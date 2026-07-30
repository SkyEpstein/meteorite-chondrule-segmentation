"""Model combination, cross-slice z-refinement, and a learned confidence model.

Three roadmap items share one file because they share one discipline: every
parameter here (a veto threshold, a blend weight, a z-window, the confidence
meta-model itself) is FIT OR SELECTED ON VALIDATION ONLY and then frozen for
the test slab. Nothing in this module may be tuned on test. [LEAKAGE][MUST]

1. Two-model combinations: recall-specialist + matrix-veto, weighted
   probability average, disagreement/agreement maps, and a logistic-regression
   stacker over the two models' probabilities.
2. Cross-slice z-refinement: kept-slices are ordered along z and adjacent
   slices place chondrules almost identically (measured adjacent-slice label
   IoU ~0.83), so smoothing/voting along z is real signal, not blur.
3. `ConfidenceModel`: a meta-model trained on the base model's own
   mistakes that emits a per-pixel trust map, judged by risk-coverage
   (selective prediction).

Self-test:  python -m src.combine
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from . import metrics

# Extends far below 0.05 on purpose. The over-predict-then-prune pipeline needs
# very high recall, and with dice loss under ~95% background the useful
# operating points sit near zero. A grid starting at 0.05 made
# `recall_threshold` silently fall back to its minimum for every target >= 0.90,
# so a "recall 1.00" experiment actually measured recall 0.69 and three
# different targets returned identical results.
THR_GRID = np.concatenate([
    np.arange(0.0002, 0.005, 0.0004),
    np.arange(0.005, 0.05, 0.005),
    np.arange(0.05, 0.96, 0.05),
])


# ---------------------------------------------------------------------------
# 1. Two-model combination strategies
# ---------------------------------------------------------------------------

def combo_veto(prob_a, prob_b_matrix, thr_a, thr_veto):
    """Recall-specialist + matrix-veto.

    Keep a pixel where model A calls it chondrule (`prob_a >= thr_a`) AND
    model B does not call it matrix (`prob_b_matrix < thr_veto`).
    `prob_b_matrix` is model B's probability that the pixel is MATRIX, so a
    high value vetoes. Returns a boolean prediction map.
    """
    return (prob_a >= thr_a) & (prob_b_matrix < thr_veto)


def fit_combo_veto(prob_a_val, prob_b_val, mask_val, valid_val):
    """Grid-search `thr_a` x `thr_veto` maximizing foreground IoU.

    [LEAKAGE][MUST] Both thresholds are fit on VALIDATION probabilities only
    and frozen before the test slab is scored.

    Returns (thr_a, thr_veto, val_iou).
    """
    best = (float(THR_GRID[0]), float(THR_GRID[0]), -1.0)
    for ta in THR_GRID:
        keep_a = prob_a_val >= ta
        for tv in THR_GRID:
            m = metrics.iou(keep_a & (prob_b_val < tv), mask_val, valid_val)
            if m == m and m > best[2]:  # skip NaN
                best = (float(ta), float(tv), float(m))
    return best


def combo_average(prob_a, prob_b, w=0.5):
    """Weighted probability average: `w * prob_a + (1 - w) * prob_b`."""
    return (w * prob_a + (1.0 - w) * prob_b).astype(np.float32)


def fit_combo_average(prob_a_val, prob_b_val, mask_val, valid_val):
    """Grid-search the blend weight `w` and the operating threshold together.

    [LEAKAGE][MUST] Both are selected on VALIDATION only, then frozen.

    Returns (w, thr, val_iou).
    """
    best_w, best_thr, best_iou = 0.0, 0.5, -1.0
    for w in np.arange(0.0, 1.01, 0.1):
        p = combo_average(prob_a_val, prob_b_val, float(w))
        thr, val = metrics.select_threshold(p, mask_val, valid_val)
        if val == val and val > best_iou:
            best_w, best_thr, best_iou = float(w), float(thr), float(val)
    return best_w, best_thr, best_iou


def disagreement_map(prob_a, prob_b):
    """|prob_a - prob_b|: where the two models most disagree."""
    return np.abs(prob_a - prob_b).astype(np.float32)


def agreement_confidence(prob_a, prob_b):
    """1 - |prob_a - prob_b| in [0,1]: high where the two models concur."""
    return (1.0 - np.abs(prob_a - prob_b)).astype(np.float32)


def _stack_features(prob_a, prob_b):
    """Stacker features per pixel: [p_a, p_b, |p_a - p_b|, p_a * p_b] -> (N,4)."""
    a = np.asarray(prob_a, np.float32).ravel()
    b = np.asarray(prob_b, np.float32).ravel()
    return np.column_stack([a, b, np.abs(a - b), a * b])


def combo_stack(prob_a, prob_b, mask, valid, max_px=300_000, seed=0):
    """Tiny logistic-regression stacker over the two models' probabilities.

    Fit on at most `max_px` VALID pixels, balanced chondrule/matrix.

    [LEAKAGE][MUST] Fit on VALIDATION pixels only. The stacker's weights are
    a tuned parameter like any threshold, and fitting them on test would leak.

    Returns the fitted sklearn estimator (apply it with `apply_stack`).
    """
    rng = np.random.default_rng(seed)
    a, b, y = prob_a[valid], prob_b[valid], mask[valid]
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    n = min(len(pos), len(neg), max_px // 2)
    sel = np.concatenate([rng.choice(pos, n, replace=False),
                          rng.choice(neg, n, replace=False)])
    est = LogisticRegression(max_iter=200)
    est.fit(_stack_features(a[sel], b[sel]), y[sel].astype(np.uint8))
    return est


def apply_stack(model, prob_a, prob_b):
    """Apply a fitted stacker; returns a probability map shaped like `prob_a`."""
    p = model.predict_proba(_stack_features(prob_a, prob_b))[:, 1]
    return p.reshape(np.shape(prob_a)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Cross-slice z-refinement
# ---------------------------------------------------------------------------

def z_smooth(prob, k=3, mode="mean"):
    """Smooth the probability volume along z (axis 0) ONLY, window `k` (odd).

    modes: "mean" (average), "median" (robust to a single bad slice), "min"
    (conservative, a pixel only survives if the whole z-window is confident)
    and "max" (inclusive, recall-oriented). Edges use replicate padding so
    every window holds k values. Returns a same-shape float32 volume.
    """
    if k % 2 == 0:
        raise ValueError(f"k must be odd, got {k}")
    p = np.asarray(prob, np.float32)
    if mode == "mean":
        out = ndi.uniform_filter1d(p, size=k, axis=0, mode="nearest")
    elif mode == "median":
        out = ndi.median_filter(p, size=(k, 1, 1), mode="nearest")
    elif mode == "min":
        out = ndi.minimum_filter1d(p, size=k, axis=0, mode="nearest")
    elif mode == "max":
        out = ndi.maximum_filter1d(p, size=k, axis=0, mode="nearest")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return out.astype(np.float32)


def z_agreement(prob, thr, k=3):
    """Fraction of the k z-neighbors (including this slice) whose binarized
    prediction AGREES with this slice's binarized prediction.

    1.0 = the whole z-window calls this pixel the same way; low values flag
    slice-to-slice flicker. Replicate padding keeps the denominator at k at the
    volume ends. Returns (n,H,W) float32 in [0,1], a confidence signal.
    """
    b = (np.asarray(prob) >= thr).astype(np.float32)
    frac_pos = ndi.uniform_filter1d(b, size=k, axis=0, mode="nearest")
    agree = np.where(b > 0.5, frac_pos, 1.0 - frac_pos)
    return np.clip(agree, 0.0, 1.0).astype(np.float32)


def fit_z_refine(prob_val, mask_val, valid_val):
    """Grid-search the z-refinement (mode, k) and the operating threshold.

    [LEAKAGE][MUST] Selected on VALIDATION only; the winner is frozen and then
    applied to test.

    Returns dict(mode, k, thr, val_iou).
    """
    best = {"mode": "mean", "k": 3, "thr": 0.5, "val_iou": -1.0}
    for mode in ("mean", "median", "min", "max"):
        for k in (3, 5, 7):
            p = z_smooth(prob_val, k=k, mode=mode)
            thr, val = metrics.select_threshold(p, mask_val, valid_val)
            if val == val and val > best["val_iou"]:
                best = {"mode": mode, "k": int(k), "thr": float(thr),
                        "val_iou": float(val)}
    return best


# ---------------------------------------------------------------------------
# 3. Learned confidence model
# ---------------------------------------------------------------------------

def _local_mean_std(vol, w):
    """In-plane (per-slice) local mean and std over a w x w window."""
    m = ndi.uniform_filter(vol, size=(1, w, w))
    m2 = ndi.uniform_filter(vol * vol, size=(1, w, w))
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return m.astype(np.float32), s.astype(np.float32)


class ConfidenceModel:
    """Per-pixel TRUST map: predicts whether the BASE model's prediction at
    that pixel is CORRECT (the REE-logD confidence idea applied to pixels).

    Fit it on the base model's validation-zone predictions, then apply the
    frozen meta-model to test / unlabeled slices and read `risk_coverage` to
    see whether high-trust pixels really are more accurate.
    """

    def __init__(self):
        self.model = None

    @staticmethod
    def build_features(prob, raw, thr, k=3):
        """(n,H,W) base probability + raw intensity -> (n,H,W,F) float32 stack.

        F = 11 confidence features, in order:
          0  prob      the base model's probability itself
          1  margin    |prob - 0.5|, the distance from the decision boundary;
                       the model's errors concentrate near it
          2  z_agree   z_agreement(prob, thr, k): do the neighboring slices
                       binarize this pixel the same way?
          3  z_mean    z_smooth(prob, k, "mean"), the z-neighborhood's mean
                       probability
          4  pmean5    local mean of prob, 5x5 in-plane window
          5  pstd5     local std of prob, 5x5 (a contested neighborhood)
          6  pmean11   local mean of prob, 11x11 in-plane window
          7  pstd11    local std of prob, 11x11
          8  raw       raw CT intensity / 255
          9  rmean5    local mean of raw intensity, 5x5
         10  rstd5     local std of raw intensity, 5x5 (texture/edge proxy)

        Local statistics use `scipy.ndimage.uniform_filter` with a (1, w, w)
        footprint, so only features 2-3 look across z.
        """
        p = np.asarray(prob, np.float32)
        r = np.asarray(raw, np.float32) / 255.0
        feats = [p, np.abs(p - 0.5).astype(np.float32),
                 z_agreement(p, thr, k), z_smooth(p, k, "mean")]
        for w in (5, 11):
            m, s = _local_mean_std(p, w)
            feats += [m, s]
        feats.append(r)
        m, s = _local_mean_std(r, 5)
        feats += [m, s]
        return np.stack(feats, axis=-1)

    def _chunks(self, prob, raw, thr, k=3, chunk=6):
        """Yield (i0, i1, X_interior) over z-windows instead of materializing the
        whole (n,H,W,11) stack.

        The full stack is ~6 GB for a real zone (962x1450x11 float32 per slice),
        which violates the project's memory rule. Each window is expanded by a
        halo of k//2 slices and the halo rows are then dropped, so the two
        z-dependent features (z_agreement, z_smooth) see exactly the neighbors
        they would in a full-volume pass. The output is identical, the peak
        memory is not. At the true volume ends the halo is clipped, which
        reproduces the replicate padding the full-volume path uses.
        """
        n = prob.shape[0]
        h = k // 2
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            a, b = max(0, i0 - h), min(n, i1 + h)
            X = self.build_features(prob[a:b], raw[a:b], thr, k)
            yield i0, i1, X[i0 - a:i1 - a]

    def fit(self, prob, raw, mask, valid, thr, max_px=400_000, seed=0):
        """Learn where the base model is right: y = ((prob >= thr) == mask),
        over VALID pixels only.

        [LEAKAGE][MUST] The caller fits this on VALIDATION-zone predictions,
        from a zone the base model never trained on. Fitting it on train-zone
        predictions would be dishonest: the base model memorizes train, so
        nearly every train pixel is "correct" and the confidence model would
        learn a mistake distribution that exists nowhere else. Once fit, the
        meta-model is frozen for the test slab.

        Pixels are subsampled to at most `max_px`, BALANCED between correct and
        incorrect, because the base model is right on the large majority of
        pixels.
        """
        rng = np.random.default_rng(seed)
        nsl = prob.shape[0]
        budget = max(2, max_px // max(1, nsl))     # per-slice sampling budget
        Xs, ys = [], []
        for i0, i1, X in self._chunks(prob, raw, thr):
            for j in range(i1 - i0):
                i = i0 + j
                yi = ((prob[i] >= thr) == mask[i])
                vi = valid[i]
                Xv, yv = X[j][vi], yi[vi]
                ok, bad = np.flatnonzero(yv), np.flatnonzero(~yv)
                m = min(len(ok), len(bad), budget // 2)
                if m == 0:
                    continue
                sel = np.concatenate([rng.choice(ok, m, replace=False),
                                      rng.choice(bad, m, replace=False)])
                Xs.append(Xv[sel])
                ys.append(yv[sel])
        if not Xs:
            raise ValueError("no slice had both correct and incorrect valid pixels")
        self.model = HistGradientBoostingClassifier(max_iter=150,
                                                    random_state=seed)
        self.model.fit(np.concatenate(Xs), np.concatenate(ys).astype(np.uint8))
        return self

    def predict(self, prob, raw, thr):
        """(n,H,W) float32 trust map in [0,1]: P(the base prediction here is
        correct), from the meta-model frozen at fit time."""
        out = np.zeros(np.shape(prob), np.float32)
        for i0, i1, X in self._chunks(prob, raw, thr):
            flat = X.reshape(-1, X.shape[-1])
            p = self.model.predict_proba(flat)[:, 1]
            out[i0:i1] = p.reshape(X.shape[:-1]).astype(np.float32)
        return out

    @staticmethod
    def risk_coverage(trust, prob, mask, valid, thr, n_points=20):
        """Selective prediction: do the most-trusted pixels really score better?

        Valid pixels are sorted by `trust` descending; at each coverage level in
        `linspace(0.05, 1.0, n_points)` we keep that top fraction and report the
        foreground IoU and the pixel error rate among the KEPT pixels. A trust
        map that means something shows error rising with coverage.

        Returns [{"coverage", "iou", "error_rate"}, ...].
        """
        t = trust[valid]
        pred, gt = (prob[valid] >= thr), mask[valid]
        order = np.argsort(-t, kind="stable")
        n = t.size
        out = []
        for c in np.linspace(0.05, 1.0, n_points):
            keep = order[:max(1, int(round(float(c) * n)))]
            p, g = pred[keep], gt[keep]
            out.append({"coverage": float(c),
                        "iou": float(metrics.iou(p, g)),
                        "error_rate": float(np.mean(p != g))})
        return out


# ---------------------------------------------------------------------------
# 4. Learned ERROR model (spec R12): regression, not classification
# ---------------------------------------------------------------------------

class ErrorModel:
    """Predicts HOW WRONG the base model is, not merely whether it is wrong.

    Sibling to `ConfidenceModel`, and deliberately different:

      * `ConfidenceModel` is a CLASSIFIER on `(prob >= thr) == gt`. Its answer
        depends on the operating threshold and is binary: "trust / don't".
      * `ErrorModel` is a REGRESSOR on the residual `|prob - gt|`. It is
        threshold-INDEPENDENT and returns an expected error magnitude, i.e. an
        error bar. That is what lets you rank whole slices or chondrules by how
        much they need a human's eyes.

    Both are meta-models over the same feature stack; keeping both means we can
    report which one actually ranks errors better rather than assuming.

    [LEAKAGE][MUST] Fit on VALIDATION-zone predictions only. Fitting on the
    train zone would be dishonest: the base model memorizes train (measured
    train IoU 1.000 on seen patches), so its train residuals are near zero and
    describe a mistake distribution that exists nowhere else.
    """

    def __init__(self):
        self.model = None

    def fit(self, prob, raw, mask, valid, thr, max_px=400_000, seed=0):
        """Regress the per-pixel residual |prob - gt| on the confidence features.

        Sampling is uniform over valid pixels (NOT balanced): the residual
        distribution itself is the training signal, so re-weighting it would
        distort the very quantity being predicted.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor
        cm = ConfidenceModel()
        rng = np.random.default_rng(seed)
        nsl = prob.shape[0]
        budget = max(2, max_px // max(1, nsl))
        Xs, ys = [], []
        for i0, i1, X in cm._chunks(prob, raw, thr):
            for j in range(i1 - i0):
                i = i0 + j
                vi = valid[i]
                resid = np.abs(prob[i] - mask[i].astype(np.float32))[vi]
                Xv = X[j][vi]
                if Xv.shape[0] == 0:
                    continue
                m = min(Xv.shape[0], budget)
                sel = rng.choice(Xv.shape[0], m, replace=False)
                Xs.append(Xv[sel])
                ys.append(resid[sel])
        if not Xs:
            raise ValueError("no valid pixels to fit the error model")
        self.model = HistGradientBoostingRegressor(max_iter=150, random_state=seed)
        self.model.fit(np.concatenate(Xs), np.concatenate(ys))
        return self

    def predict(self, prob, raw, thr):
        """(n,H,W) float32 predicted error magnitude, clipped to [0,1]."""
        cm = ConfidenceModel()
        out = np.zeros(np.shape(prob), np.float32)
        for i0, i1, X in cm._chunks(prob, raw, thr):
            flat = X.reshape(-1, X.shape[-1])
            out[i0:i1] = self.model.predict(flat).reshape(X.shape[:-1])
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def evaluate(pred_err, prob, mask, valid, bins=10):
        """Is the predicted error actually informative about the real error?

        Reports Pearson + Spearman correlation against the true residual and a
        binned calibration table (mean predicted vs mean actual error per bin).
        A useful model shows positive correlation and a roughly increasing
        actual-error column.
        """
        from scipy.stats import pearsonr, spearmanr
        pe = pred_err[valid].astype(np.float64)
        ae = np.abs(prob[valid] - mask[valid].astype(np.float32)).astype(np.float64)
        out = {"n_pixels": int(pe.size),
               "mean_predicted_error": float(pe.mean()),
               "mean_actual_error": float(ae.mean()),
               "mae_of_error_prediction": float(np.mean(np.abs(pe - ae)))}
        if pe.size > 2 and pe.std() > 0 and ae.std() > 0:
            sub = slice(None, None, max(1, pe.size // 200_000))  # cap the stats cost
            out["pearson"] = float(pearsonr(pe[sub], ae[sub])[0])
            out["spearman"] = float(spearmanr(pe[sub], ae[sub])[0])
        edges = np.quantile(pe, np.linspace(0, 1, bins + 1))
        table = []
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            sel = (pe >= lo) & (pe <= hi) if b == bins - 1 else (pe >= lo) & (pe < hi)
            if sel.sum():
                table.append({"bin": b, "n": int(sel.sum()),
                              "pred": float(pe[sel].mean()),
                              "actual": float(ae[sel].mean())})
        out["calibration"] = table
        return out

    @staticmethod
    def aggregate(pred_err, axis_slices=True):
        """Per-slice mean predicted error: ranks slices for human review."""
        if axis_slices:
            return pred_err.reshape(pred_err.shape[0], -1).mean(axis=1)
        return float(pred_err.mean())


# ---------------------------------------------------------------------------
# 4b. False-positive detector: the CORRECT target for pruning
# ---------------------------------------------------------------------------

class FPDetector:
    """P(this predicted-positive pixel is actually a FALSE POSITIVE).

    This is the right objective for pruning, and neither `ConfidenceModel` nor
    `ErrorModel` supplies it:

      * `ErrorModel` regresses ``|prob - gt|``. At a high-recall threshold every
        probability is small, so a TRUE positive (gt=1, prob~0.1) has residual
        ~0.9 while a FALSE positive (gt=0, prob~0.1) has residual ~0.1. The
        signal is INVERTED for pruning. Measured on the GNN: mean predicted
        error 0.331 on TPs vs 0.259 on FPs, so pruning "high error" deletes
        chondrules. That is why every prune experiment returned ~zero gain
        despite the error model itself being sound (Spearman 0.767 vs true
        error).
      * `ConfidenceModel` asks "was the prediction correct" over ALL pixels,
        which is dominated by the ~95% easy background.

    Restricting to the predicted-positive set and asking "is this a false
    positive?" targets exactly the pixels pruning can act on. The upper bound is
    large: deleting every FP took the GNN from IoU 0.146 to 0.810.

    [LEAKAGE][MUST] Fit on VALIDATION-zone predictions only.
    """

    def __init__(self, estimator=None):
        self.model = estimator
        self.extra = None          # optional callable: slice2d -> (H,W) extra feats

    def fit(self, prob, raw, mask, valid, thr, max_px=400_000, seed=0,
            estimator=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        cm = ConfidenceModel()
        rng = np.random.default_rng(seed)
        Xs, ys = [], []
        budget = max(2, max_px // max(1, prob.shape[0]))
        for i0, i1, X in cm._chunks(prob, raw, thr):
            for j in range(i1 - i0):
                i = i0 + j
                sel = (prob[i] >= thr) & valid[i]      # predicted-positive only
                if not sel.any():
                    continue
                Xi, yi = X[j][sel], (~mask[i][sel])    # 1 == false positive
                pos, neg = np.flatnonzero(yi), np.flatnonzero(~yi)
                m = min(len(pos), len(neg), budget // 2)
                if m == 0:
                    continue
                idx = np.concatenate([rng.choice(pos, m, replace=False),
                                      rng.choice(neg, m, replace=False)])
                Xs.append(Xi[idx])
                ys.append(yi[idx])
        if not Xs:
            raise ValueError("no slice had both TPs and FPs among predictions")
        self.model = estimator or self.model or HistGradientBoostingClassifier(
            max_iter=150, random_state=seed)
        self.model.fit(np.concatenate(Xs), np.concatenate(ys).astype(np.uint8))
        return self

    def predict(self, prob, raw, thr):
        """(n,H,W) P(false positive). Higher = more likely a false alarm, so it
        is used with higher_is_worse=True in `fit_prune`."""
        cm = ConfidenceModel()
        out = np.zeros(np.shape(prob), np.float32)
        for i0, i1, X in cm._chunks(prob, raw, thr):
            flat = X.reshape(-1, X.shape[-1])
            out[i0:i1] = self.model.predict_proba(flat)[:, 1].reshape(X.shape[:-1])
        return out


# ---------------------------------------------------------------------------
# 5. Over-predict, then prune (constitution P9)
# ---------------------------------------------------------------------------

def recall_threshold(prob_val, mask_val, valid_val, target_recall=0.95):
    """Lowest-precision-cost threshold that still reaches `target_recall`.

    The deliberate over-prediction step: a missed chondrule cannot be recovered
    downstream, so the base model is opened up until it catches ~everything and
    precision is left to the prune stage. Selected on VAL. [LEAKAGE][MUST]
    """
    best = None
    for t in THR_GRID:
        m = metrics.metrics_from_pred(prob_val >= t, mask_val, valid_val)
        if m["recall"] == m["recall"] and m["recall"] >= target_recall:
            if best is None or m["precision"] > best[1]["precision"]:
                best = (float(t), m)
    if best is None:                      # target unreachable: take max recall
        t = float(THR_GRID[0])
        best = (t, metrics.metrics_from_pred(prob_val >= t, mask_val, valid_val))
    return best


def apply_prune(prob, thr_recall, score_map, cutoff, higher_is_worse=True):
    """Over-predicted mask minus the pixels the meta-model distrusts.

    `score_map` is either a predicted-error map (higher_is_worse=True) or a
    trust map (higher_is_worse=False).
    """
    keep = prob >= thr_recall
    drop = score_map > cutoff if higher_is_worse else score_map < cutoff
    return keep & ~drop


def fit_prune(prob_val, mask_val, valid_val, score_map_val, thr_recall,
              higher_is_worse=True, n_cut=25):
    """Grid-search the prune cutoff on VAL; returns (cutoff, val_iou, table).

    Also reports the un-pruned baseline so the gain from pruning is explicit
    rather than implied. [LEAKAGE][MUST] Fit on validation only.
    """
    base_iou = float(metrics.iou(prob_val >= thr_recall, mask_val, valid_val))
    lo, hi = float(np.min(score_map_val)), float(np.max(score_map_val))
    best, table = (None, -1.0), []
    for c in np.linspace(lo, hi, n_cut):
        pred = apply_prune(prob_val, thr_recall, score_map_val, float(c),
                           higher_is_worse)
        m = metrics.metrics_from_pred(pred, mask_val, valid_val)
        table.append({"cutoff": float(c), "iou": m["iou"],
                      "precision": m["precision"], "recall": m["recall"]})
        if m["iou"] == m["iou"] and m["iou"] > best[1]:
            best = (float(c), float(m["iou"]))
    return {"cutoff": best[0], "val_iou": best[1],
            "unpruned_val_iou": base_iou,
            "gain": (best[1] - base_iou) if best[0] is not None else 0.0,
            "table": table}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _synthetic(n=12, H=64, W=80, seed=0):
    """A bright disk drifting along z (ground truth) plus noisy probability
    maps of it: a chondrule-shaped toy volume with known answers."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = np.zeros((n, H, W), bool)
    raw = np.zeros((n, H, W), np.float32)
    for i in range(n):
        cy, cx = 30.0 + 0.5 * i, 22.0 + 2.0 * i
        mask[i] = (yy - cy) ** 2 + (xx - cx) ** 2 <= 12.0 ** 2
        raw[i] = 40.0 + 70.0 * mask[i] + rng.normal(0, 6, (H, W))
    prob = np.clip(mask + rng.normal(0, 0.35, mask.shape), 0, 1).astype(np.float32)
    prob_b_matrix = np.clip(~mask + rng.normal(0, 0.35, mask.shape), 0, 1).astype(np.float32)
    return raw, mask, prob, prob_b_matrix, np.ones((n, H, W), bool)


def _selftest():
    raw, mask, prob, prob_b_matrix, valid = _synthetic()
    truth = mask.astype(np.float32)

    sm = z_smooth(prob, k=3, mode="mean")
    mse_raw = float(np.mean((prob - truth) ** 2))
    mse_sm = float(np.mean((sm - truth) ** 2))
    assert sm.shape == prob.shape and sm.dtype == np.float32, sm.shape
    assert mse_sm < mse_raw, (mse_sm, mse_raw)
    print(f"  z_smooth      shape {sm.shape} kept; MSE to truth "
          f"{mse_raw:.4f} -> {mse_sm:.4f}   OK")

    za = z_agreement(prob, 0.5, k=3)
    assert za.shape == prob.shape, za.shape
    assert za.min() >= 0.0 and za.max() <= 1.0, (za.min(), za.max())
    print(f"  z_agreement   range [{za.min():.3f}, {za.max():.3f}] within [0,1]   OK")

    ta, tv, viou = fit_combo_veto(prob, prob_b_matrix, mask, valid)
    lo, hi = float(THR_GRID[0]), float(THR_GRID[-1])
    assert lo <= ta <= hi and lo <= tv <= hi, (ta, tv)
    assert np.isfinite(viou), viou
    print(f"  combo_veto    thr_a={ta:.2f} thr_veto={tv:.2f} val IoU={viou:.4f}   OK")

    # NOTE: this fits and scores the trust model on the SAME toy volume. It is a
    # smoke test of the plumbing, not an honest confidence evaluation.
    cm = ConfidenceModel().fit(prob, raw, mask, valid, thr=0.5)
    trust = cm.predict(prob, raw, 0.5)
    assert trust.shape == prob.shape, trust.shape
    assert trust.min() >= 0.0 and trust.max() <= 1.0, (trust.min(), trust.max())
    print(f"  ConfidenceModel trust {trust.shape} range "
          f"[{trust.min():.3f}, {trust.max():.3f}]   OK")

    rc = cm.risk_coverage(trust, prob, mask, valid, 0.5)
    cov = [r["coverage"] for r in rc]
    assert all(b > a for a, b in zip(cov, cov[1:])), cov
    # IoU is legitimately NaN at low coverage. Pixels sorted by trust descending
    # are background-dominated (matrix is the large majority of every slice:
    # ~91% in this toy volume, ~95% in the real data), so the most-trusted
    # fraction can contain no foreground at all: empty union -> undefined IoU,
    # per metrics.iou. That is a property of the ranking, not a defect, so the
    # check requires finite error_rate everywhere and finite IoU once coverage
    # is wide enough to include foreground.
    assert all(np.isfinite(r["error_rate"]) for r in rc), rc
    assert np.isfinite(rc[-1]["iou"]), rc[-1]
    assert any(np.isfinite(r["iou"]) for r in rc), "no coverage level kept foreground"
    assert rc[0]["error_rate"] <= rc[-1]["error_rate"], (rc[0], rc[-1])
    em = ErrorModel().fit(prob, raw, mask, valid, thr=0.5)
    pe = em.predict(prob, raw, 0.5)
    ev = ErrorModel.evaluate(pe, prob, mask, valid)
    assert pe.shape == prob.shape and pe.min() >= 0.0 and pe.max() <= 1.0, pe.shape
    assert ev["pearson"] > 0.0, ev            # predicted error must track real error
    per_slice = ErrorModel.aggregate(pe)
    assert per_slice.shape == (prob.shape[0],), per_slice.shape
    print(f"  ErrorModel    pearson={ev['pearson']:.3f} spearman={ev['spearman']:.3f} "
          f"mean pred={ev['mean_predicted_error']:.3f} vs actual="
          f"{ev['mean_actual_error']:.3f}   OK")

    # FPDetector: the one mechanism every reported prune number flows through.
    # Guards the two lines that distinguish it from ConfidenceModel: the
    # restriction to predicted-positive pixels, and the polarity of the label
    # (1 == false positive). A polarity flip here silently reproduces the
    # retracted inverted-ErrorModel result, so assert the DIRECTION explicitly.
    thr_fp, _ = recall_threshold(prob, mask, valid, target_recall=0.90)
    fd = FPDetector().fit(prob, raw, mask, valid, thr=thr_fp)
    sc = fd.predict(prob, raw, thr_fp)
    keep = prob >= thr_fp
    tp_sel, fp_sel = keep & mask & valid, keep & (~mask) & valid
    assert sc.shape == prob.shape and 0.0 <= sc.min() and sc.max() <= 1.0, sc.shape
    assert sc[fp_sel].mean() > sc[tp_sel].mean(), (
        f"FPDetector polarity inverted: P(FP)={sc[fp_sel].mean():.3f} on FPs vs "
        f"{sc[tp_sel].mean():.3f} on TPs, so pruning would delete true positives")
    fpf = fit_prune(prob, mask, valid, sc, thr_fp, higher_is_worse=True)
    assert fpf["val_iou"] >= fpf["unpruned_val_iou"] - 1e-9, fpf
    print(f"  FPDetector    P(FP) on FP={sc[fp_sel].mean():.3f} > on TP="
          f"{sc[tp_sel].mean():.3f}; prune {fpf['unpruned_val_iou']:.3f} -> "
          f"{fpf['val_iou']:.3f}   OK")

    tr, trm = recall_threshold(prob, mask, valid, target_recall=0.95)
    pr = fit_prune(prob, mask, valid, pe, tr, higher_is_worse=True)
    assert trm["recall"] >= 0.95 or tr == float(THR_GRID[0]), trm
    assert np.isfinite(pr["val_iou"]) and pr["val_iou"] >= pr["unpruned_val_iou"], pr
    print(f"  prune         recall-thr={tr:.2f} (recall={trm['recall']:.3f}) "
          f"IoU {pr['unpruned_val_iou']:.3f} -> {pr['val_iou']:.3f} "
          f"(gain {pr['gain']:+.3f})   OK")

    print(f"  risk_coverage coverage {cov[0]:.2f}->{cov[-1]:.2f} increasing; "
          f"error {rc[0]['error_rate']:.4f} @cov {cov[0]:.2f} <= "
          f"{rc[-1]['error_rate']:.4f} @cov {cov[-1]:.2f}   OK")
    return True


if __name__ == "__main__":
    assert _selftest()
    print("combine self-test: OK")
