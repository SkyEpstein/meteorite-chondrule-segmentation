"""Recorded bake-off for the confidence meta-model (constitution: choose by
bake-off, never arbitrarily).

`src/combine.py` hard-codes ONE estimator per meta-model task:
`HistGradientBoostingClassifier` for the trust map and
`HistGradientBoostingRegressor` for the error bar. That choice was never
measured against anything. This module measures it:

  * trust task: classify `(prob >= thr) == mask`, "was the base model's
    prediction at this pixel correct";
  * error task: regress the residual `|prob - gt|`, i.e. how wrong it is;
  * free baselines: margin, raw-probability uncertainty and z-agreement,
    which cost NOTHING to compute. They are the controls: a learned meta-model
    that cannot beat `|prob - 0.5|` has not earned its training time.

Every candidate is scored the same way: build a score map, grid-search the
prune cutoff on VAL, report the pruned validation IoU.

[LEAKAGE][MUST] Every estimator here is FIT on VALIDATION-zone arrays and every
winner is chosen on VALIDATION. Fitting on train-zone predictions would be
dishonest: the base model memorizes train, so its train residuals describe a
mistake distribution that exists nowhere else. Test arrays are only ever passed
to `evaluate_on_test`, after the winner is frozen.

Self-test:  python -m src.confidence_bakeoff
"""
from __future__ import annotations
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import combine, metrics


# ---------------------------------------------------------------------------
# 1. Candidate registries
# ---------------------------------------------------------------------------

def classifier_candidates():
    """Name -> estimator for the TRUST task (target: `(prob >= thr) == mask`).

    Five families deliberately spanning different inductive biases: boosted
    trees, bagged trees at two split rules, a linear model and a small MLP.
    The scaled families are pipelined with `StandardScaler` because the 11
    confidence features are on wildly different scales.
    """
    return {
        "hist_gb": HistGradientBoostingClassifier(max_iter=150, random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=100,
                                                min_samples_leaf=8,
                                                n_jobs=-1, random_state=0),
        "extra_trees": ExtraTreesClassifier(n_estimators=100,
                                            min_samples_leaf=8,
                                            n_jobs=-1, random_state=0),
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=300)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=60, random_state=0)),
    }


def regressor_candidates():
    """Name -> estimator for the ERROR task (target: `|prob - gt|`)."""
    return {
        "hist_gb": HistGradientBoostingRegressor(max_iter=150, random_state=0),
        "random_forest": RandomForestRegressor(n_estimators=100,
                                               min_samples_leaf=8,
                                               n_jobs=-1, random_state=0),
        "extra_trees": ExtraTreesRegressor(n_estimators=100,
                                           min_samples_leaf=8,
                                           n_jobs=-1, random_state=0),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32),
                                          max_iter=60, random_state=0)),
    }


# ---------------------------------------------------------------------------
# 2. FREE baselines: no training at all, the controls the learned models must
#    beat to justify their existence.
# ---------------------------------------------------------------------------

def free_baselines(prob, thr, k=3):
    """Name -> (score_map, higher_is_worse) for the zero-training baselines.

    ONLY TWO OF THESE ARE INDEPENDENT. `margin`, `margin_trust` and
    `raw_prob_uncertainty` are monotone transforms of one another (verified:
    |Spearman rho| = 1.000 between all three pairs), so they must yield
    IDENTICAL pruned IoU, because `fit_prune` sweeps a cutoff over each map's
    own range and `higher_is_worse` flips the comparison. They are kept as an
    internal consistency check: if their scores ever differ, something is wrong.
    The genuinely independent zero-cost control is `z_agreement`.

      margin                -|prob - 0.5| as an ERROR-like score: high means
                            close to the decision boundary, i.e. worse.
      margin_trust          |prob - 0.5| as the equivalent TRUST-like map.
      raw_prob_uncertainty  1 - max(prob, 1 - prob).
      z_agreement           combine.z_agreement(prob, thr, k), a trust map.

    None of these costs a fit, so any learned candidate that does not beat them
    is not worth running.
    """
    p = np.asarray(prob, np.float32)
    return {
        "margin": ((-np.abs(p - 0.5)).astype(np.float32), True),
        "margin_trust": (np.abs(p - 0.5).astype(np.float32), False),
        "raw_prob_uncertainty": ((1.0 - np.maximum(p, 1.0 - p)).astype(np.float32),
                                 True),
        "z_agreement": (combine.z_agreement(p, thr, k), False),
    }


# ---------------------------------------------------------------------------
# 3. Sampling helper
# ---------------------------------------------------------------------------

def sample_xy(prob, raw, mask, valid, thr, target, max_px=300_000, seed=0):
    """Sample a (X, y) training set for one meta-model task from VALID pixels.

    Features come from `ConfidenceModel._chunks`, so the (n,H,W,11) stack is
    never materialized: it is ~6 GB for a real zone.

    `target="trust"` -> y is the correctness bool `(prob >= thr) == mask`,
    sampled BALANCED between correct and incorrect because the base model is
    right on the large majority of pixels.
    `target="error"` -> y is the residual `|prob - gt|`, sampled UNIFORMLY:
    the residual distribution is itself the training signal, so re-weighting it
    would distort the very quantity being regressed.

    [LEAKAGE][MUST] The caller passes VALIDATION-zone arrays. Sampling from
    train-zone predictions would describe a mistake distribution that exists
    nowhere else, because the base model memorizes train.

    Returns (X, y).
    """
    cm = combine.ConfidenceModel()
    rng = np.random.default_rng(seed)
    nsl = prob.shape[0]
    budget = max(2, max_px // max(1, nsl))     # per-slice sampling budget
    Xs, ys = [], []
    for i0, i1, X in cm._chunks(prob, raw, thr):
        for j in range(i1 - i0):
            i = i0 + j
            vi = valid[i]
            Xv = X[j][vi]
            if Xv.shape[0] == 0:
                continue
            if target == "trust":
                yv = ((prob[i] >= thr) == mask[i])[vi]
                ok, bad = np.flatnonzero(yv), np.flatnonzero(~yv)
                m = min(len(ok), len(bad), budget // 2)
                if m == 0:
                    continue
                sel = np.concatenate([rng.choice(ok, m, replace=False),
                                      rng.choice(bad, m, replace=False)])
            elif target == "error":
                yv = np.abs(prob[i] - mask[i].astype(np.float32))[vi]
                sel = rng.choice(Xv.shape[0], min(Xv.shape[0], budget),
                                 replace=False)
            else:
                raise ValueError(f"unknown target {target!r}")
            Xs.append(Xv[sel])
            ys.append(yv[sel])
    if not Xs:
        raise ValueError(f"no valid pixels to sample for target {target!r}")
    y = np.concatenate(ys)
    return np.concatenate(Xs), (y.astype(np.uint8) if target == "trust" else y)


# ---------------------------------------------------------------------------
# 4. Prediction helper
# ---------------------------------------------------------------------------

def predict_map(est, prob, raw, thr, kind):
    """(n,H,W) float32 score map from a fitted candidate, chunked over z.

    `kind="trust"` reads `predict_proba(...)[:, 1]`; `kind="error"` reads
    `predict(...)` clipped to [0,1].
    """
    cm = combine.ConfidenceModel()
    out = np.zeros(np.shape(prob), np.float32)
    for i0, i1, X in cm._chunks(prob, raw, thr):
        flat = X.reshape(-1, X.shape[-1])
        if kind == "trust":
            p = est.predict_proba(flat)[:, 1]
        elif kind == "error":
            p = np.clip(est.predict(flat), 0.0, 1.0)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        out[i0:i1] = p.reshape(X.shape[:-1]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 5. The bake-off
# ---------------------------------------------------------------------------

def _ranking_metric(score_map, prob, mask, valid, thr, higher_is_worse,
                    max_n=200_000):
    """How well does the score map RANK the base model's mistakes?

    Trust-like maps (higher_is_worse=False) get `roc_auc_score` against the
    correctness label; error-like maps get `spearmanr` against the true
    residual. Subsampled by stride to at most `max_n` valid pixels to cap the
    statistics cost, the same trick `ErrorModel.evaluate` uses. NaN when the
    statistic is undefined (one class only, or a constant score).
    """
    s = score_map[valid].astype(np.float64)
    sub = slice(None, None, max(1, s.size // max_n))
    s = s[sub]
    if higher_is_worse:
        ae = np.abs(prob[valid] - mask[valid].astype(np.float32))
        ae = ae.astype(np.float64)[sub]
        if s.size > 2 and s.std() > 0 and ae.std() > 0:
            return float(spearmanr(s, ae)[0])
        return float("nan")
    correct = ((prob[valid] >= thr) == mask[valid])[sub]
    if correct.size > 2 and 0 < int(correct.sum()) < correct.size:
        return float(roc_auc_score(correct.astype(np.uint8), s))
    return float("nan")


def run_bakeoff(prob_val, raw_val, mask_val, valid_val, thr_ref,
                target_recall=0.95, seed=0):
    """Score every candidate meta-model and every free baseline on VALIDATION.

    The base model is first opened up to `target_recall` via
    `combine.recall_threshold` (constitution P9: over-predict, then prune), and
    each candidate's score map is then handed to `combine.fit_prune`, so every
    family is judged on the one number that matters downstream: the pruned
    validation IoU.

    [LEAKAGE][MUST] Every array here is from the VALIDATION zone: the recall
    threshold, the fitted estimators and the prune cutoffs are all selected on
    val and frozen before `evaluate_on_test` ever sees the test slab.

    Returns (rows, fitted): `rows` is a list of dicts sorted by
    `pruned_val_iou` descending, `fitted` maps `f"{task}_{name}"` to the fitted
    estimator so the winner can be reused without refitting.
    """
    thr_rec, _ = combine.recall_threshold(prob_val, mask_val, valid_val,
                                          target_recall)
    rows, fitted = [], {}

    def record(name, task, score_map, higher_is_worse, fit_sec):
        pr = combine.fit_prune(prob_val, mask_val, valid_val, score_map,
                               thr_rec, higher_is_worse=higher_is_worse)
        rank = _ranking_metric(score_map, prob_val, mask_val, valid_val,
                               thr_ref, higher_is_worse)
        row = {"name": name, "task": task,
               "pruned_val_iou": float(pr["val_iou"]),
               "unpruned_val_iou": float(pr["unpruned_val_iou"]),
               "gain": float(pr["gain"]), "cutoff": pr["cutoff"],
               "fit_sec": round(float(fit_sec), 1),
               "ranking_metric": rank,
               "higher_is_worse": bool(higher_is_worse)}
        rows.append(row)
        print(f"[bake] {task:5s} {name:22s} pruned IoU={row['pruned_val_iou']:.4f} "
              f"(unpruned {row['unpruned_val_iou']:.4f}, gain {row['gain']:+.4f}) "
              f"rank={rank:.4f} {row['fit_sec']:.1f}s", flush=True)
        return row

    for task, candidates, worse in (("trust", classifier_candidates(), False),
                                    ("error", regressor_candidates(), True)):
        X, y = sample_xy(prob_val, raw_val, mask_val, valid_val, thr_ref, task,
                         seed=seed)
        for name, est in candidates.items():
            t0 = time.time()
            est.fit(X, y)
            fit_sec = time.time() - t0
            score_map = predict_map(est, prob_val, raw_val, thr_ref, task)
            fitted[f"{task}_{name}"] = est
            record(name, task, score_map, worse, fit_sec)

    for name, (score_map, worse) in free_baselines(prob_val, thr_ref).items():
        record(name, "free", score_map, worse, 0.0)

    return sorted(rows, key=lambda r: r["pruned_val_iou"], reverse=True), fitted


def evaluate_on_test(winner_name, est_or_none, kind, prob_test, raw_test,
                     mask_test, valid_test, thr_rec, cutoff, higher_is_worse,
                     thr_k=3, thr_feat=None):
    """Apply the frozen winner to TEST exactly once and report hard metrics.

    A learned winner arrives as `est_or_none` and its score map is rebuilt with
    `predict_map`; a free winner arrives as `est_or_none=None` and is recomputed
    by name from `free_baselines`.

    `thr_feat` is the threshold the SCORE MAP is built at and MUST match the one
    used when the estimator was fit (`thr_ref` in `run_bakeoff`). Two of the
    confidence features depend on it: `z_agreement` and the `(prob >= thr)`
    binarization. Building them at `thr_rec` here while fitting at `thr_ref`
    would feed the model a different feature distribution than it was trained
    on. Defaults to `thr_rec` only for the degenerate case where the two are the
    same. The PRUNE itself still operates at `thr_rec`.
    """
    tf = thr_rec if thr_feat is None else thr_feat
    if est_or_none is None:
        score_map = free_baselines(prob_test, tf, thr_k)[winner_name][0]
    else:
        score_map = predict_map(est_or_none, prob_test, raw_test, tf, kind)
    pred = combine.apply_prune(prob_test, thr_rec, score_map, cutoff,
                               higher_is_worse=higher_is_worse)
    return metrics.metrics_from_pred(pred, mask_test, valid_test)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    # NOTE: this fits and scores every candidate on the SAME toy volume. It
    # is a smoke test of the plumbing, not an honest meta-model comparison.
    raw, mask, prob, _, valid = combine._synthetic()
    thr_ref, _ = metrics.select_threshold(prob, mask, valid)
    rows, fitted = run_bakeoff(prob, raw, mask, valid, thr_ref)

    assert all(np.isfinite(r["pruned_val_iou"]) for r in rows), rows
    ious = [r["pruned_val_iou"] for r in rows]
    assert all(a >= b for a, b in zip(ious, ious[1:])), ious
    free_names = set(free_baselines(prob, thr_ref))
    got_free = {r["name"] for r in rows if r["task"] == "free"}
    assert got_free == free_names, (got_free, free_names)
    learned = [r for r in rows if r["task"] in ("trust", "error")]
    assert any(np.isfinite(r["ranking_metric"]) for r in learned), learned
    assert len(fitted) == len(learned), (len(fitted), len(learned))

    print(f"\n  {'name':22s} {'task':5s} {'prunedIoU':>9s} {'gain':>8s} "
          f"{'rank':>8s} {'fit_s':>7s}")
    for r in rows:
        print(f"  {r['name']:22s} {r['task']:5s} {r['pruned_val_iou']:9.4f} "
              f"{r['gain']:+8.4f} {r['ranking_metric']:8.4f} {r['fit_sec']:7.1f}")
    return True


if __name__ == "__main__":
    assert _selftest()
    print("confidence bakeoff self-test: OK")
