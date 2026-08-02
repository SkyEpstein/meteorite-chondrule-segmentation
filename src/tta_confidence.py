"""Test-time augmentation as training data for the CONFIDENCE (error) model.

Sky's idea (2026-07-30), and the measurement that justifies it
-------------------------------------------------------------
The confidence model takes the base model's predictions and learns where they
are wrong. So more validation predictions = more (prediction, correctness) pairs
to fit on. Augmenting the validation slab supplies them.

The subtlety that makes it work: it is NOT enough to rotate a saved probability
map. All 11 existing confidence features are invariant under the dihedral group
(pointwise values, z-only statistics, and SQUARE uniform windows), so rotating a
prediction reproduces bit-identical feature rows — measured max delta 0.0 for
flips, 3.6e-07 for rot90. Nothing is learned from those.

The base model is **not rotation-equivariant**, so re-running it on a rotated
image yields a genuinely different prediction and therefore genuinely new rows.
That is what this module does.

[CORRECTED 2026-07-30] An earlier version of this docstring blamed tiled
inference and per-window GroupNorm. **That was wrong.** Verified with the tiler
removed entirely:

    TILED   predict_proba         mean|dp| 0.0499
    WHOLE   predict_proba_whole   mean|dp| 0.0505     <- tiling contributes ~1%
    tiler alone, local model      mean|dp| 0.000000

`nn.GroupNorm` normalises over (C/G, H, W) and is spatially permutation-
invariant, so a dihedral transform leaves its statistics identical. The real
cause is **convolution filter anisotropy**.

[!] That inversion matters: filter anisotropy is exactly what dihedral training
augmentation removes. A model trained with `--aug=geom` is trained TOWARD the
invariance TTA needs violated, so it is the LEAST favourable subject for this
method, not the most.

[!] The motivating number (mean|dp| 0.0570) came from an UNTRAINED net and does
not transfer: an independent training probe measured the gap collapsing 0.0784
(untrained) -> 0.0020 (trained), a 40x drop, and the trained control's val map is
94% saturated below 0.01. **The effect size on a trained model is UNMEASURED.**
Run `label_flip_rate` on a real checkpoint before building on any of this — that
is the go/no-go, and it costs 8 validation inference passes.

Two products
------------
1. **More rows.** Each of the 8 dihedral orientations is scored by the base
   model and mapped back to the canonical frame, giving 8 prediction maps over
   the SAME labels — ~4,250 slice-equivalents from 531.
2. **A disagreement feature.** The spread across those 8 predictions is a
   per-pixel instability measure. Where the model's answer depends on how the
   image was oriented, it is likely wrong. No existing feature can express this
   (they are all rotation-invariant by construction), and it needs no new labels.

[LEAKAGE][MUST] Everything here runs on the VALIDATION zone, which the base model
never trained on. The labels are the canonical validation mask throughout; only
the model's prediction varies across orientations.

[HONESTY] These rows are NOT independent samples. They are the same 19
validation chondrules seen from 8 angles, so effective sample size grows
sublinearly and the minimum detectable effect (0.0142 IoU) does NOT improve.
This makes the confidence model better FIT; it does not make the comparison
between confidence models easier to resolve.

[SEAM] Features are built PER ORIENTATION and only then pooled as rows. The 8
volumes are never concatenated along z: `z_agreement` and `z_smooth` look along
axis 0, so a stacked volume would compute z-features across an artificial seam
between orientations — the same defect class as concatenating two meteorites.
"""
from __future__ import annotations
import numpy as np

from . import combine

# (name, forward, inverse) on (n, H, W) arrays. Forward maps canonical -> view;
# inverse maps a view-frame prediction back to canonical.
DIHEDRAL = [
    ("id", lambda a: a, lambda a: a),
    ("rot90", lambda a: np.rot90(a, 1, (1, 2)), lambda a: np.rot90(a, -1, (1, 2))),
    ("rot180", lambda a: np.rot90(a, 2, (1, 2)), lambda a: np.rot90(a, -2, (1, 2))),
    ("rot270", lambda a: np.rot90(a, 3, (1, 2)), lambda a: np.rot90(a, -3, (1, 2))),
    ("flipx", lambda a: a[:, :, ::-1], lambda a: a[:, :, ::-1]),
    ("flipx_rot90", lambda a: np.rot90(a[:, :, ::-1], 1, (1, 2)),
     lambda a: np.rot90(a, -1, (1, 2))[:, :, ::-1]),
    ("flipy", lambda a: a[:, ::-1, :], lambda a: a[:, ::-1, :]),
    ("flipy_rot90", lambda a: np.rot90(a[:, ::-1, :], 1, (1, 2)),
     lambda a: np.rot90(a, -1, (1, 2))[:, ::-1, :]),
]


def _selftest_dihedral():
    """Every inverse must undo its forward exactly."""
    a = np.arange(2 * 5 * 7, dtype=np.float32).reshape(2, 5, 7)
    for name, fwd, inv in DIHEDRAL:
        back = inv(np.ascontiguousarray(fwd(a)))
        assert back.shape == a.shape, (name, back.shape, a.shape)
        assert np.array_equal(back, a), name
    return True


def tta_predict(runner, raw):
    """Score `raw` under all 8 orientations; return (8, n, H, W) in CANONICAL frame.

    The model is run on the transformed image — that is the whole point. Mapping
    a single prediction through the transforms instead would produce identical
    features and teach the confidence model nothing.
    """
    out = []
    for name, fwd, inv in DIHEDRAL:
        view = np.ascontiguousarray(fwd(raw))
        p = runner.predict_proba(view)
        out.append(np.ascontiguousarray(inv(p)))
    return np.stack(out, axis=0)


def disagreement(preds, how="std"):
    """(8,n,H,W) canonical-frame predictions -> (n,H,W) instability per pixel."""
    if how == "std":
        return preds.std(axis=0).astype(np.float32)
    if how == "range":
        return (preds.max(axis=0) - preds.min(axis=0)).astype(np.float32)
    raise ValueError(how)


def label_flip_rate(preds, thr, mask, valid):
    """THE GO/NO-GO for this whole module. Do orientations change the ANSWER?

    Disagreement in probability space (`disagreement`) can be large while every
    orientation still binarizes to the same label — in which case TTA supplies
    the confidence model with nothing it can act on, because the prune operates
    on labels. This measures the quantity that actually decides it:

      * `flip_frac`      — fraction of VALID pixels where the 8 orientations do
                           NOT unanimously agree on the binarized label.
      * `flip_on_error`  — flip rate among pixels the identity pass gets WRONG.
      * `flip_on_correct`— flip rate among pixels it gets RIGHT.
      * `lift`           — flip_on_error / flip_on_correct. This is the number
                           that matters: > 1 means instability CONCENTRATES on
                           mistakes, which is what makes it a usable confidence
                           signal. ~1 means the flips are noise, evenly spread,
                           and the feature carries no information about error.

    A near-zero `flip_frac` means the trained model is effectively equivariant
    and TTA is dead regardless of `lift`.
    """
    b = (preds >= thr)                      # (8,n,H,W) bool
    v = valid.astype(bool)
    unanimous = b.all(axis=0) | (~b).all(axis=0)
    flipped = (~unanimous) & v
    n_valid = int(v.sum())
    gt = mask.astype(bool)
    correct = ((b[0] == gt) & v)
    wrong = ((b[0] != gt) & v)
    f_err = float((flipped & wrong).sum()) / max(1, int(wrong.sum()))
    f_cor = float((flipped & correct).sum()) / max(1, int(correct.sum()))
    return {
        "threshold": float(thr),
        "n_valid_px": n_valid,
        "flip_frac": float(flipped.sum()) / max(1, n_valid),
        "flip_on_error": f_err,
        "flip_on_correct": f_cor,
        # inf (all flips on errors — the ideal) must NOT collapse to nan, which
        # would be indistinguishable from "no flips at all" (the useless case).
        "lift": (float(f_err / f_cor) if f_cor > 0
                 else (float("inf") if f_err > 0 else float("nan"))),
        # [INTERPRETATION] The null for `lift` is NOT exactly 1. Correctness is
        # judged from the identity pass, and identity is one of the 8 voters, so
        # a pixel where identity is the odd one out is both "wrong" and
        # "flipped" by construction. A synthetic control with flips scattered at
        # random still measured lift 1.85. Read lift COMPARATIVELY (control vs
        # augmented model) rather than against an absolute threshold of 1.
        "identity_error_frac": float(wrong.sum()) / max(1, n_valid),
    }


def tta_average(preds, how="mean"):
    """(8,n,H,W) canonical-frame predictions -> (n,H,W) ensembled probability."""
    if how == "mean":
        return preds.mean(axis=0).astype(np.float32)
    if how == "median":
        return np.median(preds, axis=0).astype(np.float32)
    raise ValueError(how)


def compare_single_vs_tta(runner, raw, mask, valid, grid=None):
    """Does averaging the 8 orientations beat a single forward pass?

    [LEAKAGE][MUST] Run this on VALIDATION. Each variant gets its own
    val-selected threshold, and only the variant chosen here should later be
    scored on TEST — constitution §2 is "touch test once", and scoring both
    variants on test to pick the better one would be selection on test.

    Returns a dict with both variants' full metric sets (§5: never a bare IoU)
    and the delta, flagged against the noise floor.
    """
    from . import metrics
    preds = tta_predict(runner, raw)
    single = preds[0]                     # identity orientation == plain predict
    avg = tta_average(preds)
    gt = mask.astype(bool)
    out = {}
    for name, p in (("single_pass", single), ("tta_mean_8", avg)):
        t, _ = metrics.select_threshold(p, gt, valid, grid=grid)
        m = metrics.metrics_from_pred(p >= t, mask, valid)
        m["auprc"] = float(metrics.auprc(p, gt, valid))
        m["threshold"] = float(t)
        out[name] = {k: float(v) for k, v in m.items()}
    d = out["tta_mean_8"]["iou"] - out["single_pass"]["iou"]
    out["delta_iou_val"] = float(d)
    out["below_noise_floor"] = bool(abs(d) < 0.0142)
    out["disagreement_mean"] = float(disagreement(preds).mean())
    out["disagreement_max"] = float(disagreement(preds).max())
    return out


def build_rows(preds, raw, mask, valid, thr, max_px=400_000, seed=0,
               use_disagreement=True):
    """Pool per-orientation feature rows into one fitting set for the error model.

    Returns (X, y) where y is the per-pixel residual |prob - gt| (the ErrorModel
    target), sampled uniformly over VALID pixels so the residual distribution is
    not distorted.
    """
    dis = disagreement(preds) if use_disagreement else None
    rng = np.random.default_rng(seed)
    n_or = preds.shape[0]
    per_or = max(2, max_px // n_or)
    gt = mask.astype(np.float32)
    Xs, ys = [], []
    for o in range(n_or):
        # [SEAM] features per orientation, never on a stacked volume
        F = combine.ConfidenceModel.build_features(preds[o], raw, thr, extra=dis)
        idx = np.flatnonzero(valid.ravel())
        if idx.size > per_or:
            idx = rng.choice(idx, per_or, replace=False)
        Xs.append(F.reshape(-1, F.shape[-1])[idx])
        ys.append(np.abs(preds[o].ravel()[idx] - gt.ravel()[idx]))
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


if __name__ == "__main__":
    assert _selftest_dihedral()
    print("dihedral round-trip: OK (all 8 inverses exact)")
    # feature-shape check with and without the extra channel
    rng = np.random.default_rng(0)
    p = rng.random((3, 40, 50)).astype(np.float32)
    r = (rng.random((3, 40, 50)) * 255).astype(np.uint8)
    f0 = combine.ConfidenceModel.build_features(p, r, 0.3)
    f1 = combine.ConfidenceModel.build_features(p, r, 0.3, extra=p * 0 + 1)
    print(f"features without extra: {f0.shape[-1]}   with extra: {f1.shape[-1]}")
    assert f1.shape[-1] == f0.shape[-1] + 1
    assert np.array_equal(f1[..., :-1], f0), "extra channel perturbed the base 11"
    print("extra-channel append: OK (base 11 unchanged)")
