"""Constitution stage 7: STACK the three error signals, then prune once.

Why this module exists
----------------------
`memory/constitution.md` section 12 defines the confidence stage as three
INDEPENDENT error signals combined into one score:

    (4) CONFIDENCE: regressors    -\\
    (5) CONFIDENCE: error U-Net    +-> three independent error signals
    (6) Z-NEIGHBOUR CONFIDENCE    -/
    (7) PIXEL DROPPING (prune)    -> LAST; stacks all three into one score

That stacking was never built. `confidence_bakeoff.run_bakeoff` returns its
candidates "sorted by pruned_val_iou descending" so "the winner can be reused",
and `combine.fit_prune` takes exactly ONE `score_map_val`. The pipeline therefore
picks a single best signal and prunes with that alone.

So what the existing measurements establish is narrower than it has been read as:
**no SINGLE signal beats the free margin baseline.** Ten of fourteen candidates
gained exactly 0.00000, and the error U-Net cut test IoU from 0.29443 to 0.11752.
Whether the three COMBINED help has never been tested. This module tests it.

There is a specific reason to expect combination to behave differently from
selection. Eight of the regressor's eleven features are functions of `prob`
itself (prob, margin, z_agree, z_mean, and the four local prob statistics), so it
is largely a restatement of the base model's own probability. `z_agreement` looks
ACROSS SLICES and the error U-Net has a full receptive field over raw pixels.
Those two carry information the regressor structurally cannot. Signals that fail
individually can still combine usefully exactly when they fail for different
reasons.

This is a test, not a claim. If the stacked arm loses to margin, that is reported
as plainly as a win would be.

Leakage
-------
[LEAKAGE][MUST] Everything is fitted on VALIDATION and frozen before test is
touched once. The stacker is CROSS-FIT inside val on contiguous buffered folds,
so the prune cutoff is chosen on out-of-fold scores for all val slices rather
than on scores the stacker has already seen. One final stacker is refit on all
of val and is the one applied to test.

[HONESTY] The val slab is 60 slices holding 19 chondrules by binary connected
components or 36 by the same-id rule. A 4-parameter logistic stacker is far less
exposed to that than the 14.0M-parameter error U-Net, but it is still a small fit
set and a favourable val result should not be read as transfer.

Run:  python -m src.confidence_stack [--model=NAME] [--recall=0.95] [--no-unet]
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import combine, error_cnn, metrics, zones
from .classical_crossfit import contiguous_folds

OUT = "results/confidence_stack.json"
MODEL = "unet2d_s30000"
VAL_BUFFER = 4
K_FOLDS = 5


def _iou(pred, mask, valid):
    return float(metrics.iou(pred, mask, valid))


def regressor_cv(pv, Rv, Mv, Vv, thr, k=K_FOLDS, buffer=VAL_BUFFER):
    """Cross-fit `ConfidenceModel` inside val. Returns (oof_error, final_model).

    [MUST] The regressor has to be cross-fit for the same reason the U-Net and
    the stacker do. An earlier draft of this module called `predict` on an
    UNFITTED `ConfidenceModel`, which would simply have crashed; the tempting
    fix, fitting on all of val and predicting on all of val, is worse than a
    crash because it silently reports a signal graded on its own training data.

    `ConfidenceModel` predicts TRUST, P(the base is correct), so the returned map
    is flipped to error orientation with 1 - trust.
    """
    n = Mv.shape[0]
    oof = np.zeros(Mv.shape, np.float32)
    for j, (held, fit_idx) in enumerate(contiguous_folds(n, k, buffer)):
        cm = combine.ConfidenceModel()
        cm.fit(pv[fit_idx], Rv[fit_idx], Mv[fit_idx], Vv[fit_idx], thr)
        oof[held] = 1.0 - cm.predict(pv[held], Rv[held], thr)
        print(f"  [regressor cv] fold {j + 1}: fit {len(fit_idx)} slices, "
              f"held {len(held)}", flush=True)
    final = combine.ConfidenceModel()
    final.fit(pv, Rv, Mv, Vv, thr)
    return oof, final


def signals(prob, raw, thr, unet=None, regressor=None):
    """The error signals for one zone, ALL oriented so HIGHER MEANS MORE WRONG.

    Orientation matters and is easy to get silently backwards:
      * `ConfidenceModel.predict` returns TRUST, P(the base is correct here), so
        it is flipped with 1 - trust.
      * `combine.z_agreement` returns a TRUST map too, 1.0 where the whole
        z-window binarizes the pixel the same way, so it is flipped as well.
      * the error U-Net already regresses |prob - gt|, which is error-oriented,
        so it is used as-is.
    Getting either flip wrong inverts the signal and the stacker would learn to
    keep exactly the pixels it should drop.

    `regressor` and `unet` must already be FITTED. This function never fits.
    """
    out = {"z_agreement": 1.0 - combine.z_agreement(prob, thr, k=3)}
    if regressor is not None:
        out["regressor"] = 1.0 - regressor.predict(prob, raw, thr)
    if unet is not None:
        out["error_unet"] = error_cnn.predict_error(unet, raw, prob)
    return out


def _pixels(sig_names, sigs, kept, mask, valid, max_px=400_000, seed=0):
    """Balanced pixel sample INSIDE the over-predicted mask only.

    The prune can only remove pixels that are already kept, so those are the only
    pixels on which "should this be dropped" is a real question. Fitting on all
    pixels would teach the stacker mostly about background it can never act on.
    """
    rng = np.random.default_rng(seed)
    sel = kept & valid
    y_all = (~mask) & sel                      # kept AND wrong = false positive
    idx_pos = np.flatnonzero(y_all.ravel())
    idx_neg = np.flatnonzero((sel & mask).ravel())
    if not len(idx_pos) or not len(idx_neg):
        raise SystemExit("no positives or no negatives inside the kept mask")
    n = min(max_px // 2, len(idx_pos), len(idx_neg))
    take = np.concatenate([rng.choice(idx_pos, n, replace=False),
                           rng.choice(idx_neg, n, replace=False)])
    X = np.column_stack([sigs[s].ravel()[take] for s in sig_names])
    y = np.concatenate([np.ones(n, np.int8), np.zeros(n, np.int8)])
    return X, y


def stack_cv(sig_names, sigs_val, kept_val, Mv, Vv, k=K_FOLDS,
             buffer=VAL_BUFFER, seed=0):
    """Cross-fit the stacker inside val. Returns (oof_score, final_model).

    `oof_score[i]` comes from a stacker that never saw slice i nor any slice
    within `buffer` of it, so the prune cutoff selected on it is honest. Folds
    are contiguous because adjacent CT slices are near-duplicates.
    """
    n = Mv.shape[0]
    oof = np.zeros(Mv.shape, np.float32)
    for j, (held, fit_idx) in enumerate(contiguous_folds(n, k, buffer)):
        sub = {s: sigs_val[s][fit_idx] for s in sig_names}
        X, y = _pixels(sig_names, sub, kept_val[fit_idx], Mv[fit_idx],
                       Vv[fit_idx], seed=seed + j)
        m = LogisticRegression(max_iter=1000).fit(X, y)
        Xh = np.column_stack([sigs_val[s][held].ravel() for s in sig_names])
        oof[held] = m.predict_proba(Xh)[:, 1].reshape(
            (len(held),) + Mv.shape[1:]).astype(np.float32)
        print(f"  [stack cv] fold {j + 1}: fit {len(fit_idx)} slices, "
              f"held {len(held)}", flush=True)
    X, y = _pixels(sig_names, sigs_val, kept_val, Mv, Vv, seed=seed)
    final = LogisticRegression(max_iter=1000).fit(X, y)
    return oof, final


def run(root=".", model=MODEL, target_recall=0.95, k=K_FOLDS, use_unet=True):
    z = zones.load_zones(root, zones=("val", "test"))
    Rv, Mv, Vv = z["val"]
    Rt, Mt, Vt = z["test"]
    pv = np.load(f"results/probs/{model}_val.npy").astype(np.float32)
    pt = np.load(f"results/probs/{model}.npy").astype(np.float32)

    thr_rec, rm = combine.recall_threshold(pv, Mv, Vv, target_recall)
    kept_v, kept_t = pv >= thr_rec, pt >= thr_rec
    unpruned = {"val": _iou(kept_v, Mv, Vv), "test": _iou(kept_t, Mt, Vt)}
    print(f"  over-predicted to recall {rm['recall']:.4f} at thr {thr_rec:.2e}; "
          f"unpruned val {unpruned['val']:.4f} test {unpruned['test']:.4f}")

    # Every learned signal is cross-fit inside val, so the val-side map used to
    # choose the prune cutoff is out-of-fold, and the model applied to test is
    # refit on all of val. The z-agreement signal needs no fitting at all.
    t0 = time.time()
    oof_reg, reg_final = regressor_cv(pv, Rv, Mv, Vv, thr_rec, k=k)
    print(f"  regressor cross-fit in {time.time() - t0:.0f}s")

    unet, oof_unet = None, None
    if use_unet:
        t0 = time.time()
        oof_unet, unet, _ = error_cnn.fit_error_unet_cv(Rv, pv, Mv, Vv, k=k)
        print(f"  error U-Net cross-fit in {time.time() - t0:.0f}s")

    # VAL signals are the OUT-OF-FOLD maps; TEST signals come from the models
    # refit on all of val. Mixing these up would grade val on in-fold scores.
    sv = signals(pv, Rv, thr_rec)                 # z_agreement only, no fitting
    sv["regressor"] = oof_reg
    if oof_unet is not None:
        sv["error_unet"] = oof_unet
    st = signals(pt, Rt, thr_rec, unet=unet, regressor=reg_final)
    sv["margin"] = -np.abs(pv - 0.5)          # free baseline, error-oriented
    st["margin"] = -np.abs(pt - 0.5)
    sig_names = [s for s in ("regressor", "error_unet", "z_agreement")
                 if s in sv]

    # --- the stacked arm -------------------------------------------------
    oof, final = stack_cv(sig_names, sv, kept_v, Mv, Vv, k=k)
    pr = combine.fit_prune(pv, Mv, Vv, oof, thr_rec, higher_is_worse=True)
    Xt = np.column_stack([st[s].ravel() for s in sig_names])
    stacked_t = final.predict_proba(Xt)[:, 1].reshape(pt.shape).astype(np.float32)
    arms = {"stacked": {
        "val": pr["val_iou"],
        "test": _iou(combine.apply_prune(pt, thr_rec, stacked_t, pr["cutoff"],
                                         True), Mt, Vt),
        "cutoff": pr["cutoff"],
        "coefficients": dict(zip(sig_names,
                                 [float(c) for c in final.coef_[0]])),
        "intercept": float(final.intercept_[0])}}

    # --- every single signal, same protocol ------------------------------
    for s in sig_names + ["margin"]:
        p1 = combine.fit_prune(pv, Mv, Vv, sv[s], thr_rec, higher_is_worse=True)
        arms[s] = {"val": p1["val_iou"], "cutoff": p1["cutoff"],
                   "test": _iou(combine.apply_prune(pt, thr_rec, st[s],
                                                    p1["cutoff"], True),
                                Mt, Vt)}

    order = sorted(arms, key=lambda a: -arms[a]["val"])
    print(f"\n  {'arm':<14}{'val IoU':>10}{'test IoU':>11}")
    print(f"  {'unpruned':<14}{unpruned['val']:>10.5f}{unpruned['test']:>11.5f}")
    for a in order:
        print(f"  {a:<14}{arms[a]['val']:>10.5f}{arms[a]['test']:>11.5f}")
    best_single = max((a for a in arms if a != "stacked"),
                      key=lambda a: arms[a]["val"])
    d = arms["stacked"]["val"] - arms[best_single]["val"]
    print(f"\n  stacked minus best single ({best_single}) on VAL: {d:+.5f}")
    print(f"  stacker coefficients: {arms['stacked']['coefficients']}")

    out = {"regime": ("constitution stage 7: the three error signals stacked by "
                      "a logistic model cross-fit inside val on contiguous "
                      "buffered folds, then ONE prune. Cutoff selected on the "
                      "out-of-fold stacked score over all val slices; test "
                      "scored once. Signals are all oriented so higher means "
                      "more likely wrong."),
           "caveat": ("Previous measurements established only that no SINGLE "
                      "signal beats the free margin baseline. This tests the "
                      "combination. The val slab is 60 slices holding 19 "
                      "chondrules by binary CC or 36 by the same-id rule, so a "
                      "favourable val result is not evidence of transfer."),
           "model": model, "target_recall": float(target_recall),
           "achieved_recall": float(rm["recall"]), "k": int(k),
           "thr_recall": float(thr_rec), "signals": sig_names,
           "unpruned": unpruned, "arms": arms}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kw = {}
    for a in [x for x in argv if x.startswith("--")]:
        key, _, v = a[2:].partition("=")
        if key == "model":
            kw["model"] = v
        elif key == "recall":
            kw["target_recall"] = float(v)
        elif key == "no-unet":
            kw["use_unet"] = False
        else:
            raise SystemExit(f"unknown option {a}")
    return run(**kw)


if __name__ == "__main__":
    main()
