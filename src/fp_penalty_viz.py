"""Show what Sky's mentor's false-positive penalty actually did to the errors.

The proposal, and why it was reasonable
---------------------------------------
The shipped base model overpredicts: 5,805,884 false positives against 2,444,882
false negatives, precision 0.372 against recall 0.584. Punishing false positives
harder is the obvious correction. `run_cascade_arms.arm_loss` implements it as
`tversky_loss(alpha=0.7, beta=0.3)`, weighting a false positive 2.33x a false
negative, the exact mirror of the 0.3/0.7 recall direction already measured to
hurt. Tversky at alpha = beta = 0.5 IS Dice, so the `control` arm is the matched
symmetric baseline.

The scalar answer is in `results/cascade_arms.json`: test IoU 0.2033 +/- 0.0178
against the control's 0.2493 +/- 0.0289, worst of five arms, but inside the
2-sigma bar. This module asks the question a scalar cannot: did the loss do what
it was designed to do? A loss that weights false positives 2.33x should buy
precision by giving up recall. That is a specific, checkable prediction.

[HONESTY] THE MATCHED CONTROL IS NOT IN THIS FIGURE, AND CANNOT BE.
`save_prob` was added to `run_cascade_arms` after the control, cleaned and
cascade arms had already run, so `results/probs/` holds maps for `fp_penalty`
and `full` only. There is no 7,200-step, dice-loss, raw-only map on disk. Rather
than quietly substitute a different arm, this figure BRACKETS the missing
control with the two dice arms that do exist, each differing from it in exactly
one stated way:

    unet2d_s30000   dice, raw only, 30,000 steps   <- differs in BUDGET
    arm_full_s0     dice, raw + 4 aux, 7,200 steps <- differs in INPUT CHANNELS
    arm_fp_penalty  tversky 0.7/0.3, raw only, 7,200 steps

Neither bracket licenses a causal claim on its own. What they do license: if the
penalty had bought precision, `fp_penalty` would sit above BOTH dice arms on
precision at matched recall, and it does not.

To remove the confound properly, run the control and save its map:

    python -m src.run_cascade_arms --arm=control --seed=0

[DEFECT, found here] The AUPRC this module reports for `unet2d_s30000` is 0.309,
against the 0.427 in its leaderboard row. That is not an error in this module.
`metrics.auprc` on the live map returns 0.30937 exactly, strides of 2/4/8 agree
to 0.0005, and no definition tried (no valid mask, per-slice mean, per-slice
median, pre-fix backup map) reproduces 0.427. The same row's IoU and tp/fp/fn
DO match the map. See CHANGELOG.md's KNOWN ISSUES. The numbers in these figures
are measured from the maps on disk; where they disagree with a leaderboard
field, the disagreement is the finding.

Outputs:
  results/fp_penalty_anatomy.json        the numbers behind the figures
  results/figures/fp_penalty_anatomy.png error maps, same slices, every arm
  results/figures/fp_penalty_pr.png      precision-recall frontier

Run:  python -m src.fp_penalty_viz
"""
from __future__ import annotations
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_recall_curve, average_precision_score

from . import zones
from .pipeline_viz import ERR_CMAP, LEGEND, _err, _norm, FIG

OUT = "results/fp_penalty_anatomy.json"

# (label, probability map, val-frozen threshold, short tag for the title line).
# Thresholds come from results/cascade_arms.json for the arms and from the
# leaderboard row for the base; each was selected on VALIDATION and frozen
# before test was scored once.
ARMS = [
    ("dice, raw, 30k steps\n(unet2d_s30000)",
     "results/probs/unet2d_s30000.npy", 0.002, "base"),
    ("dice, raw+aux, 7.2k\n(arm_full_s0)",
     "results/probs/arm_full_s0.npy", 0.90, "full"),
    ("TVERSKY .7/.3, raw, 7.2k\n(fp_penalty s0)",
     "results/probs/arm_fp_penalty_s0.npy", 0.002, "fp0"),
    ("TVERSKY .7/.3, raw, 7.2k\n(fp_penalty s1)",
     "results/probs/arm_fp_penalty_s1.npy", 0.006, "fp1"),
    ("TVERSKY .7/.3, raw, 7.2k\n(fp_penalty s2)",
     "results/probs/arm_fp_penalty_s2.npy", 0.002, "fp2"),
]

# Every 8th valid pixel, deterministic, for the PR curve only. The full test
# zone is ~55M valid pixels and an exact curve over five maps costs minutes for
# a line that is visually identical. The AUPRC printed in the legend is computed
# on the SAME subsample, so the number and the curve always agree with each
# other; `_check_subsample` verifies the subsample against the exact value.
STRIDE = 8


def _confusion(pred, mask, valid):
    tp = int((pred & mask & valid).sum())
    fp = int((pred & ~mask & valid).sum())
    fn = int((~pred & mask & valid).sum())
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "iou": tp / max(tp + fp + fn, 1),
            "fp_over_fn": fp / max(fn, 1)}


def _check_subsample(y, s, exact, tol=0.005):
    """Guard the stride. A subsampled AUPRC that drifts from the exact value
    would put a wrong number in the legend, so this asserts rather than warns."""
    got = float(average_precision_score(y, s))
    if abs(got - exact) > tol:
        raise SystemExit(
            f"stride {STRIDE} AUPRC {got:.4f} differs from exact {exact:.4f} "
            f"by more than {tol}; lower STRIDE or compute exactly")
    return got


def run(root=".", n=2, exact_auprc=None):
    z = zones.load_zones(root, zones=("test",))
    Rt, Mt, Vt = z["test"]
    area = (Mt & Vt).reshape(Mt.shape[0], -1).sum(1)
    idx = sorted(np.argsort(area)[::-1][:n].tolist())

    rows, panels, curves = [], [], []
    for label, path, thr, tag in ARMS:
        P = np.load(path).astype(np.float32)
        pred = (P >= thr) & Vt
        rec = {"arm": label.replace("\n", " "), "map": path, "threshold": thr,
               "tag": tag, **_confusion(pred, Mt, Vt)}
        panels.append([( _err(pred[i] & Vt[i], Mt[i], Vt[i]),
                        float((pred[i] & Mt[i] & Vt[i]).sum() /
                              max(((pred[i] | (Mt[i] & Vt[i])) & Vt[i]).sum(), 1)))
                       for i in idx])
        sub_y = Mt[Vt].ravel()[::STRIDE].astype(np.uint8)
        sub_p = P[Vt].ravel()[::STRIDE]
        ap = float(average_precision_score(sub_y, sub_p))
        if exact_auprc and tag in exact_auprc:
            ap = _check_subsample(sub_y, sub_p, exact_auprc[tag])
        pr, rc, _ = precision_recall_curve(sub_y, sub_p)
        curves.append((label, rc, pr, ap, tag))
        rec["auprc_subsampled"] = ap
        rows.append(rec)
        del P, pred, sub_y, sub_p

    _fig_anatomy(Rt, Mt, Vt, idx, panels, rows)
    _fig_pr(curves, rows)
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "regime": (
                "TEST zone, each arm at its own VAL-frozen threshold, scored "
                "once. AUPRC is threshold-free and computed on every "
                f"{STRIDE}th valid pixel."),
            "caveat": (
                "The matched control (7,200 steps, dice, raw only) has NO saved "
                "probability map: save_prob was added to run_cascade_arms after "
                "that arm ran. The two dice arms here differ from it in budget "
                "and in input channels respectively. Neither is a substitute."),
            "arms": rows}, f, indent=2)
    print(f"wrote {OUT}")
    return rows


def _fig_anatomy(Rt, Mt, Vt, idx, panels, rows):
    ncol = 2 + len(ARMS)
    fig, axes = plt.subplots(len(idx), ncol, figsize=(3.5 * ncol, 3.8 * len(idx)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        axes[r, 0].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 0].set_title(f"raw, test slice {i}", fontsize=10)
        axes[r, 1].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 1].imshow(np.ma.masked_where(~(Mt[i] & Vt[i]), Mt[i]),
                          cmap=ListedColormap(["#00E5FF"]), alpha=.55)
        axes[r, 1].set_title("truth", fontsize=10)
        for c, (label, _, _, _tag) in enumerate(ARMS):
            err, si = panels[c][r]
            axes[r, 2 + c].imshow(err, cmap=ERR_CMAP, vmin=0, vmax=3,
                                  interpolation="nearest")
            axes[r, 2 + c].set_title(f"{label}\nslice IoU {si:.3f}", fontsize=8)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, -1].legend(handles=LEGEND, fontsize=8, loc="upper right",
                       framealpha=.85)
    line = "   ".join(
        f"{r['tag']}: prec {r['precision']:.3f} rec {r['recall']:.3f} "
        f"FP {r['fp'] / 1e6:.2f}M" for r in rows)
    fig.suptitle(
        "Penalising false positives: what the loss actually did\n" + line +
        "\nred is false positive, the thing the penalty was meant to remove. "
        "THE MATCHED CONTROL (dice, raw, 7.2k) HAS NO SAVED MAP; the two dice "
        "arms differ in budget and in input channels respectively.",
        fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, .93))
    plt.savefig(f"{FIG}/fp_penalty_anatomy.png", dpi=110)
    plt.close(fig)
    print("  fp_penalty_anatomy.png")


def _fig_pr(curves, rows):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    style = {"base": ("#1565C0", "-", 2.2), "full": ("#2E7D32", "-", 2.2),
             "fp0": ("#E53935", "--", 1.8), "fp1": ("#E53935", "--", 1.8),
             "fp2": ("#E53935", "--", 1.8)}
    for label, rc, pr, ap, tag in curves:
        col, ls, lw = style[tag]
        ax.plot(rc, pr, ls, color=col, lw=lw,
                label=f"{label.splitlines()[0]}  AUPRC {ap:.3f}")
    for r in rows:
        col = style[r["tag"]][0]
        ax.plot(r["recall"], r["precision"], "o", color=col, ms=8, mec="k",
                mew=.8, zorder=5)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "The precision-recall frontier, threshold-free\n"
        "dots mark each arm's val-frozen operating point.\n"
        "If the penalty bought precision the red curves would sit ABOVE the "
        "blue and green ones.", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG}/fp_penalty_pr.png", dpi=130)
    plt.close(fig)
    print("  fp_penalty_pr.png")


if __name__ == "__main__":
    run()
