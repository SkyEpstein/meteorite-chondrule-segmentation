"""Run the COMPLETE model end to end on the post-fix base map, and show it.

Why this exists: `results/final_pipeline.json` records every pipeline stage
against `baseline_test = 0.31676`, which is the PRE-tiler-fix base model. That
base no longer exists. The stages have therefore never been measured on the
model the project actually ships, and the one number suggesting they helped
(prune + z = 0.32141, the withdrawn 0.3214) was both pre-fix and selected on
TEST rather than validation.

This module fixes that. Every stage is fitted on VALIDATION probabilities and
applied to TEST once, so the result is leakage-clean. [LEAKAGE][MUST]

Stage order follows the constitution as Sky reordered it on 2026-07-30:

  0. BASE          the U-Net probability map at its val-frozen threshold
  1. HIERARCHY     superpixel cascade used as a FILTER. Compute only: it is
                   tuned for 99.9% chondrule retention, so it can remove
                   predictions and never add them. Reported, not credited.
  2. Z-REFINE      smooth probabilities along z, (mode, k, thr) fitted on val
  3. OVER-PREDICT  open the base up to a recall target, then
     + PRUNE       drop the pixels the confidence score distrusts

Stage 3 uses the FREE margin baseline |p - 0.5| as its score, because the
confidence bake-off found no learned model beat it by more than 0.0005 IoU.

Outputs:
  results/pipeline_stages.json      per-stage val and test IoU
  results/figures/pipeline_stages.png   the same test slices at every stage
  results/figures/full_pipeline.png     the same slices with the classical
                                        Stage-2 columns the CNN figure omits

Run:  python -m src.pipeline_viz
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import classical_pipeline, combine, metrics, zones

FIG = "results/figures"
OUT = "results/pipeline_stages.json"
MODEL = "unet2d_s30000"
PROB_TEST = f"results/probs/{MODEL}.npy"
PROB_VAL = f"results/probs/{MODEL}_val.npy"
# Stage 2, already computed and cross-fitted by src/classical_crossfit.py.
CLASSICAL_VAL = "results/probs/classical_xf_val.npy"
CLASSICAL_TEST = "results/probs/classical_xf_test.npy"

C_TP, C_FP, C_FN = "#2E7D32", "#E53935", "#FDD835"
ERR_CMAP = ListedColormap(["#0B0B0B", C_TP, C_FP, C_FN])
LEGEND = [Patch(color=C_TP, label="true positive"),
          Patch(color=C_FP, label="false positive"),
          Patch(color=C_FN, label="false negative (missed)")]


def _row(model=MODEL, path="results/leaderboard_all.json"):
    with open(path) as f:
        hits = [r for r in json.load(f)["leaderboard"] if r.get("model") == model]
    if not hits:
        raise SystemExit(f"no leaderboard row for {model}")
    return hits[-1]


def _iou(pred, mask, valid):
    return float(metrics.iou(pred, mask, valid))


def _err(pred, mask, valid):
    e = np.zeros(pred.shape, np.uint8)
    e[pred & mask] = 1
    e[pred & ~mask] = 2
    e[~pred & mask] = 3
    e[~valid] = 0
    return e


def _norm(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def run_stages(root=".", model=MODEL, prob_test=None, prob_val=None, thr0=None):
    """Fit every stage on VAL, apply to TEST once, return per-stage masks.

    `prob_test` and `prob_val` default to results/probs/{model}.npy and
    results/probs/{model}_val.npy. `thr0` is the base threshold: leave it None
    to read the val-frozen threshold from the leaderboard row for `model`, or
    pass it to run an arm that was scored with leaderboard=False and so has no
    row. When `thr0` is given the leaderboard is never opened.
    """
    prob_test = f"results/probs/{model}.npy" if prob_test is None else prob_test
    prob_val = (f"results/probs/{model}_val.npy" if prob_val is None
                else prob_val)
    row = None
    if thr0 is None:
        try:
            row = _row(model)
        except (SystemExit, OSError, KeyError) as exc:
            raise SystemExit(
                f"no val-frozen threshold for model {model}: {exc}. If that "
                "arm is not on the leaderboard, pass thr0 explicitly "
                "(--thr=FLOAT).") from exc
        thr0 = float(row["threshold"])
        src = "from the leaderboard row"
    else:
        thr0 = float(thr0)
        src = "passed in, not from the leaderboard"
    z = zones.load_zones(root, zones=("val", "test"))
    _, Mv, Vv = z["val"]
    Rt, Mt, Vt = z["test"]
    Pv = np.load(prob_val).astype(np.float32)
    Pt = np.load(prob_test).astype(np.float32)
    print(f"  base threshold {thr0:g} (val-frozen, {src})")

    stages = []

    # --- 0. BASE -----------------------------------------------------------
    base_v = Pv >= thr0
    base_t = Pt >= thr0
    stages.append({"stage": "0. base U-Net",
                   "val_iou": _iou(base_v, Mv, Vv),
                   "test_iou": _iou(base_t, Mt, Vt),
                   "detail": f"probability >= {thr0:g}, threshold frozen on val",
                   "mask": base_t})

    # --- 1. Z-REFINE -------------------------------------------------------
    zr = combine.fit_z_refine(Pv, Mv, Vv)
    Pv_z = combine.z_smooth(Pv, k=zr["k"], mode=zr["mode"])
    Pt_z = combine.z_smooth(Pt, k=zr["k"], mode=zr["mode"])
    zr_v, zr_t = Pv_z >= zr["thr"], Pt_z >= zr["thr"]
    stages.append({"stage": "1. + z-refine",
                   "val_iou": _iou(zr_v, Mv, Vv),
                   "test_iou": _iou(zr_t, Mt, Vt),
                   "detail": f"z_smooth mode={zr['mode']} k={zr['k']} "
                             f"thr={zr['thr']:g}, all fitted on val",
                   "mask": zr_t})

    # --- 2. OVER-PREDICT then PRUNE ---------------------------------------
    # Constitution P9: open the base up so nothing is missed, then let the
    # confidence stage remove what it distrusts. A miss is unrecoverable; a
    # false positive is prunable.
    thr_r, rm = combine.recall_threshold(Pv, Mv, Vv, target_recall=0.95)
    score_v = np.abs(Pv - 0.5)          # free margin baseline; low = distrusted
    score_t = np.abs(Pt - 0.5)
    pr = combine.fit_prune(Pv, Mv, Vv, score_v, thr_r, higher_is_worse=False)
    pv_mask = combine.apply_prune(Pv, thr_r, score_v, pr["cutoff"], False)
    pt_mask = combine.apply_prune(Pt, thr_r, score_t, pr["cutoff"], False)
    stages.append({"stage": "2. over-predict + prune",
                   "val_iou": _iou(pv_mask, Mv, Vv),
                   "test_iou": _iou(pt_mask, Mt, Vt),
                   "detail": f"opened to recall {rm['recall']:.4f} at "
                             f"thr={thr_r:.2e}, pruned below margin "
                             f"{pr['cutoff']:.4f} (fitted on val)",
                   "mask": pt_mask,
                   "recall_target_unreachable": bool(
                       rm.get("recall_target_unreachable", False)),
                   "unpruned_val_iou": pr["unpruned_val_iou"],
                   "prune_val_gain": pr["gain"]})

    del Pv, Pv_z, base_v, zr_v, pv_mask, score_v
    return {"row": row, "model": model, "thr0": float(thr0), "z_refine": zr,
            "stages": stages,
            "raw": Rt, "mask": Mt, "valid": Vt, "prob": Pt}


def fig_stages(res, n=3, tag=""):
    """The same test slices at every pipeline stage, with IoU under each.

    A non-empty `tag` writes pipeline_stages_{tag}.png and marks the title, so
    two model versions cannot be confused.
    """
    Rt, Mt, Vt = res["raw"], res["mask"], res["valid"]
    stages = res["stages"]
    area = (Mt & Vt).reshape(Mt.shape[0], -1).sum(1)
    idx = sorted(np.argsort(area)[::-1][:n].tolist())

    ncol = 2 + len(stages)
    fig, axes = plt.subplots(n, ncol, figsize=(3.7 * ncol, 3.9 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        axes[r, 0].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 0].set_title(f"raw — test slice {i}", fontsize=10)
        axes[r, 1].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 1].imshow(np.ma.masked_where(~(Mt[i] & Vt[i]), Mt[i]),
                          cmap=ListedColormap(["#00E5FF"]), alpha=.55)
        axes[r, 1].set_title("truth", fontsize=10)
        for c, st in enumerate(stages):
            pred = st["mask"][i] & Vt[i]
            si = ((pred & Mt[i] & Vt[i]).sum() /
                  max(((pred | (Mt[i] & Vt[i])) & Vt[i]).sum(), 1))
            axes[r, 2 + c].imshow(_err(pred, Mt[i], Vt[i]), cmap=ERR_CMAP,
                                  vmin=0, vmax=3, interpolation="nearest")
            axes[r, 2 + c].set_title(f"{st['stage']}\nslice IoU {si:.3f}",
                                     fontsize=9)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, -1].legend(handles=LEGEND, fontsize=8, loc="upper right",
                       framealpha=.85)
    line = "   ".join(f"{s['stage']}: {s['test_iou']:.4f}" for s in stages)
    suf = f"_{tag}" if tag else ""
    fig.suptitle("The complete model, stage by stage, on the POST-tiler-fix base\n"
                 f"test IoU   {line}\n"
                 "every stage fitted on validation, test scored once"
                 + (f" [{tag}]" if tag else ""),
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, .955))
    plt.savefig(f"{FIG}/pipeline_stages{suf}.png", dpi=110)
    plt.close()
    print(f"  pipeline_stages{suf}.png")


def fig_confidence(res, tag="", n=2):
    """The confidence signals themselves, beside what stage 3 does with them.

    `fig_stages` draws base, z-refine and prune as masks, so the scores that
    drive stage 3 never appear. This figure draws them on the same test slices:
    the free margin baseline |p - 0.5| and the z-agreement score.
    """
    Rt, Mt, Vt, Pt = res["raw"], res["mask"], res["valid"], res["prob"]
    thr0 = float(res["thr0"])
    stages = res["stages"]
    area = (Mt & Vt).reshape(Mt.shape[0], -1).sum(1)
    idx = sorted(np.argsort(area)[::-1][:n].tolist())
    agree = combine.z_agreement(Pt, thr0, k=3)

    ncol = 6   # raw, base prob, margin, z-agreement, base error, pruned error
    fig, axes = plt.subplots(n, ncol, figsize=(3.7 * ncol, 3.9 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        axes[r, 0].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 0].set_title(f"raw micro-CT, test slice {i}", fontsize=9)
        axes[r, 1].imshow(Pt[i], cmap="magma", vmin=0,
                          vmax=max(Pt[i].max(), 1e-3))
        axes[r, 1].set_title("base probability", fontsize=9)
        axes[r, 2].imshow(np.abs(Pt[i] - 0.5), cmap="magma", vmin=0, vmax=0.5)
        axes[r, 2].set_title("margin |p - 0.5|\nlow = distrusted", fontsize=9)
        axes[r, 3].imshow(agree[i], cmap="magma", vmin=0, vmax=1)
        axes[r, 3].set_title(f"z-agreement, k=3 at thr {thr0:g}\n"
                             "1.0 = the z-window agrees", fontsize=9)
        for c, (m, title) in enumerate(
                [(stages[0]["mask"], "base"),
                 (stages[2]["mask"], "over-predict + prune")]):
            pred = m[i] & Vt[i]
            axes[r, 4 + c].imshow(_err(pred, Mt[i], Vt[i]), cmap=ERR_CMAP,
                                  vmin=0, vmax=3, interpolation="nearest")
            axes[r, 4 + c].set_title(
                f"{title}\nslice IoU {_iou(pred, Mt[i], Vt[i]):.3f}", fontsize=9)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 5].legend(handles=LEGEND, fontsize=8, loc="upper right",
                      framealpha=.85)

    suf = f"_{tag}" if tag else ""
    fig.suptitle("confidence signals: the learned regressors and the error "
                 "U-Net gained at most +0.0005 val IoU in the bake-off; the "
                 "free margin baseline was never beaten",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, .95))
    plt.savefig(f"{FIG}/confidence_signals{suf}.png", dpi=110)
    plt.close()
    print(f"  confidence_signals{suf}.png")


def fig_full_pipeline(root=".", n=2, res=None, tag=""):
    """The complete model including the classical work `fig_stages` leaves out.

    `fig_stages` draws the CNN side only, so the classical Stage-2 chain
    (denoise, CLAHE, top-hat) is invisible there even though the constitution
    puts that stage ahead of the CNN. This figure shows those images.

    The classical and CNN columns are ALTERNATIVES, not a chain. The cascade arm
    that would feed the classical map to the CNN as an input channel has never
    been run (constitution 12 marks it UNSETTLED), so the CNN sees raw only.
    Drawing them side by side is the honest layout; a left-to-right composition
    would picture a pipeline that does not exist.

    [LEAKAGE] The classical threshold is selected on the val cross-fitted map and
    then applied to test once. Returns that arm's threshold, val IoU and test IoU
    so `main` can record them.

    `res` is a `run_stages()` result. It is optional so the figure can be drawn
    on its own, but `main` passes the one it already holds: recomputing it would
    load the val and test zones a second time for no new information.

    A non-empty `tag` writes full_pipeline_{tag}.png and marks the title, so two
    model versions cannot be confused.
    """
    if res is None:
        res = run_stages(root)
    Rt, Mt, Vt = res["raw"], res["mask"], res["valid"]
    stages = res["stages"]

    # The val mask is needed only to FIT the classical threshold. Val raw is
    # never drawn, so drop it rather than hold ~0.3 GB for nothing.
    zv = zones.load_zones(root, zones=("val",))["val"]
    Mv, Vv = zv[1], zv[2]
    del zv

    Pv_c = np.load(CLASSICAL_VAL).astype(np.float32)
    thr_c, val_iou_c = metrics.select_threshold(Pv_c, Mv, Vv)
    del Pv_c, Mv, Vv
    Pt_c = np.load(CLASSICAL_TEST).astype(np.float32)
    cls_t = Pt_c >= thr_c
    test_iou_c = _iou(cls_t, Mt, Vt)
    print(f"  classical arm: thr {thr_c:g} fitted on val (val IoU "
          f"{val_iou_c:.5f}), applied to test once for test IoU {test_iou_c:.5f}")

    area = (Mt & Vt).reshape(Mt.shape[0], -1).sum(1)
    idx = sorted(np.argsort(area)[::-1][:n].tolist())

    ncol = 10   # raw, 3 classical channels, classical prob, 4 error maps, truth
    fig, axes = plt.subplots(n, ncol, figsize=(3.7 * ncol, 3.9 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        ch = classical_pipeline.channels(Rt[i])
        axes[r, 0].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 0].set_title(f"raw micro-CT, test slice {i}", fontsize=9)
        for c, (plane, title) in enumerate(
                [(1, "denoised (median, 5 px)"),
                 (2, "CLAHE (local contrast)"),
                 (3, "top-hat (disk, 15 px)")]):
            axes[r, 1 + c].imshow(_norm(ch[..., plane]), cmap="gray")
            axes[r, 1 + c].set_title(title, fontsize=9)
        axes[r, 4].imshow(Pt_c[i], cmap="magma", vmin=0,
                          vmax=max(Pt_c[i].max(), 1e-3))
        axes[r, 4].set_title("classical probability", fontsize=9)
        for c, (m, title) in enumerate(
                [(cls_t, f"classical at val thr {thr_c:.3f}"),
                 (stages[0]["mask"], "CNN base"),
                 (stages[1]["mask"], "CNN + z-refine"),
                 (stages[2]["mask"], "CNN + prune")]):
            pred = m[i] & Vt[i]
            axes[r, 5 + c].imshow(_err(pred, Mt[i], Vt[i]), cmap=ERR_CMAP,
                                  vmin=0, vmax=3, interpolation="nearest")
            axes[r, 5 + c].set_title(
                f"{title}\nslice IoU {_iou(pred, Mt[i], Vt[i]):.3f}", fontsize=9)
        axes[r, 9].imshow(_norm(Rt[i]), cmap="gray")
        axes[r, 9].imshow(np.ma.masked_where(~(Mt[i] & Vt[i]), Mt[i]),
                          cmap=ListedColormap(["#00E5FF"]), alpha=.55)
        axes[r, 9].set_title("truth", fontsize=9)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    # The legend belongs on an error map, so put it on the last one.
    axes[0, 8].legend(handles=LEGEND, fontsize=8, loc="upper right",
                      framealpha=.85)

    tio = [s["test_iou"] for s in stages]
    line2 = ("test IoU, thresholds fitted on val and applied to test once:   "
             f"classical {test_iou_c:.4f}   CNN base {tio[0]:.4f}   "
             f"+ z-refine {tio[1]:.4f}   + prune {tio[2]:.4f}")
    suf = f"_{tag}" if tag else ""
    # [CORRECTED 2026-08-06] This line used to assert flatly that the cascade
    # arm had never been run, which was true when written and became false the
    # moment `arm_full_s0` trained. A figure that contradicts the model it is
    # rendering is worse than one with no caption, so the sentence now depends
    # on which model produced the CNN columns.
    composed = res.get("model", MODEL) != MODEL
    stage3 = ("stage 3 IS composed here: the CNN sees raw plus the cleaned "
              "channels plus the classical probability"
              if composed else
              "stage 3's classical-channel cascade arm has never been run, so "
              "the CNN sees raw only")
    fig.suptitle("The complete model, all seven stages\n"
                 f"{line2}\n"
                 "stage 1 hierarchy is a compute filter (99.9% retention, not "
                 f"shown as a column); {stage3}"
                 + (f" [{tag}]" if tag else ""),
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, .90))
    plt.savefig(f"{FIG}/full_pipeline{suf}.png", dpi=110)
    plt.close()
    print(f"  full_pipeline{suf}.png")
    return {"threshold": float(thr_c),
            "val_iou": float(val_iou_c),
            "test_iou": float(test_iou_c),
            "detail": ("cross-fitted classical map "
                       "(results/probs/classical_xf_{val,test}.npy) thresholded "
                       f"at {thr_c:g}, selected on val. This is an ALTERNATIVE "
                       "stage-2/3 arm, not composed with the CNN: the "
                       "classical-channel cascade has never been run.")}


def _cli(argv):
    """Parse `--model=NAME --thr=FLOAT --tag=NAME`. No args gives an empty dict."""
    opts = {}
    for a in argv:
        for key in ("model", "thr", "tag"):
            if a.startswith(f"--{key}="):
                opts[key] = a.split("=", 1)[1]
                break
        else:
            raise SystemExit(f"unknown argument {a}: expected --model=NAME, "
                             "--thr=FLOAT or --tag=NAME")
    return opts


def main(root=".", argv=None):
    """Run the stage chain and draw it.

    With no arguments this is the default model and today's filenames. With
    `--model=NAME --thr=FLOAT --tag=NAME` it runs that arm instead, writes the
    tagged figures plus the confidence-signal figure, and leaves
    results/pipeline_stages.json untouched.
    """
    opts = _cli(sys.argv[1:] if argv is None else argv)
    model = opts.get("model", MODEL)
    tag = opts.get("tag", "")
    thr0 = float(opts["thr"]) if "thr" in opts else None
    res = run_stages(root, model=model, thr0=thr0)
    print()
    print(f"  {'stage':<26}{'val IoU':>10}{'test IoU':>11}{'test delta':>12}")
    base = res["stages"][0]["test_iou"]
    for s in res["stages"]:
        d = s["test_iou"] - base
        print(f"  {s['stage']:<26}{s['val_iou']:>10.5f}{s['test_iou']:>11.5f}"
              f"{d:>+12.5f}")
    fig_stages(res, tag=tag)
    ca = fig_full_pipeline(root, res=res, tag=tag)
    if opts:
        fig_confidence(res, tag=tag)

    out = {"regime": ("every stage fitted on VALIDATION probabilities and "
                      "applied to TEST once; base is the POST-tiler-fix "
                      f"{model} map. Supersedes results/final_pipeline.json, "
                      "whose stages were measured against the pre-fix base "
                      "(0.31676)."),
           "model": model,
           "base_threshold": float(res["thr0"]),
           "z_refine": res["z_refine"],
           "classical_arm": ca,
           "stages": [{k: v for k, v in s.items() if k != "mask"}
                      for s in res["stages"]]}
    if opts:
        print(f"\nran {model} from argv, so {OUT} was left as it was")
        return out
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
