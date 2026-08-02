"""Does MORE ANNOTATED DATA help? — the corrected scaling test.

Why this file exists
--------------------
`results/data_scaling.json` (the first attempt) reported

    25% -> 0.2421 | 50% -> 0.2972 | 75% -> 0.2593 | 100% -> 0.2314

which is non-monotonic and *worst at 100% data*. That result is not evidence
that data hurts; it is evidence that the experiment was confounded. Two defects:

  1. **Contiguous subsets.** Each fraction took a contiguous block of the train
     slab, so "25% of the data" was also "one particular region of the
     meteorite". Region difficulty and data volume were varied together and
     cannot be separated.
  2. **One seed per point.** The bootstrap SE on a val-IoU difference here is
     ~0.0071 IoU (19 validation chondrules). Single-seed differences of 0.03-0.06
     sit only ~4-8 SE apart *before* counting training-seed variance, which is
     itself large. With n=1 per point there is no way to tell a real trend from
     run-to-run scatter.

This module fixes both:

  * subsets are drawn **at random across the whole train slab** (a 25% subset
    still spans the full z-range, so region and volume are decoupled);
  * every fraction is run with **3 seeds**, and the curve is reported as
    mean +/- SD so the trend can be compared against the noise.

Budget caveat [HONESTY]
-----------------------
Every point trains for a FIXED 7,200 steps. That is deliberate - it holds
compute constant so the only thing varying is how many distinct annotated
slices the model sees. It also means the absolute IoUs here are BELOW the
30,000-step headline: budget was the largest single lever we measured.
[WITHDRAWN] The 0.3159/0.3168 anchors were produced by the defective inference
tiler; the post-fix retrain scores 0.29409. Read this curve for its SHAPE, not
its height, and do not compare its heights against pre-fix numbers.

The subsets are drawn from TRAIN only. Val and test are untouched and identical
across every run, so the scores stay comparable. [LEAKAGE][MUST]

Run:  python -m src.data_scaling
"""
from __future__ import annotations
import json
import os
import numpy as np

from . import evaluate
from .models_cnn import CNNRunner, UNet2D, dice_loss

OUT = "results/data_scaling_v2.json"
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)
STEPS = 7200
EPOCHS = 24                       # 24 x 300 = 7,200 steps
STEPS_PER_EPOCH = STEPS // EPOCHS


def subset_train(train, frac, seed):
    """Random (not contiguous) subset of train slices. [LEAKAGE] train-only."""
    n = train["raw"].shape[0]
    k = max(1, int(round(frac * n)))
    idx = np.sort(np.random.default_rng(1000 + seed).permutation(n)[:k])
    return {"raw": train["raw"][idx], "mask": train["mask"][idx],
            "valid": train["valid"][idx]}, idx


def load_done(path=OUT):
    """Resume support: completed (frac, seed) runs are never repeated."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("runs", [])


def save(runs, path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    by_frac = {}
    for r in runs:
        if r.get("test_iou") == r.get("test_iou"):        # drop NaN/errored
            by_frac.setdefault(r["frac"], []).append(r["test_iou"])
    summary = []
    for f in sorted(by_frac):
        v = np.asarray(by_frac[f], float)
        summary.append({
            "frac": f,
            "n_slices": int(round(f * 213)),
            "n_seeds": int(v.size),
            "test_iou_mean": float(v.mean()),
            "test_iou_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "test_iou_all": [float(x) for x in v],
        })
    with open(path, "w") as f:
        json.dump({
            "regime": (
                "TRAIN subsets drawn at RANDOM across the full train slab; "
                f"fixed {STEPS}-step budget for every point so compute is held "
                "constant; val/test slabs identical and untouched across runs; "
                "3 seeds per fraction; foreground IoU on TEST, threshold "
                "selected on VAL. N=1 meteorite (ALH 83010)."),
            "caveat": (
                f"Absolute IoU is depressed by the fixed {STEPS}-step budget "
                "(the 30,000-step run of the same model scores 0.29409 post-tiler-fix; "
                "the 0.3159/0.3168 anchors are WITHDRAWN). Read "
                "the SHAPE of the curve, not its height."),
            "supersedes": "results/data_scaling.json (contiguous subsets, 1 seed)",
            "summary": summary,
            "runs": runs,
        }, f, indent=2)
    return summary


def run(root="."):
    d = evaluate.load(root)
    runs = load_done()
    done = {(r["frac"], r["seed"]) for r in runs}
    print(f"train slices available: {d['train']['raw'].shape[0]}", flush=True)

    for frac in FRACTIONS:
        for seed in SEEDS:
            if (frac, seed) in done:
                print(f"skip frac={frac} seed={seed} (already done)", flush=True)
                continue
            sub, idx = subset_train(d["train"], frac, seed)
            name = f"ds2_f{int(frac * 100)}_s{seed}"
            print(f"\n=== {name}: {sub['raw'].shape[0]} slices, "
                  f"{STEPS} steps ===", flush=True)
            runner = CNNRunner(
                name, lambda ic: UNet2D(in_ch=ic, base=32, depth=4), dice_loss,
                epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch=16,
                patch=128, pos_frac=0.5, lr=3e-4, seed=seed)
            # Score through the shared harness so the leakage rules and the
            # threshold-on-val discipline are identical to every other model.
            # [MUST] leaderboard=False: these runs train on SUBSETS at a reduced
            # 7.2k budget, so they are not comparable to the frozen-split roster.
            # They used to land in results/leaderboard_all.json, which
            # fig_leaderboard plots as one ranked chart — a `ds2_*` row there
            # reads as a model that beat or lost to unet2d_s30000. Their home is
            # OUT (results/data_scaling_v2.json) with mean +/- SD per fraction.
            row = evaluate.run_and_score(
                runner, data={"train": sub, "val": d["val"], "test": d["test"]},
                extra={"note": f"data-scaling frac={frac} seed={seed}"},
                leaderboard=False)
            runs.append({
                "frac": frac, "seed": seed, "n_slices": int(sub["raw"].shape[0]),
                "steps": STEPS,
                "test_iou": float(row.get("iou", float("nan"))),
                "val_iou": float(getattr(runner, "best_full_val_iou", float("nan"))),
                "threshold": float(row.get("threshold", float("nan"))),
                "error": row.get("error"),
            })
            save(runs)
            print(f"    -> test IoU {runs[-1]['test_iou']:.4f}", flush=True)

    summary = save(runs)
    print("\n=== corrected data-scaling curve (test foreground IoU) ===")
    print(f"{'slices':>8} {'frac':>6} {'mean':>8} {'SD':>8}   runs")
    for s in summary:
        sd = "   n/a" if s["test_iou_sd"] != s["test_iou_sd"] else f"{s['test_iou_sd']:.4f}"
        vals = " ".join(f"{v:.4f}" for v in s["test_iou_all"])
        print(f"{s['n_slices']:>8} {s['frac']:>6.2f} {s['test_iou_mean']:>8.4f} "
              f"{sd:>8}   {vals}")
    if len(summary) >= 2:
        lo, hi = summary[0], summary[-1]
        delta = hi["test_iou_mean"] - lo["test_iou_mean"]
        print(f"\n{lo['n_slices']} -> {hi['n_slices']} slices: "
              f"{delta:+.4f} IoU (absolute)")
        pooled = np.nanmean([s["test_iou_sd"] for s in summary
                             if s["test_iou_sd"] == s["test_iou_sd"]])
        if pooled == pooled:
            print(f"pooled seed SD = {pooled:.4f} IoU; the change is "
                  f"{abs(delta) / pooled:.1f}x that SD")
    return summary


if __name__ == "__main__":
    run()
