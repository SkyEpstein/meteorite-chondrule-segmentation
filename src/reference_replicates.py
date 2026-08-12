"""Replicate `unet2d_s30000`, the number every comparison in this repo is against.

Why this exists
---------------
`unet2d_s30000` (test IoU 0.29409) is the project's reference model. The
pretraining probe, the cascade arms, the geometry-augmentation result, the
complete-model stages and the README's status block all quote a gap measured
against it. It has never been run twice.

On 2026-08-08 the full-budget seed SD was measured at **0.04517** on an identical
configuration (`resunet18_scratch_s30000`, three seeds spanning 0.08135). That
makes the reference a single draw from a distribution about that wide. If 0.29409
sits high or low in its own spread, every gap quoted against it moves by the same
amount, in the same direction, all at once. Replicating it does not test any
method. It is what makes the existing comparisons interpretable.

The recipe
----------
`results/leaderboard_all.json` records the row's recipe as
`phase1:p1_ctrl_b32 (val_iou=0.3874)`. `results/phase1/p1_ctrl_b32.json` gives
loss=dice, lr=3e-4, base=32, wd=0.0, aug=false, pos_frac=0.5. It does NOT record
depth, patch, batch or val_stride. Those come from `pipeline_v2.BEST_RECIPE`,
which is the recorded form of the same winning configuration and agrees with
p1_ctrl_b32 on every field the two share: depth=4, patch=128, batch=16,
val_stride=6. The budget is 60 x 500 = 30,000 steps, from the `_s30000` suffix.

[HONESTY] This is a faithful reconstruction, not a byte-identical re-run. Four
fields were recovered from a second source rather than from the run's own record,
and seed 0 predates checkpointing so there is no artifact to diff against. If a
replicate lands far from 0.29409, "the recipe was reconstructed wrong" is a live
explanation alongside "the reference was a lucky draw", and the two cannot be
separated from these runs alone. That the recipe is not fully recorded in one
place is itself worth fixing.

[MUST] Seed 0 keeps the historical name `unet2d_s30000`. Replicates are
`unet2d_s30000_seed{n}`. Writing a replicate under the bare name would overwrite
the reference's probability maps and, because `append_leaderboard` dedupes on
`model`, its leaderboard row too. That is the same clobber found in
`pretrained_probe` on 2026-08-08, before it had fired.

`leaderboard=True`, deliberately and unlike `run_cascade_arms`: these ARE the
frozen split, the frozen budget and the frozen recipe, which is exactly the
roster the leaderboard's regime string describes. The replicates belong in it.

Run:  python -m src.reference_replicates --seed=1,2
      python -m src.reference_replicates --report
"""
from __future__ import annotations
import json
import os
import statistics
import sys
import time

from . import evaluate
from .models_cnn import CNNRunner, UNet2D, dice_loss

OUT = "results/reference_replicates.json"
BASE_NAME = "unet2d_s30000"
REFERENCE_IOU = 0.29409          # seed 0, the value every gap is quoted against
EPOCHS, STEPS_PER_EPOCH = 60, 500
RECIPE = dict(base=32, depth=4, patch=128, batch=16, pos_frac=0.5, lr=3e-4,
              augment=False, val_stride=6)


def name_for(seed):
    return BASE_NAME if seed == 0 else f"{BASE_NAME}_seed{seed}"


def load_done():
    try:
        with open(OUT) as f:
            return json.load(f).get("runs", [])
    except (OSError, ValueError):
        return []


def save(runs):
    ious = [r["test_iou"] for r in runs if r.get("test_iou") == r.get("test_iou")]
    summary = {"n": len(ious), "test_iou_all": sorted(ious, reverse=True)}
    if ious:
        summary["test_iou_mean"] = float(statistics.fmean(ious))
    if len(ious) > 1:
        summary["test_iou_sd"] = float(statistics.stdev(ious))
        summary["spread"] = float(max(ious) - min(ious))
    with open(OUT, "w") as f:
        json.dump({
            "regime": (
                "UNet2D base 32 depth 4, dice, 30,000 steps, lr 3e-4, "
                "pos_frac 0.5, batch 16, patch 128, no augmentation; the FROZEN "
                "split; threshold selected on VAL and frozen for a single TEST "
                "scoring. Seed 0 is the historical `unet2d_s30000` row and was "
                "NOT re-run; its recipe was reconstructed from "
                "results/phase1/p1_ctrl_b32.json plus pipeline_v2.BEST_RECIPE, "
                "so replicates are a faithful reconstruction rather than a "
                "byte-identical re-run."),
            "purpose": (
                "The reference model had n=1 while the measured full-budget seed "
                "SD is 0.04517. Every gap in this repo is quoted against it."),
            "reference_seed0_iou": REFERENCE_IOU,
            "summary": summary, "runs": runs}, f, indent=2)
    return summary


def run(seeds=(1, 2), root="."):
    runs = load_done()
    done = {int(r["seed"]) for r in runs}
    for seed in seeds:
        if seed == 0:
            print("refusing to re-run seed 0: it IS the reference row, and "
                  "rewriting it would destroy the number this module exists to "
                  "put error bars on", flush=True)
            continue
        if seed in done:
            print(f"skip seed={seed} (already done)", flush=True)
            continue
        nm = name_for(seed)
        print(f"\n=== {nm}: {EPOCHS * STEPS_PER_EPOCH} steps ===", flush=True)
        t0 = time.time()
        runner = CNNRunner(
            nm, lambda ic: UNet2D(in_ch=ic, base=RECIPE["base"],
                                  depth=RECIPE["depth"]),
            dice_loss, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
            batch=RECIPE["batch"], patch=RECIPE["patch"],
            pos_frac=RECIPE["pos_frac"], lr=RECIPE["lr"],
            augment=RECIPE["augment"], val_stride=RECIPE["val_stride"],
            seed=seed, ckpt_dir="results/checkpoints")
        row = evaluate.run_and_score(
            runner, root=root,
            extra={"recipe": "phase1:p1_ctrl_b32 (reconstructed)", "seed": seed,
                   "replicate_of": BASE_NAME},
            save_prob=f"results/probs/{nm}.npy", leaderboard=True)
        runs.append({
            "seed": int(seed), "model": nm,
            "test_iou": float(row.get("iou", float("nan"))),
            "val_iou": float(getattr(runner, "best_full_val_iou", float("nan"))),
            "threshold": float(row.get("threshold", float("nan"))),
            "sec": round(time.time() - t0, 1), "error": row.get("error")})
        save(runs)
        print(f"    -> test {runs[-1]['test_iou']:.5f}  "
              f"val {runs[-1]['val_iou']:.4f}", flush=True)
    return report(save(runs))


def report(summary):
    ious = [REFERENCE_IOU] + list(summary.get("test_iou_all", []))
    print("\n=== the reference model, with replicates ===")
    print(f"  seed 0 (the reference, not re-run): {REFERENCE_IOU:.5f}")
    for v in summary.get("test_iou_all", []):
        print(f"  replicate:                          {v:.5f}")
    if len(ious) > 1:
        m, sd = statistics.fmean(ious), statistics.stdev(ious)
        print(f"\n  all seeds: n={len(ious)}  mean {m:.5f}  SD {sd:.5f}  "
              f"spread {max(ious) - min(ious):.5f}")
        print(f"  the reference sits {(REFERENCE_IOU - m) / sd:+.2f} SD from the "
              f"mean of its own configuration")
        print("  every gap quoted against 0.29409 moves by "
              f"{m - REFERENCE_IOU:+.5f} if the mean is used instead")
        print("\n  [HONESTY] n=3 SD has 2 degrees of freedom and a wide interval. "
              "This does not")
        print("            re-open any comparison on its own; it says how much "
              "the anchor itself moves.")
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds, report_only = (1, 2), False
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "seed":
            seeds = tuple(int(x) for x in v.split(","))
        elif k == "report":
            report_only = True
        else:
            raise SystemExit(f"unknown option {a}")
    if report_only:
        return report(save(load_done()))
    return run(seeds)


if __name__ == "__main__":
    main()
