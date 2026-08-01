"""Replicate the headline pair so an effect size can be stated as an interval.

Spec R6a, A3a. Single-seed cells cannot carry a number. Two families of noise
floor, both persisted rather than asserted:

  fixed rule   results/noise_floors.json: split-draw sd 0.0071 over 40 draws,
               contiguous-window sd 0.0261 over 238 interior windows. These bound
               what the DATA alone does.
  trained      results/replication.json: split-draw sd 0.0331 and test-window sd
               0.0988 over four runs each. These are what a trained-model gap must
               actually clear, and they are 4.7x and 3.8x the fixed-rule figures.

On this data no trained-model gap below roughly 0.15 IoU is separable.

`P-slice-random` against `P-zone-buffer-straddler` is the only protocol pair with
a matched inference path (both score full 768x1280 slices with real halo and
overlap-averaging), matched patch-sampler geometry, and identical pixel counts.
The tile arms carry handicaps that work against them (spec C5), so they are
reported as a lower bound and are not replicated here.

Two sources of variation are swept:

* the leaky arm over 4 split draws (seeds 0-3), which is what a paper using a
  random split would have drawn;
* the leakage-free arm over 4 contiguous test windows, offset along the band,
  which is what this project would have drawn had it placed its zones elsewhere.

Reporting the pair as two distributions rather than two points is what decides
whether an inflation figure can be stated at all. On this data it could not: the
interval spans zero.

Run:  python "Audits of Other Papers/05_replication.py" [--budget 3000]
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)

from src import metrics, split as splitmod                       # noqa: E402
from src import models_cnn as mc                                 # noqa: E402
import protocols as P                                            # noqa: E402
from importlib import import_module                              # noqa: E402
runmod = import_module("02_run_protocols")

OUT = os.path.join(ROOT, "results", "replication.json")

SEEDS = [0, 1, 2, 3]
# Offsets slide the whole (val, test, buffer) block along the band. The block
# spans band slices 68 to 251 at offset 0, so these four all fit inside 0..396
# with the buffer intact and train size unchanged at 213.
WINDOW_OFFSETS = [-60, 0, 60, 120]


def zone_shifted(N, offset, buffer=12):
    """Reference zone geometry translated by `offset` band slices."""
    v0, v1 = 80 + offset, 139 + offset
    t0, t1 = 160 + offset, 239 + offset
    if v0 - buffer < 0 or t1 + buffer >= N:
        raise ValueError(f"offset {offset} pushes the block off the band")
    slab = np.array(["train"] * N, dtype=object)
    slab[v0:v1 + 1] = "val"
    slab[t0:t1 + 1] = "test"
    for a, b in ((v0, v1), (t0, t1)):
        for i in list(range(max(0, a - buffer), a)) + \
                 list(range(b + 1, min(N, b + 1 + buffer))):
            if slab[i] == "train":
                slab[i] = "buffer"
    return {z: np.where(slab == z)[0] for z in ("train", "val", "test")}


def run_cell(z, name, budget):
    epochs, steps = 15, max(1, budget // 15)
    r = mc.CNNRunner(name, lambda in_ch: mc.UNet2D(in_ch=in_ch, base=32, depth=4),
                     mc.dice_loss, k=1, epochs=epochs, steps_per_epoch=steps)
    r.fit({k: z["train"][k] for k in ("raw", "mask", "valid")},
          {k: z["val"][k] for k in ("raw", "mask", "valid")})
    thr = float(r.best_thr)
    pt = r.predict_proba(z["test"]["raw"])
    m = metrics.metrics_from_pred(pt >= thr, z["test"]["mask"], z["test"]["valid"])
    pv = runmod.otsu_probs(z["train"]["raw"], z["val"]["raw"])
    othr, _ = metrics.select_threshold(pv, z["val"]["mask"], z["val"]["valid"])
    po = runmod.otsu_probs(z["train"]["raw"], z["test"]["raw"])
    mo = metrics.metrics_from_pred(po >= othr, z["test"]["mask"], z["test"]["valid"])
    return {
        "test_iou": m["iou"], "test_dice": m["dice"],
        "test_precision": m["precision"], "test_recall": m["recall"],
        "threshold": thr,
        "otsu_test_iou": mo["iou"],
        "gain_over_otsu": m["iou"] - mo["iou"],
        "train_pos_frac": float((z["train"]["mask"] & z["train"]["valid"]).mean()),
        "test_pos_frac": float((z["test"]["mask"] & z["test"]["valid"]).mean()),
    }


def main(budget=3000, crop=True):
    raw, ids = runmod.load_band(crop)
    N = raw.shape[0]
    mask = ids > 0
    inst, ninst = splitmod.instance_label(ids)
    del ids
    gc.collect()
    print(f"band N={N} instances={ninst} budget={budget}", flush=True)

    results = {
        "regime": (f"replication of the headline pair; U-Net at {budget} steps; "
                   f"leaky arm = P-slice-random over 4 split seeds; "
                   f"leakage-free arm = zone+buffer12+straddler over 4 "
                   f"contiguous test windows; both arms score full-slice inputs "
                   f"so the inference path is matched; "
                   f"{'centered 768x1280 crop' if crop else 'full field'}; "
                   f"N=1 meteorite"),
        "budget_steps": budget,
        "runs": [],
    }
    if os.path.exists(OUT):
        with open(OUT) as f:
            results["runs"] = json.load(f).get("runs", [])
    done = {(r["arm"], r["variant"]) for r in results["runs"]}

    jobs = ([("leaky_slice_random", s) for s in SEEDS] +
            [("clean_zone_straddler", o) for o in WINDOW_OFFSETS])

    for arm, variant in jobs:
        if (arm, variant) in done:
            print(f"{arm} {variant}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        row = {"arm": arm, "variant": variant, "budget_steps": budget}
        try:
            if arm == "leaky_slice_random":
                sel = P.slice_random(N, seed=variant)
                valid = np.ones(raw.shape, bool)
            else:
                sel = zone_shifted(N, variant)
                zc = np.zeros(raw.shape, np.uint8)
                for code, zz in ((1, "train"), (2, "val"), (3, "test")):
                    zc[sel[zz]] = code
                valid = ~P.straddler_ignore(inst, zc, ninst)
                del zc
            z = P.materialize_slices(raw, mask, valid, sel)
            del valid
            gc.collect()
            row.update(run_cell(z, f"{arm}_{variant}", budget))
            print(f"{arm:22s} v={variant:4d} test IoU={row['test_iou']:.4f} "
                  f"(otsu {row['otsu_test_iou']:.4f}, gain "
                  f"{row['gain_over_otsu']:+.4f}) "
                  f"trainpos={row['train_pos_frac']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del z
        except Exception as e:
            import traceback
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-600:]
            row["test_iou"] = float("nan")
            print(f"{arm} {variant}: FAILED {e}", flush=True)
        results["runs"].append(row)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()

    # Summary: two distributions, plus the gap with its uncertainty.
    def stats(arm, key="test_iou"):
        v = [r[key] for r in results["runs"]
             if r["arm"] == arm and isinstance(r.get(key), float)
             and r[key] == r[key]]
        if not v:
            return None
        return {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
                if len(v) > 1 else float("nan"),
                "min": float(np.min(v)), "max": float(np.max(v))}

    summ = {}
    for key in ("test_iou", "gain_over_otsu"):
        a, b = stats("leaky_slice_random", key), stats("clean_zone_straddler", key)
        if a and b:
            # Welch t, not z: n=4 per arm. A 1.96 multiplier is anticonservative
            # by roughly 40% here and would understate exactly the uncertainty
            # this replication exists to expose.
            va, vb = (a["sd"] ** 2) / a["n"], (b["sd"] ** 2) / b["n"]
            se = float(np.sqrt(va + vb))
            df = float((va + vb) ** 2 /
                       (va ** 2 / (a["n"] - 1) + vb ** 2 / (b["n"] - 1)))
            from scipy import stats as _st
            tcrit = float(_st.t.ppf(0.975, df))
            summ[key] = {
                "leaky": a, "clean": b,
                "difference": a["mean"] - b["mean"],
                "difference_se": se, "welch_df": df, "t_critical_95": tcrit,
                "interval_method": "Welch t, unpaired difference of means, n=4 per arm",
                "difference_ci95": [a["mean"] - b["mean"] - tcrit * se,
                                    a["mean"] - b["mean"] + tcrit * se],
            }
    results["summary"] = summ
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    for k, s in summ.items():
        print(f"\n{k}: leaky {s['leaky']['mean']:.4f} (sd {s['leaky']['sd']:.4f}, "
              f"n={s['leaky']['n']}) vs clean {s['clean']['mean']:.4f} "
              f"(sd {s['clean']['sd']:.4f}, n={s['clean']['n']})\n"
              f"    difference {s['difference']:+.4f} "
              f"95% CI [{s['difference_ci95'][0]:+.4f}, "
              f"{s['difference_ci95'][1]:+.4f}]", flush=True)
    print(f"\nwrote {OUT}", flush=True)
    return results


if __name__ == "__main__":
    a = sys.argv
    b = int(a[a.index("--budget") + 1]) if "--budget" in a else 3000
    main(budget=b, crop="--full" not in a)
