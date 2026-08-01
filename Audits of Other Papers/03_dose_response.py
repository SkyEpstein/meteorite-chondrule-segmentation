"""Buffer sweep along z. NOT a confound-free measurement, see the retraction below.

Spec R6. The test zone is FIXED at band slices 160-239 for every point on the
curve, so every run is scored on identical pixels, and the training set size is
FIXED at 197 slices, so no run gets more slices than another. That removes the
test-composition confound that the protocol comparison in 02 carries.

IT DOES NOT REMOVE EVERYTHING. An earlier version of this file claimed the
buffer was the only thing that moved. That was wrong, and the retraction is
kept here rather than deleted. Training slices are chosen nearest-first to the
test zone, so widening the buffer backfills from the band edges, and the band
edges are nearly empty of chondrules. Measured training positive fraction falls
monotonically with the buffer:

    buffer  0 -> 0.07475      buffer 12 -> 0.05700
    buffer  4 -> 0.06805      buffer 20 -> 0.05171

That is a 31% relative decline, perfectly collinear with the treatment, and at
buffer 20 forty of the 197 training slices come from band region 0-39 whose mean
positive fraction is 0.00279. A decline in test IoU across this curve is
therefore jointly explained by proximity leakage AND by a collapse in training
foreground, and this design cannot separate them.

The curve is still reported, as a covariate-annotated observation rather than a
causal measurement: `train_pos_frac` is recorded at every point so a reader can
see the confound moving. The Otsu control used in 02 does NOT help here, because
its val-selected operating point is measured identical (0.35) across every
training set on this volume and its fitted cut moves only from 36.357 to 37.354,
so it is very nearly blind to training composition (results/noise_floors.json).

Kept slices are every 4th original slice. The measured minimum separation is
4(b+1) original slices, not 4b, because a buffer of b leaves b+1 slices of gap.

Run:  python "Audits of Other Papers/03_dose_response.py" [--budget 3000]
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

from src import metrics                                          # noqa: E402
from src import models_cnn as mc                                 # noqa: E402
import protocols as P                                            # noqa: E402
from importlib import import_module                              # noqa: E402
runmod = import_module("02_run_protocols")

OUT = os.path.join(ROOT, "results", "dose_response.json")

BUFFERS = [0, 2, 4, 8, 12, 16, 20]
DOSE_TRAIN = 197        # slices available at the widest buffer in the sweep
VAL_ZONE = (80, 139)
TEST_ZONE = (160, 239)


def zones_for_buffer(N, buffer, train_size=DOSE_TRAIN):
    """Fixed val/test zones, fixed train size, only the buffer moves.

    Training slices are chosen NEAREST-FIRST to the test zone, then truncated to
    `train_size`. Widening the buffer therefore removes exactly the slices
    closest to test and backfills from further away, holding the count constant.
    """
    slab = np.array(["train"] * N, dtype=object)
    (v0, v1), (t0, t1) = VAL_ZONE, TEST_ZONE
    slab[v0:v1 + 1] = "val"
    slab[t0:t1 + 1] = "test"
    for a, b in ((v0, v1), (t0, t1)):
        for i in list(range(max(0, a - buffer), a)) + \
                 list(range(b + 1, min(N, b + 1 + buffer))):
            if slab[i] == "train":
                slab[i] = "buffer"
    tr = np.where(slab == "train")[0]
    if len(tr) < train_size:
        raise ValueError(f"buffer {buffer}: only {len(tr)} train slices available, "
                         f"need {train_size}")
    centre = (t0 + t1) / 2.0
    tr = np.sort(tr[np.argsort(np.abs(tr - centre))[:train_size]])
    return {"train": tr,
            "val": np.where(slab == "val")[0],
            "test": np.where(slab == "test")[0]}


def min_gap(sel):
    tr, te = sel["train"], sel["test"]
    return int(np.abs(te[:, None] - tr[None, :]).min())


def main(budget=3000, crop=True):
    raw, ids = runmod.load_band(crop)
    N = raw.shape[0]
    mask = ids > 0
    valid = np.ones(raw.shape, bool)
    del ids
    gc.collect()

    epochs, steps = 15, max(1, budget // 15)
    results = {
        "regime": (f"buffer dose-response; test zone FIXED at band slices "
                   f"{TEST_ZONE[0]}-{TEST_ZONE[1]} and train size FIXED at "
                   f"{DOSE_TRAIN} slices for every point. NOT confound-free: "
                   f"train positive fraction falls monotonically with buffer "
                   f"(see train_pos_frac per point), collinear with the "
                   f"treatment, so proximity and training-foreground collapse "
                   f"cannot be separated here; budget {budget} steps; U-Net; "
                   f"{'centered 768x1280 crop' if crop else 'full field'}; "
                   f"no straddler resolution at any point (buffer is the only "
                   f"lever under test); N=1 meteorite"),
        "budget_steps": budget,
        "train_size": DOSE_TRAIN,
        "test_zone_band": list(TEST_ZONE),
        "points": [],
    }
    if os.path.exists(OUT):
        with open(OUT) as f:
            results["points"] = json.load(f).get("points", [])
    done = {p["buffer"] for p in results["points"]}

    for b in BUFFERS:
        if b in done:
            print(f"buffer {b}: cached, skipping", flush=True)
            continue
        sel = zones_for_buffer(N, b)
        z = P.materialize_slices(raw, mask, valid, sel)
        gap = min_gap(sel)
        t0 = time.time()
        pt_row = {"buffer": b, "min_z_gap_band_slices": gap,
                  "min_gap_original_slices": gap * 4,
                  "n_train": int(len(sel["train"])),
                  "budget_steps": budget,
                  "train_pos_frac": float(
                      (z["train"]["mask"] & z["train"]["valid"]).mean()),
                  "test_pos_frac": float(
                      (z["test"]["mask"] & z["test"]["valid"]).mean())}
        try:
            r = mc.CNNRunner(f"dose_b{b}",
                             lambda in_ch: mc.UNet2D(in_ch=in_ch, base=32, depth=4),
                             mc.dice_loss, k=1, epochs=epochs,
                             steps_per_epoch=steps)
            r.fit({k: z["train"][k] for k in ("raw", "mask", "valid")},
                  {k: z["val"][k] for k in ("raw", "mask", "valid")})
            thr = float(r.best_thr)
            pt = r.predict_proba(z["test"]["raw"])
            m = metrics.metrics_from_pred(pt >= thr, z["test"]["mask"],
                                          z["test"]["valid"])
            pt_row.update({"test_iou": m["iou"], "test_dice": m["dice"],
                           "test_precision": m["precision"],
                           "test_recall": m["recall"], "threshold": thr})
            print(f"buffer={b:2d} (gap {gap} band / {gap * 4} original slices) "
                  f"test IoU={m['iou']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            del pt
        except Exception as e:
            import traceback
            pt_row["error"] = f"{type(e).__name__}: {e}"
            pt_row["traceback"] = traceback.format_exc()[-600:]
            pt_row["test_iou"] = float("nan")
            print(f"buffer={b}: FAILED {e}", flush=True)
        results["points"].append(pt_row)
        results["points"].sort(key=lambda p: p["buffer"])
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        del z
        gc.collect()

    print(f"wrote {OUT}", flush=True)
    return results


if __name__ == "__main__":
    a = sys.argv
    b = int(a[a.index("--budget") + 1]) if "--budget" in a else 3000
    main(budget=b, crop="--full" not in a)
