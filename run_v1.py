"""v0/v1 pipeline (spec R6): build split -> classical baseline -> metrics + figure.

Run:  python run_v1.py
Writes: split.json, results/v1_baseline.json, results/figures/v1_test_overlays.png
"""
from __future__ import annotations
import json
import os
import platform
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data, split as splitmod, baseline, metrics

ROOT = "."


def _versions():
    import numpy, scipy, skimage, sklearn, tifffile, matplotlib as mpl
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "skimage": skimage.__version__, "sklearn": sklearn.__version__,
        "tifffile": tifffile.__version__, "matplotlib": mpl.__version__,
    }


def _load_zone(raw, lab, idxs, slab, ignore, zone):
    ii = [i for i in range(len(idxs)) if slab[i] == zone]
    abs_i = [idxs[i] for i in ii]
    R = data.load_stack(raw, abs_i, "raw").astype(np.float32)
    M = data.load_stack(lab, abs_i, "mask")
    IG = ignore[ii]
    return R, M, ~IG, abs_i


def main():
    t0 = time.time()
    assert metrics._selfcheck(), "metrics self-check failed"  # A2

    res, slab, ignore, idxs = splitmod.build_split(ROOT)
    print("=== SPLIT ===")
    for k in ("band", "N", "buffer", "n_chondrule_instances", "n_straddler_instances",
              "straddlers_remaining", "ignored_positive_frac"):
        print(f"  {k}: {res[k]}")
    for z in ("train", "val", "test"):
        s = res["zone_stats"][z]
        print(f"  {z:5s}: {s['n_slices']:3d} slices  pos_frac_valid={s['pos_frac_valid']*100:5.2f}%"
              f"  pos_px={s['pos_pixels']:,}")

    raw, lab = data.list_pairs(ROOT)
    Rv, Mv, Vv, _ = _load_zone(raw, lab, idxs, slab, ignore, "val")
    Rt, Mt, Vt, abs_t = _load_zone(raw, lab, idxs, slab, ignore, "test")

    # Tune intensity threshold on validation (touch test only once, after this).
    thr, val_iou = baseline.tune_threshold(Rv, Mv, Vv)
    print(f"\n=== BASELINE (threshold+morphology) ===\n  tuned intensity thr={thr}  val IoU={val_iou:.4f}")

    # Frozen threshold -> test.
    pred_t = baseline.predict(Rt, thr)
    prob_t = Rt / 255.0  # fixed scale (not test-derived) for an honest pseudo-prob ECE
    test = metrics.metrics_from_pred(pred_t, Mt, Vt)
    test["auprc"] = metrics.auprc(prob_t, Mt, Vt)
    test["ece"] = metrics.expected_calibration_error(prob_t, Mt, Vt)
    psi = metrics.per_slice_iou(pred_t, Mt, Vt)
    test["per_slice_iou_median"] = float(np.nanmedian(psi))
    test["per_slice_iou_iqr"] = [float(np.nanpercentile(psi, 25)), float(np.nanpercentile(psi, 75))]

    print(f"  TEST  IoU={test['iou']:.4f}  Dice={test['dice']:.4f}  "
          f"P={test['precision']:.3f}  R={test['recall']:.3f}  "
          f"AUPRC={test['auprc']:.3f}  ECE={test['ece']:.3f}")
    print(f"  TEST per-slice IoU median={test['per_slice_iou_median']:.3f} "
          f"IQR={test['per_slice_iou_iqr']}")

    out = {
        "regime": "leakage-free contiguous z-slab split; threshold tuned on val, "
                  "frozen for test; foreground=chondrule; N=1 meteorite (ALH 83010)",
        "baseline": {"method": "intensity>=thr + closing + remove_small_objects",
                     "tuned_intensity_threshold": int(thr), "val_iou": float(val_iou)},
        "test_metrics": test,
        "split_summary": {k: res[k] for k in
                          ("band", "N", "buffer", "n_straddler_instances",
                           "ignored_positive_frac")},
        "versions": _versions(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    os.makedirs("results/figures", exist_ok=True)
    with open("results/v1_baseline.json", "w") as f:
        json.dump(out, f, indent=2)

    # Qualitative overlays: 3 densest test slices, raw | GT | prediction.
    order = np.argsort([-(Mt[i] & Vt[i]).sum() for i in range(len(Mt))])[:3]
    fig, ax = plt.subplots(len(order), 3, figsize=(13, 3.4 * len(order)))
    for r, i in enumerate(order):
        ax[r, 0].imshow(Rt[i], cmap="gray"); ax[r, 0].set_title(f"raw test z={abs_t[i]}")
        ax[r, 1].imshow(Rt[i], cmap="gray")
        ax[r, 1].imshow(np.ma.masked_where(~Mt[i], Mt[i]), cmap="autumn", alpha=.6)
        ax[r, 1].set_title("ground truth")
        ax[r, 2].imshow(Rt[i], cmap="gray")
        ax[r, 2].imshow(np.ma.masked_where(~pred_t[i], pred_t[i]), cmap="winter", alpha=.6)
        ax[r, 2].set_title(f"baseline pred (thr={thr})")
        for c in range(3):
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("results/figures/v1_test_overlays.png", dpi=90, bbox_inches="tight")
    print(f"\nwrote results/v1_baseline.json and results/figures/v1_test_overlays.png "
          f"({out['runtime_sec']}s)")


if __name__ == "__main__":
    main()
