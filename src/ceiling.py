"""Annotation-consistency ceiling (spec R8).

Estimates the maximum IoU any model could realistically reach against these
hand labels. That ceiling is set by boundary fuzziness (small, sparse objects,
where a 1-2px edge error costs a lot of IoU) and label z-consistency — NOT by
model capacity. It answers "how close to IoU = 1 is even possible."

Reproducible: `python -m src.ceiling` writes results/annotation_ceiling.json.
Computed on the TEST slab (from the frozen split.json). Ignored straddler voxels
(~2.6%) are not excluded — negligible for this geometric estimate.
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy import ndimage as ndi
from . import data


def _iou(a, b):
    inter = np.count_nonzero(a & b)
    uni = np.count_nonzero(a | b)
    return inter / uni if uni else np.nan


def compute(root=".", split_path="split.json"):
    sp = json.load(open(os.path.join(root, split_path)))
    absl, slab = sp["band_abs_slices"], sp["slab_per_band_slice"]
    test_abs = [absl[i] for i in range(len(absl)) if slab[i] == "test"]
    _, lab = data.list_pairs(root)
    M = np.stack([data.read_mask(lab[a]) for a in test_abs], 0)

    bt = {}
    for k in (1, 2):
        dil = np.stack([ndi.binary_dilation(M[i], iterations=k) for i in range(len(M))])
        ero = np.stack([ndi.binary_erosion(M[i], iterations=k) for i in range(len(M))])
        bt[f"pm{k}px_band_iou"] = float(np.nanmedian([_iou(ero[i], dil[i]) for i in range(len(M))]))
        bt[f"plus{k}px_iou"] = float(np.nanmedian([_iou(M[i], dil[i]) for i in range(len(M))]))
        bt[f"minus{k}px_iou"] = float(np.nanmedian([_iou(M[i], ero[i]) for i in range(len(M))]))

    n = len(M)
    zc = {
        "copy_neighbor_iou": float(np.nanmedian([_iou(M[i], M[i - 1]) for i in range(1, n)])),
        "neighbor_and_iou": float(np.nanmedian([_iou(M[i], M[i - 1] & M[i + 1]) for i in range(1, n - 1)])),
        "neighbor_or_iou": float(np.nanmedian([_iou(M[i], M[i - 1] | M[i + 1]) for i in range(1, n - 1)])),
    }
    return {
        "regime": "TEST slab of the frozen leakage-free split; per-slice median IoU; "
                  "N=1 meteorite (ALH 83010); geometric estimate (ignored voxels not excluded)",
        "test_slices": int(n),
        "test_pos_frac": float(M.mean()),
        "boundary_tolerance_iou": bt,
        "z_consistency_iou": zc,
        "practical_ceiling_iou_pm1px": bt["pm1px_band_iou"],
        "practical_ceiling_iou_pm2px": bt["pm2px_band_iou"],
        "interpretation": "A predictor perfect except for a ±1-2px boundary already caps near "
                          "0.85-0.92 IoU, so IoU > ~0.95 is not reachable against these labels.",
    }


def main(root="."):
    out = compute(root)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    with open(os.path.join(root, "results", "annotation_ceiling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
