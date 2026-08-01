"""Measure what each split protocol leaks, before any model is trained.

Spec R4. For every protocol this reports:

  shared_instances        chondrules with voxels in BOTH train and test
  test_pos_frac_shared    fraction of test positive pixels belonging to one
  min_z_gap               smallest z-distance from a test unit to a train unit
  adjacent_test_frac      fraction of test units with a train unit touching them
  overlap_test_frac       fraction of test units sharing pixels with a train unit

The instance labelling reuses src.split.instance_label so a "chondrule" here
means the same thing it means in the frozen split: a 3D connected component
constrained to a single annotator id.

Run:  python "Audits of Other Papers/01_leakage_diagnostics.py"
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)

from src import data, split as splitmod                      # noqa: E402
sys.path.insert(0, ROOT)
import protocols as P                                        # noqa: E402

OUT = os.path.join(ROOT, "results", "leakage_diagnostics.json")


def load_band_ids(crop=True):
    """Instance-id volume for the annotated band, center-cropped."""
    raw_paths, lab_paths = data.list_pairs(REPO)
    fg = data.fg_fraction(lab_paths)
    b0, b1 = data.annotated_band(fg)
    idxs = list(range(b0, b1 + 1))
    vol_ids = data.load_stack(lab_paths, idxs, kind="raw")
    return P.crop_slices(vol_ids, crop), idxs


def zone_volume_slices(zone_idx, shape):
    """(N,H,W) uint8 zone code: 0 unused, 1 train, 2 val, 3 test."""
    zc = np.zeros(shape, np.uint8)
    for code, z in ((1, "train"), (2, "val"), (3, "test")):
        zc[zone_idx[z]] = code
    return zc


def zone_volume_tiles(zone_coords, shape, tile=P.TILE):
    zc = np.zeros(shape, np.uint8)
    for code, z in ((1, "train"), (2, "val"), (3, "test")):
        for (zz, y0, x0) in zone_coords[z]:
            zc[zz, y0:y0 + tile, x0:x0 + tile] = code
    return zc


def instance_zone_counts(inst, zc, ninst):
    """counts[instance_id, zone_code] of voxels, accumulated slice by slice."""
    counts = np.zeros((ninst + 1, 4), np.int64)
    for z in range(inst.shape[0]):
        iz, cz = inst[z].ravel(), zc[z].ravel()
        for code in (1, 2, 3):
            m = cz == code
            if m.any():
                counts[:, code] += np.bincount(iz[m], minlength=ninst + 1)
    return counts


def slice_gap_stats(zone_idx):
    tr = np.asarray(zone_idx["train"])
    te = np.asarray(zone_idx["test"])
    d = np.abs(te[:, None] - tr[None, :]).min(axis=1)
    return {
        "min_z_gap": int(d.min()),
        "median_z_gap": float(np.median(d)),
        "adjacent_test_frac": float((d <= 1).mean()),
        "within_4_test_frac": float((d <= 4).mean()),
    }


def tile_gap_stats(zone_coords, tile=P.TILE):
    tr = np.asarray(zone_coords["train"], np.int64)
    te = np.asarray(zone_coords["test"], np.int64)
    tr_set = set(map(tuple, tr))
    adj = 0
    for (z, y, x) in te:
        neigh = [(z + dz, y + dy * tile, x + dx * tile)
                 for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                    (0, -1, 0), (0, 0, 1), (0, 0, -1))]
        if any(n in tr_set for n in neigh):
            adj += 1
    # z-distance to the nearest train tile anywhere
    dz = np.abs(te[:, 0][:, None] - tr[:, 0][None, :]).min(axis=1)
    # pixel overlap with any train patch (only possible when patches overlap)
    overlap, ov_frac = 0, []
    for (z, y, x) in te:
        same_z = tr[tr[:, 0] == z]
        if len(same_z) == 0:
            continue
        oy = np.clip(tile - np.abs(same_z[:, 1] - y), 0, tile)
        ox = np.clip(tile - np.abs(same_z[:, 2] - x), 0, tile)
        a = (oy * ox).max()
        if a > 0:
            overlap += 1
            ov_frac.append(a / (tile * tile))
    return {
        "min_z_gap": int(dz.min()),
        "median_z_gap": float(np.median(dz)),
        "adjacent_test_frac": adj / len(te),
        "overlap_test_frac": overlap / len(te),
        "mean_overlap_area_frac": float(np.mean(ov_frac)) if ov_frac else 0.0,
        "n_test_units": int(len(te)),
        "n_train_units": int(len(tr)),
    }


def main(crop=True):
    vol_ids, idxs = load_band_ids(crop)
    N = vol_ids.shape[0]
    print(f"band: N={N} shape={vol_ids.shape} crop={crop}", flush=True)

    inst, ninst = splitmod.instance_label(vol_ids)
    print(f"instances (same-id 3D CC): {ninst}", flush=True)
    pos = vol_ids > 0
    del vol_ids

    report = {
        "band_slices": N,
        "crop": [P.CROP_H, P.CROP_W] if crop else "full",
        "n_chondrule_instances": int(ninst),
        "unit_sizes": {"train": P.N_TRAIN, "val": P.N_VAL, "test": P.N_TEST},
        "protocols": {},
    }

    for name, cfg in P.PROTOCOLS.items():
        sel = cfg["fn"](N, seed=0)
        if cfg["kind"] == "slice":
            zc = zone_volume_slices(sel, inst.shape)
            gaps = slice_gap_stats(sel)
        else:
            zc = zone_volume_tiles(sel, inst.shape)
            gaps = tile_gap_stats(sel)

        # Straddler resolution removes voxels from the zones they do not belong
        # to. Those voxels must leave the diagnostic too, or the protocol that
        # fixes the leak is scored as if it still had it. Recoding them to zone
        # 0 drops them from the instance counts and from the test positive
        # denominator in one move.
        n_ignored = 0
        if cfg.get("straddler"):
            ign = P.straddler_ignore(inst, zc, ninst)
            n_ignored = int((ign & (zc != 0)).sum())
            zc[ign] = 0

        counts = instance_zone_counts(inst, zc, ninst)
        in_train = counts[:, 1] > 0
        in_test = counts[:, 3] > 0
        shared = in_train & in_test
        shared[0] = False                      # id 0 is background

        # Positive pixels in test that belong to a shared chondrule.
        test_pos = 0
        test_pos_shared = 0
        for z in range(inst.shape[0]):
            m = (zc[z] == 3) & pos[z]
            if not m.any():
                continue
            ids = inst[z][m]
            test_pos += ids.size
            test_pos_shared += int(shared[ids].sum())

        n_test_inst = int((in_test & (np.arange(ninst + 1) > 0)).sum())
        entry = {
            "label": cfg["label"],
            "kind": cfg["kind"],
            "straddler_resolution": bool(cfg.get("straddler")),
            "voxels_removed_by_straddler_resolution": n_ignored,
            "n_instances_in_test": n_test_inst,
            "shared_instances_train_and_test": int(shared.sum()),
            "shared_instance_frac_of_test": (
                float(shared.sum() / n_test_inst) if n_test_inst else float("nan")),
            "test_positive_pixels": int(test_pos),
            "test_pos_frac_from_shared_chondrules": (
                float(test_pos_shared / test_pos) if test_pos else float("nan")),
        }
        entry.update(gaps)
        report["protocols"][name] = entry
        print(f"\n{name}: {cfg['label']}", flush=True)
        for k, v in entry.items():
            if k not in ("label", "kind"):
                print(f"    {k}: {v}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)
    return report


if __name__ == "__main__":
    main(crop="--full" not in sys.argv)
