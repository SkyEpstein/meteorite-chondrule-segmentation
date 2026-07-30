"""Leakage-free contiguous z-slab split (spec R2, R3; constitution P1).

Two mechanisms together guarantee no chondrule leaks across slabs:

1. **Buffer gaps** of `buffer` z-slices separate train / val / test. This kills
   *near-duplicate* leakage: adjacent kept-slices (4 original slices apart)
   look almost identical, so a slice must not sit right next to a slice in
   another slab.

2. **Straddler resolution.** Chondrules are labeled as 3D connected components
   constrained to the same annotator id (`instance_label`): same-id z-extent
   median 24, p95 48, max 70 band-slices (a plain *binary* CC over-merges
   touching chondrules to max 182, which is why it is not used). The max extent
   (70) is too long to buffer away without destroying the split, so instead of
   dropping, every chondrule whose z-range covers >= 2 of {train,val,test} is
   assigned to the slab holding most of its voxels and its voxels in the other
   core zones are ignored. Verified by an independent same-id CC recount: 0
   chondrules keep labels in >= 2 slabs (~2.6% of positive pixels ignored).

The split is deterministic (no RNG) and written to split.json. Zone geometry is
fixed by band-relative index ranges so the same band always yields the same
split.
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy import ndimage as ndi
from . import data

# Band-relative zone ranges (inclusive). Placed in the chondrule-dense interior
# so val/test carry representative positive mass; train is the remainder.
DEFAULT_VAL_ZONE = (80, 139)     # ~15% of band
DEFAULT_TEST_ZONE = (160, 239)   # ~20% of band
DEFAULT_BUFFER = 12              # band-slices (~48 original slices of separation)


def assign_slabs(N, val_zone, test_zone, buffer):
    """Return an object array slab[0..N-1] in {train,val,test,buffer}."""
    slab = np.array(["train"] * N, dtype=object)
    (v0, v1), (t0, t1) = val_zone, test_zone
    assert 0 <= v0 <= v1 < t0 <= t1 < N, \
        f"zones must be ordered within band: val={val_zone} test={test_zone} N={N}"
    assert t0 - v1 - 1 >= buffer, \
        f"val/test gap {t0 - v1 - 1} < buffer {buffer} (would silently shrink separation)"
    slab[v0:v1 + 1] = "val"
    slab[t0:t1 + 1] = "test"

    def mark_buffer(a, b):
        for i in list(range(max(0, a - buffer), a)) + list(range(b + 1, min(N, b + 1 + buffer))):
            if slab[i] == "train":
                slab[i] = "buffer"

    mark_buffer(v0, v1)
    mark_buffer(t0, t1)
    return slab


def instance_label(vol_ids):
    """3D connected components constrained to the SAME annotator id.

    Two touching voxels are the same chondrule only if they share the same
    nonzero label id. This separates distinct-but-touching chondrules (which a
    plain binary CC fuses into mega-clusters) so that straddler-dropping removes
    only chondrules that genuinely cross a slab seam. Runs per-id on the id's
    bounding box for speed. Returns (int32 label volume, n_instances).
    """
    out = np.zeros(vol_ids.shape, dtype=np.int32)
    n = 0
    for idx, sl in enumerate(ndi.find_objects(vol_ids)):  # bbox per id value
        if sl is None:
            continue
        uid = idx + 1
        comp, k = ndi.label(vol_ids[sl] == uid)
        seg = out[sl]
        m = comp > 0
        seg[m] = comp[m] + n
        n += k
    return out, n


def _component_zone_span(lbl, slab):
    """For each 3D component id, the set of core zones its z-range covers."""
    core = {"train", "val", "test"}
    spans = {}
    for cid, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        z0, z1 = sl[0].start, sl[0].stop
        spans[cid] = set(np.unique(slab[z0:z1])) & core
    return spans


def build_split(root=".", val_zone=DEFAULT_VAL_ZONE, test_zone=DEFAULT_TEST_ZONE,
                buffer=DEFAULT_BUFFER, out="split.json"):
    """Build, verify and persist the split.

    Returns (result_dict, slab, ignore_volume, band_idxs). `ignore_volume` is a
    (N,H,W) bool mask of dropped straddler voxels, recomputable from the saved
    params.
    """
    raw, lab = data.list_pairs(root)
    data.check_alignment(raw, lab)
    fg = data.fg_fraction(lab)
    bstart, bend = data.annotated_band(fg)
    idxs = list(range(bstart, bend + 1))
    vol_ids = data.load_stack(lab, idxs, kind="raw")       # (N,H,W) uint8 instance ids
    vol = vol_ids > 0                                       # binary chondrule mask
    N = vol.shape[0]

    slab = assign_slabs(N, val_zone, test_zone, buffer)
    inst, ninst = instance_label(vol_ids)                  # same-id 3D CC = chondrule instances
    spans = _component_zone_span(inst, slab)
    straddlers = [cid for cid, z in spans.items() if len(z) >= 2]

    # Resolve each straddler to the slab holding most of its voxels; ignore only
    # its voxels in *other* core zones. Keeps large chondrules in the evaluation
    # (rather than discarding them) while guaranteeing each is seen in one slab.
    zonecode = np.array([{"train": 1, "val": 2, "test": 3}.get(str(slab[i]), 0)
                         for i in range(N)])
    counts = np.zeros((ninst + 1, 4))
    for z in range(N):
        counts[:, zonecode[z]] += np.bincount(inst[z].ravel(), minlength=ninst + 1)
    maj = np.zeros(ninst + 1, dtype=np.int64)
    for cid in straddlers:
        maj[cid] = int(np.argmax(counts[cid, 1:4])) + 1   # 1=train 2=val 3=test
    ignore = np.zeros(vol.shape, dtype=bool)
    for z in range(N):
        zc = zonecode[z]
        if zc == 0:                                        # buffer slice: unused anyway
            continue
        m = maj[inst[z]]
        ignore[z] = (m != 0) & (m != zc)                   # straddler voxel outside its slab

    # Verify: no surviving chondrule keeps voxels in >= 2 core zones.
    inst2, _ = instance_label(np.where(ignore, 0, vol_ids).astype(vol_ids.dtype))
    remaining = [cid for cid, z in _component_zone_span(inst2, slab).items() if len(z) >= 2]
    assert not remaining, f"leakage remains after resolving: {len(remaining)} chondrules"

    total_pos = int(vol.sum())
    ignored_pos = int((vol & ignore).sum())
    zone_stats = {}
    for z in ("train", "val", "test"):
        sel = np.array([slab[i] == z for i in range(N)])
        vv, ig = vol[sel], ignore[sel]
        valid = ~ig
        zone_stats[z] = {
            "n_slices": int(sel.sum()),
            "abs_slices": [int(idxs[i]) for i in range(N) if slab[i] == z],
            "pos_frac_valid": float(vv[valid].mean()) if valid.any() else 0.0,
            "pos_pixels": int((vv & ~ig).sum()),
        }

    result = {
        "band": [int(bstart), int(bend)],
        "N": int(N),
        "band_abs_slices": [int(i) for i in idxs],
        "buffer": int(buffer),
        "val_zone_band": list(val_zone),
        "test_zone_band": list(test_zone),
        "slab_per_band_slice": [str(slab[i]) for i in range(N)],
        "n_chondrule_instances": int(ninst),
        "n_straddler_instances": len(straddlers),
        "straddler_resolution": "assign-to-majority-slab; minority-zone voxels ignored",
        "straddlers_remaining": len(remaining),
        "total_positive_pixels": total_pos,
        "ignored_positive_pixels": ignored_pos,
        "ignored_positive_frac": ignored_pos / total_pos if total_pos else 0.0,
        "zone_stats": zone_stats,
    }
    with open(os.path.join(root, out), "w") as f:
        json.dump(result, f, indent=2)
    return result, slab, ignore, idxs
