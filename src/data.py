"""Data loading & label binarization (spec R1).

Pairs each raw CT slice with its label slice by sorted z-order, binarizes the
labels (any value > 0 == chondrule), and locates the annotated central band.

Alignment: raw files are `ALH83010_z_0000..2120.tif` (every 4th original slice)
and labels are `ALHA83010_labeled_0000..0530.tif` (sequential). Sorted order
puts index i at the same physical depth in both lists.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import tifffile

RAW_DIR = "ALH83010_z_crop_every4"
LAB_DIR = "ALH83010_z_crop_every4_labeled"


def list_pairs(root="."):
    """Return (raw_paths, lab_paths), sorted and 1:1 by z-order."""
    raw = sorted(glob.glob(os.path.join(root, RAW_DIR, "*.tif")))
    lab = sorted(glob.glob(os.path.join(root, LAB_DIR, "*.tif")))
    if not raw or not lab:
        raise FileNotFoundError(f"missing tifs under {RAW_DIR}/ or {LAB_DIR}/ (root={root!r})")
    if len(raw) != len(lab):
        raise ValueError(f"slice count mismatch: raw={len(raw)} lab={len(lab)}")
    return raw, lab


def read_raw(path):
    return np.asarray(tifffile.imread(path))


def read_mask(path):
    """Binary chondrule mask: label > 0."""
    return np.asarray(tifffile.imread(path)) > 0


def check_alignment(raw_paths, lab_paths, n=5):
    """Assert a sample of raw/label pairs share dimensions. Returns the shape."""
    shape = None
    idxs = np.linspace(0, len(raw_paths) - 1, n).astype(int)
    for i in idxs:
        rs, ls = read_raw(raw_paths[i]).shape, read_mask(lab_paths[i]).shape
        if rs != ls:
            raise ValueError(f"raw/label shape mismatch at {i}: {rs} vs {ls}")
        shape = rs
    return shape


def fg_fraction(lab_paths):
    """Foreground (chondrule) pixel fraction per slice."""
    return np.array([read_mask(p).mean() for p in lab_paths], dtype=float)


def annotated_band(fg_frac, max_gap=10):
    """Longest run of labeled slices, merging interior gaps <= max_gap slices.

    Returns (start, end) inclusive indices into the sorted slice list. This
    isolates the central annotated band and drops the un-annotated end runs
    (which are treated as unlabeled per the constitution, never as negatives).
    """
    present = np.where(fg_frac > 0)[0]
    if present.size == 0:
        raise ValueError("no labeled slices found")
    groups, s, prev = [], present[0], present[0]
    for i in present[1:]:
        if i - prev > max_gap:
            groups.append((s, prev))
            s = i
        prev = i
    groups.append((s, prev))
    start, end = max(groups, key=lambda g: g[1] - g[0])
    return int(start), int(end)


def load_stack(paths, idxs, kind="mask"):
    """Load a (len(idxs), H, W) stack for the given slice indices."""
    read = read_mask if kind == "mask" else read_raw
    return np.stack([read(paths[i]) for i in idxs], axis=0)
