"""Cached zone loader: memory-efficient access to the frozen split.

`split.build_split()` is deterministic but expensive in RAM: it materializes a
full-resolution int32 instance-label volume (~2.2 GB) *twice* (build + verify)
plus the band label volume. Having every search worker recompute that was
costing ~4.4 GB each and pushed the machine into swap.

This module computes it ONCE, caches the per-zone valid masks to disk, and lets
workers load only the zones they actually need. Search workers load
("train", "val"). The test zone is never even read during a search, which is
also the right default for the touch-test-once rule. [LEAKAGE]

Build the cache:  python -m src.zones
"""
from __future__ import annotations
import os
import numpy as np

from . import data, split as splitmod

CACHE = "split_cache.npz"


def cache_zones(root=".", path=CACHE):
    """Run the split once and cache per-zone valid masks + absolute indices."""
    res, slab, ignore, idxs = splitmod.build_split(root)
    payload = {}
    for z in ("train", "val", "test"):
        ii = [i for i in range(len(idxs)) if slab[i] == z]
        payload[f"valid_{z}"] = ~ignore[ii]                       # bool (n,H,W)
        payload[f"abs_{z}"] = np.asarray([idxs[i] for i in ii], np.int32)
    np.savez_compressed(os.path.join(root, path), **payload)
    return res


def load_zones(root=".", zones=("train", "val"), path=CACHE):
    """Load only the requested zones. Returns {zone: (raw f32, mask bool, valid bool)}.

    Raises if the cache is missing so a worker can never silently fall back to
    the expensive full rebuild.
    """
    full = os.path.join(root, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{full} missing: run `python -m src.zones` first")
    d = np.load(full)
    raw_paths, lab_paths = data.list_pairs(root)
    out = {}
    for z in zones:
        abs_i = d[f"abs_{z}"].tolist()
        R = data.load_stack(raw_paths, abs_i, "raw").astype(np.float32)
        M = data.load_stack(lab_paths, abs_i, "mask")
        out[z] = (R, M, d[f"valid_{z}"])
    return out


if __name__ == "__main__":
    r = cache_zones()
    sz = os.path.getsize(CACHE) / 1e6
    print(f"cached split -> {CACHE} ({sz:.1f} MB on disk); "
          f"band={r['band']} straddlers_remaining={r['straddlers_remaining']}")
