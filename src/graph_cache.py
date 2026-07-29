"""Disk cache for the tier-D superpixel graphs (built ONCE, shared by all GNNs).

Building one slice graph costs ~9.5 s — SLIC plus the 21-channel
`features.extract_features` stack averaged over every superpixel — and every
GNN variant would otherwise redo that identical work on every run. This module
builds each slice graph once and stores it under `results/graph_cache/`.

Stored per slice (`np.savez_compressed`):

  * `x`           node features, float32 (N, Fn)
  * `edge_index`  adjacency in COMPACT COO form, int32 (2, E), both directions
                  kept — never a dense N x N matrix (N ~ 6000 per slice)
  * `labels`      superpixel id map, int16 (H, W); N < 32767 at N_SEGMENTS=6000
                  so int16 is exact and halves the file size versus int32

Only RAW slices enter graph construction here — no mask and no VALID mask — so
the cache is exactly what `predict_proba` would have computed and adds no
leakage path. [LEAKAGE]

Build the cache:  python -m src.graph_cache
"""
from __future__ import annotations
import os
import time
import numpy as np
import scipy.sparse as sp

from . import models_gnn
from .zones import load_zones

CACHE_DIR = "results/graph_cache"


def cache_path(zone, i):
    """Cache file for slice `i` of `zone`."""
    return f"{CACHE_DIR}/{zone}_{i:04d}.npz"


def have_cache(zone, i):
    """True iff slice `i` of `zone` is already cached."""
    return os.path.exists(cache_path(zone, i))


def _adj_to_edges(adj):
    """Symmetric CSR adjacency -> COO `edge_index` (2, E) int32 (both directions)."""
    a = sp.coo_matrix(adj)
    return np.vstack([a.row, a.col]).astype(np.int32)


def edges_to_adj(edge_index, n_nodes):
    """`edge_index` (2, E) -> symmetric scipy CSR float32 adjacency (0/1 weights).

    Lets any code path that wants the adjacency object `build_graph` returned
    (e.g. `normalized_adjacency`) work straight off a cached edge list.
    """
    r = np.asarray(edge_index[0], dtype=np.int64)
    c = np.asarray(edge_index[1], dtype=np.int64)
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    a = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, cols)),
                      shape=(n_nodes, n_nodes)).tocsr()
    a.data[:] = 1.0             # tocsr() sums duplicate entries; weights are 0/1
    return a


def _save_slice(zone, i, node_feats, adj, labels):
    """Write one slice graph to its cache file (compact COO + int16 label map)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    assert int(labels.max()) < np.iinfo(np.int16).max, \
        f"{zone}[{i}]: {int(labels.max()) + 1} superpixels overflow int16"
    np.savez_compressed(cache_path(zone, i),
                        x=np.asarray(node_feats, np.float32),
                        edge_index=_adj_to_edges(adj),
                        labels=np.asarray(labels, np.int16))


def load_cached(zone, i):
    """-> (node_feats float32 (N,Fn), edge_index int32 (2,E), labels int32 (H,W))."""
    p = cache_path(zone, i)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing — run `python -m src.graph_cache` to build the graph cache")
    d = np.load(p)
    return (d["x"].astype(np.float32),
            d["edge_index"].astype(np.int32),
            d["labels"].astype(np.int32))


def cache_size_mb():
    """Total size of the cache directory in MB."""
    if not os.path.isdir(CACHE_DIR):
        return 0.0
    return sum(os.path.getsize(os.path.join(CACHE_DIR, f))
               for f in os.listdir(CACHE_DIR)) / 1e6


def build_cache(zones=("train", "val", "test"), stride_train=2):
    """Build every MISSING slice graph for `zones`; already-cached slices are skipped.

    Train honors `stride_train` (only every 2nd slice) because that is how
    `GNNRunner` samples its train graphs. One zone's raw stack is loaded at a
    time, as `zones.load_zones` intends, so the three zones never sit in RAM
    together. Returns a summary dict.
    """
    t0 = time.time()
    built = skipped = 0
    for z in zones:
        R = load_zones(zones=(z,))[z][0]
        stride = max(1, stride_train) if z == "train" else 1
        idxs = list(range(0, R.shape[0], stride))
        print(f"[graph_cache] {z}: {len(idxs)} slices (stride {stride})", flush=True)
        for k, i in enumerate(idxs):
            if have_cache(z, i):
                skipped += 1
            else:
                x, adj, labels = models_gnn.build_graph(R[i])
                _save_slice(z, i, x, adj, labels)
                built += 1
            if (k + 1) % 20 == 0:
                print(f"[graph_cache] {z}: {k + 1}/{len(idxs)} "
                      f"({built} built, {skipped} skipped, "
                      f"{time.time() - t0:.0f}s)", flush=True)
        del R
    n_files = len(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else 0
    mb = cache_size_mb()
    print(f"[graph_cache] total {n_files} files, {mb:.1f} MB", flush=True)
    return {"built": built, "skipped": skipped, "files": n_files, "mb": mb,
            "sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    s = build_cache()
    print(f"graph cache -> {CACHE_DIR}: {s['files']} files, {s['mb']:.1f} MB "
          f"({s['built']} built, {s['skipped']} already cached) in {s['sec']}s")
