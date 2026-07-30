"""Tier D: superpixel Graph Neural Network segmenter (spec v2 roster D).

Every 2D CT slice becomes a graph: SLIC superpixels are the nodes, region
adjacency (4-connectivity) supplies the edges, and each node carries the MEAN of
the tier-B texture stack over its pixels plus a few intensity/shape summaries.
A 3-layer GCN labels each node chondrule-vs-matrix and the node probability is
painted back onto its pixels, so the runner exposes the same
``fit`` / ``predict_proba`` API as tiers A/B/C.

Same leakage-free contract as the other tiers:

  * Normalization is ALWAYS ``raw / 255.0``, a *fixed* scale, never a statistic
    estimated from any data split. [LEAKAGE][MUST]
  * ``fit`` trains on TRAIN graphs ONLY; VAL is touched solely to pick the
    operating threshold (and to print progress); no test set is passed or
    referenced anywhere in this file. [LEAKAGE][MUST]
  * The ignore (VALID) mask enters through the node targets and weights: a node
    votes with its VALID pixels only and its loss weight is its valid fraction,
    so a node with zero valid pixels contributes exactly zero. [MUST]

Runtime note: GRAPH BUILDING DOMINATES. SLIC plus the 21-channel texture stack
costs seconds per slice, whereas a forward/backward pass over ~1200 nodes is
negligible. Graphs are therefore cached per slice (keyed by slice content), so
repeated epochs and repeated val passes never recompute SLIC.

Self-test:  python -m src.models_gnn      (synthetic image; no training)
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from skimage.segmentation import slic
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import features, metrics
from .models_cnn import normalize, _norm

# The GNN runs on CPU, NOT on the MPS device the CNNs use. PyTorch has no
# SparseMPS implementation for `aten::_sparse_coo_tensor_with_dims_and_tensors`,
# so every sparse adjacency op raises NotImplementedError there (verified: it
# aborts GNNRunner.fit immediately). The cost is nil: a graph is ~6k nodes, so
# the message passing is microseconds; the expensive part (SLIC + texture
# features) was always CPU-bound and is now disk-cached anyway.
DEVICE = "cpu"

torch.manual_seed(0)

# Superpixel granularity sets a HARD ceiling on this model: painting labels back
# from superpixels can never beat the "every superpixel gets its majority label"
# oracle. Measured on real val slices, 1200 segments give 0.771, 3000 give 0.837
# and 6000 give 0.878, at essentially identical cost (~9.5 s/slice, dominated by
# feature extraction, not SLIC). So use 6000: no reason to cap the GNN at 0.77.
N_SEGMENTS = 6000
COMPACTNESS = 0.1


# ---------------------------------------------------------------------------
# Graph construction (the expensive part; cached by GNNRunner).
# ---------------------------------------------------------------------------
def build_graph(raw_slice, n_segments=N_SEGMENTS, compactness=COMPACTNESS):
    """(H,W) raw slice -> (node features (N,Fn), adjacency (N,N), labels (H,W)).

    SLIC runs on the fixed-scale ``raw/255`` image (no data-derived statistic).
    Node feature layout, per superpixel:

        [0 .. F-1]  mean of ``features.extract_features`` over its pixels
        [F .. F+3]  mean / std / min / max of the normalized intensity
        [F+4, F+5]  centroid y/H, x/W
        [F+6]       area fraction (area / (H*W))

    `adjacency` is a symmetric scipy CSR matrix with a 1.0 wherever two
    superpixels touch under 4-connectivity (no self-loops; those are added by
    `normalized_adjacency`). `labels` is the (H,W) int32 superpixel id map,
    contiguous over 0..N-1, so it covers every pixel.
    """
    img = normalize(raw_slice)                      # fixed raw/255 [LEAKAGE][MUST]
    H, W = img.shape
    labels = slic(img, n_segments=n_segments, compactness=compactness,
                  channel_axis=None, start_label=0).astype(np.int32)
    n = int(labels.max()) + 1
    lab_flat = labels.ravel()
    area = np.bincount(lab_flat, minlength=n).astype(np.float64)
    denom = np.maximum(area, 1.0)

    # --- mean of each texture channel over each superpixel ---
    feats = features.extract_features(img)          # (H,W,F)
    flat = feats.reshape(-1, feats.shape[-1])
    fmean = np.empty((n, flat.shape[1]), np.float64)
    for j in range(flat.shape[1]):
        fmean[:, j] = np.bincount(lab_flat, weights=flat[:, j].astype(np.float64),
                                  minlength=n) / denom

    # --- intensity mean / std / min / max ---
    ints = img.ravel().astype(np.float64)
    i_mean = np.bincount(lab_flat, weights=ints, minlength=n) / denom
    i_sq = np.bincount(lab_flat, weights=ints * ints, minlength=n) / denom
    i_std = np.sqrt(np.maximum(i_sq - i_mean * i_mean, 0.0))
    idx = np.arange(n)
    i_min = np.asarray(ndi.minimum(img, labels, idx), dtype=np.float64)
    i_max = np.asarray(ndi.maximum(img, labels, idx), dtype=np.float64)

    # --- normalized centroid + area fraction ---
    ys = np.repeat(np.arange(H, dtype=np.float64), W)
    xs = np.tile(np.arange(W, dtype=np.float64), H)
    cy = (np.bincount(lab_flat, weights=ys, minlength=n) / denom) / float(H)
    cx = (np.bincount(lab_flat, weights=xs, minlength=n) / denom) / float(W)

    node_feats = np.concatenate([
        fmean,
        np.stack([i_mean, i_std, i_min, i_max], axis=1),
        np.stack([cy, cx, area / float(H * W)], axis=1),
    ], axis=1).astype(np.float32)

    # --- region adjacency: superpixels touching under 4-connectivity ---
    pairs = []
    a, b = labels[:, :-1].ravel(), labels[:, 1:].ravel()
    sel = a != b
    pairs.append(np.stack([a[sel], b[sel]], axis=1))
    a, b = labels[:-1, :].ravel(), labels[1:, :].ravel()
    sel = a != b
    pairs.append(np.stack([a[sel], b[sel]], axis=1))
    und = np.unique(np.sort(np.concatenate(pairs, axis=0), axis=1), axis=0)
    rows = np.concatenate([und[:, 0], und[:, 1]])   # both directions -> symmetric
    cols = np.concatenate([und[:, 1], und[:, 0]])
    adj = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(n, n))
    return node_feats, adj, labels


def normalized_adjacency(adj):
    """A -> D^-1/2 (A + I) D^-1/2 as a coalesced torch sparse COO tensor (CPU)."""
    n = adj.shape[0]
    a = sp.coo_matrix(adj + sp.eye(n, dtype=np.float32, format="csr"))
    deg = np.asarray(a.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    vals = (a.data * dinv[a.row] * dinv[a.col]).astype(np.float32)
    idx = torch.from_numpy(np.vstack([a.row, a.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n, n)).coalesce()


def node_targets(labels, mask, valid, n_nodes):
    """Per-node (target, weight) from a pixel mask + VALID mask. [LEAKAGE][MUST]

    target = 1 iff the MAJORITY of a node's VALID pixels are chondrule (strictly
    more than half); weight = fraction of the node's pixels that are valid, so a
    node with zero valid pixels gets weight 0 and cannot contribute to the loss.
    """
    lab_flat = labels.ravel()
    area = np.bincount(lab_flat, minlength=n_nodes).astype(np.float64)
    n_valid = np.bincount(lab_flat, weights=valid.ravel().astype(np.float64),
                          minlength=n_nodes)
    n_pos = np.bincount(lab_flat, weights=(mask & valid).ravel().astype(np.float64),
                        minlength=n_nodes)
    target = (n_pos > 0.5 * n_valid).astype(np.float32)
    weight = (n_valid / np.maximum(area, 1.0)).astype(np.float32)
    return target, weight


def paint(labels, node_values):
    """(H,W) label map + (N,) per-node values -> (H,W) values painted per pixel."""
    return node_values[labels]


# ---------------------------------------------------------------------------
# Model.  H' = act(GroupNorm(A_norm @ (H @ W) + b)); GroupNorm carries no batch
# statistics, so it introduces no batch-statistic leakage (as in tier C).
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """One graph-convolution layer: aggregate over A_norm, norm, ReLU."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = _norm(out_dim)

    def forward(self, x, a_norm):
        h = torch.sparse.mm(a_norm, self.lin(x)) + self.bias
        return F.relu(self.norm(h))


class GCN(nn.Module):
    """3 graph-conv layers (width 64) + a linear head -> 1 logit per node."""

    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden)
        self.g2 = GraphConv(hidden, hidden)
        self.g3 = GraphConv(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        h = self.drop(self.g1(x, a_norm))
        h = self.drop(self.g2(h, a_norm))
        h = self.drop(self.g3(h, a_norm))
        return self.head(h)             # (N, 1) logits


def _best_thr_iou(prob, mask, valid, grid=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Threshold maximizing pixel-level foreground IoU (metrics.iou honors valid)."""
    best_t, best_v = 0.5, -1.0
    for t in sorted(set((0.5,) + tuple(grid))):
        v = metrics.iou(prob >= t, mask, valid)
        if v == v and v > best_v:       # skip NaN
            best_v, best_t = v, float(t)
    return best_t, best_v


# ---------------------------------------------------------------------------
# Runner (matches the tier-A/B/C fit / predict_proba API).
# ---------------------------------------------------------------------------
class GNNRunner:
    """Trains the superpixel GCN and exposes predict_proba -> (n,H,W) in [0,1].

    `.tier` = "D_gnn"; `.name` identifies the run. `train_stride` subsamples the
    TRAIN slices (graph building is the cost); `val_every` sets how often the
    val IoU is printed during training. `model_cls` swaps the message-passing
    architecture: any module with the `GCN(in_dim, hidden, dropout)` signature
    (see `models_gnn_variants`); the default keeps the 3-layer GCN.
    """

    def __init__(self, name="superpixel_gcn", n_segments=N_SEGMENTS,
                 compactness=COMPACTNESS, hidden=64, dropout=0.2, epochs=40,
                 lr=1e-2, train_stride=2, val_every=10, seed=0, *,
                 model_cls=GCN):
        self.name = name
        self.tier = "D_gnn"
        self.n_segments = n_segments
        self.compactness = compactness
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.train_stride = train_stride
        self.val_every = val_every
        self.seed = seed
        self.model_cls = model_cls
        self.model = None
        self.best_thr = 0.5
        self._cache = {}                # slice content -> built graph

    def _graph(self, sl):
        """Cached build_graph for one slice: SLIC is never recomputed.

        Two cache tiers, both keyed by SLICE CONTENT (not by zone/index): an
        in-process dict, then a DISK cache shared across runs and across GNN
        variants. Content-keying is what makes the disk tier work at all:
        `predict_proba(raw)` receives a bare array and never learns which zone
        it belongs to, so a (zone, index) key could not be formed here.
        Building a graph costs ~9.5 s/slice, so without the shared disk tier
        each of the 7 GNN variants would repeat ~40 min of identical SLIC +
        texture-feature work.
        """
        key = (sl.shape, hash(sl.tobytes()))
        g = self._cache.get(key)
        if g is None:
            g = self._disk_load(sl)
            if g is None:
                g = build_graph(sl, self.n_segments, self.compactness)
                self._disk_save(sl, g)
            self._cache[key] = g
        return g

    def _disk_key(self, sl):
        import hashlib
        h = hashlib.sha1(np.ascontiguousarray(sl).tobytes()).hexdigest()[:16]
        return f"results/graph_cache/n{self.n_segments}_{h}.npz"

    def _disk_load(self, sl):
        import os
        p = self._disk_key(sl)
        if not os.path.exists(p):
            return None
        try:
            d = np.load(p)
            from . import graph_cache
            adj = graph_cache.edges_to_adj(d["edge_index"], int(d["n_nodes"]))
            return d["x"], adj, d["labels"].astype(np.int32)
        except Exception:
            return None                      # corrupt/partial file: just rebuild

    def _disk_save(self, sl, g):
        import os
        x, adj, labels = g
        p = self._disk_key(sl)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            from . import graph_cache
            ei = graph_cache._adj_to_edges(adj)
            np.savez_compressed(p, x=x.astype(np.float32), edge_index=ei,
                                labels=labels.astype(np.int16),
                                n_nodes=np.int32(x.shape[0]))
        except Exception:
            pass                             # cache is an optimization, never a hard dep

    def _standardize(self, x):
        """Apply the TRAIN-fitted node-feature scaling (identity before fit)."""
        mu = getattr(self, "_feat_mu", None)
        if mu is None:
            return np.asarray(x, np.float32)
        return ((np.asarray(x, np.float32) - mu) / self._feat_sd).astype(np.float32)

    def fit(self, train, val):
        """Train on `train` ONLY; use `val` ONLY for the threshold. [LEAKAGE][MUST]

        train/val are dicts {"raw":(n,H,W) float32, "mask":bool, "valid":bool}.
        No test set is passed or referenced anywhere in this method.
        """
        torch.manual_seed(self.seed)
        R, M, V = train["raw"], train["mask"], train["valid"]
        stride = max(1, self.train_stride)

        # --- TRAIN-ONLY graphs (+ node targets/weights from the VALID mask). ---
        t0 = time.time()
        raw_graphs = []
        for i in range(0, R.shape[0], stride):
            x, adj, labels = self._graph(R[i])
            tgt, w = node_targets(labels, M[i], V[i], x.shape[0])
            raw_graphs.append((x, adj, tgt, w))

        # --- Node-feature standardization, fit on TRAIN nodes ONLY. [LEAKAGE][MUST]
        # Without this the GNN collapses to a constant output. The 26 node
        # features span a ~20,000x scale ratio (per-column std 1.5e-05 .. 0.29:
        # intensity/centroid terms vs Hessian-eigenvalue and Gabor channels), and
        # graph convolutions on inputs that badly conditioned cannot learn. In
        # measured degenerate runs, GAT emitted a constant 0.446 everywhere
        # (recall 1.000, IoU 0.053) and the plain GCN emitted just two distinct
        # values. Tier B already standardizes (StandardScaler) and the CNNs get a
        # fixed raw/255; the GNN was the one path with no scaling at all.
        allx = np.concatenate([g[0] for g in raw_graphs], axis=0)
        self._feat_mu = allx.mean(axis=0, keepdims=True).astype(np.float32)
        self._feat_sd = np.maximum(allx.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
        del allx

        graphs = []
        for x, adj, tgt, w in raw_graphs:
            graphs.append((torch.from_numpy(self._standardize(x)).to(DEVICE),
                           normalized_adjacency(adj).to(DEVICE),
                           torch.from_numpy(tgt[:, None]).to(DEVICE),
                           torch.from_numpy(w[:, None]).to(DEVICE)))
        del raw_graphs
        n_nodes = sum(int(g[0].shape[0]) for g in graphs)
        print(f"    [{self.name}] {len(graphs)} train graphs / {n_nodes} nodes "
              f"built in {time.time() - t0:.0f}s", flush=True)

        # Class balance from TRAIN nodes only; nodes with weight 0 are excluded
        # because they cannot contribute to the loss.
        pos = sum(int(((t > 0.5) & (w > 0)).sum()) for _, _, t, w in graphs)
        neg = sum(int(((t <= 0.5) & (w > 0)).sum()) for _, _, t, w in graphs)
        pw = float(np.clip(neg / max(pos, 1), 1.0, 20.0))
        pos_weight = torch.tensor([pw], device=DEVICE)

        self.model = self.model_cls(int(graphs[0][0].shape[1]), hidden=self.hidden,
                                    dropout=self.dropout).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"    [{self.name}] nodes pos={pos} neg={neg} pos_weight={pw:.2f} "
              f"in_dim={int(graphs[0][0].shape[1])} device={DEVICE}", flush=True)

        for ep in range(self.epochs):
            te = time.time()
            self.model.train()
            tot = 0.0
            for x, a, tgt, w in graphs:     # one pass, full-batch per graph
                opt.zero_grad()
                per_node = F.binary_cross_entropy_with_logits(
                    self.model(x, a), tgt, pos_weight=pos_weight, reduction="none")
                loss = (per_node * w).sum() / w.sum().clamp(min=1.0)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
            msg = (f"    [{self.name}] epoch {ep + 1}/{self.epochs} "
                   f"loss={tot / max(len(graphs), 1):.4f} {time.time() - te:.1f}s")
            if (ep + 1) % max(1, self.val_every) == 0 or ep == self.epochs - 1:
                prob = self.predict_proba(val["raw"])
                thr, viou = _best_thr_iou(prob, val["mask"], val["valid"])
                msg += f" valIoU={viou:.4f} (thr={thr:.2f})"
            print(msg, flush=True)

        # --- VAL-ONLY: freeze the operating threshold on the full val zone. ---
        prob = self.predict_proba(val["raw"])
        self.best_thr, viou = _best_thr_iou(
            prob, val["mask"], val["valid"],
            grid=tuple(float(round(x, 2)) for x in np.arange(0.05, 0.96, 0.05)))
        self.best_full_val_iou = float(viou)
        print(f"    [{self.name}] FULL-val IoU={viou:.4f} "
              f"frozen thr={self.best_thr:.2f}", flush=True)
        return self

    def predict_proba(self, raw):
        """(n,H,W) raw -> (n,H,W) float32 probabilities in [0,1].

        Per slice: build (or reuse) its graph, run the GCN once, then paint each
        superpixel's probability back onto its pixels via the label map.
        """
        assert self.model is not None, "call fit() before predict_proba"
        self.model.eval()
        n, H, W = raw.shape
        out = np.zeros((n, H, W), np.float32)
        with torch.no_grad():
            for i in range(n):
                x, adj, labels = self._graph(raw[i])
                # Same TRAIN-fitted scaling as fit(): applying the model to
                # unstandardized features would be a train/apply mismatch.
                logits = self.model(torch.from_numpy(self._standardize(x)).to(DEVICE),
                                    normalized_adjacency(adj).to(DEVICE))
                p = torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float32)
                out[i] = paint(labels, p)
        return out


# ---------------------------------------------------------------------------
# Self-test: synthetic image only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    H, W = 128, 160
    img = rng.random((H, W)) * 60.0 + 40.0
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - 60) ** 2 + (xx - 80) ** 2 <= 30 ** 2
    img[disk] += 120.0
    img = np.clip(img, 0.0, 255.0).astype(np.float32)

    x, adj, labels = build_graph(img, n_segments=60, compactness=COMPACTNESS)
    n = x.shape[0]
    assert x.ndim == 2 and n > 0, x.shape
    assert adj.shape == (n, n), adj.shape
    dense = adj.toarray()
    assert np.array_equal(dense, dense.T), "adjacency not symmetric"
    assert labels.shape == (H, W), labels.shape
    assert labels.min() == 0 and labels.max() == n - 1, (labels.min(), labels.max())
    assert np.bincount(labels.ravel(), minlength=n).min() > 0, "empty superpixel"
    assert np.unique(labels).size == n, "label map does not cover 0..N-1"

    tgt, w = node_targets(labels, disk, np.ones((H, W), bool), n)
    assert tgt.shape == (n,) and w.shape == (n,), (tgt.shape, w.shape)
    assert float(w.min()) == 1.0, w.min()        # all pixels valid -> weight 1

    net = GCN(x.shape[1])
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(x), normalized_adjacency(adj))
    assert tuple(out.shape) == (n, 1), tuple(out.shape)

    print(f"  image {img.shape}  nodes={n}  node features {x.shape}")
    print(f"  adjacency {adj.shape} nnz={adj.nnz}  labels {labels.shape} "
          f"{labels.dtype}")
    print(f"  positive nodes={int(tgt.sum())}  logits {tuple(out.shape)}")
    print("gnn self-test: OK")
