"""Tier D — alternative message-passing schemes for the superpixel graph.

Six drop-in alternatives to the 3-layer GCN in ``models_gnn``. Every class here
obeys that model's contract, so any of them plugs straight into
``GNNRunner(model_cls=...)``:

  * ``__init__(self, in_dim, hidden=64, dropout=0.2)``
  * ``forward(x, a_norm)`` with ``x`` of shape ``(N, in_dim)`` float and
    ``a_norm`` the SAME coalesced torch sparse COO tensor that
    ``models_gnn.normalized_adjacency`` hands to ``GCN.forward``
    (D^-1/2 (A + I) D^-1/2 — self-loops already included)
  * returns raw LOGITS of shape ``(N, 1)`` — no sigmoid inside forward.

Plain torch sparse ops only (``torch.sparse.mm`` / ``index_add_``): no
torch_geometric, and no dense N x N intermediate is ever materialized (N ~ 6000
nodes per slice, so one dense adjacency would be ~144 MB).

Because the adjacency carries self-loops, every "neighborhood" here includes
the node itself — that is a property of the fixed input contract, not a choice
made per model.

Normalization is GroupNorm (``models_cnn._norm``) exactly as in the GCN
baseline: no running statistics, no coupling between samples. [LEAKAGE]

Self-test:  python -m src.models_gnn_variants     (random graph; shapes only)
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_cnn import _norm
from .models_gnn import GCN, GNNRunner, GraphConv, normalized_adjacency


# ---------------------------------------------------------------------------
# Shared sparse helpers.
# ---------------------------------------------------------------------------
def _edges(a_norm):
    """(2, E) index pairs of the sparse adjacency: row = destination, col = source."""
    a = a_norm if a_norm.is_coalesced() else a_norm.coalesce()
    return a.indices()


def _agg_adj(a_norm, mean):
    """Adjacency with the SAME sparsity but plain aggregation weights.

    `mean=True` -> 1/deg(i) on every incoming edge of node i (mean aggregator);
    `mean=False` -> 1.0 on every edge (sum aggregator).
    """
    idx = _edges(a_norm)
    n = a_norm.shape[0]
    ones = torch.ones(idx.shape[1], device=a_norm.device, dtype=torch.float32)
    if mean:
        deg = torch.zeros(n, device=a_norm.device, dtype=torch.float32)
        deg.index_add_(0, idx[0], ones)
        vals = 1.0 / deg[idx[0]].clamp(min=1.0)
    else:
        vals = ones
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()


# ---------------------------------------------------------------------------
# 1. GAT — attention over existing edges only.
# ---------------------------------------------------------------------------
class GATLayer(nn.Module):
    """Multi-head attention computed over the EXISTING edges of `a_norm` only.

    e_ij = LeakyReLU(0.2) of a^T [W h_i || W h_j] evaluated per edge, softmaxed
    over each destination node's incoming edges; the denominator comes from an
    ``index_add_`` over the edge list, so the N x N matrix is never formed.
    `concat=True` concatenates the heads, `concat=False` averages them.
    """

    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.2,
                 neg_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.neg_slope = neg_slope
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_src)

    def forward(self, x, a_norm):
        n = x.shape[0]
        idx = _edges(a_norm)
        dst, src = idx[0], idx[1]
        h = self.lin(x).view(n, self.heads, self.out_dim)
        # a^T [W h_i || W h_j] splits into a destination half and a source half.
        e = F.leaky_relu((h * self.att_dst).sum(-1)[dst]
                         + (h * self.att_src).sum(-1)[src], self.neg_slope)
        e = torch.exp(e - e.max().detach())          # (E, heads); shift is exact
        denom = torch.zeros(n, self.heads, device=x.device, dtype=e.dtype)
        denom.index_add_(0, dst, e)
        alpha = self.drop(e / denom[dst].clamp(min=1e-16))
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device, dtype=h.dtype)
        out.index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        out = out.reshape(n, -1) if self.concat else out.mean(dim=1)
        return out + self.bias


class GAT(nn.Module):
    """3 graph-attention layers (4 heads) + a linear head -> 1 logit per node.

    Hidden layers CONCAT their heads (4 x hidden/4 = `hidden` channels, so the
    width matches the GCN baseline); the output layer AVERAGES its heads.
    """

    def __init__(self, in_dim, hidden=64, dropout=0.2, heads=4):
        super().__init__()
        per = max(hidden // heads, 1)
        wide = per * heads
        self.g1 = GATLayer(in_dim, per, heads=heads, concat=True, dropout=dropout)
        self.g2 = GATLayer(wide, per, heads=heads, concat=True, dropout=dropout)
        self.g3 = GATLayer(wide, hidden, heads=heads, concat=False, dropout=dropout)
        self.n1 = _norm(wide)
        self.n2 = _norm(wide)
        self.n3 = _norm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        h = self.drop(F.relu(self.n1(self.g1(x, a_norm))))
        h = self.drop(F.relu(self.n2(self.g2(h, a_norm))))
        h = self.drop(F.relu(self.n3(self.g3(h, a_norm))))
        return self.head(h)             # (N, 1) logits


# ---------------------------------------------------------------------------
# 2. GraphSAGE — mean aggregator with a separate self transform.
# ---------------------------------------------------------------------------
class SAGELayer(nn.Module):
    """h_i' = W1 h_i + W2 * mean_{j in N(i)} h_j + b."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, a_mean):
        return self.lin_self(x) + self.lin_neigh(torch.sparse.mm(a_mean, x)) + self.bias


class GraphSAGE(nn.Module):
    """3 mean-aggregator SAGE layers (GroupNorm + ReLU + dropout) + a linear head."""

    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.g1 = SAGELayer(in_dim, hidden)
        self.g2 = SAGELayer(hidden, hidden)
        self.g3 = SAGELayer(hidden, hidden)
        self.n1 = _norm(hidden)
        self.n2 = _norm(hidden)
        self.n3 = _norm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        a_mean = _agg_adj(a_norm, mean=True)
        h = self.drop(F.relu(self.n1(self.g1(x, a_mean))))
        h = self.drop(F.relu(self.n2(self.g2(h, a_mean))))
        h = self.drop(F.relu(self.n3(self.g3(h, a_mean))))
        return self.head(h)             # (N, 1) logits


# ---------------------------------------------------------------------------
# 3. GIN — sum aggregator with a learnable eps and an MLP per layer.
# ---------------------------------------------------------------------------
class GINLayer(nn.Module):
    """h_i' = MLP((1 + eps) h_i + sum_{j in N(i)} h_j), eps LEARNABLE per layer.

    The MLP is 2 linear layers, each followed by GroupNorm + ReLU.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.norm1 = _norm(out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.norm2 = _norm(out_dim)

    def forward(self, x, a_sum):
        h = (1.0 + self.eps) * x + torch.sparse.mm(a_sum, x)
        h = F.relu(self.norm1(self.lin1(h)))
        return F.relu(self.norm2(self.lin2(h)))


class GIN(nn.Module):
    """3 GIN layers + a linear head -> 1 logit per node."""

    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.g1 = GINLayer(in_dim, hidden)
        self.g2 = GINLayer(hidden, hidden)
        self.g3 = GINLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        a_sum = _agg_adj(a_norm, mean=False)
        h = self.drop(self.g1(x, a_sum))
        h = self.drop(self.g2(h, a_sum))
        h = self.drop(self.g3(h, a_sum))
        return self.head(h)             # (N, 1) logits


# ---------------------------------------------------------------------------
# 4. ChebNet — spectral filters of Chebyshev order K.
# ---------------------------------------------------------------------------
class ChebLayer(nn.Module):
    """Chebyshev filter of order K on the normalized adjacency `L_hat`.

    T_0 = x, T_1 = L_hat x, T_k = 2 L_hat T_{k-1} - T_{k-2}; the K terms are
    concatenated and projected, then GroupNorm + ReLU (as in `GraphConv`).
    """

    def __init__(self, in_dim, out_dim, k=3):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(k * in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = _norm(out_dim)

    def forward(self, x, a_norm):
        terms = [x]
        if self.k > 1:
            terms.append(torch.sparse.mm(a_norm, x))
        for _ in range(2, self.k):
            terms.append(2.0 * torch.sparse.mm(a_norm, terms[-1]) - terms[-2])
        h = self.lin(torch.cat(terms, dim=1)) + self.bias
        return F.relu(self.norm(h))


class ChebNet(nn.Module):
    """3 Chebyshev layers (K=3) + a linear head -> 1 logit per node."""

    def __init__(self, in_dim, hidden=64, dropout=0.2, k=3):
        super().__init__()
        self.g1 = ChebLayer(in_dim, hidden, k=k)
        self.g2 = ChebLayer(hidden, hidden, k=k)
        self.g3 = ChebLayer(hidden, hidden, k=k)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        h = self.drop(self.g1(x, a_norm))
        h = self.drop(self.g2(h, a_norm))
        h = self.drop(self.g3(h, a_norm))
        return self.head(h)             # (N, 1) logits


# ---------------------------------------------------------------------------
# 5. ResGCN — 6 layers deep, residual + pre-normalized (does depth help here?).
# ---------------------------------------------------------------------------
class ResGCNLayer(nn.Module):
    """x + A_norm @ (ReLU(norm(x)) W) + b — residual with PRE-normalization."""

    def __init__(self, dim):
        super().__init__()
        self.norm = _norm(dim)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, a_norm):
        h = F.relu(self.norm(x))
        return x + torch.sparse.mm(a_norm, self.lin(h)) + self.bias


class ResGCN(nn.Module):
    """6 residual graph-conv layers (2x the baseline depth) + a linear head."""

    def __init__(self, in_dim, hidden=64, dropout=0.2, depth=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([ResGCNLayer(hidden) for _ in range(depth)])
        self.norm = _norm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        h = self.inp(x)
        for layer in self.layers:
            h = self.drop(layer(h, a_norm))
        return self.head(F.relu(self.norm(h)))      # (N, 1) logits


# ---------------------------------------------------------------------------
# 6. GraphUNetLite — gPool / gUnpool encoder-decoder with skip connections.
# ---------------------------------------------------------------------------
def _restrict(a_norm, idx, n):
    """A[idx][:, idx] on the SPARSE COO tensor — nothing is ever densified.

    Edges survive only when BOTH endpoints are kept; the endpoints are then
    renumbered into the pooled node ordering.
    """
    a = a_norm if a_norm.is_coalesced() else a_norm.coalesce()
    e, v = a.indices(), a.values()
    perm = torch.full((n,), -1, dtype=torch.long, device=a.device)
    perm[idx] = torch.arange(idx.shape[0], dtype=torch.long, device=a.device)
    r, c = perm[e[0]], perm[e[1]]
    keep = (r >= 0) & (c >= 0)
    k = int(idx.shape[0])
    return torch.sparse_coo_tensor(torch.stack([r[keep], c[keep]]), v[keep],
                                   (k, k)).coalesce()


def _unpool(x, idx, n):
    """Scatter (k, F) pooled features back to their original node positions."""
    out = torch.zeros(n, x.shape[1], device=x.device, dtype=x.dtype)
    out[idx] = x
    return out


class GPool(nn.Module):
    """gPool: keep the top `ratio * N` nodes by a learned projection score.

    Kept features are gated by tanh(score) so the projection stays trainable.
    """

    def __init__(self, dim, ratio=0.5):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.ratio = ratio

    def forward(self, x, a_norm):
        n = x.shape[0]
        score = (self.proj(x) / self.proj.weight.norm().clamp(min=1e-12))[:, 0]
        k = max(1, int(n * self.ratio))
        idx = torch.topk(score, k).indices
        xp = x[idx] * torch.tanh(score[idx])[:, None]
        return xp, _restrict(a_norm, idx, n), idx


class GraphUNetLite(nn.Module):
    """Graph U-Net: 2 gPool levels (ratio 0.5) -> bottleneck GCN -> gUnpool.

    Unpooled features are scattered back to their original node positions
    (zeros elsewhere) and ADDED to the encoder features of that level (skip
    connections), as in Gao & Ji (2019).
    """

    def __init__(self, in_dim, hidden=64, dropout=0.2, ratio=0.5):
        super().__init__()
        self.enc0 = GraphConv(in_dim, hidden)
        self.pool1 = GPool(hidden, ratio)
        self.enc1 = GraphConv(hidden, hidden)
        self.pool2 = GPool(hidden, ratio)
        self.bottleneck = GraphConv(hidden, hidden)
        self.dec1 = GraphConv(hidden, hidden)
        self.dec0 = GraphConv(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a_norm):
        n0 = x.shape[0]
        h0 = self.drop(self.enc0(x, a_norm))            # level 0: every node
        x1, a1, i1 = self.pool1(h0, a_norm)             # -> level 1
        h1 = self.drop(self.enc1(x1, a1))
        x2, a2, i2 = self.pool2(h1, a1)                 # -> level 2
        h2 = self.drop(self.bottleneck(x2, a2))
        u1 = _unpool(h2, i2, h1.shape[0]) + h1          # skip at level 1
        u1 = self.drop(self.dec1(u1, a1))
        u0 = _unpool(u1, i1, n0) + h0                   # skip at level 0
        u0 = self.drop(self.dec0(u0, a_norm))
        return self.head(u0)            # (N, 1) logits


# ---------------------------------------------------------------------------
# Configured roster.
# ---------------------------------------------------------------------------
def gnn_models():
    """{name: class} — every entry is a drop-in `GNNRunner(model_cls=...)`."""
    return {
        "gat": GAT,
        "graphsage": GraphSAGE,
        "gin": GIN,
        "chebnet": ChebNet,
        "resgcn": ResGCN,
        "graph_unet": GraphUNetLite,
    }


# ---------------------------------------------------------------------------
# Self-test: random graph only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    N, IN_DIM, E = 200, 26, 800
    r = rng.integers(0, N, E)
    c = rng.integers(0, N, E)
    sel = r != c                                        # no self-loops (A + I later)
    rows = np.concatenate([r[sel], c[sel]])
    cols = np.concatenate([c[sel], r[sel]])             # both directions -> symmetric
    adj = sp.csr_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(N, N))
    adj.data[:] = 1.0
    a_norm = normalized_adjacency(adj)
    x = torch.from_numpy(rng.random((N, IN_DIM)).astype(np.float32))
    print(f"  random graph N={N} in_dim={IN_DIM} edges(nnz)={adj.nnz}")

    for name, cls in gnn_models().items():
        net = cls(IN_DIM)
        net.eval()
        n_par = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            y = net(x, a_norm)
        assert tuple(y.shape) == (N, 1), (name, tuple(y.shape))
        assert torch.isfinite(y).all(), name
        print(f"  OK  {name:12s} {n_par:>8,d} params  -> {tuple(y.shape)}")

    runner = GNNRunner(model_cls=GAT)
    assert runner.model_cls is GAT, runner.model_cls
    assert GNNRunner().model_cls is GCN, "default model_cls must stay GCN"
    print(f"  OK  GNNRunner(model_cls=GAT) name={runner.name} tier={runner.tier}")
    print("gnn variants self-test: OK")
