"""Hierarchical region cascade: split coarse, DROP the empty regions, subdivide.

A slice is cut into a few coarse regions; every region that contains no
chondrule is DROPPED outright; the survivors are subdivided and the test is
repeated, down to fine regions. Only the pixels still alive at the finest level
ever need to reach a pixel-level model.

Why the hierarchy is built BOTTOM-UP by merging: SLIC at different
granularities does NOT nest (a fine superpixel is not contained in any single
coarse superpixel), so a top-down re-segmentation would not give a tree at all.
The fine SLIC segmentation is therefore built once and agglomerated upward on
the region-adjacency graph, which makes every level an exact refinement of the
level above it.

Measured facts that motivate the design (established elsewhere, not re-derived
here): at 20 regions SLIC regions are 69.8% droppable (zero chondrule pixels)
versus 53.0% for a regular grid; a trivial region feature (99th-percentile
intensity) drops 39.7%-68.4% of regions at PERFECT recall; ideal dropping
leaves ~8.4% of the image at the finest level (~12x compute saving).

Leakage contract:

  * Intensities are ALWAYS ``raw / 255.0`` (``models_cnn.normalize``): a fixed
    scale, never a statistic estimated from any split. [LEAKAGE][MUST]
  * Drop thresholds are FIT ON VAL and FROZEN FOR TEST: call `run_cascade` with
    `thresholds=None` on the VAL zone, take `[lv["threshold"] for lv in
    res["levels"]]`, and pass that list back as `thresholds=` when running the
    test zone. Fitting a threshold on the data it is scored on would read the
    test labels through the drop rule. [LEAKAGE][MUST]
  * The LEARNED drop rule obeys the same contract: call `run_cascade_learned`
    with `classifiers=None` on the VAL zone, take `res["classifiers"]`, and pass
    that list back as `classifiers=` when running the test zone. Both the model
    and its probability cut come from val. [LEAKAGE][MUST]
  * Chondrule pixels count only where the VALID mask is True, matching the rest
    of the harness. [MUST]

The fine segmentation reuses the tier-D graph disk cache (same content-keyed
path as ``GNNRunner._disk_key``), so a slice whose graph a GNN run already
built is not re-SLIC'd here.

Self-test:  python -m src.hierarchy      (4 real VAL slices; then 8 val + 8 test
slices for the learned drop rule, fit on val and frozen for test)
"""
from __future__ import annotations
import hashlib
import heapq
import os
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier

from . import features, graph_cache, models_gnn, zones
from .models_cnn import normalize
from .models_gnn import N_SEGMENTS

# Region counts per level, COARSEST FIRST. The finest level is the SLIC
# granularity itself (models_gnn.N_SEGMENTS = 6000), which is also what makes
# the shared graph cache keys (`n6000_*`) hit.
LEVELS = (20, 80, 320, 1280, N_SEGMENTS)


# ---------------------------------------------------------------------------
# Fine segmentation (the expensive step): content-keyed disk cache, exactly the
# scheme in GNNRunner._graph / _disk_key.
# ---------------------------------------------------------------------------
def _disk_key(sl, n_segments):
    """Cache path for one slice's graph, keyed by SLICE CONTENT."""
    h = hashlib.sha1(np.ascontiguousarray(sl).tobytes()).hexdigest()[:16]
    return f"results/graph_cache/n{n_segments}_{h}.npz"


def _cached_graph(sl, n_segments):
    """`models_gnn.build_graph` for one slice, through the shared disk cache."""
    p = _disk_key(sl, n_segments)
    if os.path.exists(p):
        try:
            d = np.load(p)
            adj = graph_cache.edges_to_adj(d["edge_index"], int(d["n_nodes"]))
            return d["x"], adj, d["labels"].astype(np.int32)
        except Exception:
            pass                             # corrupt/partial file: just rebuild
    g = models_gnn.build_graph(sl, n_segments)
    x, adj, labels = g
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        ei = graph_cache._adj_to_edges(adj)
        np.savez_compressed(p, x=x.astype(np.float32), edge_index=ei,
                            labels=labels.astype(np.int16),
                            n_nodes=np.int32(x.shape[0]))
    except Exception:
        pass                                 # cache is an optimization, never a hard dep
    return g


# ---------------------------------------------------------------------------
# 1. Bottom-up hierarchy.
# ---------------------------------------------------------------------------
class Hierarchy:
    """Nested region partitions of one slice, COARSEST FIRST.

    `.labels_fine` is the (H,W) int32 SLIC label map, contiguous over 0..N-1.
    `.levels` is a list of `{"n_regions": int, "map": (N,) int32}`, where `map`
    sends a fine label to its region id at that level (contiguous over
    0..n_regions-1). Level `k` is an exact refinement of level `k-1`.
    """

    def __init__(self, labels_fine, levels):
        self.labels_fine = labels_fine
        self.levels = levels

    def region_labels(self, level_idx):
        """(H,W) int32 region map at `level_idx` (0 = coarsest)."""
        return self.levels[level_idx]["map"][self.labels_fine]


def _find(parent, i):
    """Union-find root of `i` with path compression."""
    r = i
    while parent[r] != r:
        r = parent[r]
    while parent[i] != r:
        parent[i], i = r, parent[i]
    return r


def build_hierarchy(raw_slice, n_fine=N_SEGMENTS, levels=LEVELS):
    """(H,W) raw slice -> `Hierarchy` over the region counts in `levels`.

    The fine SLIC segmentation is built (or loaded) ONCE, then agglomerated on
    its region-adjacency graph: the adjacent pair with the most similar mean
    intensity is merged repeatedly, and the partition is snapshotted every time
    the live region count first drops to a target in `levels`.
    """
    _, adj, labels_fine = _cached_graph(raw_slice, n_fine)
    n = int(labels_fine.max()) + 1

    img = normalize(raw_slice)
    lab_flat = labels_fine.ravel()
    size = np.bincount(lab_flat, minlength=n).astype(np.float64)
    mean = (np.bincount(lab_flat, weights=img.ravel().astype(np.float64),
                        minlength=n) / np.maximum(size, 1.0))

    tri = sp.triu(adj, k=1).tocoo()
    eu, ev = tri.row.astype(np.int64), tri.col.astype(np.int64)
    nb = [set() for _ in range(n)]
    for u, v in zip(eu, ev):
        nb[u].add(int(v))
        nb[v].add(int(u))

    parent = list(range(n))
    ver = [0] * n

    # Merge cost. A PURE intensity difference |mean_u - mean_v| is degenerate:
    # a region that has already grown pays no more to absorb the next neighbour
    # than a tiny one does, so one region runs away and swallows the slice.
    # Measured with the pure-intensity cost, the largest level-0 region covered
    # 98.4-99.4% of the slice and the oracle droppable area collapsed to 0.5%
    # (direct SLIC at the same region count leaves 69% droppable), which made
    # the whole coarse-to-fine idea untestable.
    #
    # Ward-style cost scales the intensity difference by the harmonic size term
    # su*sv/(su+sv), so merging two already-large regions is expensive and the
    # partition stays roughly equal-area, like SLIC.
    def _cost(u, v):
        su, sv = float(size[u]), float(size[v])
        return float((mean[u] - mean[v]) ** 2 * (su * sv) / max(su + sv, 1.0))

    heap = [(_cost(int(u), int(v)), int(u), int(v), 0, 0)
            for u, v in zip(eu, ev)]
    heapq.heapify(heap)

    targets = sorted({int(t) for t in levels}, reverse=True)
    snaps = []                                # finest first (targets descend)

    def _record():
        roots = np.array([_find(parent, i) for i in range(n)], dtype=np.int64)
        _, inv = np.unique(roots, return_inverse=True)
        inv = np.asarray(inv).reshape(-1).astype(np.int32)
        snaps.append({"n_regions": int(inv.max()) + 1, "map": inv})

    n_active = n
    ti = 0
    while ti < len(targets) and n_active <= targets[ti]:
        _record()
        ti += 1
    while ti < len(targets) and heap:
        _, u, v, vu, vv = heapq.heappop(heap)
        if parent[u] != u or parent[v] != v:
            continue                          # one side was absorbed
        if ver[u] != vu or ver[v] != vv:
            continue                          # a mean changed: entry is stale
        if size[v] > size[u]:
            u, v = v, u                       # keep the larger region as root
        parent[v] = u
        tot = size[u] + size[v]
        mean[u] = (mean[u] * size[u] + mean[v] * size[v]) / tot
        size[u] = tot
        ver[u] += 1
        merged = (nb[u] | nb[v]) - {u, v}
        nb[u] = merged
        nb[v] = set()
        for w in merged:
            nb[w].discard(v)
            nb[w].add(u)
            heapq.heappush(heap, (_cost(u, w), u, w, ver[u], ver[w]))
        n_active -= 1
        while ti < len(targets) and n_active <= targets[ti]:
            _record()
            ti += 1
    while ti < len(targets):                  # disconnected RAG: no merges left
        _record()
        ti += 1

    return Hierarchy(labels_fine, snaps[::-1])


# ---------------------------------------------------------------------------
# 2. Per-level drop rule.
# ---------------------------------------------------------------------------
def region_feature(raw_slice, region_map, n_regions):
    """(n_regions,) float32: the 99th-percentile normalized intensity per region."""
    img = normalize(raw_slice).ravel()
    lab = np.asarray(region_map).ravel()
    order = np.argsort(lab, kind="stable")
    lab_s, val_s = lab[order], img[order]
    counts = np.bincount(lab_s, minlength=n_regions)[:n_regions]
    ends = np.cumsum(counts)
    starts = ends - counts
    out = np.zeros(n_regions, np.float32)
    for r in range(n_regions):
        s, e = int(starts[r]), int(ends[r])
        if e > s:
            out[r] = np.percentile(val_s[s:e], 99.0)
    return out


def region_features_rich(raw_slice, region_map, n_regions):
    """(n_regions, F+11) float32: the RICH per-region descriptor.

    One intensity statistic cannot separate a chondrule from bright matrix
    (matrix p90 = 55 CT overlaps chondrule mean ~47), so the learned drop rule
    gets the whole texture stack pooled per region instead of one number.

    Column layout, with F = len(`features.feature_names()`) (currently 19, so
    the descriptor is 30 wide):

        [0 .. F-1]   mean of each ``features.extract_features`` channel over the
                     region, in ``features.feature_names()`` order
        [F .. F+7]   normalized intensity mean / std / min / max / p25 / p75 /
                     p90 / p99 over the region
        [F+8]        area fraction (region area / (H*W))
        [F+9]        normalized centroid y (mean row / H)
        [F+10]       normalized centroid x (mean col / W)

    Column F repeats column 0 (both are the mean normalized intensity); the
    intensity block is kept as a self-contained 8-statistic summary.

    Intensity is ALWAYS ``raw/255`` (`models_cnn.normalize`): a fixed scale,
    never a statistic estimated from a split. [LEAKAGE][MUST]
    """
    img = normalize(raw_slice)
    H, W = img.shape
    lab_flat = np.asarray(region_map).ravel()
    area = np.bincount(lab_flat, minlength=n_regions)[:n_regions].astype(np.float64)
    denom = np.maximum(area, 1.0)

    # --- mean of every texture channel: ONE bincount per channel, vectorized
    # over regions (never a per-region loop) ---
    feats = features.extract_features(img)                 # (H,W,F)
    flat = feats.reshape(-1, feats.shape[-1])
    fmean = np.empty((n_regions, flat.shape[1]), np.float64)
    for j in range(flat.shape[1]):
        fmean[:, j] = np.bincount(lab_flat,
                                  weights=flat[:, j].astype(np.float64),
                                  minlength=n_regions)[:n_regions] / denom

    # --- intensity mean / std, also by bincount ---
    ints = img.ravel().astype(np.float64)
    i_mean = np.bincount(lab_flat, weights=ints,
                         minlength=n_regions)[:n_regions] / denom
    i_sq = np.bincount(lab_flat, weights=ints * ints,
                       minlength=n_regions)[:n_regions] / denom
    i_std = np.sqrt(np.maximum(i_sq - i_mean * i_mean, 0.0))

    # --- intensity min / max / percentiles: group the pixels by region id with
    # one stable argsort, then walk the regions (percentiles have no bincount
    # form). Within a group the values keep raster order, not sorted order, so
    # min/max are taken explicitly.
    order = np.argsort(lab_flat, kind="stable")
    lab_s, val_s = lab_flat[order], ints[order]
    counts = np.bincount(lab_s, minlength=n_regions)[:n_regions]
    ends = np.cumsum(counts)
    starts = ends - counts
    i_min = np.zeros(n_regions, np.float64)
    i_max = np.zeros(n_regions, np.float64)
    i_pct = np.zeros((n_regions, 4), np.float64)
    for r in range(n_regions):
        s, e = int(starts[r]), int(ends[r])
        if e > s:
            v = val_s[s:e]
            i_min[r], i_max[r] = v.min(), v.max()
            i_pct[r] = np.percentile(v, (25.0, 75.0, 90.0, 99.0))

    # --- geometry: area fraction + normalized centroid ---
    ys = np.repeat(np.arange(H, dtype=np.float64), W)
    xs = np.tile(np.arange(W, dtype=np.float64), H)
    cy = (np.bincount(lab_flat, weights=ys,
                      minlength=n_regions)[:n_regions] / denom) / float(H)
    cx = (np.bincount(lab_flat, weights=xs,
                      minlength=n_regions)[:n_regions] / denom) / float(W)

    return np.concatenate([
        fmean,
        np.stack([i_mean, i_std, i_min, i_max], axis=1),
        i_pct,
        np.stack([area / float(H * W), cy, cx], axis=1),
    ], axis=1).astype(np.float32)


def region_has_chondrule(region_map, mask, valid, n_regions):
    """(n_regions,) bool: True iff the region holds ANY valid chondrule pixel."""
    lab = np.asarray(region_map).ravel()
    pos = (np.asarray(mask, bool) & np.asarray(valid, bool)).ravel()
    return np.bincount(lab[pos], minlength=n_regions)[:n_regions] > 0


def fit_drop_threshold(feats, has_chondrule, target_recall=1.0):
    """Largest t such that keeping `feats >= t` retains >= `target_recall` of the
    regions that truly contain a chondrule. Regions below t are dropped.

    FIT ON VAL, FROZEN FOR TEST. [LEAKAGE][MUST]

    With `target_recall=1.0` this is the minimum feature value over the positive
    regions, so no region holding a chondrule is ever dropped. With no positive
    region to constrain it, the rule is vacuous and -inf (drop nothing) is
    returned.
    """
    f = np.asarray(feats, np.float64)
    pos = np.asarray(has_chondrule, bool)
    s = np.sort(f[pos])
    k = int(s.size)
    if k == 0:
        return float("-inf")
    m = int(np.ceil(float(target_recall) * k))      # positives that must survive
    m = max(1, min(m, k))
    return float(s[k - m])                          # the m-th largest positive


class LevelDropClassifier:
    """A LEARNED drop rule for ONE level of the hierarchy.

    The intensity rule above is one number per region (the 99th-percentile
    intensity) against one threshold, the weakest possible test. As measured, it
    drops only ~70% of the area even though ~92.5% of regions are pure matrix,
    because bright matrix overlaps chondrule intensity. This replaces it with a
    gradient-boosted classifier over the full `region_features_rich` descriptor,
    cut at a recall-constrained probability.

    FIT ON VAL, FREEZE, APPLY TO TEST. [LEAKAGE][MUST]

    `.model` is the fitted `HistGradientBoostingClassifier`; `.threshold` is the
    probability cut (keep a region iff p >= threshold).
    """

    def __init__(self):
        self.model = None
        self.threshold = float("-inf")

    def fit(self, raw_stack, mask_stack, valid_stack, hierarchies, level_idx,
            target_recall=0.999, seed=0):
        """Train "this region contains a chondrule" at `level_idx`, then cut.

        THE CALLER MUST PASS VALIDATION DATA. The operating threshold is chosen
        on the SAME data the model was fit on, so running this on test would
        read the test labels through the drop rule and self-grade the cascade.
        Fit on VAL, freeze, apply to TEST. [LEAKAGE][MUST]

        Regions come from `hierarchies[i].region_labels(level_idx)`, features
        from `region_features_rich`, labels from `region_has_chondrule`. The
        threshold is the largest probability cut still retaining at least
        `target_recall` of the positive regions, the same rule as
        `fit_drop_threshold`, applied to the model probability instead of the
        intensity percentile. With no positive region the rule is vacuous and
        -inf (drop nothing) is used.
        """
        X, y = [], []
        for i in range(len(hierarchies)):
            h = hierarchies[i]
            nr = int(h.levels[level_idx]["n_regions"])
            rm = h.region_labels(level_idx)
            X.append(region_features_rich(raw_stack[i], rm, nr))
            y.append(region_has_chondrule(rm, mask_stack[i], valid_stack[i], nr))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

        self.model = HistGradientBoostingClassifier(max_iter=200,
                                                    random_state=seed)
        self.model.fit(X, y.astype(np.int32))

        p = self.model.predict_proba(X)[:, 1]
        s = np.sort(p[y])
        k = int(s.size)
        if k == 0:
            self.threshold = float("-inf")
        else:
            m = int(np.ceil(float(target_recall) * k))  # positives that survive
            m = max(1, min(m, k))
            self.threshold = float(s[k - m])            # m-th largest positive
        return self

    def predict_keep(self, raw_slice, region_map, n_regions):
        """(n_regions,) bool: True = KEEP the region, False = drop it.

        Applies the FROZEN model and threshold unchanged, so this is the path a
        test slice takes; nothing here is fit. [LEAKAGE][MUST]
        """
        assert self.model is not None, "call fit() before predict_keep"
        X = region_features_rich(raw_slice, region_map, n_regions)
        return self.model.predict_proba(X)[:, 1] >= self.threshold


# ---------------------------------------------------------------------------
# 3. The cascade.
# ---------------------------------------------------------------------------
def run_cascade(raw, mask, valid, thresholds=None, levels=LEVELS,
                target_recall=1.0):
    """Descend the hierarchy over a stack, dropping empty regions level by level.

    Every region is alive at the coarsest level. At each level the region
    feature is computed and the ALIVE regions whose feature is below that
    level's threshold are dropped; a region at the next (finer) level is alive
    only if its parent survived.

    `thresholds=None` FITS one threshold per level on the data passed in, so the
    caller passes VAL. A list of thresholds APPLIES them unchanged: that is the
    test path, and the list must be the one fit on val. [LEAKAGE][MUST]

    Returns `{"levels": [...], "final_alive_mask": (n,H,W) bool}`, one dict per
    level (coarsest first) with `level`, `n_regions` and `n_alive` (totalled
    over the stack), `frac_alive` (fraction of image area still live),
    `chondrule_retention` (fraction of all TRUE chondrule pixels still inside a
    live region, the number that decides whether the cascade is usable), and
    `threshold`. `final_alive_mask` marks the pixels inside regions alive at the
    FINEST level: exactly what a downstream pixel model would have to process.
    """
    raw = np.asarray(raw)
    mask = np.asarray(mask, bool)
    valid = np.asarray(valid, bool)
    n, H, W = raw.shape
    n_fine = int(max(levels))

    hs = [build_hierarchy(raw[i], n_fine=n_fine, levels=levels) for i in range(n)]
    n_levels = len(hs[0].levels)

    total_pix = float(n * H * W)
    total_pos = float(np.count_nonzero(mask & valid))
    alive = [None] * n
    final_alive_mask = np.zeros((n, H, W), bool)
    out_levels = []

    for L in range(n_levels):
        rmaps, nregs, feats, hasch, parent_alive = [], [], [], [], []
        for i in range(n):
            h = hs[i]
            nr = int(h.levels[L]["n_regions"])
            rm = h.region_labels(L)
            if L == 0:
                a = np.ones(nr, bool)
            else:
                # Level L refines level L-1, so every fine label under one
                # L-region carries the same L-1 id: the scatter is consistent.
                par = np.zeros(nr, np.int64)
                par[h.levels[L]["map"]] = h.levels[L - 1]["map"]
                a = alive[i][par]
            rmaps.append(rm)
            nregs.append(nr)
            parent_alive.append(a)
            feats.append(region_feature(raw[i], rm, nr))
            hasch.append(region_has_chondrule(rm, mask[i], valid[i], nr))

        # Only ALIVE regions take part: dropped regions are already gone and
        # must not move the threshold.
        if thresholds is None:
            t = fit_drop_threshold(
                np.concatenate([feats[i][parent_alive[i]] for i in range(n)]),
                np.concatenate([hasch[i][parent_alive[i]] for i in range(n)]),
                target_recall)
        else:
            t = float(thresholds[L])

        live_pix = kept_pos = n_alive = 0
        for i in range(n):
            alive[i] = parent_alive[i] & (feats[i] >= t)
            lm = alive[i][rmaps[i]]
            n_alive += int(alive[i].sum())
            live_pix += int(np.count_nonzero(lm))
            kept_pos += int(np.count_nonzero(lm & mask[i] & valid[i]))
            if L == n_levels - 1:
                final_alive_mask[i] = lm

        out_levels.append({
            "level": L,
            "n_regions": int(sum(nregs)),
            "n_alive": n_alive,
            "frac_alive": live_pix / total_pix,
            "chondrule_retention": (kept_pos / total_pos if total_pos
                                    else float("nan")),
            "threshold": t,
        })

    return {"levels": out_levels, "final_alive_mask": final_alive_mask}


def run_cascade_learned(raw, mask, valid, levels=LEVELS, classifiers=None,
                        target_recall=0.999, seed=0):
    """`run_cascade` with a LEARNED per-level drop rule.

    Identical descent semantics (a region is evaluated only if its PARENT
    survived), but each level's keep/drop decision comes from a
    `LevelDropClassifier` over `region_features_rich` rather than one intensity
    percentile against a threshold.

    `classifiers=None` FITS one classifier per level on the data passed in, so
    the caller passes VAL. A list of classifiers APPLIES them frozen: that is the
    test path, and the list must be the one fit on val. [LEAKAGE][MUST]

    Returns the same dict as `run_cascade`: `{"levels": [...],
    "final_alive_mask": (n,H,W) bool}` with `level`, `n_regions`, `n_alive`,
    `frac_alive`, `chondrule_retention` and `threshold` (here the probability
    cut) per level, plus `classifiers`, the fitted-or-passed
    `LevelDropClassifier` list to freeze and hand to the test run.
    """
    raw = np.asarray(raw)
    mask = np.asarray(mask, bool)
    valid = np.asarray(valid, bool)
    n, H, W = raw.shape
    n_fine = int(max(levels))

    hs = [build_hierarchy(raw[i], n_fine=n_fine, levels=levels) for i in range(n)]
    n_levels = len(hs[0].levels)

    total_pix = float(n * H * W)
    total_pos = float(np.count_nonzero(mask & valid))
    alive = [None] * n
    final_alive_mask = np.zeros((n, H, W), bool)
    out_levels, out_clfs = [], []

    for L in range(n_levels):
        if classifiers is None:
            clf = LevelDropClassifier().fit(raw, mask, valid, hs, L,
                                            target_recall=target_recall,
                                            seed=seed)
        else:
            clf = classifiers[L]
        out_clfs.append(clf)

        live_pix = kept_pos = n_alive = n_regions = 0
        for i in range(n):
            h = hs[i]
            nr = int(h.levels[L]["n_regions"])
            rm = h.region_labels(L)
            if L == 0:
                a = np.ones(nr, bool)
            else:
                # Level L refines level L-1, so every fine label under one
                # L-region carries the same L-1 id: the scatter is consistent.
                par = np.zeros(nr, np.int64)
                par[h.levels[L]["map"]] = h.levels[L - 1]["map"]
                a = alive[i][par]
            alive[i] = a & clf.predict_keep(raw[i], rm, nr)
            lm = alive[i][rm]
            n_regions += nr
            n_alive += int(alive[i].sum())
            live_pix += int(np.count_nonzero(lm))
            kept_pos += int(np.count_nonzero(lm & mask[i] & valid[i]))
            if L == n_levels - 1:
                final_alive_mask[i] = lm

        out_levels.append({
            "level": L,
            "n_regions": n_regions,
            "n_alive": n_alive,
            "frac_alive": live_pix / total_pix,
            "chondrule_retention": (kept_pos / total_pos if total_pos
                                    else float("nan")),
            "threshold": clf.threshold,
        })

    return {"levels": out_levels, "final_alive_mask": final_alive_mask,
            "classifiers": out_clfs}


# ---------------------------------------------------------------------------
# Self-test: 4 real VAL slices (thresholds fit on val), then the learned drop
# rule fit on 8 val slices and applied frozen to 8 test slices.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw, mask, valid = zones.load_zones(".", zones=("val",))["val"]
    raw, mask, valid = raw[:4], mask[:4], valid[:4]

    h = build_hierarchy(raw[0])
    assert len(h.levels) == len(LEVELS), len(h.levels)
    assert h.labels_fine.shape == raw[0].shape, h.labels_fine.shape
    assert h.labels_fine.dtype == np.int32, h.labels_fine.dtype
    n_fine_regions = int(h.labels_fine.max()) + 1
    assert int(h.labels_fine.min()) == 0, int(h.labels_fine.min())
    assert np.unique(h.labels_fine).size == n_fine_regions, "fine map has gaps"
    r0 = h.region_labels(0)
    n0 = int(np.unique(r0).size)
    assert 15 <= n0 <= 25, n0
    assert r0.shape == raw[0].shape, r0.shape

    res = run_cascade(raw, mask, valid, target_recall=1.0)

    print(f"  val subset {raw.shape}  fine regions/slice={n_fine_regions}  "
          f"coarsest regions/slice={n0}")
    print(f"  {'level':>5} {'n_regions':>10} {'n_alive':>8} {'frac_alive':>11} "
          f"{'retention':>10} {'threshold':>10}")
    for lv in res["levels"]:
        print(f"  {lv['level']:>5} {lv['n_regions']:>10} {lv['n_alive']:>8} "
              f"{lv['frac_alive']:>11.4f} {lv['chondrule_retention']:>10.4f} "
              f"{lv['threshold']:>10.4f}")

    fin = res["levels"][-1]
    assert fin["chondrule_retention"] >= 0.99, fin["chondrule_retention"]
    fr = [lv["frac_alive"] for lv in res["levels"]]
    for a, b in zip(fr, fr[1:]):
        assert b <= a, (a, b)
    assert res["final_alive_mask"].shape == raw.shape, \
        res["final_alive_mask"].shape

    print(f"  final alive area={fin['frac_alive']:.4f} "
          f"({1.0 / max(fin['frac_alive'], 1e-12):.1f}x compute saving)  "
          f"chondrule retention={fin['chondrule_retention']:.4f}")
    print("hierarchy self-test: OK")

    # --- LEARNED drop rule: fit on 8 VAL slices, frozen, applied to 8 TEST
    # slices. The classifiers and their probability cuts come only from val;
    # the test slices are scored with them unchanged. [LEAKAGE][MUST]
    zs = zones.load_zones(".", zones=("val", "test"))
    vr, vm, vv = [a[:8] for a in zs["val"]]
    tr, tm, tv = [a[:8] for a in zs["test"]]

    learned_val = run_cascade_learned(vr, vm, vv, target_recall=0.999, seed=0)
    learned_test = run_cascade_learned(tr, tm, tv,
                                       classifiers=learned_val["classifiers"])

    # Same slices, OLD intensity rule, SAME target recall so the comparison is
    # direct: thresholds fit on the val 8, applied to the test 8.
    inten_val = run_cascade(vr, vm, vv, target_recall=0.999)
    inten_test = run_cascade(tr, tm, tv,
                             thresholds=[x["threshold"]
                                         for x in inten_val["levels"]])

    def _print_table(tag, res):
        print(f"  {tag}")
        print(f"  {'level':>5} {'n_regions':>10} {'frac_alive':>11} "
              f"{'retention':>10}")
        for x in res["levels"]:
            print(f"  {x['level']:>5} {x['n_regions']:>10} "
                  f"{x['frac_alive']:>11.4f} {x['chondrule_retention']:>10.4f}")

    print(f"  val slices {vr.shape}  test slices {tr.shape}")
    _print_table("learned drop rule - VAL (fit)", learned_val)
    _print_table("learned drop rule - TEST (frozen)", learned_test)

    lfin, ifin = learned_test["levels"][-1], inten_test["levels"][-1]
    print(f"  {'TEST rule':>22} {'frac_alive':>11} {'saving':>9} "
          f"{'retention':>10}")
    print(f"  {'learned classifier':>22} {lfin['frac_alive']:>11.4f} "
          f"{1.0 / max(lfin['frac_alive'], 1e-12):>8.1f}x "
          f"{lfin['chondrule_retention']:>10.4f}")
    print(f"  {'intensity p99':>22} {ifin['frac_alive']:>11.4f} "
          f"{1.0 / max(ifin['frac_alive'], 1e-12):>8.1f}x "
          f"{ifin['chondrule_retention']:>10.4f}")

    assert lfin["chondrule_retention"] >= 0.95, lfin["chondrule_retention"]
    print("hierarchy learned-drop self-test: OK")
