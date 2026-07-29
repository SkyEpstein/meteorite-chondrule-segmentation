"""Segmentation metrics (spec R4). Foreground = chondrule (positive) class.

All hard metrics are computed only over *valid* pixels (an optional boolean
mask); pixels marked invalid — e.g. voxels of dropped straddler chondrules —
are excluded from every count. The operating threshold is chosen on validation
and frozen for the test report (constitution P2).
"""
from __future__ import annotations
import numpy as np


def _counts(pred_bool, gt_bool, valid=None):
    if valid is not None:
        pred_bool = pred_bool & valid
        gt_bool = gt_bool & valid
    tp = int(np.count_nonzero(pred_bool & gt_bool))
    fp = int(np.count_nonzero(pred_bool & ~gt_bool))
    fn = int(np.count_nonzero(~pred_bool & gt_bool))
    return tp, fp, fn


def metrics_from_pred(pred_bool, gt_bool, valid=None):
    """Full hard-metric dict from a boolean prediction."""
    tp, fp, fn = _counts(pred_bool, gt_bool, valid)
    di, dd = tp + fp + fn, 2 * tp + fp + fn
    nan = float("nan")
    return {
        "iou": tp / di if di else nan,
        "dice": 2 * tp / dd if dd else nan,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "recall": tp / (tp + fn) if (tp + fn) else nan,
        "tp": tp, "fp": fp, "fn": fn,
    }


def iou(pred_bool, gt_bool, valid=None):
    tp, fp, fn = _counts(pred_bool, gt_bool, valid)
    d = tp + fp + fn
    return tp / d if d else float("nan")


def per_slice_iou(pred_bool, gt_bool, valid=None):
    """IoU computed independently per z-slice (axis 0). NaN where a slice has
    no positive in GT nor prediction."""
    out = []
    for i in range(pred_bool.shape[0]):
        v = None if valid is None else valid[i]
        out.append(iou(pred_bool[i], gt_bool[i], v))
    return np.array(out, dtype=float)


def auprc(prob, gt_bool, valid=None):
    """Area under precision-recall curve (threshold-free; good for imbalance)."""
    from sklearn.metrics import average_precision_score
    p = prob[valid] if valid is not None else prob.ravel()
    g = gt_bool[valid] if valid is not None else gt_bool.ravel()
    if g.sum() == 0 or g.all():
        return float("nan")
    return float(average_precision_score(g.astype(np.uint8), p))


def expected_calibration_error(prob, gt_bool, valid=None, bins=10):
    """ECE: mean |confidence - accuracy| over probability bins."""
    p = prob[valid] if valid is not None else prob.ravel()
    g = (gt_bool[valid] if valid is not None else gt_bool.ravel()).astype(np.float64)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, p.size
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & (p <= hi) if b == bins - 1 else (p >= lo) & (p < hi)
        k = int(sel.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(p[sel].mean() - g[sel].mean())
    return float(ece)


def select_threshold(prob, gt_bool, valid=None, grid=None, metric="iou"):
    """Pick the probability threshold maximizing `metric` on the given data."""
    if grid is None:
        # Floor must sit far below 0.05. Dice-trained CNNs here operate at
        # 0.002-0.05, so a grid starting at 0.05 pins the selected threshold to
        # its own boundary — measured: `unet2d_s30000` selected exactly 0.05,
        # which corrupted the val ranking used to choose between models. This is
        # the same truncated-grid defect logged three times in CHANGELOG.md.
        grid = np.concatenate([
            np.geomspace(1e-6, 0.05, 24),
            np.linspace(0.05, 0.95, 19),
        ])
    best_t, best_v = float(grid[0]), -1.0
    for t in grid:
        m = metrics_from_pred(prob >= t, gt_bool, valid)[metric]
        if m == m and m > best_v:  # skip NaN
            best_v, best_t = m, float(t)
    return best_t, best_v


def _selfcheck():
    """A2: verify counts on a hand-built toy mask."""
    gt = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], bool)      # 3 positives
    pr = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]], bool)      # 3 predicted
    # TP=2 (0,0),(1,1); FP=1 (1,2); FN=1 (0,1)  -> IoU=2/4=.5  Dice=4/6=.667
    m = metrics_from_pred(pr, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1), m
    assert abs(m["iou"] - 0.5) < 1e-9 and abs(m["dice"] - 2 / 3) < 1e-9, m
    # valid mask excluding (1,2): FP removed -> TP=2 FP=0 FN=1 -> IoU=2/3
    valid = np.ones_like(gt); valid[1, 2] = False
    m2 = metrics_from_pred(pr, gt, valid)
    assert (m2["tp"], m2["fp"], m2["fn"]) == (2, 0, 1), m2
    assert abs(m2["iou"] - 2 / 3) < 1e-9, m2
    return True


if __name__ == "__main__":
    print("metrics self-check:", _selfcheck())
