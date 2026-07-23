"""Classical intensity + morphology baseline (spec R5; constitution P4).

Chondrules are denser (brighter) than matrix in CT (mean ~50 vs ~32), so a
high-intensity threshold plus morphological cleanup is a legitimate,
non-learned reference. Its foreground-IoU on the held-out slab is the number
every learned model in v2 must beat to justify itself.
"""
from __future__ import annotations
import numpy as np
from skimage import morphology


def intensity_prob(raw_stack):
    """Pseudo-probability in [0,1] from raw intensity (1st-99th pct scaling).

    Used only for threshold-free ranking metrics (AUPRC) and calibration; the
    hard prediction uses `predict` with a tuned intensity cut.
    """
    r = raw_stack.astype(np.float32)
    lo, hi = np.percentile(r, 1), np.percentile(r, 99)
    return np.clip((r - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def predict(raw_stack, thr_intensity, min_size=32, close_radius=1):
    """Threshold at `thr_intensity`, morphologically close, drop small objects.

    Operates slice-by-slice (2D morphology) to match the 2D framing.
    """
    pred = raw_stack >= thr_intensity
    selem = morphology.disk(close_radius)
    out = np.zeros_like(pred)
    for i in range(pred.shape[0]):
        p = morphology.binary_closing(pred[i], selem)
        p = morphology.remove_small_objects(p, min_size=min_size)
        out[i] = p
    return out


def tune_threshold(raw_val, mask_val, valid_val, lo=20, hi=120, step=2,
                   min_size=32, close_radius=1):
    """Sweep intensity thresholds on validation; return (best_thr, best_iou).

    Imported here to avoid a circular import at module load.
    """
    from . import metrics
    best_thr, best_iou = lo, -1.0
    for thr in range(lo, hi, step):
        pred = predict(raw_val, thr, min_size=min_size, close_radius=close_radius)
        m = metrics.iou(pred, mask_val, valid_val)
        if m == m and m > best_iou:
            best_iou, best_thr = m, thr
    return best_thr, best_iou
