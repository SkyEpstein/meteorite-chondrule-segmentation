"""Multiscale per-pixel texture features (spec v2, tier B & GNN nodes).

Given a 2D CT slice, returns an (H, W, F) float32 feature stack. Features are
classic hand-engineered descriptors that expose intensity, edges, blobs, local
statistics, oriented texture and ridge/blob structure at several scales: the
signal a threshold alone can't capture.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def feature_names():
    names = ["intensity"]
    names += [f"gauss{s}" for s in (1, 2, 4, 8)]
    names += ["grad1", "grad2"]
    names += [f"log{s}" for s in (1, 2, 4)]
    names += ["lmean5", "lstd5", "lstd11"]
    names += [f"gabor_f{f}_t{t}" for f in (0.15, 0.3) for t in (0, 90)]
    names += ["hess_large", "hess_small"]
    return names


def extract_features(sl):
    """(H,W) slice -> (H,W,F) float32 features."""
    x = np.asarray(sl, dtype=np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    feats = [x]
    for s in (1, 2, 4, 8):
        feats.append(ndi.gaussian_filter(x, s))
    for s in (1, 2):
        g = ndi.gaussian_filter(x, s)
        gx = ndi.sobel(g, axis=1)
        gy = ndi.sobel(g, axis=0)
        feats.append(np.hypot(gx, gy))
    for s in (1, 2, 4):
        feats.append(ndi.gaussian_laplace(x, s))
    m5 = ndi.uniform_filter(x, 5)
    feats.append(m5)
    feats.append(np.sqrt(np.maximum(ndi.uniform_filter(x * x, 5) - m5 * m5, 0)))
    m11 = ndi.uniform_filter(x, 11)
    feats.append(np.sqrt(np.maximum(ndi.uniform_filter(x * x, 11) - m11 * m11, 0)))
    for f in (0.15, 0.3):
        for t in (0.0, np.pi / 2):
            re, im = gabor(x, frequency=f, theta=t)
            feats.append(np.hypot(re, im).astype(np.float32))
    H = hessian_matrix(x, sigma=2, order="rc", use_gaussian_derivatives=True)
    big, small = hessian_matrix_eigvals(H)
    feats.append(big)
    feats.append(small)
    return np.stack([f.astype(np.float32) for f in feats], axis=-1)
