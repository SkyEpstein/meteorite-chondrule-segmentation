"""Classical Stage-0 segmenter: denoise -> CLAHE -> texture separation -> watershed.

Sky's proposal (2026-07-30). This is a **Stage 0 base segmenter** under the
constitution §12 contract — it plugs into the same slot the U-Net occupies, and
stages 1-3 (hierarchy filter, learned confidence/error model, z-neighbour
confidence) apply on top unchanged.

Why this is worth trying
------------------------
The meteorite micro-CT literature attributes poor simple-classifier performance
to "an overlap in the range of intensities present in the different components"
— exactly the overlap measured here (chondrule-containing superpixels mean 0.2506
vs empty rock 0.1578, with empty rock's p90 ABOVE the chondrule mean). The same
work reports local-histogram classification beating pixel intensity 85.8% vs
73.4% percent-correct. CLAHE is a local-histogram method.

Watershed is separately well matched: chondrules are rigid spheroids that touch,
and the CNN's failure mode is blobby over-prediction — 33 predicted objects vs 10
true, at 2.06x the true area on test slice 40.

[MUST] CLAHE is an ADDITIONAL channel, never a replacement
----------------------------------------------------------
CLAHE equalises contrast locally and therefore discards ABSOLUTE intensity. But
absolute intensity is the strongest single signal in this data (chondrules ~50
vs matrix ~32 in 8-bit), and destroying it is exactly what made intensity-jitter
augmentation halve val IoU, 0.387 -> 0.202. The domain result above is about
local-histogram FEATURES ADDED ALONGSIDE intensity, not equalisation replacing
it. So `channels()` always returns raw intensity next to the CLAHE channel.

[MUST] The output is a PROBABILITY, not a mask
----------------------------------------------
Stages 1-3 consume a graded score: the confidence model's features are `prob`,
`|prob - 0.5|`, `z_agreement(prob)`, `z_smooth(prob)` and local statistics OF
prob. A bare watershed label map is binary, so it would hand the confidence stage
zero gradation — a worse version of the saturation that already flatlines it on
the CNN (91% of pixels below 1e-4, only 0.32% graded). Watershed therefore does
instance separation, while the returned score stays continuous.

[LEAKAGE][MUST] Every parameter here is tuned on VALIDATION and frozen before
test is scored once.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

# skimage 0.24 verified present.
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel
from skimage.morphology import disk, white_tophat
from skimage.restoration import denoise_bilateral
from skimage.segmentation import watershed as sk_watershed


# ---------------------------------------------------------------------------
# 1. Denoise
# ---------------------------------------------------------------------------
def denoise(sl, method="median", strength=2):
    """(H,W) uint8/float -> (H,W) float32 in [0,1].

    `median` is the default deliberately: it removes micro-CT speckle while
    preserving the step edges that chondrule boundaries consist of, and it is
    ~100x cheaper than bilateral or NL-means on a 962x1450 slice. Aggressive
    smoothing is a real risk here — the boundary IS the thing being segmented,
    and the annotation ceiling is only 0.9226 at +/-1px.
    """
    x = np.asarray(sl, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    if method == "none":
        return x
    if method == "median":
        return ndi.median_filter(x, size=int(strength) * 2 + 1).astype(np.float32)
    if method == "gaussian":
        return ndi.gaussian_filter(x, sigma=float(strength)).astype(np.float32)
    if method == "bilateral":
        return denoise_bilateral(x, sigma_color=0.05,
                                 sigma_spatial=float(strength)).astype(np.float32)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# 2. CLAHE  +  3. morphological texture separation
# ---------------------------------------------------------------------------
def clahe(sl, clip=0.01, ntiles=8):
    """Contrast Limited Adaptive Histogram Equalisation -> (H,W) float32 [0,1]."""
    x = np.asarray(sl, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return equalize_adapthist(x, kernel_size=max(x.shape) // ntiles,
                              clip_limit=clip).astype(np.float32)


def texture_separation(sl, radius=15):
    """White top-hat: bright structures SMALLER than `radius`, background removed.

    Chondrules are compact bright inclusions on a darker textured matrix, so a
    top-hat with a radius above the chondrule scale suppresses the slowly-varying
    matrix and keeps the inclusions. Returns (H,W) float32.
    """
    x = np.asarray(sl, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return white_tophat(x, disk(int(radius))).astype(np.float32)


def channels(sl, denoise_method="median", denoise_strength=2,
             clip=0.01, radius=15):
    """The stacked representation: (H,W,4) float32.

    0 raw        — ABSOLUTE intensity, always kept (see the CLAHE note above)
    1 denoised   — speckle removed, edges preserved
    2 clahe      — local contrast, computed ON the denoised image
    3 tophat     — morphological texture separation
    """
    x = np.asarray(sl, np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    dn = denoise(x, denoise_method, denoise_strength)
    return np.stack([x, dn, clahe(dn, clip), texture_separation(dn, radius)],
                    axis=-1)


# ---------------------------------------------------------------------------
# 4. Watershed  (instance separation on a CONTINUOUS score)
# ---------------------------------------------------------------------------
def watershed_split(score, thr, min_distance=8, min_size=64):
    """Split touching blobs in a continuous `score` map. Returns (labels, mask).

    Markers are local maxima of the distance transform of the thresholded mask —
    the standard construction for touching convex objects. The watershed runs on
    the SOBEL gradient of the score so basins stop at score edges.
    """
    mask = score >= thr
    if not mask.any():
        return np.zeros(score.shape, np.int32), mask
    dist = ndi.distance_transform_edt(mask)
    # local maxima of the distance transform -> one marker per blob
    peaks = (dist >= min_distance) & (dist == ndi.maximum_filter(dist, size=3))
    markers, n = ndi.label(peaks)
    if n == 0:                                   # nothing separable: keep as-is
        markers, n = ndi.label(mask)
    lab = sk_watershed(sobel(score.astype(np.float32)), markers, mask=mask)
    if min_size > 0 and lab.max() > 0:
        counts = np.bincount(lab.ravel())
        drop = np.flatnonzero(counts < min_size)
        if drop.size:
            lab[np.isin(lab, drop)] = 0
    return lab.astype(np.int32), lab > 0


def _selftest():
    """Two touching discs must come out as TWO watershed labels, not one."""
    H = W = 80
    yy, xx = np.mgrid[0:H, 0:W]
    a = ((yy - 40) ** 2 + (xx - 32) ** 2) < 15 ** 2
    b = ((yy - 40) ** 2 + (xx - 50) ** 2) < 15 ** 2
    score = np.where(a | b, 0.9, 0.1).astype(np.float32)
    merged = int(ndi.label(score >= 0.5)[1])
    lab, m = watershed_split(score, 0.5, min_distance=6, min_size=20)
    n = int(len(np.unique(lab)) - 1)
    assert merged == 1, f"discs should overlap into one blob, got {merged}"
    assert n == 2, f"watershed should split them into 2, got {n}"

    ch = channels((np.random.default_rng(0).random((40, 50)) * 255).astype(np.uint8))
    assert ch.shape == (40, 50, 4), ch.shape
    assert np.isfinite(ch).all()
    return {"merged_blobs_before": merged, "watershed_labels_after": n,
            "channels": ch.shape[-1]}


if __name__ == "__main__":
    r = _selftest()
    print(f"watershed: {r['merged_blobs_before']} merged blob -> "
          f"{r['watershed_labels_after']} separated labels  OK")
    print(f"channels(): {r['channels']} planes [raw, denoised, clahe, tophat]  OK")
    print("raw intensity is ALWAYS channel 0 — CLAHE never replaces it")
