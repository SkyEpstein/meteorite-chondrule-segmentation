"""Reconstructions of the train/test split protocols used by external papers.

Spec R3. Each protocol takes the 397-slice annotated band of ALH 83010 and
returns train/val/test index sets in the same {raw, mask, valid} form the
project's model runners already consume, so a paper's protocol can be run
against this project's data with everything else held fixed.

Controls that make the protocols comparable (spec C1, C2):

* Every protocol allocates exactly the same number of units to train/val/test
  as the reference split does (213/60/80 of 397 slices, or the same fractions
  of tiles). The 44 slices the reference spends on buffer are left unused by
  every other protocol too, so no protocol gets a training-set-size advantage.
* Every protocol runs on the same centered 768 x 1280 crop. 768 = 3 * 256 and
  1280 = 5 * 256, so a 256-px tile grid partitions the field exactly and the
  tile protocols score the same pixel population as the slice protocols. The
  full 962 x 1450 field is available via `crop=False` for the tie-back run
  against the project's published number.

`valid` carries the straddler-ignore mask only for the reference protocol.
Under the leaky protocols nothing is ignored, which is precisely what those
protocols do.
"""
from __future__ import annotations
import numpy as np

TILE = 256
CROP_H, CROP_W = 768, 1280           # 3 x 5 tiles of 256, exact partition
N_TRAIN, N_VAL, N_TEST = 213, 60, 80  # matches the reference split's slab sizes


def crop_slices(vol, crop=True):
    """Center-crop a (n,H,W) volume to CROP_H x CROP_W."""
    if not crop:
        return vol
    _, H, W = vol.shape
    y0, x0 = (H - CROP_H) // 2, (W - CROP_W) // 2
    return vol[:, y0:y0 + CROP_H, x0:x0 + CROP_W]


# --------------------------------------------------------------------------
# Slice-level protocols: each returns {zone: array of band-slice indices}
# --------------------------------------------------------------------------

def slice_random(N, seed=0):
    """Random image-level split.

    The protocol behind every '80/20 random split' and '8:1:1 ratio' in the
    corpus when the unit of splitting is the image. Leaks two ways here: the
    kept slices are 4 original slices apart so neighbours are near-duplicates,
    and a chondrule spans a median of 24 band-slices, so the same physical
    object supplies training and test pixels.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    return {
        "train": np.sort(perm[:N_TRAIN]),
        "val": np.sort(perm[N_TRAIN:N_TRAIN + N_VAL]),
        "test": np.sort(perm[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]),
    }


def zone_contiguous(N, buffer=0):
    """Contiguous z zones, optionally buffered.

    buffer=0 with no straddler resolution is the naive 'we split by region'
    protocol: it removes the scattered-neighbour leak but keeps the seam leak,
    because a chondrule sitting across a boundary still donates voxels to both
    sides. buffer=12 with straddler resolution is this project's reference.

    Zone geometry follows src/split.py so the reference case reproduces the
    frozen split exactly.
    """
    slab = np.array(["train"] * N, dtype=object)
    v0, v1 = 80, 139
    t0, t1 = 160, 239
    slab[v0:v1 + 1] = "val"
    slab[t0:t1 + 1] = "test"
    for a, b in ((v0, v1), (t0, t1)):
        for i in list(range(max(0, a - buffer), a)) + \
                 list(range(b + 1, min(N, b + 1 + buffer))):
            if slab[i] == "train":
                slab[i] = "buffer"
    out = {z: np.where(slab == z)[0] for z in ("train", "val", "test")}
    # Hold train size fixed at N_TRAIN so buffer width is the only thing that
    # varies across the dose-response sweep (spec R6). Drop the training slices
    # FURTHEST from the test zone, so growing the buffer removes near slices
    # and the remaining set stays the same size.
    tr = out["train"]
    if len(tr) < N_TRAIN:
        # Silently returning a smaller training set would break the equal-size
        # control the whole comparison rests on. Buffers >= 13 reach this.
        raise ValueError(f"buffer {buffer}: only {len(tr)} train slices available, "
                         f"need {N_TRAIN}; the equal-train-size control cannot hold")
    if len(tr) > N_TRAIN:
        centre = (t0 + t1) / 2.0
        keep = tr[np.argsort(np.abs(tr - centre))[:N_TRAIN]]
        out["train"] = np.sort(keep)
    return out


# --------------------------------------------------------------------------
# Tile-level protocols: each returns {zone: array of (z, y0, x0)}
# --------------------------------------------------------------------------

def _tile_grid(N, H=CROP_H, W=CROP_W, tile=TILE):
    ys = range(0, H - tile + 1, tile)
    xs = range(0, W - tile + 1, tile)
    return np.array([(z, y, x) for z in range(N) for y in ys for x in xs], np.int32)


def tile_random(N, seed=0, tile=TILE):
    """Random patch/tile split: cut every slice into tiles, shuffle, split.

    This is the protocol used when a paper crops a small number of large images
    into many patches and then splits the patches (MudrockNet's 70/15/15 over
    crops of the same base SEM images is the clearest case in the corpus). It is
    the most leaky protocol available: for any test tile, the tile beside it in
    x, y and z is almost certainly in train, and a single chondrule is cut
    across several tiles that land in different zones.
    """
    grid = _tile_grid(N, tile=tile)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grid))
    f_tr, f_va = N_TRAIN / 397.0, N_VAL / 397.0
    n_tr = int(round(f_tr * len(grid)))
    n_va = int(round(f_va * len(grid)))
    n_te = int(round((N_TEST / 397.0) * len(grid)))
    return {
        "train": grid[perm[:n_tr]],
        "val": grid[perm[n_tr:n_tr + n_va]],
        "test": grid[perm[n_tr + n_va:n_tr + n_va + n_te]],
    }


def patch_random_overlap(N, seed=0, tile=TILE, n_patches=6000):
    """Randomly positioned, freely overlapping patches, then split.

    The extreme of the family: patches are sampled at arbitrary (z, y, x) rather
    than on a grid, so a train patch and a test patch can share pixels outright.
    Papers that 'extract N random crops' before splitting land here.
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(0, N, n_patches)
    y = rng.integers(0, CROP_H - tile + 1, n_patches)
    x = rng.integers(0, CROP_W - tile + 1, n_patches)
    grid = np.stack([z, y, x], axis=1).astype(np.int32)
    n_tr = int(round(N_TRAIN / 397.0 * n_patches))
    n_va = int(round(N_VAL / 397.0 * n_patches))
    n_te = int(round(N_TEST / 397.0 * n_patches))
    return {
        "train": grid[:n_tr],
        "val": grid[n_tr:n_tr + n_va],
        "test": grid[n_tr + n_va:n_tr + n_va + n_te],
    }


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------

def materialize_slices(raw, mask, valid, zone_idx):
    """Subset a (N,H,W) band by slice index -> {raw, mask, valid} per zone."""
    return {z: {"raw": raw[i].astype(np.float32),
                "mask": mask[i],
                "valid": valid[i]}
            for z, i in zone_idx.items()}


def materialize_tiles(raw, mask, valid, zone_coords, tile=TILE):
    """Cut (z,y0,x0) coordinate lists out of the band -> (n_tiles,tile,tile)."""
    out = {}
    for z, coords in zone_coords.items():
        R = np.empty((len(coords), tile, tile), np.float32)
        M = np.empty((len(coords), tile, tile), bool)
        V = np.empty((len(coords), tile, tile), bool)
        for j, (zz, y0, x0) in enumerate(coords):
            R[j] = raw[zz, y0:y0 + tile, x0:x0 + tile]
            M[j] = mask[zz, y0:y0 + tile, x0:x0 + tile]
            V[j] = valid[zz, y0:y0 + tile, x0:x0 + tile]
        out[z] = {"raw": R, "mask": M, "valid": V}
    return out


def straddler_ignore(inst, zc, ninst):
    """Voxels to ignore so no chondrule keeps labels in two core zones.

    Same rule as src/split.py: assign each instance to the zone holding most of
    its voxels, then ignore that instance's voxels in every other core zone.
    Returns a (N,H,W) bool mask of voxels to drop.

    A buffer alone does NOT achieve this. Chondrules here reach 70 band-slices
    in z-extent while the buffer is 12, so the longest ones still span a seam.
    Measured on this volume, buffer-12 alone leaves 8 chondrules with voxels in
    both train and test, carrying 19.9% of test positive pixels.
    """
    counts = np.zeros((ninst + 1, 4), np.int64)
    for z in range(inst.shape[0]):
        iz, cz = inst[z].ravel(), zc[z].ravel()
        for code in (1, 2, 3):
            m = cz == code
            if m.any():
                counts[:, code] += np.bincount(iz[m], minlength=ninst + 1)
    maj = np.zeros(ninst + 1, np.int64)
    multi = ((counts[:, 1:4] > 0).sum(axis=1) >= 2)
    multi[0] = False
    maj[multi] = np.argmax(counts[multi, 1:4], axis=1) + 1
    ignore = np.zeros(inst.shape, bool)
    for z in range(inst.shape[0]):
        m = maj[inst[z]]
        ignore[z] = (m != 0) & (m != zc[z]) & (zc[z] != 0)
    return ignore


PROTOCOLS = {
    "P-tile-random": dict(kind="tile", fn=tile_random, straddler=False,
                          label="random tile/patch split"),
    "P-patch-random-overlap": dict(kind="tile", fn=patch_random_overlap,
                                   straddler=False,
                                   label="random overlapping patches"),
    "P-slice-random": dict(kind="slice", fn=slice_random, straddler=False,
                           label="random image (slice) split"),
    "P-zone-nobuffer": dict(kind="slice", straddler=False,
                            fn=lambda N, seed=0: zone_contiguous(N, buffer=0),
                            label="contiguous zones, no buffer"),
    "P-zone-buffer": dict(kind="slice", straddler=False,
                          fn=lambda N, seed=0: zone_contiguous(N, buffer=12),
                          label="contiguous zones, buffer 12, no straddler fix"),
    "P-zone-buffer-straddler": dict(kind="slice", straddler=True,
                                    fn=lambda N, seed=0: zone_contiguous(N, buffer=12),
                                    label="contiguous zones, buffer 12 + straddler "
                                          "resolution (this project's reference)"),
}
