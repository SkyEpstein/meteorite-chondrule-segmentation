"""Patch pool for the synthetic-data generator, with the leakage boundary enforced.

Stage 1 of the label-conditioned VQ-VAE + MaskGIT build
(`specs/2026-08-08-synthetic-maskgit/spec.md`).

[LEAKAGE][MUST] This module is the single place that decides what the generator
is allowed to see, and it ASSERTS the boundary rather than trusting it. A
generator that has seen val or test pixels launders them into the segmenter's
training set through its samples, and every downstream number would be void
without anything looking wrong.

Two pools, deliberately different:

  TOKENIZER pool   train-zone slices + BOTH unlabeled regions.
                   The VQ-VAE only reconstructs pixels, so it needs no labels
                   and can use the 134 slices outside the annotated band that
                   the segmenter cannot use at all. This is the one real
                   advantage the two-stage design has over simpler augmentation.

  MASKGIT pool     train-zone slices ONLY.
                   Conditioning needs label maps, which exist only inside the
                   annotated band, and only the train zone is ours to look at.

The unlabeled regions are the 134 slices OUTSIDE the annotated band, derived
from `split.json`'s band rather than by subtracting the zones. The nearest
held-out slice is 81 (val) and 158 (test) away, against this project's buffer
standard of 12. `describe()` prints the measured distances.

[HISTORY] Two boundary bugs, both in this file, both from re-deriving what
split.json already states. Subtracting the zones left all 44 BUFFER slices, one
away from held-out data. The distance-based fix used `gap < 12` while
`split.mark_buffer` marks distances 1 THROUGH 12, so it still admitted slices
168 and 351, which split.json labels `buffer`. That second one was found by
review, not by `assert_boundary`, because the assertion re-used the same
predicate that built the pool. `assert_boundary` now checks split.json's own
slab labels, and `tests/test_synth_boundary.py` fails on the old behaviour.

Patch geometry: 128 px patches, to match the segmenter's own training patch. The
VQ-VAE downsample factor is NOT fixed here and the working value is **4**, not 8:
f=8 (a 16x16 grid) was measured to strip the matrix texture the segmenter reads
and more than double its predicted-positive fraction, so it is rejected. f=4
gives a 32x32 grid, about 7 tokens across a 60 px chondrule. See
`src/synth_vqvae.py` and the 2026-08-08 CHANGELOG entry.

Run:  python -m src.synth_patches            # build and cache the pools
      python -m src.synth_patches --describe # boundary report only, no extraction
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from . import data, zones

PATCH = 128
OUT = "results/synth/patches.npz"
CACHE = "split_cache.npz"


def zone_indices(root=".", path=CACHE):
    """Absolute slice indices per zone, straight from the frozen split cache."""
    full = os.path.join(root, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{full} missing, run `python -m src.zones` first")
    d = np.load(full)
    return {z: d[f"abs_{z}"].astype(int).tolist() for z in ("train", "val", "test")}


MIN_GAP = 12          # the split's own buffer width
SPLIT_JSON = "split.json"


def band_and_slabs(root="."):
    """The annotated band's absolute indices and each one's slab label.

    `split.json` is the AUTHORITY on which slices are train / val / test /
    buffer. Every boundary decision here is taken from it rather than
    re-derived, because re-deriving it is what went wrong (see below).
    """
    with open(os.path.join(root, SPLIT_JSON)) as f:
        d = json.load(f)
    return dict(zip([int(i) for i in d["band_abs_slices"]],
                    d["slab_per_band_slice"]))


def unlabeled_indices(root=".", path=CACHE, min_gap=MIN_GAP):
    """Slices with NO labels at all, i.e. those OUTSIDE the annotated band.

    [LEAKAGE][MUST] Derived from the BAND, not by subtracting the zones. The
    subtraction is what produced two successive bugs, and the second one is
    instructive:

      1. Subtracting train/val/test leaves the **buffer** slices, which are
         in-band and sit adjacent to val and test. That admitted 178 slices
         whose nearest held-out neighbour was 1 away.
      2. The "fix" added a distance test, `gap < min_gap`. But `split.mark_buffer`
         marks `range(a - buffer, a)`, which is distances 1 THROUGH 12, so
         distance 12 is INSIDE the buffer. `gap < 12` therefore admitted exactly
         the outermost buffer slice on each side: absolute slices 168 and 351,
         both labelled `buffer` by split.json itself. Found by review, not by
         the check that was supposed to prevent it.

    Slices outside the band carry no labels by definition, so this needs no
    distance heuristic at all. The distance test is kept only as a redundant
    guard, and `assert_boundary` now checks the split's OWN labels rather than
    re-applying this function's reasoning.
    """
    slabs = band_and_slabs(root)
    zi = zone_indices(root, path)
    held = np.asarray(sorted(zi["val"] + zi["test"]), dtype=int)
    n_total = len(data.list_pairs(root)[0])
    keep = []
    for i in range(n_total):
        if i in slabs:
            continue                      # in-band: labelled, and not ours
        if held.size and int(np.min(np.abs(held - i))) <= min_gap:
            continue                      # redundant guard, <= not <
        keep.append(i)
    return keep


def pools(root=".", path=CACHE, check=True):
    """Build the pools. `check=True` asserts the boundary before returning, so
    no caller can obtain an unchecked pool by using this instead of describe()."""
    zi = zone_indices(root, path)
    unl = unlabeled_indices(root, path)
    p = {"tokenizer": sorted(zi["train"] + unl),
         "maskgit": sorted(zi["train"]),
         "heldout": sorted(zi["val"] + zi["test"]),
         "train": sorted(zi["train"]), "unlabeled": sorted(unl),
         "val": sorted(zi["val"]), "test": sorted(zi["test"])}
    if check:
        assert_boundary(p, root)
    return p


def assert_boundary(p, root="."):
    """[MUST] Fail loudly if anything the split withheld reaches a generator pool.

    [REWRITTEN 2026-08-08 after review] The previous version re-applied the same
    `gap < MIN_GAP` predicate that `unlabeled_indices` used to BUILD the pool,
    so it could only fire if that function disagreed with itself. It passed a
    pool containing two slices that `split.json` labels `buffer`. A check
    derived from the thing it is checking is not a check.

    The authority is `split.json`'s own `slab_per_band_slice`. Any in-band slice
    in a generator pool must be labelled `train`; `val`, `test` and `buffer` are
    all disqualifying, and `buffer` is the case the old test missed.
    """
    slabs = band_and_slabs(root)
    for nm in ("tokenizer", "maskgit"):
        bad = [(i, slabs[i]) for i in p[nm] if slabs.get(i, "train") != "train"]
        if bad:
            raise SystemExit(
                f"LEAKAGE: {len(bad)} slice(s) in the {nm} pool are not `train` "
                f"in split.json: {bad[:10]}. `buffer` counts: it is in-band and "
                f"adjacent to held-out data, which is what the buffer is for.")
    held = set(p["heldout"])
    for nm in ("tokenizer", "maskgit"):
        bad = sorted(held & set(p[nm]))
        if bad:
            raise SystemExit(
                f"LEAKAGE: {len(bad)} held-out slice(s) in the {nm} pool: "
                f"{bad[:10]}. The generator must never see val or test.")
    overlap = sorted(set(p["unlabeled"]) & set(p["train"]))
    if overlap:
        raise SystemExit(f"unlabeled and train pools overlap: {overlap[:10]}")
    # [MUST] Membership is not sufficient. A buffer slice is not a val or test
    # index yet sits one slice from one, and adjacent CT slices are
    # near-duplicates. Every added slice must clear the split's own buffer.
    held = np.asarray(p["heldout"], dtype=int)
    for nm in ("tokenizer", "maskgit"):
        for i in p[nm]:
            if i in set(p["train"]):
                continue              # train is inside the split's own contract
            gap = int(np.min(np.abs(held - int(i))))
            # `<=`, NOT `<`. mark_buffer covers distances 1 THROUGH MIN_GAP, so
            # distance == MIN_GAP is inside the buffer. This guard kept the
            # strict form for one round after the builder was fixed, which is
            # how the same off-by-one survived in the same file as the docstring
            # explaining it.
            if gap <= MIN_GAP:
                raise SystemExit(
                    f"LEAKAGE: slice {i} in the {nm} pool is {gap} slices from a "
                    f"held-out slice, under the buffer of {MIN_GAP}. Adjacent CT "
                    f"slices are near-duplicates; this is what the buffer is for.")
    return True


def _min_distance(a, b):
    if not a or not b:
        return None
    bb = np.asarray(sorted(b))
    return int(min(abs(int(x) - int(bb[np.argmin(np.abs(bb - x))])) for x in a))


def describe(root="."):
    p = pools(root)
    print("=== generator data boundary "
          "(slabs from split.json, zone indices from split_cache.npz) ===")
    for nm in ("train", "unlabeled", "val", "test"):
        v = p[nm]
        rng = f"{v[0]}..{v[-1]}" if v else "(empty)"
        print(f"  {nm:<10} {len(v):>4} slices   {rng}")
    print(f"  {'TOKENIZER':<10} {len(p['tokenizer']):>4} slices "
          f"(train + unlabeled)")
    print(f"  {'MASKGIT':<10} {len(p['maskgit']):>4} slices (train only)")
    for nm in ("val", "test"):
        d = _min_distance(p["unlabeled"], p[nm])
        print(f"  nearest unlabeled slice to {nm:<5} is {d} slices away "
              f"(project buffer standard is 12)")
    print("  no held-out slice appears in either generator pool   OK")
    return p


def extract(root=".", n_per_slice=24, seed=0, out=OUT, min_std=1.0):
    """Sample PATCH x PATCH crops from each pool slice.

    `min_std` rejects crops that are almost entirely background air: with ~96%
    class imbalance a uniformly random crop is usually empty, and a tokenizer
    trained mostly on flat black learns to spend its codebook there.
    """
    p = describe(root)
    rng = np.random.default_rng(seed)
    raw_paths, _ = data.list_pairs(root)

    out_patches, out_src, out_labeled = [], [], []
    train_set = set(p["train"])
    for k, idx in enumerate(p["tokenizer"]):
        img = data.load_stack(raw_paths, [idx], "raw")[0].astype(np.float32)
        H, W = img.shape
        got = 0
        for _ in range(n_per_slice * 8):
            if got >= n_per_slice:
                break
            y = int(rng.integers(0, max(1, H - PATCH)))
            x = int(rng.integers(0, max(1, W - PATCH)))
            c = img[y:y + PATCH, x:x + PATCH]
            if c.shape != (PATCH, PATCH) or float(c.std()) < min_std:
                continue
            out_patches.append(c.astype(np.float32))
            out_src.append(idx)
            out_labeled.append(idx in train_set)
            got += 1
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(p['tokenizer'])} slices, "
                  f"{len(out_patches)} patches", flush=True)

    P = np.stack(out_patches)
    lo, hi = float(np.percentile(P, 0.5)), float(np.percentile(P, 99.5))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, patches=P.astype(np.float32), src=np.asarray(out_src, np.int32),
        labeled=np.asarray(out_labeled, bool),
        norm_lo=np.float32(lo), norm_hi=np.float32(hi),
        tokenizer_slices=np.asarray(p["tokenizer"], np.int32),
        maskgit_slices=np.asarray(p["maskgit"], np.int32),
        heldout_slices=np.asarray(p["heldout"], np.int32))
    print(f"\nwrote {out}")
    print(f"  {P.shape[0]} patches of {PATCH}x{PATCH}, "
          f"{int(np.sum(out_labeled))} from labeled train slices")
    print(f"  intensity 0.5/99.5 percentile for normalisation: {lo:.1f} / {hi:.1f}")
    return P


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--describe" in argv:
        return describe()
    n = 24
    for a in argv:
        if a.startswith("--n_per_slice="):
            n = int(a.split("=", 1)[1])
    return extract(n_per_slice=n)


if __name__ == "__main__":
    main()
