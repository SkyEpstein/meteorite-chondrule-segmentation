"""The generator's leakage boundary, tested against split.json rather than itself.

Spec `specs/2026-08-08-synthetic-maskgit/spec.md` task 1 requires "an automated
check that fails if any val or test slice index enters the generator's pool".
`assert_boundary` is a runtime guard inside the production path, not a test, and
its first version was self-consistent with the bug it was meant to prevent, so
it passed a pool containing two `buffer` slices.

Every check here derives its expectation from `split.json`, which is the
authority on what each slice is. None of them re-uses `synth_patches`' own
reasoning, which is the property the original check lacked.

REGRESSION UNDER TEST (2026-08-08): absolute slices 168 and 351 are labelled
`buffer` by split.json and sat in the tokenizer pool, because the boundary used
`gap < 12` while `split.mark_buffer` marks distances 1 THROUGH 12. `test_no_buffer_slices`
fails on that bug and passes on the fix.

Run:  python -m tests.test_synth_boundary
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import synth_patches as sp                                # noqa: E402

FAILURES = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main():
    with open("split.json") as f:
        sj = json.load(f)
    slabs = dict(zip([int(i) for i in sj["band_abs_slices"]],
                     sj["slab_per_band_slice"]))
    p = sp.pools()
    held = set(p["val"]) | set(p["test"])

    print("=== the generator boundary, checked against split.json ===")

    # 1. The headline contract.
    for nm in ("tokenizer", "maskgit"):
        check(not (held & set(p[nm])), f"no val/test slice in the {nm} pool")

    # 2. THE REGRESSION. buffer slices are in-band and adjacent to held-out data.
    for nm in ("tokenizer", "maskgit"):
        bad = [(i, slabs[i]) for i in p[nm] if slabs.get(i, "train") != "train"]
        check(not bad, f"no non-train in-band slice in the {nm} pool",
              f"found {bad[:5]}" if bad else "")

    # 3. Distance, independently of how the pool was built.
    if held:
        worst = min(min(abs(i - h) for h in held) for i in p["unlabeled"])
        check(worst > sj["buffer"],
              f"every unlabeled slice clears the buffer of {sj['buffer']}",
              f"nearest is {worst}")

    # 4. The unlabeled pool is exactly the out-of-band slices, no derivation.
    check(all(i not in slabs for i in p["unlabeled"]),
          "every unlabeled slice is outside the annotated band")

    # 5. assert_boundary must actually RAISE. The original never fired, so
    #    "it passed" carried no information. Inject a buffer slice and a test
    #    slice and require a SystemExit for each.
    buffers = [i for i, s in slabs.items() if s == "buffer"]
    for label, inject in (("a buffer slice", buffers[:1]),
                          ("a test slice", sorted(p["test"])[:1])):
        if not inject:
            continue
        bad = dict(p)
        bad["tokenizer"] = sorted(set(p["tokenizer"]) | set(inject))
        try:
            sp.assert_boundary(bad)
            check(False, f"assert_boundary raises on {label}", "it did NOT raise")
        except SystemExit:
            check(True, f"assert_boundary raises on {label}")

    # 6. pools() must self-check, so no caller can obtain an unchecked pool.
    import inspect
    src = inspect.getsource(sp.pools)
    check("assert_boundary" in src, "pools() asserts the boundary itself")

    # 7. CARDINALITY. Every check above passes vacuously on an empty pool, so a
    #    corrupt split_cache would report all-OK. Pin the sizes.
    check(len(p["tokenizer"]) == 347, "tokenizer pool is 347 slices",
          f"got {len(p['tokenizer'])}")
    check(len(p["maskgit"]) == 213, "maskgit pool is 213 slices",
          f"got {len(p['maskgit'])}")
    check(len(p["unlabeled"]) == 134, "unlabeled pool is 134 slices",
          f"got {len(p['unlabeled'])}")

    # 8. THE OFF-BY-ONE, on a SYNTHETIC split. mark_buffer covers distances 1
    #    THROUGH buffer, so a slice at distance exactly `buffer` is inside it.
    #    This survived in assert_boundary for a round after the builder was
    #    fixed. It cannot be probed against the real split, because every real
    #    slice at distance 12 is in-band and the label check catches it first,
    #    so the distance branch is unreachable with this geometry. A synthetic
    #    band makes it reachable and pins the comparison operator behaviourally
    #    rather than by reading the source.
    real_bs = sp.band_and_slabs
    try:
        sp.band_and_slabs = lambda root=".": {i: "val" for i in range(100, 110)}
        probe = 100 - sp.MIN_GAP          # distance exactly MIN_GAP, out of band
        synth = {"tokenizer": [probe], "maskgit": [], "train": [],
                 "heldout": list(range(100, 110)), "unlabeled": [probe],
                 "val": list(range(100, 110)), "test": []}
        try:
            sp.assert_boundary(synth)
            check(False, f"assert_boundary raises at gap == MIN_GAP",
                  "it did NOT raise: the < vs <= off-by-one is back")
        except SystemExit:
            check(True, "assert_boundary raises at gap == MIN_GAP")
        # and one slice further out must be accepted, or the guard is too strict
        synth["tokenizer"] = synth["unlabeled"] = [100 - sp.MIN_GAP - 1]
        try:
            sp.assert_boundary(synth)
            check(True, "assert_boundary accepts gap == MIN_GAP + 1")
        except SystemExit:
            check(False, "assert_boundary accepts gap == MIN_GAP + 1",
                  "it raised: the guard is now too strict")
    finally:
        sp.band_and_slabs = real_bs

    # 9. Injection into the MASKGIT pool too, not just the tokenizer.
    if buffers:
        bad = dict(p)
        bad["maskgit"] = sorted(set(p["maskgit"]) | set(buffers[:1]))
        try:
            sp.assert_boundary(bad)
            check(False, "assert_boundary raises on a bad maskgit pool",
                  "it did NOT raise")
        except SystemExit:
            check(True, "assert_boundary raises on a bad maskgit pool")

    # 10. The cached patch pool re-validates on load, against split.json's slab
    #     labels rather than against val+test membership. The earlier guard
    #     compared to `heldout_slices`, which holds no buffer slices at all, so
    #     it could not detect the contamination it was written for.
    import numpy as np, tempfile
    from src import synth_vqvae as sv
    if buffers:
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "p.npz")
            np.savez_compressed(
                f, patches=np.zeros((2, 8, 8), np.float32),
                src=np.asarray([p["train"][0], buffers[0]], np.int32),
                labeled=np.asarray([True, False]),
                norm_lo=np.float32(0), norm_hi=np.float32(1),
                heldout_slices=np.asarray(sorted(held), np.int32))
            try:
                sv.load_patches(f)
                check(False, "load_patches rejects a buffer-contaminated pool",
                      "it did NOT raise")
            except SystemExit:
                check(True, "load_patches rejects a buffer-contaminated pool")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
