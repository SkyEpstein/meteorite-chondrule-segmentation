"""Acceptance test for the multi-meteorite ingestion path.

`build_multi_split` had never run against more than one meteorite before
2026-08-04, so the claim that adding a meteorite is a config change was untested.
It was half wrong: two sources of the SAME in-plane size ingested fine, two of
DIFFERENT sizes crashed with a bare numpy ValueError from the `np.concatenate`
over per-meteorite zone arrays in `build_multi_split`, which is the normal case
for two real scans.

This builds small synthetic scans and checks the whole contract:

  * one meteorite still works;
  * two of the same size concatenate with correct per-slice provenance;
  * two of DIFFERENT sizes pad to a common size instead of crashing;
  * the padding is marked INVALID everywhere, which is what the fix rests on.
    If padded pixels were left valid, every metric would score empty space as
    true negatives and every IoU would be wrong with nothing to catch it;
  * the leakage contract holds: no meteorite appears in only some zones.

Volumes are tiny, so this is safe to run beside a training job. Writes nothing
into the repo: OUT_DIR and SPLIT_JSON are redirected to a temp directory.

Run:  python -m tests.test_multi_meteorite     (from the repo root)
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline_v2 as pv                                  # noqa: E402


def make_meteorite(root, name, n=100, H=64, W=64, n_chond=14,
                   band=(10, 90), seed=0):
    """Write a synthetic raw/label tif stack pair and return its source.

    Chondrules are spheres so they span z and the straddler logic has real work.
    Labels carry INSTANCE IDS, not a binary mask: `build_multi_split` reads them
    with kind="raw" to count 3D instances by the same-id rule. Outside `band` the
    labels are empty, mimicking a partially annotated scan so `annotated_band`
    has a band to find.
    """
    rng = np.random.default_rng(seed)
    raw_dir = os.path.join(root, f"{name}_raw")
    lab_dir = os.path.join(root, f"{name}_labeled")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    raw = rng.integers(20, 40, size=(n, H, W)).astype(np.uint8)
    lab = np.zeros((n, H, W), np.uint16)
    zz, yy, xx = np.ogrid[:n, :H, :W]
    b0, b1 = band
    for c in range(1, n_chond + 1):
        r = int(rng.integers(4, 8))
        cz = int(rng.integers(b0 + r, b1 - r))
        cy = int(rng.integers(r, H - r))
        cx = int(rng.integers(r, W - r))
        ball = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
        lab[ball] = c
        raw[ball] = rng.integers(55, 75)          # chondrules read brighter
    lab[:b0] = 0
    lab[b1:] = 0

    for i in range(n):
        tifffile.imwrite(os.path.join(raw_dir, f"{i:04d}.tif"), raw[i])
        tifffile.imwrite(os.path.join(lab_dir, f"{i:04d}.tif"), lab[i])
    return pv.MeteoriteSource(name, raw_dir, lab_dir, every_n=1)


def check(label, sources, tmp):
    print(f"\n--- {label} ---")
    pv.OUT_DIR = tmp
    pv.SPLIT_JSON = os.path.join(tmp, "split_multi.json")
    zones = pv.build_multi_split(sources)

    for z in pv.ZONES:
        prov = zones[z]["provenance"]
        by = {p: int((prov == p).sum()) for p in sorted(set(prov))}
        assert sum(by.values()) == zones[z]["raw"].shape[0]
        print(f"  {z:>6}: raw{zones[z]['raw'].shape} provenance={by}")

    # [LEAKAGE][MUST] no meteorite may be absent from any zone.
    for src in sources:
        zs = {z for z in pv.ZONES if src.name in set(zones[z]["provenance"])}
        assert zs == set(pv.ZONES), f"{src.name} missing from {set(pv.ZONES) - zs}"
    print("  leakage contract OK: every meteorite appears in all three zones")

    meta = zones["meta"]
    if meta.get("padding_applied"):
        Hm, Wm = meta["padded_to"]
        native = {e["name"]: tuple(e["native_shape"]) for e in meta["meteorites"]}
        print(f"  native shapes {native}, padded to {Hm}x{Wm}")
        for z in pv.ZONES:
            prov = zones[z]["provenance"]
            V, M = zones[z]["valid"], zones[z]["mask"]
            for nm, (h, w) in native.items():
                sel = np.where(prov == nm)[0]
                if not len(sel) or (h, w) == (Hm, Wm):
                    continue
                assert not V[sel][:, h:, :].any(), f"{nm}/{z}: padded ROWS valid"
                assert not V[sel][:, :, w:].any(), f"{nm}/{z}: padded COLS valid"
                assert not M[sel][:, h:, :].any(), f"{nm}/{z}: padded ROWS positive"
                assert not M[sel][:, :, w:].any(), f"{nm}/{z}: padded COLS positive"
                assert V[sel][:, :h, :w].any(), f"{nm}/{z}: native region blanked"
        print("  padding marked INVALID everywhere, native region intact")
        for z in pv.ZONES:
            s = meta["zone_stats"][z]
            assert s["pos_frac_valid"] >= s["pos_frac"], \
                f"{z}: pos_frac_valid should exceed the padding-diluted pos_frac"
            print(f"    {z:>6}: pos_frac={s['pos_frac']:.5f} "
                  f"pos_frac_valid={s['pos_frac_valid']:.5f}")
    return zones


def main():
    tmp = tempfile.mkdtemp(prefix="synthmet_")
    print("workspace:", tmp)
    try:
        a = make_meteorite(tmp, "SYNTH_A", H=64, W=64, seed=1)
        b = make_meteorite(tmp, "SYNTH_B", H=64, W=64, seed=2)
        c = make_meteorite(tmp, "SYNTH_C", H=48, W=56, seed=3)

        one = check("ONE meteorite", [a], tmp)
        assert one["meta"]["n_meteorites"] == 1
        assert not one["meta"]["padding_applied"]

        same = check("TWO meteorites, SAME in-plane size", [a, b], tmp)
        assert same["meta"]["n_meteorites"] == 2
        assert not same["meta"]["padding_applied"]

        diff = check("TWO meteorites, DIFFERENT in-plane size", [a, c], tmp)
        assert diff["meta"]["padding_applied"]
        assert diff["meta"]["padded_to"] == [64, 64]
        for z in pv.ZONES:
            assert diff[z]["raw"].shape[1:] == (64, 64)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
