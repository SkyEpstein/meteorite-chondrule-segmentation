"""Authoritative per-zone 3D chondrule recount for the FROZEN split.

Three decompositions of the frozen split are on record and at most one is right:

    99 / 19 / 24   CHANGELOG.md, the 2026-07-29 "Prepare for a second meteorite"
                   entry, the "Free statistical power from slab placement"
                   bullet (99 and 19 explicit, 24 inferred)
    63 / 36 / 43   the 2026-08-02 handoff, method not recorded
    81 / 32 / 29   results_v2/split_multi.json, but that is the DIFFERENT v2 split

`SE_REF_N = 19` in `src/pipeline_v2.py` scales every noise floor in the project,
so which of these is right is not cosmetic. Nothing in the repo can settle it:
`split.json` records only `n_chondrule_instances: 142` in total and no per-zone
count, and no artifact anywhere records a per-zone BINARY connected-components
count, which is the leading hypothesis for where 19 came from.

This counts both ways and records exactly what it did, so the answer stops being
archaeology. Both rules are run over the same slices and the same valid mask:

  * SAME-ID: `pipeline_v2.count_chondrules_3d`, which groups voxels by their
    stored instance id. This is the rule the pipeline uses.
  * BINARY CC: `scipy.ndimage.label` on the boolean mask, which merges touching
    chondrules. `pipeline_v2.count_chondrules_3d`'s docstring records that this
    gives 87 against the same-id rule's 142 over the whole volume, a 39%
    shortfall.

Zone membership comes from `split.json`'s `zone_stats[zone]["abs_slices"]`, so
the slab definition is read from the frozen artifact rather than recomputed.
The valid mask comes from the same cache every model trains against, which means
straddler-ignored voxels are excluded here exactly as they are in training.

Run:  python -m src.recount_chondrules
"""
from __future__ import annotations
import json
import os

import numpy as np
from scipy import ndimage as ndi

from . import data, zones as zonemod
from .pipeline_v2 import count_chondrules_3d

OUT = "results/chondrule_recount.json"
ZONES = ("train", "val", "test")


def main(root="."):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)
    _, lab_paths = data.list_pairs(root)
    z = zonemod.load_zones(root, zones=ZONES)

    print("authoritative per-zone chondrule recount, frozen split")
    print(f"  slab source : split.json zone_stats[zone]['abs_slices']")
    print(f"  valid mask  : {zonemod.CACHE} (the same one training uses)")
    print(f"  label dir   : {data.LAB_DIR}")
    print()

    rows, tot_same, tot_bin = {}, 0, 0
    for zone in ZONES:
        sel = [int(i) for i in split["zone_stats"][zone]["abs_slices"]]
        _, _, V = z[zone]
        if V.shape[0] != len(sel):
            raise ValueError(f"{zone}: cache has {V.shape[0]} slices, split.json "
                             f"lists {len(sel)}")
        ids = data.load_stack(lab_paths, sel, kind="raw")     # instance ids
        same = int(count_chondrules_3d(ids, V))

        # Binary CC over the SAME voxels, so the two rules differ only in rule.
        binmask = (ids > 0) & V
        _, nbin = ndi.label(binmask)
        nbin = int(nbin)
        del ids, binmask

        rows[zone] = {"n_slices": len(sel), "same_id_3d": same,
                      "binary_cc_3d": nbin,
                      "shortfall_pct": round(100.0 * (1 - nbin / same), 1) if same else None}
        tot_same += same
        tot_bin += nbin
        print(f"  {zone:>6}: {len(sel):>4} slices   same-id {same:>4}   "
              f"binary-CC {nbin:>4}   shortfall {rows[zone]['shortfall_pct']}%")

    print(f"  {'TOTAL':>6}: {'':>4}          same-id {tot_same:>4}   "
          f"binary-CC {tot_bin:>4}")
    print()

    recorded = int(split.get("n_chondrule_instances", -1))
    print(f"  split.json n_chondrule_instances = {recorded}")
    print(f"  same-id sum {'MATCHES' if tot_same == recorded else 'DISAGREES WITH'} it")

    # The three decompositions on record, so the verdict is unambiguous.
    claims = {"CHANGELOG 99/19/24": (99, 19, 24),
              "handoff 63/36/43": (63, 36, 43),
              "v2 split 81/32/29 (DIFFERENT split)": (81, 32, 29)}
    got_same = tuple(rows[zn]["same_id_3d"] for zn in ZONES)
    got_bin = tuple(rows[zn]["binary_cc_3d"] for zn in ZONES)
    print(f"\n  measured same-id  : {got_same}")
    print(f"  measured binary-CC: {got_bin}")
    for name, c in claims.items():
        marks = []
        if c == got_same:
            marks.append("matches SAME-ID")
        if c == got_bin:
            marks.append("matches BINARY-CC")
        print(f"    {name:<38} {', '.join(marks) if marks else 'no match'}")

    val_same = rows["val"]["same_id_3d"]
    val_bin = rows["val"]["binary_cc_3d"]
    print(f"\n  SE_REF_N is 19. Validation measures {val_same} by the same-id rule "
          f"and {val_bin} by binary CC.")

    out = {
        "regime": ("per-zone 3D chondrule counts on the FROZEN split; zone "
                   "membership from split.json zone_stats[zone]['abs_slices']; "
                   "valid mask from the training cache so straddler-ignored "
                   "voxels are excluded; counted BOTH ways over identical voxels"),
        "same_id_rule": "pipeline_v2.count_chondrules_3d, groups voxels by stored instance id",
        "binary_cc_rule": "scipy.ndimage.label on (ids > 0) & valid, merges touching chondrules",
        "zones": rows,
        "total_same_id": tot_same,
        "total_binary_cc": tot_bin,
        "split_json_n_chondrule_instances": recorded,
        "se_ref_n_in_code": 19,
        "val_same_id": val_same,
        "val_binary_cc": val_bin,
        "claims_on_record": {k: list(v) for k, v in claims.items()},
    }
    os.makedirs(os.path.dirname(os.path.join(root, OUT)), exist_ok=True)
    with open(os.path.join(root, OUT), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
