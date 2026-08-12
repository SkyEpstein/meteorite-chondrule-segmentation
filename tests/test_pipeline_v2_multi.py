"""Prove `pipeline_v2.run_all` runs end to end on TWO meteorites.

The 2026-08-04 handoff listed this as the next thing to establish:
`build_multi_split` had been tested with two synthetic sources, but the pipeline
that consumes it had never run with more than the single frozen meteorite, and
`results_v2/pipeline_v2.json` did not exist. Ingestion should be proven BEFORE a
real second scan lands, not while debugging one.

This runs the whole chain — split, power report, hierarchy filter, stage-2
sweep, confidence sweep, noise-floor flags — on two tiny synthetic meteorites of
DIFFERENT in-plane sizes, so the padding path is exercised too. It is a
plumbing test at a deliberately absurd budget. It asserts that the stages
connect and that nothing escapes into the real records. It asserts NOTHING about
accuracy, and the IoUs it produces are meaningless by construction.

[MUST] The real artifacts must be untouched. A synthetic run that writes to
`results/leaderboard_all.json` or `results_v2/split_multi.json` would corrupt
the project's records with numbers from fake data, so both are snapshotted and
compared byte for byte at the end.

Run:  python -m tests.test_pipeline_v2_multi
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline_v2 as pv                                  # noqa: E402
from tests.test_multi_meteorite import make_meteorite              # noqa: E402

# Absurdly small on every axis. patch must be <= the smaller image dimension,
# and the superpixel levels must be far below the pixel count or SLIC returns
# more segments than there are pixels to fill them.
TINY_RECIPE = dict(patch=32, batch=4, val_stride=1)
TINY_LEVELS = (20, 60)
STEPS_PER_EPOCH = 5
BUDGET = 5                                    # -> 1 epoch of 5 steps


def _sha(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    guarded = ["results/leaderboard_all.json", "results_v2/split_multi.json",
               "results_v2/pipeline_v2.json"]
    before = {p: _sha(p) for p in guarded}

    tmp = tempfile.mkdtemp(prefix="pv2_multi_")
    out_dir = os.path.join(tmp, "out")
    try:
        # Different in-plane sizes on purpose: build_multi_split crashed on this
        # before 2026-08-04 and now pads to the common maximum, marking the pad
        # invalid. 96x96 and 80x112 pad to 96x112, so BOTH sources get padded on
        # one axis and neither is a no-op.
        sources = [
            make_meteorite(tmp, "SYNTH_A", n=100, H=96, W=96, n_chond=20,
                           seed=0),
            make_meteorite(tmp, "SYNTH_B", n=100, H=80, W=112, n_chond=20,
                           seed=1),
        ]
        print(f"\n=== run_all on {len(sources)} synthetic meteorites ===",
              flush=True)
        out = pv.run_all(
            sources, out_dir=out_dir,
            budgets=(BUDGET,), steps_per_epoch=STEPS_PER_EPOCH,
            archs={"tiny": lambda in_ch: pv.UNet2D(in_ch=in_ch, base=8,
                                                   depth=2)},
            recipe=TINY_RECIPE, hier_levels=TINY_LEVELS, leaderboard=False)

        # --- the chain produced every section it promises ---
        for k in ("split", "power", "hierarchy", "stage2_ranking", "winner",
                  "confidence_sweep", "gains", "runtime_sec"):
            assert k in out, f"run_all output is missing {k!r}"
        print("  every pipeline section present   OK")

        path = os.path.join(out_dir, "pipeline_v2.json")
        assert os.path.exists(path), f"{path} was not written"
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk["winner"]["model"] == out["winner"]["model"]
        print(f"  wrote {os.path.basename(path)} and it round-trips   OK")

        # --- two meteorites really went through, with the leakage contract ---
        meta = out["split"]
        names = [e["name"] for e in meta["meteorites"]]
        assert sorted(names) == ["SYNTH_A", "SYNTH_B"], names
        for e in meta["meteorites"]:
            for z in pv.ZONES:
                assert e["zone"][z]["n_slices"] > 0, (e["name"], z)
        print(f"  both meteorites present in all three zones   OK  {names}")

        # --- the winner was chosen on validation, never on test ---
        assert out["winner"]["selected_on"] == "validation IoU"
        ranked = out["stage2_ranking"]
        vals = [r["val_iou"] for r in ranked]
        assert vals == sorted(vals, reverse=True), vals
        print("  stage-2 ranked by val IoU, winner selected on it   OK")

        # --- probability maps landed under out_dir, not the module default ---
        probs = os.path.join(out_dir, "probs")
        assert os.path.isdir(probs) and os.listdir(probs), probs
        assert not os.path.exists(os.path.join("results_v2", "probs",
                                               f"{out['winner']['model']}.npy"))
        print(f"  probability maps confined to out_dir   OK  "
              f"({len(os.listdir(probs))} files)")

        # --- every gain carries an explicit noise-floor flag ---
        for k, g in out["gains"].items():
            assert "below_noise_floor" in g, k
        print(f"  all {len(out['gains'])} gains flagged against the floor   OK")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- [MUST] nothing escaped into the real records ---
    for p in guarded:
        assert _sha(p) == before[p], (
            f"SYNTHETIC RUN MODIFIED {p}. That is the failure this test exists "
            f"to catch: fake-data numbers in a real record.")
    print("  real leaderboard and split records untouched   OK")

    print("\nALL CHECKS PASSED")
    print("NOTE: this is a plumbing test. Its IoUs are meaningless by "
          "construction and must never be quoted.")


if __name__ == "__main__":
    main()
