# Prove two-meteorite ingestion end to end (condensed spec)

Date: 2026-08-08. Sky chose this first of three, ahead of the AUPRC defect and
seeding the scratch ResNet-18.

## Problem

The 2026-08-04 handoff, priority 2: `build_multi_split` has been tested with two
synthetic sources, but `pipeline_v2.run_all`, the thing that consumes it, has
never run with more than the single frozen meteorite. `results_v2/pipeline_v2.json`
does not exist. Ingestion should be proven before a real second scan lands, not
while debugging one.

`run_all` could not be smoke-tested as written. It hardcoded
`stage2_sweep(zones, alive_masks)` with the 30,000-step default and the full
five-architecture roster, and there was no way to shrink any of it.

## Two defects found while making it runnable

**1. [MUST] violation: `stage2_sweep` would have written synthetic and
off-roster rows onto the real leaderboard.** It called
`evaluate.run_and_score(...)` without `leaderboard=False`, while scoring against
a `build_multi_split` split that is NOT the frozen `split.json` roster.
`run_and_score`'s own docstring marks that flag a [MUST] for exactly this case:
"a different split, a data subset, or a reduced budget". `fig_leaderboard` plots
every row in that file as one comparable ranking, so the rows would have been
read as comparable models and misquoted.

Nothing is corrupted today: `results/leaderboard_all.json` contains **zero**
`hier_*` rows, confirming `stage2_sweep` has never completed a run. The defect
was latent and would have fired on the first successful `run_all`, including
this smoke test. Default is now `leaderboard=False`, opt in deliberately.

**2. `out_dir` named only the final JSON.** `run_all(sources, out_dir=X)` wrote
its summary to `X` while `build_multi_split` wrote the split record to the
module-level `results_v2/`, `save_prob` wrote probability maps to
`results_v2/probs/`, and the reload at the confidence stage read them back from
`results_v2/probs/` too. So a custom `out_dir` produced a summary in one place
describing artifacts in another, and any synthetic run would have overwritten
`results_v2/split_multi.json`.

## Approach

Thread the knobs through, defaults preserving previous behaviour except the
leaderboard flag:

- `build_multi_split(..., out_dir=None)`, None keeping the historical path.
- `stage2_sweep(..., out_dir, leaderboard=False, recipe, steps_per_epoch)`.
  `recipe` overrides individual `BEST_RECIPE` keys, which is what allows a patch
  smaller than the image.
- `run_all(..., budgets, archs, leaderboard=False, recipe, steps_per_epoch,
  hier_levels)`, and `out_dir` now used for the split record, the probability
  maps and the reload.

`tests/test_pipeline_v2_multi.py` runs the whole chain on two synthetic
meteorites of DIFFERENT in-plane sizes (96x96 and 80x112, padding to 96x112 so
both are padded on one axis and neither is a no-op), at 1 epoch of 5 steps with
one 8-channel depth-2 U-Net.

## Acceptance

1. `run_all` completes on two sources and returns all eight sections.
2. `pipeline_v2.json` is written under the passed `out_dir` and round-trips.
3. Both meteorites appear in all three zones (the leakage contract).
4. Stage-2 ranking is sorted by val IoU and the winner is selected on it.
5. Probability maps land under `out_dir`, not `results_v2/`.
6. Every reported gain carries a `below_noise_floor` flag.
7. [MUST] `results/leaderboard_all.json`, `results_v2/split_multi.json` and
   `results_v2/pipeline_v2.json` are byte-identical before and after.

## RESULT (2026-08-08): PASSED, all seven criteria

`python -m tests.test_pipeline_v2_multi`, exit 0 in 1214 s.

```
  every pipeline section present   OK
  wrote pipeline_v2.json and it round-trips   OK
  both meteorites present in all three zones   OK  ['SYNTH_A', 'SYNTH_B']
  stage-2 ranked by val IoU, winner selected on it   OK
  probability maps confined to out_dir   OK  (2 files)
  all 9 gains flagged against the floor   OK
  real leaderboard and split records untouched   OK
```

`run_all` now runs end to end on two meteorites. `results_v2/pipeline_v2.json`
still does not exist and should not: the only run so far is synthetic and wrote
to a temp directory.

## READ THIS BEFORE TRUSTING THE TEST: the fake data is trivially separable

The FP-prune sweep reports **IoU 1.0000, precision 1.000, recall 1.000** at every
recall target, and the bake-off's learned candidates beat the free baselines by
+0.32. That looks like leakage and is not. `make_meteorite` draws background
intensities from `randint(20, 40)` and chondrule intensities from
`randint(55, 75)`, which are **disjoint ranges**, so any model with access to raw
intensity separates the classes perfectly. A perfect score is the correct
outcome on this data.

The consequence is the limitation that matters: **this test cannot detect a
leakage bug.** Perfectly separable data scores 1.000 whether the confidence
model was fitted on validation or on test, so the [LEAKAGE][MUST] contract is
exercised but not verified. Verifying it needs synthetic data with OVERLAPPING
class intensities, where a val-fitted model must score below a test-fitted one.
That is worth building and is not built.

Second limitation, practical: the run takes 20 minutes and **787 s of that is a
single bake-off candidate, `mlp`**. `confidence_sweep` has no budget knob, which
is the one part of the chain this change did not parameterize. A 20-minute
plumbing test will not be run routinely.

## What this does NOT establish

Accuracy, in any form. The budget is 5 training steps on fake spheres, so every
IoU it prints is meaningless and must never be quoted. It proves the stages
connect and that a multi-source run stays inside its own output directory. It
also does not test real second-meteorite data: `MeteoriteSource.voxel_um` is
recorded and still not acted on, so differing pixel SIZES remain unhandled,
and this model has no scale invariance.
