# Replicate the reference model (condensed spec)

Date: 2026-08-08. Chosen by Sky after the seed-SD result landed.

## Problem

`unet2d_s30000`, test IoU **0.29409**, is the number every comparison in this
repo is quoted against: the pretraining probe, the five cascade arms, geometry
augmentation, the complete-model stages, the README status block. It has never
been run twice.

The same day, the full-budget seed SD was measured at **0.04517** on an identical
configuration (`resunet18_scratch_s30000`: 0.33569 / 0.26098 / 0.34233). So the
reference is one draw from a distribution roughly that wide. If 0.29409 sits high
or low within its own spread, **every gap quoted against it shifts by the same
amount, in the same direction, simultaneously.**

This is not a method question and does not test anything. It is what makes the
comparisons already in the repo interpretable.

## Approach

`python -m src.reference_replicates --seed=1,2`, about 3 h (the original run took
5195 s).

Recipe reconstructed from two sources, because no single one records it fully:
`results/phase1/p1_ctrl_b32.json` gives loss=dice, lr=3e-4, base=32, wd=0,
aug=false, pos_frac=0.5; `pipeline_v2.BEST_RECIPE` supplies depth=4, patch=128,
batch=16, val_stride=6 and agrees with p1_ctrl_b32 on every field they share. The
budget is 60 x 500 = 30,000 steps from the `_s30000` suffix.

## Acceptance

1. `results/reference_replicates.json` holds seeds 1 and 2 with a real SD across
   all three.
2. Seed 0's map, its leaderboard row and the value 0.29409 are untouched. The
   runner refuses `--seed=0` outright.
3. The report states how far the reference sits from the mean of its own
   configuration, in SDs, and by how much every gap quoted against 0.29409 would
   move if the mean were used instead.

## What this CANNOT distinguish

A faithful reconstruction is not a byte-identical re-run. Four fields (depth,
patch, batch, val_stride) were recovered from a second source rather than from
the run's own record, and seed 0 predates checkpointing so there is no artifact
to diff against.

**If a replicate lands far from 0.29409, two explanations are live and these runs
cannot separate them:** the reference was an unusual draw, or the recipe was
reconstructed wrong. Only a genuine tie-break (a third source recording the
original batch/patch, or seed 0's own checkpoint, which does not exist) would
settle it.

That the reference recipe is not fully recorded in one place is itself a defect
worth fixing, independent of this run.

## Note on the leaderboard flag

`leaderboard=True` here, unlike `run_cascade_arms`. These replicates ARE the
frozen split, the frozen budget and the frozen recipe, which is exactly the
roster the leaderboard's regime string describes, so they belong in it. The
[MUST] in `run_and_score` targets runs that are NOT roster-comparable; these are.
