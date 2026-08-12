# The AUPRC discrepancy (condensed spec, resolved)

Date: 2026-08-08. Second of Sky's three tasks. Raised 2026-08-07 while building
the fp_penalty figures.

## Problem

`unet2d_s30000`'s leaderboard row records AUPRC 0.42703. Recomputing it from
`results/probs/unet2d_s30000.npy` with the project's own `metrics.auprc` gave
0.30937. The row's IoU, precision, recall and tp/fp/fn all reproduced from that
same map exactly, so it was not a wrong-map misattribution.

## Approach and what it cost

Hypotheses tested and rejected, in order: a different valid mask (0.31676), the
pre-fix backup map (0.31580), per-slice mean (0.31705) and median (0.30083), the
subsample stride (2/4/8 all agree to 0.0005), the validation map (0.51033), and
val+test pooled (0.38151). None reproduced 0.42703.

The discriminator was ECE. `score()` calls `metrics.auprc(prob, mask, valid)` and
`metrics.expected_calibration_error(prob, mask, valid)` on consecutive lines with
identical arguments, and ECE reproduced to five decimals on all four post-fix
rows while AUPRC was inflated on all four (+0.118, +0.108, +0.055, +0.033). Same
inputs, one metric right and one wrong, means the difference is in what the
metric is sensitive to, not in the data.

**A wrong turn worth recording.** float16 was tested first by adding uniform
jitter inside each value's own float16 spacing and re-scoring. It moved AUPRC by
0.0003, and that was reported as float16 "refuted, cleanly". The test was
invalid: random jitter inside a tie group restores RANDOM order, which is exactly
how average precision already treats ties, so the experiment could only ever
return "no change". It could not have detected the effect it was aimed at. The
lesson is that a null result only counts if the test could have produced a
non-null one.

## Resolution

Regenerating the probabilities from a checkpoint settled it, because it is the
only test that produces the float32 array the metrics were originally computed
on. `resunet18_scratch_s30000`:

| | AUPRC |
|---|---|
| recorded in the leaderboard | 0.44301 |
| regenerated from checkpoint, float32 | **0.44301** (exact) |
| that same array cast to float16 | 0.33455 |
| the saved float16 map on disk | **0.33455** (exact) |

`evaluate.run_and_score` computes every metric on float32 and then saves the map
as float16. **The leaderboard values are correct; the saved maps are lossy.**
The 2026-08-07 KNOWN ISSUES entry had the implication backwards and is corrected
in place, with the reasoning error kept visible.

## Decision

Sky, 2026-08-08, from a three-option MCQ: **keep float16 and carry the rule.**
float32 would roughly double ~13 GB of probability maps to fix a metric that is
already recorded correctly. The rule is documented at the save site in
`src/evaluate.py`, in CHANGELOG.md's KNOWN ISSUES, and in `src/fp_penalty_viz.py`
whose figures are affected:

> Quote AUPRC, and any future ranking metric, from `results/leaderboard_all.json`.
> Never recompute it from `results/probs/`.

## Consequence already incurred

`results/figures/fp_penalty_pr.png` reports float16 AUPRCs, all understated by a
model-dependent amount. Its precision-recall curves and every threshold-derived
number (precision, recall, FP counts) are unaffected, so the conclusion about the
mentor's penalty stands. The AUPRC labels on it should not be quoted.
