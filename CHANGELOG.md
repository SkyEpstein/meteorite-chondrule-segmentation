# Changelog

All notable decisions and milestones for the Meteorite Chondrule Segmentation
project. Honest numbers only; evaluation regime stated with every metric.

## [Unreleased]

### ⚠️ KNOWN ISSUES IN THIS COMMIT (found by pre-commit review — read first)

- **🔴 ALL CNN-FAMILY TEST SCORES WERE MEASURED WITH TILED INFERENCE, AND TILED
  INFERENCE IS STILL THE DEFAULT — but it has TWO defects, one fixed and one
  accepted (2026-07-29).** A whole-slice replacement was tried and **reverted**;
  see below. In sequence:
  1. The tiler **accumulated its halo** instead of cropping it, producing
     axis-aligned seams in every CNN/transformer/modern-conv probability map —
     including `unet2d_s30000` (0.3168). Cropping the halo fixed *that*: the
     boundary-localised artifact is gone (1/8 largest column jumps on a
     boundary afterwards, boundary jump 1.1× median).
  2. **Cropping was not sufficient.** These nets normalise with **GroupNorm**,
     which takes statistics over whatever spatial extent it is given, so under
     tiling every window was normalised by its own statistics and its
     predictions shifted as a block. Measured with the *fixed* tiler: moving the
     tile grid 128 px changed predictions by mean 0.024 / max 0.286, and tiled
     vs a single whole-image pass differed by mean 0.022 / max 0.331. This is
     **invisible to a constant-image test** (GroupNorm is degenerate there),
     which is why the original verification passed.

  Tiling was replaced by a whole-slice pass and then **REVERTED within the
  hour**, because whole-slice inference *collapsed* trained-model output — val
  best-epoch IoU 0.4487 → 0.2632 (epochs 8-12 at 0.0000-0.0075), on two SEPARATE
  runs at the same config and seed — not one model scored two ways. These nets
  train on 128×128
  patches at `pos_frac=0.5`; a whole 962×1450 slice at ~5% foreground puts
  GroupNorm far out of distribution. **Tiled inference (halo cropped) is the
  default and the accepted state**, with its residual grid dependence documented
  rather than fixed. Fixing it properly means changing the normalisation or the
  training window, not the stitching.

- **⚠️ The val-selected threshold sits ON the grid floor — but it is NOT clamped
  (measured).** `CNNRunner.fit` selects on `np.arange(0.002, 0.05, 0.004) ∪
  np.arange(0.05, 0.96, 0.05)`, floor exactly **0.002**, and `run_and_score`
  scores with that `best_thr`. The retrained `unet2d_s30000` selected 0.002 — the
  boundary — which looks like the truncated-grid defect logged three times before.
  **It is not, here.** Re-selecting on the same saved val map with the grid
  extended to **1e-7** returns the *same* 0.002 and the *same* test IoU 0.29409.
  The val-IoU curve is flat over 5e-4…2e-2 (0.44426 / 0.44512 / 0.44475) with a
  true maximum at 0.002, collapsing only below 1e-4. So no IoU is being lost to
  the grid for this model. **Still worth fixing:** `unet2d_s15000` and
  `matrix_specialist` also sit on the floor and have not been checked, and both
  `:.2f` prints render 0.001/0.002 as `thr=0.00`, so a real clamp would be
  invisible. Extend the grid and print more digits so the distinction is
  observable rather than requiring a bespoke probe.

  **Every CNN-family leaderboard IoU in this repo was produced under tiled
  inference.** [UPDATED 2026-08-04] The counts and the provenance gap are now
  settled. Of 44 rows: **4 are post-halo-crop** (`unet2d_s30000` 0.29409,
  `unet2d_s30000_auggeom` 0.24404, `resunet18_pretrained_s30000` 0.30269,
  `resunet18_scratch_s30000` 0.33569), **33 are pre-fix** (including the
  E_cascade row, whose `predict_proba` delegates straight to `CNNRunner`), and
  **7 never touched the tiler** (the D_gnn rows paint superpixel probabilities
  directly). Every row now carries a
  `tiler_regime` field of "pre", "post" or "n/a", and `run_and_score` stamps
  each new row with the git commit and a UTC timestamp, so this never has to be
  reconstructed again. Full detail under 2026-07-29 below, backfill evidence
  under 2026-08-04.

- **`+10% from z-context` is UNBACKED.** `results/probs/unet25d_k5.npy` is
  actually the 30,000-step run's output (bare-name collision, fixed in code but
  after these files were written). Re-scoring it gives **0.2154**, not the
  leaderboard's 0.2547 — the surviving artifact points the OPPOSITE way. Treat
  every "2.5D beats 2D at equal budget" statement as unverified.
- **The 3k budget anchor's named file is wrong.** [CORRECTED 2026-08-04, twice
  over.] The original wording said `probs/unet2d.npy` is byte-identical to
  `probs/unet2d_s30000.npy`. That was true when written and is FALSE now: the
  post-fix retrain rewrote `probs/unet2d_s30000.npy` on 2026-07-29 19:54, so the
  two differ today (sha256 b58d50d9 vs e5fd4ed9). The surviving fact is sharper.
  `probs/unet2d.npy` is byte-identical to
  **`results/prefix_tiler_backup/probs/unet2d_s30000.npy`**, confirmed by hash,
  so the file holds the 30,000-step run's map written under the bare family
  name. `probs/unet25d_k5.npy` has the same defect and re-scores to 0.2154
  against its row's 0.2547. The 0.2314 anchor therefore has no correctly named
  artifact, but `probs/rt_dice_ctrl.npy` is a byte-level stand-in for the same
  configuration under the same pre-fix tiler.

  **"Independently reproduced" overclaimed.** `rt_dice_ctrl` (0.23142 @ thr
  0.25) matches `unet2d` on every metric including AUPRC to full precision.
  That is not independent reproduction: the two are the SAME configuration down
  to `seed=0`, so this is a determinism check, not a seed replicate, and it
  supplies no variance estimate. Two separate executions did occur (`sec` 871.3
  vs 947.1), so the rows are honest.

  `pick_winner` globbing both names, so one model enters the ranking twice,
  SURVIVES. `best_model_by_val` (`src/run_overnight.py:22-43`) has the same
  defect and is still called live at `:185, :218, :247`, so a future run can
  still select the mislabeled entry. The "best of 40 architectures" framing
  also over-counts by one, because that duplicated configuration is two rows.

- **[NEW 2026-08-04] `run_and_score` has no invariant tying a row to its saved
  map.** `save_prob` is a caller-supplied string that is never checked against
  `runner.name` (`src/evaluate.py`), and `append_leaderboard` dedupes on `model`
  while recording nothing about which file the row's map went to. That absence
  is the structural cause of the misattribution above, and it is still
  unguarded. The provenance stamp added today records the commit and the time
  but not the map path.
- **[NEW 2026-08-07] 🔴 `unet2d_s30000`'s recorded AUPRC of 0.42703 cannot be
  reproduced from any map on disk, under any definition tried.** Found while
  building the fp_penalty figures. Measured on the live post-fix map
  `results/probs/unet2d_s30000.npy` with the project's own `metrics.auprc`:

  | quantity | value |
  |---|---|
  | leaderboard row, `auprc` | **0.42703** |
  | `metrics.auprc` on the live map (valid-masked, pooled) | **0.30937** |
  | same, no valid mask | 0.31676 |
  | per-slice mean (valid-masked, n=80) | 0.31705 |
  | per-slice median | 0.30083 |
  | pre-fix backup map, valid-masked pooled | 0.31580 |

  **The row's other metrics DO match that map**: IoU at the frozen threshold
  0.002 reproduces 0.29409 exactly, and tp/fp/fn agree to within 6 pixels out of
  5.8M. So this is not a wrong-map misattribution like the `unet25d_k5` case; it
  is one field of an otherwise-correct row disagreeing with its own map by
  +0.118. Subsampling is excluded: strides of 2, 4 and 8 all give 0.3089.

  **AUPRC 0.427 is quoted as part of the headline metric set in `HANDOFF.md`,
  `CHANGELOG.md` and `README.md`.** Nothing has been changed on the strength of
  this, and the number is NOT retracted here, because the measurement says only
  that the row and the artifact disagree, not which one is wrong. Settle it
  before quoting 0.427 again. Note also that the no-valid-mask value 0.31676 is
  digit-for-digit the `baseline_test` in `results/final_pipeline.json`, which is
  an IoU, not an AUPRC; that is probably coincidence and is worth ten minutes to
  confirm it is.

- **`results/probs/` has no file for `unet2d_s15000`, `unet25d_k5_s15000`,
  `unet25d_k5_s30000`** — the 15k point (0.3089) quoted in the docs cannot be
  independently checked.
- **`final_pipeline.json` predates the threshold-grid fix in `metrics.py`.** Under
  the committed code `unet2d_s30000` val-selects thr 0.0076 and scores
  **0.31601** (not 0.31676); every downstream number in that file — the −6.2%
  z-refine, 0.3131 prune, 0.3214 stack, and the bake-off table — was computed at
  thr 0.05. Re-run `python -m src.run_final` to regenerate.
- **`0.3214` was selected on TEST.** The prune+z stack is never scored on
  validation, so it is chosen among four test-scored options. It is a reported
  configuration, not a val-selected result.
- **The confidence bake-off ran at recall 0.601, not the stated 0.95** —
  `recall_threshold` silently falls back to the grid floor when a target is
  unreachable and returns no flag.
- **`results/ablation.json` and `results/unlabeled.json` used `hrnet_lite`,
  selected from the TEST-sorted leaderboard** — the exact leakage
  `best_model_by_val` was added to prevent. Both predate that fix.
- **`results/ablation.json` has one arm only** — the promised
  drop-straddlers-entirely comparison was never computed.
- **v4 unlabeled inference was never produced** — the sweep saved probabilities
  but not model weights, so there is nothing to run on the 130 unlabeled slices.
  *(Fixed going forward by `CNNRunner.save_checkpoint`, 2026-07-29 — but the
  already-swept models are gone and would need retraining.)*
- **`results/data_scaling.json` is confounded** — contiguous subsets and one
  seed per point; it is non-monotonic and worst at 100% data. Do not quote it.
  **[UPDATED 2026-08-04] Its replacement is partial, not absent.**
  `src/data_scaling.py` has still never completed a full 12-run sweep, but the
  curve now stands at **3 of 12 runs**. Fraction 0.25 is complete at 3 seeds
  (test IoU 0.2823 / 0.2664 / 0.2696, `n_seeds: 3`, `test_iou_sd` about
  0.00841), and `results/data_scaling_v2.json` IS committed in this change.
  Fractions 0.5, 0.75 and 1.0 stand at 0 of 3 each, so 9 runs remain, roughly
  5 hours. The earlier wording here said 1 of 12, `n_seeds: 1`,
  `test_iou_sd: NaN`, and deliberately uncommitted; all three are now wrong. One
  completed fraction gives no slope, so the question "does more annotated data
  help?" is still unanswered.
- **`tests/` is no longer empty, and still covers almost nothing.** [UPDATED
  2026-08-04] The earlier wording here said `tests/` is empty; that is now wrong.
  Two files ship: `tests/test_aux_channels.py` (the N-channel aux input path,
  including bit-identical equivalence for a single-channel aux) and
  `tests/test_multi_meteorite.py` (multi-meteorite ingestion: padding to a common
  in-plane size, padding marked invalid, per-slice provenance). Neither covers
  the noise floor or any model's accuracy. Both run on synthetic data from
  `__main__`, so the ~5,000 new lines still have nothing that runs
  automatically.
- **Latent bug:** `run_families.py` / `cascade.py` do `bool(rec["aug"])`, so an
  `aug="geom"` recipe would silently train with intensity jitter. Not triggered
  today (winning recipe has `aug: false`).



### 2026-08-07: The complete model end to end, five stage-3 arms, and two more non-results rendered as results

*Spec (condensed). **Problem:** three days of work (2026-08-05 to 2026-08-07) sat
uncommitted and undocumented, and the headline number of its main experiment was
wrong. **Approach:** give the `full` arm the seeds it was missing, fix the
printout that manufactured the wrong number, and record the lot. **Acceptance:**
`full` at n=3 with a real seed SD, verdicts formed within a budget only, no
verdict claimed at n=1. All met. Full note in
`specs/2026-08-07-full-arm-seeds/spec.md`. The 08-05 and 08-06 items below are
documented retrospectively from their artifacts, not from a session record.*

**[RETRACTED] `full` is not val 0.5203 +/- 0.0158 / test 0.2574 +/- 0.0055.**
That row, printed in `logs/cascade_arms.log` under a header reading "3 seeds
each, 7,200 steps", is the mean of a 7,200-step run and a 30,000-step run pooled
across two incomparable budgets, n=2, with the between-budget spread reported as
a seed SD. Its companion verdict `full +0.1179, 2-sigma bar 0.0448, RESOLVED` is
retracted with it. Do not quote either. `save()` had already been fixed to group
by (arm, budget) and `results/cascade_arms.json` was correct; the printed summary
was never regenerated, which is how a corrected file and a wrong log came to
coexist.

**This one did NOT propagate**, which is worth recording because the opposite has
happened repeatedly here. Grepped repo-wide for 0.5203, 0.2574 and 0.1179: they
appear nowhere outside `logs/cascade_arms.log` and this retraction. The
2026-08-06 handoff independently reported the arm as `full (n=1) 0.5092`, the
honest number, so the wrong value never reached a document anyone reads.

**The corrected `full` arm, three seeds at 7,200 steps** (val 0.5092 / 0.4778 /
0.5089, test 0.2535 / 0.2399 / 0.2493):

| arm, 7,200 steps, n=3 | val IoU | test IoU |
|---|---|---|
| control (raw only) | 0.4024 +/- 0.0159 | 0.2493 +/- 0.0289 |
| cleaned (+ denoise/CLAHE/top-hat) | 0.4560 +/- 0.0098 | 0.2534 +/- 0.0099 |
| cascade (+ cross-fit classical probability) | 0.4757 +/- 0.0118 | 0.2467 +/- 0.0040 |
| **full (both)** | **0.4986 +/- 0.0181** | **0.2476 +/- 0.0070** |
| fp_penalty | 0.3874 +/- 0.0169 | 0.2033 +/- 0.0178 |

Against the control, same 2-sigma rule on both axes: cleaned +0.0536 (bar 0.0374)
RESOLVED on val and +0.0041 (bar 0.0612) inside the noise on test; cascade
+0.0733 (0.0397) RESOLVED and -0.0026 (0.0584) inside the noise; **full +0.0962
(0.0482) RESOLVED and -0.0017 (0.0595) inside the noise**; fp_penalty -0.0150
(0.0464) and -0.0460 (0.0679), inside the noise on both.

The val conclusion SURVIVES the correction at twice its bar, and the ordering
cleaned < cascade < full is intact, so the two classical products do stack rather
than one carrying the effect. No arm is separable from the control on test, and
`full`'s test gap changed SIGN when the missing seeds arrived, +0.0042 at n=1 to
-0.0017 at n=3. The 30,000-step `full` run (val 0.5315, test 0.2613) stays n=1
and is NOT verified as hyperparameter-matched to `unet2d_s30000` (test 0.29409,
recipe `phase1:p1_ctrl_b32`), so those two must not be differenced.

**The mentor's proposal was tested and did not help.** `fp_penalty` is Tversky
with alpha 0.7 on false positives and beta 0.3 on false negatives, the exact
mirror of the 0.3/0.7 recall direction already measured to hurt. Tversky at
alpha = beta = 0.5 IS Dice, so `control` is the matched symmetric baseline and no
separate control was needed. It is the worst of the five arms on both axes and
the only one whose val gap points down, but neither gap clears its bar, so the
honest verdict is **no benefit detected with the sign pointing the wrong way**,
not "proven harmful". Scope: one alpha/beta setting, 7,200 steps not 30,000, and
raw-only inputs, because `build_lookup` gives `fp_penalty` no aux channels. It
has never been composed with the `full` arm.

**Four more val/test pairs, and they support the val-versus-test bar rather than
the sign-flip tally.** `results/val_vs_test.json` puts the smallest val gap that
predicts anything at 0.137 to 0.172 IoU. All four arms here have val gaps below
that bar and all four produced test gaps inside the noise, including the one
comfortably resolved on validation. The 2026-08-04 entry noted only TWO effects
had both numbers recorded; these are four more, seed-replicated, landing the same
way.

**[2026-08-06] Two printout defects fixed in `report()`, both of the same
class: a number that is not a result being rendered as one.**
1. The control was selected by arm NAME alone, so every row was differenced
   against whichever control came first regardless of budget. The 30,000-step
   `full` row printed as "+0.1291 against the control" where the control was the
   7,200-step one, the cross-budget comparison this module's own `regime` string
   forbids. Gaps are now formed within a budget only, and a budget with no
   control of its own says so instead of borrowing one.
2. A single-seed row has no SD, so its bar was `nan`; `abs(gap) > nan` is False,
   so the verdict fell through to **"inside the noise"** — a claim about noise
   where n=1 supports no verdict. It now prints "no verdict (n=1)". Verified by
   exercising the path directly: same inputs, old code "inside the noise", new
   code "no verdict (n=1)".

`--report` reprints from `results/cascade_arms.json` without the multi-GB
`evaluate.load`, and rebuilding the summary from `runs` re-applies the current
(arm, budget) grouping, so a file written under the old pooling is corrected by
being read. Confirmed not to mutate the JSON.

**[2026-08-05] The complete model measured end to end on the post-fix base**
(`src/pipeline_viz.py`, `results/pipeline_stages.json`). Every stage fitted on
VALIDATION and applied to TEST once. This supersedes `results/final_pipeline.json`,
whose stages were all measured against the pre-tiler-fix base (0.31676) that no
longer exists.

| stage | val IoU | test IoU |
|---|---|---|
| 0. base U-Net `unet2d_s30000`, thr 0.002 | 0.44512 | 0.29409 |
| 1. + z-refine (mean, k=7, thr 0.3) | 0.45818 | 0.29215 |
| 2. + over-predict to recall 0.9576, then prune | 0.43807 | 0.30279 |

**Constitution section 9 remains unproven on the model the project ships.**
Over-predict-then-prune LOSES 0.00705 on validation, which is the axis that
decides, and gains 0.0087 on test. Neither clears the noise floor, and they
disagree in sign. The over-predict step alone costs almost everything: opened to
recall 0.9576 the map scores val 0.07107 / test 0.06026, so the prune is
recovering 0.367 val IoU just to end up below where it started.

**[2026-08-06] The confidence stack: combining the three signals does not beat
the best single one** (`src/confidence_stack.py`, `results/confidence_stack.json`).
Earlier measurements established only that no SINGLE signal beats the free margin
baseline; whether the three COMBINED help had never been tested. Stacked by a
logistic model cross-fit inside val on contiguous buffered folds, one prune:

| arm | val IoU | test IoU |
|---|---|---|
| regressor alone | **0.45874** | 0.29316 |
| stacked (all three) | 0.44731 | 0.28661 |
| margin (free baseline) | 0.43807 | **0.30279** |
| error U-Net alone | 0.10561 | 0.06969 |
| z-agreement alone | 0.08018 | 0.07134 |

The stack loses to its own best component on val by 0.01143, so the answer is no.
The val and test orderings are exactly reversed: val selects the regressor, which
then scores 0.0096 BELOW the free margin baseline on test. Stacker coefficients
put almost all weight on the regressor (7.99 against 1.10 and 0.75), which is
consistent with eight of its eleven features being functions of `prob` itself.

**Figures** (`results/figures/`): `full_pipeline.png` and `pipeline_stages.png`
on the raw-only base, `full_pipeline_full_arm.png`, `pipeline_stages_full_arm.png`
and `confidence_signals_full_arm.png` on the composed arm, plus `best_model.png`
and `poc_overlay_spread.png`. KNOWN DEFECT: `full_pipeline.png`'s caption reads
"stage 3's classical-channel cascade arm has never been run, so the CNN sees raw
only". True when written, false now, and not regenerated.

**TRAP: `pytest tests/` collects nothing and reports success.** Both test modules
are script-style with a `__main__` entry, so pytest returns "no tests ran" with
exit 0, which reads as a pass. Run them as modules: `python -m tests.test_aux_channels`
and `python -m tests.test_multi_meteorite`. Both pass as of this commit. This is
also why the harness test gate has no `last_test` field to write.

### 2026-08-04: The pretraining probe, and the noise floor moves the wrong way

*Spec (condensed). **Problem:** the pretrained arm of the data-starvation probe
was mid-run, the validation chondrule count was flagged as possibly wrong, and
the docs still let superseded rows be quoted alone. **Approach:** settle the
count question by adversarial verification before touching any constant, run the
control arm, and mark provenance where a number can be extracted without its
caveat. **Acceptance:** no constant changed on a single unverified count, both
probe arms scored on TEST with the threshold frozen on val, and every correction
recorded rather than left in a subagent's report.*

**Pretrained arm (ResNet-18 ImageNet encoder + U-Net decoder, 30,000 steps).**
Scored once on TEST with the threshold frozen on val.

| metric | value |
|---|---|
| test IoU | 0.30269 |
| Dice | 0.4647 |
| precision | 0.3394 |
| recall | 0.7368 |
| AUPRC | 0.4671 |
| ECE | 0.0736 |
| threshold (val-frozen) | 0.018 |
| per-slice median IoU | 0.2848 |
| val IoU | 0.5282 |
| wall clock | 8150 s |

This is +0.0086 against plain `unet2d_s30000` (0.29409), and that comparison is
NOT interpretable: it changes architecture and initialisation at once. The
scratch control arm is what makes the gap mean anything, and it was still running
when this entry was written. The error profile did shift, with recall up from
0.584 to 0.737 and precision down from 0.372 to 0.339, and AUPRC up 0.040.

**[CORRECTION] Raising SE_REF_N would LOOSEN every noise floor, not tighten it.**
The 2026-08-02 handoff proposed that validation holds 36 chondrules rather than
19, and that the minimum detectable effect would therefore tighten from 0.0142 to
roughly 0.0103. Two independent verifications refute the second half.

`SE_REF_IOU` and `SE_REF_N` are a MATCHED ANCHOR PAIR from one bootstrap on one
zone, not an independent SE and an independent count. `MDE(n) = 2 * SE_REF_IOU *
sqrt(SE_REF_N / n)`. If that zone really holds 36 chondrules, the measured SE at
n=36 IS 0.0071, the floor at that zone stays 0.0142, and every projection to
larger n loosens by sqrt(36/19) = 1.376. At n=150 the projected MDE goes from
0.0051 to 0.0070. A recount adds no data, so it cannot buy power. The 0.0103
figure is what `power_report(36)` prints today with the anchor left at 19, which
is the inconsistent state. **HANDOFF.md's 0.0103 is WITHDRAWN.**

Corroborating detail found while checking this: the bootstrap CI of
[-0.013, +0.015] recorded in this file's 2026-07-23 "Full model sweep, budget
curve, confidence pipeline" entry, under "Model selection is unreliable here",
is centred on the +0.000976 gap between the top two models in
`results/final_pipeline.json`, with half-width 0.0140. At
SE = 0.0071 the bounds reproduce exactly; at 0.0075 neither bound reproduces.
[HONESTY] That is strong circumstantial evidence that 0.0071 and the CI are the
same measurement, not proof. No bootstrap code exists anywhere in this repo or
its history, so the derivation cannot be re-run, and the claim that the
resampling unit was SLICES rests entirely on prose at three sites.

**[RECORD] Test was scored five more times today.** Two probe arms and three
data-scaling seeds. The frozen split's test slab is not a clean holdout and has
not been one for a long time: 44 leaderboard rows, plus thresholds, plus pipeline
variants, plus retrains have each queried it. README concedes roughly 0.02 to
0.03 of optimism from the 40-model sweep alone, which understates the total.
Recording the count here because the constitution's "touch test once" rule is
being deviated from routinely and the deviations were not previously written
down.

**[SETTLED] The recount ran, and `SE_REF_N = 19` is a BINARY-CC count.**
`src/recount_chondrules.py` counts both ways over identical voxels, taking zone
membership from `split.json`'s `abs_slices` and the valid mask from the same
cache training uses. Output in `results/chondrule_recount.json`.

| zone | slices | same-id 3D | binary CC | shortfall |
|---|---|---|---|---|
| train | 213 | 63 | 49 | 22.2% |
| val | 60 | **36** | **19** | 47.2% |
| test | 80 | 43 | 24 | 44.2% |
| total | | **142** | 92 | |

Validation holds exactly 19 by binary connected components and 36 by the same-id
rule. The constant therefore froze the precise undercount that
`pipeline_v2.count_chondrules_3d`'s own docstring warns against.

**[RETRACTED, same day] An earlier draft of this paragraph read "The predicted
shortfall for that hypothesis was 47.2% and the measured shortfall is 47.2%."
That is a TAUTOLOGY presented as a confirming prediction, and it is withdrawn.**
Both figures are `1 - 19/36` computed from the same two measured integers, so
they cannot disagree. The only genuinely prior number on record was the 39%
whole-volume shortfall in `pipeline_v2.count_chondrules_3d`'s docstring (87
against 142), which does NOT match 47.2%.
This is the same self-confirming-check defect flagged in `cascade.py` earlier the
same day, reproduced in prose here.

Of the three decompositions on record: the 2026-08-02 handoff's **63/36/43 is
CONFIRMED** as the same-id decomposition, and the same-id total of 142 matches
`split.json`'s `n_chondrule_instances`. The 99/19/24 in this file's 2026-07-29
"Prepare for a second meteorite" entry, under "Free statistical power from slab
placement", is NOT reproduced: its 19 and 24 are the binary-CC val and test
counts, but 99 matches neither rule for train, which measures 63 same-id and 49
binary-CC. 81/32/29 belongs to the different v2 split and is not comparable.

[HONESTY] The COUNTS above were measured directly and stand. The claim about
where `SE_REF_N` came from does not stand as firmly: the numeric match holds on
two zones of three and the third contradicts it, since 99 matches neither 63 nor
49. No per-zone binary-CC number was ever recorded in this repo before today, so
this is strong circumstantial evidence that the constant was taken from a
binary-CC count, not proof of its provenance.

**The constant was still NOT changed, and re-pointing it to 36 is still wrong.**
Confirming that 19 is a binary-CC artifact does not make 36 the right anchor.
`SE_REF_IOU = 0.0071` was bootstrapped over 60 validation SLICES, so neither 19
nor 36 was ever the resampling unit. Substituting 36 would leave the unit
mismatch untouched while loosening every projection by sqrt(36/19) = 1.376. The
defect is that the anchor is expressed in chondrules at all. Fixing it properly
means re-expressing the anchor in the unit that was actually resampled, which
rewrites every projection row, and that decision is still open.

**The floor's blast radius is six files, not three.** Previously believed to be
`pipeline_v2.py` plus two hardcodes. Also affected: the `abs(dv) < 0.0142` and
`abs(dm) < 0.0142` comparisons in `src/run_error_cnn.py`'s reporting block (an
executable gate), `src/confidence_compare.py`'s `[REGIME]` MDE print (hardcodes
the count 19) and its `json.dump` of `mde_iou` into
`results/confidence_compare.json` (persists the value), and the `axhspan` band
in `src/make_figures.py`'s `fig_data_scaling` (the +/-0.0071 band rendered into
`results/figures/data_scaling.png`). Two floors already coexist before any edit:
`run_all` takes n from the split metadata and computes 0.0109 on the v2 split,
while three modules print and store 0.0142.

**Unresolved and larger than the count question.** The code uses MDE = 2.0 * SE
while project prose cites 2.8 * SE at three sites with no derivation anywhere.
That is 1.4x on every floor. Separately, the SE was bootstrapped over slices
whose adjacent-slice label IoU is about 0.83 and whose prediction agreement is
96.0%, so an i.i.d. slice bootstrap understates the SE. Both point the same way:
the true floor is probably wider than 0.0142, which makes more of this project's
measured method effects unresolvable, not fewer.

**Fixed: the probe destroyed its own first result.** `src/pretrained_probe.py`
built its output rows from empty and rewrote `results/pretrained_probe.json`
after every arm, so running the two arms as separate processes (the documented
workflow, and what the handoff instructed) had the second silently delete the
first. Added `_prior_rows`, which keeps the other arm's row and still replaces a
re-run of the same arm. The pretrained row was backed up before the control arm
launched.

**Docs: per-row provenance markers.** ROADMAP.md's results table and budget curve
carried a block-level supersession warning but no per-row marker, so any single
row quoted away from the warning read as current. Every superseded row now
carries [PRE-FIX] in its own cell. Rows that never touched the CNN tiler
(classical, GNN, intensity threshold) are marked as such instead.

**The control arm, and the probe's verdict: NULL.** The scratch arm, identical
in every respect except that its ResNet-18 encoder starts from random weights,
scores **test IoU 0.33569** against the pretrained arm's 0.30269.

| metric | pretrained | scratch | gap |
|---|---|---|---|
| test IoU | 0.30269 | 0.33569 | -0.03300 |
| Dice | 0.4647 | 0.5026 | -0.0379 |
| precision | 0.3394 | 0.4076 | -0.0682 |
| recall | 0.7368 | 0.6554 | +0.0813 |
| AUPRC | 0.4671 | 0.4430 | +0.0240 |
| ECE | 0.0736 | 0.0633 | +0.0103 |
| val IoU | 0.5282 | 0.4957 | +0.0325 |
| val-frozen threshold | 0.018 | 0.002 | |

Judged against the threshold registered BEFORE the numbers arrived: a test gap
under about 0.05 is ambiguous at n=1 seed. **|-0.033| is inside that band, so
this is not evidence either way, and it is not being reinterpreted now that it
has a direction.** What can be said is that the ImageNet encoder did not deliver
a large win, and the sign is opposite to "pretraining substitutes for data".

**Ninth val/test sign flip.** Pretrained won validation by +0.0325 and lost test
by -0.0330. The pre-registration said to judge on test precisely because of this
history, and the history repeated.

**[UNVERIFIED] The scratch arm's threshold sits on the grid floor.** Its
val-frozen threshold is 0.002, which the KNOWN ISSUES block above records as the
grid's lowest value, and the log prints `thr=0.00`. The two arms were therefore
scored at thresholds an order of magnitude apart, one of them pinned at the grid
boundary, so the comparison may not sit at either model's real operating point.
The verifier launched to settle this died on a session limit and the question is
OPEN. Do not treat the -0.033 as clean until it is answered.

**Second reading, the architecture one.** Scratch ResNet18-UNet (0.33569) beats
plain `unet2d_s30000` (0.29409) by +0.0416. Also n=1 seed, also inside the
ambiguity band, and it is the comparison the probe's own docstring says to make.
**[CORRECTED 2026-08-04] It does not isolate architecture.** `resunet18_*` is
BatchNorm throughout (a torchvision ResNet-18 encoder plus a decoder built from
`nn.BatchNorm2d`) while `unet2d_s30000` uses GroupNorm (`models_cnn._norm`), so
this comparison crosses a normalisation change as well. The two arms of the
probe still differ only in encoder initialisation; it is the comparison against
`unet2d_s30000` that is confounded. One consequence: the two `resunet18_*` rows
carry `tiler_regime: post` for a net that does not use GroupNorm, so the
tile-grid dependence documented on the GroupNorm nets does not transfer to them.

**Validation predicts test well, and is still useless here. Both are true.**
Sky, 2026-08-04: *"things that score well on validation are overfitting"*. Tested
against the 12 models that carry a recorded `best_full_val_iou` (10 from
`final_pipeline.json`'s ranking, 2 from the probe), correlated with their
leaderboard test IoU. The result went AGAINST the expectation:

| statistic | all 12 | dropping `unet2d` |
|---|---|---|
| Pearson r | +0.794 | +0.861 |
| Spearman rho | +0.862 | |
| slope | +0.825 | +0.898 |
| residual SD | 0.02565 | 0.02224 |

(`unet2d` is dropped in the second column because its recorded val IoU is
identical to `unet2d_s30000`'s, which is the probs-collision defect above: its
map is the 30,000-step run's.)

Validation is a strong predictor of test ACROSS ARCHITECTURES. It is also range
restricted here, since 10 of the 12 are the top 10 by val, which should have
depressed the correlation rather than inflated it.

**The reconciliation is the residual.** Scatter around the fit is 0.0222 IoU. A
val gap has to be large enough that the predicted test gap clears the noise on a
DIFFERENCE of two residuals, which has SD sqrt(2) times the residual SD. Both
this figure and the 39-model figures below were first published without that
sqrt(2) and understated the bar by 1.41x; corrected the same day. See the
corrected table. Two effects this project has measured, for scale:

| effect | val gap | test gap | outcome |
|---|---|---|---|
| pretrained vs scratch (today) | +0.0325 | -0.0330 | sign flipped |
| geometry augmentation | +0.0157 | -0.0501 | sign flipped |

So the nine sign flips are not evidence that validation is broken. They are
evidence that it has been read at a resolution it does not have. Sky's claim
holds in the sharper form: **val differences below roughly 0.15 IoU carry no
information about test**, and every experiment run on ALH 83010 has lived far
inside that band.

**[CORRECTED, same day]** That sentence first read "under about 0.05 IoU are
noise", using the 12-model bar computed without the sqrt(2). Corrected, the
12-model bar is 0.070 and the 39-model bar is 0.151. The 0.05 figure is
withdrawn twice over: for the missing sqrt(2), and because that sample was
restricted to the top ten by validation.

**Three routes, and they do NOT agree as neatly as first claimed.** An earlier
draft said "three independent routes now agree on the wall", citing 0.05, 0.032
and 0.050. After both corrections the three give:

| route | bar on a two-run difference |
|---|---|
| pre-registered probe band, set before any number arrived | 0.05 |
| measured seed SD 0.0084 (n=3), so 2 x SD x sqrt(2) | 0.024 |
| val-versus-test residual, 39 models | 0.151 |

They span a factor of six, not "the same answer". The honest reading is that
they measure different things, and they are not even bars on the same quantity:
the residual route bounds a VAL gap, while the seed route and the pre-registered
band bound a TEST difference. The seed route captures retest variance at one
configuration, the residual route captures how poorly val transfers across
models, and the pre-registered band was a judgement call.

**[CORRECTED, same day]** This paragraph first ended "All three exceed every
effect this project has measured, which is the conclusion that survives." That
is false, and it reinstated the census overclaim retracted elsewhere in this
entry. Checked against the two effects with both numbers recorded: the 0.151 val
bar does exceed both val gaps (0.0325, 0.0157), but the 0.024 seed-derived bar
does NOT exceed either test gap (-0.0330, -0.0501), and the 0.05 band does not
exceed the -0.0501 augmentation test gap. What survives is narrower: the val
residual bar exceeds both recorded val gaps by 4.6x and 9.6x, and no route places
any recorded effect comfortably above the noise.

**[SUPERSEDED, same day] The full 39-model version puts the bar at 0.151, not
0.05.** Val IoU was recomputed from the saved `results/probs/<model>_val.npy` at
each row's own frozen threshold, for every leaderboard model that has one (39 of
44; five have no saved map). That removes the range restriction. Written to
`results/val_vs_test.json`.

Method self-check: 11 of the 12 recomputed values match their recorded
`best_full_val_iou` to within 0.0024.

The bar is `2 * sqrt(2) * residual_SD / slope`. The sqrt(2) is there because the
quantity that has to clear the noise is a DIFFERENCE of two residuals.

| subset | n | r | slope | residual SD | val gap needed (2 sigma) |
|---|---|---|---|---|---|
| all | 39 | +0.898 | +0.473 | 0.02524 | **0.1510** |
| excluding collapsed (test >= 0.10) | 37 | +0.827 | +0.436 | 0.02547 | 0.1653 |
| excluding weak (test >= 0.15) | 35 | +0.806 | +0.403 | 0.02449 | 0.1720 |
| competitive only (test >= 0.20) | 24 | +0.744 | +0.426 | 0.02225 | 0.1476 |
| C_cnn only | 31 | +0.899 | +0.465 | 0.02434 | 0.1479 |
| pre-fix only, one regime | 28 | +0.904 | +0.416 | 0.02011 | 0.1368 |

**[CORRECTION, same day]** These were first published as 0.1068, 0.1169, 0.1217,
0.1044, 0.1046 and 0.0967, computed as `2 * residual_SD / slope` with no sqrt(2).
That understated every bar by 1.41x. The corrected figures are above; the r,
slope and residual SD columns are unchanged. The earlier 0.05 from the 12-model
sample is separately WITHDRAWN, since that sample was restricted to the top ten
by validation.

Robust: the bar sits at 0.137 to 0.172 however the set is cut, residual SD stays
at 0.020 to 0.025, and the slope stays at 0.40 to 0.47.

**The scale of the problem, in one line.** The 24 competitive models (test IoU
above 0.20) span just 0.2564 in validation IoU, and the bar is 0.1476. The bar
is 58% of the entire spread of usable models, so this dataset can reliably
separate about two tiers of model and nothing finer.

For scale, the two effects measured here that have both a val and a test number
recorded have val gaps of 0.0157 and 0.0325, four to nine times below the bar.
That is two data points, not a census: the project's other effects (hierarchy,
prune, z-refine) were not scored into this analysis, so no claim is made that
EVERY effect ever measured falls below the line.

**[DEFECT] `final_pipeline.json`'s ranking holds a stale val IoU for
`unet2d_s30000`.** It records 0.47696, the PRE-fix value. Recomputing from the
current saved val map gives **0.44512**, which matches the 0.4452 independently
recorded in `results/error_cnn.json`. The saved map was rewritten by the
2026-07-29 retrain; the ranking file was not. This also explains why `unet2d`
and `unet2d_s30000` appeared to share a val IoU of 0.47696: `probs/unet2d.npy`
is the pre-fix s30000 map, as established above.

CAVEATS, not buried:

  * A cross-model residual is not the same quantity as within-model retest
    variance, and the two should not be added naively.
  * Five models have no saved val map and are absent.
  * Test IoU is systematically lower than val IoU, which is part of why the
    slope is well under 1.
  * **Two of the 39 are degenerate and carry heavy leverage.** `gnn_gat` and
    `matrix_specialist` both collapsed to predicting everything foreground, so
    their val and test IoUs are exactly the zone positive fractions (0.06661 and
    0.05279) and they sit as coincident points at the bottom of the range.
    Dropping them takes r from +0.898 to +0.827. The bar is stable across that
    cut (0.151 to 0.165), but the headline correlation is not, and r = +0.898
    should not be quoted without this.
  * **"39 models" is not 39 independent models.** `unet2d` and `rt_dice_ctrl`
    are the same configuration at the same seed, established elsewhere in this
    entry, so they are one measurement entered twice.
  * The val axis is recomputed from float16 maps while the test axis was scored
    in float32 at run time. Quantisation noise on one axis only would inflate
    the residual and therefore the bar.

**[MEASURED] First seed variance in this project: SD 0.0084.** Fraction 0.25 of
train at the 7,200-step budget, three seeds, test IoU 0.2823 / 0.2664 / 0.2696,
mean 0.2727, SD 0.0084 with 2 degrees of freedom.

Quoting `src/data_scaling.py:16-20` exactly, because an earlier draft of this
entry misrepresented it: *"Single-seed differences of 0.03-0.06 sit only ~4-8 SE
apart before counting training-seed variance, which is itself large. With n=1 per
point there is no way to tell a real trend from run-to-run scatter."* It does NOT
assert that scatter is 0.03 to 0.06; that straw man is withdrawn. What the
measurement does contradict is the clause "training-seed variance, which is
itself large": measured at 0.0084, it is smaller than the 0.0142 val noise floor,
not larger.

Separately, CHANGELOG twice attributes run-to-run variation to non-deterministic
MPS. That mechanism looks wrong. Two executions of the same arm in
`logs/data_scaling_v2.log` agree on every printed digit for the three epochs that
overlap, and `unet2d` and `rt_dice_ctrl` (the same configuration at the same
seed) agree on every metric. [HONESTY] Three epochs of one log and one row pair
is thin evidence for a general determinism claim; no full-run or test-IoU
determinism check has been made.

[HONESTY] This does NOT retroactively make the day's pretraining gap
significant. The 0.05 ambiguity band was registered before the numbers arrived
and is not being revised afterwards. Three separate reasons to leave it alone:
an SD from n=3 has a very wide confidence interval; this was measured at 7,200
steps on 25% of train, not the probe's 30,000 steps on all of it; and the probe's
arms differ in initialisation, which carries its own variance. Seed replicates
at the probe's own regime would settle it.

Also noted: the three seeds froze thresholds of 0.002, 0.8 and 0.002. Two of
three sit on the documented grid floor, and the same configuration selected a
400x different operating point across seeds while test IoU moved only 0.016.
**[CORRECTED 2026-08-04]** The third threshold was written here as 0.00. That is
the log's two-decimal render, the same render this file warns about elsewhere;
`results/data_scaling_v2.json` records 0.002 for seed 2.

**The multi-meteorite path was untested, and it is half broken.** `run_all` has
never completed and `results_v2/pipeline_v2.json` does not exist, so the claim
that adding a meteorite is a config change had never been exercised. Tested here
against synthetic sources:

  * ONE meteorite: works.
  * TWO meteorites of the SAME in-plane size: works. Provenance is tracked per
    slice, per-zone chondrule counts come out, and the leakage contract holds,
    with every meteorite present in all three zones.
  * TWO meteorites of DIFFERENT in-plane size: **crashes** at
    the `np.concatenate` over per-meteorite zone arrays in
    `pipeline_v2.build_multi_split` with a bare `ValueError` that
    mentions neither meteorites nor scans. Two real scans essentially never
    share dimensions, so this would have fired on the first day of the project's
    stated next step.

Fixed by padding each meteorite up to the common maximum and marking the padding
INVALID, which reuses the `valid` mask the metrics already honour, and physical
scale is preserved. `MeteoriteSource` also gained `voxel_um`, recorded and not
acted on: padding reconciles differing pixel COUNTS but not differing pixel
SIZES, and this model has no scale invariance.

[CORRECTION, same day] An earlier draft of this entry said "nothing downstream
changes". That is an overclaim and it is withdrawn. `valid` does exclude padded
pixels from every loss and every metric, which was verified. But `_norm` is
GroupNorm, which normalises over whatever spatial extent an inference tile
covers, so a tile straddling the zero padding computes its statistics partly on
that padding and the predictions for REAL pixels inside that tile shift.
Constant-0 padding maximises the effect, since matrix reads around 32 CT. The
magnitude is second order and unmeasured; the claim of no downstream effect was
simply wrong.

Verified by `tests/test_multi_meteorite.py`, which asserts the property the fix
rests on: padded rows and columns are invalid in every zone, carry no positives,
and the native region survives. If padding were left valid, every metric would
score empty space as true negatives and every IoU would be wrong with nothing to
catch it. The test also records `pos_frac` against `pos_frac_valid` (0.03884 vs
0.04834 on the synthetic train zone) so the dilution is visible.

Sky, 2026-08-04, on the one loose end: `PatchSet` draws negative patch centres
uniformly and does not consult `valid`, so padded area costs a proportional
share of negative draws. Not worth acting on, because the hierarchy runs as a
FILTER ahead of the second stage and uniform-zero padding is dropped at the
coarsest level, so the complete model never sees it.

**[DEFECT] `results_v2/split_multi.json`'s regime string misdescribes its own
counting rule.** It says 3D chondrule counts are made "by scipy.ndimage.label on
mask & valid", which is binary connected components. The code in
`pipeline_v2.build_multi_split`'s per-zone loop loads instance IDs and calls
`count_chondrules_3d`, which is the same-id rule, and whose own docstring warns
binary CC undercounts by 39% (87 against 142). The stored provenance string
therefore names the exact method the code was written to avoid. This matters
because that string is one of the artifacts anyone would consult to reconcile
the 19-versus-36 question.

**Data-scaling curve: 3 of 12, up from 1.** The handoff's "1 of 12" was correct
when written, and an earlier claim in this session that the JSON held 5 runs was
wrong. At the start of today `results/data_scaling_v2.json` held one completed
run (frac 0.25, seed 0, test IoU 0.2823); a second (frac 0.25, seed 1) had
trained to full-val IoU 0.3897 in an earlier session but its test scoring was cut
off and never saved. Both remaining seeds of fraction 0.25 were run here, so that
fraction is now complete at 3 of 3 and the curve stands at **3 of 12**. The
intended design is 4 fractions times 3 seeds. Roughly 5 hours remain for the 9
outstanding runs, and the runner skips completed pairs, so it is resumable.

`src/data_scaling.py` gained `--frac=` and `--seed=` so a single fraction can be
finished without committing to the whole grid. The grid had been abandoned three
times as an all-or-nothing job.

### 2026-07-31: Classical Stage-0, the image confidence model, and what is still broken

*Spec (condensed). **Problem:** Sky proposed a classical pipeline (denoise,
CLAHE, morphological texture separation, watershed) and an image-based confidence
model, and reordered the complete model so pruning happens last on three stacked
signals. **Approach:** build each as an addition to the existing paths so the
comparison is like-for-like, and adversarially review before trusting anything.
**Acceptance:** every arm scored under the frozen split with test touched once,
and every unfixed defect written down rather than left in the code.*

**Classical Stage-0: the harness reproduces, and Sky's additions help.**

| arm | test IoU | Dice | prec | recall | AUPRC | graded% |
|---|---|---|---|---|---|---|
| `base18` (the existing 18 features) | 0.2028 | 0.3373 | 0.2486 | 0.5241 | 0.2428 | 30.62% |
| `plus3` (+ denoise, CLAHE, top-hat) | 0.2127 | 0.3508 | 0.2910 | 0.4414 | 0.2799 | 30.65% |
| `plus3_ws` (+ watershed) | 0.2185 | 0.3586 | 0.3209 | 0.4063 | 0.2799 | 30.65% |

`base18` reproduces the published classical-ML baseline (0.203) to three
decimals, so the harness is validated. Sky's cleaning channels add **+0.0099** and
watershed a further **+0.0058**; unlike most comparisons here these are *paired*
(identical sampling, estimator and seed), so they carry no training-seed noise.
Two runs produced byte-identical numbers. **But at 0.2185 it loses to the CNN's
0.29409 by 0.076**, so it is not a competitive Stage 0 on its own. Its one
striking property is **30.6% graded pixels against the CNN's 1.19%**, 26x more
gradation, which is what stages 3-6 consume.

**The image confidence model (error U-Net) HURT, badly.**

```
base (threshold only)   IoU 0.2944
+ margin baseline       IoU 0.3028
+ error U-Net           IoU 0.1175      -0.1769 vs base, 12x the noise floor
```

On the held-out validation half it *worked* (unpruned 0.0732 -> pruned 0.1948,
+0.1217) and then transferred catastrophically. **[HONESTY] Three confounds, and
the review found a fourth that may dominate:** it was fit on ~9 chondrules
(val halved again); the base map is 91% saturated so there is little to learn
from; the val/test split has flipped sign seven times in this project; and
`fit_error_unet` samples patches **uniformly**, with no positive weighting, while
`PatchSet` uses `pos_frac=0.5` precisely because the target is ~0 across ~95%
background. The result is therefore **not yet evidence about image confidence
models.** Do not cite it as one.

**Cross-fitting measured, and the first hypothesis was wrong.** To give the error
model more than 9 chondrules, the base model's train predictions must be
out-of-fold. A 2-fold classical cross-fit gave out-of-fold train IoU 0.1507 vs
0.2875 full-fit on val, which looked like a channel-quality problem; a 5-fold
redo (2 h) moved it only to 0.1591. The decisive measurement, the full-fit model
scored **in-fold on its own train data**, settles it:

```
same 27 train slices, thr 0.5:  in-fold 0.1736   out-of-fold 0.1655   gap +0.0081
foreground fraction:            train 0.0359   val 0.0666   test 0.0528
```

Gradient boosting barely memorises here (+0.008). The apparent 0.13 gap is **zone
prevalence**: train has half the foreground density of val, and IoU is strongly
prevalence-sensitive. Cross-fitting was still correct (the constitution requires
it) but is cheap insurance for this model class, not a load-bearing fix. The
2 h re-run was spent on a wrong diagnosis that a 90-second in-fold check would
have refuted.

**Constitution §12 reordered (Sky):** hierarchy, then classical, then CNN (whether the
classical map is an input channel is UNSETTLED; Sky chose to build both arms
and let validation decide), then regressor confidence, then error
U-Net, then z-neighbour, and **pixel dropping LAST**, stacking all three confidence signals into
one val-fit score. The cascade channel must be cross-fitted;
`src/classical_crossfit.py` produces it (5 contiguous folds, 12-slice buffer, all
213 train slices covered, verified disjoint).

**The confidence bake-off, which answers Sky's actual question.** Run at recall
0.99 on both base models (`results/confidence_compare.json`):

| base model | graded% | base IoU | pruned IoU | gain (test) | gain (VAL) |
|---|---|---|---|---|---|
| `unet2d_s30000` | 1.19% | 0.2944 | 0.3028 | +0.0084 | **-0.0071** |
| `unet2d_s30000_auggeom` | 0.92% | 0.2436 | 0.2461 | +0.0025 | **-0.0077** |

**Sky's question was whether an augmented base model gives its confidence model
more to learn from. The answer is no, and it moved the wrong way**: the augmented
map is LESS graded (0.92% vs 1.19%), and its prune gains less.

**[HONESTY] Both "gains" are below the 0.0142 floor AND negative on validation,
the split everything was selected on.** The positive test numbers are sign flips
below the noise floor, the seventh and eighth recorded in this project. Do not
cite +0.0084 as a gain.

**The learned models pruned nothing.** Every error regressor and every trust
classifier returned a pruned val IoU equal to the UNPRUNED value to five decimals
(0.06661 control, 0.06429 augmented), i.e. the fitted cutoff was "remove
nothing". The winner on both arms was the zero-parameter free baseline `margin`
(0.4381 / 0.4532), whose winning cutoff is bit-identical to keeping `p == 1.0`.
Since pruning by `|p - 0.5|` on a map opened to recall 0.99 is **re-thresholding
in disguise**, constitution §9's over-predict-then-prune collapsed back to
choosing a threshold, and ten learned meta-models could not beat it. Extending
the cutoff grid (linear plus quantile, up to 48 points) changed nothing, so the
search was fair. Constitution v3+b requires exactly this to be reported rather
than hidden: **the learned confidence model has not earned its complexity.**

**New modules in this commit, with their status:**

| module | what it does | state |
|---|---|---|
| `classical_pipeline.py` | denoise, CLAHE, top-hat, watershed | scored; watershed defective (below) |
| `run_classical.py` | scores the classical arms, saves probs + estimators | reproduces the 0.203 baseline |
| `classical_crossfit.py` | 5-fold out-of-fold classical maps for the cascade channel | run; all 213 train slices covered |
| `cnn_crossfit.py` | 5-fold out-of-fold CNN maps, so the error model gets more than 28 slices | **written, never completed a run** |
| `error_cnn.py`, `run_error_cnn.py` | image-based confidence model (U-Net on raw+prob) | run once; result confounded (below) |
| `confidence_compare.py` | confidence bake-off across base models at a chosen recall | run at recall 0.99 |
| `tta_confidence.py` | test-time augmentation as error-model training data | **dead code, four defects (below)** |

`cnn_crossfit.py` was started once and killed 3 epochs in after a review found
three defects in it, all now fixed and none ever exercised: epoch selection was
being done on the held-out fold itself (with `val_stride=6` every held-out slice
sat within 3 of a selection slice, inside the project's 12-slice buffer, so the
contamination covered the whole fold); the incremental save wrote a
NaN-containing partial to the FINAL path, where it would look complete to every
reader and silently poison a `PatchSet` aux channel or return `gain 0.0` from
`fit_prune`; and fold weights were discarded. It now selects on the real val
slab, writes `.partial` and renames only after asserting every slice is finite,
persists per-fold checkpoints, and records genuine out-of-fold IoU per fold
rather than the val IoU it was selected on.

**SHIPPED WITH KNOWN DEFECTS, recorded and not fixed.** A pre-commit review found
these in code that is being committed.

**[RETRACTED] One of them DOES affect a published number in this very entry.**
An earlier draft of this block claimed "none affects a published number". That is
false: the `plus3_ws` row above (test IoU 0.2185, credited with **+0.0058** from
watershed) is a direct product of the defective `watershed_split` below. Its
recall falls 0.4414 to 0.4063 while precision rises, which is the signature of
**silently deleting small objects**, not of better instance separation. **Treat
the +0.0058 watershed gain as unverified**; it may be an artifact of dropping
objects the metric then stops penalising. The `plus3` row (+0.0099 from the
cleaning channels) is unaffected, it involves no watershed.

- `classical_pipeline.watershed_split` **silently deletes small objects** (any
  component whose distance-transform max is below `min_distance` gets no marker
  and vanishes) and **over-segments ~22x** (22,375 labels over 80 test slices
  against a measured 999 true 2D components = 12.5 per slice). The label count is
  itself an upper bound: `run_classical` accumulates `lab.max()`, not surviving
  labels, so `min_size` dropping punches holes in the id range. This is what
  makes the `plus3_ws` row above unverified.
- `error_cnn` uses a **4-slice val buffer** where the project mandates 12, and
  the unweighted sampling above.
- `tta_confidence.build_rows` delivers **no extra rows** at default `max_px`
  (it divides the budget by 8 rather than multiplying), samples **disjoint**
  pixels per orientation rather than the same pixels under 8 views, needs ~6 GB
  per orientation, and has **no inference path** (`_chunks` has no `extra`
  parameter, so a 12-feature model raises on predict). The module is currently
  unreachable dead code.
- `tests/` remains **empty** across ~1,400 new lines.

### 2026-07-30 — Geometry augmentation at a matched 30k budget: −0.0501 test IoU

*Spec (condensed) — **problem:** the only augmentation evidence was a 7,200-step
comparison that structurally penalises augmentation, so "augmentation loses" was
not established. **Approach:** run geometry-only augmentation at the 30,000-step
headline budget, everything else identical, and adversarially verify the result
before recording it. **Acceptance:** the two arms differ ONLY in augmentation,
verified from the run's own saved config rather than from intent; and every
inference drawn from the numbers survives a hostile check.*

**Result — metrics together per §5:**

| model | test IoU | Dice | precision | recall | AUPRC | thr |
|---|---|---|---|---|---|---|
| `unet2d_s30000` (control) | **0.29409** | 0.45452 | 0.37188 | 0.58437 | 0.42703 | 0.002 |
| `unet2d_s30000_auggeom` | **0.24404** | 0.39233 | 0.30375 | 0.55385 | 0.26341 | 0.002 |
| **Δ (augmented − control)** | **−0.05005** | −0.0622 | −0.0681 | −0.0305 | −0.1636 | |

**Comparability was verified, not assumed.** The saved checkpoint
(`results/checkpoints/unet2d_s30000_auggeom.pt` — the first this project has ever
written) records the run's ACTUAL config: `augment="geom"`, `seed=0`, 60×500,
batch 16, patch 128, lr 3e-4, wd 0.0, pos_frac 0.5, k=1, dice loss. Every value
matches the control's recipe. The training and inference code were shown
AST-identical across the two runs, and a tiler seam-signature probe confirms both
used the post-fix tiler (top-8 column jumps at 255/511/767/1023, versus the pre-fix
signature 223/287/479/543/735/799/991/1055 carried by both 7.2k arms).

**Not a thresholding artifact.** The deficit is **−0.047 … −0.054 at every
threshold from 1e-4 to 0.95**, and **−0.0504** when each arm is re-thresholded at
its own validation optimum on a grid extended to 1e-7. (0.002 is the floor of
`CNNRunner.fit`'s own grid, but re-selecting freely moves the augmented arm's
optimum to 8.76e-4 and makes it *slightly worse*, so the floor is not penalising
it.)

**[HONESTY] This is a replicated point estimate, NOT a decisive result.**

- **n=1 seed per arm**, on non-deterministic MPS. This project's own
  `src/data_scaling.py` docstring states that single-seed differences of
  0.03–0.06 "sit only ~4-8 SE apart *before* counting training-seed variance"
  and that with n=1 "there is no way to tell a real trend from run-to-run
  scatter". −0.05 is squarely in that regime.
- **The 0.0142 MDE is a VALIDATION-side figure** (`pipeline_v2.py`:
  `SE_REF_IOU=0.0071`, `SE_REF_N=19` val chondrules). There is no test-side MDE
  anywhere in this repo, and the figure excludes training-seed variance entirely.
  Quoting "3.5× the MDE" against a test delta overstates what it means.
- **Validation says the opposite.** Full-val 0.4451 → 0.4608, i.e. **+0.0157**,
  stable at every threshold — but that is only **1.1× the val MDE**, so "won
  validation" would overstate it too. Seventh val/test sign flip recorded here.
- **The two SCORED checkpoints did not consume equal training.** Best-by-val
  epoch was **21/60 for the control** (~10,500 steps, then decaying 0.089 from
  peak) and **43/60 for the augmented arm** (~21,500 steps, still within 0.023 of
  peak at the end). "Matched 30k budget" is true of compute *allotted*, not
  compute *used* — and the augmented arm was still near its optimum when the
  budget ran out.

**[RETRACTED] "This refutes the budget-confound hypothesis."** It does not, and
the claim switched metrics mid-argument. Tracked consistently on VALIDATION the
gap runs 0.165 (3k) → 0.0133 (7.2k) → **−0.0157** (30k), which is what the budget
hypothesis predicts; the earlier "augmentation simply needed steps to amortize"
entry was itself made on val. The epoch-of-peak evidence above points the same
way. **Val and test disagree; the hypothesis is CONTESTED, not closed.**

**[RETRACTED] "CT slices have a privileged orientation."** Offered as the likely
mechanism; it is unsupported. Grep finds zero occurrences of ring artifact, beam
hardening, beam direction, settling or any isotropy measurement anywhere in the
repo — and the augmentation is **in-plane only**, so settling and beam direction
(z-axis phenomena) *cannot* be penalised by it. Only ring artifacts would even be
mechanically applicable. The `PatchSet` comment it contradicted ("CT texture has
no privileged orientation") is equally unmeasured; both sides are speculation.

**[DEMOTED] AUPRC as independent corroboration.** −0.164 is real at float32, but
recomputing at the float16 resolution the maps are actually stored in gives
0.309 → 0.231, a gap of −0.079 — same sign, half the magnitude. ~90% of pixels
sit below 1e-4 for both models, so AP is dominated by ordering inside a saturated
mass, and the integral includes recall regions nobody operates at. The threshold
sweep settles the question better; AUPRC is a consistency note only.

### 2026-07-30 — Calibration (DSC++), the recall floor, and what the literature says

*Spec (condensed) — **problem:** three separate failures (a saturated probability
map, a confidence stage that never beats a free baseline, and a prefilter that
cannot use the CNN) all trace to one suspected root cause: an overconfident base
model. **Approach:** check the literature before inventing anything, then attack
the root cause rather than the symptoms. **Acceptance:** each proposed change is
traced to a published result, implemented from the actual formulation rather than
memory, and unit-tested for the behaviour it is supposed to produce.*

**Literature review — three of our findings are named, documented problems.**

1. **Dice loss is documented to cause exactly our saturation.** Yeung et al.,
   [arXiv:2111.00528](https://arxiv.org/abs/2111.00528): the DSC loss "is known
   to yield poorly calibrated, overly confident predictions", specifically
   dangerous on small medical datasets. Their **DSC++** fix exponentiates only
   the FP/FN penalty terms; at γ=2 negative log-likelihood fell ~5× across five
   datasets (e.g. DRIVE 0.204 → 0.041) with Dice preserved or improved.
2. **Our tile artifact is a named problem and normalization is the cause.**
   [arXiv:2503.19545](https://arxiv.org/abs/2503.19545), "Tiling artifacts and
   trade-offs of feature normalization": *"we investigate the root cause of these
   issues and show that they stem from the normalization layers within the neural
   networks."* Independently the same conclusion reached here from the figures.
   They recommend **BatchRenorm**; global normalization (BatchNorm) gives
   artifact-free stitching at a quality cost, tile-wise gives quality with
   artifacts. **This also means the whole-slice attempt was the wrong axis
   entirely** — the fix belongs in the normalization, not the stitching.
3. **Our hierarchy result is the known domain finding.** The meteorite micro-CT
   literature attributes poor simple-classifier performance to "an overlap in the
   range of intensities present in the different components" — the exact overlap
   measured here (chondrule regions 0.2506 vs empty rock 0.1578, p90 of empty
   above the chondrule mean). Local-histogram features beat pixel intensity
   85.8% vs 73.4% percent-correct, which indicates the prefilter's next feature
   set.

**`dscpp_loss` added and unit-tested** (`models_cnn.py`). γ=2 default.
[IMPLEMENTATION NOTE] The published equation renders with the exponent on the
SUMMED FP/FN terms; that reading is rejected and the reason is recorded in the
docstring — it is numerically explosive (a squared pixel-count sum swamps the
linear intersection term) and it would rescale total error uniformly rather than
"selectively modulate the penalty associated with overconfident, incorrect
predictions". The exponent is applied per-pixel. `_dscpp_selfcheck` asserts the
intended behaviour and passes:

```
                    hesitant wrong   confident wrong    gap
Dice                    0.008018         0.054999    0.046981
DSC++ (gamma=2)         0.000595         0.050529    0.049934   <- more selective
perfect 0.00000   inverted 0.98428
```

**🔴 `THR_GRID` floor fixed — third instance of this defect.** Extended to 1e-7
(`combine.py`). Before: every recall target ≥0.80 silently fell back to the 2e-4
floor, so a bake-off advertised at recall 0.95 **actually ran at 0.60** and four
"different" prune targets returned one identical row. After: recall 0.99 is
reachable at thr 2.14e-07 (achieved 0.9910). 0.995+ is genuinely unreachable
(max 0.9945) and now **prints a warning and sets `recall_target_unreachable` on
the returned metrics** instead of failing silently — the silence is why this
survived three investigations. [HONESTY] At recall 0.99 the base model's val IoU
is **0.0666** (precision 0.0667): that is the true cost of the over-predict step,
and the prune must climb back past 0.294 to beat plain thresholding.

**Augmentation — the record, corrected.** Both arms had already run, at 7,200
steps: `long_p1_ctrl_b32` (none) **0.2834** vs `long_p1_geom_b32` (flips +
rot90) **0.2329** — augmentation lost by **0.0505**, 3.6× the MDE. Caveats: both
predate the tiler fix, both n=1 seed, and a *fixed* budget structurally penalises
augmentation because the model sees each real configuration less often. Sky's
call is to re-test geometry-only at the 30k headline budget; `run_families` now
takes `--aug=` and `--loss=` overrides, each of which renames the run so a
bake-off arm can never overwrite the row it is compared against.

**[WITHDRAWN] Elastic deformation was proposed and retracted.** It was suggested
from generic biomedical practice; Sky challenged whether it preserves the data,
and it does not. Chondrules are rigid mineral spheroids — they solidified from
melt droplets, which is *why* they are round — and roundness is a feature
separating them from irregular matrix. Warping them produces shapes that do not
occur and would plausibly worsen the measured over-fragmentation (33 predicted
objects vs 10 true, 2.06× the area). Same error class as intensity jitter:
importing generic augmentation without checking domain validity.

**[HONESTY] Augmentation cannot lower the MDE.** It multiplies pixel
configurations, not independent chondrules. 3,000 transformed copies of 531
slices still contain the same 142 chondrules and the same 19 in validation, so
the 0.0142 floor is unchanged. For a *confidence* model — fit on the base model's
errors — rotated copies of one chondrule are one chondrule's worth of
information, and copies spanning fit and evaluation would be leakage.

**[DEFINITION, Sky 2026-07-30] "Confidence model" and "learned error model" are
the SAME THING in this project — one stage, not two.** The constitution had them
split as v3+ (trust classifier) and v3++ (error model); §12 now records that
stage 2 **is** the learned error model — a regression on the *magnitude* of the
base model's error, threshold-independent, yielding an error bar. The trust
classifier (`combine.ConfidenceModel`) is demoted to a **comparison arm** beside
the free baselines and is not a separate deliverable.

**`src/confidence_compare.py`** added: runs the bake-off across base models at a
chosen recall target, holding split/candidates/prune fitting fixed, and reports
each base map's **graded fraction** so "does the augmented model give its
confidence model more to rank" is answered with a measurement. Reports the
**error model** as the deliverable with trust/free beside it, plus explicit
`error_beats_trust` / `error_beats_free` flags — never a collapsed "best learned"
number, which would hide whether the deliverable actually won.

### 2026-07-29 (later) — Complete-model contract; scaling-job triage; headline retrain

*Spec (condensed) — **problem:** the handoff listed three tasks (verify the
scaling job, retrain the headline model under the fixed tiler, commit the pending
work) and Sky added a fourth: define "the complete model" ahead of a larger
dataset. **Approach:** triage the running jobs before spending compute on them;
back up any artifact a re-run would overwrite; define the contract as a spec +
constitution amendment rather than code, and verify every decision-feeding claim
adversarially before acting on it. **Acceptance:** each in-flight job's true state
established from process/file evidence rather than inference; the headline delta
reported against the reproducible baseline with its sign and its noise-floor flag;
the contract recorded in three places that agree; every claim that failed
verification retracted in writing.*

**Constitution §12 — "the complete model" is now one named thing (Sky's call).**
From here on it means the ordered composition of four pluggable stages: best base
segmenter (high-recall operating point) → hierarchy filter → learned confidence
model → z-layer neighbor confidence. Headline metric is **end-to-end test
foreground IoU** against the plain-U-Net-at-matched-budget baseline, with **every
stage independently ablatable** so a stage that does not earn its place is
visible. Recorded as constitution §12 and
`specs/2026-07-29-complete-model-contract/spec.md`.

**[HONESTY] This is an architecture contract, not a performance claim.** At n=19
val chondrules the MDE is 0.0142 IoU; the hierarchy measured **−0.012 vs the plain
U-Net — inside that floor** — and is justified on **compute** (2.41× at target
recall 0.999), not accuracy. The stages cannot currently be told apart from
break-even; the larger dataset is what makes the measurement possible.

**Scaling-curve triage — two jobs were in flight; the weaker one was stopped.**
The `src/data_scaling.py` run (4 fractions × 3 seeds × 7,200 steps) had **died at
run 1 of 12 with zero results saved** — `results/data_scaling_v2.json` has never
existed. A separate scratchpad job (`data_scaling2.py`, PID 39992) was still alive
after 71 min running a **single-seed 15,000-step** variant. It was killed, on
these grounds: (a) **one seed per fraction**, so a trend cannot be separated from
seed scatter (bootstrap SE ±0.0071 IoU); (b) it **bypasses
`evaluate.run_and_score`** and the shared leakage checks; and (c) it writes a JSON
**list** to the same path `src/data_scaling.py` reads with `.get("runs", [])` —
reproduced: `AttributeError: 'list' object has no attribute 'get'`, which would
have crashed the good module on its next start. The 12-run module is relaunched
after the retrain (serialized deliberately — both jobs share one GPU).

**[CORRECTION — the record of that kill was initially wrong.]** It was first
reported here, and to Sky, that the killed job "had written nothing". **That is
false**, found by the pre-commit overseer pass. The job's progress log was
`results/data_scaling_v2.log`, not the `logs/data_scaling_v2.log` that was
inspected, and it shows **one completed data point**:

```
[ds25] FULL-val IoU=0.4488 frozen thr=0.00
 25% ( 53 random slices): test IoU=0.2863 val=0.4488
```

with the 50% run 9/30 epochs in when it was killed. Two consequences: the claim
that it "carries the exact confound `src/data_scaling.py` was written to remove"
was also wrong — `src/data_scaling.py` names **two** defects (contiguous subsets
*and* one seed), and **`53 random slices` shows this job had already fixed the
contiguity confound**; only the single-seed defect remained, at a larger 15k
budget. The kill still stands on grounds (a)–(c), but ~71 min of compute produced
a real measurement. That measurement **was never written to JSON** (no
`data_scaling_v2.json` exists anywhere on disk), so its only record is the log,
which `.gitignore` excludes via `results/*.log` — it is therefore preserved in this
commit as `results/prefix_tiler_backup/killed_data_scaling2_job.log`.

**[HONESTY] Do not quote `test IoU=0.2863` as a scaling-curve point.** It is
n=1 seed, at a 15k budget, measured under the **pre-fix tiler**, and it has no
companion points to form a curve.

**Headline retrain under the fixed tiler — in flight, result not yet in.**
`python -m src.run_families run 0 60 500` → `unet2d_s30000`, 30,000 steps.
Pre-fix artifacts were copied to `results/prefix_tiler_backup/` first
(leaderboard, `final_pipeline.json`, the 213 MB probability map, and the four POC
figures) because `run_and_score` overwrites the leaderboard row and the very
probability map the POC figures were drawn from.

**[HONESTY] Which baseline the delta will be quoted against.** Three different
"headline" numbers exist for this model: **0.3168** (README, predates the
threshold-grid fix *and* the tiler fix), **0.31601** (committed code, thr 0.0076,
predates the tiler fix), and **0.31587** (`leaderboard_all.json`, thr 0.006). The
apples-to-apples comparison for the retrain is against the reproducible ~0.3159–
0.3160, not the published 0.3168; both will be reported with the sign.

#### Adversarial verification of the contract — one claim retracted, four gaps found

The first draft of the spec asserted that the z-neighbor stage was **not wired
into `run_all`**. **That was false and is retracted.** It is step 4 of
`confidence_sweep` (`pipeline_v2.py:573-574`), and `combine.z_agreement` is live in
two further places — a bake-off free baseline (`confidence_bakeoff.py:120`) and
feature 2 of `ConfidenceModel.build_features` (`combine.py:250`). All four stages
were already wired. What is actually missing is the **composite**:

- **`run_all` has never been executed** — no caller in the repo, `__main__` is
  plumbing-only by its own comment, and `results_v2/` holds only
  `split_multi.json`.
- **The three post-base treatments never chain** — bake-off winner (`:520`),
  prune (`:555`) and z-refine (`:574`) each read the same base map `pt`.
- **Stages 0 and 1 are fused** — `stage2_sweep` only builds `HierMaskedRunner`
  (`:454`), so no hierarchy-off arm exists and the ablation cannot be computed.
- Hierarchy thresholds were checked and are **genuinely leakage-free** (val-fit,
  frozen, decision rule reads raw only — `hierarchy.py:483-493`).

#### 🔴 `recall_threshold` cannot reach any recall target — constitution §9 is not running

`THR_GRID` floors at 2e-4 (`combine.py:34-38`) and `recall_threshold` **silently**
returns that floor when a target is unreachable (`combine.py:551-553`). Measured on
the real validation zone:

```
unet2d_s30000     targets 0.80/0.90/0.95/0.98 -> thr 2e-4, val recall 0.6014  (all four identical)
hier_cnn_s30000   targets 0.80/0.90/0.95/0.98 -> thr 2e-4, val recall 0.5531  (all four identical)
```

A grid-floor defect, **not** model saturation — recall at `prob > 1e-7` is 0.985
(unet2d) / 0.970 (hier_cnn). So the four prune targets in
`results/final_pipeline.json` are one row repeated (0.3011→0.3131 ×4),
`BAKEOFF_TARGET_RECALL = 0.95` actually ran at **0.60**, and **"over-predict, then
prune" (constitution §9) has never actually executed.** The CHANGELOG already
recorded the symptom; the cause is that the earlier fix moved the floor
0.05 → 0.0002, still ~3 orders too high. Secondary defects recorded in the spec:
unmatched threshold grids across arms, gains that conflate treatment with
threshold change, ~12 test scorings flagged against a single-comparison MDE, `_gain`
treating NaN as measurable, `out_dir` half-honored, and no collision guard in
`alive_index`.

**Sky's decisions:** (D1) extend `THR_GRID` to ~1e-7 log-spaced and make the
fallback raise or flag rather than fail silently — this **invalidates** the prune
and bake-off numbers in `final_pipeline.json`, which must be regenerated;
(D2) chain order is **z-refine the probability map first, then prune**, so no stage
binarizes early and the final threshold stays val-selected — deliberately *not*
reusing `run_final.py`'s `prune_plus_z`, whose 0.3214 was test-selected with a
hard-coded 0.5 cut; (D3) this session is **spec + commit only** — the composite and
the D1 fix are the next session's first task, because building the contract on a
base score that is currently being re-measured would have to be redone.

#### 📊 HEADLINE RETRAIN RESULT — the halo crop LOWERED IoU but RAISED AUPRC

`unet2d_s30000` retrained from scratch, 30,000 steps, same recipe
(`phase1:p1_ctrl_b32`), scored through `evaluate.run_and_score`. Metrics reported
together per constitution §5:

| metric | pre-fix (halo accumulated) | post-fix (halo cropped) | Δ |
|---|---|---|---|
| **test IoU** | 0.31587 | **0.29409** | **−0.0218** |
| Dice | 0.48009 | 0.45452 | −0.0256 |
| precision | 0.41118 | 0.37188 | −0.0393 |
| recall | 0.57674 | 0.58437 | +0.0076 |
| **AUPRC** | 0.35151 | **0.42703** | **+0.0755** |
| val-selected threshold | 0.006 | 0.002 | |
| full-val IoU | 0.47696 | 0.44510 | −0.0319 |

**Answering the handoff's question with its sign:** against the published 0.3168
the delta is **−0.0227**; against the reproducible `leaderboard_all.json` value
0.31587 it is **−0.0218**. Both **negative** — at the val-selected operating point
the defective tiler was *flattering* the model. The magnitude is ~1.5× the MDE
(0.0142), so it is nominally outside the noise floor.

**[HONESTY] But "the bug was helping" is too simple, and this run does not settle
it.** Four caveats, none of them small:

1. **AUPRC, which is threshold-free, improved by +0.0755** — a 21% relative gain.
   The probability map's *ranking quality* got substantially better while IoU at
   the chosen operating point got worse. That is the signature of a better map
   with a worse-transferring threshold, not of a degraded model.
2. **This run still used TILED inference.** It started before tiling was removed,
   so the old code was loaded in memory. It measures *halo-accumulated vs
   halo-cropped tiling* — **not** whole-slice inference. The whole-image number
   does not exist yet.
3. **The threshold is NOT the explanation — this was checked directly.** The new
   threshold 0.002 is exactly the floor of the grid `CNNRunner.fit` uses, which
   looks like the truncated-grid defect logged three times before. It is not.
   Re-selecting on the retrain's own saved val map with the grid extended to
   **1e-7** returns the same 0.002 and the same test IoU 0.29409; the val-IoU
   curve is flat over 5e-4…2e-2 with a true maximum at 0.002. Both operating
   points (0.006 pre-fix, 0.002 post-fix) sit on that plateau, so **the −0.0218
   is not a thresholding artifact.** *(This entry went through two wrong versions
   before this one: first claiming 0.002 was an interior point of a grid that
   isn't the one used, then accepting a review's inference that it was a clamp.
   The measurement above settles it.)* n=1 seed and nondeterministic MPS remain as
   confounds, so the delta is real but not cleanly attributable to the tiler.
4. **Weights were not saved** — the `ckpt_dir` wiring landed after this job
   launched — so re-scoring *this* model under whole-slice inference is not
   possible. That comparison needs one more training run, which will now persist
   its weights.

**Do not update the README headline from this number yet.** It is a
tiled-inference number for an inference path that no longer exists.

#### 🔑 The halo crop made the probability map ~9× MORE SATURATED — this explains the rest

Spotted in the regenerated POC figures (the probability panels look binary), then
measured on the 111.6 M-pixel test map:

| band | post-fix | pre-fix |
|---|---|---|
| < 1e-4 (hard 0) | **91.02%** | |
| 1e-4 … 0.01 | 0.72% | |
| 0.01 … 0.1 | 0.18% | |
| **0.1 … 0.9 (graded)** | **0.32%** | **3.00%** |
| 0.9 … 0.9999 | 0.51% | |
| > 0.9999 (hard 1) | 7.25% | |

The old tiler averaged each window's **zero-padded borders** into the output,
manufacturing intermediate values. Cropping the halo removed that averaging, so
the network's own near-binary outputs stand alone — the graded band shrank
**3.00% → 0.32%**. Three things fall out of this:

1. **It explains the flat threshold plateau.** With 98.3% of pixels outside
   0.01–0.9, no threshold in that range can move IoU much — which is why
   extending the selection grid to 1e-7 returned the identical 0.002 / 0.29409.
2. **It explains AUPRC ↑0.0755 with IoU −0.0218.** Ranking improved (AUPRC reads
   the ordering, including within the saturated masses — the maps are float16, so
   subnormals preserve ordering down to ~6e-8), while calibration became more
   extreme, so the achievable IoU at a single global threshold fell.
3. **[CONTRACT IMPACT] It undermines Stage 2 and constitution §9.** A learned
   confidence model and the free `|p − 0.5|` baseline both rank pixels by
   gradation; with 0.32% graded there is almost nothing to rank. This is a
   candidate explanation for the standing result that *no* confidence method beat
   a zero-parameter baseline by more than 0.0005 IoU — not that confidence
   modelling fails, but that this base map gives it nothing to work with.
   "Over-predict, then prune" needs a confidence gradient to prune along.

**Implication for the roadmap:** before Stage 2 can be judged, the base model
needs to produce a *calibrated* map — e.g. a loss that does not drive logits to
saturation, temperature scaling fit on val, or label smoothing. Also note
`save_prob` stores **float16**, which barely represents the 1e-5…1e-7 range where
this model's remaining signal lives; consider float32 for the val maps that
meta-models are fit on.

#### Pre-commit review found the halo crop was necessary but NOT sufficient

*Spec (condensed) — problem: the pre-commit reviewer claimed the seam fix was
incomplete. Approach: reproduce it independently before believing or dismissing
it. Acceptance: a measurement that distinguishes "seam fixed" from "tiled
inference correct".*

**Independently reproduced.** `UNet2D` normalises with **GroupNorm**, which
computes statistics over the spatial extent it is handed, so under tiling each
window is normalised by its own statistics. Measured with the *fixed* (halo-
cropping) tiler, untrained base=8 depth-4 `UNet2D`, non-constant 512×1024 image:

```
tile grid shifted 128 px, identical content:   mean|dp| 0.024   max 0.286
tiled vs one whole-image forward pass:         mean|dp| 0.022   max 0.331
largest 8 column jumps on a tile boundary:     1/8  (x=511 jump 1.1x median)
```

**The distinction that matters:** the *seam* — the boundary-localised artifact —
really was fixed by cropping the halo (1/8, 1.1× median). What survived was
**diffuse**: whole windows offset from one another. So "the seam is fixed" was
true; "tiled inference is correct" was not. A constant-image test cannot see
this because GroupNorm is degenerate on constant input — which is exactly what
the original ✅ verification used.

#### ❌ The whole-slice "fix" was WRONG and was reverted within the hour

`predict_proba` was switched to a single whole-slice forward pass. It passed every
check that was run on it — and **it destroyed trained-model quality.** Caught by
the second pre-commit review, from the repo's own logs: `ds2_f25_s0`, identical
config and seed, epoch for epoch:

| epoch | tiled | whole-slice |
|---|---|---|
| 8 | 0.4476 | **0.0000** |
| 11 | 0.4487 | **0.0004** |
| 15 | 0.4403 | **0.0075** |

0.0655 (its epoch-1 value) is the val foreground base rate — "predict everything
positive". By epoch 8 the map was empty at every grid threshold. **Cause:** these
nets train on 128×128 patches at `pos_frac=0.5`, and GroupNorm computes statistics
over whatever extent it is handed. A whole 962×1450 slice is ~110× larger and ~5%
foreground, so features land far outside the training distribution and the logits
saturate. Worse, the collapse happens *inside* `fit()` — the per-epoch val pass
uses `predict_proba`, so `best_state` and `best_thr` were being selected on
collapsed maps.

**Reverted.** Tiled inference (halo cropped) is the default again;
`predict_proba_whole` is retained, documented as unusable for these models. The
relaunched 10-h `data_scaling` job was killed 15 epochs in — it had written no
results file, so nothing corrupt reached disk.

**[LESSON] With GroupNorm plus patch training, the inference window size is a
model hyperparameter, not an implementation detail.** Matching the training
window's statistics matters more than eliminating tile-grid dependence, and the
residual grid dependence is now **accepted and documented** rather than fixed —
fixing it properly means changing the normalisation or the training window.

**[LESSON] My verification was the wrong kind.** I checked an *untrained* net's
probability differences and exact agreement with a hand-rolled forward pass. Those
cannot detect a decision-boundary collapse — the identical blind spot I had just
criticised the original constant-image test for. The check that would have caught
it: compare val IoU across the two paths **on the same saved weights**.

What the whole-slice path was verified to do correctly (all still true, and all
irrelevant to whether it should be used):

- output **exactly equals** a hand-rolled single forward pass (max |Δ| = 0.0);
- shapes/ranges correct at 512×1024, 521×1250, 257×33, 100×140, 962×1450, and
  for k=3 (2.5D);
- interior shift-sensitivity **~3e-4 mean / 0.01 max**, vs 0.024 / 0.286 tiled
  (~80× reduction). The residual is GroupNorm's *global* statistics changing
  because replicate-padding replicates different edge columns — inherent to
  GroupNorm, not to tiling. A residual measured with `np.roll` is dominated by
  the wrap discontinuity (mean 1.16e-02 within 32 px of the seam vs 2.7e-04 in
  the true interior), so that test overstates it.

**[HONESTY]** These magnitudes come from an *untrained* base=8 net on synthetic
data; the mechanism is architectural (it holds trained or not), but the size of
the effect on the real trained model is not yet measured. `run_families` now
persists weights (`ckpt_dir` defaults to `results/checkpoints`), which is what
makes that measurement possible without retraining — previously weights died
with the process.

**Leaderboard isolation.** `evaluate.run_and_score` takes `leaderboard=False`,
and `src/data_scaling.py` now passes it. Its 12 subset runs (25–100% of train, a
reduced 7.2k budget) were about to be appended to `results/leaderboard_all.json`,
whose regime string promises one split and which `fig_leaderboard` plots as a
single ranked chart — a `ds2_*` row there reads as a model that beat or lost to
`unet2d_s30000`. The same hazard applies to `pipeline_v2.stage2_sweep` on the v2
split and is **not yet fixed**.

**Scope note.** D3 said spec-and-commit-only. Sky reopened it to include these
code fixes once the GroupNorm finding landed, because the retrain in flight would
otherwise have measured the wrong inference path.

### 2026-07-29 — Prepare for a second meteorite

*Spec (condensed): the binding constraint is measured to be **data**, not
method — 19 validation chondrules give a bootstrap SE of ±0.0071 IoU and a
minimum detectable effect of ~0.014 IoU, which is larger than almost every
method effect we measured. Approach: make adding a meteorite a config change,
and fix the artifacts that would mislead that decision. Acceptance: a
data-agnostic pipeline with a passing self-test; weights that survive their
process; a scaling curve that is not confounded.*

- **Model weights now persist** (`CNNRunner.save_checkpoint` /
  `load_checkpoint`, plus a `ckpt_dir=` constructor argument that auto-saves at
  the end of `fit`). This clears the known issue that the sweep saved
  probabilities but not weights, which is why v4 unlabeled inference could
  never be produced. The **validation-selected threshold travels inside the
  checkpoint**, so a reloaded model cannot be scored at an arbitrary operating
  point. Verified by round-trip: reloaded predictions are bit-identical
  (`max |p1-p2| = 0.0`), the threshold matches exactly, and a mismatched `k`
  is rejected rather than silently accepted.
- **`src/data_scaling.py` — the corrected scaling test.** The existing
  `results/data_scaling.json` is **confounded and should not be quoted**: it
  reports 0.2421 / 0.2972 / 0.2593 / 0.2314 for 25/50/75/100% of the training
  slices — non-monotonic and *worst at 100% data*. Two defects: subsets were
  **contiguous** (so "25% of the data" was also "one region of the meteorite",
  varying difficulty and volume together), and there was **one seed per point**
  against a seed-to-seed scatter we never measured. The replacement draws
  subsets **at random across the whole train slab** (verified: a 25% subset
  spans slices 1–211 of 0–212, so region is decoupled from volume) and runs
  **3 seeds per fraction** at a fixed 7,200-step budget, reporting mean ± SD.
  Absolute IoU will sit below the 30k headline by construction — the curve is
  to be read for shape, not height. *Status as of the later 2026-07-29 entry:
  **never completed a run.** It died at run 1 of 12 having saved nothing, so
  `results/data_scaling_v2.json` does not exist and `data_scaling.json` is
  superseded in intent only, not in fact.*
- **🔴 TILED-INFERENCE SEAM BUG — found, confirmed, fixed.**
  `CNNRunner.predict_proba` tiled at `tile=256, halo=32` but **accumulated the
  halo instead of cropping it**. With stride equal to tile, overlap was confined
  to 64-px bands where the divisor jumped 1 → 2 → 4 discontinuously, and the
  averaged-in values were each window's *zero-padded* border pixels, computed
  with 32 px of context against a depth-4 U-Net's far larger receptive field.
  - *Found* by rendering real predictions (`poc_images`): a predicted chondrule
    stopped on a perfectly straight vertical edge (TP 14193 / FN 15723 / FP 1),
    and probability maps showed rectangular bands across round chondrules.
  - *Confirmed* independently: on every CNN-family artifact, **5–7 of the 8
    largest per-column jumps land exactly on a window boundary** (223, 287,
    479, 543, 735, 799, 991, 1055); the three GNN models, which bypass this
    tiler, score **0–1 of 8** with a different periodicity. That control is
    what makes the attribution airtight.
  - *Not* the classic bugs: accumulator/count indexing was correct, there was
    no coverage gap, and normalization is per-volume. It was the blend itself.
  - *Fixed* by discarding the halo so cores tile the slice exactly once
    (`cnt == 1` everywhere, asserted). Edge windows are also replicate-padded
    to a multiple of `PAD_MULT = 32`, because the last window (202 px) was not
    a multiple of the encoder's downsampling factor and the decoder padded it
    asymmetrically. Verified: a constant-logit model now returns an **exactly
    constant** map (spread 0.000e+00, was non-zero), and a real depth-4 U-Net on
    a constant image shows **0/8** boundary-aligned jumps.
  - **⚠️ Every CNN/transformer/modern-conv number in this repo predates the fix,
    including the 0.3168 headline.** They were measured with a defective blend
    and are unreliable *in an unknown direction* — the sign is not yet
    established. Because weights were never persisted, corrected numbers
    require **retraining**, not re-inference.
- **POC imagery** (`src/poc_images.py`, 4 figures) — rendered from the real
  saved probability maps of `unet2d_s30000` on held-out test slices, at the
  validation-selected threshold, not illustrated. Measured en route:
  **96.0% adjacent-slice prediction agreement** (the signal z-refinement is
  built on), and an **oracle** droppable fraction rising **60.0% → 89.1%** from
  20 to 6000 superpixels, against the learned drop rule's 58.5% — that gap is
  the hierarchy's remaining headroom, and it is large.
- **`src/pipeline_v2.py` — the data-agnostic pipeline.** `MeteoriteSource` /
  `build_multi_split` / `power_report` / `stage2_sweep` / `confidence_sweep` /
  `run_all`. Adding a meteorite is a config change. Self-test output:
  `531/531 pairs OK`, `13 straddler instance(s) resolved to majority slab;
  871822 voxels ignored; 0 chondrules span 2+ zones`, zone counts summing to
  the band, no slice index in two zones.
- **Two defects caught and fixed in that pipeline before it was trusted:**
  - *Straddler resolution was missing* — the first draft relied on the buffer
    alone. `split.py` measured chondrule z-extents reaching **70 band-slices**
    against a **12-slice** buffer, so chondrules could appear in two core
    zones. Now reuses the audited same-id-CC + majority-slab rule, with the
    post-hoc recount assertion. **P1 violation, caught before use.**
  - *Chondrule counting used plain binary CC*, which fuses touching chondrules
    — the same failure that once dropped 53.6% of positives. It reported
    **87** where the same-id rule reports **142** (a 39% shortfall). This feeds
    `power_report`'s `n`, so every minimum-detectable-effect figure would have
    been wrong. Now uses `split.instance_label`; the corrected counts are
    **train 81 / val 32 / test 29 = 142**, matching the known total exactly.
- **Free statistical power from slab placement.** The v2 split puts
  **32 chondrules in validation** versus the frozen split's 19, which lowers
  the minimum detectable effect from **0.0142 to 0.0109 absolute IoU** with no
  new data. It is a trade, not a free lunch: training chondrules fall from 99
  to 81, and the two splits' scores are not comparable to each other.
- **`hier_cnn` at a matched 30,000-step budget: test IoU 0.3037**
  (val 0.4460, threshold 0.000177, 41.5% of pixels alive), up from 0.2695 at
  8k. Against the plain U-Net's 0.3159 at the same budget that is **−0.012
  IoU — inside the ±0.014 noise floor**. The honest reading: the hierarchy
  buys a ~2.4× reduction in the area the CNN must process at no *measurable*
  accuracy cost. It is a compute optimization, and we cannot currently
  distinguish it from break-even.

### 2026-07-23 — Full model sweep, budget curve, confidence pipeline

**Headline: best test IoU 0.3168** (`unet2d_s30000` — a plain 2D U-Net trained
30,000 steps), vs 0.203 for classical ML and a 0.130 intensity floor.
Annotation ceiling ~0.92. **All numbers are "best-of-40, selection-biased"**:
the winner was chosen from a 40-model sweep, so its test score is the max of 40
noisy estimates (optimistic by roughly 0.02–0.03). Thresholds were always fit on
validation; the model itself was never tuned on test.

- **Budget beat architecture.** Same U-Net at 3k / 7.2k / 15k / 30k steps →
  0.2314 / 0.2834 / 0.3089 / 0.3159. Gains halve each doubling and have
  plateaued (+0.007 for the last doubling). The best of 40 *architectures* at a
  fixed budget reached only 0.2598. **Sweeping architectures was a worse use of
  compute than training one model longer.**
- **41 leaderboard rows** on the identical leakage-free split — **36 distinct
  architecture/budget configurations** swept, plus 4 recall-training loss
  variants and 1 hierarchy-filtered model. (The commit message says "37-model
  sweep"; the exact figure is 36 sweep configurations. Counts differ by which
  follow-up runs are included — this is the reconciled breakdown.) Two rows are
  **degenerate constant predictors, not successes**: `gnn_gat` and
  `matrix_specialist` both sit at IoU 0.0528 with recall 1.000. Families:
  U-Nets, attention/2.5D variants, 9 extra conv families, 4 from-scratch
  transformers (`vit_seg`, `swin_lite`, `segnext_lite`, `axial_unet`), 6 modern
  conv families, 7 GNNs, and a GNN→CNN cascade.
- **Over-predict-then-prune works, once the objective is right** (Sky's idea).
  `ErrorModel` regressing `|prob-gt|` is INVERTED for pruning: at a low
  threshold a true positive has residual ≈0.9 and a false positive ≈0.1, so
  "prune high error" deletes chondrules (measured: 0.331 on TPs vs 0.259 on
  FPs). `FPDetector` — *"is this predicted-positive pixel a false positive?"* —
  fixes it: `swin_lite` 0.2552 → 0.2675, `gnn_graphsage` 0.1783 → **0.2133**.
  Oracle prune ceiling is 0.810 from 0.146, so the mechanism has large headroom.
- **Cross-slice z-refinement works** (Sky's idea): +5.4% IoU for free. Pixels
  where z-neighbours disagree are wrong ~62% of the time vs ~6% where they
  agree — a strong zero-cost confidence signal.
- **Confidence bake-off: nothing earned its keep on the final model.** Best
  learned candidate (`hist_gb`, error task) reached pruned val IoU 0.455582 vs
  0.455048 for every free baseline (`|p-0.5|`, raw-prob uncertainty,
  z-agreement) — a **+0.0005** gain, i.e. the learned models marginally *won*
  but by nothing. The free baselines gained exactly 0.000000. Earlier wording
  ("the free baseline beat every learned model") was backwards. Learned
  confidence DID help a weak model substantially (`gnn_graphsage` 0.1783 →
  0.2133), so the honest reading is: confidence recovers slack in a poor model
  and adds nothing to a good one.
- **Asymmetric recall-oriented training HURTS**: symmetric dice 0.2314 vs
  tversky-recall 0.1529, focal-tversky 0.1603, dilated-target 0.2196. Train
  symmetrically and move the threshold instead.
- **Hierarchical superpixel cascade** (Sky's idea): **2.71× compute saving at
  99.9% chondrule retention** after fixing a merge-rule bug (mean-intensity
  agglomeration with no size term let one region absorb 98–99% of each slice;
  fixed with a Ward-style cost). Relaxing per-level recall from 1.0 to 0.999
  raises the saving from 1.66× to 2.70× for 0.05% of chondrule pixels. As a
  standalone segmenter it reaches only 0.1437 — at pixel level the rule
  degenerates to the intensity threshold — so it is an efficiency mechanism, not
  an accuracy one.
- **RETRACTED — there is no hard recall ceiling.** An earlier entry claimed the
  CNN assigns probability ~0 to ~42% of chondrule pixels and saturates at recall
  0.709. Recomputed from the model's own saved probabilities: only **0.40%** of
  chondrule pixels are exactly zero, and max recall as the threshold approaches
  0 is **0.996**. The "42%" was the fraction below **2e-4 — the floor of the
  threshold grid**, not of the model's output. This is the THIRD occurrence of
  the truncated-threshold-grid bug already logged twice below.
  *Honest replacement:* those pixels carry probabilities 3–4 orders of magnitude
  below the useful operating range, so recovering them costs nearly all
  precision — thr 1e-6 gives recall 0.961 but IoU 0.065; thr 1e-5 gives recall
  0.781, IoU 0.125; thr 1e-4 gives recall 0.592, IoU 0.282; the max-IoU point is
  0.3168. A *soft economic* ceiling, not a hard one.
- **Surviving finding: the model's confidence falls as chondrules get brighter.**
  Mean predicted probability over TRUE chondrule pixels by CT intensity:
  0.739 (0–35) → 0.655 (35–45) → 0.374 (45–55) → 0.130 (55–70) → 0.085 (100+).
  Threshold-free, so unaffected by the grid bug.
  The cause is UNIDENTIFIED. An earlier explanation — that bright pixels are
  mostly not-chondrule, so the model learns "bright ⇒ matrix" — is **refuted**:
  the global not-chondrule:chondrule ratio is 17.9:1, so the 10:1 ratio in the
  CT 55–70 band means bright pixels are actually **1.7x ENRICHED** in chondrules
  relative to base rate. A brightness-only Bayes rule would rank them ABOVE the
  prior; this model ranks them far below it. Also note the trend is not strictly
  monotone and the darkest bin (~2% of chondrule pixels) does not replicate
  across splits — the robust statement is that above CT ≈ 40 the model's output
  on true chondrule pixels falls steeply (test 0.655 → 0.374 → 0.129 → 0.085;
  val 0.600 → 0.409 → 0.076 → 0.030).
- **Generalization-limited, not capacity-limited**: train IoU 1.000 vs ~0.32
  held-out, budget plateaued, 40 architectures clustered, augmentation and
  weight decay both unhelpful. N=1 meteorite, ~142 chondrules, adjacent slices
  ~0.83 IoU similar — the effective sample size is far below 213 slices.
- **Model selection is unreliable here**: a bootstrap over validation slices
  gives a 95% CI of [-0.013, +0.015] on the val-IoU gap between the top two
  models (P(better) = 0.50), while their test gap is 0.057.
- **New modules**: `zones` (cached split), `evaluate` (one scoring path),
  `models_extra` / `models_transformer` / `models_modern` / `models_gnn_variants`,
  `graph_cache`, `combine` (z-refine, ConfidenceModel, ErrorModel, FPDetector,
  prune), `confidence_bakeoff`, `cascade`, `hierarchy`, `autoresearch`,
  `run_families` / `run_final` / `run_overnight` / `make_figures`.
- **Bugs caught that would have produced confidently wrong conclusions**: GNN
  node features unstandardized (~20,000× scale spread → constant outputs);
  budget-run name collisions overwriting both leaderboard rows and probability
  files; truncated threshold grids (twice) making a "recall 1.00" experiment
  actually measure 0.69; queue deadlocks; dead `pos_weight` / `invert_target`
  knobs; and the hierarchy merge-rule defect above.
- **Final pipeline on the locked model: test IoU 0.3214** (`unet2d_s30000` +
  FP-prune + z-smoothing), vs 0.3168 for the model alone.
- **NEGATIVE, on the final model — both must be reported.** Constitution P9
  requires the head-to-head, it was run, and both lost:
  **z-refinement gives 0.2972 vs 0.3168 baseline (−6.2%)**, and **FP-prune gives
  0.3131 vs 0.3168**. The earlier "+5.4%" and "prune works" claims were measured
  on `swin_lite` and `gnn_graphsage` — models the roadmap itself calls
  disposable. On the model actually shipped, neither helps in isolation; only
  the stacked combination (0.3214) exceeds the baseline.
- **The FP-prune recall sweep is degenerate**: all four targets (0.80/0.90/0.95/
  0.98) resolved to the same threshold 2e-4 with byte-identical metrics — the
  same grid-floor cause as the retracted ceiling claim. It is a single point,
  not a sweep.
- **Two-model veto (Sky's is/is-not idea): 0.2750 test IoU vs 0.3168 reference —
  it loses on IoU**, but raises precision to 0.463 from the reference's 0.4147 (+0.049; the 0.332 quoted
  earlier was `hrnet_lite`'s, not this model's). About 60% of that gain is
  reproduced by simply thresholding the same model to a smaller prediction
  budget (precision 0.4435 at a matched budget), so only ~+0.020 is attributable
  to the matrix specialist — whose own output spans just 0.705–0.996 at recall
  0.403. The matrix specialist genuinely vetoes false positives; it over-vetoes.
  Probability-averaging the two gives 0.3162, indistinguishable from the base.
- **Not delivered, disclosed rather than glossed:** (a) v4 inference on the ~130
  unlabeled slices was NOT produced — the sweep persisted probability maps but
  not model weights, so the winner must be retrained to run it;
  `results/unlabeled.json` records only the slice list. (b) The straddler-drop
  ablation promised in constitution P1 has a single arm in
  `results/ablation.json` — nothing was compared, so the majority-assignment
  policy remains unquantified.
- **Reproducibility gap, disclosed:** `results/hierarchy_fixed.json` (the 2.71x
  headline) has **no producer script in the repo** — it was generated
  out-of-band while iterating on the merge-rule fix. The committed
  `src/run_overnight.py:run_hierarchy` still writes the superseded 1.58x
  `results/hierarchy.json`. The 2.71x number should be treated as
  not-yet-reproducible until that stage is rewired.

### 2026-07-23 — Karpathy-style autoresearch (Phase 0 + search rounds)
- **Phase 0 (sanity) — the pivotal finding.** A U-Net overfits 4 fixed patches to
  **train IoU 1.000** (dice@3e-4; bce/tversky ≈0.997) and the valid-mask
  provably neutralizes every loss (all four → 0.0000 when nothing is valid).
  So the earlier ~0.19 was **not** a bug, not capacity, and not step count:
  **train 1.000 vs held-out 0.19 is a generalization gap.** This invalidated the
  "just train longer" plan and redirected the search to regularization.
- **Round 1 (one variable, ranked on FULL val).**
  `p1_ctrl_b32` (no augmentation) **val IoU 0.3874** vs `p1_aug_b32`
  (augmentation) **0.2021**.
  - **LR fix alone was worth ~57%**: val 0.246 → 0.387 from `lr=3e-4` + cosine
    decay (both prescribed by Phase 0), with no architecture change.
  - **Augmentation HURT, nearly halving IoU.** Hypothesis under test in round 2:
    the bundle included **intensity jitter**, and intensity is the discriminative
    signal (chondrules ≈50 vs matrix ≈32 CT), so jitter corrupts the main
    feature while flips/rotations should be harmless. `augment="geom"` now
    isolates geometry (verified: constant intensity across draws).
- **Phase 3 (long budget, 7,200 steps) — CNNs overtake classical ML.**
  `long_p1_ctrl_b32` **test IoU 0.2834** (Dice 0.442, P 0.360, R 0.571; val
  0.4508) vs the classical-ML champion **0.203** — a **+40%** improvement, and
  the first learned model to beat texture features on the held-out slab.
  `long_p1_geom_b32` reached test 0.2329 (val 0.4375).
- **Augmentation: budget artifact confirmed.** At the search budget the
  augmentation gap was 0.165 val IoU; at 7,200 steps it closed to **0.013**
  (0.4375 vs 0.4508). So augmentation was never orientation-sensitivity — it
  simply needed steps to amortize. It still does not win, so the no-aug recipe
  stands, but the earlier round-2 reading is now superseded.

  **[AMENDED 2026-07-30 — this conclusion was drawn on VALIDATION only, and TEST
  disagrees.]** The 30,000-step arm continues the val trend exactly as predicted
  (0.165 → 0.0133 → **−0.0157**, i.e. augmentation overtakes on val), but on TEST
  the gap does not close at all: **−0.0505 at 7.2k and −0.0501 at 30k**, both
  threshold-robust. Within the 30k run the budget reading also still holds — the
  control peaks at epoch 21/60 and decays, while the augmented arm peaks at 43/60
  and is still near its optimum at the end. So "augmentation simply needed steps
  to amortize" is supported by every validation-side measurement and refuted by
  every test-side one. **The hypothesis is CONTESTED, not confirmed.** Do not
  quote this bullet without the amendment. n=1 seed throughout, on both sides.
- **Threshold grid was truncated (fixed).** Both long runs selected exactly
  `thr=0.05`, the grid minimum — a parameter pinned to its boundary, meaning the
  true optimum lay outside the search and those two test numbers are likely
  understated. Grid extended down to 0.002; dice loss under ~95% background
  genuinely pushes the useful operating point very low.
- **Round 2 — the jitter hypothesis was WRONG.** Ranked on full val:
  `p1_ctrl_b32` **0.3874** > `p1_ctrl_wd_b32` 0.3640 > `p1_geom_b32` 0.2221 >
  `p1_aug_b32` 0.2021. Removing intensity jitter recovered only **0.02** of the
  0.185 gap, so the *geometric* flips/rotations cause nearly all the damage —
  not the jitter. Most likely a budget artifact (augmentation slows convergence
  and 1,500 steps is too short to amortize it) rather than orientation
  sensitivity; tested directly by a long-budget run rather than assumed.
  Weight decay also did not help. **Best recipe so far: no augmentation,
  lr=3e-4 + cosine decay, dice, base=32.**
- **GNN superpixel ceiling measured before training** (`n_segments` vs the
  "every superpixel gets its majority label" oracle, real val slices):
  1200 → **0.771**, 3000 → 0.837, 6000 → **0.878**, at essentially identical
  cost (~9.5 s/slice, dominated by feature extraction, not SLIC). Default
  raised to 6000 so the GNN is not capped at 0.77. Note this is a hard ceiling
  for any superpixel method, below the 0.92 annotation ceiling.
- **Confidence-model memory fix:** the per-pixel feature stack was ~6.1 GB for a
  real zone (violating the memory rule); now built in z-windows with a halo,
  verified **bit-identical** to the full-volume path (max diff 0.0) at ~600 MB
  peak. Also documented that risk-coverage IoU is legitimately NaN at low
  coverage (the most-trusted pixels are background-dominated → empty union).
- **Infrastructure:** split cached once (`src/zones.py`) — search workers load
  train+val only and never read test, cutting RAM from ~30 GB (4 workers) to
  ~12.6 GB (2 workers) after the earlier run swamped the machine.
  Added `src/evaluate.py` (one train→val-threshold→score-test-once path for
  every family), `src/run_families.py`, and `results/excluded_models.json`
  (SegFormer / TransUNet / pretrained ResNet-UNet / SAM — declared with reasons,
  no silent caps).
- **New model code:** `src/models_extra.py` — 9 additional NN families
  (pixel-MLP, patch-CNN, dilated CNN, UNet++, FCN-8, SegNet, DeepLabV3+,
  autoencoder, ConvLSTM-over-z); shape self-test passes, including
  non-power-of-two tiles. Untrained as of this entry.

### 2026-07-23 — v2 sweep (Tier A+B), audits & evaluation ceiling
- **Adversarial audits (2 independent verifiers).** Eval/metrics **confirmed
  honest & leakage-free**; split **confirmed leakage-free at the label level**
  (the one recycled id 22 shown to be 3 distinct chondrules). Reconciled
  constitution / spec / docstring to the real mechanism (small buffer +
  instance-level straddler resolution + independent verification); added a
  zone-separation guard; relabeled the intensity-baseline ECE as a
  pseudo-probability metric.
- **Tier A+B leaderboard** (leakage-free split; threshold tuned on val, test
  scored once; foreground IoU): hist_gboost **0.203**, extra_trees 0.201,
  random_forest 0.195 (classical-ML on 19 texture features) vs. intensity
  threshold floor **0.130**. Texture ~1.5× the floor by trading flood-recall for
  precision (0.13→0.25); linear/NB still flood.
- **Annotation ceiling** (answers "how close to 1 is possible"): ±1px-boundary
  IoU ≈ **0.92**, ±2px ≈ 0.85; adjacent-slice (copy-neighbor) IoU ≈ 0.83. →
  practical IoU ceiling ~**0.85–0.92**, not ~1.0. Current best 0.20 leaves real
  headroom for CNNs.
- **Tier C scaffolding** (`src/models_cnn.py`): 2D / 2.5D / attention U-Nets;
  Dice/Tversky/focal (valid-masked); train-only patches; val-only early-stop;
  tiled inference. Shape-verified, **not yet trained**.
- **v2 spec expanded**: two-model combinations (build all 3, keep val-winner),
  recall-optimized operating point, ceiling, straddler-drop ablation; GNN queued.
- **Pre-commit gate (reviewer + overseer) applied.** Fixed: `torch` added to
  requirements; ceiling made reproducible (`src/ceiling.py` +
  `results/annotation_ceiling.json`); leaderboard regime reconciled (audited, not
  "provisional"); buffer doc corrected to 48 original slices; v1 pseudo-ECE now
  uses a fixed /255 scale. **Deferred to the Tier C stage** (untrained-CNN
  training quality — documented, not lost): 2.5D train-seam channel mixing,
  `pos_weight` threading, focal α, and a regression-test suite. Core split &
  metrics were audited leakage-free with no changes required.
- **Scope expansion (Sky):** sweep **all** neural-network families (not only the
  named CNN/GNN), and add a **learned confidence model** trained on the CV
  models' own mistakes to flag where predictions are trustworthy (cf. the REE
  logD separation-confidence approach).

### 2026-07-23 — Project bootstrap (SDD Phase 0–1)
- **Discovery.** Identified the dataset as a micro-CT scan of **ALH 83010**:
  531 raw slices (1450×962, 8-bit) paired 1:1 with a hand-labeled instance
  segmentation. Verified from pixels that labels are **individual chondrule
  instances** (compact blobs persisting across z), not mineral classes; positive
  target collapses to binary **chondrule vs. matrix**. Chondrules are denser
  (brighter) than matrix in CT (mean ≈ 50 vs. 32). Annotation covers a
  **central band ≈ slices 100–496**; ~130 end slices are unlabeled. `BP.CSV` is
  unrelated (hydrocarbon table) and ignored.
- **Constitution** written (`memory/constitution.md`, v0.1): mission, scope,
  data facts, local-Mac stack, 8 non-negotiable principles (leakage-free z-split,
  touch-test-once, honest negatives, baseline-first, joint metrics, N=1 honesty,
  reproducibility, SDD-every-change), and a v0→v4 roadmap.
- **Decisions (Sky, MCQ):** task = binary semantic segmentation with per-pixel
  probability; metric = foreground IoU; empty ends **excluded as unlabeled**;
  test split = **contiguous z-slab + buffer**; compute = **local Mac only**.
- **Spec v0/v1** written (`specs/2026-07-23-chondrule-segmentation/spec.md`):
  evaluation harness + frozen leakage-free split + classical baseline; awaiting
  sign-off.
