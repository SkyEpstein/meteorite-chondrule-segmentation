# Changelog

All notable decisions and milestones for the Meteorite Chondrule Segmentation
project. Honest numbers only; evaluation regime stated with every metric.

## [Unreleased]

### KNOWN ISSUES IN THIS COMMIT (found by pre-commit review, read first)

- **ALL CNN-FAMILY TEST SCORES PREDATE THE TILED-INFERENCE FIX
  (2026-07-29).** The inference tiler accumulated its halo instead of cropping
  it, producing axis-aligned seams in every CNN/transformer/modern-conv
  probability map, including `unet2d_s30000` (0.3168). Cropping the halo removed
  the seams but was NOT sufficient: GroupNorm normalises each window by its own
  statistics, so the residual is diffuse rather than seam-shaped and a
  constant-image test cannot detect it. Inference is now WHOLE-SLICE. No model has
  been retrained yet, so **every leaderboard IoU in this repo is unreliable in an
  unknown direction.** A headline retrain is in flight with no result yet. The
  authoritative account is the `predict_proba` docstring in `src/models_cnn.py`;
  the 2026-07-29 entry below predates the whole-slice change and describes only
  the halo-crop step.

- **`+10% from z-context` is UNBACKED.** `results/probs/unet25d_k5.npy` is
  actually the 30,000-step run's output (bare-name collision, fixed in code but
  after these files were written). Re-scoring it gives **0.2154**, not the
  leaderboard's 0.2547. The surviving artifact points the OPPOSITE way. Treat
  every "2.5D beats 2D at equal budget" statement as unverified.
- **The 3k budget anchor has no artifact.** `probs/unet2d.npy` is byte-identical
  to `probs/unet2d_s30000.npy`. The 0.2314 value is independently reproduced by
  `rt_dice_ctrl` (0.23142 @ thr 0.25), but the named file is wrong, and
  `pick_winner` globs both names so one model enters the ranking twice.
- **`results/probs/` has no file for `unet2d_s15000`, `unet25d_k5_s15000`,
  `unet25d_k5_s30000`**, so the 15k point (0.3089) quoted in the docs cannot be
  independently checked.
- **`final_pipeline.json` predates the threshold-grid fix in `metrics.py`.** Under
  the committed code `unet2d_s30000` val-selects thr 0.0076 and scores
  **0.31601** (not 0.31676). Every downstream number in that file (the -6.2%
  z-refine, 0.3131 prune, 0.3214 stack, and the bake-off table) was computed at
  thr 0.05. Re-run `python -m src.run_final` to regenerate.
- **`0.3214` was selected on TEST.** The prune+z stack is never scored on
  validation, so it is chosen among four test-scored options. It is a reported
  configuration, not a val-selected result.
- **The confidence bake-off ran at recall 0.601, not the stated 0.95.**
  `recall_threshold` silently falls back to the grid floor when a target is
  unreachable and returns no flag.
- **`results/ablation.json` and `results/unlabeled.json` used `hrnet_lite`,
  selected from the TEST-sorted leaderboard**, which is the exact leakage
  `best_model_by_val` was added to prevent. Both predate that fix.
- **`results/ablation.json` has one arm only.** The promised
  drop-straddlers-entirely comparison was never computed.
- **v4 unlabeled inference was never produced.** The sweep saved probabilities
  but not model weights, so there is nothing to run on the 130 unlabeled slices.
  *(Fixed going forward by `CNNRunner.save_checkpoint`, 2026-07-29, but the
  already-swept models are gone and would need retraining.)*
- **`results/data_scaling.json` is confounded**: contiguous subsets and one
  seed per point, and it is non-monotonic and worst at 100% data. Nothing has
  superseded it yet: `src/data_scaling.py` cannot run (see below) and
  `results/data_scaling_v2.json` does not exist. Do not quote it.
- **`tests/` is empty.** ~5,000 new lines ship with only synthetic `__main__`
  smoke tests, none run automatically.
- **`src/data_scaling.py` cannot run: it crashes before training.** Line 141
  passes `leaderboard=False` to `evaluate.run_and_score`, which takes
  `(runner, data, root, extra, save_prob)` and has no such parameter and no
  `**kwargs`, so the call raises `TypeError: got an unexpected keyword argument
  'leaderboard'` on run 1 of 12. This is why the job died with zero results
  saved, and relaunching it unchanged will die the same way. Deleting the kwarg
  alone is not the fix: the comment above it exists because `ds2_*` rows must
  stay out of `results/leaderboard_all.json`, and `run_and_score` appends
  unconditionally (`src/evaluate.py:125`), so the suppression still needs
  implementing. Left unfixed deliberately in this commit.
- **The in-flight retrain will still discard its weights.** `save_checkpoint`
  only fires when `CNNRunner.ckpt_dir is not None` (`src/models_cnn.py:603`), and
  NO caller anywhere passes `ckpt_dir`: `run_families.py`, `sweep.py`,
  `run_final.py` and `pipeline_v2.py` all construct runners without it. So the
  claim that checkpointing "clears the known issue that the sweep saved
  probabilities but not weights" is true of the mechanism but false in practice,
  and v4 unlabeled inference stays blocked until a caller passes it.
- **Whole-slice inference does not remove GroupNorm's extent sensitivity, it
  moves it.** GroupNorm statistics span the spatial extent it is handed. Training
  draws 128x128 patches (`patch=128`) while `predict_proba` now runs a full
  962x1450 slice, so the train-to-inference extent gap is now LARGER than the old
  tiler's was. Whole-slice output cannot depend on a tile grid, which was the
  point, but part of any retrain delta may be this normalisation-extent shift
  rather than the tiler correction alone. Not yet quantified.
- **Machine-readable artifacts carry no tiler marker.** The `regime` strings in
  `results/leaderboard_all.json` and `results/final_pipeline.json` still describe
  a clean evaluation regime, so any programmatic consumer, including
  `fig_leaderboard`, sees all 41 rows as valid. The tiler disclosure exists only
  in prose.
- **`results_v2/split_multi.json` misdescribes its own counting method.** Its
  `regime` field says 3D counts came from `scipy.ndimage.label` on mask AND
  valid, but `count_chondrules_3d` uses `split.instance_label` (same-id), which
  is the corrected method; plain binary CC gives 87 where the same-id rule gives
  142. The field also still contains a literal em dash that the code no longer
  emits, so the artifact is not byte-reproducible from the committed source.
- **`results/figures/data_scaling.png` renders the confounded curve** that this
  block says must not be quoted, and has no producer in the repo: the new
  `fig_data_scaling` reads `results/data_scaling_v2.json`, which does not exist,
  so it silently returns.
- **The 96.0% adjacent-slice agreement figure below is the metric
  `src/poc_images.py` explicitly refuses to report**, because at ~90% background
  a constant all-background map already scores ~90%. The honest measure is
  slice-to-slice IoU over the union of positives: truth 85.2%, predictions 65.9%.
  Prefer those.
- **The four `poc_*.png` figures are deliberately NOT in this commit.** They were
  rendered from pre-fix probability maps, so their probability panels show the
  tiler's rectangular seam bands, and their titles predate the style pass.
  Regenerate them after the retrain.
- **Latent bug:** `run_families.py` / `cascade.py` do `bool(rec["aug"])`, so an
  `aug="geom"` recipe would silently train with intensity jitter. Not triggered
  today (winning recipe has `aug: false`).



### 2026-07-29: Documentation rewrite, style guide, and attribution cleanup

A prose and documentation pass, committed together with in-progress work from the
primary worktree that the rewrite had to describe honestly: whole-slice inference,
model checkpointing, `pipeline_v2.py`, `data_scaling.py`, `poc_images.py` and a new
spec. The prose pass itself trained no model, recomputed no metric and changed no
number. The folded-in work does change inference behaviour and adds one new
measurement (`results/hier_then_cnn_s30000.json`).

**Problem.** The public docs read as machine-generated: 176 em dashes across the
markdown, emoji section headings, arrows and middot separators, and heavy bold.
The README opened with three stacked warning blocks before saying what the
project does. The GitHub repo had no description. The tip commit carried an AI
co-author trailer.

**Approach.** Rewrote the README into reading order: what this is, what it
measured, how to read the numbers, why the evaluation is trustworthy, preparing
for more data, how to run it, and what is and is not finished. Removed every em
dash, en dash, Unicode minus, arrow, middot and emoji from all tracked markdown
and from 26 Python files (comments, docstrings, human-readable message strings
and matplotlib titles), then regenerated the four figures whose titles changed.
Codified the rules in `memory/style.md` and added constitution principles 13
(writing style) and 14 (attribution and commits), taking the constitution to
version 0.4. Set the repo description. Rewrote the tip commit to drop its
`Co-Authored-By` trailer and force-pushed `main`, with the tree byte-identical
and author and dates preserved.

**Acceptance.** Zero em dashes, en dashes, Unicode minus, arrows, middots or
emoji remain in tracked markdown or Python. `python -m compileall src/` passes on
every module, and a smoke test imported 11 of them cleanly; the rest were
syntax-checked but not imported. For the style pass specifically, the multiset of
numeric tokens was compared before and after in every file it touched and was
unchanged, so no reported number moved by a recast. That check does not extend to
the folded-in work, which legitimately adds numbers. No caveat, retraction,
withdrawal or status label was removed.

**Two content fixes, not just style.** The README's results table asserted "+10%
over 2D at equal budget" for `unet25d_k5`, a claim the KNOWN ISSUES block above
already calls UNBACKED and the roadmap marks CLAIM WITHDRAWN; the front page now
carries the withdrawal. The README also now opens with the tiled-inference bug
instead of presenting 0.3168 as a current result, and marks every affected row,
so a reader cannot quote the headline without the caveat. A first draft of that
section described the fix as halo-cropping and cited the constant-image test as
proof; a review caught that `predict_proba` documents cropping as insufficient and
that test as unable to detect the GroupNorm residual. The section was rewritten
from the code, and the ROADMAP bug log now carries the tiler entry it was missing.

### 2026-07-29 (later): Complete-model contract; scaling-job triage; headline retrain

**Constitution §12: "the complete model" is now one named thing (Sky's call).**
From here on it means the ordered composition of four pluggable stages: best base
segmenter (high-recall operating point), then hierarchy filter, then learned
confidence model, then z-layer neighbor confidence. The headline metric is
end-to-end test foreground IoU against the plain-U-Net-at-matched-budget
baseline, and every stage is independently ablatable so a stage that does not
earn its place is visible. Recorded as constitution §12 and
`specs/2026-07-29-complete-model-contract/spec.md`.

**[HONESTY] This is an architecture contract, not a performance claim.** At n=19
val chondrules the MDE is 0.0142 IoU. The hierarchy measured -0.012 vs the plain
U-Net, inside that floor, and is justified on compute (2.41× at target recall
0.999), not accuracy. The stages cannot currently be told apart from
break-even; the larger dataset is what makes the measurement possible.

**Scaling-curve triage: two jobs were in flight, and the weaker one was stopped.**
The `src/data_scaling.py` run (4 fractions × 3 seeds × 7,200 steps) had died at
run 1 of 12 with zero results saved, so `results/data_scaling_v2.json` has never
existed. A separate scratchpad job (`data_scaling2.py`, PID 39992) was still alive
after 71 min running a single-seed 15,000-step variant. It was killed, on
these grounds: (a) one seed per fraction, so a trend cannot be separated from
seed scatter (bootstrap SE ±0.0071 IoU); (b) it bypasses
`evaluate.run_and_score` and the shared leakage checks; and (c) it writes a JSON
list to the same path `src/data_scaling.py` reads with `.get("runs", [])`,
reproduced as `AttributeError: 'list' object has no attribute 'get'`, which would
have crashed the good module on its next start. The 12-run module is relaunched
after the retrain, serialized deliberately because both jobs share one GPU.

**[CORRECTION: the record of that kill was initially wrong.]** It was first
reported here, and to Sky, that the killed job "had written nothing". **That is
false**, found by the pre-commit overseer pass. The job's progress log was
`results/data_scaling_v2.log`, not the `logs/data_scaling_v2.log` that was
inspected, and it shows one completed data point:

```
[ds25] FULL-val IoU=0.4488 frozen thr=0.00
 25% ( 53 random slices): test IoU=0.2863 val=0.4488
```

with the 50% run 9/30 epochs in when it was killed. Two consequences: the claim
that it "carries the exact confound `src/data_scaling.py` was written to remove"
was also wrong, because `src/data_scaling.py` names **two** defects (contiguous
subsets *and* one seed), and `53 random slices` shows this job had already fixed
the contiguity confound; only the single-seed defect remained, at a larger 15k
budget. The kill still stands on grounds (a) to (c), but ~71 min of compute
produced a real measurement. That measurement was never written to JSON (no
`data_scaling_v2.json` exists anywhere on disk), so its only record is the log,
which `.gitignore` excludes via `results/*.log`. It is therefore preserved in this
commit as `results/prefix_tiler_backup/killed_data_scaling2_job.log`.

**[HONESTY] Do not quote `test IoU=0.2863` as a scaling-curve point.** It is
n=1 seed, at a 15k budget, measured under the pre-fix tiler, and it has no
companion points to form a curve.

**Headline retrain under the fixed tiler: in flight, result not yet in.**
`python -m src.run_families run 0 60 500` trains `unet2d_s30000` for 30,000 steps.
Pre-fix artifacts were copied to `results/prefix_tiler_backup/` first
(leaderboard, `final_pipeline.json`, the 213 MB probability map, and the four POC
figures) because `run_and_score` overwrites the leaderboard row and the very
probability map the POC figures were drawn from.

**[HONESTY] Which baseline the delta will be quoted against.** Three different
"headline" numbers exist for this model: 0.3168 (README, predates the
threshold-grid fix *and* the tiler fix), 0.31601 (committed code, thr 0.0076,
predates the tiler fix), and 0.31587 (`leaderboard_all.json`, thr 0.006). The
apples-to-apples comparison for the retrain is against the reproducible
~0.3159 to 0.3160, not the published 0.3168; both will be reported with the sign.

#### Adversarial verification of the contract: one claim retracted, four gaps found

The first draft of the spec asserted that the z-neighbor stage was not wired
into `run_all`. **That was false and is retracted.** It is step 4 of
`confidence_sweep` (`pipeline_v2.py:573-574`), and `combine.z_agreement` is live in
two further places: a bake-off free baseline (`confidence_bakeoff.py:120`) and
feature 2 of `ConfidenceModel.build_features` (`combine.py:250`). All four stages
were already wired. What is actually missing is the composite:

- **`run_all` has never been executed.** There is no caller in the repo,
  `__main__` is plumbing-only by its own comment, and `results_v2/` holds only
  `split_multi.json`.
- **The three post-base treatments never chain.** Bake-off winner (`:520`),
  prune (`:555`) and z-refine (`:574`) each read the same base map `pt`.
- **Stages 0 and 1 are fused.** `stage2_sweep` only builds `HierMaskedRunner`
  (`:454`), so no hierarchy-off arm exists and the ablation cannot be computed.
- Hierarchy thresholds were checked and are **genuinely leakage-free** (val-fit,
  frozen, decision rule reads raw only, see `hierarchy.py:483-493`).

#### `recall_threshold` cannot reach any recall target: **constitution §9 is not running**

`THR_GRID` floors at 2e-4 (`combine.py:34-38`) and `recall_threshold` **silently**
returns that floor when a target is unreachable (`combine.py:551-553`). Measured on
the real validation zone:

```
unet2d_s30000     targets 0.80/0.90/0.95/0.98 -> thr 2e-4, val recall 0.6014  (all four identical)
hier_cnn_s30000   targets 0.80/0.90/0.95/0.98 -> thr 2e-4, val recall 0.5531  (all four identical)
```

A grid-floor defect, **not** model saturation: recall at `prob > 1e-7` is 0.985
(unet2d) / 0.970 (hier_cnn). So the four prune targets in
`results/final_pipeline.json` are one row repeated (0.3011 to 0.3131 ×4),
`BAKEOFF_TARGET_RECALL = 0.95` actually ran at 0.60, and **"over-predict, then
prune" (constitution §9) has never actually executed.** The CHANGELOG already
recorded the symptom; the cause is that the earlier fix moved the floor
0.05 to 0.0002, still ~3 orders too high. Secondary defects recorded in the spec:
unmatched threshold grids across arms, gains that conflate treatment with
threshold change, ~12 test scorings flagged against a single-comparison MDE, `_gain`
treating NaN as measurable, `out_dir` half-honored, and no collision guard in
`alive_index`.

**Sky's decisions:** (D1) extend `THR_GRID` to ~1e-7 log-spaced and make the
fallback raise or flag rather than fail silently. This invalidates the prune
and bake-off numbers in `final_pipeline.json`, which must be regenerated.
(D2) chain order is z-refine the probability map first, then prune, so no stage
binarizes early and the final threshold stays val-selected, deliberately *not*
reusing `run_final.py`'s `prune_plus_z`, whose 0.3214 was test-selected with a
hard-coded 0.5 cut. (D3) this session is spec + commit only: the composite and
the D1 fix are the next session's first task, because building the contract on a
base score that is currently being re-measured would have to be redone.

### 2026-07-29: Prepare for a second meteorite

*Spec (condensed): the binding constraint is measured to be **data**, not
method. 19 validation chondrules give a bootstrap SE of ±0.0071 IoU and a
minimum detectable effect of ~0.014 IoU, which is larger than almost every
method effect we measured. Approach: make adding a meteorite a config change,
and fix the artifacts that would mislead that decision. Acceptance: a
data-agnostic pipeline with a passing self-test; weights that survive their
process; a scaling curve that is not confounded.*

- **Model weights now persist** (`CNNRunner.save_checkpoint` /
  `load_checkpoint`, plus a `ckpt_dir=` constructor argument that auto-saves at
  the end of `fit`). This clears the known issue that the sweep saved
  probabilities but not weights, which is why v4 unlabeled inference could
  never be produced. The validation-selected threshold travels inside the
  checkpoint, so a reloaded model cannot be scored at an arbitrary operating
  point. Verified by round-trip: reloaded predictions are bit-identical
  (`max |p1-p2| = 0.0`), the threshold matches exactly, and a mismatched `k`
  is rejected rather than silently accepted.
- **`src/data_scaling.py`, the corrected scaling test.** The existing
  `results/data_scaling.json` is **confounded and should not be quoted**: it
  reports 0.2421 / 0.2972 / 0.2593 / 0.2314 for 25/50/75/100% of the training
  slices, non-monotonic and *worst at 100% data*. Two defects: subsets were
  contiguous (so "25% of the data" was also "one region of the meteorite",
  varying difficulty and volume together), and there was one seed per point
  against a seed-to-seed scatter we never measured. The replacement draws
  subsets at random across the whole train slab (verified: a 25% subset
  spans slices 1 to 211 of 0 to 212, so region is decoupled from volume) and
  runs 3 seeds per fraction at a fixed 7,200-step budget, reporting mean ± SD.
  Absolute IoU will sit below the 30k headline by construction, so the curve is
  to be read for shape, not height. *It is NOT running: it crashes before training
  on a bad kwarg (see KNOWN ISSUES). The old file is not yet replaced.*
- **TILED-INFERENCE SEAM BUG: found, confirmed, fixed.**
  `CNNRunner.predict_proba` tiled at `tile=256, halo=32` but accumulated the
  halo instead of cropping it. With stride equal to tile, overlap was confined
  to 64-px bands where the divisor jumped 1 to 2 to 4 discontinuously, and the
  averaged-in values were each window's *zero-padded* border pixels, computed
  with 32 px of context against a depth-4 U-Net's far larger receptive field.
  - *Found* by rendering real predictions (`poc_images`): a predicted chondrule
    stopped on a perfectly straight vertical edge (TP 14193 / FN 15723 / FP 1),
    and probability maps showed rectangular bands across round chondrules.
  - *Confirmed* independently: on every CNN-family artifact, 5 to 7 of the 8
    largest per-column jumps land exactly on a window boundary (223, 287,
    479, 543, 735, 799, 991, 1055); the three GNN models, which bypass this
    tiler, score 0 to 1 of 8 with a different periodicity. That control is
    what makes the attribution airtight.
  - *Not* the classic bugs: accumulator/count indexing was correct, there was
    no coverage gap, and normalization is per-volume. It was the blend itself.
  - *Fixed* by discarding the halo so cores tile the slice exactly once
    (`cnt == 1` everywhere, asserted). Edge windows are also replicate-padded
    to a multiple of `PAD_MULT = 32`, because the last window (202 px) was not
    a multiple of the encoder's downsampling factor and the decoder padded it
    asymmetrically. Verified at the time with a constant-logit model returning an
    exactly constant map (spread 0.000e+00, was non-zero) and a depth-4 U-Net on a
    constant image showing 0/8 boundary-aligned jumps. **That verification was
    inadequate and this account is superseded:** GroupNorm is degenerate on a
    constant image, so the test could not see the residual. Cropping left a
    diffuse, tile-grid-dependent error (1/8 on a boundary afterwards), and
    inference was subsequently replaced with a whole-slice forward pass. See the
    `predict_proba` docstring.
  - **Every CNN/transformer/modern-conv number in this repo predates the fix,
    including the 0.3168 headline.** They were measured with a defective blend
    and are unreliable *in an unknown direction*: the sign is not yet
    established. Because weights were never persisted, corrected numbers
    require **retraining**, not re-inference.
- **POC imagery** (`src/poc_images.py`, 4 figures), rendered from the real
  saved probability maps of `unet2d_s30000` on held-out test slices, at the
  validation-selected threshold, not illustrated. Measured en route:
  96.0% adjacent-slice prediction agreement (the signal z-refinement is
  built on), and an *oracle* droppable fraction rising 60.0% to 89.1% from
  20 to 6000 superpixels, against the learned drop rule's 58.5%. That gap is
  the hierarchy's remaining headroom, and it is large.
- **`src/pipeline_v2.py`, the data-agnostic pipeline.** `MeteoriteSource` /
  `build_multi_split` / `power_report` / `stage2_sweep` / `confidence_sweep` /
  `run_all`. Adding a meteorite is a config change. Self-test output:
  `531/531 pairs OK`, `13 straddler instance(s) resolved to majority slab;
  871822 voxels ignored; 0 chondrules span 2+ zones`, zone counts summing to
  the band, no slice index in two zones.
- **Two defects caught and fixed in that pipeline before it was trusted:**
  - *Straddler resolution was missing*, because the first draft relied on the
    buffer alone. `split.py` measured chondrule z-extents reaching 70
    band-slices against a 12-slice buffer, so chondrules could appear in two
    core zones. Now reuses the audited same-id-CC + majority-slab rule, with
    the post-hoc recount assertion. **P1 violation, caught before use.**
  - *Chondrule counting used plain binary CC*, which fuses touching chondrules,
    the same failure that once dropped 53.6% of positives. It reported
    87 where the same-id rule reports 142 (a 39% shortfall). This feeds
    `power_report`'s `n`, so every minimum-detectable-effect figure would have
    been wrong. Now uses `split.instance_label`; the corrected counts are
    **train 81 / val 32 / test 29 = 142**, matching the known total exactly.
- **Free statistical power from slab placement.** The v2 split puts
  32 chondrules in validation versus the frozen split's 19, which lowers
  the minimum detectable effect from 0.0142 to 0.0109 absolute IoU with no
  new data. It is a trade, not a free lunch: training chondrules fall from 99
  to 81, and the two splits' scores are not comparable to each other.
- **`hier_cnn` at a matched 30,000-step budget: test IoU 0.3037**
  (val 0.4460, threshold 0.000177, 41.5% of pixels alive), up from 0.2695 at
  8k. Against the plain U-Net's 0.3159 at the same budget that is -0.012
  IoU, inside the ±0.014 noise floor. The honest reading: the hierarchy
  buys a ~2.4× reduction in the area the CNN must process at no *measurable*
  accuracy cost. It is a compute optimization, and we cannot currently
  distinguish it from break-even.

### 2026-07-23: Full model sweep, budget curve, confidence pipeline

**Headline: best test IoU 0.3168** (`unet2d_s30000`, a plain 2D U-Net trained
30,000 steps), vs 0.203 for classical ML and a 0.130 intensity floor.
Annotation ceiling ~0.92. **All numbers are "best-of-40, selection-biased"**:
the winner was chosen from a 40-model sweep, so its test score is the max of 40
noisy estimates (optimistic by roughly 0.02 to 0.03). Thresholds were always fit
on validation; the model itself was never tuned on test.

- **Budget beat architecture.** Same U-Net at 3k / 7.2k / 15k / 30k steps gives
  0.2314 / 0.2834 / 0.3089 / 0.3159. Gains halve each doubling and have
  plateaued (+0.007 for the last doubling). The best of 40 *architectures* at a
  fixed budget reached only 0.2598. Sweeping architectures was a worse use of
  compute than training one model longer.
- **41 leaderboard rows** on the identical leakage-free split: 36 distinct
  architecture/budget configurations swept, plus 4 recall-training loss
  variants and 1 hierarchy-filtered model. (The commit message says "37-model
  sweep"; the exact figure is 36 sweep configurations. Counts differ by which
  follow-up runs are included, and this is the reconciled breakdown.) Two rows
  are **degenerate constant predictors, not successes**: `gnn_gat` and
  `matrix_specialist` both sit at IoU 0.0528 with recall 1.000. Families:
  U-Nets, attention/2.5D variants, 9 extra conv families, 4 from-scratch
  transformers (`vit_seg`, `swin_lite`, `segnext_lite`, `axial_unet`), 6 modern
  conv families, 7 GNNs, and a GNN-then-CNN cascade.
- **Over-predict-then-prune works, once the objective is right** (Sky's idea).
  `ErrorModel` regressing `|prob-gt|` is INVERTED for pruning: at a low
  threshold a true positive has residual ≈0.9 and a false positive ≈0.1, so
  "prune high error" deletes chondrules (measured: 0.331 on TPs vs 0.259 on
  FPs). `FPDetector` asks *"is this predicted-positive pixel a false positive?"*
  and fixes it: `swin_lite` 0.2552 to 0.2675, `gnn_graphsage` 0.1783 to 0.2133.
  Oracle prune ceiling is 0.810 from 0.146, so the mechanism has large headroom.
- **Cross-slice z-refinement works** (Sky's idea): +5.4% IoU for free. Pixels
  where z-neighbours disagree are wrong ~62% of the time vs ~6% where they
  agree, a strong zero-cost confidence signal.
- **Confidence bake-off: nothing earned its keep on the final model.** Best
  learned candidate (`hist_gb`, error task) reached pruned val IoU 0.455582 vs
  0.455048 for every free baseline (`|p-0.5|`, raw-prob uncertainty,
  z-agreement), a +0.0005 gain, i.e. the learned models marginally *won*
  but by nothing. The free baselines gained exactly 0.000000. Earlier wording
  ("the free baseline beat every learned model") was backwards. Learned
  confidence DID help a weak model substantially (`gnn_graphsage` 0.1783 to
  0.2133), so the honest reading is: confidence recovers slack in a poor model
  and adds nothing to a good one.
- **Asymmetric recall-oriented training HURTS**: symmetric dice 0.2314 vs
  tversky-recall 0.1529, focal-tversky 0.1603, dilated-target 0.2196. Train
  symmetrically and move the threshold instead.
- **Hierarchical superpixel cascade** (Sky's idea): 2.71× compute saving at
  99.9% chondrule retention after fixing a merge-rule bug (mean-intensity
  agglomeration with no size term let one region absorb 98 to 99% of each slice;
  fixed with a Ward-style cost). Relaxing per-level recall from 1.0 to 0.999
  raises the saving from 1.66× to 2.70× for 0.05% of chondrule pixels. As a
  standalone segmenter it reaches only 0.1437, because at pixel level the rule
  degenerates to the intensity threshold. So it is an efficiency mechanism, not
  an accuracy one.
- **RETRACTED: there is no hard recall ceiling.** An earlier entry claimed the
  CNN assigns probability ~0 to ~42% of chondrule pixels and saturates at recall
  0.709. Recomputed from the model's own saved probabilities: only 0.40% of
  chondrule pixels are exactly zero, and max recall as the threshold approaches
  0 is 0.996. The "42%" was the fraction below 2e-4, the floor of the
  threshold grid, not of the model's output. This is the THIRD occurrence of
  the truncated-threshold-grid bug already logged twice below.
  *Honest replacement:* those pixels carry probabilities 3 to 4 orders of
  magnitude below the useful operating range, so recovering them costs nearly
  all precision. Thr 1e-6 gives recall 0.961 but IoU 0.065; thr 1e-5 gives
  recall 0.781, IoU 0.125; thr 1e-4 gives recall 0.592, IoU 0.282; the max-IoU
  point is 0.3168. A *soft economic* ceiling, not a hard one.
- **Surviving finding: the model's confidence falls as chondrules get brighter.**
  Mean predicted probability over TRUE chondrule pixels by CT intensity:
  0.739 (0 to 35), 0.655 (35 to 45), 0.374 (45 to 55), 0.130 (55 to 70),
  0.085 (100+). Threshold-free, so unaffected by the grid bug.
  The cause is UNIDENTIFIED. An earlier explanation, that bright pixels are
  mostly not-chondrule so the model learns that bright means matrix, is
  **refuted**: the global not-chondrule:chondrule ratio is 17.9:1, so the 10:1
  ratio in the CT 55 to 70 band means bright pixels are actually 1.7x ENRICHED
  in chondrules relative to base rate. A brightness-only Bayes rule would rank
  them ABOVE the prior; this model ranks them far below it. Also note the trend
  is not strictly monotone and the darkest bin (~2% of chondrule pixels) does
  not replicate across splits. The robust statement is that above CT ≈ 40 the
  model's output on true chondrule pixels falls steeply (test 0.655, 0.374,
  0.129, 0.085; val 0.600, 0.409, 0.076, 0.030).
- **Generalization-limited, not capacity-limited**: train IoU 1.000 vs ~0.32
  held-out, budget plateaued, 40 architectures clustered, augmentation and
  weight decay both unhelpful. N=1 meteorite, ~142 chondrules, adjacent slices
  ~0.83 IoU similar, so the effective sample size is far below 213 slices.
- **Model selection is unreliable here**: a bootstrap over validation slices
  gives a 95% CI of [-0.013, +0.015] on the val-IoU gap between the top two
  models (P(better) = 0.50), while their test gap is 0.057.
- **New modules**: `zones` (cached split), `evaluate` (one scoring path),
  `models_extra` / `models_transformer` / `models_modern` / `models_gnn_variants`,
  `graph_cache`, `combine` (z-refine, ConfidenceModel, ErrorModel, FPDetector,
  prune), `confidence_bakeoff`, `cascade`, `hierarchy`, `autoresearch`,
  `run_families` / `run_final` / `run_overnight` / `make_figures`.
- **Bugs caught that would have produced confidently wrong conclusions**: GNN
  node features unstandardized (~20,000× scale spread giving constant outputs);
  budget-run name collisions overwriting both leaderboard rows and probability
  files; truncated threshold grids (twice) making a "recall 1.00" experiment
  actually measure 0.69; queue deadlocks; dead `pos_weight` / `invert_target`
  knobs; and the hierarchy merge-rule defect above.
- **Final pipeline on the locked model: test IoU 0.3214** (`unet2d_s30000` +
  FP-prune + z-smoothing), vs 0.3168 for the model alone.
- **NEGATIVE, on the final model. Both must be reported.** Constitution P9
  requires the head-to-head, it was run, and both lost:
  **z-refinement gives 0.2972 vs 0.3168 baseline (-6.2%)**, and **FP-prune gives
  0.3131 vs 0.3168**. The earlier "+5.4%" and "prune works" claims were measured
  on `swin_lite` and `gnn_graphsage`, models the roadmap itself calls
  disposable. On the model actually shipped, neither helps in isolation; only
  the stacked combination (0.3214) exceeds the baseline.
- **The FP-prune recall sweep is degenerate**: all four targets (0.80/0.90/0.95/
  0.98) resolved to the same threshold 2e-4 with byte-identical metrics, the
  same grid-floor cause as the retracted ceiling claim. It is a single point,
  not a sweep.
- **Two-model veto (Sky's is/is-not idea): 0.2750 test IoU vs 0.3168 reference, so
  it loses on IoU**, but raises precision to 0.463 from the reference's 0.4147 (+0.049; the 0.332 quoted
  earlier was `hrnet_lite`'s, not this model's). About 60% of that gain is
  reproduced by simply thresholding the same model to a smaller prediction
  budget (precision 0.4435 at a matched budget), so only ~+0.020 is attributable
  to the matrix specialist, whose own output spans just 0.705 to 0.996 at recall
  0.403. The matrix specialist genuinely vetoes false positives; it over-vetoes.
  Probability-averaging the two gives 0.3162, indistinguishable from the base.
- **Not delivered, disclosed rather than glossed:** (a) v4 inference on the ~130
  unlabeled slices was NOT produced, because the sweep persisted probability maps but
  not model weights, so the winner must be retrained to run it;
  `results/unlabeled.json` records only the slice list. (b) The straddler-drop
  ablation promised in constitution P1 has a single arm in
  `results/ablation.json`. Nothing was compared, so the majority-assignment
  policy remains unquantified.
- **Reproducibility gap, disclosed:** `results/hierarchy_fixed.json` (the 2.71x
  headline) has **no producer script in the repo**. It was generated
  out-of-band while iterating on the merge-rule fix. The committed
  `src/run_overnight.py:run_hierarchy` still writes the superseded 1.58x
  `results/hierarchy.json`. The 2.71x number should be treated as
  not-yet-reproducible until that stage is rewired.

### 2026-07-23: Karpathy-style autoresearch (Phase 0 + search rounds)
- **Phase 0 (sanity), the pivotal finding.** A U-Net overfits 4 fixed patches to
  **train IoU 1.000** (dice@3e-4; bce/tversky ≈0.997) and the valid-mask
  provably neutralizes every loss (all four give 0.0000 when nothing is valid).
  So the earlier ~0.19 was **not** a bug, not capacity, and not step count:
  **train 1.000 vs held-out 0.19 is a generalization gap.** This invalidated the
  "just train longer" plan and redirected the search to regularization.
- **Round 1 (one variable, ranked on FULL val).**
  `p1_ctrl_b32` (no augmentation) **val IoU 0.3874** vs `p1_aug_b32`
  (augmentation) **0.2021**.
  - **LR fix alone was worth ~57%**: val 0.246 to 0.387 from `lr=3e-4` + cosine
    decay (both prescribed by Phase 0), with no architecture change.
  - **Augmentation HURT, nearly halving IoU.** Hypothesis under test in round 2:
    the bundle included **intensity jitter**, and intensity is the discriminative
    signal (chondrules ≈50 vs matrix ≈32 CT), so jitter corrupts the main
    feature while flips/rotations should be harmless. `augment="geom"` now
    isolates geometry (verified: constant intensity across draws).
- **Phase 3 (long budget, 7,200 steps): CNNs overtake classical ML.**
  `long_p1_ctrl_b32` **test IoU 0.2834** (Dice 0.442, P 0.360, R 0.571; val
  0.4508) vs the classical-ML champion **0.203**, a **+40%** improvement, and
  the first learned model to beat texture features on the held-out slab.
  `long_p1_geom_b32` reached test 0.2329 (val 0.4375).
- **Augmentation: budget artifact confirmed.** At the search budget the
  augmentation gap was 0.165 val IoU; at 7,200 steps it closed to **0.013**
  (0.4375 vs 0.4508). So augmentation was never orientation-sensitivity. It
  simply needed steps to amortize. It still does not win, so the no-aug recipe
  stands, but the earlier round-2 reading is now superseded.
- **Threshold grid was truncated (fixed).** Both long runs selected exactly
  `thr=0.05`, the grid minimum, a parameter pinned to its boundary, meaning the
  true optimum lay outside the search and those two test numbers are likely
  understated. Grid extended down to 0.002; dice loss under ~95% background
  genuinely pushes the useful operating point very low.
- **Round 2: the jitter hypothesis was WRONG.** Ranked on full val:
  `p1_ctrl_b32` **0.3874** > `p1_ctrl_wd_b32` 0.3640 > `p1_geom_b32` 0.2221 >
  `p1_aug_b32` 0.2021. Removing intensity jitter recovered only **0.02** of the
  0.185 gap, so the *geometric* flips/rotations cause nearly all the damage,
  not the jitter. Most likely a budget artifact (augmentation slows convergence
  and 1,500 steps is too short to amortize it) rather than orientation
  sensitivity; tested directly by a long-budget run rather than assumed.
  Weight decay also did not help. **Best recipe so far: no augmentation,
  lr=3e-4 + cosine decay, dice, base=32.**
- **GNN superpixel ceiling measured before training** (`n_segments` vs the
  "every superpixel gets its majority label" oracle, real val slices):
  1200 gives **0.771**, 3000 gives 0.837, 6000 gives **0.878**, at essentially identical
  cost (~9.5 s/slice, dominated by feature extraction, not SLIC). Default
  raised to 6000 so the GNN is not capped at 0.77. Note this is a hard ceiling
  for any superpixel method, below the 0.92 annotation ceiling.
- **Confidence-model memory fix:** the per-pixel feature stack was ~6.1 GB for a
  real zone (violating the memory rule); now built in z-windows with a halo,
  verified **bit-identical** to the full-volume path (max diff 0.0) at ~600 MB
  peak. Also documented that risk-coverage IoU is legitimately NaN at low
  coverage (the most-trusted pixels are background-dominated, giving an empty union).
- **Infrastructure:** split cached once (`src/zones.py`), so search workers load
  train+val only and never read test, cutting RAM from ~30 GB (4 workers) to
  ~12.6 GB (2 workers) after the earlier run swamped the machine.
  Added `src/evaluate.py` (one path for every family: train, pick the threshold
  on val, then score test once), `src/run_families.py`, and `results/excluded_models.json`
  (SegFormer / TransUNet / pretrained ResNet-UNet / SAM, declared with reasons,
  no silent caps).
- **New model code:** `src/models_extra.py` adds 9 additional NN families
  (pixel-MLP, patch-CNN, dilated CNN, UNet++, FCN-8, SegNet, DeepLabV3+,
  autoencoder, ConvLSTM-over-z); shape self-test passes, including
  non-power-of-two tiles. Untrained as of this entry.

### 2026-07-23: v2 sweep (Tier A+B), audits & evaluation ceiling
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
  precision (0.13 to 0.25); linear/NB still flood.
- **Annotation ceiling** (answers "how close to 1 is possible"): ±1px-boundary
  IoU ≈ **0.92**, ±2px ≈ 0.85; adjacent-slice (copy-neighbor) IoU ≈ 0.83. So the
  practical IoU ceiling ~**0.85 to 0.92**, not ~1.0. Current best 0.20 leaves real
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
  training quality, documented and not lost): 2.5D train-seam channel mixing,
  `pos_weight` threading, focal α, and a regression-test suite. Core split &
  metrics were audited leakage-free with no changes required.
- **Scope expansion (Sky):** sweep **all** neural-network families (not only the
  named CNN/GNN), and add a **learned confidence model** trained on the CV
  models' own mistakes to flag where predictions are trustworthy (cf. the REE
  logD separation-confidence approach).

### 2026-07-23: Project bootstrap (SDD Phase 0 to 1)
- **Discovery.** Identified the dataset as a micro-CT scan of **ALH 83010**:
  531 raw slices (1450×962, 8-bit) paired 1:1 with a hand-labeled instance
  segmentation. Verified from pixels that labels are **individual chondrule
  instances** (compact blobs persisting across z), not mineral classes; positive
  target collapses to binary **chondrule vs. matrix**. Chondrules are denser
  (brighter) than matrix in CT (mean ≈ 50 vs. 32). Annotation covers a
  **central band ≈ slices 100 to 496**; ~130 end slices are unlabeled. `BP.CSV` is
  unrelated (hydrocarbon table) and ignored.
- **Constitution** written (`memory/constitution.md`, v0.1): mission, scope,
  data facts, local-Mac stack, 8 non-negotiable principles (leakage-free z-split,
  touch-test-once, honest negatives, baseline-first, joint metrics, N=1 honesty,
  reproducibility, SDD-every-change), and a v0 to v4 roadmap.
- **Decisions (Sky, MCQ):** task = binary semantic segmentation with per-pixel
  probability; metric = foreground IoU; empty ends **excluded as unlabeled**;
  test split = **contiguous z-slab + buffer**; compute = **local Mac only**.
- **Spec v0/v1** written (`specs/2026-07-23-chondrule-segmentation/spec.md`):
  evaluation harness + frozen leakage-free split + classical baseline; awaiting
  sign-off.
