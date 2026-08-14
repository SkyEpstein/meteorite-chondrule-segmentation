# Session handoff, 2026-08-08

Sixth handoff. Supersedes the 2026-08-04 one; the 2026-08-06 body is kept and
amended in place, with later changes marked [UPDATED]. Read
[CHANGELOG.md](CHANGELOG.md)'s KNOWN ISSUES block before quoting any number.

## READ THIS FIRST: the noise floor on TEST is 5.4x what this project assumed

`resunet18_scratch_s30000` now has three seeds at 30,000 steps:

```
   seed 0  0.33569      seed 1  0.26098      seed 2  0.34233
   mean 0.31300      SD 0.04517      spread 0.08135
```

**0.33569 was the best of three draws, not the model's score.** Every place this
repo called it "the highest post-fix score" is retracted; the arm's score is
0.31300 +/- 0.04517.

**AND SO IS THE REFERENCE MODEL.** `unet2d_s30000`, the anchor every gap in this
repo is quoted against, was also run three times the same day:

```
   seed 0 (the reference)  0.29409      seed 2  0.27631      seed 1  0.26639
   mean 0.27893      SD 0.01403
```

0.29409 is the TOP of its own spread, +1.08 SD above its mean. **Every gap
measured against it shifts by -0.01516 in the same direction**, because they all
share that anchor: geometry augmentation -0.0501 becomes -0.0349, the 30k `full`
arm -0.0328 becomes -0.0176, the prune +0.0087 becomes +0.0239, the scratch arm
+0.0189 becomes +0.0341.

**There is NO single full-budget noise number, and an earlier draft of this file
claimed there was.** The two configurations measured at 30,000 steps give
**0.01403** (UNet2D) and **0.04517** (ResNet18-UNet), 3.2x apart. Do not carry
one across architectures. Do not conclude UNet2D is the more stable one either:
the SD intervals at n=3 are [0.0073, 0.0882] and [0.0235, 0.2840] and overlap
heavily. Both are far above the 0.0084 the project used to reason with.

Use the SD of the arms actually being compared. On that basis:

| comparison | gap | test |
|---|---|---|
| scratch ResNet-18 vs UNet2D, both n=3 | **+0.0341** | SE 0.0273, **t = 1.25** |
| geometry augmentation vs the UNet2D mean | -0.0349 | 2-sigma bar about 0.032 at the point-estimate SD |
| 30k `full` arm vs the UNet2D mean | -0.0176 | inside |
| pretraining, pretrained n=1 vs scratch n=3 | -0.0103 | inside |

**Still nothing clears its noise, but two things are now close enough to be
worth settling.** Geometry augmentation is marginally outside its bar at the
point-estimate SD, though that bar ranges from about 0.017 to 0.20 across the
SD's own interval, so it is the best CANDIDATE for a real effect here, not an
established one. Two seeds of `unet2d_s30000_auggeom` would settle it. The
ResNet-vs-UNet gap at t = 1.25 is the largest apparent effect in the project.

It is real model variance, NOT threshold-selection variance. Seed 1 froze thr
0.95 while seeds 0 and 2 chose 0.002, which made selection the obvious suspect.
Holding the threshold fixed, SD is 0.0551 at 0.002 and 0.0488 at 0.95, against
0.0452 with val-selected thresholds, so selection slightly REDUCES spread. Seed 1
is below the others at both thresholds.

## What 2026-08-08 changed

1. The seed result above.
2. **`pipeline_v2.run_all` now runs end to end on TWO meteorites** (2026-08-04
   priority 2, previously never done). `python -m tests.test_pipeline_v2_multi`,
   ~20 min. Two latent defects had to be fixed first: `stage2_sweep` called
   `run_and_score` WITHOUT `leaderboard=False` while scoring a
   `build_multi_split` split, which that function's docstring marks a [MUST]
   (nothing was corrupted, there are zero `hier_*` rows); and `out_dir` named
   only the final JSON while the split record, the maps and the reload all used
   module-level `results_v2/`. CAVEAT: the synthetic chondrules are drawn from
   disjoint intensity ranges, so the pipeline scores a perfect 1.000 and the test
   CANNOT detect a leakage bug. Overlapping-intensity fixtures are not built.
3. **The AUPRC question is settled and the 2026-08-07 entry had it backwards.**
   The leaderboard values are CORRECT; the saved maps are lossy.
   `run_and_score` computes metrics on float32 and saves float16, which
   threshold metrics and 10-bin ECE survive and ranking metrics do not. Proved by
   regenerating from a checkpoint: `resunet18_scratch_s30000` is 0.44301 in
   float32, exactly its leaderboard value, and 0.33455 cast to float16, exactly
   what the saved map gives. **Quote AUPRC from `results/leaderboard_all.json`,
   never recompute it from `results/probs/`.** Sky chose to keep float16 and
   carry the rule rather than double ~13 GB. AUPRC labels on
   `results/figures/fp_penalty_pr.png` are understated; its curves and
   precision/recall/FP counts are fine.
4. **A clobber fixed before it fired.** `pretrained_probe` had no `--seed` and
   left the seed out of the model name, so any replicate would have overwritten
   seed 0's checkpoint, both maps and its leaderboard row under the same name.
   Seed 0 keeps the historical name, others get `_seed{n}`.
5. Sky's `src/split_protocol_audit.py` (untracked before this commit) measured
   what the literature's random-slice protocol buys: leakage-free 0.1956,
   no-buffer 0.1738, random-slice 0.2479 +/- 0.0331, **inflation +0.0524 IoU
   (+27%)**. NOT MINE, and one thing to check before quoting: `no_buffer` came
   out BELOW `leakage_free`, which is backwards for a leakier split, and both are
   single deterministic runs with no seed spread.

## What 2026-08-07 changed

1. `full` went from n=1 to n=3 at 7,200 steps. Section 1's table is updated. The
   headline is unchanged and slightly stronger: the test gap flipped sign.
2. **[RETRACTED]** `logs/cascade_arms.log` prints
   `full 0.5203 +/- 0.0158 val, 0.2574 +/- 0.0055 test` and
   `full +0.1179, bar 0.0448, RESOLVED`, under a header reading "3 seeds each".
   Those are the mean of a 7,200-step and a 30,000-step run pooled across
   incomparable budgets, n=2, with the between-budget spread printed as a seed
   SD. **Do not quote that log.** Checked repo-wide: the numbers never
   propagated out of the log file, and the 2026-08-06 handoff correctly showed
   `full (n=1) 0.5092`. Regenerate with `python -m src.run_cascade_arms --report`.
3. Two defects fixed in that printout, both rendering a non-result as a result:
   the control was matched by arm name alone so rows were differenced ACROSS
   budgets, and an n=1 row's `nan` bar fell through to printing "inside the
   noise" where no verdict is possible. See the CHANGELOG entry.
4. **TRAP:** `pytest tests/` collects nothing and exits 0, which reads as a pass.
   Both test modules are script-style. Run `python -m tests.test_aux_channels`
   and `python -m tests.test_multi_meteorite`. Both pass as of this commit.

## Nothing is running

Every job finished. Verify with `pgrep -f "python3 -u -m src"` before starting
anything.

## THE HEADLINE: three independent attacks on the same errors, all null

Every method tried this session targets the same thing, the confident false
positives that dominate the error (5.8M FP against 2.4M FN). None of them moved
test IoU.

| attack | where it acts | validation | test |
|---|---|---|---|
| classical pipeline into the CNN | input | **+0.086** | **-0.033** |
| penalising false positives in the loss | training | -0.015 | -0.046 |
| stacking three confidence signals | prune | -0.011 vs best single | -0.016 vs margin |

The first is the important one and it is the cleanest comparison in the repo.

## 1. The full pipeline at full budget is WORSE on test

`arm_full_s0_b30000`: a `UNet2D` base 32 depth 4 at 30,000 steps taking raw plus
denoised plus CLAHE plus top-hat plus the cross-fitted classical probability.
Same architecture, same budget, same split as `unet2d_s30000`. Only the input
channels differ.

| | validation | test |
|---|---|---|
| `unet2d_s30000`, raw only | 0.4451 | **0.2941** |
| `arm_full_s0_b30000`, composed | **0.5315** | **0.2613** |
| difference | +0.086 | **-0.033** |

Validation went up by 0.086 and test went DOWN by 0.033.

The 7,200-step arms (3 seeds each, `results/cascade_arms.json`) show the same
thing with error bars. Validation climbs monotonically with more classical input
and test does not move:

| arm | val | val vs control | test | test vs control |
|---|---|---|---|---|
| control | 0.4024 +/- 0.0159 | | 0.2493 +/- 0.0289 | |
| cleaned | 0.4560 +/- 0.0098 | +0.054 RESOLVED | 0.2534 +/- 0.0099 | +0.004 noise |
| cascade | 0.4757 +/- 0.0118 | +0.073 RESOLVED | 0.2467 +/- 0.0040 | -0.003 noise |
| full | **0.4986 +/- 0.0181** | **+0.096 RESOLVED** | **0.2476 +/- 0.0070** | **-0.002 noise** |

[UPDATED 2026-08-07] `full` was n=1 in the 2026-08-06 handoff and is now n=3;
seeds 1 and 2 were run for exactly this reason. Its val mean fell 0.0106 and its
test mean fell 0.0059, and **the test gap changed SIGN, +0.004 at n=1 to -0.002
at n=3**. All three validation gains are RESOLVED against a seed-derived bar.
No test difference is. This is validation overfitting through the INPUT CHANNELS
rather than through the weights, which the project had not previously
demonstrated, and the extra seeds strengthened it rather than softening it.

The composition ordering cleaned < cascade < full holds on validation, so the two
classical products stack rather than one carrying the whole effect. That is a
real result about the architecture. It just does not reach test.

[HONESTY] The 30k comparison is n=1 seed and the measured test-side seed SD is
0.0289, so -0.033 alone is about 1.1 SD and not resolved. What makes it
persuasive is four arms at two budgets all showing the same direction.

[UPDATED 2026-08-08] Two corrections, pulling in opposite directions. The anchor
`unet2d_s30000` is the top of its own three-seed spread (mean 0.27893), so the
gap is **-0.0176**, not -0.033. But the right SD for a UNet2D-family comparison
is **0.01403**, not 0.0289, so -0.0176 is about 1.25 of THAT SD. Still not
resolved, and the arithmetic is now anchored rather than floating.

Do NOT use the ResNet arm's 0.04517 here: it is a different architecture and 3.2x
larger. The `full` arm's own seed SD is unmeasured at n=1, and it carries four
aux channels, so borrowing plain UNet2D's 0.01403 is an assumption, not a
measurement. The "four arms same direction" argument also thins: three of those
four are at 7,200 steps, where the val gain, not the test gap, is what resolved.

## 2. Penalising false positives did not help

Sky's mentor's proposal, and it had genuinely never been tried: the recall
bake-off only ever ran beta > alpha. `tversky_loss(alpha=0.7, beta=0.3)` is the
exact mirror, and Tversky at alpha = beta = 0.5 IS Dice, so the existing control
is the matched baseline.

| | val | test |
|---|---|---|
| control (dice), 3 seeds | 0.4024 | 0.2493 |
| fp_penalty (tversky 0.7/0.3), 3 seeds | 0.3874 | 0.2033 |
| difference | -0.0150 (bar 0.0464) | -0.0460 (bar 0.0679) |

Formally inside the noise on both. But the direction is consistent: per-seed test
was 0.2186 / 0.2073 / 0.1838 against the control's 0.2669 / 0.2159 / 0.2650, and
two of three fp_penalty seeds sit below the control's worst seed. Read it as "did
not help and probably hurt", not as "no effect".

## 3. Constitution stage 7 is now BUILT, and it does not beat one free signal

The constitution specifies three independent error signals stacked into one
score. That was never implemented: `run_bakeoff` returns candidates ranked so
"the winner can be reused" and `combine.fit_prune` takes ONE score map. So every
previous conclusion about the confidence stage was about SINGLE signals only.

`src/confidence_stack.py` builds it. All three signals cross-fit inside val on
contiguous buffered folds, cutoff chosen on out-of-fold scores over all 60
slices, test scored once.

| arm | val IoU | test IoU |
|---|---|---|
| unpruned (opened to recall 0.958) | 0.07107 | 0.06026 |
| regressor | **0.45874** | 0.29316 |
| stacked (all three) | 0.44731 | 0.28661 |
| margin, zero parameters | 0.43807 | **0.30279** |
| error U-Net | 0.10561 | 0.06969 |
| z_agreement | 0.08018 | 0.07134 |

Stacked is 0.0114 BELOW the regressor alone on validation, and below margin on
test. Stacker coefficients: regressor 7.99, error U-Net 1.10, z_agreement 0.75.
The stacker correctly identified that only one input carries signal.

**The error U-Net was NOT rescued by more data.** The fit set went from 28 slices
to 40-44 per fold via cross-validation, and it still only prunes to 0.10561
against an unpruned 0.07107. Sky's hypothesis that fit-set size was the binding
constraint is not supported at this scale.

**Almost all the value is the threshold, not the ranking.** Opening to recall
0.958 drops IoU to 0.060; every arm then recovers to 0.29-0.30. A zero-parameter
score does that best.

## Why the confidence stage cannot work here (mechanism, from the images)

`results/figures/confidence_signals_full_arm.png` shows it. The margin map is
high (confident) almost everywhere, with thin uncertainty rings only at object
boundaries. The errors are large, coherent, CONFIDENT blobs on matrix that looks
like chondrule. The signal and the error do not overlap, so no threshold on that
signal can separate them. The model is not uncertain about its mistakes.

Also structural: **the prune can only REMOVE**. With 2.4M false negatives, a
third of the error is permanently out of its reach.

And 8 of the regressor's 11 features are functions of `prob` itself, so it is
largely a noisy restatement of margin. That is why it never beats it.

## Where the numbers stand

Post-tiler-fix only. Everything else on the leaderboard is pre-fix and marked.

| | test IoU | note |
|---|---|---|
| `resunet18_scratch_s30000` | **0.31300 +/- 0.04517** | [UPDATED 2026-08-08] n=3 seeds (0.33569 / 0.26098 / 0.34233). The single-seed 0.33569 that this row used to quote, and called the highest post-fix score in the repo, was the best of three draws and is RETRACTED as a headline |
| `resunet18_pretrained_s30000` | 0.30269 | its ImageNet twin |
| `unet2d_s30000` | 0.29409 | Dice 0.455, prec 0.372, recall 0.584, AUPRC 0.427 |
| `arm_full_s0_b30000` | 0.2613 | the composed full pipeline, 30k |
| `unet2d_s30000_auggeom` | 0.24404 | |
| annotation ceiling | 0.9226 | so 0.336 is about 36% of what the labels allow |

Classical and GNN never touched the tiler: `plus3_ws` 0.2185
(defect-contaminated), `plus3` 0.2127, `hist_gboost` / `base18` 0.20283 (one
measurement at two precisions), best GNN `gnn_resgcn` 0.19548, threshold 0.130.

**Our best model was never meant to be a model.** `resunet18_scratch_s30000` is
the random-init CONTROL arm of the pretraining probe, an instrument for measuring
data starvation. It scored highest by accident. It also differs from
`unet2d_s30000` in encoder, parameter count (14.0M vs 7.8M) and normalisation
(BatchNorm vs GroupNorm) at once, so its +0.0416 attributes to nothing in
particular.

## Model selection is not honest and cannot be

`results/final_pipeline.json` records it plainly: `selected_on` is a
"FORCE_MODEL override, NOT val-selected. hrnet_lite ranked #1 on validation
(0.47794) vs this model (0.47696) ... a TEST-informed tiebreak."

Validation ranked `hrnet_lite` first; the shipped winner is `unet2d_s30000`,
forced, because it topped a leaderboard SORTED BY TEST IoU. And validation could
not have chosen: those two differ by 0.001 on val against a measured bar of
0.148. There is no honest selection procedure available on this dataset.

## The measurement limits (unchanged, and now better evidenced)

- **A val gap must exceed 0.137 to 0.172 IoU to carry information about test**
  (39 models, `results/val_vs_test.json`), while the whole competitive field
  spans 0.2564. This dataset separates about two tiers of model.
- **Seed SD is regime-dependent and larger than recorded.** 0.0084 at fraction
  0.25 / 7.2k, but **0.0289** for the control at full data / 7.2k. The 0.0084 in
  the docs is not general.
- `SE_REF_N = 19` is the val zone's BINARY connected-components count, measured
  (`results/chondrule_recount.json`: train 63/49, val **36/19**, test 43/24). It
  was NOT changed, because `SE_REF_IOU = 0.0071` was bootstrapped over 60 val
  SLICES, so neither count was ever the resampling unit.
- Code uses `MDE = 2.0 * SE`; prose cites 2.8 at three sites, only one deriving
  it, as (1.96+0.84) for 80% power. They answer different questions. The field
  NAME `mde_iou` is wrong for what it computes.

## Uncommitted work, and it still cannot be committed from a worktree

[UPDATED 2026-08-14] The 2026-08-04 backlog LANDED: `180bee0` and `a653490` are
committed. HEAD is `a653490`, and 5 commits are unpushed. The staged work now is
the synthetic-data arm (see below).

## The synthetic-data arm: 2 of 5 tasks, and the mentor's idea is UNTESTED

Sky's mentor proposed synthetic data via a conditioned MaskGIT + VQ-VAE.
Spec: `specs/2026-08-08-synthetic-maskgit/spec.md`. Sky retargeted it at the
CONFIDENCE stages, held OUT of base training, because the base records train IoU
1.000 and so errs on nothing it trained on.

**Nothing has been generated, trained on, or scored.** Tasks 3 (MaskGIT
transformer), 4 (error-structure gate) and 5 (confidence stages on real test) are
NOT started. Do not describe this arm as tested.

What tasks 1 and 2 established, which is real:

| tokenizer | agreement IoU | flips | positive fraction orig -> recon |
|---|---|---|---|
| f=8 (16x16 grid) | 0.3302 | 23.6% | 0.1429 -> 0.3255 (**2.28x**) |
| f=4 (32x32 grid) | 0.6168 | 7.3% | 0.1429 -> 0.1634 (1.14x) |

At the compression the cited abstract uses, the tokenizer strips matrix texture
and more than doubles the segmenter's predicted-positive rate, because this
model's documented failure is calling confusable matrix "chondrule". f=4 recovers
most of it at 16x the transformer attention. **f=2 was never measured**: launched,
killed when the pool was found contaminated, never relaunched.

OPEN, and they are Sky's calls, not the next session's to assume:
1. The reconstruction holdout is SAME-SLICE, so the tokenizer's absolute figures
   are optimistic by an unmeasured amount. Rebuilding it slice-disjoint is cheap
   and not done. It is the only quantitative evidence in that commit.
2. Tasks 3-5 are the expensive two, and `HANDOFF.md`'s own priority 1 is "get a
   second meteorite, nothing else moves the binding constraint". The spec
   pre-registers that a real +0.03 gain would sit at or below test noise. This
   deserves an explicit go/no-go rather than continuing by default.
3. The generator is fit on the base segmenter's OWN train slices, so the base's
   errors on synthetic samples may be systematically softer than on real
   held-out data. That is the premise the whole retarget rests on, and task 4's
   gate is what would catch it.

**A WORKTREE SESSION CANNOT COMMIT IN THIS REPO.** All three gates (review,
oversee, test) key state by SESSION directory in
`~/.claude/harness/state/review/<session-path-hash>`. A session under
`.claude/worktrees/` writes that key while a commit in the main checkout reads
`_Users_skyepstein_Meteorite_Computer_Vision_0ebe756c75f6`, which nothing in the
worktree session can write. The test gate is worse: **neither key has a
`last_test` field at all**, so it blocks unconditionally.

Do the work in the MAIN checkout if you intend to commit. That is also where the
data lives. Hatches are `[review-skip]`, `[oversee-skip]`, `[test-skip]`, each
needs Sky's explicit approval, each is logged, and they are read only from `-m`.

## GitHub is diverged and needs reconciling

`origin/main` is `83d6d48`; local is `18ab53a`. Three commits each way from the
initial commit, NOT a fast-forward. `c313916` (local) and `14a2349` (remote) are
content-identical duplicates. Remote-only `153c48d` removed em dashes across 28
`src/` files and added `memory/style.md`, which exists ONLY on the remote. Seven
files conflict; three more were added independently on both sides.

Sky's stated policy for the cleanup: **GitHub carries only accurate results, no
disclaimers. If something is wrong, remove it. The CHANGELOG keeps the full
record.** The README status block was rewritten under that policy on 2026-08-05
and now leads with 0.29409 and 0.33569 rather than the withdrawn 0.3168.
[UPDATED 2026-08-08] The README status block was rewritten again under the same
policy: it now leads with the three-seed 0.31300 +/- 0.04517, and the 0.33569 it
briefly led with is marked retracted there.

## Built this session and NOT yet run

- `src/pipeline_v2.run_all` with two sources. [DONE 2026-08-08] It completes end
  to end on two synthetic meteorites; see the top of this file.
- `cnn_crossfit.py` has never run; there are no `cnn_xf` files. Fitting a
  confidence model on out-of-fold TRAIN predictions needs 5 folds at 30k first,
  roughly 11 hours. Sky's judgement was it would suffer distribution mismatch
  anyway, since the error model would learn a weaker base model's mistakes.
- Resizing the split (shrink test, grow val) was approved in principle for later.
  The arithmetic favours it: test SE would grow about 0.0026 against a seed SD of
  0.0289. The cost is that all 44 leaderboard rows become non-comparable.

## Next steps, in priority order

1. **Get a second meteorite.** Nothing else moves the binding constraint. As of
   2026-08-08 this is no longer an argument from principle: the measured
   full-budget seed SD (0.01403 for UNet2D, 0.04517 for ResNet18-UNet) is on the
   order of, or larger than, every effect anyone has tried to measure on test, so
   ALH 83010 can barely answer method questions at all. The gate
   is ANNOTATION, not acquisition: raw chondrite CT is plentiful, public
   chondrule labels do not exist.
2. **Reconcile with `origin/main`, then push. Not a fast-forward.** Measured
   2026-08-08 with `git merge-tree` (non-destructive); the earlier "seven files
   conflict" in this file was WRONG.

   The two histories share only the initial commit `824e811` as a git ancestor,
   but the second commit exists TWICE under different hashes: local `c313916`
   and remote `14a2349` are byte-identical (same tree `0210266...`, same
   patch-id `b329cec...`). That work was committed locally and also landed on
   GitHub via PR #1, so git sees two unrelated commits holding the same content.

   Genuinely one-sided: the remote has `153c48d` ("Rewrite docs for clarity,
   remove em dashes, add style guide and repo description"), which rewrote
   CHANGELOG +515, README +298, ROADMAP +318, constitution +203 and added
   `memory/style.md`, which exists ONLY there. The laptop has everything from
   `e37aa0e` onward.

   **A straight merge conflicts in 37 files**, most of them `add/add` on
   `src/*.py`, because the true merge base predates every source file.
   **Rebasing instead collapses that to 14** and removes every spurious source
   conflict, because the duplicate commits are content-identical:

   ```
   git rebase --onto origin/main c313916 main
   ```

   The 14 are the real overlap: CHANGELOG, README, ROADMAP,
   memory/constitution.md, .gitignore, two figures, the complete-model spec, and
   the six src files 153c48d actually edited (data_scaling, evaluate,
   make_figures, models_cnn, pipeline_v2, poc_images).

   Sky's call on 2026-08-08 was to keep this SEPARATE from the commit, because
   the four prose files are exactly the ones carrying the retractions and
   resolving one the wrong way silently un-retracts a corrected number.
   **After rebasing, re-grep `0.33569`** and confirm it still reads as retracted
   in README, ROADMAP and this file. Also re-grep `0.5203` and the AUPRC block.

   [MUST] Do NOT rebase while a training job is running: `reference_replicates`
   writes probability maps, checkpoints and `results/leaderboard_all.json` while
   a rebase rewrites the tree under it. `origin/main` was last fetched
   2026-08-07; re-fetch first.
4. Seeds for anything still quoted at n=1. The list is short and now known to
   matter: `resunet18_pretrained_s30000`, the 30k `full` arm, and
   `unet2d_s30000` itself, which is the project's reference number and has never
   been replicated.
5. Overlapping-intensity synthetic fixtures, so `tests/test_pipeline_v2_multi`
   can actually detect a leakage bug instead of scoring a trivial 1.000.
6. A budget knob for `confidence_sweep`. 787 s of that test's 20 minutes is one
   bake-off candidate (`mlp`), and a 20-minute plumbing test will not be run.

DROPPED: "settle the scratch arm's grid-floor threshold, which gates the
0.33569." Measured 2026-08-08 and it gates nothing. Holding the threshold fixed
across the three seeds gives SD 0.0551 at 0.002 and 0.0488 at 0.95, against
0.0452 with val-selected thresholds, so selection is not the variance source.

## Respect the machine

64 GB, one training job at a time. The 2026-08-02 swap incident was four
concurrent jobs. A figure job alongside one trainer is fine. The composed arm at
4 aux channels peaks around 7.5 GB in the patch sampler alone. `evaluate.load`
takes `want_test=False`; `zones.load_zones(root, zones=("val",))` loads one zone.

## Traps that have cost time

- **Corrections propagate badly.** Five corrections were applied at the exact
  line cited and nowhere else; an oversight pass found the same wrong claims in
  four other files. Grep the whole repo, including code comments.
- **`run_and_score` saves nothing unless you ask.** A 6-hour arm sweep was
  launched with neither `save_prob` nor `ckpt_dir`, so its probability maps and
  weights would both have been discarded. Caught before it wasted the night.
- **Summaries can pool incomparable regimes.** `save()` grouped by arm only, so
  the 30,000-step composed run was silently averaged with its 7,200-step
  namesake. Now keyed by (arm, budget).
- **A re-run at a new budget will overwrite the old one** unless the budget is in
  the name. `arm_full_s0` vs `arm_full_s0_b30000`.
- **Fit and grade must not share slices.** A first draft of `confidence_stack`
  fitted the regressor on val and predicted on val, which would have rigged the
  comparison the module exists to make. Everything learned is cross-fit now.
- **`results/probs/unet2d.npy` is not `unet2d`.** It is byte-identical to
  `results/prefix_tiler_backup/probs/unet2d_s30000.npy`.
- **Untrained-net probes cannot validate an inference change.** A whole-slice
  "fix" passed every check on an untrained net, then collapsed the trained model.
