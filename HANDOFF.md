# Session handoff, 2026-08-07

Fifth handoff. Supersedes the 2026-08-04 one; the 2026-08-06 body is kept and
amended in place, with today's changes marked [UPDATED 2026-08-07]. Read
[CHANGELOG.md](CHANGELOG.md)'s KNOWN ISSUES block before quoting any number.

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
| `resunet18_scratch_s30000` | **0.33569** | highest post-fix in the repo. n=1 seed, threshold on the grid floor |
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

## Uncommitted work: 44 files, and it still cannot be committed from a worktree

`HEAD` is still `18ab53a`. Everything from 2026-08-04 onward is staged or
untracked. Commit message drafted at
`/tmp/claude-501/METEORITE_COMMIT_MSG_2026-08-04.txt`.

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

## Built this session and NOT yet run

- `src/pipeline_v2.run_all` with two sources. Still never completed end to end.
- `cnn_crossfit.py` has never run; there are no `cnn_xf` files. Fitting a
  confidence model on out-of-fold TRAIN predictions needs 5 folds at 30k first,
  roughly 11 hours. Sky's judgement was it would suffer distribution mismatch
  anyway, since the error model would learn a weaker base model's mistakes.
- Resizing the split (shrink test, grow val) was approved in principle for later.
  The arithmetic favours it: test SE would grow about 0.0026 against a seed SD of
  0.0289. The cost is that all 44 leaderboard rows become non-comparable.

## Next steps, in priority order

1. **Get a second meteorite.** Nothing else moves the binding constraint, and
   this session closed off three method routes at once.
2. Commit the 44 pending files from a main-checkout session.
3. Reconcile with `origin/main`, keeping `memory/style.md` and the em-dash pass.
4. Run `pipeline_v2.run_all` with two synthetic sources so ingestion is proven
   before real data lands.
5. Settle the scratch arm's grid-floor threshold, which gates the 0.33569.

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
