# Changelog

All notable decisions and milestones for the Meteorite Chondrule Segmentation
project. Honest numbers only; evaluation regime stated with every metric.

## [Unreleased]

### ⚠️ KNOWN ISSUES IN THIS COMMIT (found by pre-commit review — read first)

- **`+10% from z-context` is UNBACKED.** `results/probs/unet25d_k5.npy` is
  actually the 30,000-step run's output (bare-name collision, fixed in code but
  after these files were written). Re-scoring it gives **0.2154**, not the
  leaderboard's 0.2547 — the surviving artifact points the OPPOSITE way. Treat
  every "2.5D beats 2D at equal budget" statement as unverified.
- **The 3k budget anchor has no artifact.** `probs/unet2d.npy` is byte-identical
  to `probs/unet2d_s30000.npy`. The 0.2314 value is independently reproduced by
  `rt_dice_ctrl` (0.23142 @ thr 0.25), but the named file is wrong, and
  `pick_winner` globs both names so one model enters the ranking twice.
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
- **`tests/` is empty.** ~5,000 new lines ship with only synthetic `__main__`
  smoke tests, none run automatically.
- **Latent bug:** `run_families.py` / `cascade.py` do `bool(rec["aug"])`, so an
  `aug="geom"` recipe would silently train with intensity jitter. Not triggered
  today (winning recipe has `aug: false`).



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
