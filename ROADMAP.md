# Roadmap — Meteorite Chondrule Segmentation

**Canonical, durable roadmap.** The session task list is ephemeral; this file is
the source of truth. Update it whenever a stage completes or scope changes.

Governed by [memory/constitution.md](memory/constitution.md) ·
results in [CHANGELOG.md](CHANGELOG.md) ·
specs in [specs/2026-07-23-chondrule-segmentation/](specs/2026-07-23-chondrule-segmentation/)

**Last updated:** 2026-08-04

> ⚠️ **Read the KNOWN ISSUES block at the top of [CHANGELOG.md](CHANGELOG.md) before quoting any number here.** Several results in this file are withdrawn or non-reproducing.

---

## 2026-08-04: the pretraining probe is NULL, and the first measured seed variance

**The pretraining probe is COMPLETE and its verdict is NULL.** The pretrained
ResNet18-UNet scored test IoU 0.30269 against the scratch control's 0.33569, a
gap of -0.03300. The band registered before the numbers arrived says a gap under
about 0.05 is ambiguous at n=1 seed, so this is not evidence either way, and it
is not being reinterpreted now that it has a direction. The sign is opposite to
"pretraining substitutes for data". This is the ninth val/test sign flip:
pretrained won validation by +0.0325 and lost test by -0.0330. [UNVERIFIED] The
scratch arm's val-frozen threshold is 0.002, the documented grid floor, so do not
treat -0.033 as clean until that is settled.

> **[UPDATED 2026-08-08]** The scratch arm now has three seeds: 0.33569 /
> 0.26098 / 0.34233, **mean 0.31300, SD 0.04517**. Both readings in this entry,
> the one above and the one below, were computed against 0.33569, which was its
> luckiest draw.
> The pretraining gap is **-0.0103** against the three-seed mean, still null and
> now further inside the band. The +0.0416 architecture gap is **+0.0189**,
> about 0.4 SD. The [UNVERIFIED] threshold caveat is RESOLVED and was a dead
> end: with the threshold held fixed across the three seeds the SD is 0.0551 at
> 0.002 and 0.0488 at 0.95, against 0.0452 with val-selected thresholds, so
> selection is not the variance source. See CHANGELOG.md 2026-08-08.

**Second reading, the architecture one.** Scratch ResNet18-UNet (0.33569) beats
plain `unet2d_s30000` (0.29409) by +0.0416. Also n=1 seed, also inside the
ambiguity band. [CORRECTED 2026-08-04] It does not isolate architecture:
`resunet18_*` is BatchNorm throughout (torchvision ResNet-18 encoder plus an
`nn.BatchNorm2d` decoder) while `unet2d_s30000` uses GroupNorm
(`models_cnn._norm`), so this comparison crosses a normalisation change too.

**First measured seed variance in this project: SD 0.0084.** Fraction 0.25 of
train at the 7,200-step budget, three seeds, test IoU 0.2823 / 0.2664 / 0.2696.
The data-scaling curve now stands at 3 of 12.

**Validation predicts test across models, and is still too coarse to use here.**
Over 39 leaderboard models r = +0.898, but the bar a val gap must clear to mean
anything is 0.137 to 0.172 IoU, while the competitive field spans only 0.2564 in
validation IoU. This dataset separates about two tiers of model and nothing
finer.

**SE_REF_N left at 19, and the recount settled what 19 is.** The proposed
correction to 36 had the sign backwards: the anchor is a matched pair, so raising
the count LOOSENS every floor by sqrt(36/19) = 1.376. The 0.0103 figure is
WITHDRAWN. `python -m src.recount_chondrules` then measured the frozen split both
ways over identical voxels:

| zone | slices | same-id 3D | binary CC |
|---|---|---|---|
| train | 213 | 63 | 49 |
| val | 60 | **36** | **19** |
| test | 80 | 43 | 24 |
| total | | **142** | 92 |

So `SE_REF_N = 19` is exactly the binary connected-components count of the val
zone, the undercount `count_chondrules_3d` was written to avoid. The handoff's
63/36/43 is confirmed; the 99/19/24 in CHANGELOG.md's 2026-07-29 "Prepare for a
second meteorite" entry, under "Free statistical power from slab placement", is
not reproduced. The constant is STILL not changed to 36, because `SE_REF_IOU`
was bootstrapped over 60 val SLICES, so neither 19 nor 36 was ever the
resampling unit.

**The 2.0 versus 2.8 multiplier is traced.** Only one of three mentions carries a
derivation (`specs/2026-07-29-complete-model-contract/spec.md:206-208`), which
gives 2.8 as (1.96 + 0.84), the 80%-power sample-sizing constant. The code's 2.0
has no derivation anywhere. They answer different questions: 2.0 (about 1.96)
tests whether an OBSERVED gain differs from zero, which is what `_gain` actually
does, while 2.8 states what effect a study would detect 80% of the time. The code
is defensible for its use; the naming is not, since a field called `mde_iou`
means the second thing while computing the first.

**Multi-meteorite ingestion was untested and was half broken.** Two scans of
different in-plane size crashed. Fixed by padding to a common size with the
padding marked invalid. `pipeline_v2.run_all` with two sources has still never
completed end to end.

**Leaderboard provenance.** All 44 rows now carry a per-row `tiler_regime` field,
and `run_and_score` stamps new rows.

---

## THE COMPLETE MODEL (constitution §12, Sky 2026-07-29) — SUPERSEDED BELOW

> **This section lists the OLD four-stage, base-first order.** Sky reordered it
> on 2026-07-30 to seven stages, hierarchy first and pixel dropping last. The
> constitution is authoritative; this list has not been rewritten.


From here on, **"the complete model" names exactly one thing** — the ordered
composition of four pluggable stages:

| # | stage | fit on | role |
|---|---|---|---|
| 0 | **base segmenter** — best arch by leakage-free val bake-off, high-recall operating point | train, selected on val | proposes; misses are unrecoverable |
| 1 | **hierarchy filter** — superpixel cascade, thresholds frozen after val | val | compute (2.41× at recall 0.999) |
| 2 | **confidence model** — learned trust/error meta-model vs free baselines | val | prunes false positives |
| 3 | **z-layer neighbor confidence** — cross-slice agreement | val | exploits z-continuity |

- Headline metric: **end-to-end test foreground IoU** vs the
  plain-U-Net-at-matched-budget baseline.
- **Every stage must be independently ablatable** — a stage that does not earn its
  place has to be visible, not assumed.
- **Architecture contract, not a performance claim.** At n=19 val chondrules the
  MDE is 0.0142 IoU and the hierarchy measured −0.012 — inside the floor. The
  stages are not currently distinguishable from break-even.
  **[CORRECTED 2026-08-04] "n=19 val chondrules" is the wrong label.**
  `src/recount_chondrules.py` counted the val zone both ways over identical
  voxels: 19 is the BINARY connected-components count, and the pipeline's own
  same-id rule gives 36. `SE_REF_N` was left at 19 deliberately, because the
  `SE_REF_IOU = 0.0071` it pairs with was bootstrapped over 60 validation SLICES,
  so neither chondrule count was ever the resampling unit. The 0.0071 and the
  0.0142 floor are unchanged. The 2026-08-04 section at the top of this file
  already records the recount, so this file contradicts itself wherever it calls
  19 a chondrule count; the same correction applies at every such site below.
- No stage may hard-code ALH 83010; the entry point stays `run_all(sources)`.

**Verified state (2026-07-29):** all four stages are already *wired* into
`pipeline_v2.run_all` — what is missing is the **composite**. Next session, in
order:

1. **[FIXED 2026-07-30] `combine.recall_threshold`** — `THR_GRID` floors at 2e-4 and the
   function silently returns that floor, so **no** recall target is reachable
   (measured: all four targets collapse to val recall 0.60 / 0.55). Extend the
   grid to ~1e-7 log-spaced and make the fallback raise or flag. Constitution §9
   ("over-predict, then prune") NOW executes: recall 0.99 is reached on the
   control (0.9910), still unreached on the augmented arm (0.9355, flagged).
   Regenerate the prune
   and bake-off numbers in `final_pipeline.json` afterwards.
2. **Give `run_all` a real entry point and run it** — it currently has no caller
   anywhere; `__main__` is plumbing-only and `results_v2/` has no `pipeline_v2.json`.
3. **Chain the stages** — base → hierarchy → **z-refine probabilities → prune**
   (Sky's order, so nothing binarizes early). Today the bake-off winner, the prune
   sweep and z-refine each read the same base map independently.
4. **Add a hierarchy-off arm** to `stage2_sweep`, which currently only builds
   `HierMaskedRunner` — so stages 0 and 1 cannot be ablated apart.
5. **Settle the ablation threshold protocol** (shared grid vs per-arm) — at
   MDE 0.0142 it changes the sign of small deltas.

**🔑 Blocks Stage 2 — the base map is saturated.** Measured on the retrained
model: **91.0%** of test pixels are < 1e-4, **7.3%** are > 0.9999, and only
**0.32%** fall in the graded 0.1–0.9 band (pre-tiler-fix it was 3.00% — the old
tiler's zero-padded-border averaging was manufacturing the gradation). A
confidence model and the free `|p − 0.5|` baseline both rank by gradation, so
there is almost nothing for Stage 2 to grade — a candidate explanation for the
standing result that no confidence method beats a zero-parameter baseline by more
than 0.0005 IoU. **Calibrate the base map first** (a loss that does not saturate
logits, val-fit temperature scaling, or label smoothing) before judging Stage 2 or
concluding anything about constitution §9's "over-predict, then prune". Also
`save_prob` writes **float16**, which barely spans the 1e-5…1e-7 range where this
model's residual signal lives — use float32 for val maps that meta-models fit on.

**Also queued (2026-07-29), independent of the contract:**

- **Relaunch the data-scaling curve** — `python -m src.data_scaling`, 12 runs ×
  7,200 steps, ~10–11 h, resumable. **[UPDATED 2026-08-04] At 3 of 12 runs, not
  running.** Fraction 0.25 is complete at 3 seeds (test IoU 0.2823 / 0.2664 /
  0.2696, `n_seeds: 3`, `test_iou_sd` about 0.00841), and
  `results/data_scaling_v2.json` is committed. Fractions 0.5, 0.75 and 1.0 stand
  at 0 of 3 each, so 9 runs remain, roughly 5 hours. This entry said "1 of 12 as
  of 2026-07-30" and "uncommitted" while the 2026-08-04 section above already
  said 3 of 12, so the file contradicted itself. It has still **never completed
  the full sweep**; an earlier attempt died at run 1/12 saving nothing. One
  completed fraction gives no slope, so "does more annotated data help?" remains
  unanswered. Now safe to run:
  it passes `leaderboard=False` so its subset runs no longer pollute
  `results/leaderboard_all.json`.
- **Apply the same `leaderboard=False` isolation to `pipeline_v2.stage2_sweep`** —
  still outstanding. It scores on the **v2 split** (val `[246,305]`, test
  `[318,396]`) and writes into the frozen-split leaderboard, which
  `fig_leaderboard` plots as one ranked chart.
- **Fix the tile-boundary artifact at its source.** It is NOT cosmetic: on the
  retrained model, column jumps at x=256k are 37–88× the median and y=512 is 201×,
  cutting chondrules along straight lines (chondrule 1 of slice 69: TP 14164 vs
  FN 15752). Whole-slice inference is NOT the fix (reverted — collapses the
  model). The real options are upstream: replace GroupNorm with a normalisation
  independent of window extent, or train on windows matching inference.
  `run_families` now persists weights to `results/checkpoints/`, so
  `predict_proba` vs `predict_proba_whole` can be compared on the actual headline
  model instead of an untrained probe.
- **Regenerate `poc_images`** — the committed figures are pre-fix renders, and
  `src/poc_images.py` hard-codes `THR = 0.0076` while the probability map it loads
  was frozen at 0.006.
- **v2 split prevalence gap** — val `pos_frac` 0.0848 vs test 0.0311 (2.7×).
  `zone_ranges` puts test at the end of the annotated band, reversing the frozen
  split's deliberate placement in the chondrule-dense interior. IoU is
  prevalence-sensitive, so a val-frozen threshold mis-transfers.

Spec: [`specs/2026-07-29-complete-model-contract/spec.md`](specs/2026-07-29-complete-model-contract/spec.md)

---

## 🔜 READY FOR MORE METEORITES (`src/pipeline_v2.py`)

The binding constraint is **data**, not method. Measured: 142 3D chondrules in
the whole volume, **19 in validation**, giving a bootstrap SE of ±0.0071 IoU and
a **minimum detectable effect of ~0.014 IoU**. Almost every method tested
produced 0.005–0.014 IoU — at or below that floor, which is why nine comparisons
flipped sign between val and test. We could not measure whether they work.

**[CORRECTED 2026-08-04] "19 in validation" is the binary connected-components
count.** `src/recount_chondrules.py` counted the val zone both ways over
identical voxels, and the pipeline's own same-id rule gives 36. `SE_REF_N` was
left at 19 deliberately, because the ±0.0071 SE it pairs with was bootstrapped
over 60 validation SLICES, so neither chondrule count was ever the resampling
unit. The 0.0071 and the ~0.014 floor are unchanged.

**[CORRECTED 2026-08-04] The sign-flip count above read six when the paragraph
was written.** It is nine as of 2026-08-04; the pretraining probe was the ninth.

`pipeline_v2.py` makes adding meteorites a config change:

**[CORRECTED 2026-08-04] That claim was FALSE.** `pipeline_v2.py` crashed on two
scans of different in-plane size, so the ingestion path itself needed a fix (pad
to the common in-plane size, with the padding marked invalid), and `run_all` with
two sources has still never completed end to end. The list below describes what
the code does today.

- **`MeteoriteSource`** — point it at a new raw/label directory pair.
- **`build_multi_split`** — per-meteorite contiguous z-slabs with buffers, then
  concatenated, so no chondrule spans a seam and no meteorite lands only in test.
  Reports per-zone 3D chondrule counts (the number that governs power).
- **`power_report`** — prints the minimum detectable effect in absolute IoU for
  the actual validation size; every reported gain is flagged
  `below_noise_floor` when it is smaller.
- **`stage2_sweep`** — the hierarchical second-stage sweep: candidate
  architectures trained ONLY inside the hierarchy's alive mask, as `hier_cnn`
  does now. Two measured operating points, don't conflate them:
  **target recall 0.99 → 36.8% of pixels alive (2.71x saving), 99.91%
  of chondrule pixels retained**; **target recall 0.999 → 41.5% alive
  (2.41x saving)**, which is the setting `hier_cnn` was trained under.
- **`confidence_sweep`** — the full bake-off (10 learned + 4 free baselines),
  the FP-prune recall sweep, and z-refinement, all re-measured rather than
  assumed. On the current data none of these beat a zero-parameter baseline by
  more than 0.0005 IoU; with 19 validation chondrules that is not a finding
  that they fail, it is a finding that we cannot tell.
  **[CORRECTED 2026-08-04]** 19 is the binary connected-components count of the
  val zone; the pipeline's own same-id rule gives 36 over identical voxels
  (`src/recount_chondrules.py`). `SE_REF_N` was left at 19 deliberately, because
  the SE it pairs with was bootstrapped over 60 validation SLICES, so neither
  chondrule count was ever the resampling unit. The floor is unchanged.
- **Checkpoints** (`CNNRunner.save_checkpoint` / `load_checkpoint`) — weights
  used to die with the process, which is why the unlabeled-slice inference was
  never produced. The val-selected threshold is stored with the weights.

Defaults bake in what actually won: **plain UNet2D, 30,000 steps, lr 3e-4 with
cosine decay, dice loss, no augmentation** — budget was the largest measured
lever (+0.085 IoU from 3k→30k), larger than any architecture change.

**Expected minimum detectable effect** — 19 chondrules: 0.014 IoU ·
40: 0.010 IoU · 80: 0.007 IoU · 150: 0.005 IoU.

---

## Where the numbers stand

> 🔴 **EVERY CNN-FAMILY NUMBER IN THIS TABLE IS SUPERSEDED (2026-07-29).** All of
> them were produced under **tiled inference** with the halo-accumulation defect.
> That defect is fixed (halo now cropped) and **tiled inference remains the
> default** — but a second mechanism is ACCEPTED, not fixed: GroupNorm normalises
> per-window, so predictions still depend on where the tile grid falls. Measured
> on the retrained model, positive-pixel counts jump **83x / 37x / 59x / 88x** the
> median at x = 256 / 512 / 768 / 1024, and **201x** at y = 512 — visible as
> straight-line cuts through chondrules. A whole-slice replacement was tried and
> **reverted** (best-epoch val IoU 0.4487 → 0.2632 on a 25%-data run, epochs 8–12
> at 0.0000–0.0075, two separate runs at the same config/seed; these nets
> train on 128×128 patches). **Do not quote 0.3168, 0.3214, or any row below as
> current.** Retrained headline: **0.29409**. GNN rows bypass the tiler and are
> unaffected.

Every row marked [PRE-FIX] was produced under the defective tiler and is
superseded. The marker is repeated per row on purpose: a row quoted on its own,
away from the warning above, must still carry its own caveat.

| model | test IoU | note |
|---|---|---|
| **`unet2d_s30000`** | **0.3168** | [PRE-FIX] **best overall**: plain U-Net, just trained longer. Post-fix retrain: 0.29409 |
| final pipeline (prune + z stacked) | 0.3214 | [PRE-FIX] `unet2d_s30000` + FP-prune + z-smoothing. Also selected on TEST, not val |
| `unet2d_s15000` | 0.3089 | [PRE-FIX] earlier budget point, same architecture |
| `long_p1_ctrl_b32` (7,200 steps) | 0.2834 | [PRE-FIX] same architecture, less budget |
| `hrnet_lite` | 0.2598 | [PRE-FIX] best of 29 architectures at equal budget |
| `swin_lite` | 0.2552 | [PRE-FIX] from-scratch transformer |
| `unet25d_k5` | 0.2547 | [PRE-FIX] z-context. **CLAIM WITHDRAWN**, artifact corrupted (see CHANGELOG KNOWN ISSUES) |
| `tiramisu` | 0.2478 | [PRE-FIX] |
| `resunet` | 0.2440 | [PRE-FIX] |
| `hist_gboost` (classical ML) | 0.203 | Tier B champion. Classical, not affected by the tiler |
| `gnn_graphsage` / `superpixel_gcn` | 0.178 / 0.177 | best GNNs. Bypass the tiler, unaffected |
| `unet2d` (3,000 steps) | 0.2314 | [PRE-FIX] budget-curve anchor |
| intensity threshold | 0.130 | v1 floor. Not affected by the tiler |

**Budget curve (same architecture, only steps differ), still climbing.** Every
point below is [PRE-FIX]; read the SHAPE, not the heights. 3,000 gives 0.2314;
7,200 gives 0.2834; 15,000 gives **0.3089**; 30,000 gives **0.3159**
(plateaued: +0.052, +0.026, +0.007 per doubling). The 30,000 point re-scores to
0.29409 post-fix, so the curve's top is roughly 0.02 lower than drawn.

**On the final model these gains do NOT hold in isolation** — z-refine −6.2%,
FP-prune −0.4% vs baseline; only the stacked combination (0.3214) beats it.
See CHANGELOG.

**Ceilings:** annotation ~0.92 (±1px boundary) · superpixel quantization ~0.878
(GNN-only). **Regime:** leakage-free contiguous z-slab split, threshold tuned on
val, test scored once, N = 1 meteorite.

---

## ✅ Complete

| # | Stage | Outcome |
|---|---|---|
| — | Leakage-free split + 2 adversarial audits | 142 chondrules, **0 leak**, independently confirmed |
| — | Evaluation harness + classical baseline | IoU 0.130 floor |
| — | Tier A/B sweep (thresholds + classical ML) | 0.203 |
| — | Annotation ceiling | ~0.92 (±1px), ~0.85 (±2px) |
| 1 | Phase 0 sanity (Karpathy) | overfits to train **1.000** → gap is *generalization*, not optimization |
| 2 | Regularization search + long run | val 0.387 → **test 0.2834**, beats classical ML by 40% |
| 3 | Architecture sweep, 29 families | in progress, 28 scored, 0 failures |
| — | Public repo | [SkyEpstein/meteorite-chondrule-segmentation](https://github.com/SkyEpstein/meteorite-chondrule-segmentation) |

---

## 🔄 In progress

all stages complete; see Complete and the negative-results sections

---

## ⏳ Model selection (must finish before anything downstream)

| # | Stage | Notes |
|---|---|---|
| 15 | Transformer + modern-conv families | mostly done; corrects an earlier over-broad exclusion. [CORRECTED 2026-08-04] *Pretrained* models are not impractical locally either: a 14.0M-parameter ImageNet ResNet-18 U-Net trained here in 8150 s. [CORRECTED 2026-08-04] The 14.0M was written here as the ResNet-18's count; it is the whole net, ResNet-18 encoder plus U-Net decoder. The ResNet-18 alone is 11.69M |
| 16 | GNN variants | GAT still degenerate (R=1.000) — a second, GAT-specific bug remains |
| 17 | **Cascade: GNN coarse → CNN refinement** (Sky) | built, cross-fitting verified; escapes the 0.878 ceiling |
| 21 | **Hierarchical cascade** (Sky) | **validated by measurement** — see below. NOTE: Sky specified **superpixels**; the feasibility test used a regular GRID, an unflagged substitution. Grids nest cleanly (each cell → 4) whereas SLIC levels do not nest, so a true superpixel hierarchy needs agglomerative merging on the region-adjacency graph. Superpixels adhere to intensity edges, so they should be MORE separable — the grid numbers are likely a lower bound. **Measured: 2.71x compute saving at 99.9% retention after fixing the merge rule.** |

---

# 🚧 BASE MODEL LOCKED HERE — HARD BARRIER 🚧

**Nothing below this line runs until the model above is final** (Sky,
2026-07-23, reaffirmed after the ordering drifted).

A confidence model fits **its base model's specific error distribution**, so
every result below has to be *redone* if the base model changes. Running it
early is pure rework.

**This was violated in practice** — confidence experiments were run on
`swin_lite` and two GNNs, none of which will be the final model, so all of those
numbers are disposable. The one thing it bought was finding the inverted-target
bug (see the retraction above) on a throwaway model rather than at the end,
which was cheap insurance. That debt is now paid; the barrier holds from here.

**Order of operations:** finish the sweep → budget curve → cascade →
hierarchical → **pick the winner on VALIDATION** → train it properly → *then*
everything below, once.

**Status: barrier passed — the post-lock pipeline ran on unet2d_s30000; results in results/final_pipeline.json.**

---

---

## ⏳ Confidence & operating point (blocked on model lock)

| # | Stage | Notes |
|---|---|---|
| 19 | Recall-oriented training bake-off | `tversky_recall`, `focal_tversky`, `combo`+`pos_weight`, `dilate_target` vs symmetric+low-threshold |
| 18 | **Confidence bake-off** | 10 learned candidates **+ free baselines** (`\|p−0.5\|`, z-agreement) — a learned model must beat a zero-cost one to justify itself |
| 12 / 7 | Error regressor + trust classifier | error regressor decisively beat trust classifiers in harness testing |
| 13 | **Over-predict → prune** (Sky) | recall targets 0.80→1.00 swept, winner chosen on val |
| 6 | **Cross-slice z-refinement** (Sky) | measured +5.4% on swin_lite but **−6.2% on the final model** |
| 5 | **Two-model: is / is-not** (Sky) | machinery built (`invert_target`, `hard_frac`); the naive `1−p` version is a no-op and is *not* the plan |
| 20 | *Conditional:* GNN confidence sweep | runs only if pruned GNN clears **0.203** (pre-registered) |

---

## ⏳ Final

| # | Stage |
|---|---|
| 8 | Recall operating point + straddler-drop ablation |
| 9 | Fix 2.5D train-seam; regression tests (`pos_weight` already fixed) |
| 10 | Apply best model to the ~130 unlabeled slices |
| 11 | Docs, review gate, commit, push |

---

## ⚠️ over-predict-then-prune: WORKS ON WEAK MODELS ONLY

**On the shipped model it LOSES**: FP-prune 0.3131 vs 0.3168 baseline. The gains
below are real but were measured on `swin_lite` and `gnn_graphsage` — models this
same roadmap calls disposable. Do not cite them as properties of the final
system.

`FPDetector` (target: *among predicted-positive pixels, is this a false
positive?*) fixes the inverted signal — `P(FP)` is now 0.637 on FPs vs 0.388 on
TPs — and **beats the max-IoU baseline**:

| model | max-IoU ref | best pruned | gain |
|---|---|---|---|
| `swin_lite` | 0.2552 | **0.2675** (@R=0.90) | **+4.8%** |
| `gnn_graphsage` | 0.1783 | **0.2133** (@R=0.95) | **+19.6%** |

At R=0.95 `swin_lite` is unpruned 0.0543 → pruned 0.2659: a near-worthless
over-prediction recovered 4.9x, above the tuned baseline. The pruned score is
also nearly FLAT across recall targets (0.2659–0.2675), so the pipeline is
insensitive to the starting threshold — a practical advantage over
threshold-tuning.

## ✅ Hierarchical superpixel cascade (Sky's idea) — 2.71x saving at 99.9% retention

**FINAL, after fixing a merge-rule bug.** Honest val-fit → test-apply, levels
80 → 6000, recall target swept:

| target recall | test area alive | saving | chondrule retention |
|---|---|---|---|
| 1.000 | 0.604 | 1.66x | 0.99998 |
| **0.999** | 0.370 | **2.70x** | 0.99947 |
| 0.995 | 0.369 | 2.71x | 0.99911 |

Insisting on *perfect* recall costs ~40% of the benefit: relaxing to 0.999
raises the saving from 1.66x to 2.70x for 0.05% of chondrule pixels. Saving
plateaus below 0.999, so that is the operating point.

**Superpixels beat a grid for hierarchical dropping** (Sky specified
superpixels; the first feasibility test used a grid):

| regions | grid droppable | SLIC droppable |
|---|---|---|
| **20** | 53.0% | **69.8%** |
| 80 | 76.2% | 77.5% |
| 320 | 83.6% | 84.9% |

The gap is largest at the coarse level, which is exactly where the hierarchy
saves the most compute — so use superpixels, per the original idea.

### The bug that invalidated the first round of results

`build_hierarchy` merged the most-similar adjacent regions by **mean intensity
with no size term**, so an already-large region paid no more to absorb its next
neighbour than a tiny one did. One region ran away and swallowed the slice:
largest level-0 region **98.4–99.4% of the image**. Consequences:

- oracle droppable area at 20 regions collapsed to **0.5%** (direct SLIC: 69%)
- the four coarse levels changed the outcome by **5.4e-5 of area** — the
  "cascade" was empirically a flat threshold at the finest level
- the reported explanation ("20 regions is too coarse, every 70k-px region
  clips a chondrule") was **factually wrong** — regions were not 70k px; one
  was 1.38M px

Fixed with a **Ward-style cost** (`|Δmean|² · s_u·s_v/(s_u+s_v)`), which makes
merging two large regions expensive and keeps the partition roughly equal-area.
Droppable area at 80 regions went **23.8% → 78.2%**; at 320, **31.5% → 85.8%**.

Also corrected: the earlier **4.1x was in-sample** (threshold fit and scored on
the same 4 slices, so "100% retention" was a tautology), and
"granularity barely matters" was an oracle-threshold artefact — under honest
transfer, droppable rises 41.4% → 58.7% from 1,500 → 24,000 segments.

**Level 0 (20 regions) remains degenerate** — agglomerating all the way to 20
is inherently unbalanced; direct SLIC would be needed there. Levels 80+ now
match direct SLIC, so the cascade starts at 80.

## (superseded) earlier hierarchy claims

`src/hierarchy.py` — bottom-up agglomerative merge on the region-adjacency graph
(SLIC levels do not nest, so the tree is built by merging), then descend:
drop empty regions, subdivide survivors. Thresholds fit on val at perfect
recall. Verified independently, 4 val slices:

| level | regions | frac area alive | chondrule retention |
|---|---|---|---|
| 0 | 20/slice | 1.0000 | 1.0000 |
| 1 | 80/slice | **0.2441** | 1.0000 |
| 2–4 | 320 → 6000 | 0.2440 | 1.0000 |

SUPERSEDED — Final: 24.4% of the image survives → 4.1x compute saving at 100%
chondrule pixel retention (independently recomputed from `final_alive_mask`, not
taken from the reported table).

Refinements to the original framing:
- **20 regions is too coarse** — nothing is droppable there, because nearly every
  70k-px region clips some chondrule. The useful entry level is **80**.
- Only **14 of 320** regions are dropped at level 1, but they hold **75.6% of the
  area**: merging fuses uniform matrix into a few huge regions while
  chondrule-adjacent regions stay small. A few confident drops remove most of
  the image.
- **All the saving happens at one level** — finer levels drop regions but no
  additional area, so a 2-level hierarchy may capture nearly all the benefit.
- 4.1x vs the 12x ideal is the price of fitting at *perfect* recall; sweeping
  `target_recall` (0.999, 0.995) is the obvious next knob.

## ⚠️ RETRACTED negative result — the objective was wrong, not the strategy

The tables below were produced with `ErrorModel`, which regresses
`|prob - gt|`. **That target is inverted for pruning.** At a high-recall
threshold every probability is small, so a TRUE positive (gt=1, prob≈0.1) has
residual ≈0.9 while a FALSE positive (gt=0, prob≈0.1) has residual ≈0.1 —
pruning "high error" therefore deletes chondrules. Measured on `gnn_graphsage`:
mean predicted error **0.331 on TPs vs 0.259 on FPs**, exactly backwards.

Diagnostic that settled it (same model, high-recall operating point):

| quantity | value |
|---|---|
| unpruned | 0.1456 |
| **ORACLE prune (delete every FP)** | **0.8100** ← pruning's real ceiling |
| learned error map vs true error | Pearson 0.598, **Spearman 0.767** (informative!) |

So the mechanism has enormous headroom and the error model is sound — it was
simply pointed the wrong way. **`FPDetector`** (in `combine.py`) fixes this by
asking the correct question: *among predicted-positive pixels, is this one a
false positive?* Re-running with it. The claim below that "pruning only
subtracts and is strictly lossy" is **withdrawn**.

### Superseded tables (kept for the record)

Measured on `swin_lite`, **inverted-target** error model fit on val:

| target R | thr | unpruned | pruned | gain |
|---|---|---|---|---|
| *max-IoU reference* | 0.55 | — | **0.2552** | — |
| 0.80 | 0.010 | 0.2148 | 0.2172 | +0.0024 |
| 0.90 | 0.0014 | 0.1582 | 0.1585 | +0.0003 |
| 0.95 | 0.0006 | 0.0543 | 0.0542 | −0.0001 |
| 1.00 | 0.0002 | 0.0528 | 0.0527 | −0.0001 |

With the inverted target, every over-predicting configuration scored worse than
plain max-IoU thresholding, and GNN pruning gave exactly zero gain
(`graphsage` 0.1456→0.1456, `superpixel_gcn` 0.1528→0.1528).

**All of the above is an artefact of the wrong objective.** Re-run with
`FPDetector`, the same models and the same operating points instead gave
`swin_lite` **0.2675** and `gnn_graphsage` **0.2133** — both ABOVE their
max-IoU references. See the CONFIRMED section.

Two conclusions drawn at the time are **WITHDRAWN**, and neither should be
cited:

- ~~"Pruning only subtracts, so degrading the operating point and climbing back
  is strictly lossy."~~ False — the oracle prune reaches 0.810 from 0.146, so
  pruning has enormous headroom.
- ~~"Prefer mechanisms that add evidence over ones that filter."~~ Overgeneralised
  from the artefact. Filtering works fine when it is asked the right question.

The conditional GNN confidence sweep (task 20) was cancelled on the strength of
the bad numbers; with `FPDetector` the GNN reaches **0.2133**, above the 0.203
bar, so **that cancellation is reversed** — it qualifies.

## Key measured findings (drive the remaining design)

1. **Generalization-limited, not capacity-limited.** Train IoU 1.000 vs ~0.28
   held-out. More capacity does not help; more *data diversity* or better
   regularization would.
2. **Budget > architecture.** 3k→7.2k steps gave **+0.052** IoU; the best
   architecture swap at fixed budget gave **+0.028**. Sweeping 29 architectures
   was a worse use of compute than training one model longer.
3. **z-context — BOTH CLAIMS NOW IN DOUBT.** The "+10% at equal budget" 2.5D
   result is unbacked: its probability artifact is the 30k run's (collision
   bug), and re-scoring gives 0.2154 vs 2D's 0.2314 — the opposite sign.
   Post-hoc z-refinement measured +5.4% on `swin_lite` but **−6.2% on the
   shipped model**.
4. **z-agreement is a strong free confidence signal.** Where z-neighbours
   disagree, predictions are wrong ~62% of the time; where they agree, right
   ~94%. It isolates ~5% of pixels holding a disproportionate share of errors.
5. **Confidence cannot manufacture signal, and the OBJECTIVE decides whether it
   works at all.** On a degenerate base model (constant output) pruning
   recovered nothing. More importantly, regressing `|prob - gt|` is the WRONG
   objective for pruning — at a low threshold a true positive has the larger
   residual, so "prune high error" deletes chondrules. Asking instead *"is this
   predicted-positive pixel a false positive?"* (`FPDetector`) turns the same
   experiment from ~0 gain into +4.8% (CNN) and +19.6% (GNN).
6. **[SUPERSEDED — these were IN-SAMPLE]** Hierarchical dropping looked nearly free (Sky's idea, measured): a trivial
   region-intensity feature drops **39.7% (L1) → 68.4% (L5)** of regions at
   **perfect recall**, and the drop rate barely changes between R=1.000 and
   R=0.99 — empty regions are unambiguously empty. Ideal dropping leaves only
   **8.4%** of the image at pixel level (~12× compute reduction).
7. **GNNs genuinely underperform here** (~0.18 vs 0.28), and three different
   message-passing schemes agree — but they sit at only ~20% of their own
   0.878 ceiling, so the limit is node features, not quantization.

## Bugs caught that would have produced wrong conclusions

- **GNN feature standardization missing** (~20,000× scale disparity) → models
  collapsed to constants; would have been reported as "graph methods fail here".
- **Budget-run name collision** → 3k/15k/30k runs all wrote one leaderboard row,
  silently destroying the scaling curve they exist to measure.
- **Truncated threshold grids** (twice) → a "recall 1.00" experiment actually
  measured recall 0.69, and three different targets returned identical results.
- **Queue deadlocks** (twice) → streams waited on conditions that could never
  become true; work silently never ran.
- **Dead knobs** — `pos_weight` and `invert_target` were accepted, stored, and
  never applied.
- **Truncated threshold grid, third occurrence** — a claimed "hard recall ceiling of 0.709" was really the 2e-4 grid floor; true max recall is 0.996.
- **A retracted figure shipped as live** — make_figures plotted the pre-fix hierarchy (1.58x) after the fix (2.71x) existed; now reads hierarchy_fixed.json.
