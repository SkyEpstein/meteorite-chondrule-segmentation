# Meteorite Chondrule Segmentation

Automatically find **chondrules** (rocky/silicate inclusions) in meteorite
micro-CT scans — per-pixel *chondrule vs. matrix* segmentation — and pick the
best model by a **leakage-free** foreground-IoU sweep. First sample: Allan Hills
83010 (**ALH 83010**), a 531-slice micro-CT volume with a hand-labeled central
band.

> **Status.** Current best test IoU is **0.33569** (`resunet18_scratch_s30000`,
> a ResNet-18 encoder with a U-Net decoder, trained from random init). The
> long-standing headline model `unet2d_s30000`, a plain 2D U-Net at 30,000
> steps, scores **0.29409**. Baselines: classical-ML 0.203, intensity threshold
> 0.130. The practical annotation ceiling is **0.85 to 0.92** (1 to 2 px
> boundary tolerance), so IoU near 1.0 is not reachable against hand-drawn
> labels.
>
> Both current numbers are n=1 seed. Neither is being promoted as a result: the
> gap between them sits inside the ambiguity band registered before they were
> measured, and this dataset cannot resolve differences that small (see
> "What this dataset can and cannot measure" below).
>
> **The previous headline of 0.3168, and 0.3214 with FP-pruning and z-smoothing
> stacked, are WITHDRAWN.** They were the front-page numbers until 2026-08-04.
>
> **Why 0.3168 and 0.3214 are withdrawn.** Three independent reasons: they
> predate a threshold-grid fix; 0.3214 was selected on **test**, not validation;
> and every CNN-family score was produced by a **defective inference tiler**
> that accumulated its halo instead of cropping it. Retraining the headline
> model under the fixed tiler gives test IoU **0.29409** (Dice 0.455, precision
> 0.372, recall 0.584, AUPRC 0.427), which is **-0.0218** against the
> reproducible 0.31587. AUPRC *rose* 0.0755, so the map's ranking improved while
> IoU at the operating point fell.
>
> **Read the KNOWN ISSUES block at the top of [CHANGELOG.md](CHANGELOG.md)
> before quoting any number from this repo.**
>
> **Caveats that apply to every number above.**
> - **n=1 seed.** The only measured seed variance in this project is SD 0.0084
>   (three seeds, 25% of train, 7,200 steps). Single runs are not separable at
>   the scale of most differences reported here.
> - **N=1 meteorite** (ALH 83010, roughly 142 chondrules). Nothing here supports
>   a cross-meteorite claim.
> - **Best-of-40, selection-biased.** The winner was picked from a 40-model
>   sweep, so its test score is the max of 40 noisy estimates, optimistic by
>   roughly 0.02 to 0.03. Thresholds were always fit on validation and no model
>   was tuned on test, so this is *selection* bias, not threshold leakage. An
>   unbiased figure would need a holdout that took no part in selection.
> - **A residual tile-boundary artifact remains**, 37 to 201 times the median
>   jump at tile edges. Documented, not fixed.
> - **0.33569 against 0.29409 does not isolate architecture.** `resunet18_*` is
>   BatchNorm throughout while `unet2d_s30000` uses GroupNorm
>   (`models_cnn._norm`), so that comparison crosses a normalisation change too.
> - The ImageNet-pretrained twin of the best model scored **0.30269**. The gap of
>   -0.033 is inside the 0.05 band registered before the numbers arrived, so it
>   is not evidence that pretraining hurts, and the scratch arm's val-frozen
>   threshold sits on the documented grid floor.

## Why the evaluation is trustworthy
- **Leakage-free split** — train/val/test are contiguous z-slabs (never random
  slices); a buffer kills near-duplicate leakage and every chondrule that spans a
  seam is resolved to one slab (verified: 0 chondrules in ≥2 slabs). Both the
  split and the metrics were **independently audited leakage-free**.
- **Touch test once, per model** — thresholds and hyperparameters are tuned on
  validation only; each model is then scored on the test slab exactly once. 40+
  models were scored this way to build the leaderboard, which is why the
  *winner's* score carries the selection-bias caveat above.
- **Honest reporting** — foreground IoU is always shown with Dice, precision,
  recall, and AUPRC, plus the evaluation regime; the ~96% background imbalance
  and N=1 caveat are stated everywhere.

## Results (test foreground IoU)

| Model | IoU | note |
|---|---|---|
| **`unet2d_s30000`** | ~~0.3168~~ **0.29409** | plain U-Net, 30k steps — best. 0.3168 is WITHDRAWN (defective tiler + stale threshold grid); 0.29409 is the retrain under the fixed tiler. Every row below carries its own provenance marker. |
| `unet2d_s15000` | 0.3089 | [PRE-FIX] same model, 15k steps |
| `long_p1_ctrl_b32` | 0.2834 | [PRE-FIX] same model, 7.2k steps |
| `hrnet_lite` | 0.2598 | [PRE-FIX] best of 40 architectures at a fixed 3k budget. [UPDATED 2026-08-04] The 40 over-counts by one: `unet2d` and `rt_dice_ctrl` are the same configuration at the same seed and appear as two rows |
| `swin_lite` | 0.2552 | [PRE-FIX] from-scratch transformer |
| `unet25d_k5` | 0.2547 | [PRE-FIX] 2.5D (z-context), +10% over 2D at equal budget. **CLAIM WITHDRAWN**, artifact corrupted: re-scoring it gives 0.2154, not 0.2547 (see CHANGELOG KNOWN ISSUES) |
| `hist_gradient_boosting` | 0.203 | classical-ML on texture features. Classical, not affected by the tiler |
| best GNN (`graphsage`) | 0.178 | superpixel graph. Bypasses the tiler, unaffected |
| intensity threshold | 0.130 | non-learned floor. Not affected by the tiler |

The [PRE-FIX] marker is repeated per row on purpose: a row quoted on its own,
away from the warning above, must still carry its own caveat.

**Training budget beat architecture.** The same U-Net at 3k / 7.2k / 15k / 30k
steps scores 0.2314 / 0.2834 / 0.3089 / ~~0.3159~~ — **all four WITHDRAWN** (they
predate the tiler fix; the 30k point re-scores to 0.29409). Gains halve each doubling and
have now plateaued. No architecture at a fixed budget came close.

Annotation ceiling (`results/annotation_ceiling.json`): ±1px 0.92, ±2px 0.85,
adjacent-slice 0.83.

## Status of each stage

| stage | state |
|---|---|
| Leakage-free split + 2 adversarial audits | done — 0 chondrules span slabs |
| Evaluation harness + classical baseline | done — 0.130 floor, 0.203 classical ML |
| Annotation ceiling | done — ~0.92 (±1px), ~0.85 (±2px) |
| Architecture sweep (36 configs: CNNs, 4 from-scratch transformers, 6 modern conv, 7 GNNs) | done — ~~41~~ **44** leaderboard rows [UPDATED 2026-08-04]: 4 post-tiler-fix, 33 pre-fix, 7 tiler-independent |
| Budget-scaling curve (3k → 30k steps) | **needs re-running** — every point predates the tiler fix; only the 30k point has been re-scored (0.29409), so the curve now mixes regimes |
| Hierarchical superpixel cascade | done — 2.71x compute saving at 99.9% retention |
| GNN → CNN cascade | done — 0.2025, below a plain CNN |
| Confidence bake-off (10 learned + 4 free baselines) | done — nothing beat a free baseline by more than +0.0005 |
| FP-prune / z-refinement | done — help weak models, LOSE on the shipped model |
| Two-model is/is-not veto | done — precision 0.463 vs 0.412, IoU loss |
| Recall-oriented training bake-off | done — asymmetric losses hurt |
| Straddler-drop ablation | **incomplete — one arm only** |
| Apply to the ~130 unlabeled slices (v4) | **not done — model weights were never persisted** |
| Regression tests | **partial** [UPDATED 2026-08-04]. Two files ship: `tests/test_aux_channels.py` (the N-channel aux input path, including bit-identical equivalence for a single-channel aux) and `tests/test_multi_meteorite.py` (multi-meteorite ingestion: padding to a common in-plane size, padding marked invalid, per-slice provenance). Neither covers the noise floor or any model's accuracy |
| Model checkpointing (`save_checkpoint`/`load_checkpoint`) | done — verified bit-identical reload; threshold travels with the weights |
| Multi-meteorite pipeline (`src/pipeline_v2.py`) | **plumbing only** [UPDATED 2026-08-04]. The split self-test passes (13 straddlers resolved, 0 chondrules span 2+ zones), and the split now handles multiple sources including two scans of different in-plane size (padded to the common in-plane size, with the padding marked invalid), verified by `tests/test_multi_meteorite.py`. `run_all` with two sources has still never completed end to end, and still has no caller outside its own module docstring |
| Corrected data-scaling test (`src/data_scaling.py`) | **3 of 12 runs, not running** [UPDATED 2026-08-04]. Fraction 0.25 is complete at 3 seeds (test IoU 0.2823 / 0.2664 / 0.2696, `n_seeds: 3`, `test_iou_sd` about 0.00841); fractions 0.5, 0.75 and 1.0 stand at 0 of 3 each, so 9 runs remain, roughly 5 hours. `results/data_scaling_v2.json` is committed. Two earlier attempts produced nothing usable. Do not quote a slope until all 12 land |
| POC imagery (`src/poc_images.py`) | done — 4 figures from real saved predictions |

## Preparing for more data

The binding constraint is **measured to be data, not method**: 19 validation
chondrules give a bootstrap SE of ±0.0071 IoU and a **minimum detectable effect
of ~0.014 absolute IoU**, which is larger than nearly every method effect we
measured — nine val/test comparisons flipped sign.

**[CORRECTED 2026-08-04] "19 validation chondrules" is the wrong label for that
number.** `src/recount_chondrules.py` counted the val zone both ways over
identical voxels: 19 is the BINARY connected-components count, and the
pipeline's own same-id rule gives 36. The constant `SE_REF_N` was left at 19,
because the SE it pairs with (±0.0071) was bootstrapped over 60 validation
SLICES, so neither chondrule count was ever the resampling unit. The 0.0071 and
the ~0.014 floor are unchanged, and the conclusion that data is the binding
constraint stands.

**[CORRECTED 2026-08-04] The sign-flip count above read six when the paragraph
was written.** It is nine as of 2026-08-04; the pretraining probe was the ninth.

**[CORRECTED 2026-08-04] `src/pipeline_v2.py` does NOT make adding a meteorite a
config change.** Two scans of different in-plane size crashed the ingestion path,
which needed a fix (pad to the common in-plane size, with the padding marked
invalid), and `run_all` with two sources has still never completed end to end.
What the path does do today: `MeteoriteSource`, `build_multi_split`,
`stage2_sweep` and `confidence_sweep` report the minimum detectable effect for
whatever validation size results, and flag any gain below it as
`below_noise_floor` rather than reporting it as a finding.

See the KNOWN ISSUES block at the top of [CHANGELOG.md](CHANGELOG.md) before
quoting any number.

## Run

> **Reproducibility limit:** `results/probs/` (~13 GB of per-model probability
> maps) and `results/graph_cache/` are gitignored, so the post-hoc pipeline
> (`src/run_final.py`, which globs `results/probs/*_val.npy`) is **not runnable
> from a clone** without first retraining the models. The JSON results and
> figures capture every reported number, but they cannot be independently
> re-derived from this repository alone.

```bash
pip install -r requirements.txt          # + a torch build from pytorch.org
python run_v1.py                         # split + classical baseline + figure
python -m src.sweep                      # Tier A+B leaderboard
python -m src.ceiling                    # annotation ceiling
```
Data (`ALH83010_z_crop_every4[_labeled]/`) is not tracked in git.

## Layout
`memory/constitution.md` (principles) · `specs/` (specs) · `src/` (data, split,
metrics, features, baseline, sweep, models_cnn, ceiling) · `results/` (JSON) ·
`CHANGELOG.md`.
