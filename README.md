# Meteorite Chondrule Segmentation

Automatically find **chondrules** (rocky/silicate inclusions) in meteorite
micro-CT scans — per-pixel *chondrule vs. matrix* segmentation — and pick the
best model by a **leakage-free** foreground-IoU sweep. First sample: Allan Hills
83010 (**ALH 83010**), a 531-slice micro-CT volume with a hand-labeled central
band.

> **Status.** Best model = **test IoU 0.3168** (`unet2d_s30000`, a plain 2D
> U-Net trained 30,000 steps); **0.3214** with FP-pruning + z-smoothing stacked.
> Baselines: classical-ML 0.203, intensity threshold 0.130. The practical
> annotation ceiling is **~0.85–0.92** (±1–2 px boundary tolerance), so IoU near
> 1.0 is not reachable against hand-drawn labels.
>
> **⚠️ 0.3168 AND 0.3214 ARE BOTH WITHDRAWN — read the KNOWN ISSUES block at the top of [CHANGELOG.md](CHANGELOG.md) before quoting anything here.** Three separate reasons: (1) they predate a threshold-grid fix; (2) 0.3214 was selected on **test**, not validation; and (3) **every CNN-family score was produced by a defective inference tiler** that accumulated its halo instead of cropping it. Retraining the headline model under the fixed tiler gives **test IoU 0.29409** (Dice 0.455, precision 0.372, recall 0.584, AUPRC 0.427) — **−0.0218** against the reproducible 0.31587. Note AUPRC *rose* 0.0755, so the map's ranking improved while IoU at the operating point fell. A residual tile-boundary artifact remains (37–201× the median jump at tile edges) and is documented, not fixed. The headline above is left unchanged deliberately: the retrained number is itself a tiled-inference number and the artifact is still open.
>
> **Other caveats:**
> - **Best-of-40, selection-biased.** The winner was picked from a 40-model
>   sweep, so its test score is the max of 40 noisy estimates — optimistic by
>   roughly 0.02–0.03. Thresholds were always fit on validation and the model was
>   never tuned on test, so this is *selection* bias, not threshold leakage. An
>   unbiased figure would need a holdout that took no part in selection.
> - **N = 1 meteorite** (ALH 83010, ~142 chondrules). Nothing here supports a
>   cross-meteorite claim.

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
| **`unet2d_s30000`** | ~~0.3168~~ **0.29409** | plain U-Net, 30k steps — best. 0.3168 is WITHDRAWN (defective tiler + stale threshold grid); 0.29409 is the retrain under the fixed tiler. Every other row below is still a pre-fix number. |
| `unet2d_s15000` | 0.3089 | same model, 15k steps |
| `long_p1_ctrl_b32` | 0.2834 | same model, 7.2k steps |
| `hrnet_lite` | 0.2598 | best of 40 architectures at a fixed 3k budget |
| `swin_lite` | 0.2552 | from-scratch transformer |
| `unet25d_k5` | 0.2547 | 2.5D (z-context), +10% over 2D at equal budget |
| `hist_gradient_boosting` | 0.203 | classical-ML on texture features |
| best GNN (`graphsage`) | 0.178 | superpixel graph |
| intensity threshold | 0.130 | non-learned floor |

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
| Architecture sweep (36 configs: CNNs, 4 from-scratch transformers, 6 modern conv, 7 GNNs) | done — 41 leaderboard rows |
| Budget-scaling curve (3k → 30k steps) | **needs re-running** — every point predates the tiler fix; only the 30k point has been re-scored (0.29409), so the curve now mixes regimes |
| Hierarchical superpixel cascade | done — 2.71x compute saving at 99.9% retention |
| GNN → CNN cascade | done — 0.2025, below a plain CNN |
| Confidence bake-off (10 learned + 4 free baselines) | done — nothing beat a free baseline by more than +0.0005 |
| FP-prune / z-refinement | done — help weak models, LOSE on the shipped model |
| Two-model is/is-not veto | done — precision 0.463 vs 0.412, IoU loss |
| Recall-oriented training bake-off | done — asymmetric losses hurt |
| Straddler-drop ablation | **incomplete — one arm only** |
| Apply to the ~130 unlabeled slices (v4) | **not done — model weights were never persisted** |
| Regression tests | **not done — `tests/` is empty** |
| Model checkpointing (`save_checkpoint`/`load_checkpoint`) | done — verified bit-identical reload; threshold travels with the weights |
| Multi-meteorite pipeline (`src/pipeline_v2.py`) | **plumbing only** — split self-test passes (13 straddlers resolved, 0 chondrules span 2+ zones), but `run_all` has never been executed and has no caller |
| Corrected data-scaling test (`src/data_scaling.py`) | **stopped at 1 of 12 runs, not running** (frac 0.25, test IoU 0.2823). Two earlier attempts produced nothing usable. The partial `results/data_scaling_v2.json` is deliberately uncommitted: `n_seeds: 1`, `test_iou_sd: NaN`. Do not quote until all 12 land |
| POC imagery (`src/poc_images.py`) | done — 4 figures from real saved predictions |

## Preparing for more data

The binding constraint is **measured to be data, not method**: 19 validation
chondrules give a bootstrap SE of ±0.0071 IoU and a **minimum detectable effect
of ~0.014 absolute IoU**, which is larger than nearly every method effect we
measured — six val/test comparisons flipped sign. `src/pipeline_v2.py` makes
adding a meteorite a config change (`MeteoriteSource` → `build_multi_split` →
`stage2_sweep` → `confidence_sweep`), reports the minimum detectable effect for
whatever validation size results, and flags any gain below it as
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
