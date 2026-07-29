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
> **⚠️ Read the KNOWN ISSUES block at the top of [CHANGELOG.md](CHANGELOG.md) first — 0.3168 and 0.3214 do NOT reproduce under the committed code (they predate a threshold-grid fix), and 0.3214 was selected on test, not validation.**
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
| **`unet2d_s30000`** | **0.3168** | plain U-Net, 30k steps — best |
| `unet2d_s15000` | 0.3089 | same model, 15k steps |
| `long_p1_ctrl_b32` | 0.2834 | same model, 7.2k steps |
| `hrnet_lite` | 0.2598 | best of 40 architectures at a fixed 3k budget |
| `swin_lite` | 0.2552 | from-scratch transformer |
| `unet25d_k5` | 0.2547 | 2.5D (z-context), +10% over 2D at equal budget |
| `hist_gradient_boosting` | 0.203 | classical-ML on texture features |
| best GNN (`graphsage`) | 0.178 | superpixel graph |
| intensity threshold | 0.130 | non-learned floor |

**Training budget beat architecture.** The same U-Net at 3k / 7.2k / 15k / 30k
steps scores 0.2314 / 0.2834 / 0.3089 / 0.3159 — gains halve each doubling and
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
| Budget-scaling curve (3k → 30k steps) | done — plateaued at 0.3159 |
| Hierarchical superpixel cascade | done — 2.71x compute saving at 99.9% retention |
| GNN → CNN cascade | done — 0.2025, below a plain CNN |
| Confidence bake-off (10 learned + 4 free baselines) | done — nothing beat a free baseline by more than +0.0005 |
| FP-prune / z-refinement | done — help weak models, LOSE on the shipped model |
| Two-model is/is-not veto | done — precision 0.463 vs 0.412, IoU loss |
| Recall-oriented training bake-off | done — asymmetric losses hurt |
| Straddler-drop ablation | **incomplete — one arm only** |
| Apply to the ~130 unlabeled slices (v4) | **not done — model weights were never persisted** |
| Regression tests | **not done — `tests/` is empty** |

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
