# Meteorite Chondrule Segmentation

Automatically find **chondrules** (rocky/silicate inclusions) in meteorite
micro-CT scans — per-pixel *chondrule vs. matrix* segmentation — and pick the
best model by a **leakage-free** foreground-IoU sweep. First sample: Allan Hills
83010 (**ALH 83010**), a 531-slice micro-CT volume with a hand-labeled central
band.

> **Status: interim.** Best model so far = **IoU 0.203** (classical-ML) on a
> held-out region, vs. a 0.130 intensity-threshold floor. The practical
> annotation ceiling is **~0.85–0.92** (IoU >~0.95 is not reachable against
> hand-drawn boundaries). Deep nets (CNNs/GNN/transformers) are coded/queued —
> see the roadmap. Results describe **one meteorite**; no cross-meteorite claim.

## Why the evaluation is trustworthy
- **Leakage-free split** — train/val/test are contiguous z-slabs (never random
  slices); a buffer kills near-duplicate leakage and every chondrule that spans a
  seam is resolved to one slab (verified: 0 chondrules in ≥2 slabs). Both the
  split and the metrics were **independently audited leakage-free**.
- **Touch test once** — thresholds/hyperparameters are tuned on validation only;
  the test slab is scored once per locked model.
- **Honest reporting** — foreground IoU is always shown with Dice, precision,
  recall, and AUPRC, plus the evaluation regime; the ~96% background imbalance
  and N=1 caveat are stated everywhere.

## Results (test foreground IoU)
| Model | Tier | IoU | Dice | Prec | Rec |
|---|---|---|---|---|---|
| hist_gradient_boosting | classical-ML | 0.203 | 0.337 | 0.249 | 0.524 |
| extra_trees | classical-ML | 0.201 | 0.334 | 0.249 | 0.508 |
| random_forest | classical-ML | 0.195 | 0.326 | 0.228 | 0.569 |
| intensity threshold | baseline | 0.130 | 0.230 | 0.130 | 0.999 |

Annotation ceiling (`results/annotation_ceiling.json`): ±1px 0.92, ±2px 0.85,
adjacent-slice 0.83.

## Roadmap / stages
- [x] **v0** Data audit + frozen leakage-free split
- [x] **v1** Evaluation harness (metrics) + classical baseline
- [x] **Audits** eval + split confirmed leakage-free (adversarial)
- [x] **v2 Tier A/B** thresholds + classical-ML sweep (best IoU 0.203)
- [x] **Annotation ceiling** (~0.85–0.92)
- [ ] **v2 Tier C — CNNs** (2D / 2.5D / attention U-Net) — *coded, untrained*;
      + fix 2.5D train-seam, `pos_weight`, focal α; add regression tests
- [ ] **v2 Tier C+ — all NN families** MLP, patch-CNN, U-Net++, dilated CNN,
      ResNet-UNet, FCN/SegNet, DeepLabV3+, SegFormer/TransUNet, ConvLSTM over z,
      autoencoder
- [ ] **v2 Tier D — GNN** (superpixel graph)
- [ ] **Two-model combinations** recall+matrix-veto, disagreement, diverse
      ensemble (build all, keep the val-winner)
- [ ] **Recall-optimized operating point** (miss almost no chondrules)
- [ ] **Straddler-drop ablation** (prove the raw-input caveat is negligible)
- [ ] **v3 Cross-slice z-refinement** (confidence from z-neighbor agreement)
- [ ] **v3+ Learned confidence model** — a meta-model trained on the CV models'
      own mistakes to flag where predictions are trustworthy (REE-logD-style)
- [ ] **v4** Apply the best model to the unlabeled slices

## Run
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
