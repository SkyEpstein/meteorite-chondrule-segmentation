# Spec — v2: Broad Model Sweep

**Date:** 2026-07-23 · **Status:** in progress ·
**Governs:** [constitution](../../memory/constitution.md) · **Builds on:** v0/v1 harness

## Problem
v1 gives a leakage-free split + metrics + a classical floor (test IoU ≈ 0.13,
which floods: P=0.13 / R=0.998). We now sweep a broad set of model families to
find the best chondrule segmenter by foreground IoU on the frozen test slab.
Directive (Sky): run every practical model family, explicitly including CNNs
and GNNs.

## Approach
A uniform sweep framework: every model exposes `fit(train)` and
`predict_proba(slices) -> [0,1] map`; the runner selects its operating
threshold on **validation** and reports it **once** on test. All models share
the identical frozen split and the v1 metrics.

### Roster
- **A. Non-learned:** tuned global threshold (v1), Otsu, multi-Otsu, local
  adaptive threshold.
- **B. Classical ML (per-pixel, multiscale texture features):** logistic
  regression, Gaussian NB, random forest, extra-trees, hist-gradient-boosting.
- **C. CNNs (PyTorch/MPS, patch-based):** small patch-CNN; 2D U-Net with
  BCE / Dice / Tversky / focal losses; attention U-Net; **2.5D U-Net**
  (adjacent slices as input channels).
- **D. GNN:** SLIC superpixels → region-adjacency graph → GCN / GraphSAGE node
  classification → pixel map.

### Features (tier B & GNN nodes)
Per-slice, multiscale: raw intensity, Gaussian(σ=1,2,4,8), gradient magnitude,
Laplacian-of-Gaussian, local mean/std, Gabor, Hessian eigenvalues (~25 dims).
Classical models train on a balanced pixel subsample; predict full slices.

## Requirements
- **R1** All models scored on the identical frozen split (v1 `split.json`),
  ignore-mask honored; threshold tuned on val, frozen for test.
- **R2** Report per model: foreground IoU (primary), Dice, precision, recall,
  AUPRC, ECE, train/infer seconds, param/compute notes. One ranked leaderboard.
- **R3** Deep/GNN training uses **train slabs only**; val for early-stop /
  threshold / model selection; test touched once per locked model.
- **R4** Determinism: fixed seeds; leaderboard + per-model probs/figures saved.
- **R5** No silent caps: any excluded model (e.g. SAM/large transformers) is
  listed with the reason.
- **R6 — Two-model combinations (Sky's idea; build all three, keep the winner).**
  Only combinations of *genuinely different* models count (a flipped-label copy
  is B≈1−A, a no-op). Build and measure on **val** against the honest baseline
  of one model with two thresholds:
  (a) **recall-specialist + matrix-veto** — A high-recall proposer, B a matrix
  specialist trained on hard bright-tail negatives; keep pixels where A=chondrule
  AND B≠confident-matrix;
  (b) **disagreement-as-confidence** — agreement=confident, disagreement=fuzzy
  boundary (composes with R7-style z-confidence);
  (c) **diverse ensemble** — different features/losses, averaged or stacked.
  Keep only combinations that beat the single-model two-threshold baseline on val;
  report the winner on test once.
- **R7 — Recall-optimized operating point.** Alongside the max-IoU threshold,
  report a high-recall operating point (miss almost no chondrules) with its
  precision/IoU, per Sky's "don't-miss-chondrules" pick.
- **R8 — Annotation-consistency ceiling.** Estimate the max IoU any model could
  reach against these labels (boundary-tolerance IoU under ±1–2px morphological
  slop; z-interpolation oracle from neighbor slices). Gives a concrete "how close
  to 1 is even possible" number.
- **R9 — Straddler-policy ablation.** Re-run the best model with straddler
  chondrules dropped entirely (raw pixels excluded, not just labels) and compare
  test IoU, to prove the majority-assignment raw-input caveat is negligible.
- **Tier D (GNN)** is queued after Tier C validates on this harness.
- **R10 — All neural-network families (Sky).** Beyond the named CNN/GNN, sweep a
  broad NN roster: pixel-MLP, patch-CNN, U-Net / U-Net++ / attention U-Net,
  dilated/atrous CNN, encoder-decoder with pretrained backbone (ResNet-UNet),
  FCN / SegNet, DeepLabV3+, transformer segmenters (SegFormer / TransUNet),
  ConvLSTM/ConvGRU over the z-axis (exploits slice continuity), reconstruction
  autoencoder (anomaly-style), and the GNN. Heavy/pretrained/transformer models
  that need large downloads or exceed local-MPS budget are run if feasible and
  otherwise **listed as excluded with the reason** (no silent caps). Each is
  scored on the same frozen split by foreground IoU.
- **R11 — Learned confidence model (Sky; cf. REE logD confidence).** Train a
  separate meta-model on the CV models' OWN mistakes to predict per-pixel (and
  per-region) trustworthiness. Inputs: base-model probability, texture features,
  cross-slice z-agreement, two-model disagreement, local stats. Target: whether
  the base prediction is correct (residual/error). Output: a calibrated
  confidence/trust map. Trained on TRAIN/VAL predictions only; evaluated by
  whether high-confidence pixels are actually more accurate (selective-prediction
  risk–coverage curve). Distinct from, and composes with, the cross-slice
  z-refinement (R7-style) and the two-model disagreement (R6).
- **R12 — Learned ERROR model (Sky; proven on REE-logD).** A *regression*
  meta-model predicting the **magnitude of the base model's error**, not a
  binary correct/incorrect. Target: the per-pixel residual `|prob - gt|`
  (equivalently the soft error), plus aggregated per-slice and per-chondrule
  error. Inputs: the same confidence features as R11. Trained on
  VALIDATION-zone predictions only (never train — the base model memorizes
  train). Evaluated by (a) Spearman/Pearson correlation between predicted and
  actual error, (b) calibration of predicted vs realized error in bins, and
  (c) a risk-coverage curve compared head-to-head with the R11 classifier.
  Rationale for keeping both: the classifier answers "can I trust this pixel?",
  the regressor answers "how wrong is this likely to be?" — the latter is
  threshold-independent and gives an error bar, which is what makes it useful
  for flagging whole regions/slices for human review.

## Acceptance criteria
- **A1** A leaderboard (`results/v2_leaderboard.json` + table) ranks every run
  by test foreground IoU, all on the same split, with the val-selected
  threshold frozen. *(R1,R2)*
- **A2** At least tiers A+B complete and reported; C+D added on the same
  framework. *(roster)*
- **A3** Every reported number states its regime; excluded models are named
  with reasons. *(constitution honesty; R5)*
- **A4** Deep/GNN runs demonstrably train on train-only (no val/test in fit);
  spot-checked. *(R3)*

## Honest caveats (recorded up front)
- MPS ≪ CUDA: deep nets are patch-based, modest epochs — **compute-limited, not
  tuned SOTA**. Labeled as such.
- N=1 meteorite: leaderboard describes ALH 83010 only.
- Numbers provisional until the v1 split/metrics adversarial audit clears.
