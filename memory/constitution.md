# Project Constitution — Meteorite Chondrule Segmentation

**Version 0.1 · 2026-07-23 · Owner: Sky Epstein**

## Mission
Automatically locate **chondrules** (rocky/silicate inclusions) in meteorite
micro-CT scans by assigning every pixel a **calibrated probability of
chondrule vs. matrix**. First sample: Allan Hills 83010 (**ALH 83010**).
Success = the best model found by a **leakage-free sweep**, judged primarily by
**foreground (chondrule) Intersection-over-Union (IoU)** on a spatially
held-out region — so that unlabeled scans and regions can be segmented without
hand annotation.

## Scope
**In scope:** binary semantic segmentation (chondrule vs. not) on 2D CT slices
of one meteorite; a broad but *locally runnable* model sweep; a spatial
hold-out evaluation; a cross-slice-consistency confidence score; application to
the unlabeled slices of this scan.

**Out of scope (for now):** instance separation of individual chondrules;
mineral-species classification (the labels do not encode species); any
cross-meteorite generalization claim; GPU/cluster-scale 3D models.

## Data (ground truth)
- One micro-CT volume — 531 slices, 1450×962, 8-bit grayscale
  (`ALH83010_z_crop_every4/`), every 4th slice of a ~2120-slice scan.
- Paired hand annotation (`ALH83010_z_crop_every4_labeled/`), 8-bit, where each
  integer > 0 marks an individual chondrule instance. **For this project all
  values > 0 collapse to the single positive class "chondrule".**
- **Annotated region = contiguous central band ≈ slices 100–496.** The end runs
  (≈ 1–99 and 497–530) are treated as **UNLABELED / unknown — never as
  negatives** — and used only as inference targets.
- Chondrules are **denser (brighter) than matrix** in CT (mean ≈ 50 vs. 32),
  so raw intensity carries real signal.
- `BP.CSV` in the project root is an unrelated hydrocarbon table — ignore it.

## Tech stack
Python (Anaconda) · numpy · tifffile · scikit-image · scikit-learn ·
PyTorch (Apple-MPS) for lightweight deep models · matplotlib.
**Local Mac only** (CPU / MPS), no cloud dependency. Reproducible: pinned
environment, fixed seeds, split saved to disk.

## Non-negotiable principles
1. **Leakage-free split.** Train / val / test are **contiguous z-slabs, never
   random slices.** A buffer gap (12 kept-slices ≈ 48 original slices) separates
   slabs to prevent *near-duplicate* leakage between adjacent slices. Zero
   chondrule straddling is then **guaranteed at the label level** by resolving
   every chondrule that spans a seam to the slab holding most of its voxels and
   ignoring its voxels elsewhere — verified independently by a same-id
   connected-component recount (0 straddlers, ~2.6% of positives ignored). The
   split is deterministic and saved to disk. *(Known, measured caveat for
   learned models: a resolved chondrule's raw pixels remain visible — unlabeled
   — in the ≥48-slice-distant neighbor slab; v2 quantifies this with a
   drop-straddlers-entirely ablation.)*
2. **Touch test once.** All tuning and model selection happen on validation; the
   test slab is scored only for the final locked-model report.
3. **Honest negatives.** Un-annotated regions are never used as negative labels;
   only the annotated band supplies ground truth.
4. **Baseline first.** A classical intensity/morphology baseline is built and
   scored before any learned model; every model is compared on the *identical*
   split via a recorded bake-off.
5. **Report metrics together.** Primary = foreground IoU, always reported
   alongside Dice/F1, precision, recall, and probability calibration
   (reliability + AUPRC), given the ~96% class imbalance.
6. **N = 1 honesty.** Results describe ALH 83010 only. No cross-meteorite claim
   until more samples exist.
7. **Reproducibility.** Seeds fixed; environment pinned; splits, thresholds, and
   metrics written to disk with every run.
8. **SDD every change.** Each change gets a spec note (problem / approach /
   acceptance), a verification pass, and a CHANGELOG entry — scaled to size.

## Roadmap
- **v0** Data audit & frozen split — measure chondrule z-extent, fix the band,
  save the train/val/test slabs.
- **v1** Classical baselines — threshold + morphology + size filtering; first
  honest foreground-IoU number on the test slab.
- **v2** Sweep — classical-ML (random forest on texture features) + lightweight
  2D / 2.5D U-Net; bake-off by foreground IoU on validation.
- **v3** Cross-slice consistency — neighbor-agreement confidence + optional
  z-refinement, measured against the raw 2D prediction.
- **v4** Apply the best model to the unlabeled slices; deliver probability maps +
  confidence.
