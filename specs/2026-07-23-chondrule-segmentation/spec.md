# Spec — v0/v1: Evaluation Harness, Frozen Split & Classical Baseline

**Date:** 2026-07-23 · **Status:** draft (awaiting sign-off) ·
**Governs:** [constitution](../../memory/constitution.md)

## Problem
We have a micro-CT scan of ALH 83010 with a hand-labeled central band, and we
want to eventually sweep many CV models to find the best chondrule segmenter by
foreground IoU. **None of that is trustworthy without a leakage-free evaluation
harness first.** This milestone builds that harness — a frozen spatial split and
a metrics module — and establishes the first honest baseline number against
which every future model is judged.

## Goal
A reproducible, leakage-free evaluation setup plus a classical (non-learned)
baseline, so that v2's model sweep has a fair, fixed yardstick.

## Requirements

### R1 — Data loader & label binarization
- Pair each raw slice with its label slice by sorted z-order; assert 1:1 count
  and identical dimensions.
- Binarize labels: `mask = (label > 0)` → positive class "chondrule".
- Restrict the usable set to the **annotated band** (contiguous central run
  bounded by the long empty end-runs; expected ≈ slices 100–496). End runs and
  the isolated slice 0 are excluded from train/val/test (kept as unlabeled
  inference targets only). Interior zero slices *inside* the band count as valid
  matrix. The exact band bounds are computed from data and saved.

### R2 — Chondrule instances & straddler resolution
- Label chondrules as 3D connected components **constrained to the same
  annotator id** (`instance_label`), so touching-but-distinct chondrules are not
  fused into mega-clusters (a plain binary CC over-merges them). Measured on this
  band: **142 instances**, same-id z-extent median 24, p95 48, **max 70**
  band-slices (contrast: binary-CC over-merges to max 182 — not used).
- A small buffer (12 kept-slices ≈ 52 original) separates slabs for
  near-duplicate protection. The max instance extent (70) exceeds any affordable
  buffer, so the zero-straddle **guarantee comes from resolution, not buffer
  size**: every chondrule spanning ≥2 core zones is assigned to the slab holding
  most of its voxels; its voxels in other core zones are ignored (excluded from
  training and metrics). Verified: 0 chondrules keep labels in ≥2 slabs;
  ~2.6% of positive pixels ignored.

### R3 — Frozen split (contiguous z-slabs)
- Partition the band along z into **contiguous** train / val / test slabs with
  buffer gaps between adjacent slabs. val and test are placed in the
  chondrule-dense interior (val band-rel 80–139, test 160–239) so both carry
  representative positive mass; train is the remainder (two flanking pieces).
- Exact indices saved to `split.json` and never regenerated randomly; the build
  is deterministic (asserted bit-identical across runs).
- Emit the leakage check: number of chondrules keeping labels in ≥2 slabs after
  straddler resolution (must be **0**); guard that val/test zones are separated
  by ≥ buffer.

### R4 — Metrics module
- Given predicted probability map(s) + ground-truth mask(s), compute:
  **foreground IoU (primary)**, Dice/F1, precision, recall, and, for the
  probability output, **AUPRC** + a reliability (calibration) summary.
- Threshold for hard metrics is selected on **validation only**; test uses that
  frozen threshold. Micro-averaged over the slab's pixels; per-slice IoU
  distribution also reported.

### R5 — Classical baseline
- Intensity threshold (Otsu and a validation-tuned fixed threshold) on the raw
  CT, then morphological cleanup (open/close) and small-object removal.
- Threshold and morphology parameters tuned on validation; evaluated once on
  test.
- Optionally restrict to the sample interior (exclude pure-air background) if it
  helps precision — recorded as an ablation.

### R6 — Reproducibility & reporting
- Fixed seed; pinned key package versions logged.
- One command reproduces split → baseline → metrics; results written to
  `results/v1_baseline.json` and a qualitative overlay figure saved.

## Acceptance criteria
- **A1:** `split.json` exists, is deterministic across two runs, and the
  boundary-straddling chondrule count is **0**. *(R2, R3)*
- **A2:** Metrics module reproduces foreground IoU/Dice/precision/recall on a
  tiny hand-checked toy mask exactly. *(R4)*
- **A3:** Classical baseline produces a **foreground-IoU number on the test
  slab**, reported with Dice/precision/recall + calibration, with the
  validation-selected threshold frozen before test scoring. *(R4, R5)*
- **A4:** No random slice ever appears in more than one slab; the split is by
  contiguous z only. *(R3, constitution P1)*
- **A5:** A results JSON + overlay figure are saved and regenerable by one
  command. *(R6)*

## Non-goals (this milestone)
Learned models (RF/U-Net), the cross-slice confidence score, and inference on
the unlabeled ends — all deferred to v2/v3/v4.

## Open questions
- `[RESOLVED]` End runs → excluded as unlabeled (Sky, 2026-07-23).
- `[RESOLVED]` Split geometry → contiguous z-slab + buffer (Sky, 2026-07-23).
- `[RESOLVED]` Compute → local Mac only (Sky, 2026-07-23).
- `[NEEDS CLARIFICATION — provisional]` The 3-slice interior gap (≈135–137):
  treated as valid matrix for now; revisit only if it distorts baseline
  precision.
- `[DEFERRED]` Whether "non-labeled data" includes scans beyond this folder
  (affects v4 semi-supervised option) — assume inference-only for now.
