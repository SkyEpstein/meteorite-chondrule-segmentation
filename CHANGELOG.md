# Changelog

All notable decisions and milestones for the Meteorite Chondrule Segmentation
project. Honest numbers only; evaluation regime stated with every metric.

## [Unreleased]

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
