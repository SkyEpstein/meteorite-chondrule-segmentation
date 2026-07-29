# Project Constitution — Meteorite Chondrule Segmentation

**Version 0.2 · 2026-07-23 · Owner: Sky Epstein**

## Mission
Automatically locate **chondrules** (rocky/silicate inclusions) in meteorite
micro-CT scans by assigning every pixel a **calibrated probability of
chondrule vs. matrix**. First sample: Allan Hills 83010 (**ALH 83010**).
Success = the best model found by a **leakage-free sweep**, judged primarily by
**foreground (chondrule) Intersection-over-Union (IoU)** on a spatially
held-out region — so that unlabeled scans and regions can be segmented without
hand annotation. Secondary goals: **prioritize recall** (miss as few chondrules
as possible) and **quantify prediction confidence** so users know where to trust
the model.

## Scope
**In scope:** binary semantic segmentation (chondrule vs. not) on 2D CT slices
of one meteorite; a broad, *locally runnable* sweep across **all model
families** — classical thresholding, classical-ML on texture features, and the
**full neural-network range** (pixel-MLP, patch-CNN, U-Net / U-Net++ / attention
/ dilated, encoder–decoder with pretrained backbone, FCN / SegNet / DeepLabV3+,
transformer segmenters like SegFormer / TransUNet, ConvLSTM/ConvGRU over the
z-axis, reconstruction autoencoder) and a **superpixel GNN**; **two-model
combinations** (recall-specialist + matrix-veto, disagreement, diverse
ensemble); a **learned confidence model** trained on the CV models' own mistakes
(a trust map, in the spirit of the REE-logD confidence approach); a
**cross-slice z-consistency** confidence/refinement; a spatial hold-out
evaluation; and application to the unlabeled slices of this scan.

**Out of scope (for now):** instance separation of individual chondrules;
mineral-species classification (the labels do not encode species); any
cross-meteorite generalization claim; and any model that cannot run on the local
Mac — heavy / pretrained / transformer models are attempted when feasible and
otherwise **listed as excluded with the reason** (no silent caps).

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
PyTorch (Apple-MPS) for deep models (2D / 2.5D / transformer / GNN where
feasible) · matplotlib.
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
9. **Over-predict, then prune (Sky, 2026-07-23).** Errors are asymmetric: a
   missed chondrule is **unrecoverable** (nothing downstream can restore a
   region the model never proposed), while a false positive is **fixable** by
   the confidence/error meta-models. So the base model's operating point is
   chosen for **high recall**, and precision is recovered by pruning
   low-confidence / high-predicted-error pixels. The max-IoU threshold is kept
   only as a reference point, and "over-predict then prune" must be measured
   head-to-head against it rather than assumed better.
10. **Human-in-the-loop search.** Model search runs in **short rounds**, never as
   a long autonomous loop. Each round reports its **scores and exactly what was
   changed to produce them**, then waits for Sky's steer before the next round.
11. **Respect the machine.** Training/search must fit in RAM without swapping:
   the split is computed once and cached (`src/zones.py`), workers load only the
   zones they need (never test during a search), and concurrency defaults to
   **2 workers**. Four workers each rebuilding the split (~7 GB each) swamped
   the Mac on 2026-07-23.

## Roadmap
**The canonical, detailed roadmap lives in [`ROADMAP.md`](../ROADMAP.md)** —
per-stage status, measured findings, dependencies, and the bugs caught along the
way. Keep it current; the session task list is ephemeral and must not be the
only record. The summary below is the high-level arc only.

- **v0** ✅ Data audit & frozen leakage-free split.
- **v1** ✅ Evaluation harness (metrics) + classical baseline.
- **Audits** ✅ eval + split independently confirmed leakage-free.
- **v2 Tier A/B** ✅ thresholds + classical-ML sweep (best IoU 0.203).
- **Annotation ceiling** ✅ measured (~0.85–0.92; IoU ≈1 not reachable).
- **v2 Tier C** — CNNs (2D / 2.5D / attention U-Net); fix deferred items
  (2.5D train-seam, `pos_weight`, focal α, regression tests).
- **v2 Tier C+** — all remaining NN families (MLP, patch-CNN, U-Net++, dilated,
  ResNet-UNet, FCN/SegNet, DeepLabV3+, SegFormer/TransUNet, ConvLSTM over z,
  autoencoder).
- **v2 Tier D** — superpixel GNN + variants (GAT, GraphSAGE, GIN, ChebNet,
  ResGCN, GraphUNet), sharing one cached graph.
- **v2 Tier E — cascade (Sky):** GNN coarse regions → pixel-level CNN
  refinement, with the painted GNN map as a second input channel. The output is
  pixel-level, so this escapes the ~0.878 superpixel quantization ceiling that
  binds pure-GNN methods. Train-time GNN channels MUST be **cross-fitted**
  (out-of-fold) — a base model's predictions on its own training data are
  memorized and would teach the refiner to over-trust a channel that degrades
  at test time. Judged against both a plain CNN and the GNN alone.
- **Two-model combinations** — build all three; keep the val-winner.
- **Recall-optimized operating point** + straddler-drop ablation.
- **v3** — Cross-slice z-refinement (neighbor-agreement confidence).
- **v3+** — Learned confidence model (meta-model on the CV models' mistakes; a
  calibrated trust map).
- **v3+b** — **Confidence bake-off (Sky).** Runs **only after the base model is
  locked**: a confidence model fits its base model's specific error
  distribution, so the winner can change with the base model — running the
  bake-off early would tune it to failure modes the final system does not have.
  The meta-model is then chosen by a recorded bake-off, never by picking the
  first thing that works: several
  estimator families (gradient boosting, random forest, extra trees, linear,
  small MLP) for BOTH the trust-classifier and error-regressor tasks, ranked on
  validation. The bake-off MUST include the **free baselines** — `|p-0.5|`, the
  raw probability, and z-agreement — because a learned confidence model that
  cannot beat a zero-cost signal has not earned its complexity, and that result
  is reported rather than hidden. Ranked primarily by **downstream pruned IoU**
  (the quantity that matters), with ranking quality (AUROC / Spearman) reported
  alongside.
- **v3++** — **Learned error model** (Sky, proven on the REE-logD work): a
  *regression* meta-model predicting the **magnitude of the base model's error**
  per pixel, and aggregated per slice / per chondrule. Distinct from the trust
  classifier above — it is threshold-independent and yields an expected-error
  value (an error bar) rather than a correct/incorrect probability. Trained on
  VALIDATION-zone predictions only; judged by predicted-vs-actual error
  correlation and by whether it beats the classifier at selective prediction.
- **v4** — Apply the best model to the unlabeled slices; deliver probability
  maps + confidence.
