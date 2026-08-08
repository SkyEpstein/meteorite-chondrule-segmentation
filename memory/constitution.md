# Project Constitution — Meteorite Chondrule Segmentation

**Version 0.3 · 2026-07-29 · Owner: Sky Epstein**

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

**[AMENDED 2026-08-04] "one meteorite" now describes the dataset, not the
ingestion path.** Multi-meteorite ingestion shipped: `build_multi_split` accepts
several `MeteoriteSource` entries, pads them to a common in-plane size with the
padding marked invalid, and records each source's `voxel_um`
(`tests/test_multi_meteorite.py` covers this). `run_all` with two sources has
still never completed end to end. Only §12 was amended when this landed; this
Scope paragraph was missed.

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
12. **The complete model (Sky, 2026-07-29).** From this point forward "the
   complete model" names exactly one thing — the ordered composition of seven
   pluggable stages, **in execution order (Sky, 2026-07-30)**:

   ```
   (1) HIERARCHY FILTER          -> alive mask; runs FIRST
   (2) CLASSICAL SEGMENTER       -> denoise/CLAHE/top-hat/watershed probability
   (3) CNN                       -> raw, plus a classical channel IF the
                                    cascade arm wins; see UNSETTLED below
   (4) CONFIDENCE: regressors    -\
   (5) CONFIDENCE: error U-Net    +-> three independent error signals
   (6) Z-NEIGHBOUR CONFIDENCE    -/
   (7) PIXEL DROPPING (prune)    -> LAST; stacks all three into one score
   ```

   **The hierarchy runs FIRST** — its whole purpose is compute saving, and a
   prefilter after the segmenter saves nothing. (An earlier version of this
   clause numbered the base segmenter "(0)" ahead of it, which read backwards;
   the code always did it correctly via `HierMaskedRunner`.)

   **[UNSETTLED, do not read this as decided]** Whether the classical map is an
   INPUT CHANNEL to the CNN (the Tier E cascade pattern) or an alternative Stage
   2 candidate is **not decided**. Offered the choice on 2026-07-30, Sky declined
   both and chose "**both, and let validation decide**": build (a) a CNN taking
   the cleaned channels directly, and (b) a CNN cascaded on the classical
   probability, then compare on validation. **Neither arm has been run.** An
   earlier version of this clause asserted the cascade reading as binding, which
   recorded a decision Sky had explicitly refused to make and foreclosed the
   comparison he asked for.

   **[LEAKAGE][MUST] IF the cascade arm is built, its channel must be
   CROSS-FITTED out-of-fold over the train slices.** A base model's predictions
   on its own training data are memorised, and feeding those to the refiner
   teaches it to over-trust a channel that is weaker at inference. This is the
   same requirement Tier E imposes on the GNN to CNN cascade.
   `src/classical_crossfit.py` produces such a channel; measured, the
   memorisation it guards against is small for gradient boosting (+0.0081
   in-fold vs out-of-fold) but is expected to be much larger for the CNN.

   **Pruning happens LAST and once**, consuming all three confidence signals
   stacked into a single val-fit score, rather than pruning three times in
   sequence — sequential pruning compounds errors and can only ever remove
   pixels, never recover one an earlier stage wrongly dropped.
   **[DEFINITION, Sky 2026-07-30] "Confidence model" and "learned error model"
   mean the SAME THING in this project — stage (5) IS the error model: a regression
   on the MAGNITUDE of the base model's error, threshold-independent, yielding an
   error bar. One stage, not two.** The trust classifier
   (`combine.ConfidenceModel`, a correct/incorrect probability) survives only as a
   COMPARISON arm in the bake-off next to the free baselines; it is not a separate
   deliverable, and v3+/v3++ in the roadmap below describe one thing.
   Its headline number
   is **end-to-end test foreground IoU** against the plain-U-Net-at-matched-budget
   baseline — reported alongside Dice, precision, recall and calibration **as §5
   requires**; a bare IoU does not satisfy this clause. **Every stage must be
   ablatable** so a stage that does not earn its place is visible rather than
   assumed. This is an **architecture contract, not a performance claim**: at
   today's data volume (MDE 0.0142 IoU at n=19 val chondrules. **[CORRECTED
   2026-08-04]** 19 is the val zone's BINARY connected-components count, not the
   chondrule count that label implies; the pipeline's own same-id rule gives 36
   over identical voxels, measured by `src/recount_chondrules.py`. `SE_REF_N` was
   left at 19 deliberately, because the SE it pairs with, `SE_REF_IOU = 0.0071`,
   was bootstrapped over 60 validation SLICES, so neither count was ever the
   resampling unit) the stages are
   individually indistinguishable from break-even — the hierarchy measured
   −0.012, *inside* that floor, and is justified on **compute** (2.41× at target
   recall 0.999), not accuracy. **[REGIME] Those two figures, and every
   CNN-family number they rest on, were measured under TILED inference — which
   remains the default. Its halo-accumulation defect was fixed on 2026-07-29 (so
   the numbers are being re-measured and may move), but a residual per-window
   GroupNorm dependence is ACCEPTED and documented: measured on the retrained
   model, positive-pixel counts jump 37-88x the median at x = 256 / 512 / 768 /
   1024 and 201x at y = 512 (x = 1280 shows no jump) — i.e. at MOST but not every
   tile
   boundary. A whole-slice replacement was tried and reverted — it collapsed
   best-epoch val IoU on a 25%-data run from 0.4487 to 0.2632, with epochs 8-12
   at 0.0000-0.0075. [REGIME] Two SEPARATE training runs at the same config and
   seed, not one model scored two ways — the same-weights comparison was never
   run.** The base segmenter's
   "high-recall operating point" is inherited from §9. It was NOT achieved until
   2026-07-30, when `THR_GRID` was extended to 1e-7: recall 0.99 is now reached
   on the control (achieved 0.9910). It is still NOT reached on the augmented arm
   (achieved 0.9355, flagged unreachable), which is a comparability defect in any
   bake-off across those two arms. No stage may
   hard-code ALH 83010; the entry point stays `run_all(sources)` over
   `MeteoriteSource`. **[CORRECTED 2026-08-04] The claim that more meteorites are
   therefore "an argument, not a code change" was FALSE.** `pipeline_v2.py`
   crashed on two scans of different in-plane size, so the ingestion path itself
   needed a fix (pad to the common in-plane size, with the padding marked
   invalid). `run_all` with two sources has still never completed end to end.
   Full spec: [`specs/2026-07-29-complete-model-contract/spec.md`](../specs/2026-07-29-complete-model-contract/spec.md).

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
- **v3++** — **Learned error model = THE confidence model** (Sky, proven on the
  REE-logD work; and per §12's definition note, v3+ and v3++ are ONE stage, with
  the trust classifier demoted to a comparison arm): a
  *regression* meta-model predicting the **magnitude of the base model's error**
  per pixel, and aggregated per slice / per chondrule. Distinct from the trust
  classifier above — it is threshold-independent and yields an expected-error
  value (an error bar) rather than a correct/incorrect probability. Trained on
  VALIDATION-zone predictions only; judged by predicted-vs-actual error
  correlation and by whether it beats the classifier at selective prediction.
- **v3+++** — **The complete model** (§12): compose the seven stages of §12 in
  `pipeline_v2.run_all`, score end-to-end with per-stage ablations, and carry the
  contract unchanged onto the larger multi-meteorite dataset.
- **v4** — Apply the best model to the unlabeled slices; deliver probability
  maps + confidence.
