# Spec: audit of external papers and their split protocols

Date: 2026-07-30
Status: complete, with R6 and A3 retracted after verification (see below)
Owner: Sky Epstein
Governs: everything under `Audits of Other Papers/`

## Problem

This project reports test IoU 0.3168 for a 2D U-Net on ALH 83010 micro-CT under a
contiguous z-slab split with a 12 band-slice buffer and straddler resolution. That number
looks weak next to the published literature on geological and petrographic image
segmentation, where reported scores of 0.80 to 0.91 IoU are common.

The suspicion driving this audit: the gap is not mostly method quality, it is mostly
evaluation regime. Papers in this space overwhelmingly split randomly at the image or
patch level. When every image comes from a small number of physical samples, a random
split puts near-duplicate views of the same physical object in both train and test, and
the reported score measures memorization as much as generalization.

An unaudited comparison against those numbers would make this project look bad for the
wrong reason, and would also let us adopt a leaky protocol later to "catch up".

## Scope

In scope:

- Chondrule and meteorite computer vision as the core corpus: any paper that segments,
  detects, or counts chondrules or meteorite phases in micro-CT, BSE/SEM, or optical
  imagery.
- For each paper: extract the exact split protocol, the dataset structure, and every
  reported metric, then audit the numbers for internal consistency and for claims the
  evidence cannot support.
- Reconstruct each distinct split protocol on the ALH 83010 volume and measure what it
  reports, crossed against architecture.

Out of scope:

- Obtaining and rerunning on the papers' own data. Chondrule CV datasets are essentially
  never public, so a like-for-like rerun is not available. This is a stated limitation,
  not a silent one.
- Any claim that a specific paper's specific number is wrong on its own data. The claim
  this audit can support is narrower and is stated in Acceptance below.

## Requirements

R1. A verified corpus. Every paper cited carries an author list, year, venue, and a DOI
or arXiv ID that was fetched and checked. Any citation that cannot be verified is either
dropped or marked UNVERIFIED in place. No paper enters the report on an agent's summary
alone.

R2. Split extraction. For each paper, record the split protocol verbatim where the text
permits, and classify it into a protocol family. Where the paper does not state the
split, record UNSPECIFIED as the finding rather than guessing.

R3. Protocol reconstruction. Implement each observed protocol family against the ALH
83010 band so it can be run with the existing model runners:

- `P-slice-random`: randomly assign the 397 band slices to train/val/test.
- `P-tile-random`: cut the band into 2D tiles and randomly assign tiles.
- `P-patch-random-overlap`: randomly positioned, freely overlapping 2D patches.
- `P-zone-nobuffer`: contiguous z zones, buffer 0, no straddler resolution.
- `P-zone-buffer`: contiguous z zones, buffer 12, straddlers NOT resolved.
- `P-zone-buffer-straddler` (this project's reference): buffer 12 plus straddler
  resolution.

RETRACTED from this requirement: a `P-patch3d-random` protocol sampling genuine 3D patches
was specified and was never implemented. `P-patch-random-overlap` samples 2D patches, not
3D volumes, and is not a substitute. This matters because Mahmud 2023, the corpus paper
whose data structure most resembles this project's (448 volumes of 512^3 split 80:10:10 at
random), is the one a 3D-patch protocol would have reproduced. No claim in this audit rests
on a 3D-patch measurement, and any sentence implying one has been corrected.

R4. Leakage diagnostics per protocol, computed and reported alongside every score:
number of chondrule instances with voxels in both train and test, fraction of test
positive pixels belonging to such instances, and the minimum z-distance from any test
slice to the nearest train slice.

R5. Crossed design. Each protocol runs against each selected architecture at a fixed
training budget, so protocol effect and architecture effect are separable. The budget is
identical across all cells and is stated in the report.

R6. Dose-response observation. Run a buffer sweep on a fixed test zone with a fixed
training-set size. RETRACTED as originally written: this was specified as a confound-free
control and it is not one. Verification measured that nearest-first slice selection makes
the training positive fraction fall from 0.07475 to 0.05171 monotonically with the buffer,
collinear with the treatment. The sweep is kept as a covariate-annotated observation, with
`train_pos_frac` recorded at every point, and the causal claim is withdrawn. It is not
redesigned, because the audit's remit is to run published protocols rather than to
construct new ones.

R6a. Replication of the headline pair. `P-slice-random` against
`P-zone-buffer-straddler` is the only protocol pair with matched inference path, patch
sampler geometry and pixel counts, so it is the only pair that can carry a stated effect
size. Replicate it over multiple split seeds and multiple alternative contiguous test
windows, and report an interval rather than a point.

R7. Numeric audit. For each paper, check what is checkable from the reported numbers:
the Dice/IoU identity `Dice = 2*IoU/(1+IoU)`, precision/recall against reported F1,
whether the metric is micro or macro and pixel or instance, whether reported significant
figures are supportable at the stated test-set size, and whether model selection touched
the test set.

R8. Style. The report follows `memory/style.md` and the global writing agreement.

## Acceptance criteria

A1. Every paper in the report has a verified identifier, or is explicitly marked
UNVERIFIED.

A2. The protocol comparison table reports, for every (protocol, architecture) cell, the
test IoU that protocol would have licensed, next to the leakage diagnostics from R4.

A3. RETRACTED. The original criterion claimed the dose-response carried no confound. It
carries a train-composition confound collinear with the treatment. Replacement: the
dose-response is reported with its training positive fraction at every point and with the
causal claim explicitly withdrawn.

A3a. Any stated inflation figure comes from the replicated headline pair and is reported
as an interval over split seeds and test windows, never as a single-seed point estimate.
Two families of noise floor, both now persisted:

- FIXED-RULE floors, results/noise_floors.json, which bound how much the DATA alone moves
  a score: split-draw sd 0.0071 over 40 draws, contiguous-window sd 0.0261 over 238
  interior windows.
- TRAINED-MODEL floors, results/replication.json, which are the ones a trained-model gap
  must actually clear and are far larger: split-draw sd 0.0331 and test-window sd 0.0988
  over four runs each. An earlier version of this criterion quoted only the fixed-rule
  numbers, which understated the real floor by 4.7x and 3.8x.

A3b. A cell is read as a protocol effect only if it beats its own protocol's otsu control.
An undertrained cell is not evidence about leakage.

A4. The report states plainly, for each paper, what this audit does and does not
establish about it. Specifically it must not claim a paper's number is wrong on its own
data when the evidence only supports "this protocol family inflates by X on a dataset
with this structure".

A5. Every EXPERIMENTAL number in the report is traceable to a JSON file under
`Audits of Other Papers/results/`. Two classes of number are deliberately outside that
guarantee and are labelled in place: figures quoted from the audited papers themselves,
which are traceable to their DOIs via papers.md, and figures inherited from this project's
prior work (the 0.0142 minimum detectable effect, the 0.85 to 0.92 annotation ceiling, the
chondrule z-extent statistics), which are produced elsewhere in the repository and are
attributed as such rather than being reproduced here.

A6. The N=1 limitation is stated wherever a generalization claim appears. This audit runs
on one meteorite with 142 chondrules; the measured inflation is an effect size on this
volume, not a universal constant.

## Evaluation contract

Unchanged from the project constitution and reused deliberately so the audit numbers are
comparable to the existing leaderboard:

- Fit sees train and val only.
- The operating threshold is selected on val and frozen for a single test scoring.
- Foreground is the chondrule class; primary metric is foreground IoU over valid pixels.
- Straddler-ignored voxels are excluded from every count via the valid mask.

The one deliberate deviation: under the reconstructed leaky protocols, "val" and "test"
are whatever that protocol defines. That is the point. The leakage is inside the
protocol's own definition of its splits, and reproducing it faithfully is what makes the
measurement meaningful.

## Known confounds, declared up front

C1. Test composition. Different protocols score different pixel populations. The original
mitigation named here, reporting positive-pixel fraction, was WRONG and is withdrawn.
Within the slice family the ranking inverts (zone test sets carry more foreground, 0.0767,
yet score 0.1387 for a fixed rule, against 0.0687 and 0.1463 for slice-random), but the
inversion is not universal: random overlapping patches carry the most foreground, 0.09498,
and also score highest, 0.1544. The honest measure is the contiguous-window sweep in
results/noise_floors.json: across 238 interior 80-slice windows a fixed rule scores mean
0.1682 with sd 0.0261, and this audit's test zone scores 0.1344, the 5.9th percentile.
Mitigated instead by the R6a replication over test windows.

C2. Effective training-set composition. Random splits scatter training slices through the
whole band, which is a genuine information advantage separate from near-duplicate leakage.
Quantified: the reference protocol trains at a positive fraction of 0.05081, against
0.06777 for random slices and 0.06999 for contiguous-no-buffer, because a contiguous split
forces training data out to the near-empty band edges. The two tile protocols ran before
this covariate was recorded, so it is UNMEASURED for them and no claim is made about
"every" arm. This is a real cost of the protocol, is reported as such, and is
NOT subtracted out. Part of any measured gap is therefore data efficiency, not leakage.

C3. N=1 meteorite. Per A6.

C5. Tile-arm handicaps, working AGAINST the leaky tile protocols. Measured: training
patches drawn from 256-px tiles are 23.7% reflect-padded synthetic pixels against 6.7% for
full slices (results/noise_floors.json); tiled inference on a 256-px input yields a single window with no halo and no
overlap-averaging, against 1.4 predictions per pixel plus real halo context for full
slices; and the U-Net's receptive field is exactly 256 px, so every output pixel of a tile
is border-affected. Inflation measured for the tile arms is therefore a LOWER BOUND.

C6. Otsu control scope. The control fits one threshold on train and one operating point on
val. Measured (results/noise_floors.json), the val-selected operating point is identical
across every protocol (0.35) and the fitted cut varies only from 36.357 to 37.354, so its
spread across test sets indexes test composition rather than learning. An earlier version
of this confound claimed both parameters were exactly constant; the cut is not. It is a
sanity check on a per-pixel rule, NOT a subtraction, and it is blind to every CNN-specific
difference in C5.

C7. Single seed. Base grid cells are one run at seed 0.

C8. TEST-SLAB TOUCHES, a declared exception to constitution principle 2. The audit's zone
protocols use `t0, t1 = 160, 239`, which IS the project's frozen test slab in `split.json`.
Across the crossed grid, the buffer sweep and the replication it was scored on the order of
twenty times, and the audit publishes characterisations of it: it scores 0.1344 for a fixed
intensity rule, the 5.9th percentile of 238 interior windows, while the best interior
window scores 0.2205, 1.64 times as much. Principle 2 says the test slab is scored only for the final
locked-model report, and that is not what happened here.

Why it was allowed to stand: the audit's purpose is to measure protocols, not to select a
model, and no model or hyperparameter for the main project was chosen using any of it.
The residual risk is real and is stated rather than hidden: the project now knows its test
region is unusually hard, which creates a standing temptation to relocate the zone. That
temptation must be refused. Relocating the frozen split now would be selection on the test
set, and this file is the record that the knowledge exists. Measured noise: split-draw sd
0.0071 over 40 draws for a fixed rule and 0.0331 over 4 draws for a trained U-Net;
contiguous-window sd 0.0261 for a fixed rule over 238 windows and 0.0988 for a trained
U-Net over 4 windows. Both fixed-rule figures are in results/noise_floors.json and both
trained figures in results/replication.json. The trained figures are the ones a trained-model
gap must clear, and on this data no such gap below roughly 0.15 IoU is separable.

C4. This project's own 0.3168 carries roughly 0.02 to 0.03 of best-of-40 selection bias,
already recorded in the README. Comparisons in this report use freshly trained
fixed-budget runs, not that selected number, wherever a like-for-like cell is needed.
