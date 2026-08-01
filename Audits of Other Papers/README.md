# Audits of Other Papers

An audit of the published literature this project would be compared against, and
a measurement of what those papers' evaluation protocols would report if run on
this project's data.

Written 2026-07-30, results and retraction 2026-07-31. Governed by [spec.md](spec.md).

| File | Contents |
| --- | --- |
| [spec.md](spec.md) | the specification, acceptance criteria and declared confounds |
| [papers.md](papers.md) | per-paper dossiers, verification status and numeric audit |
| [datasets.md](datasets.md) | dataset and research-design differences |
| `results/*.json` | every experimental number in this report |
| `figures/*.png` | plots, all generated from those JSON files |
| `protocols.py`, `01`-`07` scripts | the reconstruction harness |

---

## Summary

1. **There is no published deep-learning chondrule segmentation work.** A search
   across the meteoritics and planetary-science literature, NASA ADS, arXiv and
   the LPSC and Meteoritical Society abstract archives found none. The nearest
   prior art is semi-automated intensity-histogram segmentation (Griffin et al.
   2012) and a manual measurement protocol. This project has no direct
   competitor, so the comparison that would actually be made is against
   thin-section and digital-rock segmentation, where 0.80 to 0.91 IoU is common.

2. **That adjacent literature mostly splits randomly, or does not say.** Of the
   ten external works with a stated task and split, five split randomly at the
   image, patch or crop level, three never state the level at which folds were
   formed, and two group by physical sample or session. Seven of the ten give no
   usable specimen count, which is the number that decides what a random split
   means. One case is mechanically certain rather than
   suspected: a published benchmark dataset images each crystal at 72 rotations
   and splits them with a function that has no grouping argument.

3. **On this volume those protocols leak completely.** Measured before any model
   is trained: under every random protocol, 100% of test chondrules and 100% of
   test positive pixels belong to objects also present in training. The
   leakage-free reference reaches 0 of 43. A buffer alone is not enough; it
   leaves 19.9% leaking, and only straddler resolution closes it.

4. **Whether that leakage materially inflates reported IoU could not be
   determined from one meteorite.** A single-split comparison suggested about
   40% inflation and agreed across three architectures. Replication over four
   split draws and four test windows gave +0.0290 IoU, 95% CI [-0.1210,
   +0.1789]. The interval spans zero. Direction positive, magnitude
   unestablished. See Finding 5, which explains why the single-draw result
   looked so convincing.

5. **That failure is the audit's strongest point about precision.** If 142
   chondrules cannot resolve an effect of this size, a paper quoting IoU to four
   significant figures from a 10% random split of 448 volumes (Mahmud 2023,
   91.12%) is reporting digits its data cannot support. Of the segmentation papers surveyed, none reports an
   interval or an annotation ceiling, and only Griffin et al. 2012 reports a
   non-learned baseline (85.8% against 73.4% for pixel intensity).

6. **Absolute IoU is not comparable across these papers anyway.** Chondrules
   share their mineral phases with the matrix, so a fixed intensity rule reaches
   only 0.1357 to 0.1544 here, while pore-versus-solid and birefringence tasks
   have strong non-learned baselines. See [datasets.md](datasets.md).

---

## Finding 0: there is no chondrule segmentation literature to benchmark against

The premise of the audit was that this project's test IoU of about 0.316 looks
weak against published work. The first result is that the published work being
imagined does not exist.

(The repository records this figure two ways. The README reports 0.3168; the
CHANGELOG records that under the committed threshold-grid fix the same model
val-selects 0.0076 and scores 0.31601. This audit uses "about 0.316" and does
not rely on the difference.)

A search across roughly 200 query combinations covering Meteoritics and
Planetary Science, Geochimica et Cosmochimica Acta, Earth and Planetary Science
Letters, Icarus, Computers and Geosciences, NASA ADS, arXiv, EarthArXiv, and the
LPSC and Meteoritical Society abstract archives found no peer-reviewed paper
applying deep learning to chondrule segmentation or detection. The nearest work
is:

- Griffin, Elangovan, Mundell and Hezel (2012), semi-automated segmentation of
  meteorite micro-CT by local intensity histograms, reporting 85.8%
  percent-correct against 73.4% for plain pixel intensity. It segments
  chondrules, CAIs and dark inclusions, and it is not machine learning. This is
  the correct baseline for this project to beat.
- The CIS method (Meteoritical Society 2022), a standardised **manual** protocol
  using GIMP and ImageJ.
- Floyd et al. 2024, which measured 1,937 CM chondrules by hand.

One possible counterexample exists and is flagged rather than suppressed: LPSC
2024 abstract 2549 (Mills et al.) claims automated machine-learning segmentation
of chemical maps of chondritic meteorites. The abstract PDF could not be
retrieved, so it is UNVERIFIED. It is a conference abstract, not a peer-reviewed
paper.

The consequence is that this project cannot be benchmarked against a chondrule
baseline, because none is published. The comparison that would actually be made,
by a reviewer or by the project itself, is against the adjacent literature on
thin-section and digital-rock segmentation, where reported IoU of 0.80 to 0.91
is common. That comparison is what the rest of this audit tests.

---

## Finding 1: most of that literature splits randomly, and three do not say at what level

Ten external works have a stated task and split. Five of them split randomly at
the image, patch or crop level; three never state the level at which folds were
formed, which is itself the finding; two group by physical sample or analytical
session. Seven of the ten give no usable specimen count: six state nothing
(Morales, Bihani, Mahmud, Amer, Nath, Jung) and Thoresen groups by sample without
saying how many. That count is what bounds the meaning of any random split.

Verified split descriptions, quoted from the sources:

| Work | Split, as stated | Grouped by specimen |
| --- | --- | --- |
| Thoresen 2024 | stratified k-fold, "For each split, images of the same sample end up in the same set" | yes |
| Fan 2025 | random 4:1 internal, plus four held-out geographic cohorts | partly, by region |
| Leichter 2022 | by analytical session within one thin section | partly, by session |
| Morales 2025 | "randomly divided into training, validation, and test sets in an 8:1:1 ratio" | not stated |
| Bihani 2021 | "split randomly into training, validation and test datasets in a ratio 70:15:15" | not stated |
| Mahmud 2023 | "80% ... validation 10% ... testing 10%" | not stated |
| Su 2020 | five-fold cross-validation, level not stated | not stated |
| Amer 2025 | "stratified 80:20 train-test split implemented through scikit-learn's train_test_split" | no, see below |
| Nath 2025 | 75:25 | not stated |
| Jung 2026 | augmentation to 17,248 images from 1,078 base images; split relative to augmentation not stated | not stated |

The Amer 2025 case is mechanically certain rather than merely suspected. The
dataset images each mineral specimen at 72 rotational positions, and the split
is a stratified random one from scikit-learn's `train_test_split`. That function
has no grouping argument. About 80% of each crystal's 72 views therefore land in
training and the rest in test, so almost every test image has roughly 50
near-duplicate views of the same physical crystal in the training set. This is a
published benchmark dataset, so any future paper adopting its split inherits the
leak.

---

## Finding 2: on this volume, those protocols leak almost completely

Before any model is trained, each protocol can be measured for what it puts on
both sides of the split. Computed on the 397-slice annotated band of ALH 83010,
142 chondrules, with a chondrule defined as a 3D connected component constrained
to one annotator id (the same definition the frozen split uses).

Full results in [`results/leakage_diagnostics.json`](results/leakage_diagnostics.json).

| Protocol | Chondrules in test also in train | Test positive pixels from such chondrules | Nearest train unit |
| --- | --- | --- | --- |
| random overlapping patches | 139 of 139 (100%) | 100% | 89.7% of test patches physically overlap a train patch, mean 48% of area |
| random tiles | 140 of 140 (100%) | 100% | 97.8% of test tiles touch a train tile |
| random slices | 141 of 141 (100%) | 100% | 80% of test slices are adjacent to a train slice; all are within 4 |
| contiguous zones, no buffer | 27 of 51 (53%) | 51.6% | 2.5% adjacent |
| zones + buffer 12 | 8 of 51 (16%) | 19.9% | 13 band slices (52 original) |
| zones + buffer + straddler resolution | **0 of 43 (0%)** | **0.0%** | 13 band slices (52 original) |

Under all three random protocols, **every single chondrule that appears in the
test set also appears in the training set**, and 100% of the test positive
pixels belong to a chondrule the model was trained on. There is no held-out
object at all. A score computed on such a split measures how well the model
re-identifies objects it has already seen at slightly different depths.

A result worth recording in its own right: a buffer alone is not sufficient. At
buffer 12, eight chondrules still span train and test, carrying 19.9% of test
positive pixels, because chondrules reach 70 band-slices in z-extent while the
buffer is 12. Adding straddler resolution takes that to zero. This is the first
independent confirmation that `src/split.py`'s extra mechanism was doing real
work rather than being defensive over-engineering: without it, a fifth of the
test signal comes from objects the model trained on.

The instance count also drops from 51 to 43 in that last row, which is the
mechanism made visible. Straddler resolution does not merely relabel the eight
shared chondrules, it removes their minority-zone voxels from scoring entirely,
so the reference protocol evaluates on 43 chondrules that are wholly unseen
rather than 51 of which eight are partly memorized.

---

## Finding 3: what each protocol lets you report

Every cell below is the same data, the same 3,000-step budget and the same
evaluation contract. Only the split protocol changes. Full results in
[`results/protocol_comparison.json`](results/protocol_comparison.json).

| Protocol | U-Net | UNet++ | DeepLabV3+ | otsu control |
| --- | --- | --- | --- | --- |
| random overlapping patches | 0.2922 | 0.2830 | 0.2830 | 0.1544 |
| random tiles | 0.2369 | 0.2229 | 0.2389 | 0.1466 |
| random slices | 0.2996 | 0.2811 | 0.2734 | 0.1463 |
| contiguous zones, no buffer | 0.1952 | 0.2007 | 0.1834 | 0.1387 |
| zones + buffer 12 | 0.2417 | 0.2022 | 0.1855 | 0.1387 |
| zones + buffer + straddler (reference) | 0.2132 | 0.2057 | 0.1945 | 0.1357 |

The headline pair is random slices against the leakage-free reference. These two
are the only protocols in the table with a matched inference path, matched patch
sampler geometry and identical pixel counts, so the comparison is not carrying
the tile handicaps described above.

| Architecture | random slices | reference | gap | relative |
| --- | --- | --- | --- | --- |
| U-Net | 0.2996 | 0.2132 | +0.0864 | +40.5% |
| UNet++ | 0.2811 | 0.2057 | +0.0754 | +36.7% |
| DeepLabV3+ | 0.2734 | 0.1945 | +0.0789 | +40.6% |
| mean | | | +0.0803 (sd 0.0056) | about 40% |

Three architectures, trained and evaluated independently, agree on the size of
this gap to within 0.011 IoU (sd 0.0056), and the otsu control spans only 0.0187
across these six test sets. That looked at the time like test difficulty being
ruled out. It was not: Finding 5 shows the choice of contiguous test window moves
a fixed rule by sd 0.0261 across the band, and the reference cell's window sits
at the 5.9th percentile of interior windows. Test difficulty is in fact the main
thing this table failed to control.

**DO NOT READ THAT TABLE AS AN EFFECT SIZE. IT DOES NOT SURVIVE REPLICATION.**

Every cell above uses one split draw and one test window. Finding 5 repeats this
exact pair over four split draws and four contiguous test windows and finds the
difference is +0.0290 IoU with a 95% interval of [-0.1210, +0.1789], which spans
zero. The agreement across architectures above is real but misleading: all three
architectures share the same split and the same test window, so they are three
measurements of one draw, not three independent estimates of the effect.

The table is kept because it is what the crossed design produced and because
removing it would hide how convincing a single-draw result can look. The number
to quote from this audit is in Finding 5, not here.

### Full metric set, U-Net

The constitution requires IoU never to be reported alone, because under about
95% background it hides which way a model fails. Generated by `06_tables.py`.

| Protocol | IoU | Dice | precision | recall | AUPRC |
| --- | --- | --- | --- | --- | --- |
| random overlapping patches | 0.2922 | 0.4522 | 0.4253 | 0.4828 | 0.3665 |
| random tiles | 0.2369 | 0.3831 | 0.3034 | 0.5196 | 0.2437 |
| random slices | 0.2996 | 0.4611 | 0.3653 | 0.6251 | 0.3813 |
| contiguous zones, no buffer | 0.1952 | 0.3266 | 0.2323 | 0.5498 | 0.2468 |
| zones + buffer 12 | 0.2417 | 0.3893 | 0.3003 | 0.5531 | 0.3237 |
| zones + buffer + straddler (reference) | 0.2132 | 0.3515 | 0.2474 | 0.6066 | 0.2538 |

Recall is broadly similar across protocols (0.48 to 0.63) and the separation sits
mostly in precision (0.232 to 0.425). That pattern is consistent with leakage,
which should let a model that has already seen the test objects place boundaries
more confidently rather than find more of them, but this is a single draw per
cell and the same warning as above applies: it is suggestive, not an effect size.

AUPRC does NOT reproduce the IoU ordering, and saying so matters. By IoU the
order is slices, patches, buffer, tiles, straddler, no-buffer; by AUPRC it is
slices, patches, buffer, straddler, no-buffer, tiles. Random tiles falls from
fourth to last. That is what confound C5 predicts, since the tile arms are
handicapped by padding and haloless inference rather than by their leakage, but
it means AUPRC cannot be used to argue the gap is threshold-independent. It is
reported here because it is the threshold-free metric the constitution asks for,
not because it supports the case.

### What this table does NOT show

- **The three zone variants are not separable from each other.** Adding the
  buffer moved the three architectures by +0.0465, +0.0015 and +0.0021. That is
  scatter around zero driven by one U-Net run, not a monotonic decline. Anyone
  reading a clean "less leakage, lower score" gradient into the bottom three
  rows is reading noise. The model-free diagnostics in Finding 2 are what
  establish that those protocols differ in leakage; the trained scores at a
  single seed do not resolve it.
- **The tile arms understate their own protocol.** Random tiles scores below
  random slices despite leaking just as completely, because of the padding and
  haloless-inference handicaps. Its row is a floor, not an estimate.
- **These are single-seed points.** The interval is in Finding 5.

## Finding 4: validation-to-test exchangeability, a second inflation channel

This one was not anticipated by the spec; it fell out of the results. Alongside
each test score the harness records the validation score the same model reached.

Means over the three CNN architectures, generated by `06_tables.py`.

| Protocol | mean val IoU | mean test IoU | val minus test |
| --- | --- | --- | --- |
| random overlapping patches | 0.2719 | 0.2861 | -0.0141 |
| random tiles | 0.2297 | 0.2329 | -0.0033 |
| random slices | 0.2824 | 0.2847 | -0.0023 |
| contiguous zones, no buffer | 0.3626 | 0.1931 | +0.1696 |
| zones + buffer 12 | 0.3540 | 0.2098 | +0.1442 |
| zones + buffer + straddler (reference) | 0.3695 | 0.2044 | +0.1651 |

Under a random split, validation and test are draws from the same pool, so they
agree to within 0.0141 and quoting either is equivalent. Under all three
contiguous splits they are different regions of one meteorite and they differ by
0.144 to 0.170.

Per architecture on the no-buffer protocol the gap is +0.1809 (U-Net), +0.1792
(UNet++) and +0.1486 (DeepLabV3+), so the direction is consistent but the
magnitude is not identical across architectures.

So a paper that splits contiguously but reports its validation number overstates
its true held-out result by roughly 0.15 IoU here, with no training leakage
involved at all. This matters for a specific paper in the corpus: Nath et al.
2025 reports 93% as a **validation** accuracy, with no separate held-out test set
described in the material that could be accessed. Under a random split that
distinction is harmless. Under a structured split it is the difference between
about 0.36 and about 0.20 on this data.

Note this channel is orthogonal to Finding 5. It does not depend on the random
versus contiguous inflation surviving replication, because it is a gap measured
within a single protocol between two sets that protocol itself defines.

## Finding 5: the effect does not survive replication, and that is the result

The headline pair was rerun over four random split draws and four contiguous
test windows. U-Net, 3,000 steps, everything else identical. Full results in
[`results/replication.json`](results/replication.json).

| Arm | Variant | test IoU | otsu | gain over control |
| --- | --- | --- | --- | --- |
| random slices | seed 0 | 0.2996 | 0.1463 | +0.1533 |
| random slices | seed 1 | 0.2690 | 0.1372 | +0.1319 |
| random slices | seed 2 | 0.2765 | 0.1436 | +0.1330 |
| random slices | seed 3 | 0.2210 | 0.1316 | +0.0894 |
| leakage-free | window -60 | 0.1306 | 0.0507 | +0.0798 |
| leakage-free | window 0 | 0.2132 | 0.1357 | +0.0775 |
| leakage-free | window +60 | 0.3688 | 0.1871 | +0.1817 |
| leakage-free | window +120 | 0.2377 | 0.1383 | +0.0995 |

Random-split arm: mean 0.2665, sd 0.0331. Leakage-free arm: mean 0.2376, sd
0.0988.

**Difference +0.0290 IoU, 95% CI [-0.1210, +0.1789]. The interval spans zero.**

In gain-over-control terms, which normalizes for how hard each test region is:
**+0.0173 IoU, 95% CI [-0.0563, +0.0908]. Also spans zero.**

### What went wrong with the single-draw estimate

The leakage-free arm has a standard deviation of 0.0988, three times the random
arm's. Its raw scores range from 0.1306 to 0.3688 depending only on which
contiguous 80-slice window is used as test.

Most of that is intrinsic difficulty of different regions of rock, but not all of
it, and the difference matters. Comparing each replication window against the
independent fixed-rule sweep in
[`results/noise_floors.json`](results/noise_floors.json):

| window offset | test start | fixed-rule sweep IoU | replication control | model IoU |
| --- | --- | --- | --- | --- |
| -60 | 100 | 0.1988 | 0.0507 | 0.1306 |
| 0 | 160 | 0.1344 | 0.1357 | 0.2132 |
| +60 | 220 | 0.1823 | 0.1871 | 0.3688 |
| +120 | 280 | 0.1345 | 0.1383 | 0.2377 |

Three of the four agree to within 0.005, so for those the control really is
reading region difficulty. The -60 window does not: the independent sweep ranks
it at the 82.8th percentile of interior windows, comfortably easier than average,
while the replication's own control calls it the hardest of the four at 0.0507.
The likely cause is that at this offset the validation zone slides to band slices
20 to 79, against the sparse end of the annotated band, so the operating point
selected there transfers badly to test. That is threshold-selection instability,
not test-region difficulty, and it is a third variance source alongside the two
already declared.

It does not rescue the retracted claim. An additional source of variance can only
widen the interval, and the interval already spans zero.

Finding 3's reference cell used window 0, which the earlier contiguous-window
sweep placed at the 6th percentile of interior windows. It was an unusually hard
test region. Meanwhile window +60 produced 0.3688, higher than any random-split
run in the entire audit. One random draw, seed 3, produced 0.2210, essentially
level with the reference. The single-draw gap of 0.0864 was largely the distance
between an unlucky reference window and a lucky random draw.

### What this does and does not overturn

It does NOT overturn Finding 2. The leakage is measured, not estimated: under
every random protocol, 100% of test chondrules and 100% of test positive pixels
come from objects present in training, against 0 of 43 for the reference. Those
are counts. The leakage is definitely there.

What could not be established is its **effect on reported IoU**, on this dataset,
at this budget. Direction is positive in both metrics and in every architecture,
but the magnitude is not separable from window and draw variance. The
"every architecture" part of that direction claim rests on the crossed grid,
which is three measurements of one draw, so it is weaker evidence than it looks;
the direction is also positive in both replication metrics, which is the
independent part.

### Why this was foreseeable, and what it means for the corpus

This project already knows its binding constraint: with roughly 19 to 43
chondrules in a held-out zone, the project's minimum detectable effect is 0.0142
IoU at n = 19 and 0.0094 at n = 43 for a paired comparison and far higher for unpaired ones across different
regions of rock. An audit run on one meteorite runs into the same wall the
project runs into, and it did.

That failure is itself the audit's strongest argument about the literature. If
142 chondrules in a single volume cannot resolve a protocol effect of this size,
then Mahmud 2023 reporting IoU as 91.12% from a 10% random split of 448 volumes,
or Su 2020 reporting 89.97% accuracy from 92 rock samples, is reporting digits
its data cannot support. The papers in Group 2 of
[papers.md](papers.md) do not report intervals at all. This audit tried to and
found the interval wider than the effect.

The honest summary of the experimental half of this audit: **the protocols leak,
completely and measurably; whether that materially changes a reported IoU could
not be determined from one meteorite.**

## Finding 6: buffer width does not measurably matter, and that is informative

The buffer sweep holds the test zone fixed at band slices 160-239 and the
training set fixed at 197 slices, varying only how close the nearest training
slice may sit to the test zone. Full results in
[`results/dose_response.json`](results/dose_response.json).

| Buffer (band slices) | Physical gap (original slices) | test IoU | train positive fraction |
| --- | --- | --- | --- |
| 0 | 4 | 0.2071 | 0.07475 |
| 2 | 12 | 0.2136 | 0.07137 |
| 4 | 20 | 0.2043 | 0.06805 |
| 8 | 36 | 0.2139 | 0.06151 |
| 12 | 52 | 0.2283 | 0.05700 |
| 16 | 68 | 0.2178 | 0.05430 |
| 20 | 84 | 0.2049 | 0.05171 |

Mean 0.2129, sd 0.0085, range 0.2043 to 0.2283. The correlation between buffer
width and IoU is **+0.215**, slightly positive, with a slope of +0.0024 IoU per
10 band-slices. There is no detectable decline.

Read this carefully, because it cuts against the hypothesis the sweep was built
to test, and it is reported as a null rather than buried.

- **Within contiguous splitting, buffer width does not measurably change the
  score** at separations from 4 to 84 original slices on this data. The sweep's
  own sd (0.0085) is well below the trained-model split-draw sd of 0.0331 measured
  in the replication, so it is scatter.
- **The confound moved and the score still did not.** Training positive fraction
  fell 30.8% across the sweep, from 0.07475 to 0.05171, which should have pushed
  the score down if training foreground were driving results at this budget. It
  did not. The retracted causal claim is therefore not merely unproven, the
  confound it worried about also fails to show up.
- **This is consistent with the diagnostics rather than in tension with them.**
  The random protocols leak 100% of test positive pixels; contiguous-no-buffer
  already drops that to 51.6%. The large score change happens at that step. The
  buffer refines 51.6% down to 19.9%, and straddler resolution to 0%, which are
  second-order adjustments that this budget and single seed cannot resolve.

The practical implication for anyone designing a split on data like this: the
decision that matters is random versus structured. Getting the buffer width
exactly right is a smaller effect than choosing to split by structure at all.
The straddler mechanism is still worth keeping, because Finding 2 shows it is
the only thing that takes measured leakage to zero, and a protocol should be
correct by construction rather than because its residual leak happened to be too
small to detect at one budget.

---

## Method

Each protocol is reconstructed against the same ALH 83010 band and run through
the project's existing model runners, so everything except the split is held
fixed.

Controls built into the design:

- **Identical training set size.** Every protocol allocates 213 train, 60
  validation and 80 test slices, or the tile equivalent. The 44 slices the
  reference protocol spends on its buffer are left unused by every other
  protocol too, so no protocol gets more training slices. The 44 are not the
  SAME slices in every protocol: `P-zone-nobuffer` keeps the 213 nearest the
  test zone and therefore drops the near-empty band edges, which is why it
  trains at 0.06999 foreground against the reference's 0.05081. That is
  confound C2, and it is not controlled away. Because a 768 x 1280 field holds exactly 15 tiles of 256 px, a
  213-slice training set and a 3,195-tile training set contain exactly the same
  209,387,520 pixels. Slice and grid-tile protocols are matched to the pixel.
  `P-patch-random-overlap` is the exception: its patches overlap, so it draws
  3,219 patches covering fewer distinct pixels than the others, and its test
  population is not a clean pixel set. Its row is indicative only.
- **Identical field of view.** All runs use the same centred 768 x 1280 crop, so
  a 256-px tile grid partitions the field exactly and tile protocols score the
  same pixel population as slice protocols.
- **Identical budget.** Every CNN cell trains from scratch for 3,000 steps.
- **Identical evaluation contract.** Fit on train, select the operating
  threshold on that protocol's own validation set, score its test set once.
  Under the leaky protocols the validation set is leaked too, which is correct:
  the point is to reproduce the number the protocol would have licensed a paper
  to publish, not to grade the model.
- **A control arm whose fitted parameters barely move.** The `otsu`
  arm fits one scalar cut on training intensities and one operating threshold on
  validation. On this volume the operating point is identical across every
  protocol (0.35) and the cut varies only from 36.357 to 37.354, so its spread
  across the six test sets indexes test composition rather than learning. It is a measured sanity check on a per-pixel rule, not a
  subtraction, and it is blind to every CNN-specific difference listed below.

### Limits found by adversarial verification, and kept

The measurement code was checked by two independent verifiers before any number
here was written. Two defects were found in it and fixed: the diagnostics were
not applying the straddler mask, so the project's own leakage-free reference was
being scored as if it still leaked; and a guard silently returned a smaller
training set instead of raising. Four limits could not be fixed and are declared
instead.

- **Tile-arm inflation is a LOWER BOUND.** The tile protocols are handicapped
  against themselves in three measured ways. Training patches drawn from 256-px
  tiles are 23.7% reflect-padded synthetic pixels against 6.7% for full slices
  (`results/noise_floors.json`).
  Tiled inference on a 256-px input produces a single window with no halo and no
  overlap-averaging, against 1.4 predictions per pixel plus real halo context on
  full slices. And the U-Net's receptive field is exactly 256 px, the tile size,
  so every output pixel of a tile is border-affected. Whatever inflation the
  tile arms show, the true figure is larger.
- **The reference protocol trains on less foreground.** It sees a training
  positive fraction of 0.05081, against 0.06777 for random slices and 0.06999
  for contiguous-no-buffer (the two tile arms predate the covariate being
  recorded, so this is unmeasured for them),
  because a contiguous split forces training data out to the near-empty band
  edges. Part of any measured gap is therefore data efficiency, not leakage, and
  it is not subtracted out.
- **Positive-pixel fraction is the wrong difficulty proxy.** Within the
  slice-family protocols the ranking is inverted: the zone test sets carry more
  foreground (0.0767) yet score lower for a fixed intensity rule (0.1387) than
  random-slice does (0.0687, 0.1463). The inversion is not universal, and the
  earlier version of this bullet overstated it: random overlapping patches carry
  the most foreground of all (0.09498) and also score highest (0.1544). The
  honest measure is the contiguous-window sweep in
  [`results/noise_floors.json`](results/noise_floors.json): across 238 interior
  80-slice windows a fixed rule scores mean 0.1682 with sd 0.0261, and this
  audit's test zone scores 0.1344, the 5.9th percentile, so the reference is
  scored on an unusually hard window.
- **The buffer sweep is not confound-free, and that claim is withdrawn.** It was
  specified as the clean anchor. Verification measured that nearest-first slice
  selection makes the training positive fraction fall from 0.07475 to 0.05171
  monotonically with the buffer, collinear with the treatment. The sweep is kept
  with that covariate reported at every point and the causal reading retracted.
  It was not redesigned, because the audit's remit is to run published protocols
  rather than invent new ones, and no paper runs a buffer sweep.

Because of the last two points, the quantitative claim rests on a replication of
the single protocol pair that is genuinely matched: random-slice against the
leakage-free reference, which share an inference path, a patch-sampler geometry
and a pixel count. That pair is run over four split seeds and four contiguous
test windows, and reported as an interval. Measured noise floors any claim must
clear. Two families, both persisted in `results/`: fixed-rule floors, which bound
what the data alone does (split-draw sd 0.0071 over 40 draws, contiguous-window
sd 0.0261 over 238 windows, in `noise_floors.json`); and trained-model floors,
which are what a trained gap must actually clear and are much larger (split-draw
sd 0.0331, test-window sd 0.0988, in `replication.json`).

---

## What this audit establishes, and what it does not

It establishes:

- On a dataset with this structure, the split protocols used by most of the
  surveyed literature put every test object in the training set. This is a
  count, not an estimate: 100% of test chondrules and 100% of test positive
  pixels, against 0 of 43 for the leakage-free reference.
- That the resulting effect on reported IoU could NOT be resolved on one
  meteorite. The measured difference is +0.0290 IoU with a 95% interval of
  [-0.1210, +0.1789]. Direction positive, magnitude unestablished.
- That a single-split-draw comparison looked convincing and was wrong. Three
  architectures agreed to within 0.011 IoU on a gap of 0.080 that replication
  then failed to confirm, because all three shared one draw and one test window.
- The specific claim about Amer et al. 2025 is mechanical: the named function
  cannot group rotational views, so views of one crystal are split across train
  and test.
- Several citations circulating for this literature carry wrong metadata, and
  are corrected in [papers.md](papers.md).

It does not establish:

- That any specific paper's number is wrong on its own data. This audit did not
  obtain their data, and mostly could not: these datasets are largely not
  public. What it measures is what a protocol family does on a dataset with a
  known structure.
- That this project's method is better than theirs. It is not comparable at all,
  for the reasons in [datasets.md](datasets.md): different modality, different
  task separability, different annotation ceiling. Comparing absolute IoU across
  these papers is not defensible in either direction.

## Limitations

- **N = 1 meteorite.** Every number here is an effect size on one ALH 83010
  volume with 142 chondrules, not a universal constant.
- **This audit spent test-slab touches, against constitution principle 2.** The
  zone protocols test on band slices 160 to 239, which is the project's frozen
  test slab, and it was scored roughly twenty times here. The audit also now
  publishes that this window is unusually hard (0.1344 for a fixed rule, the
  5.9th percentile of 238 interior windows). No project model or hyperparameter
  was selected using any of it, but the knowledge now exists and the frozen split
  must not be relocated in response to it. Declared as spec confound C8.
- **Single seed.** Each cell is one run at seed 0.
- **Reduced budget.** 3,000 steps per cell, against the 30,000 steps behind the
  project's headline figure, so absolute IoU values here sit below the project's
  best. The protocol contrast, not the absolute level, is the measurement.
- **One citation in the source literature search was fabricated.** The closest
  prior work was returned as "Grimm and Woolum (2011)". The Crossref record for
  its DOI gives Griffin, Elangovan, Mundell and Hezel (2012). Every citation in
  this report was checked against a publisher or Crossref record for exactly
  this reason, and the corrections are listed in [papers.md](papers.md) rather
  than applied silently.
- **One number could not be verified and is not used.** The "up to 40% more
  optimistic" figure widely attributed to Roberts et al. 2017 could not be found
  in the original text across two attempts. The paper is cited for its principle
  only.
- **The two versions of MudrockNet disagree.** The published paper and its arXiv
  preprint report different per-class IoU values. The published values are used
  and the discrepancy is recorded.
- **Paywalls.** Nath 2025, Jung 2026 and Hezel 2025 could not be read in full,
  so the decisive split questions for those three remain UNVERIFIED.
- **This project audits itself on the same terms.** Its published figure is a
  four-significant-figure number, best of about 40 configurations, carrying an
  estimated 0.02 to 0.03 selection bias, measured against a validation set of 19
  chondrules whose minimum detectable effect the project computes as 0.0142 IoU
  (from the project's own power calculation, 2 x 0.0071 x sqrt(19/n) at n = 19;
  it is not produced by this audit and is in none of the JSONs here). It
  should be reported as approximately 0.32 with an interval, and the
  repository's own two values for it (0.3168 and 0.31601) differ by far less
  than that floor.

## Reproducing

```bash
python "Audits of Other Papers/01_leakage_diagnostics.py"
python "Audits of Other Papers/02_run_protocols.py" --budget 3000
python "Audits of Other Papers/03_dose_response.py" --budget 3000
python "Audits of Other Papers/05_replication.py" --budget 3000
python "Audits of Other Papers/07_noise_floors.py"
python "Audits of Other Papers/04_figures.py"
python "Audits of Other Papers/06_tables.py" > "Audits of Other Papers/results/tables.md"
```

Script 05 produces Finding 5, the headline. Script 07 produces the noise floors
that Finding 5 is judged against. Script 06 regenerates every experimental table
in this report.

The scripts require the `ALH83010_z_crop_every4` and
`ALH83010_z_crop_every4_labeled` directories at the repository root. Scripts 02
and 03 cache completed cells to their JSON output and skip them on re-run, so an
interrupted sweep resumes.
