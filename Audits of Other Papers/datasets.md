# Dataset and research-design differences

Split protocol is only one axis on which this project and the published corpus
differ. Several of the others move reported scores as hard as leakage does, and
ignoring them would replace one unfair comparison with another. This file sets
out what actually differs, so the headline IoU numbers can be read honestly.

## 1. The comparison table

"Specimens" is the count of distinct physical objects the imagery came from. It
is the single most important column for leakage, because it bounds the effective
sample size no matter how many images were cut from them.

| Work | Modality | Specimens | Images | Annotated objects | Task | Split unit |
| --- | --- | --- | --- | --- | --- | --- |
| This project | X-ray micro-CT, 3D volume | 1 meteorite (ALH 83010) | 531 slices, 397 annotated | 142 chondrules | binary chondrule vs matrix, per pixel | contiguous z-zone + buffer + straddler resolution |
| Leichter 2022 | BSE-SEM, 2D | 1 thin section | 9,000 tiles (902/820/492 used) | 20,293 microlites, 975 microphenocrysts, 660 phenocrysts | olivine vs rest, per pixel | analytical session |
| Morales 2025 | optical PPL, 2D | UNSTATED | 414 | 1,472 polygons | 3-class instance seg | random image, 8:1:1 |
| Bihani 2021 | SEM, 2D | UNSTATED | 1,785 crops | 3 classes | silt/clay/pore, per pixel | random crop, 70:15:15 |
| Mahmud 2023 | micro-CT, 3D | UNSTATED | 448 volumes of 512^3 | 4 classes | pore/mineral, per voxel | random, 80:10:10 |
| Su 2020 | optical PPL+XPL, 2D | 92 rock samples | 588 images, 63,504 patches | 13 rock types | whole-image classification | 5-fold, level UNSTATED |
| Fan 2025 | optical, 2D | 10,789 thin sections | 30,336 | about 2 million particles | hierarchical instance seg | random 4:1 internal, plus 4 held-out regions |
| Amer 2025 | optical PPL+XPL, 2D | UNSTATED specimens, 72 rotations each | 14,400 | 5 mineral classes | classification | stratified random 80:20 |
| Thoresen 2024 | optical, 2D | Apollo samples, grouped | UNSTATED | 2 rock types | whole-image classification | stratified k-fold, grouped by sample |
| Nath 2025 | reflectance spectra | UNSTATED (1,781 spectra, specimen count not given) | not imagery | meteorite classes | classification | random 75:25 |
| Jung 2026 | SEM, 2D | UNSTATED | 1,078 base, 17,248 augmented | mineral phases | multi-class seg | UNSTATED relative to augmentation |

Seven of the ten external works give no specimen count in the table above: six
state nothing (Morales, Bihani, Mahmud, Amer, Nath, Jung) and Thoresen groups by
sample without saying how many samples. Only Leichter (1), Su (92) and Fan
(10,789) state one. The eleventh row is this project.
That omission alone makes their reported numbers uninterpretable, because it is
the difference between a large study and a single-specimen study with a large
crop count.

## 2. The differences that matter most

### 2.1 Dimensionality: 3D volumes leak in a way 2D sections cannot

This project's data is a single 3D volume. Its slices are every 4th original CT
slice, so consecutive kept slices are 4 original slices apart and look nearly
identical, and a chondrule spans a median of 24 kept slices (p95 48, max 70).
That means a random slice-level split does not merely put similar images on both
sides, it puts *the same physical object* on both sides, sliced at slightly
different depths.

Most of the corpus is 2D thin sections, where a section is one physical plane
and the leak runs spatially between neighbouring tiles instead of through-plane.
Mahmud 2023 is the exception and the closest structural match: 448 volumes of
512^3, split 80:10:10 at random. If those volumes are sub-volumes of one scan,
that is structurally the same failure mode. NOT MEASURED HERE: the audit's
3D-patch protocol was specified and never implemented (see spec R3), so the
closest reconstruction actually run is the 2D `P-patch-random-overlap`.

The nearest published analogue outside geoscience is Yagis 2021 on brain MRI,
where splitting a 3D scan at the 2D slice level rather than the subject level
moved accuracy from the 46 to 70% range up to the 93 to 97% range, and reached
95% on data with randomly assigned labels.

### 2.2 Task separability: chondrules have almost no intensity contrast

This is the difference that makes raw IoU comparison across papers meaningless,
and it is independent of leakage.

Chondrules are polymineralic and share their constituent phases with the
surrounding matrix. In X-ray attenuation they are close to the matrix, which is
why the field still segments them by hand. The fixed-intensity Otsu control in
this audit scores 0.1357 to 0.1544 test IoU across all six protocols:
thresholding on intensity barely works at all.

Compare that with the tasks reporting high scores:

- Pore versus solid in digital rock physics is a near-binary density contrast.
  Thresholding is already a strong baseline, so 0.91 IoU is a modest improvement
  over a trivial method, not evidence of a hard problem solved.
- Mineral classification under cross-polarized light uses birefringence colour,
  which separates species by hue. Again a strong non-learned baseline exists.
- Olivine phenocrysts in BSE are compositionally distinct from groundmass and
  are large. Leichter's 98.5% overall accuracy on a task with modest foreground
  fraction is consistent with an easy contrast.

A reported IoU is only interpretable against that task's non-learned floor. A
paper that reports 0.91 IoU without reporting what a threshold achieves has not
established that its model did the work. Of the segmentation papers in this
corpus only Griffin et al. 2012 reports one, and it is the oldest entry and the
only non-learned method in the set.

### 2.3 Annotation ceiling

This project measured its own ceiling: agreement with the human annotation at
plus or minus 1 pixel of boundary tolerance is about 0.92 IoU, and at plus or
minus 2 pixels about 0.85. An IoU near 1.0 is not attainable because the label
boundary is not that precise.

No paper in the corpus reports an annotation ceiling or any inter-annotator
agreement. Without it, a reported IoU cannot be placed on a scale: 0.80 against
a ceiling of 0.85 is a very different result from 0.80 against a ceiling of
0.99, and the reader cannot tell which they are being shown.

### 2.4 Class balance

Chondrules occupy 3.6% of training pixels, 6.7% of validation and 5.3% of test
in the project's frozen full-field split (`split.json`). On the audit's cropped
band the reference protocol measures 0.0508 train and 0.0748 test
(`results/protocol_comparison.json`); the two are different fields of view and
are not interchangeable.
Under that imbalance, pixel accuracy is close to useless: predicting all
background scores above 94%. Several corpus papers lead with overall accuracy
(Leichter 98.5%, Mahmud 96.78%, Jung 0.82, Su 89.97%) and this project's
constitution requires IoU precisely because accuracy hides the failure.

Where a paper reports accuracy alongside IoU, the two can be compared. Where a
paper reports only accuracy, the number is not evidence of segmentation quality
on an imbalanced class.

### 2.5 Uncertainty and model selection

Only Thoresen 2024 reports an uncertainty on its headline figure (98.44%
plus or minus 1.47). Every other paper in the corpus reports a bare point
estimate, several to four significant figures.

This project is not exempt. Its published 0.3168 is a four-significant-figure
number selected as the best of about 40 configurations, carrying an estimated
0.02 to 0.03 of selection bias that is recorded in the repository README, on a
validation set of 19 chondrules where the project's own power calculation gives
a minimum detectable effect of 0.0142 IoU (2 x 0.0071 x sqrt(19/n) at n = 19).
That figure comes from the project, not from this audit, and is in none of the
JSONs here. By the standard applied to the corpus above, this project should be
reporting roughly 0.32 with an interval, not 0.3168. The audit applies the same
rule to its own numbers.

### 2.6 Data and code availability

| Work | Data public | Code public |
| --- | --- | --- |
| This project | no, user-provided scan | yes, this repository |
| Thoresen 2024 | NASA PDS source | yes, ESA GitHub |
| Bihani 2021 | partial | yes, GitHub |
| Morales 2025 | yes, Zenodo | not stated |
| Amer 2025 | yes, FigShare | yes, GitHub |
| Su 2020 | not stated | not stated |
| Fan 2025 | not stated | not stated |
| Mahmud 2023 | open-source sandstone set | not stated |
| Leichter 2022 | not stated | visualization tool only |

This project cannot publish its data and that is a genuine limitation. It is
also the reason the audit reconstructs protocols on this volume rather than
rerunning the papers on theirs: the reverse experiment is not available either,
because most of these datasets are not distributable in a form that would allow
a re-split.

## 3. What none of this excuses

The differences above explain why absolute IoU should not be compared across
these papers at all. Across datasets, no IoU comparison in either direction is
defensible, including any that flatters this project.

UPDATED after the replication. An earlier version of this paragraph said the
protocol effect was "measured cleanly" within one dataset. It was not. The
crossed grid holds dataset, task, architecture and budget fixed, but it holds
the split DRAW and the test WINDOW fixed too, and those turned out to dominate.
The replicated estimate is +0.0290 IoU with a 95% interval of [-0.1210, +0.1789],
which spans zero. What is measured cleanly is the leakage itself (100% of test
chondrules against 0 of 43), not its effect on the score.
