# Paper dossiers

Per-paper record backing the audit. Every entry states what was verified against
the source and what was not. Metadata that arrived wrong from the literature
search and was corrected against the publisher record is shown as CORRECTED, so
the correction is auditable rather than silent.

Verification was done by fetching the DOI or publisher page. Where a paper is
paywalled, the entry says so and reports only what the abstract or landing page
supports. Two conflicts could not be resolved before the session's web-fetch
budget ran out; both are marked UNRESOLVED rather than decided by guess.

## Contents

- [Group 1: meteorite and planetary](#group-1-meteorite-and-planetary)
- [Group 2: thin-section and petrographic segmentation](#group-2-thin-section-and-petrographic-segmentation)
- [Group 3: leakage methodology](#group-3-leakage-methodology)
- [Excluded citations](#excluded-citations)

---

## Group 1: meteorite and planetary

### 1.1 Floyd et al. 2024, CM chondrule sizes

Floyd, C.J., Benito, S., Martin, P.-E., Jenkins, L.E., Dunham, E., Daly, L.,
Lee, M.R. (2024). "Chondrule sizes within the CM carbonaceous chondrites and
measurement methodologies." Meteoritics and Planetary Science 59(10), 2769-2788.
DOI 10.1111/maps.14250.

CORRECTED: the search returned volume 109 issue 8. The publisher record is
volume 59, issue 10, pages 2769-2788.

- Task: manual measurement of chondrule size, not segmentation by algorithm.
- Data: 1,937 chondrules across CM chondrites.
- Headline: revised average CM chondrule size 194 micrometres.
- Split protocol: none. Not a machine-learning paper.
- Relevance: establishes that chondrule measurement in this field is still done
  by hand, which is the gap this project sits in.

### 1.2 Hezel et al. 2025, 2-D versus 3-D chondrule size distribution

Hezel, D.C., Metzler, K., Hochstein, M.L. (2025). Meteoritics and Planetary
Science 60(4). DOI 10.1111/maps.14336.

- PAYWALLED. Authors, year, volume and DOI verified from the landing page.
- Segmentation method (manual or automated): UNVERIFIED.
- Relevance: the 2-D versus 3-D bias it documents is the reason a volumetric
  method matters, but it supplies no evaluation protocol to audit.

### 1.3 Nath et al. 2025, MetNet

Nath, R., Mali, S., K., D., Panwar, N., Verma, A.J., Kumar, A., Mahajan, R.R.,
Basu Sarbadhikari, A., Varela, M.E., Ehgamberdiev, S.A., Kapadia, T.,
Srivastava, N. (2025). "Classifying meteorites with MetNet: a deep learning
approach using reflectance spectroscopy." Meteoritics and Planetary Science
60(5), 1063-1072. DOI 10.1111/maps.14342.

- Task: meteorite type classification from reflectance spectra. Not imaging, not
  segmentation.
- Data: 1,781 reflectance spectra.
- Split: 75/25 train/validation, verified.
- Headline: 93% validation accuracy.
- AUDIT, unresolved: whether the 75/25 split is by spectrum or by meteorite is
  UNVERIFIED (paywalled). This is the decisive question. Reflectance spectra are
  routinely measured multiple times per specimen; if several spectra of one
  meteorite fall on both sides of a per-spectrum split, the 93% is partly a
  measure of specimen recognition. The paper's own text is needed to settle it.
- AUDIT: the reported figure is a validation accuracy. No held-out test set
  distinct from the set used for model selection is described in the abstract,
  so 93% is a selection-set number unless the full text says otherwise.

### 1.4 Thoresen et al. 2024, Apollo thin sections (BEST PRACTICE)

Thoresen, F., Cowley, A., Haak, R., Lewe, J., Moriceau, C., Knapczyk, P.,
Engelschiøn, V.S. (2024). "Breccia and basalt classification of thin sections of
Apollo rocks with deep learning." arXiv:2410.21024.

- Task: binary rock-type classification of Apollo thin sections.
- Model: Inception-ResNet-v2 with SimCLR contrastive pretraining.
- Split, verified from the paper: stratified K-fold cross-validation, initial
  allocation 60% train, 20% validation, 20% test, and the load-bearing sentence,
  "For each split, images of the same sample end up in the same set."
- Code: github.com/esa/Apollo-rock-thin-section-classifier, contains
  `stratified_group_kfold.py`. Verified to exist.
- Headline: 98.44% accuracy, reported with an uncertainty of 1.47.
- AUDIT: this is the only paper in the corpus that both groups by physical
  sample and reports an uncertainty on its headline number. It is the reference
  for how the rest should have been done. The audit finds nothing to correct.

### 1.5 Jung et al. 2026, lunar mare basalt SEM

Jung et al. (2026). "Automated mineral identification and rock-type
classification of lunar mare basalts using SEM images." Journal of Geophysical
Research: Machine Learning and Computation. DOI 10.1029/2025JH001124.

- Task: multi-class mineral segmentation, U-Net with two scale-aware variants.
- Data: 1,078 unique base images, augmented to 17,248; 12,800 used for training.
- Headline: pixel accuracy 0.85 train, 0.81 validation, 0.82 test.
- AUDIT, unresolved: whether augmentation was applied BEFORE the split is
  UNVERIFIED (paywalled). The numbers are suggestive. 17,248 augmented images
  from 1,078 base images is a 16x expansion, and 12,800 of 17,248 is 74%. If the
  split was taken over the 17,248 augmented images rather than the 1,078 base
  images, then rotated and flipped copies of the same base image appear in train
  and test, and the 0.82 test accuracy is inflated. If the split was over base
  images first, it is sound. The abstract does not say.
- Note in the paper's favour: a train-to-test gap of only 0.03 (0.85 to 0.82) is
  what you would expect from a well-regularized model, but it is also what
  leakage produces. The gap alone cannot distinguish the two.

### 1.6 Griffin et al. 2012, meteorite micro-CT local histograms

Griffin, L.D., Elangovan, P., Mundell, A., Hezel, D.C. (2012). "Improved
segmentation of meteorite micro-CT images using local histograms." Computers and
Geosciences 39, 129-134. DOI 10.1016/j.cageo.2011.07.002.

RESOLVED, and the correction matters. The first literature pass returned these
authors as "Grimm and Woolum (2011)". That attribution is wrong. The Crossref
record for the DOI gives Griffin, Elangovan, Mundell and Hezel, 2012, volume 39,
pages 129-134. A fabricated author list on the single closest piece of prior art
would have been the worst citation error available in this report.

- Method: semi-automated classification from local intensity histograms. Not
  machine learning.
- Segments chondrules specifically, along with calcium-aluminium-rich inclusions
  and dark inclusions.
- Reported: 85.8% percent-correct against 73.4% for pixel-intensity
  classification.
- Relevance: this is the closest prior work to the present project by domain, and
  it is the correct baseline to beat. It segments chondrules in meteorite micro-CT
  without a learned model.

### 1.7 Mills et al. 2024, LPSC abstract 2549

"Automated segmentation of microscopic and spectroscopic images of primitive
astromaterials by machine learning." LPSC 2024 abstract 2549.

- UNVERIFIED. The abstract PDF could not be retrieved. Listed because it is the
  only candidate counterexample to the negative claim in the report, and
  suppressing it would make that claim look stronger than it is.

---

## Group 2: thin-section and petrographic segmentation

This is where the split protocols worth auditing live.

### 2.1 Leichter et al. 2022, olivine phenocrysts

Leichter, A., Almeev, R.R., Wittich, D., Beckmann, P., Rottensteiner, F.,
Holtz, F., Sester, M. (2022). "Automated segmentation of olivine phenocrysts in
a volcanic rock thin section using a fully convolutional neural network."
Frontiers in Earth Science 10, 740638. DOI 10.3389/feart.2022.740638.

- Data: ONE thin section, 48 x 28 mm, 1 micrometre per pixel, 9,000 BSE tiles.
- Model: DeepLabV3 pretrained on COCO.
- Split, verified: by analytical session. Train sessions 5 to 9 (902 tiles),
  test sessions 17, 24, 32, 33, 23, 11 (820 tiles), validation sessions 18, 28,
  29 (492 tiles).
- Headline: test IoU 85.0%, validation IoU 84.7%, overall accuracy 98.5%.
- AUDIT: splitting by analytical session is better than random tiling and the
  authors deserve credit for it. It is still a within-single-section split. The
  sessions are different areas of the same 48 x 28 mm slide, so the test tiles
  share the specimen, the preparation, the instrument settings and the operator
  with the training tiles. The number that generalizes to a NEW thin section is
  not measured here, and the paper does not claim it is. The honest reading of
  85.0% IoU is "on unseen areas of the same slide".
- AUDIT: test and VALIDATION IoU differ by 0.3 percentage points (85.0 against
  84.7). No train IoU is reported. A validation-to-test gap that small is itself
  a signal that the two session groups are highly exchangeable, which is the
  same quantity Finding 4 of the report measures at 0.144 to 0.170 for a
  contiguous split on this project's data.

### 2.2 Morales et al. 2025, petrographic thin sections

Morales, J., Saldivia, C., Lobo, R., del Pino, M., Muñoz, M., Caniggia, G.,
Catalán, J., Hayde, R., Poblete, V. (2025). "Visual analysis of deep learning
semantic segmentation applied to petrographic thin sections." Scientific Reports
15, 14612. DOI 10.1038/s41598-025-99767-2.

- Data: 414 images, 1,472 polygon annotations, three classes.
- Model: YOLO11n-seg.
- Split, verified verbatim: "The datasets were randomly divided into training,
  validation, and test sets in an 8:1:1 ratio."
- Headline: AP50-95 of 0.69 andalusite, 0.70 biotite, 0.74 ooids.
- AUDIT: the split is random at the image level and the paper does not state
  whether images from one thin section were kept together. The number of
  distinct physical thin sections behind the 414 images is UNVERIFIED, and the
  paper does not report it. Without that number the reported AP cannot be
  interpreted, because 414 images from 414 sections and 414 images from 8
  sections are entirely different claims.
- AUDIT, precision: the test split is 10% of 414 images, roughly 41 images
  carrying on the order of 147 annotations, and those annotations are split
  across three classes, so perhaps 50 per class. AP is not a simple proportion,
  so an exact interval cannot be recovered from the published numbers, but a
  binomial standard error at a rate near 0.70 gives the right order of
  magnitude: about 0.038 at n = 147, and about 0.065 at n = 50. The three
  reported values, 0.69, 0.70 and 0.74, sit within roughly one such standard
  error of each other. They cannot be read as a ranking of the three classes,
  and the second decimal place is not resolved by a test set this size. The
  paper reports no interval, so a reader cannot see this.

### 2.3 Bihani et al. 2021, MudrockNet

Bihani, A., Daigle, H., Santos, J.E., Landry, C., Prodanovic, M., Milliken, K.
(2021). "MudrockNet: semantic segmentation of mudrock SEM images through deep
learning." Computers and Geosciences 157, 104927. arXiv:2102.03393.

- Model: DeepLab-v3+ with a ResNet101 backbone.
- Split, verified verbatim from the repository README: "The images were split
  randomly into training, validation and test datasets in a ratio 70:15:15."
- Code: github.com/abhishekdbihani/MudrockNet, verified to exist.
- REPORTED IoU, RESOLVED with a caveat worth recording. The published Computers
  and Geosciences version reports mean IoU of silt 0.6663, clay 0.7797, pores
  0.6751. The arXiv preprint reports different values (silt 0.6591, pores
  0.6642) and no clay result. The two versions of the same work disagree, which
  is why the earlier verification passes conflicted. Cite the published values,
  and note that a reader working from the preprint gets different numbers.
- AUDIT, and this is the strongest leakage case in the corpus after Amer 2025:
  the reported dataset structure is 1,785 images produced by cropping and
  augmenting a much smaller set of base SEM images, described as 49 SEM images
  from 5 core samples, and the augmentation is applied BEFORE the 70:15:15
  random split. If that is right, then flipped copies and neighbouring crops of
  the same base image sit on both sides of the split, and the effective sample
  size for a generalization claim is 5 cores rather than 1,785 images.
  PARTIALLY VERIFIED: the split sentence is quoted from the repository README
  and is solid; the augmentation ordering and the 49-images-from-5-cores figures
  come from the dataset description and could not be confirmed against the
  paper's own methods text, because the PDF would not extract. Treated as
  reported-not-quoted, and the audit does not lean on it.
- This is the protocol reconstructed in the audit as `P-tile-random`.

### 2.4 Mahmud et al. 2023, 3D sandstone volume

Mahmud, B.U., Hong, G.Y., Mamun, A.A., Ping, E.P., Wu, Q. (2023). "Deep
learning-based segmentation of 3D volumetric image and microstructural
analysis." Sensors 23(5), 2640. DOI 10.3390/s23052640.

- Data: an open-source sandstone dataset, 448 images of 512 x 512 x 512, four
  classes.
- Model: 3D U-Net with a VGG19 backbone, 64-voxel patches.
- Split, verified verbatim: "The data are further split into training data which
  is 80% of the total, while the validation data comprise 10% and the testing
  data comprise 10% of total."
- Headline: IoU 91.12%, accuracy 96.78%, F1 95.25%.
- AUDIT, aggregation: the reported IoU and F1 are consistent with each other
  only under macro-averaging. Per class, Dice = 2 x IoU / (1 + IoU), so an IoU
  of 0.9112 implies a Dice of 0.9535. The reported F1 is 0.9525, slightly below.
  That direction is exactly what averaging over four classes produces, because
  the map from IoU to Dice is concave, so the mean of the per-class Dice values
  is at or below the value implied by the mean IoU. The numbers are internally
  consistent, and they confirm the averaging is macro over classes rather than
  micro over voxels. Micro and macro are not interchangeable under the class
  imbalance in this data, and the paper does not say which it used.
- AUDIT, precision: 91.12% is four significant figures from a test split of 10%
  of 448 images, roughly 45 volumes. The fourth digit is not supportable.
- AUDIT, structure: whether the 448 images come from one physical sandstone
  specimen or many is UNVERIFIED and the paper does not say. If they are
  sub-volumes of one scan, the 80/10/10 random split is the 3D analogue of the
  audit's random protocols and the 91.12% is a within-specimen number. NOT
  MEASURED HERE: the audit's 3D-patch protocol was specified and never
  implemented (spec R3), so the closest reconstruction actually run is 2D.

### 2.5 Su et al. 2020, rock classification

Su, C., Xu, S.-J., Zhu, K.-Y., Zhang, X.-C. (2020). "Rock classification in
petrographic thin section images based on concatenated convolutional neural
networks." Earth Science Informatics 13, 1379-1391.
DOI 10.1007/s12145-020-00505-1. arXiv:2003.10437.

- Data: 92 rock samples, 196 thin sections, 588 images, 63,504 patches.
- Split, verified: five-fold cross-validation. The level at which folds were
  formed is NOT stated in the paper.
- Headline: overall accuracy 89.97%.
- AUDIT: this is the clearest case of illusory precision in the corpus. The
  accuracy is quoted to four significant figures over 63,504 patches, but those
  patches come from only 92 rock samples. If the folds were formed over patches,
  the effective sample size for a generalization claim is 92, not 63,504, and
  the confidence interval is roughly 26 times wider than the patch count
  suggests. The paper's own numbers make the leakage question unavoidable, and
  the paper does not address it.

### 2.6 Fan et al. 2025, RoImAI foundation model

Fan, J., Yu, X., Di, Y., Lv, T., Zhang, R., Bao, J., Liu, Y., Li, L., Pan, X.
(2025). "A foundation model for rock thin-section images analysis."
Communications Engineering 5, 9. DOI 10.1038/s44172-025-00565-5.

- Data: a large multi-region Chinese thin-section corpus, plus four independent
  external validation cohorts from different regions.
- Split, verified: the internal cohort was "randomly divided into internal
  training and validation cohorts in a 4:1 ratio", with four separate external
  cohorts held out by region.
- Headline: internal AP 0.59, Dice 0.88, F1 0.83, IoU 0.80. External AP 0.64,
  0.59, 0.55, 0.56.
- AUDIT, aggregation: same check as Mahmud. IoU 0.80 implies a per-class Dice of
  0.889, and the reported Dice is 0.88, just below. Consistent with macro
  averaging over classes; not an error.
- AUDIT, and this is the corpus's best practice after Thoresen: this paper does
  the thing the others do not, which is hold out entire geographic regions and
  report the drop. Three of the four external AP values (0.59, 0.55, 0.56) sit
  at or below the internal 0.59. The paper therefore supplies its own evidence
  that the internal random split is the optimistic number. That is exactly the
  comparison this audit had to reconstruct by experiment for everyone else.

### 2.7 Amer et al. 2025, MUMDMC2025 mineral dataset

Amer, B.G., Mousa, H.M., Dawoud, M., Youssef, A. (2025). "A photomicrographic
dataset of rocks for the accurate classification of minerals." Scientific Data
12, 1775. DOI 10.1038/s41597-025-05879-9.

- Data: 14,400 photomicrographs, five mineral classes, 2,880 images per class.
- Structure, verified verbatim: "Each mineral specimen was systematically imaged
  at 72 rotational positions (5 degree increments, 360 degree coverage)" under
  both plane- and cross-polarized light.
- Split, verified verbatim: "Dataset partitioning employed a stratified 80:20
  train-test split implemented through scikit-learn's train_test_split
  function", maintaining class proportions.
- AUDIT, and this one is mechanically certain rather than suspected: sklearn's
  `train_test_split` with `stratify` balances class proportions. It has no
  grouping argument and does not keep records from the same source together.
  Given 72 rotational views of each specimen and a stratified random 80/20
  split, roughly 80% of each crystal's 72 views land in train and the rest in
  test, so essentially every test image has 50-odd near-duplicate views of the
  same physical crystal in the training set. Any benchmark accuracy computed on
  this split measures crystal re-identification, not mineral classification. The
  paper does not state that views are grouped, and the named function cannot do
  it.
- This is a dataset paper, so the harm propagates: every future paper that
  adopts the published split inherits the leak.

### 2.8 Yu et al. 2022, superpixel annotation

Yu, J., Wellmann, F., Virgo, S., von Domarus, M., Jiang, M., Schmatz, J.,
Leibe, B. (2022). "Superpixel segmentations for thin sections: evaluation of
methods to enable the generation of machine learning training data sets."
Computers and Geosciences.

CORRECTED: the search returned article number 104984 with DOI
10.1016/j.cageo.2021.104936, which are mutually inconsistent. The verification
pass returned DOI 10.1016/j.cageo.2022.105232.

- An annotation-tooling paper, not a model-evaluation paper. No split to audit.
  Included because it documents how training data in this field gets made:
  superpixel-assisted labelling of a small number of large sections, which is
  precisely the structure that makes random patch splits leak.

---

## Group 3: leakage methodology

Used to place the measured effect sizes in context.

### 3.1 Kapoor and Narayanan 2023

Kapoor, S., Narayanan, A. (2023). "Leakage and the reproducibility crisis in
machine-learning-based science." Patterns 4(9), 100804.
DOI 10.1016/j.patter.2023.100804.

CORRECTED: the search returned issue 8 and article 100859. The publisher record
is issue 9, article 100804.

- Survey finding: leakage affects 294 papers across 17 fields.
- Taxonomy: eight distinct leakage types. (The L1/L2/L3 grouping quoted in the
  first search pass is a coarser paraphrase, not the paper's own numbering.)
- Case study: in civil-war onset prediction, the apparent advantage of complex
  models over logistic regression falls from 0.14 AUC to 0.01 once leakage is
  corrected.

### 3.2 Yagis et al. 2021, brain MRI (the closest analogue)

Yagis, E., Atnafu, S.W., García Seco de Herrera, A., et al. (2021). "Effect of
data leakage in brain MRI classification using 2D convolutional neural
networks." Scientific Reports 11, 22544. DOI 10.1038/s41598-021-01681-w.

Subject-level split against slice-level split, verified from Table 4:

| Dataset | Subject-level | Slice-level |
| --- | --- | --- |
| OASIS | 66 to 69% | 94 to 97% |
| ADNI | 66 to 70% | 95 to 96% |
| PPMI | 46 to 48% | 93 to 96% |

- Control: on randomly relabelled data, the slice-level split still reaches 95%
  while the subject-level split sits at 50%. The leaked protocol reports a
  strong result on data with no signal at all.
- This is the direct analogue of the present audit. A 3D scan cut into 2D slices
  and split at the slice level is structurally the same object as a micro-CT
  volume cut into slices and split at the slice level.

### 3.3 Kattenborn et al. 2022

Kattenborn, T., Schiefer, F., Frey, J., Feilhauer, H., Mahecha, M.D.,
Dormann, C.F. (2022). "Spatially autocorrelated training and validation samples
inflate performance assessment of convolutional neural networks." ISPRS Open
Journal of Photogrammetry and Remote Sensing 5, 100018.
DOI 10.1016/j.ophoto.2022.100018.

CORRECTED: the search attributed this to Environmental Research Letters with DOI
10.1088/1748-9326/ac7921. That is the wrong journal and the DOI does not
resolve. The correct venue is ISPRS Open Journal.

- Finding: random validation overestimates CNN performance by up to 28% relative
  to spatially blocked validation, and the overestimate grows as the training
  sample shrinks.

### 3.4 Karasiak et al. 2021

Karasiak, N., Dejoux, J.-F., Monteil, C., Sheeren, D. (2021). "Spatial
dependence between training and test sets: another pitfall of classification
accuracy assessment in remote sensing." Machine Learning 111, 2715-2740.
DOI 10.1007/s10994-021-05972-1.

- Finding: up to 28% overestimation from random rather than spatial
  cross-validation.

### 3.5 Roberts et al. 2017

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J.,
Guillera-Arroita, G., et al. (2017). "Cross-validation strategies for data with
temporal, spatial, hierarchical, or phylogenetic structure." Ecography 40(8),
913-929. DOI 10.1111/ecog.02881.

- The paper is verified. The widely repeated "up to 40% more optimistic" figure
  could NOT be verified against the original text (access blocked), only against
  secondary sources. It is therefore NOT used as a number in the audit. The
  paper is cited for the blocked-cross-validation principle only.

### 3.6 Valavi et al. 2019, blockCV

Valavi, R., Elith, J., Lahoz-Monfort, J.J., Guillera-Arroita, G. (2019).
"blockCV: an R package for generating spatially or environmentally separated
folds for k-fold cross-validation of species distribution models." Methods in
Ecology and Evolution 10(2), 225-232. DOI 10.1111/2041-210X.13107.

- Cited for the buffered and blocked fold construction that the reference
  protocol in this project independently implements along z.

---

## Excluded citations

These surfaced in the literature search and are deliberately NOT used anywhere
in the audit, because they could not be verified to a standard that would let
them carry a number.

| Citation | Status | Reason not used |
| --- | --- | --- |
| arXiv 2604.04199, Roth, "Which leakage types matter? A quantitative landscape across 2,047 benchmark datasets" | EXISTS, title and author confirmed on the abstract page | Its effect sizes were never read from the paper, so no number from it is used. |
| arXiv 2607.12278, Zhang, Zhou, Kang, Li, "Auditing data leakage in whole-slide image multimodal benchmarks" | EXISTS, confirmed | Same. |
| arXiv 2606.24944, Albzour, leakage-aware leukemia benchmark | EXISTS, confirmed | Same. |
| arXiv 2604.10965, Korkmaz, "bioLeak" | EXISTS, confirmed | Same. |
| ResearchGate 403828036, trustcv taxonomy | UNVERIFIED, server returned 403 | Content never read. Not cited. |
| PLOS ONE micromorphology CNN, PMC12818656 | UNVERIFIED | The PMC identifier does not correspond to the article. A DOI was offered as correct but never independently confirmed. Not cited. |
| Siyari and Hosseini, sandstone thin sections | UNVERIFIED | The citation carried an impossible volume number and an article number inconsistent with its DOI. Not cited. |
| Roberts et al. 2017, the "up to 40% more optimistic" figure | Paper VERIFIED, the number UNVERIFIED | Two attempts failed to find the figure in the original text; publisher access was blocked and the PDF would not extract. The paper is cited for its principle only, never for that number. |

CORRECTION, recorded rather than quietly fixed. The first four rows were
originally excluded here on the grounds that their existence could not be
confirmed. A later pass reached the arXiv abstract pages and confirmed all four,
including author names. The exclusion was over-cautious, and the table above now
says so. They remain unused because their reported effect sizes were never read
from the papers themselves, which is a different and lower bar than not existing.

The general rule the audit applies to itself: a citation is used for a number
only when that number was read from the source. Existence is necessary and not
sufficient.
