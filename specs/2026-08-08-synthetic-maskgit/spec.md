# Synthetic training data via a label-conditioned VQ-VAE + MaskGIT

Date: 2026-08-08. Proposed by Sky's mentor, citing Microscopy and Microanalysis
32(S1) ozag053.1004, which applies a multi-scale conditioned MaskGIT with a
VQ-VAE to SEM micrographs. Sky chose the faithful two-stage build over three
cheaper alternatives.

## Problem

Annotation, not acquisition, is this project's gate: raw chondrite CT is
plentiful and published chondrule labels do not exist. We have **142 labeled
chondrules total** (63 train / 36 val / 43 test) across 213 labeled training
slices. If labeled examples can be synthesised with exact ground truth, the gate
moves.

## What the cited abstract does and does NOT do

It demonstrates unconditional synthesis, semantic inpainting, and outpainting of
SEM micrographs, conditioned on "the underlying spatial distribution of
primitives". It reports **no quantitative metrics** and does **not** demonstrate
generating (image, label) pairs to train a segmenter. The label-conditioned path
is the part we have to build, and it is the part the abstract does not evidence.
Recorded so nobody later cites the abstract as support for a result it never
claimed.

## Approach

Stage 1, **tokenizer**. A VQ-VAE over 2D patches: encoder to a token grid,
codebook lookup, decoder back to pixels. Trained with reconstruction plus
commitment loss.

Stage 2, **conditioned MaskGIT**. A bidirectional transformer over the token
grid. Training masks a random fraction of image tokens and predicts them from
the visible ones plus a spatially aligned label embedding. Sampling starts fully
masked and decodes iteratively in parallel, keeping the most confident
predictions each round.

Stage 3, **pair generation**. Sample a label map, condition on it, decode an
image. The pair is (generated image, the label map that conditioned it), so
labels are exact by construction.

Stage 4, **the confidence stages**, and this is the target. [RETARGETED
2026-08-08 by Sky, after the memorisation finding below.]

## [MUST] Synthetic data is HELD OUT from base training. Here is why.

The obvious use, adding synthetic pairs to the segmenter's training set, gives
the confidence stages **nothing**. The base model records **train IoU 1.000**
against ~0.32 on test (CHANGELOG 2026-07-23): it memorises its training set
completely. The confidence stages learn "where is this model wrong?", so they
need data where the model is actually wrong. Anything in base training has no
errors to learn from. This project already enforced the same principle in a
different place, requiring the cascade channel to be out-of-fold because "a base
model's predictions on its own training data are memorised".

So synthetic pairs never enter base training. The base predicts on them, errs
genuinely, and those errors become confidence-fitting data with exact labels.

**The constraint this attacks.** The confidence stages currently fit on **60
validation slices holding 19 to 36 chondrules**, cross-fit over five contiguous
buffered folds of about 12 slices each. That is the stage whose every measured
effect has landed below the noise floor: no single learned signal beats the free
`|p - 0.5|` margin, and the three-signal stack loses to its own best component by
0.0114 on val. Synthetic held-out data can supply thousands of slices, a 50 to
100x increase for the one stage that is plausibly data-starved rather than
constrained by the meteorite count.

## The gate that decides this arm

**The base model's errors on synthetic data must resemble its errors on real
held-out data.** If the generator produces systematically easier images, the
confidence model learns a different problem and transfers nothing, however good
the samples look. Before any confidence model is fitted, compare on the base
model:

| statistic | on real val | on synthetic held-out |
|---|---|---|
| IoU at the frozen threshold | 0.44512 | must be comparable |
| precision / recall | 0.372 / 0.584 (test-side ref) | must be comparable |
| FP / FN pixel ratio | 2.37 (test-side ref) | must be comparable |
| error blob size distribution | measured | must be comparable |

If the base scores far higher on synthetic than on real, the arm is dead and the
result is "the generator makes images that are too easy", which is a real and
publishable finding about this approach, not a failure to report.

## [LEAKAGE][MUST] The generator's data boundary

Measured from `split.json`, absolute slice indices in the every-4 volume:

```
annotated band   100..496
val zone         180..239     OFF LIMITS  (60 slices)
test zone        260..339     OFF LIMITS  (80 slices)
unlabeled head     0..99      81 slices from val   -> allowed
unlabeled tail   497..530    158 slices from test  -> allowed
```

- The **tokenizer** may see train-zone slices and both unlabeled regions:
  213 + 134 = **347 slices**. Both unlabeled regions sit far outside the
  annotated band and are separated from the nearest held-out slice by 81 and 158
  slices, against this project's own buffer standard of 12.

  [2026-08-08] The implementation initially exceeded this authorised set twice,
  admitting first 44 and then 2 `buffer` slices. The spec was correct both
  times; the code was not. `tests/test_synth_boundary.py` now checks the
  implementation against `split.json` rather than against its own reasoning.
- The **MaskGIT** stage needs labels, so it sees the **213 train slices only**.
- No synthetic sample may be conditioned on a val or test label map.

A generator that has seen val or test pixels launders them into the segmenter's
training set and voids every downstream number. This boundary is asserted in
code, not left to convention.

## Why this could work, and the ceiling it cannot pass

It could work because 63 training chondrules is very few for learning boundary
and shape priors, because synthetic labels are exact where the human annotation
ceiling is 0.9226, and because the tokenizer exploits 134 slices the segmenter
cannot currently use at all.

It cannot pass this ceiling: **a generator fitted to ALH 83010 samples from ALH
83010.** It can multiply examples; it cannot invent chondrule diversity absent
from this meteorite. So it addresses "too few labeled examples" and does nothing
for "only one meteorite". Those are different constraints and only the first is
in scope.

## The measurement problem, stated before any number arrives

Test holds 43 chondrules, and full-budget seed SD is **0.01403** for UNet2D and
**0.04517** for ResNet18-UNet. A real +0.03 gain from synthetic data would be at
or below the noise. **Registered in advance:** a gain under about 0.03 on test
will NOT be called a result, whatever it looks like, and the comparison is run at
3 seeds per arm so the arms have spreads rather than points.

## Tasks, each with its own acceptance

1. **Patches and the leakage boundary.** Extract patches; assert the allowed
   slice set matches the table above. Acceptance: an automated check that fails
   if any val or test slice index enters the generator's pool.
2. **VQ-VAE.** Acceptance, in two parts.

   (a) PRE-REGISTERED: reconstructions beat a matched-bitrate downsample
   baseline on PSNR, at the model's OWN token grid; codebook usage is not
   collapsed (perplexity at least 10% of the codebook). Both are necessary and
   neither is sufficient.

   (b) [ADDED 2026-08-08, AFTER seeing the data. Not pre-registered, and it
   should have been.] SEGMENTER PRESERVATION, measured by `--segcheck`: run the
   base segmenter on an original patch and on its reconstruction and compare.
   A tokenizer that changes what the segmenter sees cannot produce synthetic
   images whose ERROR STRUCTURE resembles real data, which is the only thing
   this arm needs from it. Part (a) passed at every setting while f=8 more than
   doubled the predicted-positive fraction, so (a) alone is not a gate.

   No numeric bar was registered in advance, and f=8 was rejected on the
   judgment that a 2.28x inflation of predicted-positive cannot yield a
   transferable error structure. Registering one now, for task 3 onward:
   **agreement IoU at least 0.60, decision flips at most 10%, and
   predicted-positive inflation at most 1.20x.** f=4 clears all three (0.6168,
   7.3%, 1.14x) and f=8 clears none (0.3302, 23.6%, 2.28x). Because this bar was
   set after the measurement, it must not be treated as evidence that f=4 is
   good enough; it is a floor for rejecting future candidates, and the real test
   of f=4 is task 4.

   NOTE: `gate_passed` in `results/synth/vqvae_f*.json` refers to part (a) only.
   `vqvae_f8_g1.json` records `gate_passed: true` for a tokenizer this arm
   rejected.
3. **Conditioned MaskGIT.** Acceptance: samples conditioned on a given label map
   place chondrule-like structure where the label says, measured by running the
   EXISTING segmenter on generated images and reporting IoU against the
   conditioning mask. A generator that ignores its condition fails here, and
   this is the check that catches it.
4. **Error-structure match.** Run the EXISTING base model on synthetic held-out
   pairs and compare its error statistics against real val, per the gate table
   above. Acceptance: comparable, or the arm stops here with that as the result.
5. **Confidence stages on synthetic.** Fit the regressor, error U-Net and
   stacker on synthetic held-out data instead of, and then alongside, the 60
   real val slices. Apply to real TEST once. Acceptance: reported against the
   current numbers (`results/confidence_stack.json`: stacked val 0.44731 /
   test 0.28661, best single `regressor` val 0.45874 / test 0.29316, free margin
   val 0.43807 / test 0.30279) and against the 0.03 band registered above. The
   bar to beat is the FREE margin baseline, which no learned signal has yet
   cleared.

## Honest risks, recorded up front

- **Data volume.** The abstract's regime is "a few thousand images"; we have 347
  slices, croppable into many patches but with far less true diversity. A
  data-starved tokenizer producing blurry or mode-collapsed codes is the single
  most likely failure, which is why task 2 has its own acceptance gate.
- **Modality.** The abstract is 2D SEM surface contrast; this is 3D X-ray
  attenuation. Generated slices are independent 2D samples with no z-coherence,
  so synthetic data can train the 2D segmenter but must never feed the
  z-refine or z-agreement stages.
- **Compute.** Two networks trained locally on a Mac, on top of a project where
  a single segmenter run takes 1.5 to 4.7 hours.
- **It may be unmeasurable.** See the measurement problem above. Building it and
  reporting "no resolvable difference" is a legitimate and likely outcome, and is
  worth more than not knowing, but it should be expected rather than discovered.
- **[ADDED 2026-08-14, and it cuts against the arm's central premise.] The
  generator is fitted on the base segmenter's OWN training slices.** The
  retarget rests on "held out of base training, so the base errs genuinely on
  synthetic data". But the same memorisation argument that motivated the
  retarget (train IoU 1.000) also predicts that samples drawn from the
  distribution the base has memorised may be systematically EASIER for it than
  real held-out data, so its errors on them could be softer and differently
  shaped. Held-out-ness is necessary and may not be sufficient. Task 4's gate is
  what would catch this, and it is the reason that gate is not optional. This
  risk was not named when the retarget was decided; it should have been.
