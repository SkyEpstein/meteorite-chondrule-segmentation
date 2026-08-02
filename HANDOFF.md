# Session handoff, 2026-08-01

Supersedes the 2026-07-29 handoff entirely. Read [CHANGELOG.md](CHANGELOG.md)'s
KNOWN ISSUES block before quoting any number.

**Nothing is running.** No background jobs, no suspended processes. The machine
is idle.

## Sky's current position

> "I think there is potential, we just need more data."

And, separately, on using a pretrained model:

> "The end goal is not pretrained. The pretrained model having better success
> will just show us we need more data."

That reframing matters: **pretraining is an instrument, not a deliverable.**
Pretraining substitutes for data, so the gap between a pretrained encoder and the
from-scratch model on the same 213 slices is an *estimate of how much data-driven
headroom exists*. Sky's chosen next action is **commit, then test pretraining on
those terms**. The sharper version, if it can be afforded: measure the pretrained
model's own scaling curve, so the answer is a substitution rate ("pretraining is
worth roughly Nx our data") rather than a single delta.

## The one thing that matters most

**Almost nothing measured this session cleared the noise floor.** MDE is 0.0142
IoU at 19 validation chondrules. Against that: the prune gain +0.0084, every
confidence model 1e-9 to 1e-5, z-refine inside the floor. The recurring failure
mode has been *inability to measure*, not absence of effect.

The two results that DID clear it were both negative: geometry augmentation
-0.0501 test IoU, and the image confidence model -0.1769 (the latter confounded,
see below).

## Where the numbers stand

| | test IoU | note |
|---|---|---|
| CNN `unet2d_s30000` | **0.29409** | the current base model, post-tiler-fix |
| + prune (margin baseline) | 0.3028 | +0.0084 on test, but **-0.0071 on val**: below the floor, sign-flipped |
| CNN + geometry augmentation | 0.24404 | -0.0501, replicated at two budgets, n=1 seed each |
| classical `plus3_ws` | 0.2185 | **defect-contaminated**, see KNOWN DEFECTS |
| classical `plus3` | 0.2127 | +0.0099 over the 18-feature baseline, paired comparison |
| classical `base18` | 0.2028 | reproduces the published 0.203 baseline |
| annotation ceiling | 0.9226 | so we are at ~32% of achievable |

## Decisions Sky made, still open (do NOT re-ask, DO act on)

1. **The complete model** is constitution §12: seven stages, hierarchy first,
   pixel dropping last on three stacked confidence signals. That supersedes the
   four-stage base-first order still written in the spec and ROADMAP.
2. **"Both, and let validation decide"** on the classical map: build (a) a CNN
   taking the cleaned channels directly and (b) a CNN cascaded on the classical
   probability, then compare on val. **Neither arm has been run.** The cascade
   channel is ready (`results/probs/classical_xf_*.npy`, 5-fold, all 213 train
   slices). Arm (a) needs `CNNRunner`'s aux path widened from 1 channel to N.
3. **Cross-fit the CNN** so the error model gets more than 28 slices.
   `src/cnn_crossfit.py` is written and its three blocking defects are fixed, but
   it has **never completed a run** (~6 h).
4. **Fix `error_cnn`'s unweighted sampling, then re-run it** before trusting its
   -0.1769. Not done.

## What is confounded and must not be cited as-is

- **The image confidence model's -0.1769.** Four confounds: fit on ~9 chondrules;
  the base map is 91% saturated; `fit_error_unet` samples patches **uniformly**
  with no positive weighting while `PatchSet` uses `pos_frac=0.5`; and val/test
  has flipped sign eight times in this project. On the held-out val half it
  reached 0.1948 pruned versus **0.4092 for the free margin baseline**, so it
  lost to a zero-parameter signal on validation too, before any transfer
  question.
- **The classical `plus3_ws` +0.0058 watershed gain.** `watershed_split`
  silently deletes small objects, so the gain may be an artifact of dropping
  objects the metric then stops penalising.
- **The hierarchy's -0.012.** `results/hier_then_cnn_s30000.json` carries no
  regime string and predates the retrain. If its 0.30374 is post-fix, the honest
  delta against 0.29409 is **+0.0097**, opposite in sign.

## Traps this session hit, in order of how much time they cost

1. **Untrained-net probes cannot validate an inference change.** A whole-slice
   inference "fix" passed every check I ran on an untrained net, then collapsed
   the trained model. Compare val IoU on the same saved weights instead.
2. **Corrections propagate badly.** Retractions landed in the CHANGELOG and were
   left standing in code docstrings, the README and the ROADMAP. When retracting
   a number, grep the whole repo for it.
3. **Diagnose before acting.** A 2 h cross-fit re-run was spent on a hypothesis
   that a 90-second in-fold-vs-out-of-fold check refuted.
4. **Two logs can share a name.** `logs/data_scaling_v2.log` and
   `results/data_scaling_v2.log` were different jobs; checking only one produced
   a wrong claim about what a killed job had written.
5. **Four concurrent jobs pushed the Mac into 3.4 GB of swap** and took epochs
   from 70 s to 250 s, against constitution §11's 2-worker default.

## Next steps

1. **Test pretraining as an instrument** (Sky's choice). A torchvision ResNet-18
   or ResNet-34 encoder with the existing U-Net decoder is ~45 MB, nothing like
   the SAM-scale downloads that motivated `run_families.EXCLUDED`. Hold the
   split, budget and threshold discipline fixed so the gap is interpretable.
   **Recount chondrules first**: the "~99 in train" figure is unverified (3D
   connected components give 52 or 47; `split.json` implies ~76), and any
   substitution rate depends on it.
2. **Finish the data-scaling curve.** It stands at **1 of 12 runs** after three
   attempts and is the direct test of Sky's "we need more data". ~10 h,
   resumable.
3. Run the two arms from decision 2 above.
4. Fix the defects in the KNOWN DEFECTS block of CHANGELOG.md.
