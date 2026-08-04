# Session handoff, 2026-08-02

Supersedes the 2026-08-01 handoff. Read [CHANGELOG.md](CHANGELOG.md)'s KNOWN
ISSUES block before quoting any number.

## A JOB IS RUNNING RIGHT NOW

```
python -m src.pretrained_probe --arm=pretrained     PID 81137
log: logs/pretrain_probe.log       ~13/60 epochs at handoff, ~100-150 s/epoch
```

It survives the session that started it. **Check it first:**
`tail -3 logs/pretrain_probe.log` and `pgrep -f pretrained_probe`.

When it finishes it writes `results/pretrained_probe.json` and a leaderboard row
`resunet18_pretrained_s30000`. **Then run the control arm:**

```
python -m src.pretrained_probe --arm=scratch
```

One at a time. Do not run both together (see "respect the machine" below).

## What the probe is for, and why the control arm is not optional

Sky, 2026-08-01: *"The end goal is not pretrained. The pretrained model having
better success will just show us we need more data."*

Pretraining substitutes for data, so the gap between an ImageNet encoder and the
**identical architecture randomly initialised** estimates how much data-driven
headroom exists. It is an instrument, not a candidate model.

Both arms are ResNet-18 encoder + U-Net decoder, 14.0M params, differing ONLY in
encoder initialisation (verified: conv1 weights differ, and the pretrained arm's
conv1 is bit-identical to torchvision's ImageNet tensor). Read the results as:

| comparison | measures |
|---|---|
| pretrained − scratch | **pretraining = the data-starvation estimate** |
| scratch − 0.29409 | architecture (ResNet-UNet vs plain U-Net), a separate question |
| pretrained − 0.29409 | both mixed; NOT interpretable alone |

**[MUST] Judge on TEST IoU, not validation.** Sky, 2026-08-02: "the real test is
the test IoU". This project has had **eight val/test sign flips**; geometry
augmentation won val by +0.0157 and lost test by −0.0501. Each arm scores test
once through `evaluate.run_and_score` with the threshold frozen on val.

**How big a gap counts as evidence, decided BEFORE the numbers arrived:** both
arms are n=1 seed, and `src/data_scaling.py` states single-seed differences of
0.03-0.06 cannot be separated from run-to-run scatter. So a test gap under ~0.05
is ambiguous; 0.10 or more is hard to explain as scatter and would genuinely
support "data is the binding constraint".

## Where the numbers stand (all post-tiler-fix, the only comparable set)

| | test IoU |
|---|---|
| CNN `unet2d_s30000` | **0.29409** (Dice 0.455, prec 0.372, recall 0.584, AUPRC 0.427) |
| + prune (margin baseline) | 0.3028 test, but **−0.0071 on val**: below the floor and sign-flipped, not a gain |
| CNN + geometry augmentation | 0.24404 (−0.0501, replicated at two budgets) |
| classical `plus3_ws` | 0.2185 (**defect-contaminated**, see KNOWN DEFECTS) |
| classical `plus3` | 0.2127 |
| classical `base18` | 0.2028 (reproduces the published 0.203 baseline) |
| annotation ceiling | 0.9226, so 0.294 is ~32% of achievable |

**Do not read the leaderboard's top row.** `unet2d_s15000` at 0.3089 sits above
`unet2d_s30000` but is pre-tiler-fix, as is every other row there.

## UNVERIFIED and potentially important: the chondrule count may be wrong

Measured this session with the same-id 3D rule (`count_chondrules_3d`) plus the
valid mask:

```
train 63    val 36    test 43    sum 142   (matches split.json n_chondrule_instances: 142)
```

But `src/pipeline_v2.py:81` sets `SE_REF_N = 19`, and **every noise-floor figure
in this project scales from that 19**. If val really holds 36, the MDE tightens
from 0.0142 to roughly 0.0103.

Hypothesis for where 19 came from: it may be a plain BINARY connected-components
count of the val zone, i.e. the exact undercount this repo documents (87 vs 142,
a 39% shortfall) frozen into the constant. `split.json` records no per-zone
counts, and `CHANGELOG.md:950` says the 0.0071 SE was bootstrapped over
validation **slices**, not chondrules, so the scaling unit may also be wrong.

**Two verifiers were launched to check this and were stopped for memory before
reporting. Nothing has been changed on the strength of it.** Countervailing
consideration: the floor uses `2·SE` where the convention is `2.8·SE`, and it
ignores training-seed variance, both of which would WIDEN the floor. Settle both
before touching `SE_REF_N`, because it propagates into every
`below_noise_floor` flag the pipeline emits.

## Respect the machine (violated twice this session)

1. Four concurrent training jobs took epochs from 70 s to 250 s and pushed the
   Mac into swap.
2. Two `harness-verifier` agents each called `evaluate.load(".")`, which pulls
   train, val AND test into memory, several GB apiece. Sky flagged it.

`evaluate.load` takes `want_test=False`. Use it. One data-loading process at a
time, and tell any agent that touches these volumes which zones it actually
needs. Note that macOS reports swap as the cumulative swap-FILE size, which does
not shrink; check `vm_stat` pageout RATE for real pressure, not the swap total.

## Next steps

1. Let the pretrained arm finish, then run `--arm=scratch`, then report both TEST
   IoUs with full metric sets and the gap against the ~0.05 threshold above.
2. Settle the `SE_REF_N` question above (verifiers, then fix or leave with a
   recorded reason).
3. Finish the data-scaling curve: still **1 of 12 runs** after three attempts,
   and it is the direct test of "we need more data". ~10 h, resumable.
4. The two arms Sky authorised on 2026-07-30 and that have never run: CNN with
   cleaned channels, and CNN cascaded on the classical probability. Cascade
   channel is ready (`results/probs/classical_xf_*.npy`, 5-fold, all 213 train
   slices). The cleaned-channels arm needs `CNNRunner`'s aux path widened from 1
   channel to N.
5. Fix the defects in CHANGELOG.md's KNOWN DEFECTS block.

## Traps that cost time this session

- **Untrained-net probes cannot validate an inference change.** A whole-slice
  inference "fix" passed every check on an untrained net, then collapsed the
  trained model (val IoU 0.4487 to 0.2632).
- **Corrections propagate badly.** Retractions landed in the CHANGELOG and were
  left standing in code docstrings, the README and the ROADMAP. When retracting a
  number, grep the whole repo for it.
- **Diagnose before acting.** A 2 h cross-fit re-run was spent on a hypothesis a
  90-second in-fold-vs-out-of-fold check refuted.
- **Two logs can share a name.** `logs/data_scaling_v2.log` and
  `results/data_scaling_v2.log` were different jobs.
- **The commit gate keys review state by the SESSION's directory.** Reviews run
  from a worktree session do not satisfy a commit made in the main checkout, and
  no agent in that session can update the main key. Also, its `[review-skip]`
  hatch is only read from `-m`/`--message`; with `-F file` the hatch is invisible.
  Commit e37aa0e used the hatch with Sky's explicit approval, logged.
