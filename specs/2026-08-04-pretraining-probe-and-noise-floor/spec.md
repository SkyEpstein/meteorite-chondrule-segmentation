# Pretraining probe, noise floor, and the two unrun arms

Date: 2026-08-04
Status: implemented, pending commit
Supersedes nothing. Picks up the handoff committed in 18ab53a.

## Problem

Three questions are open at the same time, and two of them can invalidate
numbers the project already reports.

1. **How much headroom is data?** A ResNet-18 U-Net is training right now with an
   ImageNet encoder (PID 81137, `--arm=pretrained`). The identical architecture
   with a random encoder has never been run. Without that control arm the
   pretrained number measures nothing, because architecture and initialisation
   are confounded.
2. **Is the noise floor wrong?** `src/pipeline_v2.py` sets `SE_REF_N = 19`
   validation chondrules. A same-id 3D recount gives val 36. Every
   `below_noise_floor` flag the pipeline emits scales from that constant.
3. **Do the docs still quote retracted numbers?** The 2026-07-29 tiled-inference
   fix invalidated roughly 40 leaderboard rows. Retractions were recorded in
   CHANGELOG.md and left standing elsewhere.

Plus two arms authorised on 2026-07-30 that have never run, blocked on
`CNNRunner` supporting more than one auxiliary input channel.

## Scope

In scope: the scratch control arm; settling `SE_REF_N`; purging stale numbers;
widening the `CNNRunner` aux path; getting `src/pretrained_probe.py` into git;
multi-meteorite ingestion (padding to a common in-plane size, `voxel_um`, and a
test); the seed-variance run; the val-versus-test analysis.

Multi-meteorite ingestion was moved from out of scope to in scope on 2026-08-04,
mid-session, at Sky's direction, after he identified having one meteorite as the
binding constraint. The seed-variance run and the val-versus-test analysis were
added the same way.

Out of scope: the data-scaling curve (still incomplete, roughly 10 h, resumable);
any change to the frozen split.

## Requirements

R1. The scratch arm runs alone. One data-loading process at a time, and
`evaluate.load` is called with `want_test=False` wherever test is not needed.
Two concurrent jobs pushed this Mac into swap on 2026-08-02.

R2. Both probe arms are judged on TEST IoU with the threshold frozen on val.
This project has had eight val/test sign flips. Validation IoU is reported but
does not decide anything.

R3. The evidence threshold is pre-registered and is not revised after the
numbers arrive: a test gap under about 0.05 is ambiguous at n=1 seed, and 0.10
or more is hard to attribute to run-to-run scatter. Justified by
`src/data_scaling.py:16-20`, quoted exactly: *"Single-seed differences of
0.03-0.06 sit only ~4-8 SE apart before counting training-seed variance, which is
itself large. With n=1 per point there is no way to tell a real trend from
run-to-run scatter."*

[CORRECTED 2026-08-04] An earlier draft of this requirement paraphrased that as
"single-seed differences of 0.03 to 0.06 cannot be separated from scatter", which
is a straw man of the source. Note also that the measured seed SD of 0.0084
contradicts the source's "training-seed variance, which is itself large". The
0.05 band was NOT revised on the strength of that measurement, because it was
pre-registered and because the SD comes from n=3 in a different budget regime.

R4. `SE_REF_N` is not changed on the strength of a single count. Two independent
verifications must agree, and the unit question (the SE was bootstrapped over
validation slices, the constant is expressed in chondrules) must be settled
separately from the count question.

Corrected 2026-08-04 after two independent verifications. The blast radius is
six code files, not three, and the floor appears as four separate numeric
literals plus one literal count:

  * `src/pipeline_v2.py`'s `SE_REF_IOU` / `SE_REF_N` / `POWER_REFERENCE_N` block,
    the anchor itself, read three times inside `power_report` (the actual-n SE,
    the reference-row comprehension, and the returned `se_reference_*` fields)
  * `src/tta_confidence.py`'s `compare_single_vs_tta` hardcodes 0.0142 in the
    `below_noise_floor` flag it returns
  * `src/confidence_compare.py`'s module-level `MDE` hardcodes 0.0142, its
    `[REGIME]` print hardcodes the count 19, and its `json.dump` of `mde_iou`
    persists the value to `results/confidence_compare.json`
  * `src/run_error_cnn.py`'s reporting block hardcodes 0.0142 in an executable
    gate, the `abs(dv) < 0.0142` and `abs(dm) < 0.0142` comparisons
  * `src/make_figures.py` hardcodes the +/-0.0071 band, which is rendered into
    `results/figures/data_scaling.png` (it was at :125 when this was written and
    moved to :158 when `fig_budget` grew, so it is cited without a line number)

Two floors already coexist before any edit: `run_all` takes n from the measured
split metadata (its `power_report(zones["meta"]["n_chondrules_3d"]["val"])` call,
n=32 on the v2 split, giving 0.0109) while three modules print and store 0.0142.
[CORRECTED 2026-08-04, before commit] This said "Nothing in the repo imports
`pipeline_v2`, so the constant's reach inside code is confined to that module
plus the unlinked hardcodes." That was true when written, and THIS change set
falsified it: `src/recount_chondrules.py` imports `count_chondrules_3d` from it
and `tests/test_multi_meteorite.py` imports the module wholesale. Neither reads
`SE_REF_N`, so the blast-radius conclusion still stands, but the premise does
not. The only automated check on
the anchor is the `__main__` self-test, which fails at its
`p19["mde_iou"] - 0.0142` assertion if it moves. `tests/` was empty when this was
written; it now holds `test_aux_channels.py` and `test_multi_meteorite.py`,
neither of which covers the noise floor.

R4a. [CORRECTION] `SE_REF_IOU` and `SE_REF_N` are a MATCHED ANCHOR PAIR from one
bootstrap on one zone, not an independent SE and an independent count. Raising
the anchor count therefore LOOSENS every floor by sqrt(36/19) = 1.376x. The
claim in HANDOFF.md's retracted 0.0103 block that the MDE "tightens from 0.0142
to roughly 0.0103" is WITHDRAWN: 0.0103 is `power_report(36)` evaluated with the
anchor left at 19, which is the inconsistent state. A recount adds no data and
cannot buy power. HANDOFF.md has been rewritten since, and the block is marked in
place there rather than deleted: under "Noise floor: what is settled and what is
not" it now reads "is WITHDRAWN, sign backwards".

R5. No number, caveat, retraction, or status label is altered to improve prose.
Any documentation pass is verified mechanically to leave numeric tokens intact
except where a value is being explicitly retracted.

R6. Widening the aux path does not change behaviour for existing single-channel
callers. A one-channel aux must produce a bit-identical input tensor to what it
produces today.

## Acceptance criteria

A1. [MET] `results/pretrained_probe.json` and its scratch counterpart both exist,
each with test IoU, Dice, precision, recall and AUPRC, and the val-frozen
threshold. Both arms are rows of the one file, which is what `_prior_rows` was
added to guarantee.

A2. [MET] The pretrained-minus-scratch test gap is reported against the R3
threshold, with the verdict stated in the pre-registered language rather than
reinterpreted.

A3. [MET] `SE_REF_N` is either changed with both verifications recorded, or left
unchanged with the reason recorded in the spec and the CHANGELOG. Silence is not
an acceptable outcome. Left unchanged, with the reason recorded in R4a and in the
CHANGELOG.

A4. [MET] Every standing unmarked pre-tiler-fix number in README.md and
ROADMAP.md either carries a retraction marker or is removed. Both tables now
carry a per-row marker.

A5. [MET] `CNNRunner` accepts N aux channels, the single-channel path is
unchanged, and the checkpoint guard rejects a channel-count mismatch at load.

A6. [OUTSTANDING] `src/pretrained_probe.py` is tracked in git. Outstanding until
the pending commit lands.

A7. [MET] Two meteorites of different in-plane size ingest without crashing, the
padding is marked invalid, and a test verifies it
(`tests/test_multi_meteorite.py`).

A8. [MET] A measured seed SD exists for at least one data fraction: SD 0.0084 at
fraction 0.25, three seeds.

A9. [MET] The val-versus-test analysis is reproducible from a script in the repo
(`src/val_vs_test.py`), not just an output file.

## Pre-registered, before the numbers arrive

Recorded so it cannot be adjusted afterwards:

- pretrained minus scratch, on TEST, under 0.05: ambiguous, reported as such.
- 0.05 to 0.10: suggestive, not evidence.
- 0.10 or more: supports "data is the binding constraint".
- Either arm beating 0.29409 does NOT by itself say anything about pretraining,
  because that comparison also changes the architecture.

## Open questions

- [OPEN] Whether `SE_REF_IOU` itself must be re-derived if the scaling unit is
  wrong, rather than only re-pointing `SE_REF_N`. Not resolved: the resampling
  unit was slices while the constant is expressed in chondrules, so re-pointing
  `SE_REF_N` alone leaves a unit mismatch. Fixing it properly means re-expressing
  the anchor in slices, which rewrites every projection row.

- [RESOLVED 2026-08-04] What 19 actually is. `python -m src.recount_chondrules`
  counted the frozen split both ways over identical voxels, taking zone
  membership from `split.json` and the valid mask from the training cache:
  train 63/49, val **36/19**, test 43/24 (same-id / binary CC), totals 142/92.
  `SE_REF_N = 19` is exactly the binary connected-components count of the val
  zone, which is the undercount `count_chondrules_3d` exists to avoid. The
  handoff's 63/36/43 is confirmed; the 99/19/24 in CHANGELOG.md's 2026-07-29
  "Prepare for a second meteorite" entry, under "Free statistical power from slab
  placement", is not reproduced (its 19 and 24 are binary-CC values but 99
  matches neither rule).
  Output in `results/chondrule_recount.json`. This does NOT change the decision
  in R4a: 36 is not the right anchor either, because neither count was the
  resampling unit.

- [OPEN] The multiplier. `MDE = 2.0 * SE` in code; project prose cites 2.8 at
  three sites, only one of which derives it, as (1.96 + 0.84) for 80% power. The
  two answer different questions: 2.0 (about 1.96) tests whether an OBSERVED gain
  differs from zero, which is what `_gain` does, while 2.8 is an a-priori
  study-sizing constant. The code is defensible for its use. The naming is not: a
  field called `mde_iou` means the second thing while computing the first.

- [RESOLVED 2026-08-04] Whether the aux path takes one callable returning C
  channels or a list of C callables. It takes one callable returning (C,H,W); a
  callable that returns (H,W) is promoted.
