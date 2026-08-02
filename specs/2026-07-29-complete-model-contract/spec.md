# Spec — The "complete model" contract

**Date:** 2026-07-29 (stage order superseded 2026-07-30) · **Owner:** Sky Epstein
**Status:** specified, NOT implemented; stage list below is SUPERSEDED by constitution §12
**Decision source:** Sky, 2026-07-29 — "define the *complete model* from this point
forward as the best model for the hierarchy + the hierarchy + confidence model +
the z-layer neighbor confidence."

> **Revision note (same day).** The first draft of this spec claimed the
> z-neighbor stage was "not wired into `run_all`". **That was false** — it is
> step 4 of `confidence_sweep`. An adversarial verification pass corrected it and
> found the real gaps, which are different and larger. Everything below reflects
> the verified state, with `file:line` evidence.

## Problem

All four components exist **and all four are already invoked** by
`pipeline_v2.run_all`:

| stage | where | verified state |
|---|---|---|
| 0 base segmenter | `stage2_sweep` — `pipeline_v2.py:678` | wired |
| 1 hierarchy filter | `hierarchy.run_cascade` — `pipeline_v2.py:647,650,652` | wired; **leakage-free** (val-fit, frozen, decision rule uses raw only — `hierarchy.py:483-493`) |
| 2 confidence model | `cb.run_bakeoff` — `pipeline_v2.py:506` | wired |
| 3 z-neighbor | `combine.fit_z_refine` / `z_smooth` — `pipeline_v2.py:573-574` | wired; `z_agreement` additionally live as a bake-off free baseline (`confidence_bakeoff.py:120`) and as feature 2 of `ConfidenceModel.build_features` (`combine.py:250`) |

So the problem is **not** a missing stage. It is that there is no *composite*:

- **G1 — `run_all` has never been executed.** No caller exists anywhere in the
  repo; `__main__` (`pipeline_v2.py:732-771`) is plumbing-only by its own comment,
  and `results_v2/` holds only `split_multi.json` — no `pipeline_v2.json`.
- **G2 — the three post-base treatments are applied *independently* to the same
  base map**, never chained: bake-off winner (`:520`), prune (`:555`) and z-refine
  (`:574`) each read `pt`. No variable carries a pruned map into the z block.
  A partial chain exists only in `run_final.py:113-124` (`prune_plus_z`) — it
  skips the hierarchy and hard-codes a `>= 0.5` cut instead of val-selecting it,
  so it is not a valid template.
- **G3 — Stage 0 and Stage 1 are fused and cannot be ablated apart.**
  `stage2_sweep` only ever builds a `HierMaskedRunner` (`pipeline_v2.py:454`), so
  every base candidate is already hierarchy-masked and `run_all` contains no
  hierarchy-off arm.
- **G4 [FIXED] the "high-recall operating point" did not happen.** `THR_GRID`
  now reaches 1e-7. See the section below.

## The contract

Four stages, composed in order. The contract fixes the **slots and their
interfaces**, not the occupants.

```
Stage 0  BASE        best architecture by leakage-free VAL bake-off,
                     trained at the locked budget, at a HIGH-RECALL
                     operating point                   [constitution §9]
Stage 1  HIERARCHY   superpixel cascade as a filter; thresholds fit on VAL,
                     frozen, then applied to train/test
Stage 2  CONFIDENCE  learned trust/error meta-model, chosen by recorded
                     bake-off against the free baselines  [constitution §v3+b]
Stage 3  Z-NEIGHBOR  cross-slice neighbour-agreement confidence / refinement
```

### Requirements

- **R1** `run_all` produces **one chained end-to-end score** — base → hierarchy →
  confidence-prune → z-neighbour — not four independent deltas against the base
  map. *(closes G2)*
- **R2** Every stage is **independently ablatable**, which requires a
  hierarchy-off arm in `stage2_sweep`. *(closes G3)*
- **R3** The headline number is **end-to-end test foreground IoU** against the
  plain-U-Net-at-matched-budget baseline. Per-stage deltas are secondary and
  always reported beside the MDE.
- **R4** Every reported gain carries the `below_noise_floor` flag from
  `power_report`. A gain smaller than the MDE is reported as *indistinguishable*,
  never as an improvement.
- **R5** No stage may hard-code ALH 83010; the entry point stays
  `run_all(sources)` over `MeteoriteSource`.
- **R6** Leakage rules inherited unchanged: meta-models and thresholds fit on VAL
  only, test touched once, straddlers resolved. [constitution §1, §2]
- **R7** `run_all` must actually be **runnable and run** — a real entry point that
  writes `pipeline_v2.json`. *(closes G1)*

### Acceptance criteria

- **A1** `python -m src.pipeline_v2` (or a documented entry point) executes all
  four stages end-to-end and writes one JSON with the chained score.
- **A2** That JSON contains the four ablation arms, each with its sign and its
  `below_noise_floor` flag.
- **A3** Adding a second `MeteoriteSource` needs no code change.
- **A4** Spec, constitution and ROADMAP name the same stages in the same order.
  **[FAILING as of 2026-08-01]** The constitution was reordered by Sky on
  2026-07-30 to SEVEN stages, hierarchy first, pruning last (constitution §12).
  This spec and ROADMAP.md still describe the superseded FOUR-stage, base-first
  order below. The authoritative order is the constitution's; the stage list in
  this spec has not been rewritten to match, and this criterion is therefore not
  met by the document you are reading.
- **A5** The recall-target defect below is either fixed or explicitly specified
  around, and the contract's wording matches whichever was chosen.

## [FIXED 2026-07-30] Blocking bug: the over-predict-then-prune step never happened

**This section describes a defect that has since been FIXED.** `THR_GRID` was
extended to 1e-7 and the silent fallback now warns and sets
`recall_target_unreachable` on the returned metrics. Recall 0.99 is reached on
the control (achieved 0.9910); it is still NOT reached on the augmented arm
(achieved 0.9355, flagged), which is a comparability defect in any bake-off
across those two arms. The measured cost of actually over-predicting: val IoU
falls to 0.0666 at recall 0.99, so the prune must climb back past 0.294 to beat
plain thresholding. It did not: every learned model pruned nothing and the free
`margin` baseline won, which is re-thresholding in disguise. The description
below is kept for the record of what was wrong and how it was found.

### Original report

`combine.recall_threshold` cannot reach **any** configured recall target for the
dice-trained models this pipeline selects. `THR_GRID` floors at 2e-4
(`combine.py:34-38`) and the function **silently** falls back to `THR_GRID[0]`
when the target is unreachable (`combine.py:551-553`). Measured on the real
validation zone:

```
unet2d_s30000     targets 0.80 / 0.90 / 0.95 / 0.98  ->  thr 2e-4, val recall 0.6014  (all four identical)
hier_cnn_s30000   targets 0.80 / 0.90 / 0.95 / 0.98  ->  thr 2e-4, val recall 0.5531  (all four identical)
```

This is a **grid-floor defect, not model saturation**: recall at `prob > 1e-7` is
0.985 (unet2d) and 0.970 (hier_cnn); only 0.86% / 0.66% of ground-truth positive
pixels are exactly zero.

**[REGIME] These numbers were computed on the saved *pre-tiler-fix* validation
probability maps** (`results/probs/*_val.npy`, written before 2026-07-29). The
structural conclusion is robust — the grid floor sits orders of magnitude above
the thresholds these models actually need — but the specific 0.985 / 0.970
headroom figures will move once the models are re-scored under whole-slice
inference. Re-measure before quoting them.

Consequences:

- `PRUNE_RECALL_TARGETS` yields four **identical** rows — visible in
  `results/final_pipeline.json` as 0.3011→0.3131 repeated four times;
- `BAKEOFF_TARGET_RECALL = 0.95` actually opens the model to **0.60** recall;
- `best_prune_by_val` breaks ties arbitrarily;
- **constitution §9 ("over-predict, then prune") is not being executed**, and
  Stage 0's "high-recall operating point" is currently false.

This was partly known — the CHANGELOG KNOWN ISSUES block already records "the
confidence bake-off ran at recall 0.601, not the stated 0.95". The new part is the
cause (grid floor ~3 orders of magnitude too high) and that the same defect class
was previously "fixed" by moving the floor 0.05 → 0.0002, which was not far enough
(`combine.py:28-33`).

**Any end-to-end number computed before this is settled measures the wrong
operating point.**

## Secondary defects found (not blocking, must not be silently inherited)

- **Unmatched threshold grids across arms.** Base grid min 0.002
  (`pipeline_v2.py:598-599`); `fit_z_refine` → `metrics.select_threshold` grid min
  1e-6 (`metrics.py:88-91`); prune grid min 2e-4. `gains["z_refine_test"]` (`:701`)
  therefore credits z-smoothing with part of a finer threshold search the base arm
  never got.
- **Headline gains conflate treatment with threshold change.**
  `gains["bakeoff_winner_test"]` (`:697-698`) and `gains["fp_prune_R*_test"]`
  (`:704-705`) subtract `base_iou` measured at `thr` while the treatments run at
  `thr_rec`. The honest delta (`test_gain_vs_unpruned`, `:564`) is computed but is
  not what the noise-floor block prints (`:721-724`).
- **Multiplicity.** One `run_all` scores the test slab ~12 times, each flagged
  against the same *single-comparison* MDE. `below_noise_floor=False` will fire by
  chance more often than the flag implies.
- **`_gain` mislabels NaN as measurable** — `abs(nan) < mde` is `False`
  (`:618-621`).
- **`out_dir` half-honored** — JSON to `out_dir` (`:718`), probs to module-level
  `OUT_DIR` (`:464, :684-687`).
- **`alive_index` has no collision guard** (`:392-397`) — the one place a train
  alive mask could reach a test slice.

## Honesty constraints [MUST]

This contract is an **architecture decision, not a performance claim.** At the
current data volume the composition cannot be validated stage-by-stage:

- MDE at n=19 val chondrules = **0.0142 IoU**; the v2 split raises n to 32,
  MDE **0.0109**.
- `hier_cnn` at a matched 30k budget scored **0.3037 vs 0.3159** — **−0.012,
  inside the noise floor.** The hierarchy is justified as a **compute**
  optimization (2.41× at target recall 0.999), not an accuracy one.
- z-continuity headroom is real and measured — truth 85.2% slice-to-slice IoU vs
  predictions 65.9% — but headroom is not a realized gain.

**All CNN-family numbers above were measured under the tiled-inference defect
fixed on 2026-07-29 and are being re-measured.** They are used here only to size
the noise floor, which the bug does not change.

## Why now (larger-dataset readiness)

Fixing the contract before the new scans land means they are evaluated against a
stable target. The MDE improves directly with validation chondrule count:

| val chondrules | MDE (IoU) | basis |
|---|---|---|
| 19 (frozen split) | 0.0142 | **measured** — bootstrap SE ±0.0071 |
| 32 (v2 split) | 0.0109 | measured count, extrapolated SE |
| 40 | 0.0098 | **projection** (1/√n) |
| 80 | 0.0069 | **projection** (1/√n) |
| 150 | 0.0051 | **projection** (1/√n) |

Only near the bottom of that table do the per-stage effects measured so far become
resolvable at all — and the last three rows are `2·SE·√(19/n)` extrapolations, not
observations. Two caveats on the formula itself: the `2·SE` factor is weaker than
the conventional `(1.96+0.84)·SE ≈ 2.8·SE` for α=0.05 at 80% power, so the true
floor is likely **higher** than tabulated; and the reference SE was bootstrapped
over validation *slices* while the scaling here is by *chondrule* count, so the
exponent may be attached to the wrong unit. Both need settling before the MDE is
used as a publication gate.

## Decisions (Sky, 2026-07-29). D1 has SHIPPED; the rest have not

- **D1 Recall floor, extend `THR_GRID` to ~1e-7, log-spaced. [SHIPPED]** The headroom is
  there (recall at `prob > 1e-7` is 0.985), so the configured 0.80–0.98 targets
  become reachable and constitution §9 becomes real. The silent fallback in
  `combine.recall_threshold:551-553` must also **raise or return a flag** instead
  of quietly returning the grid floor. **Invalidates** the prune and bake-off
  numbers in `results/final_pipeline.json`, which must be regenerated after.
- **D2 Chain order → z-refine the probability map first, then prune.** Every stage
  operates on probabilities, so nothing is destroyed by an early binarization and
  the final threshold stays val-selected. This deliberately does **not** reuse
  `run_final.py`'s `prune_plus_z` path, whose 0.3214 was selected on test and
  whose `>= 0.5` cut is hard-coded.
- **D3 Scope for 2026-07-29 → spec + commit only.** The composite (R1, R2, R7) and
  the D1 fix are **not** implemented in this session: the base model is being
  re-measured under the fixed tiler, and building the contract on a score that is
  about to change would have to be redone. Implementation is the next session's
  first task.

## Remaining open question

- **Q3 Threshold protocol for the ablations.** Re-select the threshold per arm on
  one shared grid, or let each arm keep its own? At the current MDE this changes
  the *sign* of small deltas. Must be settled before the first composite run —
  the unmatched-grid defect listed above is the same issue in a different guise.
