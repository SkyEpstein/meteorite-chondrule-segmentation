# Meteorite Chondrule Segmentation

Chondrules are the rocky, silicate inclusions that make up much of a chondrite
meteorite. This project finds them automatically in micro-CT scans by labeling
every pixel of a scan slice as either chondrule or matrix, so that parts of a
scan nobody has annotated by hand can still be mapped.

The first and so far only sample is Allan Hills 83010 (ALH 83010), a 531-slice
micro-CT volume with a hand-labeled band through the middle. Candidate models are
ranked by foreground IoU on a spatially held-out slab of that volume, using a
split built to be leakage-free.

## Read this before quoting any score

**Every neural-network score in this repository was measured with a confirmed
inference bug and is unreliable in an unknown direction.** On 2026-07-29 a
tiled-inference defect was found in `CNNRunner.predict_proba`: it accumulated its
32-pixel halo instead of cropping it. Because stride equalled tile size, the
overlap fell in 64-pixel bands where the divisor jumped from 1 to 2 to 4,
averaging in each window's zero-padded border. That left visible axis-aligned
seams: 5 to 7 of the 8 largest per-column jumps landed exactly on a window
boundary, against 0 to 1 of 8 for GNN models, which bypass the tiler.

Cropping the halo was the obvious fix, and it was **not sufficient**. That is the
part worth understanding:

- **Cropping removed the seams.** With cores tiling exactly once the
  boundary-localised artifact is gone, measured at 1 of 8 on a boundary
  afterwards, with the boundary column jump only 1.1x the median.
- **But these networks normalise with GroupNorm**, which computes its statistics
  over whatever spatial extent it is handed. Under tiling every window is
  normalised by its own statistics, so a window's predictions shift as a block.
  The residual error is therefore diffuse rather than seam-shaped, and **a
  constant-image test cannot detect it at all**, because GroupNorm is degenerate
  on a constant image. That is precisely why the first round of verification
  passed. Measured with the fixed tiler on a non-constant image (untrained
  base=8 depth-4 UNet2D, 512x1024): shifting the tile grid by 128 pixels over
  identical content moves predictions by mean |dp| 0.024, max 0.286, and tiled
  against a single whole-image pass gives mean |dp| 0.022, max 0.331.
- **Inference is now whole-slice.** Each slice is replicate-padded to a multiple
  of `PAD_MULT` and pushed through the network in one forward pass, so the output
  cannot depend on a tile grid. A 962x1450 slice takes about 0.8 s, roughly a
  minute for the 80-slice test slab, so tiling bought nothing here. The
  halo-cropping tiler survives only as `predict_proba_tiled`, for reproducing old
  numbers, and must not be used for new ones.
- **The direction is not established.** The defect could have helped or hurt. No
  claim that scores will rise or fall after retraining is supported.
- **Scores measured between the halo-crop fix and the whole-slice change were
  tile-grid dependent**, so they are not a trustworthy middle ground either.
- **A retrain of the headline model is in flight** (`unet2d_s30000`, about 2
  hours) and its result is not yet in.
- **Affected:** every CNN, transformer and modern-conv score, including the
  0.3168 headline and the 0.3214 pipeline figure.
- **Not affected:** the GNN scores, the classical-ML result and the intensity
  floor, which never go through the tiler. The GNN figure above was measured at
  bug-discovery time as a control, not re-measured afterwards.

Full detail, along with several other open issues, is in the KNOWN ISSUES block
at the top of [CHANGELOG.md](CHANGELOG.md). Read it before citing any figure from
any file here.

## What it measured

These are the numbers as recorded. Every neural-network row is a superseded
measurement pending the retrain described above, and is kept for continuity
rather than as a current result.

| Model | Test IoU | Tiler bug | Note |
|---|---|---|---|
| `unet2d_s30000` | 0.3168 | affected | plain U-Net, 30k steps, best recorded |
| `unet2d_s15000` | 0.3089 | affected | same model, 15k steps |
| `long_p1_ctrl_b32` | 0.2834 | affected | same model, 7.2k steps |
| `hrnet_lite` | 0.2598 | affected | best of 40 architectures at a fixed 3k budget |
| `swin_lite` | 0.2552 | affected | from-scratch transformer |
| `unet25d_k5` | 0.2547 | affected | 2.5D (z-context). The "+10% over 2D at equal budget" reading of this row is a WITHDRAWN CLAIM: see CHANGELOG |
| `hist_gradient_boosting` | 0.203 | not affected | classical ML on texture features |
| best GNN (`graphsage`) | 0.178 | not affected | superpixel graph |
| intensity threshold | 0.130 | not affected | non-learned floor |

Stacking FP-pruning and z-smoothing on the winner recorded 0.3214, but that
configuration was selected on test rather than validation, so it is a reported
number and not a fairly selected one.

Training budget mattered more than architecture. The same U-Net at 3k, 7.2k, 15k
and 30k steps recorded 0.2314, 0.2834, 0.3089 and 0.3159. Gains roughly halved
with each doubling and then flattened. No architecture tested at a fixed budget
came close to simply training one model longer. All four points share the same
inference defect, so the shape of the curve is more trustworthy than its height,
but two of the four cannot be checked at all: the 3k anchor's named artifact is
byte-identical to the 30k one, and no artifact exists for the 15k point. Treat the
trend as indicative and the absolute values as pending the retrain.

For scale, hand annotation is not perfectly repeatable, so IoU near 1.0 is not a
reachable target against hand-drawn labels. Measured ceiling
(`results/annotation_ceiling.json`): 0.92 at a ±1px boundary band, 0.85 at ±2px,
0.83 against an adjacent slice. The tolerance is a two-sided band; the one-sided
figures differ materially (+1px alone is 0.961).

## How to read these numbers

Beyond the tiler bug, three further caveats apply.

**The headline does not reproduce from this code even ignoring the tiler.** The
0.3168 figure also predates a fix to the threshold grid in `metrics.py`. Under
the committed code `unet2d_s30000` selects a different threshold and scores
0.31601, and `results/leaderboard_all.json` records 0.31587 at threshold 0.006.
Three different numbers have been called the headline; the retrain delta will be
quoted against the reproducible value near 0.3159.

**The winner was picked from a 40-model sweep, so its score is optimistic.** Its
test IoU is the maximum of 40 noisy estimates, which inflates it by roughly 0.02
to 0.03. Thresholds were always fit on validation and the model was never tuned
on test, so this is selection bias rather than threshold leakage. An unbiased
figure would need a holdout that played no part in selection.

**Most measured differences are smaller than the noise floor.** With 19
validation chondrules, the bootstrap standard error is about 0.0071 IoU, giving a
minimum detectable effect near 0.014 absolute IoU. That is larger than nearly
every method effect measured in this project, and six validation-versus-test
comparisons flipped sign. Treat small gains as unresolved rather than real. One
meteorite and about 142 chondrules support no cross-meteorite claim at all.

## Why the evaluation is trustworthy

The caveats above concern measurement and selection. The split and the metrics
were built to remove the more common failure modes:

- **The split is leakage-free.** Train, validation and test are contiguous
  z-slabs, never randomly drawn slices. A buffer gap separates the slabs so
  adjacent, near-duplicate slices cannot leak across them, and every chondrule
  spanning a seam is assigned to exactly one slab. An independent recount
  confirmed 0 chondrules appear in 2 or more slabs. Both the split and the
  metrics were separately audited and confirmed leakage-free at the label level.
  One measured caveat remains: a resolved chondrule's raw pixels are still
  visible, though unlabeled, in the neighbour slab at least 48 original slices
  away. The ablation meant to quantify that has only one arm, so its size is
  still unknown.
- **Each model touches the test slab once.** Thresholds and hyperparameters are
  tuned on validation only, then the model is scored on test a single time. Over
  40 models were scored this way, which is why the winner carries the
  selection-bias caveat above.
- **Metrics are recorded together.** Roughly 96% of pixels are background, so IoU
  alone is easy to misread. Every run writes Dice, precision, recall and AUPRC
  alongside foreground IoU into the JSON under `results/`, with the evaluation
  regime stated. The tables on this page quote IoU only, for brevity; consult the
  JSON before drawing a conclusion from a single number.

## Preparing for more data

The binding constraint has been measured to be data rather than method. Because
the minimum detectable effect is larger than most method effects, adding another
meteorite is worth more than another architecture sweep.

`src/pipeline_v2.py` makes adding a meteorite a configuration change rather than
a rewrite: a `MeteoriteSource` feeds `build_multi_split`, then `stage2_sweep` and
`confidence_sweep`. It reports the minimum detectable effect for whatever
validation size results, and flags any gain below that as `below_noise_floor`
instead of reporting it as a finding. Model weights now persist through
`save_checkpoint` and `load_checkpoint`, with the operating threshold travelling
alongside the weights, so a trained model can be reapplied without retraining.

## Running it

```bash
pip install -r requirements.txt          # plus a torch build from pytorch.org
python run_v1.py                         # split, classical baseline, figure
python -m src.sweep                      # Tier A+B leaderboard
python -m src.ceiling                    # annotation ceiling
```

Two limits on reproducing this from a clone. The scan data
(`ALH83010_z_crop_every4[_labeled]/`) is not tracked in git. And
`results/probs/`, about 13 GB of per-model probability maps, is gitignored along
with `results/graph_cache/`, so the post-hoc pipeline in `src/run_final.py`
cannot run from a fresh clone without retraining first. The committed JSON
results and figures record every reported number, but they cannot be
independently re-derived from this repository alone.

## What is and is not finished

| Stage | State |
|---|---|
| Leakage-free split plus 2 adversarial audits | done, 0 chondrules span slabs |
| Evaluation harness and classical baseline | done, 0.130 floor and 0.203 classical ML |
| Annotation ceiling | done, ~0.92 at 1px and ~0.85 at 2px |
| Architecture sweep (36 configs: CNNs, 4 from-scratch transformers, 6 modern conv, 7 GNNs) | done, 41 leaderboard rows, all NN rows now pending retrain |
| Budget-scaling curve (3k to 30k steps) | done, flattened at 0.3159, pending retrain |
| Hierarchical superpixel cascade | done, 2.71x compute saving at 99.9% retention |
| GNN-then-CNN cascade | done, 0.2025, below a plain CNN |
| Confidence bake-off (10 learned candidates, 4 free baselines) | done, nothing beat a free baseline by more than +0.0005 |
| FP-prune and z-refinement | done, they help weak models but lose on the shipped model |
| Two-model is/is-not veto | done, precision 0.463 against 0.412, IoU got worse |
| Recall-oriented training bake-off | done, asymmetric losses hurt |
| Tiled-inference fix | done, fixed and verified, **no model retrained yet** |
| Model checkpointing (`save_checkpoint` / `load_checkpoint`) | done, verified bit-identical reload, threshold travels with the weights |
| Multi-meteorite pipeline (`src/pipeline_v2.py`) | done, self-test passes, 13 straddlers resolved and 0 chondrules span 2 or more zones |
| Straddler-drop ablation | **incomplete, one arm only** |
| Corrected data-scaling test (`src/data_scaling.py`) | **blocked, does not run**: crashes before training on a bad `leaderboard` kwarg, see KNOWN ISSUES |
| Headline retrain under whole-slice inference | **in flight, result not yet in** |
| Complete-model contract (`specs/2026-07-29-complete-model-contract/`) | **specified, not implemented** |
| Apply to the ~130 unlabeled slices (v4) | **not done** |
| Regression tests | **not done, `tests/` is empty** |

Every score in that table computed from a neural-network probability map carries
the tiler caveat above. That includes the cascade, the confidence bake-off, the
FP-prune and z-refinement results and the veto, not just the leaderboard rows.

The superseded scaling result in `results/data_scaling.json` is confounded, by
contiguous rather than random subsets and a single seed per fraction, and must
not be quoted.

## Layout

`memory/constitution.md` holds the project principles and `specs/` the specs.
`src/` holds the code (data, split, metrics, features, baseline, sweep,
models_cnn, ceiling, pipeline_v2), `results/` the JSON outputs, and
[ROADMAP.md](ROADMAP.md) the per-stage detail. Decisions and measured numbers are
logged in [CHANGELOG.md](CHANGELOG.md).
