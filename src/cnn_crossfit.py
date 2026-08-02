"""Out-of-fold CNN predictions on TRAIN, so the error model can use all of it.

The problem this solves
-----------------------
The confidence/error model may only be fit where the base model's mistakes are
HONEST. The base CNN is fit on all of train, so it memorises train and is right
almost everywhere there — an error distribution that exists nowhere else. That
forces the error model onto the validation slab, and splitting val again (a U-Net
would otherwise memorise the very errors it is scored on) leaves roughly

    28 slices, ~9 chondrules

to fit an image model on. That is not enough to distinguish "image confidence
models do not work here" from "we gave it nine objects".

Cross-fitting removes the premise: train 5 CNNs, each on ~80% of the train slab,
and predict the held-out fold with a model that never saw it. The result is
genuine, non-memorised base predictions across all 213 train slices.

[COUNT, UNVERIFIED] An earlier draft of this docstring claimed "~99 chondrules,
about 10x more". That number could not be reproduced: 3D connected components on
the train mask give 52 (6-connectivity) or 47 (26-connectivity), and split.json
records 142 instances across the whole 397-slice annotated band, of which train's
proportional share is ~76. The honest statement is that cross-fitting raises the
error model's training set from 28 slices to 213, and the chondrule multiplier is
somewhere around 5x, not 10x. Recount before quoting it.

[LEAKAGE][MUST] Folds are CONTIGUOUS in z with a buffer, reusing
`classical_crossfit.contiguous_folds`. Adjacent CT slices are near-duplicates, so
a random fold split would make each fold's "out-of-fold" predictions effectively
in-fold. VAL and TEST predictions are NOT produced here — they already exist from
the full-fit model (`results/probs/{model}{,_val}.npy`) and are correct, because
the full-fit model never saw val or test.

[COST] 5 training runs at the full 30,000-step budget, ~75 min each: about 6
hours. Deliberately the same budget as the shipped model — a shorter budget would
make the out-of-fold channel systematically weaker than the one the error model
meets at inference, which is the mismatch measured on the classical cascade
channel (out-of-fold 0.1591 vs full-fit 0.2875, though that turned out to be zone
prevalence rather than fitting; the CNN memorises far more than gradient
boosting, so here the guard is load-bearing).

Run:  python -m src.cnn_crossfit [--folds=5] [--epochs=60] [--steps=500]
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

from . import evaluate, metrics
from .classical_crossfit import BUFFER, contiguous_folds
from .models_cnn import CNNRunner, UNet2D, dice_loss
from .run_families import best_recipe

OUTDIR = "results/probs"
OUT = "results/cnn_crossfit.json"


def run(root=".", folds=5, epochs=60, steps=500, seed=0):
    d = evaluate.load(root, want_test=False)
    Rtr, Mtr, Vtr = (d["train"][k] for k in ("raw", "mask", "valid"))
    n = Rtr.shape[0]
    rec = best_recipe()
    base = int(rec.get("base", 32))
    print(f"train {n} slices -> {folds} contiguous folds, buffer {BUFFER}\n"
          f"recipe {rec.get('_source')}; budget {epochs}x{steps}="
          f"{epochs * steps} steps per fold", flush=True)

    train_aux = np.full(Rtr.shape, np.nan, np.float32)
    rows, t0 = [], time.time()
    for j, (held, fit) in enumerate(contiguous_folds(n, folds, BUFFER)):
        ts = time.time()
        print(f"\n=== fold {j + 1}/{folds}: fit on {len(fit)} slices, "
              f"held out {held[0]}-{held[-1]} ({len(held)}) ===", flush=True)
        runner = CNNRunner(
            f"xf{j}", lambda ic: UNet2D(in_ch=ic, base=base, depth=4), dice_loss,
            epochs=epochs, steps_per_epoch=steps,
            lr=float(rec.get("lr", 3e-4)),
            weight_decay=float(rec.get("wd", 0.0)),
            pos_frac=float(rec.get("pos_frac", 0.5)),
            augment=False, val_stride=6, seed=seed,
            ckpt_dir="results/checkpoints")     # [F4] 6 GPU-hours must persist
        # [F1][LEAKAGE][MUST] Epoch selection uses the REAL val slab, never the
        # held-out fold. An earlier version passed the held-out fold here, which
        # would have let `fit` pick `best_state` by IoU on `val[::val_stride]` —
        # every 6th slice of the very fold this model then predicts. With
        # val_stride=6 that puts EVERY held-out slice within 3 of a selection
        # slice, well inside the project's 12-slice near-duplicate buffer, so the
        # contamination would cover the whole fold. The resulting channel would
        # look better than the base model achieves on genuinely unseen data, and
        # an error model fit on it would learn that the base is right more often
        # than it is — precisely the memorisation this module exists to prevent.
        runner.fit({"raw": Rtr[fit], "mask": Mtr[fit], "valid": Vtr[fit]},
                   {k: d["val"][k] for k in ("raw", "mask", "valid")})
        train_aux[held] = runner.predict_proba(Rtr[held])
        # The number this module exists to justify: how strong is the OUT-OF-FOLD
        # channel, versus the full-fit channel the error model meets at
        # inference? Pure evaluation of an already-frozen prediction, so it
        # leaks nothing. Without it, cnn_crossfit.json cannot answer the question
        # its own docstring poses, and `best_full_val_iou` is NOT a substitute:
        # since the F1 fix that is the fold model's score on the shared VAL slab,
        # which it was epoch-selected on, not its score on the held-out block.
        oof = metrics.metrics_from_pred(
            train_aux[held] >= float(getattr(runner, "best_thr", 0.5)),
            Mtr[held], Vtr[held])
        rows.append({"fold": j, "n_fit": int(len(fit)), "n_held": int(len(held)),
                     "held_range": [int(held[0]), int(held[-1])],
                     "val_iou_selected_on": float(getattr(
                         runner, "best_full_val_iou", float("nan"))),
                     "oof_iou": float(oof["iou"]),
                     "oof_dice": float(oof["dice"]),
                     "oof_precision": float(oof["precision"]),
                     "oof_recall": float(oof["recall"]),
                     "sec": round(time.time() - ts, 1)})
        print(f"  fold {j + 1} done in {rows[-1]['sec']:.0f}s: "
              f"OUT-OF-FOLD IoU {oof['iou']:.4f} "
              f"(val IoU it was selected on: "
              f"{rows[-1]['val_iou_selected_on']:.4f})", flush=True)
        # [F3] Write a PARTIAL path, never the final one. A NaN-containing
        # volume at the final path looks complete to every downstream reader:
        # NaN in a PatchSet aux channel poisons every conv output, and
        # `combine.fit_prune` on a NaN score map silently returns gain 0.0 —
        # indistinguishable from "the method found nothing".
        os.makedirs(OUTDIR, exist_ok=True)
        np.save(f"{OUTDIR}/cnn_xf_train.partial.npy", train_aux.astype(np.float16))
        json.dump({"folds": folds, "epochs": epochs, "steps": steps,
                   "recipe": rec.get("_source"), "rows": rows},
                  open(OUT, "w"), indent=2)

    covered = np.isfinite(train_aux).all(axis=(1, 2))
    assert covered.all(), f"{int((~covered).sum())} train slices left unpredicted"
    # [F3] Only now, with every fold present and finite, does the final path
    # appear. A reader that finds `cnn_xf_train.npy` can trust it is complete.
    os.replace(f"{OUTDIR}/cnn_xf_train.partial.npy",
               f"{OUTDIR}/cnn_xf_train.npy")
    print(f"\ncovered {int(covered.sum())}/{n} train slices in "
          f"{(time.time() - t0) / 60:.0f} min -> {OUTDIR}/cnn_xf_train.npy")
    print("[NOTE] val/test predictions are NOT written here: the full-fit model's "
          "existing maps are already honest for those zones.")
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    kw = {}
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k in ("folds", "epochs", "steps"):
            kw[k] = int(v)
        else:
            raise SystemExit(f"unknown option {a}")
    return run(**kw)


if __name__ == "__main__":
    main()
