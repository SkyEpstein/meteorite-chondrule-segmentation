"""Cross-fitted classical probability maps, for use as a CNN cascade channel.

The CNN takes the classical map as an extra input channel (Sky's ordering:
hierarchy -> classical -> CNN -> confidence -> z -> prune). That channel must be
**out-of-fold on TRAIN**, and this module is what makes it so.

[LEAKAGE][MUST] Why cross-fitting is not optional
--------------------------------------------------
A model's predictions on its own training data are memorised. Feeding those to a
refiner teaches it to trust a channel that is far weaker at test time, and the
refiner then collapses when the channel degrades. The constitution imposes this
on the GNN->CNN cascade already (Tier E), and `src/cascade.py` establishes the
pattern this module follows:

    for each of 5 CONTIGUOUS folds:
        fit on the other ~80% (minus a buffer)  -> predict the held-out fold
    fit on ALL of train                         -> predict VAL and TEST

Folds are CONTIGUOUS in z with a buffer, never random: adjacent CT slices are
near-duplicates, so a random split would leave each fold's "out-of-fold"
predictions effectively in-fold — the same reason the project's main split uses
contiguous slabs with a 12-slice buffer.

[WHY 5 FOLDS, MEASURED] A first pass used 2 folds. Each fold-model then saw only
~50% of the data, and the resulting channel was far WEAKER than what the CNN
would meet at inference — out-of-fold train IoU 0.1507 vs 0.2875 for the full-fit
model used on val/test (both at a fixed 0.5 threshold). That is the mirror image
of the memorisation trap: instead of over-trusting a channel that degrades, the
refiner would learn to distrust a channel that IMPROVES, wasting the signal. It
would also have made a null cascade result uninterpretable. Five folds put each
fit at ~80% of the data, and additionally cover all 213 train slices instead of
201 — with 2 folds the 12 buffer slices had no out-of-fold prediction at all.

Writes `results/probs/classical_xf_{train,val,test}.npy`.

Run:  python -m src.classical_crossfit
"""
from __future__ import annotations
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import evaluate
from .run_classical import SEED, _prob_stack, _sample, extract_plus3

BUFFER = 12          # matches the main split's near-duplicate buffer
N_FOLDS = 5
OUTDIR = "results/probs"


def contiguous_halves(n, buffer=BUFFER):
    """Two contiguous z-halves with a gap between them. [kept for reference]"""
    half = (n - buffer) // 2
    return np.arange(0, half), np.arange(half + buffer, n)


def contiguous_folds(n, k=N_FOLDS, buffer=BUFFER):
    """`k` contiguous z-folds. Returns [(held_out_idx, fit_idx), ...].

    [WHY 5 AND NOT 2] With 2 folds each fold-model trains on only ~50% of the
    data, and measured out-of-fold train IoU was 0.1507 against 0.2875 for the
    full-fit model used on val/test. The CNN would then learn to distrust a
    channel that is markedly BETTER at inference than in training — the mirror
    image of the memorisation trap, wasting signal rather than inventing it.
    With 5 folds each model sees ~80%, so out-of-fold quality lands much closer
    to what the CNN actually meets.

    The fit set EXCLUDES a `buffer` of slices on either side of the held-out
    block, because adjacent CT slices are near-duplicates — without that gap the
    "out-of-fold" predictions are effectively in-fold at the block edges.
    """
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        held = np.arange(lo, hi)
        fit = np.concatenate([np.arange(0, max(0, lo - buffer)),
                              np.arange(min(n, hi + buffer), n)])
        out.append((held, fit))
    return out


def _fit(R, M, V, idx, stride=4):
    X, y = _sample(R[idx], M[idx], V[idx], extract_plus3, slice_stride=stride)
    est = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
    est.fit(X, y)
    return est, X.shape


def run(root=".", stride=4):
    d = evaluate.load(root)
    Rtr, Mtr, Vtr = (d["train"][k] for k in ("raw", "mask", "valid"))
    n = Rtr.shape[0]
    folds = contiguous_folds(n)
    print(f"train {n} slices -> {len(folds)} contiguous folds, buffer {BUFFER}",
          flush=True)

    t0 = time.time()
    train_aux = np.full(Rtr.shape, np.nan, np.float32)

    for j, (held, fit) in enumerate(folds):
        est, sh = _fit(Rtr, Mtr, Vtr, fit, stride)
        print(f"  fold {j + 1}/{len(folds)}: fit on {len(fit)} slices {sh} "
              f"-> predicting held-out {held[0]}-{held[-1]} ({len(held)})",
              flush=True)
        train_aux[held] = _prob_stack(Rtr[held], est, extract_plus3)

    est_full, sh = _fit(Rtr, Mtr, Vtr, np.arange(n), stride)
    print(f"  fit on ALL train {sh} -> predicting val/test", flush=True)
    val_aux = _prob_stack(d["val"]["raw"], est_full, extract_plus3)
    test_aux = _prob_stack(d["test"]["raw"], est_full, extract_plus3)

    covered = np.isfinite(train_aux).all(axis=(1, 2))
    assert covered.all(), f"{int((~covered).sum())} train slices left unpredicted"
    os.makedirs(OUTDIR, exist_ok=True)
    np.save(f"{OUTDIR}/classical_xf_train.npy", train_aux.astype(np.float16))
    np.save(f"{OUTDIR}/classical_xf_val.npy", val_aux.astype(np.float16))
    np.save(f"{OUTDIR}/classical_xf_test.npy", test_aux.astype(np.float16))
    print(f"\ncovered {int(covered.sum())}/{n} train slices "
          f"({n - int(covered.sum())} in the buffer, left NaN)")
    print(f"wrote {OUTDIR}/classical_xf_{{train,val,test}}.npy "
          f"in {time.time() - t0:.0f}s")
    return {"n_train": n, "covered": int(covered.sum()),
            "buffer": BUFFER, "sec": round(time.time() - t0, 1)}


if __name__ == "__main__":
    run()
