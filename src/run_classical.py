"""Score Sky's classical pipeline as a Stage-0 segmenter, head to head.

    denoise -> CLAHE -> morphological texture separation -> watershed

Built as a strict ADDITION to the existing tier-B classical-ML path so the
comparison is like-for-like: identical pixel sampling (`sweep.sample_train_pixels`
semantics), identical estimator (`HistGradientBoostingClassifier(max_iter=200)`),
identical threshold-on-val discipline. The ONLY difference between arms is which
features go in and whether watershed is applied.

Arms
----
  base18      the existing 18 texture features                  -> reproduces ~0.203
  plus3       + denoised, CLAHE, top-hat                        -> Sky's preprocessing
  plus3_ws    + watershed instance separation on the probability

[MUST] CLAHE is ADDED, never substituted — raw intensity stays as feature 0. It
is the strongest single signal here (chondrules ~50 vs matrix ~32), and removing
it is what made intensity-jitter augmentation halve val IoU (0.387 -> 0.202).

[MUST] The output stays a PROBABILITY. Watershed only splits merged blobs; it
never replaces the score, because stages 1-3 of the complete-model contract
(hierarchy filter, learned error model, z-neighbour confidence) all consume
graded probabilities. The CNN's map is ~1% graded, which is why every confidence
model there prunes nothing — a boosted probability should be far better spread,
and `graded_frac` is reported here so that is measured rather than hoped.

[LEAKAGE][MUST] Fit on TRAIN. Select every threshold and watershed parameter on
VAL. Score TEST exactly once per arm.

Run:  python -m src.run_classical [--quick]
"""
from __future__ import annotations
import json
import os
import sys
import time

import joblib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import classical_pipeline as cp
from . import evaluate, features, metrics

SEED = 0
OUT = "results/classical_pipeline.json"


def extract_plus3(sl):
    """The 18 existing features + denoised + CLAHE + top-hat -> (H,W,21)."""
    F = features.extract_features(sl)               # (H,W,18), feature 0 = intensity
    ch = cp.channels(sl)                            # (H,W,4) raw/denoised/clahe/tophat
    return np.concatenate([F, ch[..., 1:]], axis=-1)  # drop ch[0]: raw == F[...,0]


def _sample(Rtr, Mtr, Vtr, extract, per_slice_cap=8000, slice_stride=4, seed=SEED):
    """Balanced pos/neg pixel sample from TRAIN only (sweep.py semantics)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(0, Rtr.shape[0], slice_stride):
        valid = Vtr[i]
        pos = np.argwhere(Mtr[i] & valid)
        if len(pos) == 0:
            continue
        neg = np.argwhere((~Mtr[i]) & valid)
        npos = min(len(pos), per_slice_cap)
        nneg = min(len(neg), npos)
        pidx = pos[rng.choice(len(pos), npos, replace=False)]
        nidx = neg[rng.choice(len(neg), nneg, replace=False)]
        coords = np.vstack([pidx, nidx])
        F = extract(Rtr[i])
        X.append(F[coords[:, 0], coords[:, 1], :])
        y.append(np.r_[np.ones(len(pidx)), np.zeros(len(nidx))])
    return np.vstack(X), np.concatenate(y)


def _prob_stack(R, est, extract):
    out = np.zeros(R.shape, np.float32)
    for i in range(R.shape[0]):
        F = extract(R[i])
        out[i] = est.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(R.shape[1:])
    return out


def _report(name, pv, Mv, Vv, pt, Mt, Vt, watershed=False, ws_params=None):
    """Threshold on VAL, score TEST once. Metrics together per §5."""
    thr, _ = metrics.select_threshold(pv, Mv.astype(bool), Vv)
    pred_t = pt >= thr
    ws_info = None
    if watershed:
        md, ms = (ws_params or (8, 64))
        split = np.zeros(pt.shape, bool)
        n_lab = 0
        for i in range(pt.shape[0]):
            lab, m = cp.watershed_split(pt[i], thr, min_distance=md, min_size=ms)
            split[i] = m
            n_lab += int(lab.max())
        ws_info = {"min_distance": md, "min_size": ms, "n_labels_test": n_lab}
        pred_t = split
    m = metrics.metrics_from_pred(pred_t, Mt, Vt)
    m = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
    m["auprc"] = float(metrics.auprc(pt, Mt.astype(bool), Vt))
    m["threshold"] = float(thr)
    # gradation is what stages 1-3 need; the CNN's map has ~1%
    m["graded_frac_val"] = float(((pv > 1e-4) & (pv < 0.9999)).mean())
    m["arm"] = name
    if ws_info:
        m["watershed"] = ws_info
    return m


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    stride = 12 if quick else 4

    d = evaluate.load(".")
    Rtr, Mtr, Vtr = (d["train"][k] for k in ("raw", "mask", "valid"))
    Rv, Mv, Vv = (d["val"][k] for k in ("raw", "mask", "valid"))
    Rt, Mt, Vt = (d["test"][k] for k in ("raw", "mask", "valid"))
    if quick:                       # smoke path: a few slices, not the full slabs
        Rv, Mv, Vv = Rv[:4], Mv[:4], Vv[:4]
        Rt, Mt, Vt = Rt[:4], Mt[:4], Vt[:4]

    rows = []
    for name, extract in (("base18", features.extract_features),
                          ("plus3", extract_plus3)):
        t0 = time.time()
        print(f"\n=== arm {name}: sampling train (stride {stride}) ===", flush=True)
        X, y = _sample(Rtr, Mtr, Vtr, extract, slice_stride=stride)
        print(f"  X={X.shape} pos_frac={y.mean():.3f}  fitting...", flush=True)
        est = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        est.fit(X, y)
        pv = _prob_stack(Rv, est, extract)
        pt = _prob_stack(Rt, est, extract)
        # [MUST] Persist the maps. Without them this module can only report
        # scores — the classical probability could not be shown in a figure, and
        # more importantly could not be fed to the CNN as a cascade input
        # channel, which is what the pipeline ordering needs it for. Same
        # float16 convention and naming as evaluate.run_and_score.
        if not quick:
            os.makedirs("results/probs", exist_ok=True)
            np.save(f"results/probs/classical_{name}.npy", pt.astype(np.float16))
            np.save(f"results/probs/classical_{name}_val.npy", pv.astype(np.float16))
            joblib.dump(est, f"results/classical_{name}.joblib")
            print(f"  saved probs + estimator for {name}", flush=True)
        r = _report(name, pv, Mv, Vv, pt, Mt, Vt)
        r["fit_sec"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  {name}: test IoU={r['iou']:.4f} Dice={r['dice']:.4f} "
              f"P={r['precision']:.4f} R={r['recall']:.4f} "
              f"graded={100*r['graded_frac_val']:.2f}%", flush=True)
        if name == "plus3":         # watershed only on the richer arm
            rw = _report("plus3_ws", pv, Mv, Vv, pt, Mt, Vt, watershed=True)
            rw["fit_sec"] = 0.0
            rows.append(rw)
            print(f"  plus3_ws: test IoU={rw['iou']:.4f} "
                  f"({rw['watershed']['n_labels_test']} watershed labels)", flush=True)
        json.dump({"regime": ("classical Stage-0; fit on TRAIN, threshold selected "
                              "on VAL, TEST scored once per arm; N=1 meteorite "
                              "(ALH 83010)"),
                   "quick": quick, "rows": rows}, open(OUT, "w"), indent=2)

    print(f"\n=== CLASSICAL PIPELINE (test) — reference: CNN 0.29409, "
          f"classical-ML 0.203, threshold 0.130 ===")
    print(f"{'arm':<12}{'IoU':>9}{'Dice':>9}{'prec':>9}{'recall':>9}{'AUPRC':>9}{'graded%':>9}")
    for r in rows:
        print(f"{r['arm']:<12}{r['iou']:9.4f}{r['dice']:9.4f}{r['precision']:9.4f}"
              f"{r['recall']:9.4f}{r['auprc']:9.4f}{100*r['graded_frac_val']:8.2f}%")
    print(f"\nwrote {OUT}")
    return rows


if __name__ == "__main__":
    main()
