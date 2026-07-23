"""v2 broad model sweep — tiers A (non-learned) and B (classical ML).

One leaderboard, the identical frozen v1 split, ignore-mask honored, operating
threshold tuned on validation only and frozen for test. Tier C (CNNs) and
tier D (GNN) are added later on this same harness.

Run:  python -m src.sweep
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_local
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import data, split as splitmod, features, metrics

SEED = 0


def _zone_idx(slab, N, zone):
    return [i for i in range(N) if slab[i] == zone]


def load_split_zones(root="."):
    res, slab, ignore, idxs = splitmod.build_split(root)
    raw, lab = data.list_pairs(root)
    N = len(idxs)

    def load(zone):
        ii = _zone_idx(slab, N, zone)
        abs_i = [idxs[i] for i in ii]
        R = data.load_stack(raw, abs_i, "raw").astype(np.float32)
        M = data.load_stack(lab, abs_i, "mask")
        return R, M, ~ignore[ii]

    return res, load("train"), load("val"), load("test")


# ---------- tier B: classical-ML on texture features ----------

def sample_train_pixels(Rtr, Mtr, Vtr, per_slice_cap=8000, slice_stride=4, seed=SEED):
    """Balanced (pos≈neg) pixel sample with features, from train slabs only."""
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
        F = features.extract_features(Rtr[i])
        X.append(F[coords[:, 0], coords[:, 1], :])
        y.append(np.r_[np.ones(len(pidx)), np.zeros(len(nidx))])
    return np.vstack(X), np.concatenate(y)


def sklearn_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, C=1.0)),
        "gaussian_nb": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=4,
                                                n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=4,
                                            n_jobs=-1, random_state=SEED),
        "hist_gboost": HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
    }


def predict_zone_feat(models, Rz):
    """slice-outer / model-inner: extract features once per slice, predict all."""
    n = Rz.shape[0]
    out = {name: np.zeros((n,) + Rz.shape[1:], np.float32) for name in models}
    for i in range(n):
        F = features.extract_features(Rz[i])
        flat = F.reshape(-1, F.shape[-1])
        for name, est in models.items():
            out[name][i] = est.predict_proba(flat)[:, 1].reshape(Rz.shape[1:])
    return out


# ---------- tier A: non-learned intensity/threshold ----------

def local_adaptive_prob(Rz, block=51, offset=0.0):
    out = np.zeros_like(Rz, np.float32)
    for i in range(Rz.shape[0]):
        t = threshold_local(Rz[i], block_size=block, offset=offset)
        out[i] = (Rz[i] >= t).astype(np.float32)
    return out


def _score(prob_test, Mt, Vt, thr, t_train, t_infer):
    m = metrics.metrics_from_pred(prob_test >= thr, Mt, Vt)
    m["auprc"] = metrics.auprc(prob_test, Mt, Vt)
    m["ece"] = metrics.expected_calibration_error(prob_test, Mt, Vt)
    m["threshold"] = float(thr)
    m["train_sec"] = round(t_train, 1)
    m["infer_sec"] = round(t_infer, 1)
    return m


def run(root=".", out="results/v2_leaderboard.json"):
    t0 = time.time()
    assert metrics._selfcheck()
    res, (Rtr, Mtr, Vtr), (Rv, Mv, Vv), (Rt, Mt, Vt) = load_split_zones(root)
    print(f"zones: train {Rtr.shape[0]} / val {Rv.shape[0]} / test {Rt.shape[0]} slices")
    board = {}

    # ---- Tier A ----
    ipv, ipt = Rv / 255.0, Rt / 255.0  # fixed normalization -> thresholds transfer
    otsu_t = float(threshold_otsu(Rtr[Vtr])) / 255.0
    multiotsu_t = float(threshold_multiotsu(Rtr[Vtr])[-1]) / 255.0
    for name, fixed in [("intensity_thr_tuned", None),
                        ("otsu_global", otsu_t),
                        ("multiotsu_global", multiotsu_t)]:
        thr = fixed if fixed is not None else metrics.select_threshold(ipv, Mv, Vv)[0]
        board[name] = _score(ipt, Mt, Vt, thr, 0.0, 0.0)
        board[name]["tier"] = "A_threshold"
        print(f"  [A] {name:20s} thr={thr:.3f} IoU={board[name]['iou']:.4f}")
    t = time.time(); lav = local_adaptive_prob(Rv); lat = local_adaptive_prob(Rt)
    thr = metrics.select_threshold(lav, Mv, Vv)[0]
    board["local_adaptive"] = _score(lat, Mt, Vt, thr, 0.0, time.time() - t)
    board["local_adaptive"]["tier"] = "A_threshold"
    print(f"  [A] {'local_adaptive':20s} thr={thr:.3f} IoU={board['local_adaptive']['iou']:.4f}")

    # ---- Tier B ----
    t = time.time()
    Xtr, ytr = sample_train_pixels(Rtr, Mtr, Vtr)
    print(f"  tier-B train sample: {Xtr.shape[0]:,} px x {Xtr.shape[1]} features "
          f"(pos frac {ytr.mean():.2f})")
    models = sklearn_models()
    fit_t = {}
    for name, est in models.items():
        tt = time.time(); est.fit(Xtr, ytr); fit_t[name] = time.time() - tt
    pv = predict_zone_feat(models, Rv)          # val probs (for threshold pick)
    ti = time.time(); pt = predict_zone_feat(models, Rt); infer_all = time.time() - ti
    for name in models:
        thr = metrics.select_threshold(pv[name], Mv, Vv)[0]
        board[name] = _score(pt[name], Mt, Vt, thr, fit_t[name], infer_all / len(models))
        board[name]["tier"] = "B_classical_ml"
        print(f"  [B] {name:20s} thr={thr:.3f} IoU={board[name]['iou']:.4f} "
              f"Dice={board[name]['dice']:.3f} P={board[name]['precision']:.3f} "
              f"R={board[name]['recall']:.3f} AUPRC={board[name]['auprc']:.3f}")
    print(f"  tier-B feature-extract+predict: {time.time() - t:.0f}s")

    ranked = sorted(board.items(), key=lambda kv: (kv[1]["iou"] if kv[1]["iou"] == kv[1]["iou"] else -1),
                    reverse=True)
    out_obj = {
        "regime": "leakage-free contiguous z-slab split; threshold tuned on val, "
                  "frozen for test; foreground=chondrule; N=1 (ALH 83010); split & "
                  "metrics independently audited leakage-free (2026-07-23)",
        "n_features": len(features.feature_names()),
        "feature_names": features.feature_names(),
        "split_summary": {k: res[k] for k in ("band", "N", "buffer",
                          "n_straddler_instances", "ignored_positive_frac")},
        "leaderboard": [dict(model=k, **v) for k, v in ranked],
        "runtime_sec": round(time.time() - t0, 1),
    }
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print("\n=== LEADERBOARD (test foreground IoU) ===")
    for k, v in ranked:
        print(f"  {v['iou']:.4f}  {k:22s} [{v['tier']}]  Dice={v['dice']:.3f} "
              f"P={v['precision']:.3f} R={v['recall']:.3f}")
    print(f"\nwrote {out}  ({out_obj['runtime_sec']}s)")
    return out_obj


if __name__ == "__main__":
    run()
