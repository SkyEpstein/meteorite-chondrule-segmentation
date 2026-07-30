"""Shared train-and-score harness for every model family (stages 3-10).

One code path so all tiers land on the leaderboard under identical rules:
fit on TRAIN, select the operating threshold on VAL, score the TEST slab
exactly once, and append the row to a single unified leaderboard.

Any runner works as long as it exposes the CNNRunner-style API:
  .name, .tier, .fit(train, val), .predict_proba(raw) -> (n,H,W) in [0,1]
  and (optionally) .best_thr after fit.

Leakage rules enforced here [LEAKAGE][MUST]:
  * the runner's fit() only ever receives train + val;
  * the threshold comes from VAL (the runner's own, or re-selected on val);
  * test is loaded only for the single final scoring call.
"""
from __future__ import annotations
import json
import os
import time
import traceback
import numpy as np

from . import metrics, zones

LEADERBOARD = "results/leaderboard_all.json"


def load(root=".", want_test=True):
    """Cached zones as dicts. Loads test only when actually scoring."""
    names = ("train", "val", "test") if want_test else ("train", "val")
    z = zones.load_zones(root, zones=names)
    out = {}
    for k in names:
        R, M, V = z[k]
        out[k] = {"raw": R, "mask": M, "valid": V}
    return out


def score(prob, mask, valid, thr):
    m = metrics.metrics_from_pred(prob >= thr, mask, valid)
    m["auprc"] = metrics.auprc(prob, mask, valid)
    m["ece"] = metrics.expected_calibration_error(prob, mask, valid)
    m["threshold"] = float(thr)
    psi = metrics.per_slice_iou(prob >= thr, mask, valid)
    m["per_slice_iou_median"] = float(np.nanmedian(psi)) if np.isfinite(psi).any() else float("nan")
    return m


def recall_operating_point(prob, mask, valid, min_recall=0.90):
    """Highest-precision threshold that still achieves >= min_recall.

    Serves the don't-miss-chondrules goal alongside the max-IoU point. Selected
    on whatever split the caller passes, so pass VAL to keep test untouched.
    """
    best = None
    for t in np.arange(0.02, 0.99, 0.02):
        m = metrics.metrics_from_pred(prob >= t, mask, valid)
        if m["recall"] == m["recall"] and m["recall"] >= min_recall:
            if best is None or m["precision"] > best[1]["precision"]:
                best = (float(t), m)
    return best  # (thr, metrics) or None


def append_leaderboard(entry, path=LEADERBOARD):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = json.load(f).get("leaderboard", [])
    rows = [r for r in rows if r.get("model") != entry.get("model")]  # replace re-runs
    rows.append(entry)
    rows.sort(key=lambda r: r.get("iou", -1) if r.get("iou") == r.get("iou") else -1,
              reverse=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "regime": "leakage-free contiguous z-slab split; threshold selected on VAL, "
                      "frozen for a single TEST scoring; foreground=chondrule; "
                      "N=1 meteorite (ALH 83010); local Mac (MPS/CPU), compute-limited",
            "leaderboard": rows,
        }, f, indent=2)
    return rows


def run_and_score(runner, data=None, root=".", extra=None, save_prob=None):
    """Fit the runner, score test once, append to the leaderboard.

    Returns the leaderboard row (or an {"error": ...} row if the run failed,
    because a single family blowing up must not abort the sweep).
    """
    d = data or load(root)
    t0 = time.time()
    row = {"model": runner.name, "tier": getattr(runner, "tier", "unknown")}
    try:
        runner.fit({k: d["train"][k] for k in ("raw", "mask", "valid")},
                   {k: d["val"][k] for k in ("raw", "mask", "valid")})
        thr = float(getattr(runner, "best_thr", 0.5))

        prob_test = runner.predict_proba(d["test"]["raw"])
        row.update(score(prob_test, d["test"]["mask"], d["test"]["valid"], thr))

        # Recall point is chosen on VAL, then reported on test. [LEAKAGE]
        prob_val = runner.predict_proba(d["val"]["raw"])
        rp = recall_operating_point(prob_val, d["val"]["mask"], d["val"]["valid"])
        if rp:
            rthr, _ = rp
            rm = metrics.metrics_from_pred(prob_test >= rthr,
                                           d["test"]["mask"], d["test"]["valid"])
            row["recall_point"] = {"threshold": rthr, "recall": rm["recall"],
                                   "precision": rm["precision"], "iou": rm["iou"]}
        if save_prob:
            # Save BOTH zones' probabilities: the meta-models (confidence, error,
            # prune, z-refine, combos) must be FIT on validation predictions and
            # only applied to test, so val probs are as necessary as test probs.
            os.makedirs(os.path.dirname(save_prob), exist_ok=True)
            np.save(save_prob, prob_test.astype(np.float16))
            np.save(save_prob.replace(".npy", "_val.npy"),
                    prob_val.astype(np.float16))
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-800:]
        row["iou"] = float("nan")
    row["sec"] = round(time.time() - t0, 1)
    if extra:
        row.update(extra)
    append_leaderboard(row)
    return row
