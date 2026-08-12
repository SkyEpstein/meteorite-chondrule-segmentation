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
import datetime
import json
import os
import subprocess
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
    on whatever split the caller passes — pass VAL to keep test untouched.
    """
    best = None
    for t in np.arange(0.02, 0.99, 0.02):
        m = metrics.metrics_from_pred(prob >= t, mask, valid)
        if m["recall"] == m["recall"] and m["recall"] >= min_recall:
            if best is None or m["precision"] > best[1]["precision"]:
                best = (float(t), m)
    return best  # (thr, metrics) or None


# Which tiers run through the tiled predictor (`CNNRunner.predict_proba`), and
# which never touch it. An unrecognised tier must NOT be assumed to have tiled,
# so it stamps "unknown" rather than defaulting to "post".
_TIER_REGIME = {
    "C_cnn": "post",            # tiles: CNNRunner.predict_proba
    "E_cascade": "post",        # tiles: its refiner is a CNNRunner
    "D_gnn": "n/a",             # paints superpixel probabilities directly
    "A_threshold": "n/a",       # whole-slice thresholding, no net
    "B_classical_ml": "n/a",    # per-pixel features, no net
}


def _scoring_stamp(tier):
    """Provenance for one scored row: which code produced it, and when.

    None of the rows that predate this carried any dating field, so their
    inference regime had to be reconstructed from a backup snapshot and log
    mtimes. Stamping at scoring time means that never has to be done again.
    `tiler_regime` records whether the row went through the tiled predictor,
    per `_TIER_REGIME`; the halo-crop fix is in, so a tier that does tile and is
    scored now is "post".

    The commit is resolved against THIS file's repo, not the process cwd. A run
    launched from another directory would otherwise stamp a different repo's
    commit, and a confidently wrong provenance is worse than none. Same reason
    for the dirty-tree check: `rev-parse HEAD` names the last COMMIT, so a run
    launched from a modified tree would be stamped with code that is not the
    code that ran. A hash that silently omits uncommitted changes is worse than
    no hash, so a dirty tree gets a "-dirty" suffix. If `rev-parse` succeeded but
    `status` failed, the hash is known and the tree state is not, so the stamp
    keeps the hash with an "-unknown" suffix; the commit is dropped entirely only
    when `rev-parse` itself failed.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except Exception:
        commit = None
    try:
        st = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=5)
        if st.returncode != 0:
            # `subprocess.run` does not raise on a non-zero exit, so without this
            # a failed status call leaves `dirty` empty and the row gets a BARE
            # hash, which reads as "clean". That is the exact outcome this check
            # exists to prevent, so force the unknown path.
            raise RuntimeError(f"git status exited {st.returncode}")
        if commit and st.stdout.strip():
            commit = f"{commit}-dirty"
    except Exception:
        if commit:
            commit = f"{commit}-unknown"
    return {"scored_commit": commit,
            "scored_utc": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
            "tiler_regime": _TIER_REGIME.get(tier, "unknown")}


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


def run_and_score(runner, data=None, root=".", extra=None, save_prob=None,
                  leaderboard=True):
    """Fit the runner, score test once, append to the leaderboard.

    Returns the leaderboard row (or an {"error": ...} row if the run failed —
    a single family blowing up must not abort the sweep).

    `leaderboard=False` scores the run but keeps it OUT of
    `results/leaderboard_all.json`. [MUST] Pass it whenever the run is not
    comparable to the frozen-split roster — a different split, a data subset, or
    a reduced budget. The leaderboard's own regime string promises one split and
    `fig_leaderboard` plots every row in it as one ranked chart, so a
    mixed-regime row there is read as a comparable model and misquoted.
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
            # [MUST] float16 IS DELIBERATE (~13 GB of maps as it is), and it is
            # LOSSY IN A WAY THAT MATTERS. Every metric in `row` above was
            # computed on the float32 array; what lands on disk has only ~15,000
            # representable values in [0,1]. Threshold-derived metrics survive
            # the cast (IoU, precision, recall, tp/fp/fn reproduce, and 10-bin
            # ECE reproduces to five decimals) but RANKING metrics do not.
            #
            # Measured 2026-08-08 by regenerating from a checkpoint:
            # resunet18_scratch_s30000 AUPRC is 0.44301 in float32, exactly its
            # leaderboard value, and 0.33455 once cast, exactly what the saved
            # map gives. The loss is not uniform across models (-0.033 to
            # -0.118), so saved maps cannot even RANK by AUPRC.
            #
            # So: quote AUPRC (and any future ranking metric) from
            # `results/leaderboard_all.json`. Never recompute it from
            # `results/probs/`. Sky chose to keep float16 and carry this rule
            # rather than double the disk. See CHANGELOG.md KNOWN ISSUES.
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
    row.update(_scoring_stamp(row.get("tier")))   # authoritative: after `extra`
    if leaderboard:
        append_leaderboard(row)
    return row
