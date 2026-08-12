"""Audit: how much does the literature's split protocol inflate a chondrule score?

Motivation (Sky, 2026-08-05). Published work on meteorite / rock micro-CT
segmentation frequently evaluates on a RANDOM split of slices (or patches) drawn
from a single scanned specimen. Adjacent CT slices are near-duplicates, so a
random split puts near-copies of the test data in training. This project's
constitution rejects that (principle 1: contiguous z-slabs, never random slices,
with a buffer gap), which is why its headline number is low.

The question this module answers is not "is their code wrong" but: **on OUR data,
holding everything else fixed, how many IoU points does the split protocol alone
buy?** That number is the size of the overclaim a random-split paper carries, and
it is measured here rather than asserted.

Design. Exactly one thing varies between arms: which slices land in
train/val/test. Every arm shares the same band, the same slice COUNTS per zone,
the same balanced pixel sampler (`sweep.sample_train_pixels` semantics, seed 0),
the same estimator (`HistGradientBoostingClassifier(max_iter=200)`), the same
threshold-selected-on-val discipline, and the same single scoring of test.

Arms
----
  leakage_free    contiguous z-slabs, buffer 12, straddlers resolved. The
                  project's protocol, and the honest reference.
  no_buffer       contiguous z-slabs, buffer 0, straddlers resolved. Isolates
                  what the buffer alone is worth.
  random_slice    slices assigned to zones uniformly at random, no buffer, no
                  straddler handling. The common literature protocol.

`random_slice` is run over several seeds because its result depends on the draw;
the others are deterministic.

[LEAKAGE] Only the arms are leaky, deliberately. Within every arm the discipline
holds: fit on train, choose the threshold on val, score test exactly once.

[REGIME] Classical texture-feature model, not the CNN. It reproduces the
published 0.203 baseline, costs minutes instead of hours, and the question is
about the SPLIT, which is model-independent. A CNN would very likely show a
LARGER gap, since it can memorise slice-specific texture that a 19-feature
gradient booster cannot. So the number here is a floor on the inflation, not a
point estimate for deep models.

Run:  python -m src.split_protocol_audit [--quick] [--seeds 3]
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import data, features, metrics
from . import split as splitmod

SEED = 0
OUT = "results/split_protocol_audit.json"
# Use metrics.select_threshold's own grid (1e-6 to 0.95). A narrower grid is the
# truncated-grid defect this repo has logged three times; not repeating it.
THR_GRID = None


# ---------------------------------------------------------------------------
# Zone construction. Every arm returns (slab, ignore) over the same band.
# ---------------------------------------------------------------------------
def zones_contiguous(vol_ids, buffer):
    """Project protocol: contiguous slabs, straddlers resolved to one slab.

    Mirrors `split.build_split` but takes an already-loaded label volume and a
    buffer, so the audit does not re-read the volume once per arm.
    """
    N = vol_ids.shape[0]
    slab = splitmod.assign_slabs(N, splitmod.DEFAULT_VAL_ZONE,
                                 splitmod.DEFAULT_TEST_ZONE, buffer)
    inst, ninst = splitmod.instance_label(vol_ids)
    spans = splitmod._component_zone_span(inst, slab)
    straddlers = [cid for cid, z in spans.items() if len(z) >= 2]
    zonecode = np.array([{"train": 1, "val": 2, "test": 3}.get(str(slab[i]), 0)
                         for i in range(N)])
    counts = np.zeros((ninst + 1, 4))
    for z in range(N):
        counts[:, zonecode[z]] += np.bincount(inst[z].ravel(), minlength=ninst + 1)
    maj = np.zeros(ninst + 1, dtype=np.int64)
    for cid in straddlers:
        maj[cid] = int(np.argmax(counts[cid, 1:4])) + 1
    ignore = np.zeros(vol_ids.shape, dtype=bool)
    for z in range(N):
        zc = zonecode[z]
        if zc == 0:
            continue
        m = maj[inst[z]]
        ignore[z] = (m != 0) & (m != zc)
    return slab, ignore, len(straddlers)


def zones_random(N, n_val, n_test, seed):
    """Literature protocol: slices assigned to zones uniformly at random.

    No buffer and no straddler handling, so a chondrule's own neighbouring
    slices can sit in train while the chondrule is scored in test. `ignore` is
    all-False because a naive protocol has nothing to ignore.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    slab = np.array(["train"] * N, dtype=object)
    slab[perm[:n_val]] = "val"
    slab[perm[n_val:n_val + n_test]] = "test"
    return slab, None


# ---------------------------------------------------------------------------
def sample_train_pixels(R, M, V, idx, per_slice_cap=8000, slice_stride=4, seed=SEED):
    """Balanced pos/neg pixel sample over the given train slice indices."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in idx[::slice_stride]:
        valid = V[i]
        pos = np.argwhere(M[i] & valid)
        if len(pos) == 0:
            continue
        neg = np.argwhere((~M[i]) & valid)
        npos = min(len(pos), per_slice_cap)
        nneg = min(len(neg), npos)
        pidx = pos[rng.choice(len(pos), npos, replace=False)]
        nidx = neg[rng.choice(len(neg), nneg, replace=False)]
        coords = np.vstack([pidx, nidx])
        F = features.extract_features(R[i])
        X.append(F[coords[:, 0], coords[:, 1], :])
        y.append(np.r_[np.ones(len(pidx)), np.zeros(len(nidx))])
    return np.vstack(X), np.concatenate(y)


def predict_slices(est, R, idx):
    """Probability maps for the given slice indices."""
    out = np.zeros((len(idx),) + R.shape[1:], np.float32)
    for k, i in enumerate(idx):
        F = features.extract_features(R[i])
        flat = F.reshape(-1, F.shape[-1])
        out[k] = est.predict_proba(flat)[:, 1].reshape(R.shape[1:])
    return out


def run_arm(name, slab, ignore, R, M, eval_stride, train_stride, seed=SEED,
            extra=None):
    """Fit on train, pick the threshold on val, score test once."""
    t0 = time.time()
    N = R.shape[0]
    V = np.ones(R.shape, dtype=bool) if ignore is None else ~ignore
    idx = {z: [i for i in range(N) if slab[i] == z] for z in ("train", "val", "test")}

    X, y = sample_train_pixels(R, M, V, idx["train"], slice_stride=train_stride,
                               seed=seed)
    est = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(X, y)
    t_train = time.time() - t0

    vi = idx["val"][::eval_stride]
    ti = idx["test"][::eval_stride]
    t1 = time.time()
    pv = predict_slices(est, R, vi)
    thr, val_iou = metrics.select_threshold(pv, M[vi], V[vi], grid=THR_GRID)
    pt = predict_slices(est, R, ti)
    m = metrics.metrics_from_pred(pt >= thr, M[ti], V[ti])
    m["auprc"] = metrics.auprc(pt, M[ti], V[ti])
    m.update({
        "arm": name, "threshold": float(thr), "val_iou": float(val_iou),
        "n_train_slices": len(idx["train"]), "n_val_slices": len(idx["val"]),
        "n_test_slices": len(idx["test"]),
        "n_val_scored": len(vi), "n_test_scored": len(ti),
        "n_train_pixels": int(len(y)), "seed": int(seed),
        "train_sec": round(t_train, 1), "infer_sec": round(time.time() - t1, 1),
    })
    if extra:
        m.update(extra)
    print(f"  {name:<22} seed={seed}  val IoU {val_iou:.4f}   TEST IoU {m['iou']:.4f}"
          f"   (thr {thr:.3f}, {m['train_sec'] + m['infer_sec']:.0f}s)", flush=True)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarser strides, for a smoke run")
    ap.add_argument("--seeds", type=int, default=3, help="random_slice repeats")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    train_stride = 12 if args.quick else 4
    eval_stride = 12 if args.quick else 4

    raw, lab = data.list_pairs(args.root)
    fg = data.fg_fraction(lab)
    b0, b1 = data.annotated_band(fg)
    idxs = list(range(b0, b1 + 1))
    print(f"band {b0}..{b1}  ({len(idxs)} slices)   train_stride={train_stride} "
          f"eval_stride={eval_stride}", flush=True)
    R = data.load_stack(raw, idxs, "raw")
    vol_ids = data.load_stack(lab, idxs, "raw")
    M = vol_ids > 0
    N = R.shape[0]

    results = []
    # --- honest reference, and the buffer ablation ---
    for nm, buf in (("leakage_free", splitmod.DEFAULT_BUFFER), ("no_buffer", 0)):
        slab, ignore, nstr = zones_contiguous(vol_ids, buf)
        results.append(run_arm(nm, slab, ignore, R, M, eval_stride, train_stride,
                               extra={"buffer": buf, "n_straddlers_resolved": nstr,
                                      "protocol": "contiguous z-slabs"}))
    # Match the honest arm's zone SIZES exactly, so the only difference between
    # protocols is the spatial arrangement, never the amount of training data.
    n_val, n_test = results[0]["n_val_slices"], results[0]["n_test_slices"]

    # --- the literature protocol, several draws ---
    for s in range(args.seeds):
        slab_r, ig_r = zones_random(N, n_val, n_test, seed=1000 + s)
        results.append(run_arm("random_slice", slab_r, ig_r, R, M, eval_stride,
                               train_stride, seed=1000 + s,
                               extra={"buffer": 0, "n_straddlers_resolved": 0,
                                      "protocol": "random slice assignment"}))

    ref = results[0]["iou"]
    rnd = [r["iou"] for r in results if r["arm"] == "random_slice"]
    summary = {
        "leakage_free_test_iou": ref,
        "no_buffer_test_iou": results[1]["iou"],
        "random_slice_test_iou_mean": float(np.mean(rnd)),
        "random_slice_test_iou_sd": float(np.std(rnd, ddof=1)) if len(rnd) > 1 else None,
        "random_slice_test_iou_all": rnd,
        "inflation_random_minus_leakagefree": float(np.mean(rnd) - ref),
        "inflation_relative": float((np.mean(rnd) - ref) / ref) if ref else None,
        "buffer_contribution": float(results[1]["iou"] - ref),
        "regime": ("classical texture features + HistGradientBoosting(max_iter=200); "
                   "identical sampling, estimator and threshold discipline across arms; "
                   "ONLY the slice-to-zone assignment differs"),
        "caveat": ("a floor, not a point estimate: a CNN can memorise slice-specific "
                   "texture that a 19-feature booster cannot, so its inflation is "
                   "expected to be larger"),
    }
    os.makedirs(os.path.dirname(os.path.join(args.root, OUT)), exist_ok=True)
    with open(os.path.join(args.root, OUT), "w") as f:
        json.dump({"summary": summary, "arms": results}, f, indent=2)
    print("\n" + "=" * 74)
    print(f"leakage-free (project protocol) test IoU : {ref:.4f}")
    print(f"no buffer                        test IoU : {results[1]['iou']:.4f}")
    print(f"random slice split (literature)  test IoU : {np.mean(rnd):.4f}"
          + (f" +/- {np.std(rnd, ddof=1):.4f}" if len(rnd) > 1 else ""))
    print(f"INFLATION from the split alone           : "
          f"{np.mean(rnd) - ref:+.4f} IoU "
          f"({100 * (np.mean(rnd) - ref) / ref:+.0f}%)")
    print("=" * 74)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
