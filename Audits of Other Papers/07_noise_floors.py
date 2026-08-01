"""Compute and persist the noise floors the retraction is judged against.

Spec A5 requires every number in the report to be traceable to a JSON under
results/. The floors quoted in A3a and in Finding 5 originally came from a
verification pass and existed in no artifact, which is the same defect the audit
criticises in the corpus. This script regenerates them and writes
results/noise_floors.json.

Everything here uses a fixed intensity rule, never a trained model, so it is
cheap and it isolates data properties from model behaviour:

  window_sweep     score every contiguous 80-slice test window with ONE fixed
                   threshold. The spread is how much the choice of test region
                   alone moves a reported score, and it is what makes the
                   leakage-free arm's sd 0.0988 in the replication.
  split_draw       score many random slice splits through the same otsu path
                   used by the audit's control arm. The spread is how much the
                   luck of a random draw alone moves a reported score.
  otsu_constancy   the fitted cut and selected operating point per protocol.
                   Measured: the operating point IS constant, the cut is NOT
                   (36.357 to 37.354). Earlier drafts claimed both; this is the
                   check that corrected them.
  tile_handicaps   the reflect-padding fraction and inference window counts
                   behind confound C5.

Run:  python "Audits of Other Papers/07_noise_floors.py"
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)

from src import metrics                                          # noqa: E402
import protocols as P                                            # noqa: E402
from importlib import import_module                              # noqa: E402
runmod = import_module("02_run_protocols")

OUT = os.path.join(ROOT, "results", "noise_floors.json")
WINDOW = 80          # test-zone length used everywhere in the audit
AUDIT_T0 = 160       # the audit's own test window start, band-relative
N_SPLIT_SEEDS = 40


def window_sweep(raw, mask):
    """IoU of ONE fixed intensity rule on every contiguous 80-slice window."""
    from skimage.filters import threshold_otsu
    t = float(threshold_otsu(raw.astype(np.float32)))
    pred = raw >= t
    N = raw.shape[0]
    rows = []
    for t0 in range(0, N - WINDOW + 1):
        sl = slice(t0, t0 + WINDOW)
        rows.append({"t0": t0, "iou": float(metrics.iou(pred[sl], mask[sl]))})
    all_iou = np.array([r["iou"] for r in rows])
    # "Interior" excludes windows overlapping the sparse band ends, matching the
    # region the audit's zone protocols actually place their test slabs in.
    interior = [r for r in rows if 40 <= r["t0"] <= N - WINDOW - 40]
    int_iou = np.array([r["iou"] for r in interior])
    audit = next(r["iou"] for r in rows if r["t0"] == AUDIT_T0)
    pct = float((int_iou < audit).mean() * 100.0)
    return {
        "fixed_threshold": t,
        "n_windows": len(rows),
        "all": {"mean": float(all_iou.mean()), "sd": float(all_iou.std(ddof=1)),
                "min": float(all_iou.min()), "max": float(all_iou.max())},
        "interior": {"n": len(interior), "mean": float(int_iou.mean()),
                     "sd": float(int_iou.std(ddof=1)),
                     "min": float(int_iou.min()), "max": float(int_iou.max())},
        "audit_window_t0": AUDIT_T0,
        "audit_window_iou": float(audit),
        "audit_window_percentile_of_interior": pct,
        "per_window": rows,
    }


def split_draw(raw, mask, n_seeds=N_SPLIT_SEEDS):
    """Spread of the control arm's score across random slice-split draws."""
    N = raw.shape[0]
    valid = np.ones(raw.shape, bool)
    out = []
    for s in range(n_seeds):
        sel = P.slice_random(N, seed=s)
        z = P.materialize_slices(raw, mask, valid, sel)
        pv = runmod.otsu_probs(z["train"]["raw"], z["val"]["raw"])
        thr, _ = metrics.select_threshold(pv, z["val"]["mask"], z["val"]["valid"])
        pt = runmod.otsu_probs(z["train"]["raw"], z["test"]["raw"])
        out.append({"seed": s,
                    "iou": float(metrics.iou(pt >= thr, z["test"]["mask"],
                                             z["test"]["valid"]))})
        del z
    v = np.array([o["iou"] for o in out])
    return {"n_seeds": n_seeds, "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "min": float(v.min()),
            "max": float(v.max()), "per_seed": out}


def otsu_constancy(raw, mask):
    """Fitted cut and selected operating point for each protocol."""
    from skimage.filters import threshold_otsu
    N = raw.shape[0]
    valid = np.ones(raw.shape, bool)
    rows = {}
    for name, cfg in P.PROTOCOLS.items():
        sel = cfg["fn"](N, seed=0)
        if cfg["kind"] == "slice":
            z = P.materialize_slices(raw, mask, valid, sel)
        else:
            z = P.materialize_tiles(raw, mask, valid, sel)
        cut = float(threshold_otsu(z["train"]["raw"].astype(np.float32)))
        pv = runmod.otsu_probs(z["train"]["raw"], z["val"]["raw"])
        thr, _ = metrics.select_threshold(pv, z["val"]["mask"], z["val"]["valid"])
        rows[name] = {"otsu_cut_on_train": cut, "operating_point_on_val": float(thr)}
        del z
    cuts = [r["otsu_cut_on_train"] for r in rows.values()]
    ops = [r["operating_point_on_val"] for r in rows.values()]
    return {"per_protocol": rows,
            "cut_range": [float(min(cuts)), float(max(cuts))],
            "operating_point_range": [float(min(ops)), float(max(ops))],
            "cut_is_constant": bool(max(cuts) - min(cuts) < 1e-6),
            "operating_point_is_constant": bool(max(ops) - min(ops) < 1e-9)}


def tile_handicaps(patch=128, tile=P.TILE):
    """Reflect-pad fraction and inference-window counts behind confound C5.

    PatchSet reflect-pads by patch//2 and draws centres uniformly over the unit,
    so the synthetic-pixel share depends on the unit's size. predict_proba tiles
    at 256 with a 32-px halo, so a 256-px input yields exactly one window with no
    overlap-averaging while a full slice yields many.
    """
    half = patch // 2

    def pad_fraction(H, W):
        # Expected share of a sampled patch that falls in the reflect pad, for a
        # centre drawn uniformly over the (padded) unit.
        Hp, Wp = H + 2 * half, W + 2 * half
        ys = np.arange(Hp - patch + 1)
        xs = np.arange(Wp - patch + 1)
        ov_y = np.clip(np.minimum(ys + patch, half + H) - np.maximum(ys, half), 0, patch)
        ov_x = np.clip(np.minimum(xs + patch, half + W) - np.maximum(xs, half), 0, patch)
        real = np.outer(ov_y, ov_x).mean() / (patch * patch)
        touches = 1.0 - float(((ov_y == patch)[:, None] & (ov_x == patch)[None, :]).mean())
        return 1.0 - float(real), touches

    def windows(H, W, step=256, halo=32):
        ny = len(range(0, max(1, H - step + 1), step))
        nx = len(range(0, max(1, W - step + 1), step))
        cnt = np.zeros((H, W), np.int32)
        for y0 in range(0, max(1, H - step + 1), step):
            for x0 in range(0, max(1, W - step + 1), step):
                ya, yb = max(0, y0 - halo), min(H, y0 + step + halo)
                xa, xb = max(0, x0 - halo), min(W, x0 + step + halo)
                cnt[ya:yb, xa:xb] += 1
        return ny * nx, float(cnt.mean())

    slice_pad, slice_touch = pad_fraction(P.CROP_H, P.CROP_W)
    tile_pad, tile_touch = pad_fraction(tile, tile)
    s_win, s_cnt = windows(P.CROP_H, P.CROP_W)
    t_win, t_cnt = windows(tile, tile)
    return {
        "patch": patch, "tile_px": tile,
        "full_slice": {"shape": [P.CROP_H, P.CROP_W],
                       "reflect_pad_fraction_of_patch": slice_pad,
                       "frac_patches_touching_pad": slice_touch,
                       "inference_windows": s_win,
                       "mean_predictions_per_pixel": s_cnt},
        "tile": {"shape": [tile, tile],
                 "reflect_pad_fraction_of_patch": tile_pad,
                 "frac_patches_touching_pad": tile_touch,
                 "inference_windows": t_win,
                 "mean_predictions_per_pixel": t_cnt},
    }


def main(crop=True):
    raw, ids = runmod.load_band(crop)
    mask = ids > 0
    del ids
    print("band", raw.shape, flush=True)

    rep = {"note": ("Noise floors and structural handicaps behind the audit's "
                    "retraction. All computed with a fixed intensity rule, never "
                    "a trained model."),
           "crop": [P.CROP_H, P.CROP_W] if crop else "full",
           "window_length": WINDOW}

    print("window sweep...", flush=True)
    rep["window_sweep"] = window_sweep(raw, mask)
    w = rep["window_sweep"]
    print(f"  interior n={w['interior']['n']} mean={w['interior']['mean']:.4f} "
          f"sd={w['interior']['sd']:.4f}; audit window {AUDIT_T0} scores "
          f"{w['audit_window_iou']:.4f} = {w['audit_window_percentile_of_interior']:.1f} "
          f"percentile", flush=True)

    print("split draws...", flush=True)
    rep["split_draw"] = split_draw(raw, mask)
    s = rep["split_draw"]
    print(f"  n={s['n_seeds']} mean={s['mean']:.4f} sd={s['sd']:.4f}", flush=True)

    print("otsu constancy...", flush=True)
    rep["otsu_constancy"] = otsu_constancy(raw, mask)
    o = rep["otsu_constancy"]
    print(f"  cut range {o['cut_range']} constant={o['cut_is_constant']}; "
          f"operating point range {o['operating_point_range']} "
          f"constant={o['operating_point_is_constant']}", flush=True)

    rep["tile_handicaps"] = tile_handicaps()
    t = rep["tile_handicaps"]
    print(f"  pad fraction: slice {t['full_slice']['reflect_pad_fraction_of_patch']:.3f} "
          f"vs tile {t['tile']['reflect_pad_fraction_of_patch']:.3f}; "
          f"predictions/pixel: slice {t['full_slice']['mean_predictions_per_pixel']:.2f} "
          f"vs tile {t['tile']['mean_predictions_per_pixel']:.2f}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print("wrote", OUT, flush=True)
    return rep


if __name__ == "__main__":
    main(crop="--full" not in sys.argv)
