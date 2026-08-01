"""Crossed experiment: split protocol x architecture on the ALH 83010 band.

Spec R5. Every cell trains from scratch at an identical budget, selects its
operating threshold on its own protocol's validation set, and scores its own
protocol's test set exactly once. The only thing that varies between rows of a
column is the split protocol, so the column is a direct read of what that
protocol licenses a paper to report.

The `otsu` arm is the control for spec confound C1. It is a TWO-PARAMETER
FITTED rule, not an untrained one: the cut is fitted on train intensities and
the operating point is selected on val. Measured on this volume the operating
point is identical across every protocol (0.35) and the cut varies only from
36.357 to 37.354, so with two fitted parameters it cannot memorize 142
chondrules and its spread across test sets indexes test-set difficulty for a
per-pixel rule. It is a sanity check, NOT a subtraction, and it is blind to
every CNN-specific difference in spec confound C5. See
results/noise_floors.json.

Run:  python "Audits of Other Papers/02_run_protocols.py" [--budget 3000] [--smoke]
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, ROOT)

from src import data, metrics, split as splitmod                 # noqa: E402
from src import models_cnn as mc                                 # noqa: E402
from src import models_extra as me                               # noqa: E402
import protocols as P                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "protocol_comparison.json")


# --------------------------------------------------------------------------
# Architecture arm. Names map to the models the corpus papers actually used.
# --------------------------------------------------------------------------

def architectures(budget):
    """(name, builder) pairs. builder() -> runner with .fit/.predict_proba."""
    epochs, steps = 15, max(1, budget // 15)
    extra = me.extra_models()
    nets = {
        "unet": lambda in_ch: mc.UNet2D(in_ch=in_ch, base=32, depth=4),
        "unetpp": extra["unetpp"],
        "deeplabv3p": extra["deeplabv3p"],
    }

    def runner(arch):
        return lambda n: mc.CNNRunner(f"{arch}_{n}", nets[arch], mc.dice_loss,
                                      k=1, epochs=epochs, steps_per_epoch=steps)

    return [("otsu", None)] + [(a, runner(a)) for a in nets]


def otsu_probs(raw_train, raw_eval):
    """Global Otsu threshold fitted on train intensities, applied to eval.

    Returned as a pseudo-probability so it flows through the same scoring path:
    a smooth ramp centred on the Otsu cut, which leaves the threshold search
    free to move the operating point exactly as it does for a CNN.
    """
    from skimage.filters import threshold_otsu
    t = float(threshold_otsu(raw_train.astype(np.float32)))
    scale = max(1.0, 0.25 * t)
    return 1.0 / (1.0 + np.exp(-(raw_eval.astype(np.float32) - t) / scale))


# --------------------------------------------------------------------------
# Band loading
# --------------------------------------------------------------------------

def load_band(crop=True):
    raw_paths, lab_paths = data.list_pairs(REPO)
    fg = data.fg_fraction(lab_paths)
    b0, b1 = data.annotated_band(fg)
    idxs = list(range(b0, b1 + 1))
    raw = P.crop_slices(data.load_stack(raw_paths, idxs, kind="raw"), crop)
    ids = P.crop_slices(data.load_stack(lab_paths, idxs, kind="raw"), crop)
    return raw, ids


def build_zones(name, cfg, raw, ids, inst, ninst):
    """Materialize {train,val,test} -> {raw, mask, valid} for one protocol."""
    N = raw.shape[0]
    sel = cfg["fn"](N, seed=0)
    mask = ids > 0

    if cfg["kind"] == "slice":
        zc = np.zeros(raw.shape, np.uint8)
        for code, z in ((1, "train"), (2, "val"), (3, "test")):
            zc[sel[z]] = code
    else:
        zc = np.zeros(raw.shape, np.uint8)
        for code, z in ((1, "train"), (2, "val"), (3, "test")):
            for (zz, y0, x0) in sel[z]:
                zc[zz, y0:y0 + P.TILE, x0:x0 + P.TILE] = code

    if cfg.get("straddler"):
        ignore = P.straddler_ignore(inst, zc, ninst)
        valid = ~ignore
    else:
        valid = np.ones(raw.shape, bool)
    del zc
    gc.collect()

    if cfg["kind"] == "slice":
        return P.materialize_slices(raw, mask, valid, sel)
    return P.materialize_tiles(raw, mask, valid, sel)


# --------------------------------------------------------------------------

def main(budget=3000, crop=True, smoke=False, only=None):
    raw, ids = load_band(crop)
    N = raw.shape[0]
    print(f"band N={N} shape={raw.shape} dtype={raw.dtype} budget={budget}",
          flush=True)

    # Instance labels are needed only by the straddler-resolving protocol.
    need_inst = any(c.get("straddler") for c in P.PROTOCOLS.values())
    inst, ninst = (splitmod.instance_label(ids) if need_inst else (None, 0))
    print(f"instances={ninst}", flush=True)

    archs = architectures(budget)
    if smoke:
        archs = [a for a in archs if a[0] in ("otsu", "unet")]

    results = {
        "regime": (f"crossed split-protocol x architecture on ALH 83010; "
                   f"identical budget {budget} steps per CNN cell; threshold "
                   f"selected on each protocol's own VAL and frozen for one "
                   f"TEST scoring; foreground=chondrule; "
                   f"{'centered 768x1280 crop' if crop else 'full 962x1450 field'}; "
                   f"N=1 meteorite"),
        "budget_steps": budget,
        "crop": [P.CROP_H, P.CROP_W] if crop else "full",
        "cells": [],
    }
    if os.path.exists(OUT):
        with open(OUT) as f:
            prev = json.load(f)
        results["cells"] = prev.get("cells", [])
    done = {(c["protocol"], c["arch"]) for c in results["cells"]}

    for pname, cfg in P.PROTOCOLS.items():
        if only and pname not in only:
            continue
        if all((pname, a[0]) in done for a in archs):
            print(f"\n=== {pname}: all cells done, skipping ===", flush=True)
            continue
        print(f"\n=== protocol {pname}: {cfg['label']} ===", flush=True)
        t0 = time.time()
        z = build_zones(pname, cfg, raw, ids, inst, ninst)
        print(f"    materialized in {time.time() - t0:.1f}s  "
              f"train={z['train']['raw'].shape} val={z['val']['raw'].shape} "
              f"test={z['test']['raw'].shape}", flush=True)

        for aname, builder in archs:
            if (pname, aname) in done:
                print(f"    [{aname}] cached, skipping", flush=True)
                continue
            t1 = time.time()
            cell = {"protocol": pname, "protocol_label": cfg["label"],
                    "arch": aname, "budget_steps": budget,
                    "train_pos_frac": float(
                        (z["train"]["mask"] & z["train"]["valid"]).mean())}
            try:
                if aname == "otsu":
                    pv = otsu_probs(z["train"]["raw"], z["val"]["raw"])
                    thr, _ = metrics.select_threshold(
                        pv, z["val"]["mask"], z["val"]["valid"])
                    pt = otsu_probs(z["train"]["raw"], z["test"]["raw"])
                else:
                    r = builder(f"{pname}_{aname}")
                    r.fit({k: z["train"][k] for k in ("raw", "mask", "valid")},
                          {k: z["val"][k] for k in ("raw", "mask", "valid")})
                    thr = float(r.best_thr)
                    pv = r.predict_proba(z["val"]["raw"])
                    pt = r.predict_proba(z["test"]["raw"])

                m_val = metrics.metrics_from_pred(
                    pv >= thr, z["val"]["mask"], z["val"]["valid"])
                m = metrics.metrics_from_pred(
                    pt >= thr, z["test"]["mask"], z["test"]["valid"])
                m["auprc"] = metrics.auprc(pt, z["test"]["mask"], z["test"]["valid"])
                cell.update({
                    "test_iou": m["iou"], "test_dice": m["dice"],
                    "test_precision": m["precision"], "test_recall": m["recall"],
                    "test_auprc": m["auprc"], "val_iou": m_val["iou"],
                    "threshold": thr,
                    "test_pos_frac": float(
                        (z["test"]["mask"] & z["test"]["valid"]).mean()),
                })
                print(f"    [{aname}] test IoU={m['iou']:.4f} "
                      f"dice={m['dice']:.4f} val IoU={m_val['iou']:.4f} "
                      f"thr={thr:.4f} ({time.time() - t1:.0f}s)", flush=True)
                del pv, pt
            except Exception as e:
                import traceback
                cell["error"] = f"{type(e).__name__}: {e}"
                cell["traceback"] = traceback.format_exc()[-600:]
                cell["test_iou"] = float("nan")
                print(f"    [{aname}] FAILED {e}", flush=True)
            cell["sec"] = round(time.time() - t1, 1)
            results["cells"].append(cell)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=2)
            gc.collect()

        del z
        gc.collect()

    print(f"\nwrote {OUT}", flush=True)
    return results


if __name__ == "__main__":
    a = sys.argv
    b = int(a[a.index("--budget") + 1]) if "--budget" in a else 3000
    main(budget=b, crop="--full" not in a, smoke="--smoke" in a)
