"""Run the image-based confidence model and ask whether it adds anything.

The per-pixel regressors bought 1e-9 to 1e-5 IoU — nothing — and a zero-parameter
`margin` baseline beat all ten learned candidates. The suspected reason is that
their widest spatial window is 11x11 while chondrules are 50-100 px, so they
cannot represent "this whole object is wrong", which is the dominant measured
failure. This runs a U-Net over (raw, base probability) instead, and scores it
as a prune signal against exactly the same baselines.

[LEAKAGE][MUST] Val is SPLIT (contiguous, buffered): the U-Net trains on half A,
its prune cutoff is selected on half B, and TEST is scored once with both frozen.
The regressors it is compared against fit AND select on all of val, so this
protocol is stricter, not looser.

Run:  python -m src.run_error_cnn [--model unet2d_s30000] [--recall 0.99]
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np

from . import combine, error_cnn, evaluate, metrics

OUT = "results/error_cnn.json"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    model = "unet2d_s30000"
    recall = 0.99
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "model":
            model = v
        elif k == "recall":
            recall = float(v)
        else:
            raise SystemExit(f"unknown option {a}")

    d = evaluate.load(".")
    Rv, Mv, Vv = (d["val"][k] for k in ("raw", "mask", "valid"))
    Rt, Mt, Vt = (d["test"][k] for k in ("raw", "mask", "valid"))
    pv = np.load(f"results/probs/{model}_val.npy").astype(np.float32)
    pt = np.load(f"results/probs/{model}.npy").astype(np.float32)

    a, b = error_cnn.split_val(pv.shape[0])
    print(f"val split: fit on {len(a)} slices [{a[0]}-{a[-1]}], "
          f"select cutoff on {len(b)} [{b[0]}-{b[-1]}], "
          f"buffer {error_cnn.VAL_BUFFER}", flush=True)

    thr_ref, ref_val_iou = metrics.select_threshold(pv, Mv.astype(bool), Vv)
    thr_rec, rec_m = combine.recall_threshold(pv, Mv, Vv, recall)
    print(f"base thr {thr_ref:.4g} (val IoU {ref_val_iou:.4f}); "
          f"recall thr {thr_rec:.3e} -> achieved {rec_m['recall']:.4f}", flush=True)

    t0 = time.time()
    net = error_cnn.fit_error_unet(Rv, pv, Mv, Vv, a)
    print(f"  trained in {time.time() - t0:.0f}s", flush=True)

    # --- select the prune cutoff on the HELD-OUT val half ---
    eb = error_cnn.predict_error(net, Rv[b], pv[b])
    pr = combine.fit_prune(pv[b], Mv[b], Vv[b], eb, thr_rec, higher_is_worse=True)
    print(f"  held-out half B: unpruned {pr['unpruned_val_iou']:.4f} -> "
          f"pruned {pr['val_iou']:.4f}  (gain {pr['gain']:+.4f}, "
          f"cutoff {pr['cutoff']:.4g})", flush=True)

    # --- the SAME protocol for the free baseline, so the comparison is fair ---
    from . import confidence_bakeoff as cb
    mscore, mhiw = cb.free_baselines(pv[b], thr_rec)["margin"]
    pm = combine.fit_prune(pv[b], Mv[b], Vv[b], mscore, thr_rec,
                           higher_is_worse=mhiw)
    print(f"  margin baseline on the same half: pruned {pm['val_iou']:.4f} "
          f"(gain {pm['gain']:+.4f})", flush=True)

    # --- TEST, once, both frozen ---
    et = error_cnn.predict_error(net, Rt, pt)
    final_cnn = combine.apply_prune(pt, thr_rec, et, pr["cutoff"], True) & Vt
    mst, _ = cb.free_baselines(pt, thr_rec)["margin"]
    final_mar = combine.apply_prune(pt, thr_rec, mst, pm["cutoff"], mhiw) & Vt
    base_t = (pt >= thr_ref) & Vt

    def M(p):
        m = metrics.metrics_from_pred(p, Mt, Vt)
        return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}

    res = {"model": model, "recall_target": recall,
           "val_fit_slices": len(a), "val_cutoff_slices": len(b),
           "base_val_iou": float(ref_val_iou),
           "error_cnn_heldout_val_iou": float(pr["val_iou"]),
           "error_cnn_heldout_gain": float(pr["gain"]),
           "margin_heldout_val_iou": float(pm["val_iou"]),
           "margin_heldout_gain": float(pm["gain"]),
           "beats_margin_on_heldout_val": bool(pr["val_iou"] > pm["val_iou"]),
           "test_base": M(base_t), "test_error_cnn": M(final_cnn),
           "test_margin": M(final_mar)}
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\n=== DID THE IMAGE CONFIDENCE MODEL ADD ANYTHING? ({model}) ===")
    print(f"{'arm':<26}{'IoU':>9}{'Dice':>9}{'prec':>9}{'recall':>9}")
    for nm, m in (("base (threshold only)", res["test_base"]),
                  ("+ margin baseline", res["test_margin"]),
                  ("+ error U-Net", res["test_error_cnn"])):
        print(f"{nm:<26}{m['iou']:9.4f}{m['dice']:9.4f}"
              f"{m['precision']:9.4f}{m['recall']:9.4f}")
    dv = res["test_error_cnn"]["iou"] - res["test_base"]["iou"]
    dm = res["test_error_cnn"]["iou"] - res["test_margin"]["iou"]
    print(f"\n  error U-Net vs base   : {dv:+.4f}   "
          f"({'below' if abs(dv) < 0.0142 else 'above'} the 0.0142 floor)")
    print(f"  error U-Net vs margin : {dm:+.4f}   "
          f"({'below' if abs(dm) < 0.0142 else 'above'} the 0.0142 floor)")
    print(f"  beat margin on the held-out val half: "
          f"{res['beats_margin_on_heldout_val']}")
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
