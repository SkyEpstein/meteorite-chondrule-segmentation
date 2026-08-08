"""Run the image-based confidence model and ask whether it adds anything.

The per-pixel regressors bought 1e-9 to 1e-5 IoU — nothing — and a zero-parameter
`margin` baseline beat all ten learned candidates. The suspected reason is that
their widest spatial window is 11x11 while chondrules are 50-100 px, so they
cannot represent "this whole object is wrong", which is the dominant measured
failure. This runs a U-Net over (raw, base probability) instead, and scores it
as a prune signal against exactly the same baselines.

[LEAKAGE][MUST] The U-Net never grades a slice it was fitted on. Default now
(`--cv=1`): k contiguous buffered folds inside the val slab, so every val slice
gets an out-of-fold error map, the prune cutoff is selected on all 60 of them,
and TEST is scored once with the all-val net and that cutoff frozen. The
regressors it is compared against fit AND select on all of val, so this protocol
is stricter, not looser.

[WHAT CHANGED, 2026-08-06] `--cv=0` is the old path: fit on one 28-slice half,
select the cutoff on the other 28. That run measured test IoU 0.11752 against a
base of 0.29443 and a margin baseline of 0.30279. Cross-fitting takes each fold's
fit set to 40-44 slices and the cutoff set to 60, which TESTS whether fit-set
size was the binding constraint. It does not assume the answer is yes. It costs
k + 1 fits, so roughly six times the training time at the default k = 5.

Run:  python -m src.run_error_cnn [--model unet2d_s30000] [--recall 0.99]
                                  [--cv=0]
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
    cv = True
    k_folds = error_cnn.CV_FOLDS
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "model":
            model = v
        elif k == "recall":
            recall = float(v)
        elif k == "cv":
            cv = bool(int(v))
        else:
            raise SystemExit(f"unknown option {a}")

    d = evaluate.load(".")
    Rv, Mv, Vv = (d["val"][k] for k in ("raw", "mask", "valid"))
    Rt, Mt, Vt = (d["test"][k] for k in ("raw", "mask", "valid"))
    pv = np.load(f"results/probs/{model}_val.npy").astype(np.float32)
    pt = np.load(f"results/probs/{model}.npy").astype(np.float32)

    nv = int(pv.shape[0])
    if cv:
        n_fit_report = nv
        print(f"val cross-fit: {k_folds} contiguous folds over {nv} slices, "
              f"buffer {error_cnn.VAL_BUFFER}; cutoff selected on the "
              f"out-of-fold map of all {nv}, TEST scored with the all-val net",
              flush=True)
    else:
        a, b = error_cnn.split_val(nv)
        n_fit_report = len(a)
        print(f"val split: fit on {len(a)} slices [{a[0]}-{a[-1]}], "
              f"select cutoff on {len(b)} [{b[0]}-{b[-1]}], "
              f"buffer {error_cnn.VAL_BUFFER}", flush=True)

    thr_ref, ref_val_iou = metrics.select_threshold(pv, Mv.astype(bool), Vv)
    thr_rec, rec_m = combine.recall_threshold(pv, Mv, Vv, recall)
    print(f"base thr {thr_ref:.4g} (val IoU {ref_val_iou:.4f}); "
          f"recall thr {thr_rec:.3e} -> achieved {rec_m['recall']:.4f}", flush=True)

    t0 = time.time()
    if cv:
        # k fold nets produce the out-of-fold error map; the (k+1)-th net is
        # fitted on all of val and is the one applied to TEST.
        e_sel, net, per_fold = error_cnn.fit_error_unet_cv(Rv, pv, Mv, Vv,
                                                           k=k_folds)
        sel = np.arange(nv)
        print(f"  {k_folds} + 1 fits in {time.time() - t0:.0f}s", flush=True)
    else:
        net = error_cnn.fit_error_unet(Rv, pv, Mv, Vv, a)
        print(f"  trained in {time.time() - t0:.0f}s", flush=True)
        per_fold, sel = [], b
        e_sel = error_cnn.predict_error(net, Rv[b], pv[b])

    # --- select the prune cutoff on slices no fitted net has seen ---
    pr = combine.fit_prune(pv[sel], Mv[sel], Vv[sel], e_sel, thr_rec,
                           higher_is_worse=True)
    lbl = f"out-of-fold, all {len(sel)} val slices" if cv else "held-out half B"
    print(f"  {lbl}: unpruned {pr['unpruned_val_iou']:.4f} -> "
          f"pruned {pr['val_iou']:.4f}  (gain {pr['gain']:+.4f}, "
          f"cutoff {pr['cutoff']:.4g})", flush=True)

    # --- the SAME protocol for the free baseline, so the comparison is fair ---
    from . import confidence_bakeoff as cb
    mscore, mhiw = cb.free_baselines(pv[sel], thr_rec)["margin"]
    pm = combine.fit_prune(pv[sel], Mv[sel], Vv[sel], mscore, thr_rec,
                           higher_is_worse=mhiw)
    print(f"  margin baseline on the same slices: pruned {pm['val_iou']:.4f} "
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
           "mode": "cv" if cv else "split",
           "k": k_folds if cv else None,
           "n_val_slices": nv,
           "fit_slices_per_fold": [f["n_fit"] for f in per_fold],
           "per_fold": per_fold,
           "val_fit_slices": n_fit_report, "val_cutoff_slices": len(sel),
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
    # [CORRECTED 2026-08-04] The 0.0142 floor hardcoded in the two gates below is
    # anchored on `pipeline_v2.SE_REF_N = 19`, which is the val zone's BINARY
    # connected-components count, not a chondrule count. The same-id rule gives
    # 36 there (`src/recount_chondrules.py`), and neither count was the
    # resampling unit, because the `SE_REF_IOU = 0.0071` it pairs with was
    # bootstrapped over 60 validation SLICES. The 0.0142 floor is unchanged.
    print(f"\n  error U-Net vs base   : {dv:+.4f}   "
          f"({'below' if abs(dv) < 0.0142 else 'above'} the 0.0142 floor)")
    print(f"  error U-Net vs margin : {dm:+.4f}   "
          f"({'below' if abs(dm) < 0.0142 else 'above'} the 0.0142 floor)")
    print(f"  beat margin on {lbl}: "
          f"{res['beats_margin_on_heldout_val']}")
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
