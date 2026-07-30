"""Post-hoc stages: z-refinement, meta-models, prune pipeline, combos.

Everything here is FIT ON VALIDATION predictions and only APPLIED to test
[LEAKAGE][MUST]. Nothing retrains a base model; it consumes probability maps
saved by `evaluate.run_and_score` (`results/probs/<name>.npy` for test and
`<name>_val.npy` for validation).

Stages, in the order the constitution motivates them:
  1. cross-slice z-refinement            (P: adjacent slices agree ~0.83)
  2. learned confidence classifier       (trust map, R11)
  3. learned error regressor             (error bar, R12)
  4. over-predict then prune             (P9: FNs unrecoverable, FPs fixable)
  5. two-model combinations              (R6, when a 2nd model is supplied)

Usage:
    python -m src.run_posthoc <base_name> [second_name]
"""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np

from . import combine, evaluate, metrics


def _load(name):
    t = f"results/probs/{name}.npy"
    v = f"results/probs/{name}_val.npy"
    for p in (t, v):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing: run the model with save_prob first")
    return np.load(v).astype(np.float32), np.load(t).astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    base = sys.argv[1]
    second = sys.argv[2] if len(sys.argv) > 2 else None
    t0 = time.time()

    d = evaluate.load(".")
    Rv, Mv, Vv = d["val"]["raw"], d["val"]["mask"], d["val"]["valid"]
    Rt, Mt, Vt = d["test"]["raw"], d["test"]["mask"], d["test"]["valid"]
    pv, pt = _load(base)
    out = {"base_model": base,
           "regime": "all parameters fit on VALIDATION; applied once to TEST; "
                     "foreground=chondrule; N=1 (ALH 83010)"}

    # --- reference: plain max-IoU threshold, selected on val ---
    thr_ref, val_ref = metrics.select_threshold(pv, Mv, Vv)
    out["reference_max_iou"] = {"threshold": float(thr_ref),
                                "val_iou": float(val_ref),
                                "test": evaluate.score(pt, Mt, Vt, thr_ref)}
    print(f"[ref ] max-IoU thr={thr_ref:.2f} test IoU={out['reference_max_iou']['test']['iou']:.4f}",
          flush=True)

    # --- 1. cross-slice z-refinement ---
    zr = combine.fit_z_refine(pv, Mv, Vv)
    pt_z = combine.z_smooth(pt, k=zr["k"], mode=zr["mode"])
    out["z_refine"] = {**{k: zr[k] for k in ("mode", "k", "thr", "val_iou")},
                       "test": evaluate.score(pt_z, Mt, Vt, zr["thr"])}
    print(f"[z   ] {zr['mode']} k={zr['k']} thr={zr['thr']:.2f} "
          f"test IoU={out['z_refine']['test']['iou']:.4f} "
          f"(delta {out['z_refine']['test']['iou'] - out['reference_max_iou']['test']['iou']:+.4f})",
          flush=True)

    # --- 2. confidence classifier (trust) ---
    cm = combine.ConfidenceModel().fit(pv, Rv, Mv, Vv, thr=thr_ref)
    trust_t = cm.predict(pt, Rt, thr_ref)
    out["confidence"] = {"risk_coverage_test":
                         combine.ConfidenceModel.risk_coverage(trust_t, pt, Mt, Vt, thr_ref)}
    print(f"[conf] risk-coverage computed on test", flush=True)

    # --- 3. error regressor (error bar) ---
    em = combine.ErrorModel().fit(pv, Rv, Mv, Vv, thr=thr_ref)
    err_t = em.predict(pt, Rt, thr_ref)
    ev = combine.ErrorModel.evaluate(err_t, pt, Mt, Vt)
    ev.pop("calibration_full", None)
    out["error_model"] = {k: ev[k] for k in ev if k != "calibration"}
    out["error_model"]["calibration"] = ev["calibration"]
    out["error_model"]["per_slice_predicted_error"] = \
        combine.ErrorModel.aggregate(err_t).tolist()
    print(f"[err ] pearson={ev.get('pearson')} spearman={ev.get('spearman')} "
          f"pred={ev['mean_predicted_error']:.4f} actual={ev['mean_actual_error']:.4f}",
          flush=True)

    # --- 4. over-predict then prune (P9) ---
    # How far should the base model over-predict before pruning? Pruning can
    # only REMOVE pixels, so final recall <= starting recall; and at recall 1.0
    # the threshold collapses to ~0, precision falls to the ~5% base rate, and
    # the meta-model is effectively doing the whole segmentation. So the recall
    # target is swept and the winner chosen on VAL, never assumed. [LEAKAGE]
    trust_v = cm.predict(pv, Rv, thr_ref)
    err_v = em.predict(pv, Rv, thr_ref)
    sweep = []
    for target in (0.80, 0.90, 0.95, 0.98, 0.99, 1.00):
        thr_rec, recm = combine.recall_threshold(pv, Mv, Vv, target_recall=target)
        entry = {"target_recall": target, "threshold": thr_rec,
                 "val_recall_achieved": recm["recall"],
                 "val_precision_at_threshold": recm["precision"],
                 "unpruned_test": evaluate.score(pt, Mt, Vt, thr_rec)}
        for tag, sv, st, worse in (("error", err_v, err_t, True),
                                   ("trust", trust_v, trust_t, False)):
            fit = combine.fit_prune(pv, Mv, Vv, sv, thr_rec, higher_is_worse=worse)
            pred_t = combine.apply_prune(pt, thr_rec, st, fit["cutoff"],
                                         higher_is_worse=worse)
            entry[tag] = {"cutoff": fit["cutoff"], "val_iou": fit["val_iou"],
                          "unpruned_val_iou": fit["unpruned_val_iou"],
                          "test": metrics.metrics_from_pred(pred_t, Mt, Vt)}
        sweep.append(entry)
        print(f"[prun] target R={target:.2f} thr={thr_rec:.3f} "
              f"(val R={recm['recall']:.3f} P={recm['precision']:.3f}) | "
              f"err-prune test IoU={entry['error']['test']['iou']:.4f} "
              f"R={entry['error']['test']['recall']:.3f} "
              f"P={entry['error']['test']['precision']:.3f} | "
              f"trust-prune IoU={entry['trust']['test']['iou']:.4f}", flush=True)
    best = max(sweep, key=lambda e: max(e["error"]["val_iou"], e["trust"]["val_iou"]))
    out["prune"] = {"recall_target_sweep": sweep,
                    "best_by_val": best["target_recall"],
                    "note": "pruning can only remove pixels, so final recall <= "
                            "starting recall; the recall target is selected on val"}
    print(f"[prun] best recall target by VAL = {best['target_recall']}", flush=True)

    # --- 5. two-model combinations ---
    if second:
        qv, qt = _load(second)
        ta, tv_, viou = combine.fit_combo_veto(pv, 1.0 - qv, Mv, Vv)
        veto_t = combine.combo_veto(pt, 1.0 - qt, ta, tv_)
        w, thr_w, wiou = combine.fit_combo_average(pv, qv, Mv, Vv)
        avg_t = combine.combo_average(pt, qt, w)
        stk = combine.combo_stack(pv, qv, Mv, Vv)
        stk_t = combine.apply_stack(stk, pt, qt)
        thr_s, siou = metrics.select_threshold(combine.apply_stack(stk, pv, qv), Mv, Vv)
        out["combos"] = {
            "second_model": second,
            "veto": {"thr_a": ta, "thr_veto": tv_, "val_iou": viou,
                     "test": metrics.metrics_from_pred(veto_t, Mt, Vt)},
            "average": {"w": w, "thr": thr_w, "val_iou": wiou,
                        "test": evaluate.score(avg_t, Mt, Vt, thr_w)},
            "stack": {"thr": thr_s, "val_iou": siou,
                      "test": evaluate.score(stk_t, Mt, Vt, thr_s)},
            "mean_disagreement": float(combine.disagreement_map(pt, qt).mean()),
        }
        for k in ("veto", "average", "stack"):
            print(f"[comb] {k}: test IoU={out['combos'][k]['test']['iou']:.4f}", flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    os.makedirs("results", exist_ok=True)
    with open("results/posthoc.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote results/posthoc.json ({out['runtime_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
