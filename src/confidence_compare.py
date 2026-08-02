"""Confidence bake-off across base models, at a chosen recall target.

Sky's question (2026-07-30): does training the base model on geometrically
augmented data give the CONFIDENCE model more to learn from, and therefore a
better prune?

[DEFINITION] "Confidence model" here means the **learned ERROR model** — a
regression on the magnitude of the base model's error (constitution §12). It is
one stage, not two: the trust classifier is reported only as a comparison arm
beside the free baselines.

The comparison holds everything except the base model fixed: same split, same
recall target, same candidate pool, same prune fitting. Each base model's own
VALIDATION probability map is what the meta-models are fit on, so a base model
that produces a richer (less saturated) probability map should give its
confidence model more signal to rank.

[LEAKAGE][MUST] Every estimator, recall threshold and prune cutoff is fit on the
VALIDATION zone; TEST is scored once with everything frozen.

[HONESTY] The MDE at 19 validation chondrules is 0.0142 IoU. Differences smaller
than that between arms are NOT resolvable, and augmenting the base model's
training data does not change that — augmentation adds pixel configurations, not
independent chondrules. This module therefore prints the MDE next to every
delta and flags anything under it.

Run:  python -m src.confidence_compare [--recall=0.99] name1 name2 ...
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from . import combine, confidence_bakeoff as cb, evaluate, metrics

OUT = "results/confidence_compare.json"
MDE = 0.0142            # 19 val chondrules; see pipeline_v2.power_report


def _probs(name):
    v = f"results/probs/{name}_val.npy"
    t = f"results/probs/{name}.npy"
    if not (os.path.exists(v) and os.path.exists(t)):
        return None
    return (np.load(v).astype(np.float32), np.load(t).astype(np.float32))


def run_one(name, d, target_recall=0.99, seed=0):
    """Bake-off for one base model. Returns a summary dict (never raises)."""
    p = _probs(name)
    if p is None:
        return {"model": name, "error": "no saved val/test probability maps"}
    pv, pt = p
    Rv, Mv, Vv = (d["val"][k] for k in ("raw", "mask", "valid"))
    Rt, Mt, Vt = (d["test"][k] for k in ("raw", "mask", "valid"))

    # Saturation of the base map is the thing that decides whether a confidence
    # model has anything to rank, so it is measured and reported, not assumed.
    graded = float(((pv > 1e-4) & (pv < 0.9999)).mean())

    thr_ref, ref_m = metrics.select_threshold(pv, Mv.astype(bool), Vv)
    base_test = metrics.metrics_from_pred(pt >= thr_ref, Mt, Vt)

    thr_rec, rec_m = combine.recall_threshold(pv, Mv, Vv, target_recall)
    unreachable = bool(rec_m.get("recall_target_unreachable", False))

    board, fitted = cb.run_bakeoff(pv, Rv, Mv, Vv, thr_ref,
                                   target_recall=target_recall, seed=seed)
    # [DEFINITION, Sky 2026-07-30] THE CONFIDENCE MODEL IS THE LEARNED ERROR
    # MODEL — one stage, not two (constitution §12). `best_error_model` below is
    # therefore THE deliverable. The trust classifier is reported beside it only
    # as a comparison arm, exactly like the free baselines, and a collapsed
    # "best learned" figure is not reported at all because it would hide whether
    # the deliverable actually won.
    def _best(task):
        c = [r for r in board if r.get("task") == task]
        if not c:
            return None
        b = max(c, key=lambda r: r["pruned_val_iou"])
        return {"name": b["name"], "pruned_val_iou": float(b["pruned_val_iou"])}

    best_trust, best_error, best_free = _best("trust"), _best("error"), _best("free")
    win = max(board, key=lambda r: r["pruned_val_iou"])

    # [MUST] `run_bakeoff` registers estimators as fitted[f"{task}_{name}"]
    # (confidence_bakeoff.py:283), matching pipeline_v2.py:512. Looking up the
    # bare name returns None, which sends evaluate_on_test down its free-baseline
    # branch and raises KeyError — so the arm would only ever complete when a
    # free baseline won, i.e. exactly when the learned error model LOST.
    est = fitted.get(f"{win.get('task')}_{win['name']}")
    test = cb.evaluate_on_test(win["name"], est, win.get("task"),
                               pt, Rt, Mt, Vt, thr_rec, win["cutoff"],
                               win.get("higher_is_worse", True), thr_feat=thr_ref)
    return {
        "model": name,
        "graded_frac_val": graded,
        "base_thr": float(thr_ref),
        "base_test_iou": float(base_test["iou"]),
        "recall_target": float(target_recall),
        "recall_thr": float(thr_rec),
        "recall_achieved_val": float(rec_m["recall"]),
        "recall_target_unreachable": unreachable,
        "overpredicted_val_iou": float(rec_m["iou"]),
        "winner": win["name"],
        "winner_task": win.get("task"),
        "winner_pruned_val_iou": float(win["pruned_val_iou"]),
        # [MUST] Constitution §9 requires over-predict-then-prune to be measured
        # HEAD-TO-HEAD against the max-IoU threshold. Recording only the TEST
        # gain hid that the pipeline LOSES on validation (-0.0071 / -0.0077),
        # i.e. the reported "+0.0084 gain" was a sub-noise-floor sign flip on the
        # split that was never used for selection. Both sides now ship.
        "base_val_iou": float(ref_m),
        "gain_vs_base_val": float(win["pruned_val_iou"] - ref_m),
        # Without the cutoff nobody can see that the "prune" collapsed to
        # keeping p == 1.0 — i.e. that the over-predict step never ran.
        "winner_cutoff": (None if win.get("cutoff") is None
                          else float(win["cutoff"])),
        "winner_ranking_metric": (None if win.get("ranking_metric") is None
                                  else float(win["ranking_metric"])),
        "winner_unpruned_val_iou": float(win.get("unpruned_val_iou", float("nan"))),
        "best_error_model": best_error,           # THE confidence model (§12)
        "best_trust_classifier": best_trust,      # comparison arm only
        "best_free_baseline": best_free,          # the bar it must clear
        "error_beats_trust": None if not (best_trust and best_error) else
            bool(best_error["pruned_val_iou"] > best_trust["pruned_val_iou"]),
        "error_beats_free": None if not (best_error and best_free) else
            bool(best_error["pruned_val_iou"] > best_free["pruned_val_iou"]),
        "pruned_test_iou": float(test.get("iou", float("nan"))),
        "gain_vs_base_test": float(test.get("iou", float("nan")) - base_test["iou"]),
        # [MUST] Constitution §5: IoU is never reported alone. §12 restates it
        # verbatim — "a bare IoU does not satisfy this clause" — so the full
        # metric set for BOTH the base and the pruned output is carried into the
        # JSON rather than discarded after the IoU is extracted.
        "base_test_metrics": {**{k: float(v) for k, v in base_test.items()
                                 if isinstance(v, (int, float))},
                              # metrics_from_pred does NOT compute auprc; it must
                              # be taken from the PROBABILITIES, not a hard mask.
                              "auprc": float(metrics.auprc(pt, Mt.astype(bool), Vt))},
        # No auprc here on purpose: the pruned output is a binary mask, so a
        # threshold-free ranking metric is undefined for it. Reporting a nan
        # column would read as "measured, undefined" rather than "not applicable".
        "pruned_test_metrics": {k: float(v) for k, v in test.items()
                                if isinstance(v, (int, float))},
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    recall = 0.99
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "recall":
            recall = float(v)
        else:
            raise SystemExit(f"unknown option {a}")
    names = [x for x in argv if not x.startswith("--")]
    if not names:
        raise SystemExit("usage: python -m src.confidence_compare "
                         "[--recall=0.99] <model> [<model> ...]")

    d = evaluate.load(".")
    rows = []
    for n in names:
        print(f"\n=== confidence bake-off: {n} (target recall {recall}) ===",
              flush=True)
        try:
            r = run_one(n, d, target_recall=recall)
        except Exception as e:                       # one arm must not kill the rest
            import traceback
            r = {"model": n, "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-600:]}
        rows.append(r)
        print("  " + json.dumps({k: v for k, v in r.items()
                                 if k != "traceback"}, indent=2)[:900], flush=True)
        os.makedirs("results", exist_ok=True)
        json.dump({"mde_iou": MDE, "recall_target": recall, "rows": rows},
                  open(OUT, "w"), indent=2)

    ok = [r for r in rows if "error" not in r]
    print("\n=== SUMMARY (target recall "
          f"{recall}; MDE {MDE:.4f} IoU) ===")
    print(f"{'base model':<32}{'graded%':>8}{'base IoU':>10}"
          f"{'pruned IoU':>12}{'gain':>9}")
    for r in ok:
        print(f"{r['model']:<32}{100*r['graded_frac_val']:7.2f}%"
              f"{r['base_test_iou']:10.4f}{r['pruned_test_iou']:12.4f}"
              f"{r['gain_vs_base_test']:+9.4f}")
    # The error model IS the confidence model (§12); trust/free are comparisons.
    print(f"\n{'base model':<32}{'ERROR MODEL = confidence':>26}"
          f"{'trust (comparison)':>22}{'free baseline':>18}")
    for r in ok:
        def _s(x):
            return "n/a" if not x else f"{x['name']}={x['pruned_val_iou']:.4f}"
        print(f"{r['model']:<32}{_s(r['best_error_model']):>26}"
              f"{_s(r['best_trust_classifier']):>22}{_s(r['best_free_baseline']):>18}")
        if r.get("error_beats_free") is not None:
            print(f"{'':<32}error model beats the free baseline: "
                  f"{r['error_beats_free']}   beats trust classifier: "
                  f"{r['error_beats_trust']}")
    # §5: never a bare IoU. AUPRC is threshold-free so it belongs to the BASE
    # probability map; the pruned output is a binary mask and has none.
    print(f"\n{'':<32}{'--- base map ---':>26}{'--- pruned output ---':>34}")
    print(f"{'base model':<32}{'Dice':>8}{'prec':>8}{'recall':>8}{'AUPRC':>8}"
          f"{'Dice':>10}{'prec':>8}{'recall':>8}")
    for r in ok:
        b, m = r.get("base_test_metrics", {}), r.get("pruned_test_metrics", {})
        nan = float("nan")
        print(f"{r['model']:<32}{b.get('dice', nan):8.4f}"
              f"{b.get('precision', nan):8.4f}{b.get('recall', nan):8.4f}"
              f"{b.get('auprc', nan):8.4f}"
              f"{m.get('dice', nan):10.4f}{m.get('precision', nan):8.4f}"
              f"{m.get('recall', nan):8.4f}")
    if len(ok) >= 2:
        a, b = ok[0], ok[-1]
        dv = b["pruned_test_iou"] - a["pruned_test_iou"]
        if dv != dv:                      # NaN: `abs(nan) < MDE` is False, which
            verdict = "UNDEFINED (NaN)"   # would otherwise print "above the floor"
        elif abs(dv) < MDE:
            verdict = "BELOW the noise floor — not resolvable"
        else:
            verdict = "above the noise floor"
        print(f"\n{b['model']} - {a['model']} = {dv:+.4f} IoU  ({verdict})")
        print(f"[REGIME] MDE {MDE:.4f} is derived from {19} VALIDATION chondrules "
              "and is applied here to a TEST delta; the true floor is probably "
              "higher (it uses 2*SE, not the conventional 2.8*SE).")
    return rows


if __name__ == "__main__":
    main()
