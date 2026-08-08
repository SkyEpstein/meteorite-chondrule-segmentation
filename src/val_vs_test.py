"""Does validation IoU predict test IoU? Measured over every model with a saved map.

The 12-model version of this used `final_pipeline.json`'s ranking, which is the
TOP TEN by validation, so it was range restricted. This uses every model with a
`results/probs/<model>_val.npy`, which removes that restriction.

Method: for each leaderboard row, score its saved VAL probability map at the row's
own frozen threshold. That is the same quantity as `best_full_val_iou`, and the
script checks itself against the 12 models that have it recorded.

Loads VAL ONLY (60 slices), not train, not test. IoU only, no AUPRC, because
AUPRC sorts ~83M values per model and this runs 44 of them.

[CORRECTION] An earlier version of this analysis reported the minimum
informative val gap without the sqrt(2) and so understated it by that factor.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src import metrics, zones as zonemod                          # noqa: E402

PROBS = "results/probs"

# The 2-sigma bar compares TWO models, so the observed test difference carries
# the DIFFERENCE of two residuals, whose SD is sqrt(2) * residual_sd.
BAR_FORMULA = ("2*sqrt(2)*residual_sd/slope; the difference of two residuals "
               "has SD sqrt(2)*residual_sd")


def bar_2sigma(residual_sd, slope):
    """Smallest val gap that clears 2 sigma of the difference of two residuals."""
    return float(2 * np.sqrt(2) * residual_sd / abs(slope))


SUBSETS = [
    ("all", lambda r: True),
    ("excluding collapsed (test >= 0.10)", lambda r: r["test_iou"] >= 0.10),
    ("excluding weak (test >= 0.15)", lambda r: r["test_iou"] >= 0.15),
    ("competitive only (test >= 0.20)", lambda r: r["test_iou"] >= 0.20),
    ("C_cnn only", lambda r: r["tier"] == "C_cnn"),
    ("pre-fix only", lambda r: r["tiler_regime"] == "pre"),
]


def fit_subset(label, sub):
    """n, r, slope, residual SD and the corrected bar for one subset of rows."""
    n = len(sub)
    if n <= 3:
        print(f"  {label:<36} n={n:<3} too few models to fit")
        return {"label": label, "n": n}
    x = np.array([r["val_iou"] for r in sub])
    y = np.array([r["test_iou"] for r in sub])
    s, ic_, rr_, _, _ = stats.linregress(x, y)
    sd = float((y - (s * x + ic_)).std(ddof=2))
    bar = bar_2sigma(sd, s)
    print(f"  {label:<36} n={n:<3} r={rr_:+.3f}  slope={s:+.3f}  "
          f"residSD={sd:.5f}  bar={bar:.4f}")
    return {"label": label, "n": n, "pearson_r": float(rr_), "slope": float(s),
            "residual_sd": sd, "min_informative_val_gap_2sigma": bar}

lb = {r["model"]: r for r in
      json.load(open("results/leaderboard_all.json"))["leaderboard"]}

recorded = {}
for r in json.load(open("results/final_pipeline.json")).get("ranking", []):
    if "val_iou" in r:
        recorded[r["model"]] = r["val_iou"]
for r in json.load(open("results/pretrained_probe.json"))["rows"]:
    recorded[r["model"]] = r["val_iou"]

print("loading VAL zone only")
z = zonemod.load_zones(".", zones=("val",))
Rv, Mv, Vv = z["val"]
del Rv                                    # raw is not needed to score a saved map
print(f"  val: mask{Mv.shape} positives={int(Mv.sum())}")

rows, missing, checks = [], [], []
for model, r in sorted(lb.items()):
    p = os.path.join(PROBS, f"{model}_val.npy")
    if not os.path.exists(p):
        missing.append(model)
        continue
    thr = r.get("threshold")
    if thr is None or thr != thr:
        missing.append(f"{model} (no threshold)")
        continue
    prob = np.load(p, mmap_mode="r")
    if prob.shape != Mv.shape:
        missing.append(f"{model} (shape {prob.shape} vs val {Mv.shape})")
        del prob
        continue
    m = metrics.metrics_from_pred(np.asarray(prob) >= thr, Mv, Vv)
    del prob
    viou = float(m["iou"])
    rows.append({"model": model, "val_iou": viou,
                 "test_iou": float(r["iou"]), "threshold": float(thr),
                 "tier": r.get("tier"), "tiler_regime": r.get("tiler_regime")})
    if model in recorded:
        checks.append((model, recorded[model], viou))

print(f"\nscored {len(rows)} models; {len(missing)} skipped")
for s in missing:
    print(f"  skipped: {s}")

print("\nSELF-CHECK against recorded best_full_val_iou")
worst = 0.0
for model, rec, got in checks:
    d = abs(rec - got)
    worst = max(worst, d)
    flag = "" if d < 5e-3 else "   <-- MISMATCH"
    print(f"  {model:<32} recorded={rec:.5f} recomputed={got:.5f} d={d:.5f}{flag}")
print(f"  worst absolute difference: {worst:.5f}")

v = np.array([r["val_iou"] for r in rows])
t = np.array([r["test_iou"] for r in rows])
sl, ic, rr, pp, se = stats.linregress(v, t)
resid = t - (sl * v + ic)
rsd = float(resid.std(ddof=2))

print(f"\n=== ALL {len(rows)} MODELS ===")
print(f"  Pearson  r   = {stats.pearsonr(v, t)[0]:+.3f}  p={stats.pearsonr(v, t)[1]:.2e}")
print(f"  Spearman rho = {stats.spearmanr(v, t)[0]:+.3f}  p={stats.spearmanr(v, t)[1]:.2e}")
print(f"  slope        = {sl:+.3f}")
print(f"  residual SD  = {rsd:.5f}")
print(f"  val  range {v.min():.5f} to {v.max():.5f} (spread {v.ptp():.5f})")
print(f"  test range {t.min():.5f} to {t.max():.5f} (spread {t.ptp():.5f})")
print(f"\n  a val gap must exceed ~{bar_2sigma(rsd, sl):.4f} IoU to clear 2 sigma "
      f"of the residual scatter")

print(f"\n=== SUBSETS ({BAR_FORMULA}) ===")
subsets = [fit_subset(label, [r for r in rows if keep(r)])
           for label, keep in SUBSETS]

# Same-regime subset: mixing pre- and post-tiler-fix rows adds a step change in
# test that has nothing to do with val, which would inflate the residual.
pre = [r for r in rows if r["tiler_regime"] == "pre"]
if len(pre) > 3:
    v2 = np.array([r["val_iou"] for r in pre])
    t2 = np.array([r["test_iou"] for r in pre])
    sl2, ic2, r2, p2, _ = stats.linregress(v2, t2)
    rsd2 = float((t2 - (sl2*v2 + ic2)).std(ddof=2))
    print(f"\n=== PRE-FIX ONLY ({len(pre)} models, one regime) ===")
    print(f"  r = {r2:+.3f}  slope = {sl2:+.3f}  residual SD = {rsd2:.5f}")
    print(f"  a val gap must exceed ~{bar_2sigma(rsd2, sl2):.4f} IoU")

out = {"regime": ("val IoU recomputed from saved results/probs/<model>_val.npy at "
                  "each row's own val-frozen threshold; test IoU from "
                  "results/leaderboard_all.json; VAL zone only, IoU only"),
       "n_models": len(rows), "skipped": missing,
       "self_check_worst_abs_diff": worst,
       "pearson_r": float(stats.pearsonr(v, t)[0]),
       "spearman_rho": float(stats.spearmanr(v, t)[0]),
       "slope": float(sl), "residual_sd": rsd,
       "min_informative_val_gap_2sigma": bar_2sigma(rsd, sl),
       "bar_formula": BAR_FORMULA,
       "subsets": subsets,
       "rows": sorted(rows, key=lambda r: -r["val_iou"])}
with open("results/val_vs_test.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote results/val_vs_test.json")
