"""Generate all result figures from the saved JSON/npy artefacts.

Every figure is regenerated from disk, so it always matches the recorded
numbers rather than anything held in memory.

Run:  python -m src.make_figures
Writes: results/figures/*.png
"""
from __future__ import annotations
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "results/figures"
C_BEST, C_CNN, C_GNN, C_CLASSIC = "#2E7D32", "#5B8FF9", "#E8A33D", "#9E9E9E"


def _load(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def fig_leaderboard():
    d = _load("results/leaderboard_all.json")
    if not d:
        return
    rows = [r for r in d["leaderboard"] if r.get("iou") == r.get("iou")]
    rows.sort(key=lambda r: r["iou"])
    names = [r["model"] for r in rows]
    ious = [r["iou"] for r in rows]

    def colour(n):
        if n.startswith("gnn_") or "gcn" in n:
            return C_GNN
        if "_s" in n and n.rsplit("_s", 1)[-1].isdigit():
            return C_BEST
        return C_CNN

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(names))))
    ax.barh(names, ious, color=[colour(n) for n in names])
    ax.axvline(0.203, color=C_CLASSIC, ls="--", lw=1.6, label="classical-ML 0.203")
    ax.axvline(0.130, color="#BDBDBD", ls=":", lw=1.4, label="intensity floor 0.130")
    ax.axvline(0.92, color="crimson", ls="--", lw=2, label="annotation ceiling ~0.92")
    ax.set_xlabel("test foreground IoU"); ax.set_xlim(0, 1.0)
    ax.set_title("All models on the leakage-free test slab (ALH 83010)")
    ax.legend(loc="lower right", fontsize=8)
    for i, v in enumerate(ious):
        ax.text(v + .006, i, f"{v:.3f}", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/leaderboard.png", dpi=110); plt.close()
    print("  leaderboard.png")


def fig_budget():
    d = _load("results/leaderboard_all.json")
    if not d:
        return
    m = {r["model"]: r.get("iou") for r in d["leaderboard"]}
    series = {
        "unet2d (2D)": [(3000, m.get("unet2d")), (7200, m.get("long_p1_ctrl_b32")),
                        (15000, m.get("unet2d_s15000")), (30000, m.get("unet2d_s30000"))],
        "unet25d_k5 (2.5D)": [(3000, m.get("unet25d_k5")),
                              (15000, m.get("unet25d_k5_s15000")),
                              (30000, m.get("unet25d_k5_s30000"))],
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lbl, pts in series.items():
        pts = [(s, v) for s, v in pts if v is not None]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", lw=2, ms=7, label=lbl)
        for x, y in pts:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(0.203, color=C_CLASSIC, ls="--", lw=1.5, label="classical-ML 0.203")
    ax.set_xscale("log"); ax.set_xlabel("training steps (log)")
    ax.set_ylabel("test foreground IoU")
    ax.set_title("Budget scaling: training longer beat every architecture")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/budget_scaling.png", dpi=120); plt.close()
    print("  budget_scaling.png")


def fig_data_scaling():
    """Corrected scaling curve, with the confounded original shown for contrast."""
    d = _load("results/data_scaling_v2.json")
    if not d or not d.get("summary"):
        return
    s = d["summary"]
    xs = [r["n_slices"] for r in s]
    mu = [r["test_iou_mean"] for r in s]
    sd = [0.0 if r["test_iou_sd"] != r["test_iou_sd"] else r["test_iou_sd"]
          for r in s]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(xs, mu, yerr=sd, fmt="o-", lw=2, ms=7, capsize=4,
                color=C_BEST, label="random subsets, 3 seeds (mean ± SD)")
    for r in s:                                   # every individual seed
        ax.scatter([r["n_slices"]] * len(r["test_iou_all"]), r["test_iou_all"],
                   s=18, color=C_BEST, alpha=.35, zorder=3)

    old = _load("results/data_scaling.json")
    if old:
        ax.plot([r["n_slices"] for r in old], [r["test_iou"] for r in old],
                "s--", lw=1.5, ms=6, color=C_CLASSIC, alpha=.8,
                label="superseded: contiguous subsets, 1 seed")

    # The measurement floor. Effects below this are unresolvable at n=19.
    ax.axhspan(min(mu) - 0.0071, min(mu) + 0.0071, color="k", alpha=.05)
    ax.set_xlabel("annotated training slices (of 213 available)")
    ax.set_ylabel("test foreground IoU")
    ax.set_title("Does more annotated data help?\n"
                 f"fixed {d.get('runs', [{}])[0].get('steps', '?')}-step budget; "
                 "shape, not height (30k-step run scores 0.3159)", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{FIG}/data_scaling.png", dpi=120); plt.close()
    print("  data_scaling.png")


def fig_pipeline():
    d = _load("results/final_pipeline.json")
    if not d:
        return
    stages, vals = ["base model"], [d["baseline_test"]["iou"]]
    if "z_refine" in d:
        stages.append("+ z-refine"); vals.append(d["z_refine"]["test"]["iou"])
    sw = d.get("fp_prune_sweep") or []
    if sw:
        best = max(sw, key=lambda e: e.get("val_iou", -1))
        stages.append(f"+ FP-prune\n(R={best['target_recall']:.2f})")
        vals.append(best["test"]["iou"])
    if "prune_plus_z" in d:
        stages.append("+ prune + z"); vals.append(d["prune_plus_z"]["iou"])
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, vals, color=[C_CNN] * (len(vals) - 1) + [C_BEST])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.axhline(vals[0], color="grey", ls=":", lw=1.2)
    ax.set_ylabel("test foreground IoU")
    ax.set_title(f"Pipeline gains on the locked model ({d.get('winner','?')})")
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/pipeline_gains.png", dpi=120); plt.close()
    print("  pipeline_gains.png")


def fig_bakeoff():
    d = _load("results/final_pipeline.json")
    if not d or "bakeoff" not in d:
        return
    rows = sorted(d["bakeoff"], key=lambda r: r.get("pruned_val_iou", 0))
    names = [f"{r['name']} ({r['task']})" for r in rows]
    vals = [r.get("pruned_val_iou", 0) for r in rows]
    cols = [C_CLASSIC if r["task"] == "free" else C_CNN for r in rows]
    fig, ax = plt.subplots(figsize=(9, max(4, .35 * len(names))))
    ax.barh(names, vals, color=cols)
    ax.set_xlabel("pruned VALIDATION IoU")
    ax.set_title("Confidence bake-off: grey = free baselines (no training)")
    for i, v in enumerate(vals):
        ax.text(v + .003, i, f"{v:.4f}", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig(f"{FIG}/confidence_bakeoff.png", dpi=110); plt.close()
    print("  confidence_bakeoff.png")


def fig_hierarchy():
    # Prefer the POST-FIX run. `hierarchy.json` is the superseded pre-fix result
    # (1.58x), produced when the agglomerative merge had no size term and one
    # region absorbed 98-99% of each slice. Plotting it would ship a retracted
    # number as a live figure.
    d = _load("results/hierarchy_fixed.json") or _load("results/hierarchy.json")
    if not d:
        return
    # hierarchy_fixed.json is a summary: its "levels" is a list of region COUNTS,
    # whereas the older per-level file stores a list of dicts.
    if not isinstance(d.get("levels", [{}])[0], dict):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(["area processed", "chondrules kept"],
               [d["test_frac_alive"], d["retention"]], color=[C_CNN, C_BEST])
        ax.axhline(1.0, color="grey", ls=":", lw=1)
        ax.set_ylim(0, 1.08); ax.set_ylabel("fraction")
        ax.set_title(f"Hierarchical superpixel cascade: {d['compute_saving']:.2f}x "
                     f"compute saving\n(recall target {d['best_target_recall']}, "
                     f"val-fit, test-applied)")
        for i, v in enumerate([d["test_frac_alive"], d["retention"]]):
            ax.text(i, v + .02, f"{v:.4f}", ha="center", fontsize=10)
        plt.tight_layout(); plt.savefig(f"{FIG}/hierarchy.png", dpi=120); plt.close()
        print("  hierarchy.png (post-fix)")
        return
    lv = d.get("levels", [])
    if not lv:
        return
    xs = list(range(len(lv)))
    alive = [l["frac_alive"] for l in lv]
    ret = [l["chondrule_retention"] for l in lv]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, alive, "o-", lw=2, color=C_CNN, label="image area still processed")
    ax.plot(xs, ret, "s-", lw=2, color=C_BEST, label="chondrule pixels retained")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{i}\n{l.get('n_regions','')}" for i, l in enumerate(lv)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction")
    ax.set_title("Hierarchical superpixel cascade: drop empty regions, keep every chondrule")
    ax.grid(alpha=.3); ax.legend()
    for x, a in zip(xs, alive):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/hierarchy.png", dpi=120); plt.close()
    print("  hierarchy.png")


def fig_qualitative():
    """Raw | ground truth | best-model prediction, on the densest test slices."""
    d = _load("results/final_pipeline.json")
    if not d:
        return
    name = d.get("winner")
    pt_path = f"results/probs/{name}.npy"
    if not name or not os.path.exists(pt_path):
        return
    from . import evaluate
    z = evaluate.load(".")
    Rt, Mt, Vt = z["test"]["raw"], z["test"]["mask"], z["test"]["valid"]
    prob = np.load(pt_path).astype(np.float32)
    thr = d["baseline_test"]["threshold"]
    order = np.argsort([-(Mt[i] & Vt[i]).sum() for i in range(len(Mt))])[:3]
    fig, ax = plt.subplots(len(order), 3, figsize=(14, 3.6 * len(order)))
    for r, i in enumerate(order):
        ax[r, 0].imshow(Rt[i], cmap="gray"); ax[r, 0].set_title(f"raw test slice {i}")
        ax[r, 1].imshow(Rt[i], cmap="gray")
        ax[r, 1].imshow(np.ma.masked_where(~Mt[i], Mt[i]), cmap="autumn", alpha=.65)
        ax[r, 1].set_title("ground truth")
        pred = prob[i] >= thr
        ax[r, 2].imshow(Rt[i], cmap="gray")
        ax[r, 2].imshow(np.ma.masked_where(~pred, pred), cmap="winter", alpha=.65)
        ax[r, 2].set_title(f"{name} (thr={thr:.3f})")
        for c in range(3):
            ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(f"{FIG}/qualitative.png", dpi=100); plt.close()
    print("  qualitative.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    print("figures:")
    for fn in (fig_leaderboard, fig_budget, fig_data_scaling, fig_pipeline,
               fig_bakeoff, fig_hierarchy, fig_qualitative):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print("done ->", FIG)


if __name__ == "__main__":
    main()
