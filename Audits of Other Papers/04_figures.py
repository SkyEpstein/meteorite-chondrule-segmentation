"""Figures for the audit.

Reads only the JSON under results/, so every plotted number is traceable to a
recorded result (spec A5).

Run:  python "Audits of Other Papers/04_figures.py"
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

ORDER = ["P-patch-random-overlap", "P-tile-random", "P-slice-random",
         "P-zone-nobuffer", "P-zone-buffer", "P-zone-buffer-straddler"]
SHORT = {
    "P-patch-random-overlap": "random\noverlapping\npatches",
    "P-tile-random": "random\ntiles",
    "P-slice-random": "random\nslices",
    "P-zone-nobuffer": "contiguous\nzones\n(no buffer)",
    "P-zone-buffer": "zones\n+ buffer",
    "P-zone-buffer-straddler": "zones + buffer\n+ straddler fix\n(reference)",
}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fig_protocol_bars(comp):
    cells = [c for c in comp["cells"] if c.get("test_iou") == c.get("test_iou")]
    archs = [a for a in ["unet", "unetpp", "deeplabv3p"]
             if any(c["arch"] == a for c in cells)]
    prots = [p for p in ORDER if any(c["protocol"] == p for c in cells)]
    if not prots or not archs:
        return
    lut = {(c["protocol"], c["arch"]): c["test_iou"] for c in cells}
    otsu = {c["protocol"]: c["test_iou"] for c in cells if c["arch"] == "otsu"}

    x = np.arange(len(prots))
    w = 0.72 / len(archs)
    fig, ax = plt.subplots(figsize=(11.5, 6))
    for i, a in enumerate(archs):
        vals = [lut.get((p, a), np.nan) for p in prots]
        ax.bar(x + i * w - 0.36 + w / 2, vals, w, label=a)
    if otsu:
        ov = [otsu.get(p, np.nan) for p in prots]
        # NOT "untrained": it fits a cut on train and an operating point on val.
        # The operating point is measured identical across protocols (0.35) and
        # the cut moves only 36.357 to 37.354, which is why its spread across
        # test sets reads mostly as test composition.
        ax.plot(x, ov, "k.--", ms=11, lw=1.4,
                label="otsu control (fixed intensity rule)")

    ref = lut.get(("P-zone-buffer-straddler", "unet"))
    if ref == ref and ref is not None:
        ax.axhline(ref, color="0.4", lw=1, ls=":")
        ax.text(len(prots) - 0.45, ref + 0.006,
                f"leakage-free reference {ref:.3f}", ha="right", fontsize=9,
                color="0.3")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[p] for p in prots], fontsize=8.5)
    ax.set_ylabel("test IoU (chondrule)")
    # Not "leakage increases to the left": all three random protocols leak
    # completely and equally (100% of test chondrules also in train). The real
    # division is random versus contiguous, so say that instead of implying a
    # gradient the diagnostics do not support.
    ax.set_title("Single-draw grid: DOES NOT SURVIVE REPLICATION, see Finding 5\n"
                 f"identical {comp['budget_steps']}-step budget per cell, one split "
                 "draw and one test window throughout;\nthe three random protocols "
                 "(left) each leak 100% of test chondrules into training, but the "
                 "bar gap\nis not a measured effect size (replicated difference "
                 "+0.0290 IoU, 95% CI spans zero)", fontsize=10)
    # Headroom so the legend never sits on top of the tallest bar.
    top = np.nanmax([v for v in lut.values() if v == v] + [0.1])
    ax.set_ylim(0, top * 1.28)
    ax.legend(fontsize=9, ncol=4, loc="upper center",
              frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "protocol_comparison.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_leak_vs_gain(comp, diag):
    """Learned-model gain over the control against measured leakage.

    Plotting gain rather than raw IoU removes the part of test-set difficulty a
    per-pixel rule can see: the otsu operating point is measured identical across
    protocols and its cut nearly so. It does NOT remove CNN-specific differences
    (spec C5), so this is a partial normalization, not a clean subtraction.
    """
    cells = [c for c in comp["cells"] if c.get("test_iou") == c.get("test_iou")]
    otsu = {c["protocol"]: c["test_iou"] for c in cells if c["arch"] == "otsu"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for a, mk in (("unet", "o"), ("unetpp", "s"), ("deeplabv3p", "^")):
        xs, ys, labs = [], [], []
        for c in cells:
            if c["arch"] != a or c["protocol"] not in diag["protocols"]:
                continue
            d = diag["protocols"][c["protocol"]]
            if c["protocol"] not in otsu:
                continue
            xs.append(100 * d["test_pos_frac_from_shared_chondrules"])
            ys.append(c["test_iou"] - otsu[c["protocol"]])
            labs.append(c["protocol"])
        if xs:
            ax.plot(xs, ys, mk, ms=9, label=a, alpha=0.85)
            if a == "unet":
                for xx, yy, ll in zip(xs, ys, labs):
                    ax.annotate(ll.replace("P-", ""), (xx, yy), fontsize=7.5,
                                xytext=(4, 4), textcoords="offset points",
                                color="0.35")
    ax.set_xlabel("test positive pixels belonging to a chondrule that is also in "
                  "training (%)")
    ax.set_ylabel("test IoU gain over the fixed-intensity otsu control")
    ax.set_title("Gain over control against measured leakage (one draw each)\n"
                 "this trend does NOT survive replication, see Finding 5; tile "
                 "arms also sit low\nfor their leakage because of measured "
                 "inference handicaps (C5)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "leakage_vs_gain.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_dose(dose):
    pts = [p for p in dose["points"] if p.get("test_iou") == p.get("test_iou")]
    if len(pts) < 2:
        return
    b = [p["buffer"] for p in pts]
    v = [p["test_iou"] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(b, v, "o-", lw=2, ms=8, color="#1f77b4")
    ax.set_xlabel("buffer between train and test (band slices)")
    ax.set_ylabel("test IoU (chondrule)")
    corr = float(np.corrcoef(b, v)[0, 1])
    tp = [p.get("train_pos_frac") for p in pts]
    drop = (100.0 * (1.0 - tp[-1] / tp[0])) if all(tp) else float("nan")
    ax.set_title(f"Buffer sweep: no detectable effect (correlation {corr:+.3f})\n"
                 f"test zone and train size are fixed, but training foreground "
                 f"also falls {drop:.1f}% across the sweep,\nso this is NOT a "
                 f"clean proximity measurement; the causal reading is retracted",
                 fontsize=10)
    # Separation is 4(b+1) original slices, not 4b: a buffer of b leaves b+1
    # slices of gap. The report's table uses 4 to 84; match it.
    sec = ax.secondary_xaxis(
        "top", functions=(lambda x: (x + 1) * 4, lambda x: x / 4 - 1))
    sec.set_xlabel("physical separation (original CT slices)")
    ax.grid(alpha=0.3)
    for p in pts:
        ax.annotate(f"{p['test_iou']:.3f}", (p["buffer"], p["test_iou"]),
                    fontsize=8, xytext=(0, 8), textcoords="offset points",
                    ha="center", color="0.3")
    fig.tight_layout()
    out = os.path.join(FIG, "buffer_dose_response.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def fig_diagnostics(diag):
    prots = [p for p in ORDER if p in diag["protocols"]]
    shared = [100 * diag["protocols"][p]["test_pos_frac_from_shared_chondrules"]
              for p in prots]
    inst = [100 * diag["protocols"][p]["shared_instance_frac_of_test"]
            for p in prots]
    x = np.arange(len(prots))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, inst, 0.4, label="chondrules in test that are also in train")
    ax.bar(x + 0.2, shared, 0.4,
           label="test positive pixels from such chondrules")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[p] for p in prots], fontsize=8.5)
    ax.set_ylabel("percent")
    ax.set_title("Measured leakage per protocol, before any model is trained")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "leakage_diagnostics.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    diag = load("leakage_diagnostics.json")
    comp = load("protocol_comparison.json")
    dose = load("dose_response.json")
    if diag:
        fig_diagnostics(diag)
    if comp:
        fig_protocol_bars(comp)
        if diag:
            fig_leak_vs_gain(comp, diag)
    if dose:
        fig_dose(dose)
