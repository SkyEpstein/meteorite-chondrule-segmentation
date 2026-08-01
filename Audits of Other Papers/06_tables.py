"""Emit the report's markdown tables from the recorded JSON.

Every table in README.md is generated here rather than typed, so no number in
the report can drift from the file it came from (spec A5).

Run:  python "Audits of Other Papers/06_tables.py" > "Audits of Other Papers/results/tables.md"
"""
from __future__ import annotations
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

ORDER = ["P-patch-random-overlap", "P-tile-random", "P-slice-random",
         "P-zone-nobuffer", "P-zone-buffer", "P-zone-buffer-straddler"]
NAME = {
    "P-patch-random-overlap": "random overlapping patches",
    "P-tile-random": "random tiles",
    "P-slice-random": "random slices",
    "P-zone-nobuffer": "contiguous zones, no buffer",
    "P-zone-buffer": "zones + buffer 12",
    "P-zone-buffer-straddler": "zones + buffer + straddler (reference)",
}
ARCHS = ["unet", "unetpp", "deeplabv3p"]


def load(n):
    p = os.path.join(RES, n)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def f(v, nd=4):
    return "n/a" if v is None or v != v else f"{v:.{nd}f}"


def pct(v, nd=1):
    return "n/a" if v is None or v != v else f"{100 * v:.{nd}f}%"


def table_diagnostics(d):
    print("### Measured leakage per protocol (no model trained)\n")
    print("| Protocol | Chondrules in test also in train | Test positive pixels "
          "from those chondrules | Min train-to-test gap |")
    print("| --- | --- | --- | --- |")
    for p in ORDER:
        e = d["protocols"].get(p)
        if not e:
            continue
        n_sh = e["shared_instances_train_and_test"]
        n_te = e["n_instances_in_test"]
        gap = (f"{e['min_z_gap']} band slices" if e["kind"] == "slice"
               else f"{e['min_z_gap']} band slices (tiles)")
        print(f"| {NAME[p]} | {n_sh} of {n_te} "
              f"({pct(e['shared_instance_frac_of_test'], 0)}) | "
              f"{pct(e['test_pos_frac_from_shared_chondrules'])} | {gap} |")
    print()


def table_protocols(c):
    cells = {(x["protocol"], x["arch"]): x for x in c["cells"]}
    print(f"### Test IoU by protocol and architecture "
          f"(identical {c['budget_steps']}-step budget)\n")
    hdr = "| Protocol | " + " | ".join(ARCHS) + " | otsu control | best gain over control |"
    print(hdr)
    print("| --- |" + " --- |" * (len(ARCHS) + 2))
    for p in ORDER:
        o = cells.get((p, "otsu"))
        vals = []
        for a in ARCHS:
            x = cells.get((p, a))
            vals.append(f(x["test_iou"]) if x else "pending")
        ov = o["test_iou"] if o else float("nan")
        learned = [cells[(p, a)]["test_iou"] for a in ARCHS
                   if (p, a) in cells and cells[(p, a)]["test_iou"] ==
                   cells[(p, a)]["test_iou"]]
        gain = (max(learned) - ov) if learned and ov == ov else float("nan")
        print(f"| {NAME[p]} | " + " | ".join(vals) +
              f" | {f(ov)} | {f(gain)} |")
    print()
    nf = load("noise_floors.json")
    if nf:
        oc = nf["otsu_constancy"]
        lo, hi = oc["cut_range"]
        op = oc["operating_point_range"]
        opstr = (f"identical across every protocol ({op[0]:.2f})"
                 if oc["operating_point_is_constant"]
                 else f"ranges {op[0]:.2f} to {op[1]:.2f}")
        print(f"The otsu column is a fixed global-intensity rule with two fitted "
              f"parameters. Measured (results/noise_floors.json), its val-selected "
              f"operating point is {opstr} and its fitted cut varies only from "
              f"{lo:.3f} to {hi:.3f}, so its spread across test sets indexes "
              f"test-set difficulty for a per-pixel rule. It is a sanity check, "
              f"not a subtraction, and it is blind to CNN-specific differences "
              f"(spec C5).\n")


def table_full_metrics(c):
    """Constitution principle 5: IoU is never reported alone.

    Under ~95% background, IoU on its own hides which way a model is failing.
    Precision and recall separate over-prediction from misses, and AUPRC is
    threshold-free so it survives a badly chosen operating point.
    """
    cells = {(x["protocol"], x["arch"]): x for x in c["cells"]}
    print("### Full metric set for the reference architecture (U-Net)\n")
    print("| Protocol | IoU | Dice | precision | recall | AUPRC |")
    print("| --- | --- | --- | --- | --- | --- |")
    for p in ORDER:
        x = cells.get((p, "unet"))
        if not x:
            continue
        print(f"| {NAME[p]} | {f(x.get('test_iou'))} | {f(x.get('test_dice'))} | "
              f"{f(x.get('test_precision'))} | {f(x.get('test_recall'))} | "
              f"{f(x.get('test_auprc'))} |")
    print()


def table_val_test_gap(c):
    """Validation-to-test exchangeability, a leakage channel distinct from
    training memorization.

    A random split makes val and test statistically interchangeable, so a paper
    that reports validation performance reports roughly its test performance. A
    contiguous split does not, because the val zone is a different piece of rock
    from the test zone. The size of this gap is how much a paper would overstate
    its result by quoting validation, with no training leakage involved at all.
    """
    cells = [x for x in c["cells"]
             if x["arch"] != "otsu" and x.get("val_iou") == x.get("val_iou")
             and x.get("test_iou") == x.get("test_iou")]
    if not cells:
        return
    print("### Validation-to-test exchangeability by protocol\n")
    print("| Protocol | mean val IoU | mean test IoU | val minus test |")
    print("| --- | --- | --- | --- |")
    for p in ORDER:
        rows = [x for x in cells if x["protocol"] == p]
        if not rows:
            continue
        v = float(np.mean([x["val_iou"] for x in rows]))
        t = float(np.mean([x["test_iou"] for x in rows]))
        print(f"| {NAME[p]} | {f(v)} | {f(t)} | {f(v - t, 4)} |")
    print()
    print("Under a random split val and test are drawn from the same pool, so "
          "they agree and quoting either is equivalent. Under a contiguous "
          "split they are different regions of one meteorite and they do not. A "
          "paper using a contiguous split that reports validation performance "
          "overstates its held-out result by this column, before any training "
          "leakage is considered. This channel is independent of Finding 5: it "
          "is a gap measured WITHIN one protocol, so it does not depend on the "
          "between-protocol difference surviving replication.\n")


def table_train_composition(c):
    cells = {(x["protocol"], x["arch"]): x for x in c["cells"]}
    rows = [(p, cells[(p, "unet")]) for p in ORDER
            if (p, "unet") in cells and "train_pos_frac" in cells[(p, "unet")]]
    missing = [p for p in ORDER
               if (p, "unet") in cells and "train_pos_frac" not in cells[(p, "unet")]]
    if not rows:
        return
    print("### Declared confound: training foreground differs by protocol\n")
    print("| Protocol | train positive fraction | test positive fraction |")
    print("| --- | --- | --- |")
    for p, x in rows:
        print(f"| {NAME[p]} | {f(x.get('train_pos_frac'))} | "
              f"{f(x.get('test_pos_frac'))} |")
    print()
    if missing:
        print("NOT RECORDED for " + ", ".join(NAME[p] for p in missing) +
              ": those cells were produced by an earlier revision of the runner "
              "that did not record this covariate, and the result cache was not "
              "invalidated. Any claim about training foreground across ALL "
              "protocols is unmeasured for them.\n")


def table_dose(d):
    print("### Buffer sweep (NOT confound-free, see the covariate column)\n")
    print("| Buffer (band slices) | Physical gap (original slices) | test IoU | "
          "train positive fraction |")
    print("| --- | --- | --- | --- |")
    for p in sorted(d["points"], key=lambda r: r["buffer"]):
        print(f"| {p['buffer']} | {p['min_gap_original_slices']} | "
              f"{f(p.get('test_iou'))} | {f(p.get('train_pos_frac'))} |")
    print()
    print("Training foreground falls monotonically with the buffer, collinear "
          "with the treatment, so this sweep cannot separate proximity leakage "
          "from a decline in training signal. The causal reading is withdrawn.\n")


def table_replication(r):
    print("### Replicated headline pair\n")
    print("| Arm | Variant | test IoU | otsu | gain | train positive fraction |")
    print("| --- | --- | --- | --- | --- | --- |")
    for x in sorted(r["runs"], key=lambda z: (z["arm"], z["variant"])):
        lab = ("split seed" if x["arm"].startswith("leaky") else "window offset")
        print(f"| {x['arm']} | {lab} {x['variant']} | {f(x.get('test_iou'))} | "
              f"{f(x.get('otsu_test_iou'))} | {f(x.get('gain_over_otsu'))} | "
              f"{f(x.get('train_pos_frac'))} |")
    print()
    s = r.get("summary", {}).get("test_iou")
    if s:
        print(f"Leaky arm: mean {f(s['leaky']['mean'])} "
              f"(sd {f(s['leaky']['sd'])}, n={s['leaky']['n']}). "
              f"Leakage-free arm: mean {f(s['clean']['mean'])} "
              f"(sd {f(s['clean']['sd'])}, n={s['clean']['n']}).\n")
        lo, hi = s["difference_ci95"]
        print(f"**Difference between the random-slice and leakage-free arms: "
              f"{s['difference']:+.4f} IoU, 95% CI [{lo:+.4f}, {hi:+.4f}].**\n")
        if lo <= 0 <= hi:
            print("The interval spans zero. On this volume, at this budget, the "
                  "protocol difference for the matched pair is NOT separable "
                  "from run-to-run and window-to-window variation. State the "
                  "direction, not a magnitude.\n")
    g = r.get("summary", {}).get("gain_over_otsu")
    if g:
        lo, hi = g["difference_ci95"]
        print(f"Same comparison in gain-over-control terms: "
              f"{g['difference']:+.4f} IoU, 95% CI [{lo:+.4f}, {hi:+.4f}].\n")


if __name__ == "__main__":
    print("<!-- generated by 06_tables.py; do not edit by hand -->\n")
    d, c, dr, rp = (load("leakage_diagnostics.json"),
                    load("protocol_comparison.json"),
                    load("dose_response.json"),
                    load("replication.json"))
    if d:
        table_diagnostics(d)
    if c:
        table_protocols(c)
        table_full_metrics(c)
        table_val_test_gap(c)
        table_train_composition(c)
    if dr:
        table_dose(dr)
    if rp:
        table_replication(rp)
