"""The comparison Sky asked for on 2026-07-30 and that had never been run.

Constitution section 12 leaves stage 3 UNSETTLED: whether the classical map is an
INPUT CHANNEL to the CNN (the Tier E cascade pattern) or an alternative stage-2
candidate. Offered the choice, Sky declined both and chose "both, and let
validation decide". Neither arm was ever built. This runs them.

    control   CNN on raw only
    cleaned   CNN on raw + the three cleaned channels (denoised, CLAHE, top-hat)
    cascade   CNN on raw + the CROSS-FITTED classical probability
    full      CNN on raw + cleaned channels + classical probability, together

[CORRECTED 2026-08-06] The first three arms treat the two classical products as
ALTERNATIVES, each measured against the control. Sky corrected that framing: the
intended architecture is a CHAIN, in which the traditional pipeline's outputs
feed the CNN together and the CNN then feeds the confidence mechanisms. The
`full` arm is that composition. `cleaned` and `cascade` remain useful because
they isolate which half of the input carries whatever the composition gains.

[LEAKAGE][MUST] The cascade channel is `results/probs/classical_xf_*.npy`, which
is out-of-fold over the train slices. A base model's predictions on its own
training data are memorised, and feeding those to the refiner teaches it to
over-trust a channel that is weaker at inference. `cascade.classical_aux_lookup`
bridges those maps to the aux input by content hash.

WHY A CONTROL ARM. Every existing 7,200-step number is single-seed and pre-tiler
-fix, so there was nothing valid to compare two new arms against. The control is
the same architecture, budget and seeds with no aux channel.

WHY THREE SEEDS. Measured seed SD on this project is 0.0084, and the val-versus
-test analysis puts the smallest resolvable val gap at roughly 0.15. A one-seed
comparison of these arms could not have answered the question it was set up to
ask. Three seeds per arm gives each a spread, so a difference above roughly
0.024 is separable from run-to-run scatter.

[HONESTY] The budget is 7,200 steps, not the frozen 30,000, so these numbers are
NOT comparable to `unet2d_s30000` (0.29409). They are comparable to each other,
which is the whole point. `leaderboard=False` keeps them off the frozen-split
roster for exactly that reason.

Resumable: completed (arm, seed) pairs are skipped.

Run:  python -m src.run_cascade_arms
      python -m src.run_cascade_arms --arm=cascade --seed=0
      python -m src.run_cascade_arms --report   # reprint only, no data load
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

from . import cascade, classical_pipeline, evaluate
from .models_cnn import CNNRunner, UNet2D, dice_loss, tversky_loss

OUT = "results/cascade_arms.json"
ARMS = ("control", "cleaned", "cascade", "full", "fp_penalty")
SEEDS = (0, 1, 2)
EPOCHS, STEPS_PER_EPOCH = 24, 300        # 7,200 steps, matching data_scaling
STEPS = EPOCHS * STEPS_PER_EPOCH


def cleaned_lookup(sl):
    """(H,W) raw -> (3,H,W) cleaned channels: denoised, CLAHE, top-hat.

    `classical_pipeline.channels` returns (H,W,4) with raw at index 0. The raw
    channel is dropped here because `CNNRunner` already feeds raw as its own
    input; duplicating it would waste a channel and change the normalisation the
    net sees on the copy.
    """
    ch = classical_pipeline.channels(sl)
    return np.ascontiguousarray(np.transpose(ch[..., 1:], (2, 0, 1)))


def full_lookup(d):
    """(H,W) raw -> (4,H,W): the cleaned channels AND the classical probability.

    This is the chain as Sky describes it, rather than the either/or the earlier
    arms tested: the traditional pipeline's outputs feed the CNN together, and
    the CNN then feeds the confidence stages. `cleaned` and `cascade` isolate the
    two halves; this arm is the composition.
    """
    clas = cascade.classical_aux_lookup(
        {z: d[z]["raw"] for z in ("train", "val", "test")})

    def lookup(sl):
        ch = classical_pipeline.channels(sl)               # (H,W,4)
        cleaned = np.transpose(ch[..., 1:], (2, 0, 1))     # denoise/CLAHE/tophat
        return np.ascontiguousarray(
            np.concatenate([cleaned, clas(sl)[None]], axis=0))

    return lookup


def arm_loss(arm):
    """The loss for one arm. Everything except `fp_penalty` trains on dice.

    `fp_penalty` is Sky's mentor's proposal: penalise false positives harder.
    Tversky weights false positives by alpha and false negatives by beta, and
    the recall bake-off only ever ran beta > alpha (0.3/0.7 and 0.1/0.9). Those
    arms pushed recall from 0.556 to 0.738 and dropped IoU from 0.231 to 0.153,
    so the recall direction is measured to hurt. alpha > beta was never tried.

    0.7/0.3 is the exact mirror of the tried 0.3/0.7, and it roughly offsets the
    imbalance actually measured on the base model: 5.8M false positives against
    2.4M false negatives, precision 0.372 against recall 0.584.

    No separate symmetric control is needed. Tversky at alpha = beta = 0.5 IS
    Dice, so the existing `control` arm is already the matched baseline.
    """
    if arm == "fp_penalty":
        return lambda lo, t, v: tversky_loss(lo, t, v, alpha=0.7, beta=0.3)
    return dice_loss


def build_lookup(arm, d):
    """(aux_lookup, aux_ch) for one arm. `None` aux means the control."""
    if arm in ("control", "fp_penalty"):
        return None, 1
    if arm == "cleaned":
        return cleaned_lookup, 3
    if arm == "cascade":
        raw_by_zone = {z: d[z]["raw"] for z in ("train", "val", "test")}
        return cascade.classical_aux_lookup(raw_by_zone), 1
    if arm == "full":
        return full_lookup(d), 4
    raise SystemExit(f"unknown arm {arm!r}")


def load_done(path=OUT):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("runs", [])


def save(runs, path=OUT):
    # Grouped by (arm, BUDGET). Grouping by arm alone pooled the 7,200-step and
    # 30,000-step runs of the same arm into one mean, which is averaging two
    # incomparable regimes and reporting it as a single number. Caught after the
    # 30k composed run landed and was silently averaged with its 7.2k namesake.
    by_arm = {}
    for r in runs:
        if r.get("test_iou") == r.get("test_iou"):
            by_arm.setdefault((r["arm"], r.get("steps", 7200)), []).append(r)
    summary = []
    for a, budget in sorted(by_arm, key=lambda k: (ARMS.index(k[0]), k[1])):
        rs = by_arm[(a, budget)]
        if not rs:
            continue
        v = np.array([r["val_iou"] for r in rs], float)
        t = np.array([r["test_iou"] for r in rs], float)
        summary.append({
            "arm": a, "steps": int(budget), "n_seeds": len(rs),
            "val_iou_mean": float(v.mean()),
            "val_iou_sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "test_iou_mean": float(t.mean()),
            "test_iou_sd": float(t.std(ddof=1)) if t.size > 1 else float("nan"),
            "val_iou_all": [float(x) for x in v],
            "test_iou_all": [float(x) for x in t]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "regime": (
                f"UNet2D base 32 depth 4, {STEPS} steps, lr 3e-4, pos_frac 0.5, "
                "no augmentation; frozen split; threshold selected on VAL and "
                "frozen for a single TEST scoring; 3 seeds per arm. The cascade "
                "channel is the OUT-OF-FOLD classical_xf map. Arms are "
                "comparable to EACH OTHER only: the budget is 7,200 steps, not "
                "the frozen 30,000, so these are not comparable to "
                "unet2d_s30000 (0.29409)."),
            "decision_rule": (
                "Sky, 2026-07-30: build both arms and let VALIDATION decide. "
                "Test is reported for completeness and does not select."),
            "summary": summary, "runs": runs}, f, indent=2)
    return summary


def _bar(a, b, axis):
    """2-sigma separability bar between two arms on one axis.

    Returns nan when either arm has fewer than two seeds, because there is no
    spread to pool and therefore no bar. Callers must treat nan as "no verdict",
    never as a wide bar that everything fits inside.
    """
    sa, sb = a[f"{axis}_iou_sd"], b[f"{axis}_iou_sd"]
    if sa != sa or sb != sb:
        return float("nan")
    return 2 * np.sqrt((sa ** 2 + sb ** 2) / 2) * np.sqrt(2)


def _verdict(gap, bar):
    if bar != bar:
        return "no verdict (n=1)"
    return "RESOLVED" if abs(gap) > bar else "inside the noise"


def report(summary):
    """Print the arm comparison. Gaps are formed WITHIN a budget, never across.

    [CORRECTED 2026-08-07] This replaces a printout that rendered two non-results
    as results, both found while giving the `full` arm its missing seeds:

    1. The control was selected by arm NAME alone, so every row was differenced
       against whichever control appeared first regardless of budget. The
       30,000-step `full` row printed as "+0.1291 against the control" where the
       control was the 7,200-step one — exactly the cross-budget comparison this
       module's own `regime` string forbids, and the same defect `save()` was
       already fixed for.
    2. A single-seed row has no SD, so its bar came out `nan`, and `abs(gap) >
       nan` is False, so the verdict fell through to "inside the noise". That
       asserts a fact about noise where n=1 supports no verdict at all.

    Test gaps print beside the val gaps because nine val/test sign flips are on
    record here and `results/val_vs_test.json` puts the smallest val gap that
    predicts anything at 0.137 to 0.172 IoU. Seeing both at once is what stops a
    resolved val win being read as a result: `full` is +0.0962 RESOLVED on val
    and -0.0017 on test. It does NOT change the decision rule. Validation
    selects; the test column is reported and never selects.
    """
    for budget in sorted({s["steps"] for s in summary}):
        rows = [s for s in summary if s["steps"] == budget]
        ns = {s["n_seeds"] for s in rows}
        if len(ns) == 1:
            n = ns.pop()
            head = f"{n} seed{'s' if n != 1 else ''} each"
        else:
            head = "seed counts DIFFER per arm, see the n column"
        print(f"\n=== stage-3 arm comparison ({head}, {budget:,} steps) ===")
        print(f"  {'arm':<12}{'n':>3}{'val mean':>10}{'val SD':>9}"
              f"{'test mean':>11}{'test SD':>9}")
        for s in rows:
            vsd = ("  n/a" if s["val_iou_sd"] != s["val_iou_sd"]
                   else f"{s['val_iou_sd']:.4f}")
            tsd = ("  n/a" if s["test_iou_sd"] != s["test_iou_sd"]
                   else f"{s['test_iou_sd']:.4f}")
            print(f"  {s['arm']:<12}{s['n_seeds']:>3}{s['val_iou_mean']:>10.4f}"
                  f"{vsd:>9}{s['test_iou_mean']:>11.4f}{tsd:>9}")

        ctrl = next((s for s in rows if s["arm"] == "control"), None)
        if ctrl is None:
            print(f"  no control arm at {budget:,} steps, so no gap is formed. "
                  "A control from a different budget is NOT a valid comparison.")
            continue
        print("\n  against the control AT THIS BUDGET. Validation decides "
              "(Sky, 2026-07-30);\n  the test column is reported and does not "
              "select:")
        print(f"    {'arm':<12}{'val gap':>9}{'bar':>8}  {'':<18}"
              f"{'test gap':>9}{'bar':>8}")
        for s in rows:
            if s["arm"] == "control":
                continue
            line = f"    {s['arm']:<12}"
            for axis in ("val", "test"):
                gap = s[f"{axis}_iou_mean"] - ctrl[f"{axis}_iou_mean"]
                bar = _bar(s, ctrl, axis)
                bs = "   n/a" if bar != bar else f"{bar:.4f}"
                line += f"{gap:>+9.4f}{bs:>8}  {_verdict(gap, bar):<18}"
            print(line)


def run(root=".", arms=None, seeds=None):
    d = evaluate.load(root)
    runs = load_done()
    # Keyed on the BUDGET too. Keying on (arm, seed) alone would treat a
    # 30,000-step run of an arm already done at 7,200 as complete and skip it.
    done = {(r["arm"], r["seed"], r.get("steps", 7200)) for r in runs}
    # SEED-MAJOR, deliberately. Arm-major would finish all three seeds of the
    # first arm before the third arm started, so the three-way comparison the
    # run exists to make would not exist until the last run landed. Seed-major
    # gives a complete one-seed comparison after three runs, and each later seed
    # tightens it. The cached lookups are the only cost: they are rebuilt once
    # per seed rather than once per arm, which is 2 s for the cascade hash table
    # against a 40 min training run.
    cache = {}
    for seed in (seeds or SEEDS):
        for arm in (arms or ARMS):
            if (arm, seed, STEPS) in done:
                print(f"skip {arm} seed={seed} at {STEPS} steps (already done)",
                      flush=True)
                continue
            if arm not in cache:
                cache[arm] = build_lookup(arm, d)
            lookup, aux_ch = cache[arm]
            # The budget goes in the NAME whenever it is not the 7,200-step
            # default. Without this, a 30,000-step re-run of the same arm and
            # seed would overwrite the 7,200-step run's probability maps, its
            # checkpoint, and its row in cascade_arms.json, silently destroying
            # the comparison this module exists to make.
            name = (f"arm_{arm}_s{seed}" if STEPS == 7200
                    else f"arm_{arm}_s{seed}_b{STEPS}")
            print(f"\n=== {name}: {STEPS} steps, aux_ch="
                  f"{aux_ch if lookup else 0} ===", flush=True)
            runner = CNNRunner(
                name, lambda ic: UNet2D(in_ch=ic, base=32, depth=4), arm_loss(arm),
                epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch=16,
                patch=128, pos_frac=0.5, lr=3e-4, seed=seed,
                aux_lookup=lookup, aux_ch=aux_ch,
                ckpt_dir="results/checkpoints")
            # [MUST] Persist BOTH the probability maps and the weights. Without
            # save_prob the confidence stages (4-6) and the prune (7) can never
            # be applied to this arm, because they consume a probability map;
            # without ckpt_dir the trained net dies with the process and the arm
            # has to be retrained to look at it again. An earlier version of this
            # call passed neither, which would have produced a scalar IoU and
            # nothing else after six hours of compute.
            row = evaluate.run_and_score(
                runner, data=d,
                extra={"note": f"stage-3 arm comparison: {arm}, seed {seed}"},
                save_prob=f"results/probs/{name}.npy",
                leaderboard=False)
            runs.append({
                "arm": arm, "seed": seed, "steps": STEPS,
                "aux_ch": int(aux_ch) if lookup else 0,
                "test_iou": float(row.get("iou", float("nan"))),
                "val_iou": float(getattr(runner, "best_full_val_iou",
                                         float("nan"))),
                "threshold": float(row.get("threshold", float("nan"))),
                "error": row.get("error")})
            save(runs)
            print(f"    -> val {runs[-1]['val_iou']:.4f}  "
                  f"test {runs[-1]['test_iou']:.4f}", flush=True)

    summary = save(runs)
    report(summary)
    return summary


def main(argv=None):
    global EPOCHS, STEPS_PER_EPOCH, STEPS
    argv = list(sys.argv[1:] if argv is None else argv)
    arms = seeds = None
    report_only = False
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "arm":
            arms = tuple(v.split(","))
        elif k == "seed":
            seeds = tuple(int(x) for x in v.split(","))
        elif k == "epochs":
            EPOCHS = int(v)
        elif k == "steps_per_epoch":
            STEPS_PER_EPOCH = int(v)
        elif k == "report":
            report_only = True
        else:
            raise SystemExit(f"unknown option {a}")
    # Regenerating the printed summary must not cost a data load. `run()` opens
    # with evaluate.load, several GB across three zones, which is absurd when
    # every run is already on disk and only the arithmetic was wrong. Rebuilding
    # the summary from `runs` also re-applies the current (arm, budget) grouping,
    # so a file written under the old pooling is corrected by reading it.
    if report_only:
        summary = save(load_done())
        report(summary)
        return summary
    STEPS = EPOCHS * STEPS_PER_EPOCH
    for a in arms or ():
        if a not in ARMS:
            raise SystemExit(f"--arm={a} is not one of {ARMS}")
    for s in seeds or ():
        if s not in SEEDS:
            raise SystemExit(f"--seed={s} is not one of {SEEDS}")
    return run(arms=arms, seeds=seeds)


if __name__ == "__main__":
    main()
