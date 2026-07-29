"""Tier C — CNN training/eval on the frozen split (spec R10, first pass).

Trains the k=1 U-Net family (2D dice / recall-favoring tversky / focal /
attention) on TRAIN, selects the operating threshold on VAL, and scores once on
TEST — same leakage-free discipline as Tier A/B. The 2.5D (k=5) model is
DEFERRED pending its train-seam fix (reviewer finding #2). Focal alpha is set to
up-weight the rare foreground (reviewer finding #4). Writes (incrementally, so
partial progress survives) results/v2_cnn_leaderboard.json.

Run:  python -m src.run_v2_cnn      (long-running; uses Apple-MPS)
"""
from __future__ import annotations
import functools
import json
import os
import time
from . import sweep, metrics, models_cnn as mc

EPOCHS = 15
STEPS = 400


def build_runners():
    def unet(in_ch):
        return mc.UNet2D(in_ch=in_ch, base=32, depth=4)

    def attn(in_ch):
        return mc.AttentionUNet2D(in_ch=in_ch, base=32)

    focal_fg = functools.partial(mc.focal_loss, alpha=0.75)  # up-weight rare foreground

    def mk(name, net, loss):
        return mc.CNNRunner(name, net, loss, k=1, epochs=EPOCHS, steps_per_epoch=STEPS)

    return [
        mk("unet2d_dice", unet, mc.dice_loss),
        mk("unet2d_tversky", unet, mc.tversky_loss),      # beta>alpha favors recall
        mk("unet2d_focal", unet, focal_fg),
        mk("attention_unet_dice", attn, mc.dice_loss),
    ]


def main():
    t0 = time.time()
    _, (Rtr, Mtr, Vtr), (Rv, Mv, Vv), (Rt, Mt, Vt) = sweep.load_split_zones(".")
    print(f"device={mc.DEVICE}  zones train {Rtr.shape[0]}/val {Rv.shape[0]}/test {Rt.shape[0]}"
          f"  epochs={EPOCHS} steps={STEPS}", flush=True)
    train = {"raw": Rtr, "mask": Mtr, "valid": Vtr}
    val = {"raw": Rv, "mask": Mv, "valid": Vv}

    board = []
    for r in build_runners():
        try:
            ts = time.time()
            r.fit(train, val)
            prob_t = r.predict_proba(Rt)
            thr = r.best_thr
            m = metrics.metrics_from_pred(prob_t >= thr, Mt, Vt)
            m.update({"name": r.name, "tier": "C_cnn", "threshold": float(thr),
                      "auprc": metrics.auprc(prob_t, Mt, Vt),
                      "sec": round(time.time() - ts, 1)})
            board.append(m)
            print(f"[C] {r.name}: IoU={m['iou']:.4f} Dice={m['dice']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} thr={thr:.2f} "
                  f"({m['sec']}s)", flush=True)
        except Exception as e:  # keep going; one model failing shouldn't lose the rest
            print(f"[C] {r.name} FAILED: {type(e).__name__}: {e}", flush=True)
        os.makedirs("results", exist_ok=True)
        with open("results/v2_cnn_leaderboard.json", "w") as f:
            json.dump({"regime": f"leakage-free z-slab split; threshold tuned on val, test "
                                 f"scored once; foreground=chondrule; N=1 (ALH 83010); "
                                 f"MPS, epochs={EPOCHS} (compute-limited, not tuned SOTA); "
                                 f"2.5D deferred pending train-seam fix",
                       "device": mc.DEVICE,
                       "leaderboard": sorted(board, key=lambda d: d["iou"], reverse=True),
                       "runtime_sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
