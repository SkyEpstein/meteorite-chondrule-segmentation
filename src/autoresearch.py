"""Karpathy-style autoresearch loop for the chondrule segmenter.

Recipe, in order — each phase is worthless until the previous one passes:

* **Phase 0 — sanity / overfit-a-tiny-batch.** If a U-Net cannot drive IoU
  toward ~1.0 on a handful of FIXED patches it can see over and over, that is a
  *plumbing bug* (loss, valid-mask, LR, architecture), not a tuning problem.
  Diagnosing that is far cheaper than another long training run.
* **Phase 1 — coarse search at short budgets** over the knobs that actually
  matter here (LR, loss, patch size, positive-sampling fraction, width).
* **Phase 2 — successive halving:** promote only the survivors to longer runs.
* **Phase 3 — train the winner long, then score the test slab ONCE.**

Leakage discipline is unchanged: every selection decision uses TRAIN/VAL only;
the test slab is sealed until the final locked model. [LEAKAGE][MUST]

Run Phase 0:  python -m src.autoresearch sanity
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np
import torch

from . import metrics, sweep, zones
from .models_cnn import (DEVICE, UNet2D, AttentionUNet2D, PatchSet,
                         dice_loss, tversky_loss, focal_loss, bce_loss)


# ---------------------------------------------------------------------------
# Phase 0 — can the net overfit a tiny fixed batch?
# ---------------------------------------------------------------------------
def overfit_batch(make_model, loss_fn, train, n_patches=4, steps=400, lr=1e-3,
                  patch=128, seed=0, log_every=100):
    """Train on ONE fixed batch of chondrule-containing patches; return history.

    pos_frac=1.0 so every patch actually contains chondrules (an all-background
    patch would "pass" trivially/undefined). A healthy setup drives train IoU on
    these same patches toward ~1.0.
    """
    torch.manual_seed(seed)
    ds = PatchSet(train["raw"], train["mask"], train["valid"], patch=patch,
                  pos_frac=1.0, length=n_patches, seed=seed, k=1)
    xs, ys, vs = [], [], []
    for _ in range(n_patches):
        x, y, v = ds.sample()
        xs.append(x); ys.append(y); vs.append(v)
    xb, yb, vb = (torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE),
                  torch.stack(vs).to(DEVICE))
    pos_frac = float((yb * vb).sum() / vb.sum().clamp(min=1))

    model = make_model(1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for i in range(steps):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb, vb)
        loss.backward()
        opt.step()
        if i == 0 or (i + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(xb)).cpu().numpy()
            gt = yb.cpu().numpy() > 0.5
            vd = vb.cpu().numpy() > 0.5
            hist.append({"step": i + 1, "loss": float(loss.item()),
                         "train_iou": float(metrics.iou(p >= 0.5, gt, vd))})
    return {"batch_pos_frac": pos_frac, "history": hist,
            "final_train_iou": hist[-1]["train_iou"] if hist else float("nan")}


def _mask_sanity():
    """The valid-mask must actually neutralize pixels: a fully-invalid patch
    contributes zero loss, and masking a planted error must change the loss."""
    torch.manual_seed(0)
    logits = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :4, :] = 1.0
    allvalid = torch.ones_like(target)
    novalid = torch.zeros_like(target)
    out = {}
    for nm, fn in (("bce", bce_loss), ("dice", dice_loss),
                   ("tversky", tversky_loss), ("focal", focal_loss)):
        out[nm] = {"all_valid": float(fn(logits, target, allvalid)),
                   "none_valid": float(fn(logits, target, novalid))}
    return out


def phase0(root="."):
    t0 = time.time()
    print(f"device={DEVICE}", flush=True)
    print("--- valid-mask sanity (none_valid should neutralize the loss) ---",
          flush=True)
    ms = _mask_sanity()
    for k, v in ms.items():
        print(f"  {k:8s} all_valid={v['all_valid']:.4f}  none_valid={v['none_valid']:.4f}",
              flush=True)

    _, (Rtr, Mtr, Vtr), _, _ = sweep.load_split_zones(root)
    train = {"raw": Rtr, "mask": Mtr, "valid": Vtr}
    print(f"train slices {Rtr.shape[0]} loaded ({time.time()-t0:.0f}s)", flush=True)

    def unet(in_ch):
        return UNet2D(in_ch=in_ch, base=32, depth=4)

    results = {"mask_sanity": ms, "overfit": {}}
    for nm, fn, lr in (("dice", dice_loss, 1e-3),
                       ("dice_lr3e-4", dice_loss, 3e-4),
                       ("bce", bce_loss, 1e-3),
                       ("tversky", tversky_loss, 1e-3)):
        r = overfit_batch(unet, fn, train, lr=lr)
        results["overfit"][nm] = r
        curve = " -> ".join(f"{h['train_iou']:.3f}" for h in r["history"])
        print(f"  [{nm:12s} lr={lr:.0e}] pos={r['batch_pos_frac']:.3f} "
              f"IoU: {curve}   FINAL={r['final_train_iou']:.3f}", flush=True)

    with open("results/autoresearch_phase0.json", "w") as f:
        json.dump(results, f, indent=2)
    best = max(v["final_train_iou"] for v in results["overfit"].values())
    print(f"\nPHASE0 best overfit train IoU = {best:.3f} "
          f"({'PASS - plumbing OK' if best > 0.85 else 'FAIL - likely a bug'})",
          flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Phase 1 — coarse search.
#
# Phase 0 proved the net overfits a fixed batch to IoU 1.000, so capacity and
# step count are NOT the limitation: the gap is generalization. The search
# space therefore ablates the knobs that fight memorization — augmentation,
# weight decay, and model width — rather than sweeping capacity/steps.
# Ranking is by FULL-validation IoU. The test slab is never touched. [LEAKAGE]
# ---------------------------------------------------------------------------
P1_EPOCHS, P1_STEPS, P1_VAL_STRIDE = 6, 250, 6

LOSSES = {"dice": dice_loss, "tversky": tversky_loss,
          "bce": bce_loss, "focal": focal_loss}


def phase1_configs():
    base = dict(loss="dice", lr=3e-4, base=32, wd=0.0, aug=False, pos_frac=0.5)

    def c(**kw):
        d = dict(base)
        d.update(kw)
        return d

    return [
        c(name="p1_ctrl_b32"),                                   # control
        c(name="p1_aug_b32", aug=True),                          # +augmentation
        c(name="p1_wd_b32", wd=1e-4),                            # +weight decay
        c(name="p1_aug_wd_b32", aug=True, wd=1e-4),              # both
        c(name="p1_ctrl_b16", base=16),                          # smaller net
        c(name="p1_aug_b16", base=16, aug=True),
        c(name="p1_aug_wd_b16", base=16, aug=True, wd=1e-4),
        c(name="p1_aug_wd_tversky", aug=True, wd=1e-4, loss="tversky"),
        # Round 2: round 1 showed the FULL augmentation bundle halved val IoU
        # (0.387 -> 0.202). Intensity is the discriminative signal here, so the
        # jitter is the prime suspect — these isolate geometry-only.
        c(name="p1_geom_b32", aug="geom"),                       # flips/rot only
        c(name="p1_geom_wd_b32", aug="geom", wd=1e-4),
        c(name="p1_ctrl_wd_b32", wd=1e-4),                       # control + wd
        c(name="p1_ctrl_tversky", loss="tversky"),               # recall-favoring
    ]


def run_config(idx, root="."):
    """Train ONE Phase-1 config and record its full-validation score."""
    import os
    from .models_cnn import CNNRunner
    cfg = phase1_configs()[idx]

    def mk(in_ch):
        return UNet2D(in_ch=in_ch, base=cfg["base"], depth=4)

    r = CNNRunner(cfg["name"], mk, LOSSES[cfg["loss"]], k=1,
                  epochs=P1_EPOCHS, steps_per_epoch=P1_STEPS,
                  lr=cfg["lr"], weight_decay=cfg["wd"], augment=cfg["aug"],
                  pos_frac=cfg["pos_frac"], val_stride=P1_VAL_STRIDE)

    # Cached zones: the split is built ONCE by `python -m src.zones`, and only
    # train+val are loaded here. The test zone is never even read during a
    # search — cheaper in RAM and enforces touch-test-once. [LEAKAGE]
    z = zones.load_zones(root, zones=("train", "val"))
    (Rtr, Mtr, Vtr), (Rv, Mv, Vv) = z["train"], z["val"]
    t0 = time.time()
    r.fit({"raw": Rtr, "mask": Mtr, "valid": Vtr},
          {"raw": Rv, "mask": Mv, "valid": Vv})
    out = dict(cfg)
    out.update({"val_iou": r.best_full_val_iou,
                "threshold": float(r.best_thr),
                "epochs": P1_EPOCHS, "steps": P1_STEPS,
                "sec": round(time.time() - t0, 1)})
    os.makedirs("results/phase1", exist_ok=True)
    with open(f"results/phase1/{cfg['name']}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"PHASE1 {cfg['name']} val_iou={out['val_iou']:.4f} "
          f"thr={out['threshold']:.2f} ({out['sec']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase 3 — train a chosen config at a LONG budget and score the test slab once.
# ---------------------------------------------------------------------------
P3_EPOCHS, P3_STEPS = 18, 400          # 7200 steps ~ 4.8x the search budget


def phase3(idx, root="."):
    """Long-budget run of one Phase-1 config, scored on test exactly once.

    Selection already happened on validation; this trains the chosen recipe for
    longer and reports it. Also the honest test of whether augmentation only
    looked bad because the search budget was too short to amortize it.
    """
    from . import evaluate
    from .models_cnn import CNNRunner
    cfg = phase1_configs()[idx]

    def mk(in_ch):
        return UNet2D(in_ch=in_ch, base=cfg["base"], depth=4)

    r = CNNRunner(f"long_{cfg['name']}", mk, LOSSES[cfg["loss"]], k=1,
                  epochs=P3_EPOCHS, steps_per_epoch=P3_STEPS,
                  lr=cfg["lr"], weight_decay=cfg["wd"], augment=cfg["aug"],
                  pos_frac=cfg["pos_frac"], val_stride=6)
    print(f"=== PHASE3 long run: {cfg['name']} "
          f"({P3_EPOCHS}x{P3_STEPS}={P3_EPOCHS * P3_STEPS} steps) ===", flush=True)
    row = evaluate.run_and_score(r, extra={"config": cfg,
                                           "steps": P3_EPOCHS * P3_STEPS},
                                 save_prob=f"results/probs/long_{cfg['name']}.npy")
    print(f"PHASE3 {cfg['name']}: test IoU={row.get('iou')} "
          f"dice={row.get('dice')} err={row.get('error')}", flush=True)
    return row


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sanity":
        phase0()
    elif len(sys.argv) > 2 and sys.argv[1] == "phase1":
        run_config(int(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == "phase3":
        phase3(int(sys.argv[2]))
    else:
        print(__doc__)
