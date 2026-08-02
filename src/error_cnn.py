"""An IMAGE model as the confidence model: a U-Net that predicts the error map.

Sky's proposal (2026-07-30). The existing learned confidence/error models are
per-pixel regressors over 11 hand-crafted features whose widest spatial window is
**11x11** (`combine.ConfidenceModel.build_features`). Chondrules are roughly
50-100 px across, so those models literally cannot see a whole chondrule — yet
the dominant measured failure is whole objects missed (test slice 69, chondrule
1: TP 14164 vs FN 15752, cut along a straight line). A pixel-wise regressor has
no representation for "this entire object is wrong". A U-Net does.

Why this is worth a run: in the recall-0.99 bake-off, all five error regressors
and all five trust classifiers pruned **nothing** — best pruned val IoU 0.06661
against an unpruned 0.06661 — while the zero-parameter `margin` baseline reached
0.43807. The learned stage contributed exactly zero.

Input  : 2 channels — (normalised raw, base model's probability)
Target : |base_prob - ground truth|, the same residual `ErrorModel` regresses
Output : a per-pixel predicted-error map, usable as a prune score exactly like
         any other confidence map

[LEAKAGE][MUST] Fit on the VALIDATION zone — the only zone where the base
model's mistakes are honest (it memorises train, so train-zone errors describe a
distribution that exists nowhere else; constitution v3+).

[LEAKAGE][MUST] **Val is SPLIT.** A U-Net has orders of magnitude more capacity
than a ridge regressor, so fitting it on val AND choosing its prune cutoff on the
same val would let it memorise those errors and report a val score that does not
transfer. Val is therefore halved along z with a buffer: the net trains on half
A, the prune cutoff is selected on half B, and TEST is scored once with both
frozen. The regressors it is compared against do NOT do this — they fit and
select on all of val — so if anything this handicaps the U-Net.

[HONESTY] Half of a 60-slice val slab is 30 slices and roughly 9-10 chondrules.
That is very little to fit an image model on, and it is the honest amount
available. A poor result here is as consistent with "too few chondrules" as with
"the idea does not work"; only the classical arm (25% graded vs the CNN's 1.19%)
can separate those, which is why this runs on BOTH base maps.
"""
from __future__ import annotations
import numpy as np
import torch

from .models_cnn import DEVICE, PAD_MULT, UNet2D, normalize

VAL_BUFFER = 4          # slices dropped between the two val halves


def split_val(n, buffer=VAL_BUFFER):
    """Contiguous z-halves of the val slab with a gap. Returns (idx_a, idx_b).

    Contiguous, not random: adjacent CT slices are near-duplicates, so a random
    split would leak half A into half B — the same reason the project's main
    split uses contiguous slabs with a 12-slice buffer.
    """
    half = (n - buffer) // 2
    a = np.arange(0, half)
    b = np.arange(half + buffer, n)
    return a, b


def make_inputs(raw, prob):
    """(n,H,W) raw + (n,H,W) prob -> (n,2,H,W) float32 network input."""
    r = normalize(np.asarray(raw))
    p = np.asarray(prob, np.float32)
    return np.stack([r, p], axis=1)


def residual_target(prob, mask, valid):
    """|prob - gt|, the quantity ErrorModel regresses. Invalid pixels -> 0."""
    gt = np.asarray(mask, np.float32)
    return (np.abs(np.asarray(prob, np.float32) - gt)
            * np.asarray(valid, np.float32)).astype(np.float32)


def masked_l1(pred, target, valid):
    """L1 over valid pixels only. Regression, so no dice/BCE here."""
    d = (pred - target).abs() * valid
    return d.sum() / valid.sum().clamp(min=1.0)


def fit_error_unet(raw, prob, mask, valid, idx, epochs=40, steps=120, batch=8,
                   patch=128, lr=3e-4, base=16, seed=0):
    """Train the error U-Net on the slices in `idx`. Returns the fitted model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = make_inputs(raw[idx], prob[idx])
    Y = residual_target(prob[idx], mask[idx], valid[idx])
    V = np.asarray(valid[idx], np.float32)
    n, _, H, W = X.shape
    net = UNet2D(in_ch=2, base=base, depth=4).to(DEVICE).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps):
            xs, ys, vs = [], [], []
            for _ in range(batch):
                i = int(rng.integers(n))
                y0 = int(rng.integers(max(1, H - patch)))
                x0 = int(rng.integers(max(1, W - patch)))
                xs.append(X[i, :, y0:y0 + patch, x0:x0 + patch])
                ys.append(Y[i, y0:y0 + patch, x0:x0 + patch])
                vs.append(V[i, y0:y0 + patch, x0:x0 + patch])
            xb = torch.from_numpy(np.stack(xs)).to(DEVICE)
            yb = torch.from_numpy(np.stack(ys)[:, None]).to(DEVICE)
            vb = torch.from_numpy(np.stack(vs)[:, None]).to(DEVICE)
            opt.zero_grad()
            loss = masked_l1(torch.sigmoid(net(xb)), yb, vb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    [error_unet] epoch {ep + 1}/{epochs} L1={tot / steps:.5f}",
                  flush=True)
    return net.eval()


def predict_error(net, raw, prob, tile=256, halo=32):
    """Predicted error map, (n,H,W) float32. Tiled, matching the base path."""
    X = make_inputs(raw, prob)
    n, _, H, W = X.shape
    out = np.zeros((n, H, W), np.float32)
    with torch.no_grad():
        for i in range(n):
            for y0 in range(0, H, tile):
                for x0 in range(0, W, tile):
                    ya, yb_ = max(0, y0 - halo), min(H, y0 + tile + halo)
                    xa, xb_ = max(0, x0 - halo), min(W, x0 + tile + halo)
                    win = np.ascontiguousarray(X[i, :, ya:yb_, xa:xb_][None])
                    wh, ww = win.shape[-2:]
                    ph, pw = (-wh) % PAD_MULT, (-ww) % PAD_MULT
                    if ph or pw:
                        win = np.pad(win, ((0, 0), (0, 0), (0, ph), (0, pw)),
                                     mode="edge")
                    t = torch.from_numpy(win).to(DEVICE)
                    p = torch.sigmoid(net(t)).cpu().numpy()[0, 0]
                    if ph or pw:
                        p = p[:wh, :ww]
                    cy1, cx1 = min(H, y0 + tile), min(W, x0 + tile)
                    out[i, y0:cy1, x0:cx1] = p[y0 - ya:cy1 - ya, x0 - xa:cx1 - xa]
    return out


def _selftest():
    """The net must LEARN something: L1 on a trivial synthetic task must drop."""
    rng = np.random.default_rng(0)
    n, H, W = 6, 96, 96
    mask = np.zeros((n, H, W), bool)
    mask[:, 30:60, 30:60] = True
    raw = (mask * 200 + rng.normal(0, 8, (n, H, W))).clip(0, 255).astype(np.uint8)
    prob = np.where(mask, 0.9, 0.1).astype(np.float32)
    prob[:, 40:50, 40:50] = 0.1                      # a deliberate blind spot
    valid = np.ones((n, H, W), np.float32)
    tgt = residual_target(prob, mask, valid)
    net = fit_error_unet(raw, prob, mask, valid, np.arange(n),
                         epochs=3, steps=15, batch=4, patch=48, base=8)
    pe = predict_error(net, raw, prob)
    # it should predict HIGHER error inside the blind spot than outside it
    inside = float(pe[:, 42:48, 42:48].mean())
    outside = float(pe[:, 5:15, 5:15].mean())
    return {"pred_err_in_blindspot": inside, "pred_err_elsewhere": outside,
            "true_err_in_blindspot": float(tgt[:, 42:48, 42:48].mean()),
            "learned_the_sign": inside > outside}


if __name__ == "__main__":
    a, b = split_val(60)
    print(f"val split: train on {len(a)} slices {a[0]}-{a[-1]}, "
          f"cutoff on {len(b)} slices {b[0]}-{b[-1]}, buffer {VAL_BUFFER}")
    r = _selftest()
    print(f"blind-spot predicted error {r['pred_err_in_blindspot']:.4f} vs "
          f"elsewhere {r['pred_err_elsewhere']:.4f} "
          f"(true in blind spot {r['true_err_in_blindspot']:.4f})")
    print(f"learned the sign: {r['learned_the_sign']}")
