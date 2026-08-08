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

[LEAKAGE][MUST] **The net never grades a slice it was fitted on.** A U-Net has
orders of magnitude more capacity than a ridge regressor, so fitting it on val
AND choosing its prune cutoff on the same val would let it memorise those errors
and report a val score that does not transfer. The default protocol is now k-fold
cross-fitting inside the val slab (`fit_error_unet_cv`): every val slice gets an
error map from a fold model that did not see it, the prune cutoff is chosen on
those out-of-fold maps, one final net is fitted on all of val, and TEST is scored
once with that net and that cutoff frozen. Folds are contiguous and buffered
(reused from `classical_crossfit.contiguous_folds`) because adjacent CT slices
are near-duplicates. The regressors this is compared against hold out nothing at
all: they fit and select on all of val.

[WHAT CHANGED, 2026-08-06] The previous arrangement fitted the net on ONE
contiguous 28-slice half of val and picked the cutoff on the other 28
(`split_val`, still available via `cv=False` so the recorded run reproduces).
That run measured test IoU 0.11752 against a base of 0.29443 and a free margin
baseline of 0.30279, and it did not beat margin on the held-out half either.
Cross-fitting raises the per-fold fit set from 28 slices to 40-44 and the
cutoff-selection set from 28 slices to all 60. This is a TEST of one hypothesis,
that fit-set size was the binding constraint. It is not a claim that the score
will improve. The val slab still holds the same few chondrules, so the same
collapse is a possible outcome.

[COST] k folds plus the final all-val fit is k + 1 trainings, so roughly k + 1
times the compute of the single-split run. At the default k = 5 that is six fits
where there used to be one.

[HONESTY] Half of a 60-slice val slab is 30 slices and roughly 9-10 chondrules.
Cross-fitting lifts each fold's fit set to 40-44 slices. That is more, and it is
still very little to fit an image model on, and it is the honest amount
available. A poor result here is as consistent with "too few chondrules" as with
"the idea does not work"; only the classical arm (25% graded vs the CNN's 1.19%)
can separate those, which is why this runs on BOTH base maps.
"""
from __future__ import annotations
import numpy as np
import torch

from .classical_crossfit import contiguous_folds
from .models_cnn import DEVICE, PAD_MULT, UNet2D, normalize

VAL_BUFFER = 4          # slices dropped between the two val halves
CV_FOLDS = 5            # folds inside the val slab, matching classical_crossfit


def split_val(n, buffer=VAL_BUFFER):
    """Contiguous z-halves of the val slab with a gap. Returns (idx_a, idx_b).

    Contiguous, not random: adjacent CT slices are near-duplicates, so a random
    split would leak half A into half B — the same reason the project's main
    split uses contiguous slabs with a 12-slice buffer.

    Superseded by `fit_error_unet_cv` as the default. Kept so the recorded
    28/28 run stays reproducible.
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
    # Carried on the model so a cross-fitted run can record each fold's final
    # loss without changing this function's return type.
    net.final_train_l1 = float(tot / steps)
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


def fit_error_unet_cv(raw, prob, mask, valid, k=CV_FOLDS, buffer=VAL_BUFFER,
                      **fit_kw):
    """Cross-fit the error U-Net inside the val slab. Returns (oof, net, folds).

    `oof_error[i]` is predicted by a fold model that never saw slice i, nor any
    slice within `buffer` of it, so the prune cutoff can be selected on ALL n val
    slices instead of on a 28-slice half. `final_net` is fitted on every val
    slice and is the model applied to TEST. `per_fold` records each fold's
    held-out count, fit count and final training L1 so the run is auditable.

    Folds come from `classical_crossfit.contiguous_folds`, which is contiguous in
    z and drops a buffer either side of the held-out block. Adjacent CT slices
    are near-duplicates, so random folds would leave the out-of-fold predictions
    effectively in-fold.

    `**fit_kw` goes straight to `fit_error_unet`, so epochs/steps/base stay with
    the caller.

    [COST] k + 1 fits, roughly k + 1 times the compute of the single-split path.
    """
    P = np.asarray(prob)
    n = int(P.shape[0])
    folds = contiguous_folds(n, k, buffer)
    oof = np.zeros(P.shape, np.float32)
    per_fold = []
    for j, (held, fit_idx) in enumerate(folds):
        print(f"  [error_unet cv] fold {j + 1}/{len(folds)}: fit on "
              f"{len(fit_idx)} slices, predict held-out {held[0]}-{held[-1]} "
              f"({len(held)})", flush=True)
        net = fit_error_unet(raw, prob, mask, valid, fit_idx, **fit_kw)
        oof[held] = predict_error(net, raw[held], prob[held])
        per_fold.append({"fold": j + 1, "n_held_out": int(len(held)),
                         "n_fit": int(len(fit_idx)),
                         "final_train_l1": float(net.final_train_l1)})
    print(f"  [error_unet cv] final fit on all {n} val slices, for TEST",
          flush=True)
    final_net = fit_error_unet(raw, prob, mask, valid, np.arange(n), **fit_kw)
    return oof, final_net, per_fold


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
    print(f"val split [cv=False]: train on {len(a)} slices {a[0]}-{a[-1]}, "
          f"cutoff on {len(b)} slices {b[0]}-{b[-1]}, buffer {VAL_BUFFER}")
    fs = contiguous_folds(60, CV_FOLDS, VAL_BUFFER)
    print(f"val cross-fit [default]: {len(fs)} folds, held out "
          f"{[len(h) for h, _ in fs]}, fitted on {[len(f) for _, f in fs]}, "
          f"buffer {VAL_BUFFER}")
    r = _selftest()
    print(f"blind-spot predicted error {r['pred_err_in_blindspot']:.4f} vs "
          f"elsewhere {r['pred_err_elsewhere']:.4f} "
          f"(true in blind spot {r['true_err_in_blindspot']:.4f})")
    print(f"learned the sign: {r['learned_the_sign']}")
