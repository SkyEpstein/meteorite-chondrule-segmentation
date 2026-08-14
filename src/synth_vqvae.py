"""VQ-VAE tokenizer over micro-CT patches. Stage 2 of the MaskGIT build.

`specs/2026-08-08-synthetic-maskgit/spec.md`, task 2. MaskGIT operates on
DISCRETE tokens, so a tokenizer has to exist before any transformer work starts.
This encodes a 128x128 patch to a grid of codebook indices and decodes back. The
downsample factor is configurable (`--downsample=`): 8 gives a 16x16 grid, 4 a
32x32 grid. That knob turned out to be the difference between a tokenizer the
segmenter cannot read and one it mostly can, so it is a first-class parameter
rather than a constant.

The acceptance gate is here rather than downstream on purpose. The abstract's
data regime is "a few thousand images"; ours is 347 slices, and a data-starved
tokenizer that blurs or collapses its codebook is the single most likely way this
whole approach fails. Failing loudly at this stage costs hours; discovering it
after a transformer is trained costs days. Two checks:

  1. **Reconstruction beats a matched-bitrate baseline.** The tokenizer spends
     (128/f)^2 tokens at log2(codebook) bits. Bilinear downsample to the SAME
     grid at 8 bits per pixel is a comparable budget and needs no learning. The
     control must use the model's own factor: it did not, so the f=4 gate was
     briefly scored against the f=8 baseline and its margin overstated by
     2.89 dB (31.001 - 28.115 = 2.886).
  2. **The codebook is not collapsed.** Perplexity is reported against codebook
     size; a model that routes everything through a handful of codes gives the
     transformer almost no alphabet to model, which looks fine in reconstruction
     loss and is fatal for generation.

EMA codebook updates rather than a codebook loss term: standard for VQ-VAE-2 and
MaskGIT, and markedly more stable when data is thin, which is our situation.

[LEAKAGE] Patches come from `synth_patches`, which enforces the generator's data
boundary. This module never touches raw slices and so cannot widen it. The
held-out split here is a patch-level split for measuring reconstruction, NOT a
leakage boundary; the leakage boundary is at the slice level upstream.

NOTE: these two gates are necessary and NOT sufficient. Both passed at f=8 while
`--segcheck` showed the tokenizer more than doubling the segmenter's
predicted-positive fraction. Run `--segcheck` before believing a tokenizer.

Run:  python -m src.synth_vqvae --epochs=40 --grad_w=1.0 --downsample=4
      python -m src.synth_vqvae --segcheck --grad_w=1.0 --downsample=4
      python -m src.synth_vqvae --figure   --grad_w=1.0 --downsample=4
      python -m src.synth_vqvae --eval     --grad_w=1.0 --downsample=4
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_cnn import DEVICE
from .synth_patches import PATCH        # 128; the token grid is PATCH // factor

PATCHES = "results/synth/patches.npz"
# Artifact paths come from paths_for(factor, grad_w); these fixed names are
# retired because they were the vector for the clobber that destroyed the
# MSE-only checkpoint.

CODEBOOK, EMB_DIM = 512, 64
# [MUST] 8 is the DEFAULT, not the recommendation. f=8 was measured to strip the
# matrix texture the base segmenter reads, more than doubling its
# predicted-positive fraction, and is REJECTED. The working factor is 4. This
# default is kept only so the historical f=8 artifacts stay addressable; always
# pass --downsample=4 for real work.
DOWNSAMPLE = 8                     # 128 -> 16 token grid; REJECTED, see above
HELD_FRAC = 0.1


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(),
                               nn.Conv2d(c, c, 3, padding=1),
                               nn.GroupNorm(8, c), nn.SiLU(),
                               nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.b(x)


class VectorQuantizerEMA(nn.Module):
    """Codebook with EMA updates. Returns quantized latents, loss, indices.

    Two mechanisms below exist because the first smoke run collapsed to **4 of
    512 codes** after 236 updates, which is collapse rather than slow warmup:

      * **Data-dependent init.** The codebook was initialised as `randn * 0.1`
        while nothing constrains the encoder's output scale. If the encoder
        emits vectors far from that radius, every one of them lands on whichever
        few codes happen to be nearest, and EMA then drags exactly those codes
        toward the data mean, reinforcing the collapse. Seeding the codebook
        from real encoder outputs removes the scale mismatch entirely.
      * **Dead-code restart.** Codes that stop winning receive no EMA gradient
        and can never recover on their own. Periodically reseeding unused codes
        from live encoder outputs is the standard remedy (VQGAN, Jukebox).
    """

    def __init__(self, n_codes=CODEBOOK, dim=EMB_DIM, decay=0.99, eps=1e-5,
                 commit=0.25, restart_every=100, dead_thresh=1e-3):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.eps = n_codes, dim, decay, eps
        self.commit = commit
        self.restart_every, self.dead_thresh = restart_every, dead_thresh
        emb = torch.randn(n_codes, dim) * 0.1
        self.register_buffer("embed", emb)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", emb.clone())
        self.register_buffer("inited", torch.zeros(1))
        self.register_buffer("steps", torch.zeros(1))

    @torch.no_grad()
    def _seed(self, flat, which=None):
        """Seed codes from real encoder outputs, sampling with replacement."""
        n = self.n_codes if which is None else int(which.sum())
        if n == 0:
            return
        pick = torch.randint(0, flat.shape[0], (n,), device=flat.device)
        new = flat[pick]
        if which is None:
            self.embed.copy_(new)
            self.embed_avg.copy_(new)
            self.cluster_size.fill_(1.0)
        else:
            self.embed[which] = new
            self.embed_avg[which] = new
            self.cluster_size[which] = 1.0

    def forward(self, z):
        # z: (B, D, H, W) -> flatten to (N, D)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        if self.training and float(self.inited) == 0:
            self._seed(flat.detach())
            self.inited.fill_(1)

        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embed.t()
             + self.embed.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.embed[idx].view(B, H, W, D).permute(0, 3, 1, 2)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(
                    onehot.t() @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = ((self.cluster_size + self.eps)
                      / (n + self.n_codes * self.eps) * n)
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))

                self.steps += 1
                if int(self.steps) % self.restart_every == 0:
                    dead = self.cluster_size < self.dead_thresh
                    if bool(dead.any()):
                        self._seed(flat.detach(), dead)

        loss = self.commit * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()                    # straight-through
        return q, loss, idx.view(B, H, W)


def paths_for(factor, grad_w=None):
    """Artifact names keyed by BOTH factor and gradient weight.

    [FIXED 2026-08-08 after review] The previous version keyed on `factor`
    alone, so the MSE-only and gradient-loss runs at f=8 wrote the same
    `vqvae.pt`. The MSE-only checkpoint was overwritten and could not be
    re-measured, which is the fourth instance of this clobber in this repo.
    Every distinguishing knob goes in the name.
    """
    g = "mse" if not grad_w else f"g{grad_w:g}"
    return (f"results/synth/vqvae_f{factor}_{g}.pt",
            f"results/synth/vqvae_f{factor}_{g}.json")


class VQVAE(nn.Module):
    """VQ-VAE with a configurable downsample factor (a power of two).

    `factor` is a diagnostic knob as much as a design choice. Factor 8 is the
    working design; 4 and 2 exist to answer whether the segmenter-preservation
    failure is caused by COMPRESSION or by the reconstruction OBJECTIVE. Note
    that factor 2 gives a 64x64 = 4096-token grid per patch, far too long for
    the MaskGIT stage, so it is a probe and never a candidate.
    """

    def __init__(self, ch=128, n_codes=CODEBOOK, dim=EMB_DIM,
                 factor=DOWNSAMPLE):
        super().__init__()
        n_down = int(math.log2(factor))
        if 2 ** n_down != factor or n_down < 1:
            raise ValueError(f"downsample factor must be a power of 2, got {factor}")
        self.factor = factor

        enc, c_in = [], 1
        for i in range(n_down):
            c_out = ch // 2 if i == 0 and n_down > 1 else ch
            enc += [nn.Conv2d(c_in, c_out, 4, stride=2, padding=1), nn.SiLU()]
            c_in = c_out
        enc = enc[:-1]                       # drop the trailing activation
        enc += [ResBlock(c_in), ResBlock(c_in),
                nn.GroupNorm(8, c_in), nn.SiLU(), nn.Conv2d(c_in, dim, 1)]
        self.enc = nn.Sequential(*enc)

        self.vq = VectorQuantizerEMA(n_codes, dim)

        dec = [nn.Conv2d(dim, c_in, 3, padding=1), ResBlock(c_in),
               ResBlock(c_in), nn.GroupNorm(8, c_in), nn.SiLU()]
        for i in range(n_down):
            last = i == n_down - 1
            c_out = 1 if last else (ch // 2 if i == n_down - 2 and n_down > 1
                                    else ch)
            dec += [nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1)]
            if not last:
                dec += [nn.SiLU()]
            c_in = c_out
        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        q, vq_loss, idx = self.vq(self.enc(x))
        return self.dec(q), vq_loss, idx

    @torch.no_grad()
    def encode_indices(self, x):
        return self.vq(self.enc(x))[2]

    @torch.no_grad()
    def decode_indices(self, idx):
        B, H, W = idx.shape
        q = self.vq.embed[idx.reshape(-1)].view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.dec(q)


def load_patches(path=PATCHES):
    """Load the patch pool, and RE-VALIDATE its provenance before using it.

    [LEAKAGE][MUST, added 2026-08-08 after review] `extract()` records which
    slice each patch came from and which slices were held out, and nothing used
    to check them again here. The pool is gitignored (94 MB) and regenerable in
    place, so a stale npz built by an earlier, buggier boundary would train every
    tokenizer on buffer-adjacent pixels while each artifact's regime string
    claimed the boundary was enforced. Re-checking on load costs microseconds.
    """
    d = np.load(path)
    if "src" not in d:
        raise SystemExit(
            f"{path} has no `src` field, so its provenance cannot be checked. "
            "Pools predating that field are exactly the ones most likely to "
            "have been built by a buggier boundary. Rebuild it: "
            "python -m src.synth_patches")
    # [MUST] Check against split.json's OWN slab labels, the same authority
    # assert_boundary uses. An earlier version compared `src` against the
    # npz's `heldout_slices`, which holds val+test only. The historical
    # contamination was BUFFER slices, which are in neither set, so that
    # guard could not detect the very bug its docstring cited. Verified: 0
    # of the 44 buffer slices appear in heldout_slices.
    from .synth_patches import band_and_slabs
    slabs = band_and_slabs()
    bad = sorted({(int(i), slabs[int(i)]) for i in set(d["src"].tolist())
                  if slabs.get(int(i), "train") != "train"})
    if bad:
        raise SystemExit(
            f"LEAKAGE: {path} contains patches from slices the split did "
            f"not label `train`: {bad[:10]}. Rebuild it: "
            f"python -m src.synth_patches")
    P, lo, hi = d["patches"], float(d["norm_lo"]), float(d["norm_hi"])
    X = np.clip((P - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return X, d["labeled"], lo, hi


GRAD_W = 1.0          # weight on the gradient term; --grad_w= overrides


def grad_loss(a, b):
    """Sobel-gradient MSE between two images.

    [ADDED 2026-08-08] Pixel MSE alone is minimised by predicting the local
    mean, which IS a blur, and the first trained tokenizer blurred exactly as
    that predicts: matrix speckle gone, thin cracks lost, small inclusions
    smeared. That is not cosmetic here. `segcheck` measured the consequence:
    running the base segmenter on reconstructions instead of originals flipped
    **23.6% of pixel decisions** and took the predicted-positive fraction from
    **0.1429 to 0.3255, a factor of 2.28**, because the model's documented
    failure is calling confusable matrix "chondrule" and the blur removes the
    texture that distinguishes them.

    [REGIME] Those are the LIVE f=8 figures, from the rebuilt 347-slice pool
    (`results/synth/vqvae_segcheck_f8_g1.json`). The numbers that originally
    motivated this loss term were 21.9% and 0.1633 -> 0.3271, measured on the
    MSE-only tokenizer over the pre-rebuild 349-slice pool. That checkpoint was
    destroyed by the `paths_for` clobber and those figures are NOT reproducible
    from anything committed, so they are recorded here as history and must not
    be quoted as results. Note 0.1633 is within 0.0001 of the CURRENT f=4
    reconstruction positive fraction, so the stale pair is easy to misread as a
    live one.

    Matching gradients penalises the smooth solution directly. It is the cheap,
    stable half of what VQGAN buys with a discriminator, chosen over adversarial
    training because 8,328 patches from a single meteorite is thin data for a
    GAN and a collapsed discriminator fails silently.
    """
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                     dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
    kt = k.transpose(2, 3)
    return (F.mse_loss(F.conv2d(a, k, padding=1), F.conv2d(b, k, padding=1))
            + F.mse_loss(F.conv2d(a, kt, padding=1), F.conv2d(b, kt, padding=1)))


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def baseline_psnr(X, factor=DOWNSAMPLE):
    """Bilinear downsample to the token grid and back, quantized to 8 bits.

    The matched-bitrate control: same spatial budget as the token grid, no
    learning. Quantizing to 8 bits keeps the comparison honest rather than
    handing the baseline unlimited precision.
    """
    t = torch.from_numpy(X).unsqueeze(1)
    small = F.interpolate(t, scale_factor=1 / factor, mode="bilinear",
                          align_corners=False)
    small = torch.round(small.clamp(0, 1) * 255) / 255
    back = F.interpolate(small, size=(X.shape[1], X.shape[2]), mode="bilinear",
                         align_corners=False)
    return _psnr(X, back.squeeze(1).numpy())


@torch.no_grad()
def evaluate(model, Xh, batch=64):
    model.eval()
    recon, codes = [], []
    for i in range(0, len(Xh), batch):
        xb = torch.from_numpy(Xh[i:i + batch]).unsqueeze(1).to(DEVICE)
        r, _, idx = model(xb)
        recon.append(r.squeeze(1).clamp(0, 1).cpu().numpy())
        codes.append(idx.reshape(-1).cpu().numpy())
    R = np.concatenate(recon)
    C = np.concatenate(codes)
    counts = np.bincount(C, minlength=CODEBOOK).astype(np.float64)
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum()))
    return {"psnr": _psnr(Xh, R), "perplexity": perplexity,
            "codes_used": int((counts > 0).sum()), "codebook": CODEBOOK}


def gate(res, base):
    """The two acceptance checks, reported explicitly rather than assumed."""
    beats = res["psnr"] > base
    # A tokenizer routing everything through a few codes gives the transformer
    # almost no alphabet. 10% of the codebook is a deliberately lenient floor:
    # it is meant to catch collapse, not to certify quality.
    healthy = res["perplexity"] >= 0.10 * CODEBOOK
    print(f"\n=== acceptance gate ===")
    print(f"  reconstruction PSNR {res['psnr']:.2f} dB vs matched-bitrate "
          f"baseline {base:.2f} dB   {'PASS' if beats else 'FAIL'}")
    print(f"  codebook perplexity {res['perplexity']:.1f} of {CODEBOOK} "
          f"({res['codes_used']} codes used)   "
          f"{'PASS' if healthy else 'FAIL (collapsed)'}")
    if not (beats and healthy):
        print("  GATE FAILED. Do not proceed to the transformer: a tokenizer "
              "that cannot\n  beat bilinear, or has collapsed, gives MaskGIT "
              "nothing to model.")
    return bool(beats and healthy)


def load_model(factor=DOWNSAMPLE, grad_w=None, device=None):
    """Rebuild a saved VQ-VAE, taking its factor from the checkpoint itself.

    Reading the factor from the file rather than the caller means a checkpoint
    can never be loaded into a mismatched architecture, which would fail with a
    confusing shape error or, worse, load partially.
    """
    ckpt, _ = paths_for(factor, grad_w)
    blob = torch.load(ckpt, map_location=device or DEVICE, weights_only=True)
    if "factor" not in blob:
        raise SystemExit(
            f"{ckpt} has no `factor` key, so its architecture is unknown. "
            "Silently assuming the default is how a checkpoint gets loaded "
            "into the wrong network. Retrain it.")
    f = int(blob["factor"])
    m = VQVAE(factor=f).to(device or DEVICE)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, blob


def train(epochs=30, batch=64, lr=3e-4, seed=0, grad_w=GRAD_W,
          factor=DOWNSAMPLE):
    torch.manual_seed(seed)
    X, labeled, lo, hi = load_patches()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_h = int(len(X) * HELD_FRAC)
    hold, tr = perm[:n_h], perm[n_h:]
    Xtr, Xh = X[tr], X[hold]
    print(f"patches: {len(Xtr)} train, {len(Xh)} held out for reconstruction")

    base = baseline_psnr(Xh, factor)
    print(f"matched-bitrate bilinear baseline on held-out: {base:.2f} dB")

    model = VQVAE(factor=factor).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"VQ-VAE {n_par/1e6:.2f}M params, device={DEVICE}\n")

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        idxs = rng.permutation(len(Xtr))
        tot = tot_px = tot_gr = n = 0
        for i in range(0, len(idxs), batch):
            xb = torch.from_numpy(Xtr[idxs[i:i + batch]]).unsqueeze(1).to(DEVICE)
            recon, vq_loss, _ = model(xb)
            px = F.mse_loss(recon, xb)
            gr = grad_loss(recon, xb) if grad_w else torch.zeros((), device=DEVICE)
            loss = px + grad_w * gr + vq_loss
            opt.zero_grad(); loss.backward(); opt.step()
            b = len(xb)
            tot += float(loss.detach()) * b
            tot_px += float(px.detach()) * b
            tot_gr += float(gr.detach()) * b
            n += b
        if ep % 5 == 0 or ep == epochs:
            r = evaluate(model, Xh)
            # Both components are printed because their RATIO is the thing to
            # watch: if the gradient term dwarfs the pixel term the model starts
            # chasing texture at the cost of intensity, which would trade one
            # failure for another.
            print(f"  epoch {ep:>3}/{epochs}  loss {tot/max(n,1):.5f} "
                  f"(px {tot_px/max(n,1):.5f}  grad {tot_gr/max(n,1):.5f})  "
                  f"held-out PSNR {r['psnr']:.2f} dB  perplexity "
                  f"{r['perplexity']:.1f}  {time.time()-t0:.0f}s", flush=True)

    res = evaluate(model, Xh)
    passed = gate(res, base)
    ckpt_path, out_path = paths_for(factor, grad_w)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "codebook": CODEBOOK,
                "emb_dim": EMB_DIM, "norm_lo": lo, "norm_hi": hi,
                "factor": factor, "grad_w": grad_w}, ckpt_path)
    with open(out_path, "w") as f:
        json.dump({"factor": int(factor), "grad_w": float(grad_w),
                   "token_grid": PATCH // factor, "regime": (
            f"VQ-VAE, {factor}x downsample, 128px patch -> "
            f"{PATCH // factor}x{PATCH // factor} tokens, "
            f"codebook {CODEBOOK} x {EMB_DIM}, EMA updates, {epochs} epochs, "
            f"gradient-loss weight {grad_w:g}. "
            "Patch pool from src.synth_patches (generator boundary enforced "
            "there). The held-out split here is patch-level, for measuring "
            "reconstruction only; it is NOT the leakage boundary."),
            # [MUST] These two travel WITH the artifact. The caveat existed only
            # in the CHANGELOG and in hand-edits to the JSONs, so the next
            # training run would have written a file with it stripped, and a
            # regenerate-and-diff check could not tell hand-edit from drift.
            "holdout": (
                "SAME-SLICE patch-level split: all held-out patches come from "
                "slices that also supplied training patches, and crops may "
                "overlap spatially. PSNR/perplexity are optimistic by an "
                "unmeasured amount. Valid for comparing factors, which share "
                "the holdout."),
            "gate_passed_note": (
                "`gate_passed` covers the PSNR and perplexity gates ONLY. It "
                "does NOT cover the segmenter-preservation check (--segcheck), "
                "which is what actually decided this arm; f=8 records "
                "gate_passed=true and was rejected there. See spec task 2."),
            "n_patches": int(len(X)), "holdout_seed": int(seed),
            "held_frac": float(HELD_FRAC),
            "baseline_psnr_matched_bitrate": base, "held_out": res,
            "gate_passed": bool(passed), "params": int(n_par),
            "sec": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nwrote {out_path} and {ckpt_path}")
    return passed


def figure(n=6, seed=0, factor=DOWNSAMPLE, grad_w=None,
           out="results/figures/synth_vqvae_recon.png"):
    """Render originals against reconstructions. [MUST] Do not skip this.

    PSNR and perplexity both passed, and this project has already been burned
    once by exactly that situation: the tiled-inference defect passed every
    scalar check and was only visible when someone rendered the output. A
    tokenizer can post a good PSNR while smearing chondrule rims, which are the
    structure the whole downstream task depends on, so the check has to be
    visual as well as numeric.

    Patches are picked to CONTAIN structure rather than at random: an
    all-background crop reconstructs perfectly and shows nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, _, _, _ = load_patches()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    Xh = X[perm[:int(len(X) * HELD_FRAC)]]
    pick = np.argsort([-float(x.std()) for x in Xh])[:n]     # busiest crops
    sel = Xh[pick]

    m, _ = load_model(factor, grad_w)
    with torch.no_grad():
        xb = torch.from_numpy(sel).unsqueeze(1).to(DEVICE)
        rec = m(xb)[0].squeeze(1).clamp(0, 1).cpu().numpy()
        idx = m.encode_indices(xb).cpu().numpy()

    fig, ax = plt.subplots(3, n, figsize=(2.5 * n, 7.6))
    for i in range(n):
        ax[0, i].imshow(sel[i], cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(f"original  std {sel[i].std():.3f}", fontsize=8)
        ax[1, i].imshow(rec[i], cmap="gray", vmin=0, vmax=1)
        ax[1, i].set_title(f"reconstruction  {_psnr(sel[i], rec[i]):.1f} dB",
                           fontsize=8)
        ax[2, i].imshow(np.abs(sel[i] - rec[i]), cmap="magma", vmin=0, vmax=0.3)
        ax[2, i].set_title(f"|error|   {len(np.unique(idx[i]))} distinct codes",
                           fontsize=8)
        for r in range(3):
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
    g = PATCH // m.factor
    fig.suptitle(
        f"VQ-VAE tokenizer: {PATCH}x{PATCH} patch to a {g}x{g} token grid and "
        f"back (downsample {m.factor}, gradient-loss weight {grad_w or 0:g})\n"
        "the six busiest held-out patches, which are the hardest tail of the "
        "pool rather than a random sample.\n"
        "Error row is clipped at 0.3 to make rim smearing visible if it is "
        "there.", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def segcheck(n=400, seed=0, factor=DOWNSAMPLE, grad_w=None,
             ckpt="results/checkpoints/unet2d_s30000_seed1.pt", out=None):
    """Does the tokenizer preserve what the SEGMENTER consumes?

    The task-relevant gate, and the one that matters more than PSNR. Rendering
    showed the tokenizer acting as a strong low-pass filter: matrix speckle
    gone, thin cracks lost, small inclusions smeared. That is exactly the
    signal the base model's errors are made of, since its documented failure is
    "large, coherent, CONFIDENT blobs on matrix that looks like chondrule". If
    reconstruction changes the segmenter's output, then synthetic images built
    from these tokens will not reproduce the base model's real error structure,
    and the confidence stages would be fitted on the wrong problem.

    Run the base net directly on 128px patches, original versus reconstruction.
    Fully convolutional, so no tiling is involved and the tiler's known grid
    dependence cannot contaminate the comparison.

    Uses `unet2d_s30000_seed1`: the reference recipe, and the reference's own
    seed 0 predates checkpointing so it has no weights to load.
    """
    from .models_cnn import UNet2D, normalize

    X, _, lo, hi = load_patches()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    Xh = X[perm[:int(len(X) * HELD_FRAC)]]
    pick = np.argsort([-float(x.std()) for x in Xh])[:n]      # structured crops
    sel = Xh[pick]

    m, _ = load_model(factor, grad_w)

    ck = torch.load(ckpt, map_location=DEVICE)
    net = UNet2D(in_ch=1, base=32, depth=4).to(DEVICE)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    thr = float(ck["best_thr"])

    po, pr = [], []
    with torch.no_grad():
        for i in range(0, len(sel), 32):
            xb = torch.from_numpy(sel[i:i + 32]).unsqueeze(1).to(DEVICE)
            rec = m(xb)[0].clamp(0, 1)
            # back to raw scale, then the segmenter's own fixed /255 rule
            raw_o = xb * (hi - lo) + lo
            raw_r = rec * (hi - lo) + lo
            po.append(torch.sigmoid(net(raw_o / 255.0)).squeeze(1).cpu().numpy())
            pr.append(torch.sigmoid(net(raw_r / 255.0)).squeeze(1).cpu().numpy())
    Po, Pr = np.concatenate(po), np.concatenate(pr)

    bo, br = Po >= thr, Pr >= thr
    inter = float((bo & br).sum()); union = float((bo | br).sum())
    res = {
        "n_patches": int(len(sel)), "threshold": thr, "checkpoint": ckpt,
        "downsample_factor": int(factor),
        "mean_abs_delta_prob": float(np.mean(np.abs(Po - Pr))),
        "max_abs_delta_prob": float(np.max(np.abs(Po - Pr))),
        "corr": float(np.corrcoef(Po.ravel(), Pr.ravel())[0, 1]),
        "agreement_iou": inter / max(union, 1.0),
        "decision_flip_frac": float((bo != br).mean()),
        "pos_frac_original": float(bo.mean()), "pos_frac_recon": float(br.mean()),
    }
    g = "mse" if not grad_w else f"g{grad_w:g}"
    out = out or f"results/synth/vqvae_segcheck_f{factor}_{g}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"regime": (
            "base segmenter run directly on 128px patches (no tiling), "
            "original vs VQ-VAE reconstruction, at the model's val-frozen "
            "threshold. Measures whether the tokenizer preserves the signal "
            "the segmenter consumes."), **res}, f, indent=2)

    print("\n=== does the tokenizer preserve what the segmenter reads? ===")
    print(f"  agreement IoU between the two predictions   {res['agreement_iou']:.4f}")
    print(f"  fraction of pixel decisions that flip       {res['decision_flip_frac']:.4f}")
    print(f"  mean |delta p|                              {res['mean_abs_delta_prob']:.4f}")
    print(f"  correlation of probability maps             {res['corr']:.4f}")
    print(f"  predicted-positive fraction  original {res['pos_frac_original']:.4f}"
          f"  reconstruction {res['pos_frac_recon']:.4f}")
    print(f"\nwrote {out}")
    return res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    fac, gw = DOWNSAMPLE, GRAD_W
    for a in argv:
        if a.startswith("--downsample="):
            fac = int(a.split("=", 1)[1])
        elif a.startswith("--grad_w="):
            gw = float(a.split("=", 1)[1])
    if "--segcheck" in argv:
        return segcheck(factor=fac, grad_w=gw)
    if "--figure" in argv:
        suf = f"_f{fac}_" + ("mse" if not gw else f"g{gw:g}")
        return figure(factor=fac, grad_w=gw,
                      out=f"results/figures/synth_vqvae_recon{suf}.png")
    if "--eval" in argv:
        X, _, _, _ = load_patches()
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X))
        Xh = X[perm[:int(len(X) * HELD_FRAC)]]
        m, _ = load_model(fac, gw)
        return gate(evaluate(m, Xh), baseline_psnr(Xh, fac))
    ep, gw = 30, GRAD_W
    for a in argv:
        if a.startswith("--epochs="):
            ep = int(a.split("=", 1)[1])
        elif a.startswith("--grad_w="):
            gw = float(a.split("=", 1)[1])
    return train(epochs=ep, grad_w=gw, factor=fac)


if __name__ == "__main__":
    main()
