"""Tier C — PyTorch segmentation CNNs (spec v2 roster C; constitution P2/P3).

Binary chondrule (foreground) vs matrix on micro-CT slices. Everything here
obeys the same leakage-free contract as tiers A/B:

  * Normalization is ALWAYS ``raw / 255.0`` — a *fixed* scale, never a
    statistic estimated from any data split. [LEAKAGE][MUST]
  * Training patches are drawn from the TRAIN arrays only; the validation zone
    is used solely for model selection / early stop; no test set is touched
    anywhere in this file. [LEAKAGE][MUST]
  * The ignore (VALID) mask is applied inside every loss so ignored pixels
    contribute exactly zero. [MUST]

Models return raw logits (no sigmoid inside ``forward``); ``predict_proba``
does the sigmoid and returns a [0,1] map, matching the tier-A/B runner API.

Self-test:  python -m src.models_cnn      (architecture shape check only)
"""
from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import metrics

# ---------------------------------------------------------------------------
# Device + determinism (seed everything).
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Inference windows are replicate-padded up to a multiple of this before the
# forward pass. 32 covers encoders down to depth 5 (2**5); a depth-4 U-Net only
# needs 16. Only the edge windows are ever padded, so the cost is negligible.
PAD_MULT = 32

torch.manual_seed(0)
RNG = np.random.default_rng(0)
try:  # deterministic where practical (some MPS kernels have no det. impl.)
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:  # pragma: no cover - older torch without warn_only
    pass


# ---------------------------------------------------------------------------
# Normalization — the single, fixed rule. [LEAKAGE][MUST]
# ---------------------------------------------------------------------------
def normalize(raw):
    """Fixed intensity scaling: raw/255 -> float32 in [0,1].

    No mean/std/percentile is ever estimated from data, so thresholds and
    weights transfer across splits without leakage.
    """
    return raw.astype(np.float32) / 255.0


def _norm(channels):
    """GroupNorm for `channels` feature maps.

    GroupNorm carries no running statistics and does not couple samples within
    a batch, so (unlike BatchNorm) it introduces no batch-statistic leakage.
    Group count divides the channel count; falls back toward 1 if needed.
    """
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv 3x3 -> GroupNorm -> ReLU) x2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018) on a skip connection.

    `g` is the decoder gating signal, `skip` the encoder feature (same spatial
    size here); returns the skip re-weighted by a learned [0,1] attention map.
    """

    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), _norm(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), _norm(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, skip):
        g1 = self.w_g(g)
        x1 = self.w_x(skip)
        if g1.shape[-2:] != x1.shape[-2:]:  # guard off-by-one spatial sizes
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear",
                               align_corners=False)
        att = self.psi(self.relu(g1 + x1))
        return skip * att


# ---------------------------------------------------------------------------
# Architectures.  Both accept in_ch, so 2.5D works by setting in_ch=K.
# ---------------------------------------------------------------------------
class UNet2D(nn.Module):
    """Standard 2D U-Net returning logits (no sigmoid inside forward)."""

    def __init__(self, in_ch=1, base=32, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]  # e.g. 32,64,128,256,512
        self.inc = DoubleConv(in_ch, chs[0])
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList(
            [DoubleConv(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i], chs[i - 1], kernel_size=2, stride=2)
             for i in range(depth, 0, -1)])
        self.up_convs = nn.ModuleList(
            [DoubleConv(chs[i], chs[i - 1]) for i in range(depth, 0, -1)])
        self.outc = nn.Conv2d(chs[0], 1, kernel_size=1)

    @staticmethod
    def _cat(x, skip):
        """Concat decoder feature `x` with `skip`, padding `x` to match size."""
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        skips = [self.inc(x)]
        for i in range(self.depth):
            skips.append(self.downs[i](self.pool(skips[-1])))
        x = skips[-1]  # bottleneck
        for j in range(self.depth):
            x = self.ups[j](x)
            x = self._cat(x, skips[self.depth - 1 - j])
            x = self.up_convs[j](x)
        return self.outc(x)


class AttentionUNet2D(UNet2D):
    """UNet2D with an attention gate on each skip connection."""

    def __init__(self, in_ch=1, base=32, depth=4):
        super().__init__(in_ch=in_ch, base=base, depth=depth)
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.gates = nn.ModuleList(
            [AttentionGate(chs[i - 1], chs[i - 1], max(chs[i - 1] // 2, 1))
             for i in range(depth, 0, -1)])

    def forward(self, x):
        skips = [self.inc(x)]
        for i in range(self.depth):
            skips.append(self.downs[i](self.pool(skips[-1])))
        x = skips[-1]
        for j in range(self.depth):
            x = self.ups[j](x)
            skip = self.gates[j](x, skips[self.depth - 1 - j])
            x = self._cat(x, skip)
            x = self.up_convs[j](x)
        return self.outc(x)


# ---------------------------------------------------------------------------
# 2.5D helper.
# ---------------------------------------------------------------------------
def stack_adjacent(raw01, k):
    """(n,H,W) normalized volume -> (n,k,H,W) of adjacent slices.

    Channel c of output slice i is input slice ``i + (c - k//2)``, with the
    z-stack ends replicate-padded. k must be odd.
    """
    assert k % 2 == 1, "k must be odd"
    n = raw01.shape[0]
    half = k // 2
    idx = np.arange(n)
    chans = [raw01[np.clip(idx + (c - half), 0, n - 1)] for c in range(k)]
    return np.stack(chans, axis=1)  # (n, k, H, W)


# ---------------------------------------------------------------------------
# Losses.  Every loss zeroes out ignored pixels via the VALID mask. [MUST]
# ---------------------------------------------------------------------------
def _masked_mean(per_pixel, valid):
    """Mean of a per-pixel loss over valid pixels only (guarding zero valid)."""
    denom = valid.sum().clamp(min=1.0)
    return (per_pixel * valid).sum() / denom


def bce_loss(logits, target, valid, pos_weight=None):
    """Binary cross-entropy, averaged over valid pixels only."""
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    return _masked_mean(per_pixel, valid)


def dice_loss(logits, target, valid, eps=1.0):
    """Soft foreground Dice on sigmoid(logits); invalid pixels excluded.

    Masking the probability and target by `valid` before the region sums makes
    ignored pixels contribute zero to both intersection and union.

    [CALIBRATION] Dice loss is documented to produce poorly-calibrated,
    overconfident predictions (Yeung et al. 2021, arXiv:2111.00528). Measured on
    this project's own retrained model: 91.0% of test pixels sit below 1e-4,
    7.25% above 0.9999, and only 0.32% in the graded 0.1-0.9 band — and the model
    assigns hard zeros to 38% of chondrule pixels. See `dscpp_loss`.
    """
    prob = torch.sigmoid(logits) * valid
    gt = target * valid
    inter = (prob * gt).sum()
    denom = prob.sum() + gt.sum()
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice


def dscpp_loss(logits, target, valid, gamma=2.0, eps=1.0):
    """DSC++ loss — Dice with the FP/FN penalties raised to `gamma`.

    Yeung et al., "Calibrating the Dice loss to handle neural network
    overconfidence for biomedical image segmentation" (arXiv:2111.00528).
    Standard Dice weights every error linearly, which lets the network drive
    logits to saturation at no extra cost; DSC++ leaves the true-positive term
    alone and exponentiates only the FP and FN terms, so a *confident* mistake
    costs relatively more than a hesitant one. gamma=2 is the paper's
    recommended default (NLL fell ~5x across five datasets with Dice preserved
    or improved).

    [IMPLEMENTATION NOTE] The published equation renders as the exponent applied
    to the SUMMED FP/FN terms. That reading is rejected here for two reasons:
    (a) numerically, a squared pixel-count sum (~1e8) would swamp the linear
    intersection term (~1e4) and pin the loss at 1.0; (b) it would rescale the
    total error uniformly rather than "selectively modulate the penalty
    associated with overconfident, incorrect predictions", which is the paper's
    stated mechanism. The exponent is therefore applied PER PIXEL, before the
    sum. Because p in [0,1], gamma>1 shrinks hesitant errors more than confident
    ones, which is exactly the intended selectivity. `_dscpp_selfcheck` asserts
    that behaviour.
    """
    prob = torch.sigmoid(logits) * valid
    gt = target * valid
    inter = (prob * gt).sum()
    fp = ((prob * (1.0 - gt) * valid) ** gamma).sum()
    fn = (((1.0 - prob) * gt * valid) ** gamma).sum()
    dsc = (2.0 * inter + eps) / (2.0 * inter + fp + fn + eps)
    return 1.0 - dsc


def _dscpp_selfcheck():
    """DSC++ must punish a CONFIDENT wrong pixel more than a hesitant one."""
    valid = torch.ones(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4)          # everything is background
    gt[0, 0, :2, :] = 1.0                 # ...except the top half
    def loss_at(p_wrong, fn_=dscpp_loss):
        logits = torch.full((1, 1, 4, 4), -6.0)          # confident background
        logits[0, 0, :2, :] = 6.0                        # confident foreground
        logits[0, 0, 3, 3] = float(np.log(p_wrong / (1 - p_wrong)))  # one FP
        return float(fn_(logits, gt, valid))
    hesitant, confident = loss_at(0.10), loss_at(0.95)
    assert confident > hesitant, (confident, hesitant)
    d_hes, d_con = loss_at(0.10, dice_loss), loss_at(0.95, dice_loss)
    # the RELATIVE gap between a confident and a hesitant error must widen
    assert (confident - hesitant) > (d_con - d_hes), \
        f"DSC++ gap {confident-hesitant:.5f} !> Dice gap {d_con-d_hes:.5f}"
    return {"dscpp_hesitant": hesitant, "dscpp_confident": confident,
            "dice_hesitant": d_hes, "dice_confident": d_con}


def tversky_loss(logits, target, valid, alpha=0.3, beta=0.7, eps=1.0):
    """Soft Tversky loss; beta>alpha penalizes false negatives (favors recall).

    `prob` and `gt` are pre-masked by `valid`, so invalid pixels drop out of the
    TP/FP/FN sums entirely.
    """
    prob = torch.sigmoid(logits) * valid
    gt = target * valid
    tp = (prob * gt).sum()
    fp = (prob * (1.0 - gt)).sum()
    fn = ((1.0 - prob) * gt).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky


def focal_tversky_loss(logits, target, valid, alpha=0.1, beta=0.9, gamma=0.75):
    """Focal-Tversky: (1 - Tversky)^gamma, with beta >> alpha favoring RECALL.

    beta weights false NEGATIVES, alpha weights false positives, so beta >> alpha
    trains the model to over-predict rather than miss — the regime the
    over-predict-then-prune pipeline wants (a missed chondrule is unrecoverable;
    a false alarm is prunable). gamma < 1 focuses gradient on the hard cases.
    """
    t = 1.0 - tversky_loss(logits, target, valid, alpha=alpha, beta=beta)
    return torch.pow(torch.clamp(1.0 - t, min=1e-7), gamma)


def tversky_recall_loss(logits, target, valid):
    """Tversky tuned hard for recall: a miss costs 9x a false alarm."""
    return tversky_loss(logits, target, valid, alpha=0.1, beta=0.9)


def combo_loss(logits, target, valid, pos_weight=None):
    """Weighted BCE + Dice. With pos_weight > 1 the BCE half penalizes misses."""
    return (bce_loss(logits, target, valid, pos_weight=pos_weight)
            + dice_loss(logits, target, valid))


def focal_loss(logits, target, valid, gamma=2.0, alpha=0.25):
    """Focal loss, averaged over valid pixels only."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    per_pixel = alpha_t * (1.0 - p_t) ** gamma * bce
    return _masked_mean(per_pixel, valid)


# ---------------------------------------------------------------------------
# Patch sampler.  Draws P x P patches from the GIVEN arrays ONLY. [LEAKAGE][MUST]
# The caller passes TRAIN arrays here — never val/test.
# ---------------------------------------------------------------------------
class PatchSet(Dataset):
    """Random P x P patch sampler over one set of (R, M, V) arrays.

    With probability `pos_frac` a patch is centered on a random VALID positive
    pixel (M & V); otherwise on a random location. Slices are reflect-padded so
    any center yields a full patch. For 2.5D (k>1) the input channels are the k
    adjacent slices at the SAME patch location (z-ends replicate-padded).

    `aux` is an optional (n,H,W) or (n,C,H,W) float32 volume aligned with `R`
    (e.g. a coarse GNN map for the cascade): when given it is appended as C
    EXTRA input channels after the k raw channels, so the input becomes
    (k+C, P, P).
    """

    def __init__(self, R, M, V, patch=128, pos_frac=0.5, length=2000, seed=0,
                 k=1, augment=False, dilate_target=0, aux=None,
                 invert_target=False, hard_frac=0.0, hard_bright=45):
        assert k % 2 == 1, "k must be odd"
        self.augment = augment
        self.hard_frac = float(hard_frac)
        self._invert = bool(invert_target)
        if dilate_target:
            # Grow the TRAINING target outward by `dilate_target` px so the model
            # learns to over-cover boundaries. Deliberate asymmetry for the
            # over-predict-then-prune pipeline: an outward error is prunable, an
            # inward one is unrecoverable. Only the training labels are dilated —
            # VALIDATION AND TEST ARE ALWAYS SCORED AGAINST THE TRUE MASK.
            from scipy import ndimage as _ndi
            M = np.stack([_ndi.binary_dilation(M[i], iterations=dilate_target)
                          for i in range(M.shape[0])])
        self.n, self.H, self.W = R.shape
        self.patch = patch
        self.pos_frac = pos_frac
        self.length = length
        self.k = k
        self.rng = np.random.default_rng(seed)
        half = patch // 2
        # Fixed normalization (raw/255) then reflect-pad in-plane. [LEAKAGE][MUST]
        r01 = normalize(R)
        pad = ((0, 0), (half, half), (half, half))
        self.Rp = np.pad(r01, pad, mode="reflect")
        # The matrix specialist predicts NOT-chondrule. This MUST happen before
        # the target is padded — inverting afterwards leaves `Mp` untouched and
        # the flag silently does nothing. Sampling coords are still derived from
        # the ORIGINAL chondrule mask below, so it keeps looking at the same
        # (confusable) places while learning the complementary class.
        M_t = ((~M) & V) if self._invert else M
        self.Mp = np.pad(M_t, pad, mode="reflect")
        self.Vp = np.pad(V, pad, mode="reflect")
        # Auxiliary channel volume, padded exactly like R so it stays aligned.
        # A (n,H,W) aux is promoted to (n,1,H,W) so one channel and many take
        # the identical path; the padded values are unchanged either way.
        self.Auxp = None
        if aux is not None:
            a = np.asarray(aux, np.float32)
            if a.ndim == 3:
                a = a[:, None]
            if a.ndim != 4:
                raise ValueError(f"aux must be (n,H,W) or (n,C,H,W); got {a.shape}")
            self.Auxp = np.pad(a, ((0, 0), (0, 0), (half, half), (half, half)),
                               mode="reflect")
        # Precompute valid-positive coordinates per slice (original, unpadded).
        self.pos_slices, self.pos_coords = [], {}
        for i in range(self.n):
            pv = np.argwhere(M[i] & V[i])
            if len(pv):
                self.pos_slices.append(i)
                self.pos_coords[i] = pv

        # Hard-negative coordinates for the MATRIX specialist: valid matrix
        # pixels that either touch a chondrule (dilated boundary ring) or are
        # bright enough to fall in the chondrule intensity range. These are
        # exactly the pixels a chondrule model gets wrong, so a model trained
        # here has decorrelated errors and can act as a genuine veto.
        self.hard_slices, self.hard_coords = [], {}
        if self.hard_frac > 0:
            from scipy import ndimage as _ndi
            for i in range(self.n):
                ring = _ndi.binary_dilation(M[i], iterations=4) & ~M[i]
                bright = (R[i] >= hard_bright) & ~M[i]
                hv = np.argwhere((ring | bright) & V[i])
                if len(hv):
                    self.hard_slices.append(i)
                    self.hard_coords[i] = hv


    def __len__(self):
        return self.length

    def sample(self):
        """Draw one (input, target, valid) triple of tensors."""
        if self.hard_slices and self.rng.random() < self.hard_frac:
            # HARD-NEGATIVE draw (matrix-specialist training). Centres the patch
            # on matrix pixels that actually confuse a chondrule model: those
            # bordering a chondrule, or bright enough to sit in the chondrule
            # intensity range (measured overlap: matrix p90 = 55 CT vs chondrule
            # mean ~47). Training a second model on THIS region is what makes its
            # errors decorrelate from the chondrule model's — an inverted copy of
            # the same model would just be 1 - p and veto nothing.
            i = int(self.hard_slices[self.rng.integers(len(self.hard_slices))])
            coords = self.hard_coords[i]
            y, x = coords[self.rng.integers(len(coords))]
            y, x = int(y), int(x)
        elif self.pos_slices and self.rng.random() < self.pos_frac:
            i = int(self.pos_slices[self.rng.integers(len(self.pos_slices))])
            coords = self.pos_coords[i]
            y, x = coords[self.rng.integers(len(coords))]
            y, x = int(y), int(x)
        else:
            i = int(self.rng.integers(self.n))
            y = int(self.rng.integers(self.H))
            x = int(self.rng.integers(self.W))
        p = self.patch
        # k adjacent slices at the same (y, x); z-ends replicate-padded.
        chans = []
        for c in range(self.k):
            j = min(max(i + (c - self.k // 2), 0), self.n - 1)
            chans.append(self.Rp[j, y:y + p, x:x + p])
        inp = np.stack(chans, axis=0).astype(np.float32)
        tgt = self.Mp[i, y:y + p, x:x + p][None].astype(np.float32)
        val = self.Vp[i, y:y + p, x:x + p][None].astype(np.float32)
        aux = (None if self.Auxp is None else
               self.Auxp[i, :, y:y + p, x:x + p].astype(np.float32))

        if self.augment:
            # augment="geom" -> flips/rotations only; True/"full" -> also
            # intensity jitter. Kept separable because round 1 showed the full
            # bundle HALVED val IoU (0.387 -> 0.202), and intensity is the main
            # discriminative signal here (chondrules ~50 vs matrix ~32), so
            # jitter is the prime suspect rather than the geometry.
            #
            # Dihedral group: CT texture has no privileged orientation, so
            # flips/rotations are label-preserving. Applied identically to
            # input, target and valid mask.
            if self.rng.random() < 0.5:
                inp, tgt, val = inp[:, ::-1], tgt[:, ::-1], val[:, ::-1]
                aux = aux if aux is None else aux[:, ::-1]
            if self.rng.random() < 0.5:
                inp, tgt, val = inp[:, :, ::-1], tgt[:, :, ::-1], val[:, :, ::-1]
                aux = aux if aux is None else aux[:, :, ::-1]
            r = int(self.rng.integers(4))
            if r:
                inp = np.rot90(inp, r, axes=(1, 2))
                tgt = np.rot90(tgt, r, axes=(1, 2))
                val = np.rot90(val, r, axes=(1, 2))
                aux = aux if aux is None else np.rot90(aux, r, axes=(1, 2))
            # Mild intensity jitter on the INPUT only (never the labels).
            if self.augment != "geom":
                inp = np.clip(inp * (1.0 + 0.2 * (self.rng.random() - 0.5))
                              + 0.1 * (self.rng.random() - 0.5), 0.0, 1.0)

        if aux is not None:
            # The aux channel is spatially aligned, so it takes the SAME geometry
            # above; it is concatenated AFTER the jitter block because intensity
            # jitter must never touch it (it is a probability map, not intensity).
            inp = np.concatenate([inp, aux], axis=0)

        inp = np.ascontiguousarray(inp, dtype=np.float32)
        tgt = np.ascontiguousarray(tgt, dtype=np.float32)
        val = np.ascontiguousarray(val, dtype=np.float32)
        return (torch.from_numpy(inp),      # (k [+C aux], P, P)
                torch.from_numpy(tgt),      # (1, P, P)
                torch.from_numpy(val))      # (1, P, P)

    def __getitem__(self, idx):  # idx ignored: sampling is random by design
        return self.sample()


def _draw_batch(dataset, batch):
    """Stack `batch` random samples into (B,k,P,P), (B,1,P,P), (B,1,P,P)."""
    xs, ys, vs = [], [], []
    for _ in range(batch):
        x, y, v = dataset.sample()
        xs.append(x)
        ys.append(y)
        vs.append(v)
    return torch.stack(xs), torch.stack(ys), torch.stack(vs)


def _best_val_iou(prob, mask, valid,
                  # The epoch-by-epoch CHECKPOINT is chosen with this grid, so it
                  # must span where the model actually operates. The old default
                  # (0.3-0.7) was 1-2 orders of magnitude above the eventual
                  # frozen thresholds (0.002-0.05 for dice-trained CNNs), so
                  # `best_state` was selected by IoU measured nowhere near the
                  # operating point — potentially degrading every model in the
                  # sweep. Models already trained under the old default carry
                  # this defect; see CHANGELOG.
                  grid=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                        0.1, 0.2, 0.3, 0.5, 0.7)):
    """Best foreground IoU over {0.5} u grid, via metrics.iou (valid honored)."""
    best_t, best_v = 0.5, -1.0
    for t in sorted(set((0.5,) + tuple(grid))):
        v = metrics.iou(prob >= t, mask, valid)
        if v == v and v > best_v:  # skip NaN
            best_v, best_t = v, float(t)
    return best_t, best_v


# ---------------------------------------------------------------------------
# Runner (matches the tier-A/B fit / predict_proba API).
# ---------------------------------------------------------------------------
class CNNRunner:
    """Trains one CNN and exposes predict_proba -> (n,H,W) in [0,1].

    `make_model(in_ch)` builds the net; `loss_fn(logits, target, valid)` is the
    masked loss.  `.tier` = "C_cnn"; `.name` identifies the run.

    `aux_lookup` is an optional callable returning a (H,W) or (C,H,W) float32
    aux map for one slice. When set, the aux map of every slice is fed to the
    net as C EXTRA input channels (so `in_ch = k + C`), both while training and
    at inference. `aux_ch` declares C; it defaults to 1, the historical
    behaviour, and a (H,W) return is promoted to (1,H,W).
    """

    def __init__(self, name, make_model, loss_fn, k=1, epochs=15,
                 steps_per_epoch=200, batch=16, patch=128, pos_frac=0.5,
                 lr=1e-3, pos_weight=None, seed=0, val_stride=4,
                 weight_decay=0.0, augment=False, dilate_target=0,
                 aux_lookup=None, aux_ch=1, invert_target=False, hard_frac=0.0,
                 ckpt_dir=None):
        self.name = name
        self.tier = "C_cnn"
        self.make_model = make_model
        self.loss_fn = loss_fn
        self.k = k
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch = batch
        self.patch = patch
        self.pos_frac = pos_frac
        self.lr = lr
        self.val_stride = val_stride
        self.weight_decay = weight_decay
        self.augment = augment
        self.dilate_target = dilate_target
        self.invert_target = invert_target   # matrix specialist: predict NOT-chondrule
        self.hard_frac = hard_frac           # fraction of patches on confusable matrix
        self.pos_weight = pos_weight
        self.aux_lookup = aux_lookup
        self.aux_ch = int(aux_ch)
        self.seed = seed
        # Weights were previously kept in memory only, so every trained model
        # died with its process. Applying a model to a NEW meteorite (or to the
        # ~1,590 untouched raw slices) means retraining from scratch. Persist
        # them instead.
        self.ckpt_dir = ckpt_dir
        self.model = None
        self.best_state = None
        self.best_thr = 0.5
        self.best_full_val_iou = float("nan")

    def fit(self, train, val):
        """Train on `train` ONLY; use `val` ONLY for model selection. [LEAKAGE]

        train/val are dicts {"raw":(n,H,W) float32, "mask":bool, "valid":bool}.
        No test set is passed or referenced anywhere in this method.
        """
        torch.manual_seed(self.seed)
        in_ch = self.k + (self.aux_ch if self.aux_lookup is not None else 0)
        self.model = self.make_model(in_ch).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        # Cosine LR decay: a constant lr=1e-3 made val IoU oscillate badly.
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        # --- TRAIN-ONLY data: patches sampled strictly from `train`. ---
        aux = None if self.aux_lookup is None else self._aux_volume(train["raw"])
        ds = PatchSet(train["raw"], train["mask"], train["valid"],
                      patch=self.patch, pos_frac=self.pos_frac,
                      length=self.steps_per_epoch * self.batch,
                      seed=self.seed, k=self.k, augment=self.augment,
                      dilate_target=self.dilate_target, aux=aux,
                      invert_target=self.invert_target,
                      hard_frac=self.hard_frac)

        # Cheap per-epoch selection on a VAL SUBSET; the FULL val set is used
        # once at the end to freeze the operating threshold. Still strictly
        # val-only — no test data is touched anywhere here. [LEAKAGE][MUST]
        s = slice(None, None, max(1, self.val_stride))
        vsub = (val["raw"][s], val["mask"][s], val["valid"][s])

        best_iou = -1.0
        for ep in range(self.epochs):
            t0 = time.time()
            self.model.train()
            for _ in range(self.steps_per_epoch):
                xb, yb, vb = _draw_batch(ds, self.batch)
                xb, yb, vb = xb.to(DEVICE), yb.to(DEVICE), vb.to(DEVICE)
                opt.zero_grad()
                # pos_weight was previously accepted, stored, and never used —
                # a dead knob. Thread it through for losses that accept it, so
                # recall-oriented training (penalize a MISS more than a false
                # alarm) actually takes effect.
                if self.pos_weight is None:
                    loss = self.loss_fn(self.model(xb), yb, vb)
                else:
                    pw = torch.as_tensor(float(self.pos_weight), device=DEVICE)
                    loss = self.loss_fn(self.model(xb), yb, vb, pos_weight=pw)
                loss.backward()
                opt.step()
            sched.step()

            # --- VAL-ONLY selection (subset): keep best-by-val-IoU. ---
            prob = self.predict_proba(vsub[0])
            thr, viou = _best_val_iou(prob, vsub[1], vsub[2])
            if viou == viou and viou > best_iou:
                best_iou, self.best_thr = viou, thr
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            print(f"    [{self.name}] epoch {ep + 1}/{self.epochs} "
                  f"valIoU_sub={viou:.4f} (thr={thr:.2f}) best={best_iou:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {time.time() - t0:.1f}s",
                  flush=True)

        if self.best_state is not None:  # restore best-by-val weights
            self.model.load_state_dict(self.best_state)
        # Freeze the operating threshold on the FULL val set with a finer grid.
        prob_full = self.predict_proba(val["raw"])
        self.best_thr, full_iou = _best_val_iou(
            prob_full, val["mask"], val["valid"],
            # Grid extended well below 0.05: the first long runs both selected
            # exactly 0.05, the old grid's minimum — a parameter pinned to its
            # boundary means the real optimum was outside the search. Dice loss
            # under ~95% background pushes probabilities low, so the useful
            # operating points genuinely live down here.
            grid=tuple(float(x) for x in np.concatenate([
                np.arange(0.002, 0.05, 0.004), np.arange(0.05, 0.96, 0.05)])))
        self.best_full_val_iou = float(full_iou)
        print(f"    [{self.name}] FULL-val IoU={full_iou:.4f} "
              f"frozen thr={self.best_thr:.2f}", flush=True)
        if self.ckpt_dir is not None:
            self.save_checkpoint(os.path.join(self.ckpt_dir, f"{self.name}.pt"))
        return self

    # ---- checkpoints -------------------------------------------------
    # `make_model` is a closure and cannot be pickled, so a checkpoint stores
    # weights + the frozen operating point + enough config to VERIFY that the
    # runner you rebuild matches. Reload is therefore:
    #     r = CNNRunner(name, make_model, loss_fn, k=..., ...)
    #     r.load_checkpoint(path)      # no fit() needed
    #     probs = r.predict_proba(raw_of_a_new_meteorite)

    CKPT_CFG = ("name", "k", "patch", "lr", "epochs", "steps_per_epoch",
                "batch", "pos_frac", "seed", "augment", "dilate_target",
                "invert_target", "hard_frac", "weight_decay")

    def save_checkpoint(self, path):
        """Write best-by-val weights + frozen threshold to `path`."""
        state = self.best_state
        if state is None and self.model is not None:
            state = {k: v.detach().cpu().clone()
                     for k, v in self.model.state_dict().items()}
        assert state is not None, "nothing to save: call fit() first"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "format": 1,
            "state_dict": state,
            # The threshold is part of the model: it was fit on validation and
            # must travel with the weights, or the reloaded model is scored at
            # an arbitrary operating point. [LEAKAGE][MUST]
            "best_thr": float(self.best_thr),
            "val_iou": float(self.best_full_val_iou),
            "in_ch": self.k + (self.aux_ch if self.aux_lookup is not None else 0),
            "has_aux": self.aux_lookup is not None,
            "aux_ch": self.aux_ch,
            "config": {c: getattr(self, c) for c in self.CKPT_CFG},
        }, path)
        print(f"    [{self.name}] checkpoint -> {path}", flush=True)
        return path

    def load_checkpoint(self, path, strict_config=True):
        """Restore weights + threshold. Rebuilds the net; no fit() required."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        in_ch = self.k + (self.aux_ch if self.aux_lookup is not None else 0)
        if in_ch != ck["in_ch"]:
            raise ValueError(
                f"input channels differ: runner={in_ch} checkpoint={ck['in_ch']}"
                ": k, aux_lookup and aux_ch must match how it was trained")
        if strict_config:
            bad = {c: (getattr(self, c), ck["config"][c]) for c in self.CKPT_CFG
                   if c in ck["config"] and getattr(self, c) != ck["config"][c]}
            # Only inference-relevant mismatches are fatal; budget knobs are not.
            fatal = {c: v for c, v in bad.items()
                     if c in ("k", "patch", "invert_target")}
            if fatal:
                raise ValueError(f"checkpoint config mismatch: {fatal}")
            if bad:
                print(f"    [{self.name}] note: benign config drift {bad}",
                      flush=True)
        self.model = self.make_model(in_ch).to(DEVICE)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.best_state = ck["state_dict"]
        self.best_thr = float(ck["best_thr"])
        self.best_full_val_iou = float(ck.get("val_iou", float("nan")))
        print(f"    [{self.name}] loaded {path} "
              f"(thr={self.best_thr:.3f}, val IoU={self.best_full_val_iou:.4f})",
              flush=True)
        return self

    def _aux_volume(self, raw):
        """(n,H,W) raw -> (n,C,H,W) float32 aux channels. [shared by both paths]

        `aux_lookup` may return (H,W) for a single channel or (C,H,W) for N. The
        single-channel return is promoted so the rest of the path is shape-agnostic
        and the one-channel tensor stays bit-identical to what it was.
        """
        a = np.stack([np.asarray(self.aux_lookup(raw[i]), np.float32)
                      for i in range(raw.shape[0])])
        if a.ndim == 3:                      # (n,H,W): one channel, historical shape
            a = a[:, None]
        if a.ndim != 4:
            raise ValueError(f"aux_lookup must return (H,W) or (C,H,W); "
                             f"stacked shape was {a.shape}")
        if a.shape[1] != self.aux_ch:
            raise ValueError(f"aux_lookup returned {a.shape[1]} channel(s) but "
                             f"aux_ch={self.aux_ch} was declared")
        return a

    def _channels(self, raw):
        """(n,H,W) raw -> (n,C,H,W) float32 network input. [shared by both paths]"""
        vol = stack_adjacent(normalize(raw), self.k)   # (n,k,H,W), fixed scale
        if self.aux_lookup is not None:                # extra channels -> (n,k+C,H,W)
            vol = np.concatenate([vol, self._aux_volume(raw)], axis=1)
        return vol

    def predict_proba_whole(self, raw):
        """Whole-slice single forward pass. **DO NOT USE on these models.**

        Replicate-pads each slice to a multiple of PAD_MULT and runs ONE forward
        pass, so the output has no tile-grid dependence at all. Kept only for the
        record and for architectures that are not patch-trained.

        [!] MEASURED TO COLLAPSE TRAINED-MODEL OUTPUT (2026-07-29). It was made
        the default for about an hour and reverted. `ds2_f25_s0`, same config and
        seed, but TWO SEPARATE TRAINING RUNS, not one model scored two ways: the
        same-weights comparison was never run. Best epoch reached was 0.4487
        tiled versus 0.2632 whole-slice; the epochs below are the worst of the
        collapsed run and must not be quoted as the headline (an earlier version
        of this docstring quoted 0.0004 alone, which was a cherry-pick):

            epoch    tiled          whole-slice
              8      0.4476         0.0000
             11      0.4487         0.0004
             13      0.3762         0.2632   <- best the collapsed run reached
             15      0.4403         0.0075

        Cause: these nets train on 128x128 patches at `pos_frac=0.5`, and
        `_norm` is **GroupNorm**, whose statistics are computed over whatever
        spatial extent it is handed. A whole 962x1450 slice is ~110x larger and
        ~5% foreground instead of 50%, so the normalised features land far
        outside the training distribution and the logits saturate. Worse, the
        collapse happens INSIDE `fit()` — the per-epoch val pass uses this
        method, so `best_state` and `best_thr` are selected on collapsed maps.

        The lesson: with GroupNorm plus patch training, the inference window size
        is effectively a **model hyperparameter**, not an implementation detail.
        Matching the training window's statistics matters more than eliminating
        tile-grid dependence. An untrained-net probability diff cannot detect
        this — it is a decision-boundary collapse, not a smoothness artifact.
        """
        assert self.model is not None, "call fit() before predict_proba_whole"
        self.model.eval()
        vol = self._channels(raw)
        n, _, H, W = vol.shape
        ph, pw = (-H) % PAD_MULT, (-W) % PAD_MULT
        out = np.zeros((n, H, W), np.float32)
        with torch.no_grad():
            for i in range(n):
                x = torch.from_numpy(np.ascontiguousarray(vol[i][None])).to(DEVICE)
                if ph or pw:
                    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
                p = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
                out[i] = p[:H, :W]
        return out

    def predict_proba(self, raw, tile=256, halo=32):
        """(n,H,W) raw -> (n,H,W) float32 probabilities in [0,1]. THE default path.

        Tiled inference: 256-px cores, each read with a 32-px halo of context
        which is then CROPPED, so every output pixel comes from exactly one
        window's core and `cnt == 1` everywhere. Edge windows are replicate-padded
        up to PAD_MULT so the decoder does not pad asymmetrically. Handles
        arbitrary H,W; builds the 2.5D input channels internally when k>1.

        Two defects, one fixed and one deliberately accepted (2026-07-29)
        ----------------------------------------------------------------
        1. FIXED — the halo used to be ACCUMULATED rather than cropped. With
           stride == tile, overlap sat in 64-px bands where the divisor jumped
           1 -> 2 -> 4 while averaging in each window's ZERO-PADDED borders. That
           produced visible axis-aligned seams: 5-7 of the 8 largest per-column
           jumps landed exactly on a window boundary, versus 0-1 of 8 for GNN
           models that bypass this tiler. Cropping removed it — measured 1/8 on a
           boundary afterwards, boundary column jump only 1.1x the median.

        2. ACCEPTED — `_norm` is GroupNorm, so each window is normalised by its
           OWN statistics and predictions retain a diffuse dependence on where
           the tile grid falls (untrained probe: mean|dp| 0.024, max 0.286 for a
           128-px grid shift). The obvious cure — one whole-slice forward pass —
           was tried and **made things far worse**, taking best-epoch val IoU
           from 0.4487 to 0.2632 on a 25%-data run (worst epochs 0.0000-0.0075),
           across two separate runs at the same config and seed. See
           `predict_proba_whole` for the numbers and the reason. A 256-px window is close to the 128-px training patch,
           so tiling keeps inference IN-DISTRIBUTION for GroupNorm; that matters
           more than grid-invariance. Fixing this properly means changing the
           normalisation or the training window, not the inference stitching.

        [!] Every CNN-family number produced before the halo crop was measured
        with the defective blend, including the 0.3168 headline.
        """
        assert self.model is not None, "call fit() before predict_proba"
        self.model.eval()
        vol = self._channels(raw)
        n, _, H, W = vol.shape
        out = np.zeros((n, H, W), np.float32)
        with torch.no_grad():
            for i in range(n):
                acc = np.zeros((H, W), np.float32)
                cnt = np.zeros((H, W), np.float32)
                for y0 in range(0, H, tile):
                    for x0 in range(0, W, tile):
                        # Padded window (with halo) is what the net sees...
                        ya, yb = max(0, y0 - halo), min(H, y0 + tile + halo)
                        xa, xb = max(0, x0 - halo), min(W, x0 + tile + halo)
                        win = np.ascontiguousarray(vol[i, :, ya:yb, xa:xb][None])
                        wh, ww = win.shape[-2:]
                        ph, pw = (-wh) % PAD_MULT, (-ww) % PAD_MULT
                        if ph or pw:
                            win = np.pad(win, ((0, 0), (0, 0), (0, ph), (0, pw)),
                                         mode="edge")
                        t = torch.from_numpy(win).to(DEVICE)
                        p = torch.sigmoid(self.model(t)).cpu().numpy()[0, 0]
                        if ph or pw:
                            p = p[:wh, :ww]
                        # ...but only its CORE is kept, so cores tile once.
                        cy0, cy1 = y0, min(H, y0 + tile)
                        cx0, cx1 = x0, min(W, x0 + tile)
                        acc[cy0:cy1, cx0:cx1] += p[cy0 - ya:cy1 - ya,
                                                   cx0 - xa:cx1 - xa]
                        cnt[cy0:cy1, cx0:cx1] += 1.0
                assert cnt.min() >= 1.0, "tiling left an uncovered pixel"
                out[i] = acc / cnt
        return out


# ---------------------------------------------------------------------------
# Configured roster.
# ---------------------------------------------------------------------------
def cnn_models(epochs=15):
    """Return the tier-C runners. `epochs` kept modest so they finish on MPS.

    Loss defaults already match the spec (Tversky alpha=0.3/beta=0.7 favors
    recall; focal gamma=2/alpha=0.25), so the loss functions are passed as-is.
    """
    def unet(in_ch):
        return UNet2D(in_ch=in_ch, base=32, depth=4)

    def attention(in_ch):
        return AttentionUNet2D(in_ch=in_ch, base=32)

    return [
        CNNRunner("unet2d_dice", unet, dice_loss, k=1, epochs=epochs),
        CNNRunner("unet2d_tversky", unet, tversky_loss, k=1, epochs=epochs),
        CNNRunner("unet2d_focal", unet, focal_loss, k=1, epochs=epochs),
        CNNRunner("unet25d_dice", unet, dice_loss, k=5, epochs=epochs),
        CNNRunner("attention_unet_dice", attention, dice_loss, k=1, epochs=epochs),
    ]


# ---------------------------------------------------------------------------
# Shape self-test only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    archs = [
        ("UNet2D(in_ch=1)", UNet2D(in_ch=1, base=32, depth=4), 1),
        ("AttentionUNet2D(in_ch=1)", AttentionUNet2D(in_ch=1, base=32), 1),
        ("UNet2D(in_ch=5) [2.5D]", UNet2D(in_ch=5, base=32, depth=4), 5),
        ("AttentionUNet2D(in_ch=5)", AttentionUNet2D(in_ch=5, base=32), 5),
    ]
    for label, net, in_ch in archs:
        net.eval()
        x = torch.randn(2, in_ch, 128, 128)
        with torch.no_grad():
            y = net(x)
        assert tuple(y.shape) == (2, 1, 128, 128), (label, tuple(y.shape))
        print(f"  OK  {label:26s} -> {tuple(y.shape)}")
    print("shape self-test: OK")
