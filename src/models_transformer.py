"""Tier C: FROM-SCRATCH attention-based segmentation architectures.

Same contract as ``models_cnn``: every architecture takes ``(in_ch=1, base=32)``,
``forward`` consumes ``(B, in_ch, H, W)`` floats and returns ``(B, 1, H, W)`` raw
LOGITS (no sigmoid inside forward), so these drop straight into ``CNNRunner``
via ``make_model(in_ch)``.

  * NO pretrained weights, NO model zoo, NO downloads. Every parameter here is
    randomly initialized and trained on this dataset only. [LEAKAGE][MUST]
  * LayerNorm / GroupNorm only, never BatchNorm: no running statistics and no
    coupling of samples within a batch. [LEAKAGE][MUST]
  * H and W are ARBITRARY. Full 320x320 tiles and ragged edge tiles (288x224,
    100x140, ...) alike are handled by reflect-padding up to the patch/window
    multiple internally and cropping (or resampling) back, so the output is
    EXACTLY (B, 1, H, W).

Models are deliberately small (well under ~5M parameters each) so they train on
Apple MPS at 128x128 patches, batch 16.

Self-test:  python -m src.models_transformer     (shape + param count only)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels):
    """GroupNorm for `channels` feature maps (never BatchNorm).

    GroupNorm carries no running statistics and does not couple samples within
    a batch. Group count divides the channel count; falls back toward 1.
    """
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


# ---------------------------------------------------------------------------
# Shared building blocks.
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    """Bilinear upsample -> Conv 3x3 -> GroupNorm -> GELU.

    `size=None` doubles the spatial size; passing an explicit ``(h, w)`` lands
    on an exact target, which is how the decoders hit arbitrary (H, W).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = _norm(out_ch)
        self.act = nn.GELU()

    def forward(self, x, size=None):
        if size is None:
            size = (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


class _DoubleConv(nn.Module):
    """(Conv 3x3 -> GroupNorm -> GELU) x2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class _EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block: MHSA + MLP, dropout 0.0."""

    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


def _sincos_pos_embed(dim, gh, gw, device, dtype):
    """2D sin-cos positional embedding for a gh x gw token grid -> (1, gh*gw, dim).

    Built for the ACTUAL grid at every forward pass, so no fixed sequence length
    is baked in and variable H, W need no interpolation of a learned table.
    """
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = torch.exp(-math.log(10000.0) * omega / quarter)
    ys = torch.arange(gh, device=device, dtype=torch.float32)
    xs = torch.arange(gw, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    oy = yy.reshape(-1)[:, None] * omega[None, :]
    ox = xx.reshape(-1)[:, None] * omega[None, :]
    pe = torch.cat([torch.sin(oy), torch.cos(oy),
                    torch.sin(ox), torch.cos(ox)], dim=1)
    return pe[None].to(dtype)


# ---------------------------------------------------------------------------
# 1. ViTSeg: from-scratch plain ViT encoder + convolutional decoder.
# ---------------------------------------------------------------------------
class ViTSeg(nn.Module):
    """Patch-embed -> 4 pre-LN transformer blocks -> 3-stage conv decoder.

    patch=8, dim=base*4. The input is reflect-padded up to a multiple of the
    patch size and the decoder output is cropped back to (H, W).
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.patch = 8
        dim = base * 4
        self.dim = dim
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=self.patch, stride=self.patch)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(dim, heads=4, mlp_ratio=2) for _ in range(4)])
        chs = [dim, dim // 2, dim // 4, base]
        self.ups = nn.ModuleList(
            [_UpBlock(chs[i], chs[i + 1]) for i in range(3)])
        self.outc = nn.Conv2d(chs[-1], 1, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        ph = (self.patch - H % self.patch) % self.patch
        pw = (self.patch - W % self.patch) % self.patch
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        t = self.proj(x)                              # (B, dim, gh, gw)
        gh, gw = t.shape[-2], t.shape[-1]
        tok = t.flatten(2).transpose(1, 2)            # (B, gh*gw, dim)
        tok = tok + _sincos_pos_embed(self.dim, gh, gw, tok.device, tok.dtype)
        for blk in self.blocks:
            tok = blk(tok)
        f = tok.transpose(1, 2).reshape(B, self.dim, gh, gw)
        for up in self.ups:                           # 3 x2 upsamples = x8
            f = up(f)
        return self.outc(f)[:, :, :H, :W]


# ---------------------------------------------------------------------------
# 2. SwinLiteSeg: hierarchical windowed attention.
# ---------------------------------------------------------------------------
def _window_partition(x, ws):
    """(B, H, W, C) with H, W multiples of ws -> (B*nW, ws*ws, C), b-major."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def _window_reverse(win, ws, B, H, W):
    """(B*nW, ws*ws, C) -> (B, H, W, C); inverse of `_window_partition`."""
    C = win.shape[-1]
    x = win.reshape(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


def _shift_attn_mask(H, W, ws, shift, device, dtype):
    """Standard Swin cyclic-shift attention mask -> (nW, ws*ws, ws*ws).

    Tokens that the roll made non-adjacent get -100 so they cannot attend to
    each other inside the same window.
    """
    img = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for xs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, hs, xs, :] = cnt
            cnt += 1
    win = _window_partition(img, ws).squeeze(-1)      # (nW, ws*ws)
    mask = win.unsqueeze(1) - win.unsqueeze(2)        # (nW, ws*ws, ws*ws)
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class _WindowBlock(nn.Module):
    """Pre-LN window (or shifted-window) attention block + MLP.

    The feature map is reflect-padded up to a multiple of `window` inside the
    block and cropped back on the way out, so any H, W works.
    """

    def __init__(self, dim, heads=4, window=7, shift=0, mlp_ratio=2):
        super().__init__()
        self.heads = heads
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        ws = self.window
        ph = (ws - H % ws) % ws
        pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = x.shape[-2], x.shape[-1]
        t = x.permute(0, 2, 3, 1)                     # (B, Hp, Wp, C)

        h = self.norm1(t)
        attn_mask = None
        if self.shift:
            h = torch.roll(h, shifts=(-self.shift, -self.shift), dims=(1, 2))
            m = _shift_attn_mask(Hp, Wp, ws, self.shift, h.device, h.dtype)
            # (nW,L,S) -> (B*nW,L,S) -> (B*nW*heads,L,S), batch-major/head-minor
            attn_mask = m.repeat(B, 1, 1).repeat_interleave(self.heads, dim=0)
        win = _window_partition(h, ws)
        win = self.attn(win, win, win, attn_mask=attn_mask,
                        need_weights=False)[0]
        h = _window_reverse(win, ws, B, Hp, Wp)
        if self.shift:
            h = torch.roll(h, shifts=(self.shift, self.shift), dims=(1, 2))

        t = t + h
        t = t + self.mlp(self.norm2(t))
        if ph or pw:
            t = t[:, :H, :W, :]
        return t.permute(0, 3, 1, 2).contiguous()


class _PatchMerging(nn.Module):
    """2x2 neighborhood concat + linear (4C -> 2C), halving the resolution."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        if H % 2 or W % 2:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")
        t = x.permute(0, 2, 3, 1)                     # (B, H, W, C)
        t = torch.cat([t[:, 0::2, 0::2, :], t[:, 1::2, 0::2, :],
                       t[:, 0::2, 1::2, :], t[:, 1::2, 1::2, :]], dim=-1)
        t = self.reduction(t)
        return t.permute(0, 3, 1, 2).contiguous()


class SwinLiteSeg(nn.Module):
    """Two windowed-attention stages (window 7, shifted second block) + decoder.

    Patch-embed at stride 4 to dim=base*2; stage 1 runs at that resolution,
    patch merging halves it and doubles the width for stage 2; the decoder
    bilinearly upsamples back to (H, W).
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        dim = base * 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=4, stride=4)
        self.stage1 = nn.ModuleList([
            _WindowBlock(dim, heads=4, window=7, shift=0, mlp_ratio=2),
            _WindowBlock(dim, heads=4, window=7, shift=7 // 2, mlp_ratio=2),
        ])
        self.merge = _PatchMerging(dim)
        self.stage2 = nn.ModuleList([
            _WindowBlock(dim * 2, heads=4, window=7, shift=0, mlp_ratio=2),
            _WindowBlock(dim * 2, heads=4, window=7, shift=7 // 2, mlp_ratio=2),
        ])
        self.up1 = _UpBlock(dim * 2, dim)
        self.up2 = _UpBlock(dim, dim // 2)
        self.outc = nn.Conv2d(dim // 2, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        f = self.proj(x)                              # stride 4
        for blk in self.stage1:
            f = blk(f)
        s1_size = (f.shape[-2], f.shape[-1])
        f = self.merge(f)                             # stride 8
        for blk in self.stage2:
            f = blk(f)
        f = self.up1(f, size=s1_size)
        f = self.up2(f, size=(H, W))
        return self.outc(f)


# ---------------------------------------------------------------------------
# 3. SegNeXtLite: convolutional attention (multi-scale strip convolutions).
# ---------------------------------------------------------------------------
class _MSCA(nn.Module):
    """Multi-Scale Convolutional Attention gate (no softmax attention).

    Depthwise 5x5, then three parallel depthwise strip branches (1x7/7x1,
    1x11/11x1, 1x21/21x1) which are summed, mixed by a 1x1 conv and used as a
    multiplicative gate on the input.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.b0 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim))
        self.b1 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim))
        self.b2 = nn.Sequential(
            nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim),
            nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim))
        self.mix = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        a = self.conv0(x)
        a = self.b0(a) + self.b1(a) + self.b2(a)
        return self.mix(a) * x


class _MSCABlock(nn.Module):
    """GroupNorm -> MSCA gate (+residual) -> GroupNorm -> MLP 1x1 (+residual)."""

    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        self.norm1 = _norm(dim)
        self.attn = _MSCA(dim)
        self.norm2 = _norm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SegNeXtLite(nn.Module):
    """3 stride-2 stages of MSCA blocks, fused at the stride-4 resolution."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        widths = [base, base * 2, base * 4]
        ins = [in_ch] + widths[:-1]
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ins[i], widths[i], 3, stride=2, padding=1,
                                    bias=False),
                          _norm(widths[i]),
                          nn.GELU())
            for i in range(3)])
        self.stages = nn.ModuleList([
            nn.Sequential(_MSCABlock(w), _MSCABlock(w)) for w in widths])
        self.fuse = nn.Conv2d(sum(widths), base * 2, kernel_size=1)
        self.fuse_norm = _norm(base * 2)
        self.fuse_act = nn.GELU()
        self.outc = nn.Conv2d(base * 2, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        feats = []
        f = x
        for down, stage in zip(self.downs, self.stages):
            f = stage(down(f))
            feats.append(f)
        size = (feats[1].shape[-2], feats[1].shape[-1])   # stride-4 resolution
        feats = [F.interpolate(t, size=size, mode="bilinear", align_corners=False)
                 for t in feats]
        f = self.fuse_act(self.fuse_norm(self.fuse(torch.cat(feats, dim=1))))
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.outc(f)


# ---------------------------------------------------------------------------
# 4. AxialUNet: U-Net with an axial-attention bottleneck.
# ---------------------------------------------------------------------------
class _AxialAttention(nn.Module):
    """MHSA along the height axis, then along the width axis (pre-LN, residual).

    Cost is O(H*W*(H+W)) rather than the O((H*W)^2) of full self-attention.
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm_h = nn.LayerNorm(dim)
        self.attn_h = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm_w = nn.LayerNorm(dim)
        self.attn_w = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)    # one sequence per column
        hn = self.norm_h(h)
        h = h + self.attn_h(hn, hn, hn, need_weights=False)[0]
        x = h.reshape(B, W, H, C).permute(0, 3, 2, 1)
        w = x.permute(0, 2, 3, 1).reshape(B * H, W, C)    # one sequence per row
        wn = self.norm_w(w)
        w = w + self.attn_w(wn, wn, wn, need_weights=False)[0]
        return w.reshape(B, H, W, C).permute(0, 3, 1, 2)


class AxialUNet(nn.Module):
    """2-down / 2-up U-Net (base, base*2, base*4) with axial attention at the
    bottleneck; skips are concatenated."""

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4]
        self.inc = _DoubleConv(in_ch, chs[0])
        self.pool = nn.MaxPool2d(2)
        self.down1 = _DoubleConv(chs[0], chs[1])
        self.down2 = _DoubleConv(chs[1], chs[2])
        self.axial = _AxialAttention(chs[2], heads=4)
        self.up1 = _DoubleConv(chs[2] + chs[1], chs[1])
        self.up2 = _DoubleConv(chs[1] + chs[0], chs[0])
        self.outc = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x):
        s0 = self.inc(x)
        s1 = self.down1(self.pool(s0))
        b = self.axial(self.down2(self.pool(s1)))
        y = F.interpolate(b, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up1(torch.cat([s1, y], dim=1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.up2(torch.cat([s0, y], dim=1))
        return self.outc(y)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
def transformer_models():
    """{name: constructor(in_ch) -> module}, all at base=32."""
    def vit_seg(in_ch):
        return ViTSeg(in_ch=in_ch, base=32)

    def swin_lite(in_ch):
        return SwinLiteSeg(in_ch=in_ch, base=32)

    def segnext_lite(in_ch):
        return SegNeXtLite(in_ch=in_ch, base=32)

    def axial_unet(in_ch):
        return AxialUNet(in_ch=in_ch, base=32)

    return {
        "vit_seg": vit_seg,
        "swin_lite": swin_lite,
        "segnext_lite": segnext_lite,
        "axial_unet": axial_unet,
    }


# ---------------------------------------------------------------------------
# Shape self-test only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    shapes = [(2, 1, 128, 128), (2, 1, 100, 140), (2, 1, 96, 224)]
    for name, ctor in transformer_models().items():
        net = ctor(1)
        net.eval()
        n_par = sum(p.numel() for p in net.parameters())
        for (B, C, H, W) in shapes:
            with torch.no_grad():
                y = net(torch.randn(B, C, H, W))
            assert tuple(y.shape) == (B, 1, H, W), (name, (H, W), tuple(y.shape))
        print(f"  OK  {name:14s} {n_par / 1e6:.3f}M params  "
              f"shapes {[s[2:] for s in shapes]}")
    print("transformer models self-test: OK")
