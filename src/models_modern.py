"""Modern / alternative segmentation ARCHITECTURES (tier C roster extension).

Six FROM-SCRATCH ``torch.nn.Module`` binary-segmentation nets: no pretrained
weights, no downloads, no dependency beyond torch. They plug straight into
``CNNRunner`` in ``src/models_cnn.py`` and obey the same contract:

  * ``__init__(self, in_ch=1, base=32)``
  * ``forward(x)`` with ``x`` of shape ``(B, in_ch, H, W)`` float
  * returns raw LOGITS of shape ``(B, 1, H, W)``, with no sigmoid inside forward.

H and W are arbitrary and are NOT assumed divisible by any stride: every net
that downsamples resizes intermediate features to the exact skip size and its
output to ``(H, W)``.

Normalization is GroupNorm everywhere (never BatchNorm) so no batch statistic
couples samples within a batch. Every net is kept well under ~5M parameters so
it trains on Apple-MPS in ~3000 steps at 128x128, batch 16.

Self-test:  python -m src.models_modern      (architecture shape check only)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------
def _norm(channels):
    """GroupNorm for `channels` feature maps (no running/batch statistics)."""
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _match_size(x, size):
    """Resize `x` to spatial `size` (bilinear) if it does not already match."""
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x


def _cat_skip(x, skip):
    """Concat decoder feature `x` with `skip` after resizing `x` to match."""
    return torch.cat([skip, _match_size(x, skip.shape[-2:])], dim=1)


# ---------------------------------------------------------------------------
# 1. Residual U-Net.
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """(Conv3x3-GN-GELU-Conv3x3-GN) + identity/1x1-projection shortcut, GELU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch),
        )
        self.short = (nn.Identity() if in_ch == out_ch
                      else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.short(x))


class ResUNet(nn.Module):
    """U-Net (3 down / 3 up) built entirely from residual blocks.

    Encoder widths base, base*2, base*4 with a base*8 bottleneck; skips are
    concatenated. Every block is a ``ResBlock``.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 8]
        self.pool = nn.MaxPool2d(2)
        self.inc = ResBlock(in_ch, f[0])
        self.down1 = ResBlock(f[0], f[1])
        self.down2 = ResBlock(f[1], f[2])
        self.down3 = ResBlock(f[2], f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec3 = ResBlock(f[2] * 2, f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = ResBlock(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = ResBlock(f[0] * 2, f[0])
        self.outc = nn.Conv2d(f[0], 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        s0 = self.inc(x)
        s1 = self.down1(self.pool(s0))
        s2 = self.down2(self.pool(s1))
        b = self.down3(self.pool(s2))
        d = self.dec3(_cat_skip(self.up3(b), s2))
        d = self.dec2(_cat_skip(self.up2(d), s1))
        d = self.dec1(_cat_skip(self.up1(d), s0))
        return _match_size(self.outc(d), size)


# ---------------------------------------------------------------------------
# 2. Tiramisu (FC-DenseNet).
# ---------------------------------------------------------------------------
class _DenseLayer(nn.Module):
    """One dense layer: GN -> GELU -> Conv3x3 producing `growth` maps."""

    def __init__(self, in_ch, growth):
        super().__init__()
        self.block = nn.Sequential(
            _norm(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class _DenseBlock(nn.Module):
    """`n_layers` dense layers, each fed the concatenation of all previous.

    ``keep_input=True`` returns ``cat(input, all layer outputs)`` (down path and
    the final block); ``False`` returns only the new layer outputs, which is the
    FC-DenseNet rule for anything that feeds a transition up.
    """

    def __init__(self, in_ch, growth, n_layers, keep_input=True):
        super().__init__()
        self.keep_input = keep_input
        self.layers = nn.ModuleList(
            [_DenseLayer(in_ch + i * growth, growth) for i in range(n_layers)])
        self.out_ch = (in_ch if keep_input else 0) + n_layers * growth

    def forward(self, x):
        feats = [x]
        new = []
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
            new.append(out)
        return torch.cat(feats, dim=1) if self.keep_input else torch.cat(new, dim=1)


class _TransitionDown(nn.Module):
    """1x1 conv + 2x2 average pool (channel count unchanged)."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class Tiramisu(nn.Module):
    """FC-DenseNet ("Tiramisu"), growth rate 12, 3 down / 3 up transitions.

    Dense-block layer counts are (3, 4, 5) down, 6 at the bottleneck and
    (5, 4, 3) up; transitions down are 1x1 conv + 2x2 avgpool, transitions up
    are transpose convs whose output is concatenated with the matching down-path
    dense-block output (the standard FC-DenseNet skip).
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        growth = 12
        self.first = nn.Conv2d(in_ch, base, kernel_size=3, padding=1, bias=False)

        ch = base
        self.db_down = nn.ModuleList()
        self.td = nn.ModuleList()
        skip_chs = []
        for n in (3, 4, 5):
            db = _DenseBlock(ch, growth, n, keep_input=True)
            self.db_down.append(db)
            ch = db.out_ch
            skip_chs.append(ch)
            self.td.append(_TransitionDown(ch))

        self.bottleneck = _DenseBlock(ch, growth, 6, keep_input=False)
        ch = self.bottleneck.out_ch

        self.tu = nn.ModuleList()
        self.db_up = nn.ModuleList()
        for i, n in enumerate((5, 4, 3)):
            self.tu.append(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))
            ch = ch + skip_chs[2 - i]
            last = (i == 2)
            db = _DenseBlock(ch, growth, n, keep_input=last)
            self.db_up.append(db)
            ch = db.out_ch
        self.outc = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        h = self.first(x)
        skips = []
        for db, td in zip(self.db_down, self.td):
            h = db(h)
            skips.append(h)
            h = td(h)
        h = self.bottleneck(h)
        for i, (tu, db) in enumerate(zip(self.tu, self.db_up)):
            h = db(_cat_skip(tu(h), skips[2 - i]))
        return _match_size(self.outc(h), size)


# ---------------------------------------------------------------------------
# 3. ConvNeXt U-Net.
# ---------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    """Depthwise 7x7 -> LN -> 1x1 expand x4 -> GELU -> 1x1 project -> residual.

    The LayerNorm is a channels-last LN implemented as ``GroupNorm(1, C)``; the
    residual branch is scaled by a learnable per-channel layer-scale gamma
    initialized to 1e-6.
    """

    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=7, padding=3, groups=ch)
        self.ln = nn.GroupNorm(1, ch)
        self.pw1 = nn.Conv2d(ch, ch * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(ch * 4, ch, kernel_size=1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(ch))

    def forward(self, x):
        h = self.pw2(self.act(self.pw1(self.ln(self.dw(x)))))
        return x + self.gamma.view(1, -1, 1, 1) * h


class ConvNeXtUNet(nn.Module):
    """U-Net (3 down / 3 up) whose blocks are ConvNeXt blocks.

    Level widths base, base*2, base*4 (the base*4 bottleneck sits one stride
    below the deepest listed level). Downsampling is LN + 2x2 stride-2 conv,
    upsampling a 2x2 transpose conv; each decoder stage concatenates its skip
    and fuses with a 1x1 conv before the ConvNeXt block (which preserves
    channels).
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        f = [base, base * 2, base * 4, base * 4]
        self.stem = nn.Conv2d(in_ch, f[0], kernel_size=3, padding=1)
        self.enc0 = ConvNeXtBlock(f[0])
        self.down = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(1, f[i]),
                          nn.Conv2d(f[i], f[i + 1], kernel_size=2, stride=2))
            for i in range(3)])
        self.enc = nn.ModuleList([ConvNeXtBlock(f[i + 1]) for i in range(3)])
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(f[i + 1], f[i], kernel_size=2, stride=2)
            for i in range(2, -1, -1)])
        self.fuse = nn.ModuleList([
            nn.Conv2d(f[i] * 2, f[i], kernel_size=1)
            for i in range(2, -1, -1)])
        self.dec = nn.ModuleList([ConvNeXtBlock(f[i]) for i in range(2, -1, -1)])
        self.outc = nn.Conv2d(f[0], 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        h = self.enc0(self.stem(x))
        skips = [h]
        for down, enc in zip(self.down, self.enc):
            h = enc(down(h))
            skips.append(h)
        for j in range(3):
            h = self.dec[j](self.fuse[j](_cat_skip(self.up[j](h), skips[2 - j])))
        return _match_size(self.outc(h), size)


# ---------------------------------------------------------------------------
# 4. HRNet-lite (high resolution maintained throughout).
# ---------------------------------------------------------------------------
class _HRFusion(nn.Module):
    """Multi-resolution fusion: branch i receives the sum of all branches.

    Branch j is brought to branch i's resolution by repeated stride-2 3x3 convs
    when j is shallower, and by bilinear upsampling followed by a 1x1 conv when
    j is deeper.
    """

    def __init__(self, chs):
        super().__init__()
        self.n = len(chs)
        self.paths = nn.ModuleList()
        for i in range(self.n):
            row = nn.ModuleList()
            for j in range(self.n):
                if i == j:
                    row.append(nn.Identity())
                elif i > j:  # downsample j -> i with (i - j) stride-2 convs
                    steps, prev = [], chs[j]
                    for s in range(i - j):
                        last = (s == i - j - 1)
                        out = chs[i] if last else prev
                        steps.append(nn.Conv2d(prev, out, kernel_size=3, stride=2,
                                               padding=1, bias=False))
                        steps.append(_norm(out))
                        if not last:
                            steps.append(nn.GELU())
                        prev = out
                    row.append(nn.Sequential(*steps))
                else:  # upsample j -> i: bilinear (in forward) then 1x1 conv
                    row.append(nn.Sequential(
                        nn.Conv2d(chs[j], chs[i], kernel_size=1, bias=False),
                        _norm(chs[i])))
            self.paths.append(row)
        self.act = nn.GELU()

    def forward(self, xs):
        out = []
        for i in range(self.n):
            size = xs[i].shape[-2:]
            total = None
            for j in range(self.n):
                if i < j:
                    y = self.paths[i][j](_match_size(xs[j], size))
                else:
                    y = _match_size(self.paths[i][j](xs[j]), size)
                total = y if total is None else total + y
            out.append(self.act(total))
        return out


class HRNetLite(nn.Module):
    """HRNet-style net that keeps a high-resolution branch throughout.

    A stride-2 stem feeds 3 parallel branches at strides 2, 4 and 8 (widths
    base, base*2, base*4). Each of 2 stages runs 2 residual blocks per branch
    and then a full multi-resolution fusion. The head upsamples every branch to
    the stride-2 resolution, concatenates, fuses with a 1x1 conv, upsamples to
    (H, W) and maps to a single logit.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=3, stride=2, padding=1,
                      bias=False),
            _norm(chs[0]),
            nn.GELU(),
        )
        self.trans1 = nn.Sequential(
            nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=2, padding=1,
                      bias=False),
            _norm(chs[1]),
            nn.GELU(),
        )
        self.trans2 = nn.Sequential(
            nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=2, padding=1,
                      bias=False),
            _norm(chs[2]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList([
            nn.ModuleList([nn.Sequential(ResBlock(c, c), ResBlock(c, c))
                           for c in chs])
            for _ in range(2)])
        self.fusions = nn.ModuleList([_HRFusion(chs) for _ in range(2)])
        self.head = nn.Sequential(
            nn.Conv2d(sum(chs), base * 2, kernel_size=1, bias=False),
            _norm(base * 2),
            nn.GELU(),
        )
        self.outc = nn.Conv2d(base * 2, 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        b0 = self.stem(x)
        b1 = self.trans1(b0)
        b2 = self.trans2(b1)
        xs = [b0, b1, b2]
        for blocks, fuse in zip(self.stages, self.fusions):
            xs = [blocks[i](xs[i]) for i in range(3)]
            xs = fuse(xs)
        top = xs[0].shape[-2:]
        h = torch.cat([xs[0]] + [_match_size(t, top) for t in xs[1:]], dim=1)
        h = _match_size(self.head(h), size)
        return self.outc(h)


# ---------------------------------------------------------------------------
# 5. SIREN segmenter (coordinate-based / implicit representation).
# ---------------------------------------------------------------------------
class _SineLayer(nn.Module):
    """1x1 conv followed by ``sin(omega * .)`` with the standard SIREN init."""

    def __init__(self, in_ch, out_ch, omega, is_first=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.omega = omega
        with torch.no_grad():
            bound = (1.0 / in_ch if is_first
                     else math.sqrt(6.0 / in_ch) / omega)
            self.conv.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.conv(x))


class SirenSeg(nn.Module):
    """Coordinate-based (implicit) segmenter: SIREN MLP over (y, x) + intensity.

    Per pixel the input is the intensity, the normalized (y, x) coordinate pair
    for the ACTUAL H, W at runtime, and 16-frequency random-Fourier features of
    those coordinates (fixed random projection, a non-trainable buffer sized at
    construction). A 5-layer 1x1-conv MLP of width base*4 with sine activations
    (sin(30x) first, sin(x) after) then a 1x1 conv gives the logit.

    NOTE: this net has NO spatial aggregation beyond the receptive field of a
    1x1 conv: one pixel plus its coordinate. It is a deliberate contrast to the
    convolutional models in this module.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.n_freq = 16
        self.register_buffer("freqs", torch.randn(self.n_freq, 2, 1, 1))
        width = base * 4
        n_in = in_ch + 2 + 2 * self.n_freq
        layers = [_SineLayer(n_in, width, omega=30.0, is_first=True)]
        for _ in range(4):
            layers.append(_SineLayer(width, width, omega=1.0))
        self.body = nn.Sequential(*layers)
        self.outc = nn.Conv2d(width, 1, kernel_size=1)

    def _coords(self, x):
        """(B, 2, H, W) normalized (y, x) grid for the runtime H, W."""
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        gy = ys.view(1, 1, h, 1).expand(b, 1, h, w)
        gx = xs.view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def forward(self, x):
        coords = self._coords(x)
        proj = 2.0 * math.pi * F.conv2d(coords, self.freqs)
        feats = torch.cat([x, coords, torch.sin(proj), torch.cos(proj)], dim=1)
        return self.outc(self.body(feats))


# ---------------------------------------------------------------------------
# 6. U-Net + differentiable mean-field CRF head.
# ---------------------------------------------------------------------------
class DoubleConvG(nn.Module):
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


class UNetCRF(nn.Module):
    """Small U-Net (2 down / 2 up, widths base, base*2) + mean-field CRF head.

    The U-Net emits a 2-channel unary logit map (channel 0 foreground, channel 1
    background). The head then runs 5 mean-field iterations: softmax over the
    two channels, message passing as a fixed Gaussian blur (depthwise 5x5,
    sigma=1.5, non-trainable buffer), a learnable 2x2 compatibility transform
    (1x1 conv), and adding the unary back. The returned single channel is the
    foreground-minus-background logit difference.
    """

    def __init__(self, in_ch=1, base=32, iters=5, sigma=1.5, ksize=5):
        super().__init__()
        f = [base, base * 2, base * 2]
        self.pool = nn.MaxPool2d(2)
        self.inc = DoubleConvG(in_ch, f[0])
        self.down1 = DoubleConvG(f[0], f[1])
        self.down2 = DoubleConvG(f[1], f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConvG(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConvG(f[0] * 2, f[0])
        self.unary = nn.Conv2d(f[0], 2, kernel_size=1)

        self.iters = iters
        self.ksize = ksize
        coord = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
        g = torch.exp(-(coord ** 2) / (2.0 * sigma ** 2))
        kernel2d = torch.outer(g, g)
        kernel2d = kernel2d / kernel2d.sum()
        self.register_buffer("gauss", kernel2d.view(1, 1, ksize, ksize)
                             .repeat(2, 1, 1, 1))
        self.compat = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        s0 = self.inc(x)
        s1 = self.down1(self.pool(s0))
        b = self.down2(self.pool(s1))
        d = self.dec2(_cat_skip(self.up2(b), s1))
        d = self.dec1(_cat_skip(self.up1(d), s0))
        unary = _match_size(self.unary(d), size)

        q = unary
        for _ in range(self.iters):
            p = F.softmax(q, dim=1)
            msg = F.conv2d(p, self.gauss, padding=self.ksize // 2, groups=2)
            q = unary + self.compat(msg)
        return q[:, 0:1] - q[:, 1:2]


# ---------------------------------------------------------------------------
# Registry.  Each constructor takes in_ch and returns the module with base=32.
# ---------------------------------------------------------------------------
def modern_models():
    """Name -> constructor(in_ch) for every architecture in this module."""
    return {
        "resunet": lambda in_ch: ResUNet(in_ch=in_ch, base=32),
        "tiramisu": lambda in_ch: Tiramisu(in_ch=in_ch, base=32),
        "convnext_unet": lambda in_ch: ConvNeXtUNet(in_ch=in_ch, base=32),
        "hrnet_lite": lambda in_ch: HRNetLite(in_ch=in_ch, base=32),
        "siren_seg": lambda in_ch: SirenSeg(in_ch=in_ch, base=32),
        "unet_crf": lambda in_ch: UNetCRF(in_ch=in_ch, base=32),
    }


# ---------------------------------------------------------------------------
# Shape self-test only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    SIZES = [(128, 128), (100, 140), (96, 224)]
    for name, make in modern_models().items():
        net = make(1)
        net.eval()
        n_par = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            for h, w in SIZES:
                y = net(torch.randn(2, 1, h, w))
                assert tuple(y.shape) == (2, 1, h, w), (name, (h, w),
                                                        tuple(y.shape))
        print(f"  OK  {name:14s} {n_par / 1e6:7.3f} M params  -> "
              f"(2,1,128,128), (2,1,100,140), (2,1,96,224)")
    print("modern models self-test: OK")
