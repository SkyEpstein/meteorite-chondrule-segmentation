"""Extra segmentation ARCHITECTURES (tier C roster extension).

Nine additional ``torch.nn.Module`` binary-segmentation nets that plug straight
into ``CNNRunner`` in ``src/models_cnn.py``. Every net obeys the same contract:

  * ``__init__(self, in_ch=1, base=32)``
  * ``forward(x)`` with ``x`` of shape ``(B, in_ch, H, W)`` float
  * returns raw LOGITS of shape ``(B, 1, H, W)``, with no sigmoid inside forward.

H and W are arbitrary (>= 64) and are NOT assumed divisible by any stride: every
net that downsamples either reflect-pads to a multiple of its total stride and
crops back, or resizes its output to ``(H, W)`` explicitly.

Normalization is GroupNorm everywhere (never BatchNorm) so no batch statistic
couples samples within a batch. No dependency beyond torch.

Self-test:  python -m src.models_extra      (architecture shape check only)
"""
from __future__ import annotations
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


def _cbr(in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1):
    """Conv -> GroupNorm -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        _norm(out_ch),
        nn.ReLU(inplace=True),
    )


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


def _pad_to_multiple(x, m):
    """Reflect-pad H,W up to a multiple of `m`; returns (padded, (H, W))."""
    h, w = x.shape[-2], x.shape[-1]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        x = F.pad(x, [0, pw, 0, ph], mode="reflect")
    return x, (h, w)


def _match_size(x, size):
    """Resize `x` to spatial `size` (bilinear) if it does not already match."""
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x


# ---------------------------------------------------------------------------
# 1. Per-pixel MLP (no spatial context at all).
# ---------------------------------------------------------------------------
class PixelMLP(nn.Module):
    """1x1-conv-only MLP applied independently to every pixel.

    Three hidden layers of width ``base*2`` (GroupNorm + ReLU) then a 1x1 conv
    to a single logit. Pure per-pixel baseline: receptive field is one pixel.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        h = base * 2
        self.body = nn.Sequential(
            _cbr(in_ch, h, kernel=1, padding=0),
            _cbr(h, h, kernel=1, padding=0),
            _cbr(h, h, kernel=1, padding=0),
        )
        self.outc = nn.Conv2d(h, 1, kernel_size=1)

    def forward(self, x):
        return self.outc(self.body(x))


# ---------------------------------------------------------------------------
# 2. Plain conv stack (no resolution change).
# ---------------------------------------------------------------------------
class PatchCNN(nn.Module):
    """Plain 5-layer conv stack at full resolution (no down/upsampling).

    Four 3x3 stride-1 Conv-GroupNorm-ReLU layers of widths
    ``base, base, base*2, base*2`` followed by a 1x1 conv to one logit.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        widths = [base, base, base * 2, base * 2]
        layers, prev = [], in_ch
        for w in widths:
            layers.append(_cbr(prev, w))
            prev = w
        self.body = nn.Sequential(*layers)
        self.outc = nn.Conv2d(prev, 1, kernel_size=1)

    def forward(self, x):
        return self.outc(self.body(x))


# ---------------------------------------------------------------------------
# 3. Dilated stack (large receptive field, no downsampling).
# ---------------------------------------------------------------------------
class DilatedCNN(nn.Module):
    """Six 3x3 conv layers with dilations 1, 2, 4, 8, 16, 1 at width ``base*2``.

    ``padding=dilation`` keeps the resolution fixed throughout, so the net sees
    a wide context without ever downsampling.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        w = base * 2
        layers, prev = [], in_ch
        for d in (1, 2, 4, 8, 16, 1):
            layers.append(_cbr(prev, w, kernel=3, padding=d, dilation=d))
            prev = w
        self.body = nn.Sequential(*layers)
        self.outc = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x):
        return self.outc(self.body(x))


# ---------------------------------------------------------------------------
# 4. UNet++ (nested dense skips), 3 resolution levels.
# ---------------------------------------------------------------------------
class UNetPP(nn.Module):
    """UNet++ with dense nested skips over levels 0..2 (widths b, 2b, 4b).

    Nodes X[i][j]: X[i][0] is the encoder, X[i][j>0] fuses every shallower node
    of the same level with the bilinearly upsampled X[i+1][j-1]. The head is a
    1x1 conv on X[0][2]. All upsampling targets the exact skip size, so odd H,W
    round-trip unchanged.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        f = [base, base * 2, base * 4]
        self.pool = nn.MaxPool2d(2)
        self.conv0_0 = DoubleConv(in_ch, f[0])
        self.conv1_0 = DoubleConv(f[0], f[1])
        self.conv2_0 = DoubleConv(f[1], f[2])
        self.conv0_1 = DoubleConv(f[0] + f[1], f[0])
        self.conv1_1 = DoubleConv(f[1] + f[2], f[1])
        self.conv0_2 = DoubleConv(f[0] * 2 + f[1], f[0])
        self.outc = nn.Conv2d(f[0], 1, kernel_size=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                             align_corners=False)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        return _match_size(self.outc(x0_2), size)


# ---------------------------------------------------------------------------
# 5. FCN-8s style skip fusion.
# ---------------------------------------------------------------------------
class FCN8(nn.Module):
    """FCN-8s: score maps at strides 8, 4 and 2 summed after bilinear upsample.

    Encoder is three (DoubleConv -> maxpool) stages of widths base, base*2,
    base*4. Each pooled feature gets a 1x1 score layer; the deepest score is
    upsampled and added to the next, twice, then upsampled to the input size.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.stage1 = DoubleConv(in_ch, base)          # -> stride 2 after pool
        self.stage2 = DoubleConv(base, base * 2)       # -> stride 4 after pool
        self.stage3 = DoubleConv(base * 2, base * 4)   # -> stride 8 after pool
        self.score2 = nn.Conv2d(base, 1, kernel_size=1)
        self.score4 = nn.Conv2d(base * 2, 1, kernel_size=1)
        self.score8 = nn.Conv2d(base * 4, 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        p2 = self.pool(self.stage1(x))        # stride 2,  base
        p4 = self.pool(self.stage2(p2))       # stride 4,  base*2
        p8 = self.pool(self.stage3(p4))       # stride 8,  base*4
        s = self.score8(p8)
        s = _match_size(s, p4.shape[-2:]) + self.score4(p4)
        s = _match_size(s, p2.shape[-2:]) + self.score2(p2)
        return _match_size(s, size)


# ---------------------------------------------------------------------------
# 6. SegNet (max-pool indices reused by max-unpool).
# ---------------------------------------------------------------------------
class SegNet(nn.Module):
    """Encoder-decoder that upsamples with the stored max-pool ARGMAX indices.

    Three down and three up stages (widths base, base*2, base*4). Each
    ``F.max_pool2d(..., return_indices=True)`` size is remembered and handed to
    ``F.max_unpool2d`` as ``output_size``, so odd spatial sizes are restored
    exactly.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.dec3 = DoubleConv(base * 4, base * 2)
        self.dec2 = DoubleConv(base * 2, base)
        self.dec1 = DoubleConv(base, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        e1 = self.enc1(x)
        p1, i1 = F.max_pool2d(e1, 2, return_indices=True)
        e2 = self.enc2(p1)
        p2, i2 = F.max_pool2d(e2, 2, return_indices=True)
        e3 = self.enc3(p2)
        p3, i3 = F.max_pool2d(e3, 2, return_indices=True)
        d3 = self.dec3(F.max_unpool2d(p3, i3, 2, output_size=e3.shape[-2:]))
        d2 = self.dec2(F.max_unpool2d(d3, i2, 2, output_size=e2.shape[-2:]))
        d1 = self.dec1(F.max_unpool2d(d2, i1, 2, output_size=e1.shape[-2:]))
        return _match_size(self.outc(d1), size)


# ---------------------------------------------------------------------------
# 7. DeepLabV3+ (ASPP head + low-level decoder).
# ---------------------------------------------------------------------------
class DeepLabV3P(nn.Module):
    """ASPP (dilations 1, 6, 12, 18 + image pooling) with a low-level decoder.

    Encoder: three stride-2 convs to width base*4 (output stride 8). ASPP
    branches are concatenated and projected back to base*4 by a 1x1 conv, then
    concatenated with the 1x1-reduced first-stage (stride 2) feature, upsampled
    to the input size and scored by a final 1x1 conv.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        c = base * 4
        self.enc1 = _cbr(in_ch, base, stride=2)          # stride 2, low level
        self.enc2 = _cbr(base, base * 2, stride=2)       # stride 4
        self.enc3 = _cbr(base * 2, c, stride=2)          # stride 8
        self.aspp1 = _cbr(c, c, kernel=1, padding=0)
        self.aspp6 = _cbr(c, c, kernel=3, padding=6, dilation=6)
        self.aspp12 = _cbr(c, c, kernel=3, padding=12, dilation=12)
        self.aspp18 = _cbr(c, c, kernel=3, padding=18, dilation=18)
        self.pool_branch = _cbr(c, c, kernel=1, padding=0)
        self.project = _cbr(c * 5, c, kernel=1, padding=0)
        self.low = _cbr(base, base, kernel=1, padding=0)
        self.outc = nn.Conv2d(c + base, 1, kernel_size=1)

    def forward(self, x):
        size = (x.shape[-2], x.shape[-1])
        e1 = self.enc1(x)
        e3 = self.enc3(self.enc2(e1))
        gp = self.pool_branch(F.adaptive_avg_pool2d(e3, 1))
        gp = _match_size(gp, e3.shape[-2:])
        y = self.project(torch.cat(
            [self.aspp1(e3), self.aspp6(e3), self.aspp12(e3),
             self.aspp18(e3), gp], dim=1))
        y = _match_size(y, e1.shape[-2:])
        y = torch.cat([y, self.low(e1)], dim=1)
        return self.outc(_match_size(y, size))


# ---------------------------------------------------------------------------
# 8. Plain conv autoencoder (deliberately NO skip connections).
# ---------------------------------------------------------------------------
class SegAutoencoder(nn.Module):
    """Encoder-decoder with no skips at all: the U-Net contrast case.

    Three stride-2 conv downs (widths base, base*2, base*4) and three
    transpose-conv ups, then a 1x1 conv to one logit. The input is reflect
    padded to a multiple of 8 and the output cropped back to (H, W).
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.down1 = _cbr(in_ch, base, stride=2)
        self.down2 = _cbr(base, base * 2, stride=2)
        self.down3 = _cbr(base * 2, base * 4, stride=2)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.conv3 = _cbr(base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.conv2 = _cbr(base, base)
        self.up1 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.conv1 = _cbr(base, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        x, size = _pad_to_multiple(x, 8)
        y = self.down3(self.down2(self.down1(x)))
        y = self.conv3(self.up3(y))
        y = self.conv2(self.up2(y))
        y = self.conv1(self.up1(y))
        y = self.outc(y)
        return y[..., :size[0], :size[1]]


# ---------------------------------------------------------------------------
# 9. ConvLSTM over the channel (z) axis.
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """Minimal ConvLSTM cell: one 3x3 conv producing the i, f, o, g gates."""

    def __init__(self, in_ch, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(in_ch + hidden, 4 * hidden, kernel_size=3,
                               padding=1, bias=True)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], dim=1)), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMSeg(nn.Module):
    """Treats the ``in_ch`` axis as a Z-SEQUENCE of K single-channel slices.

    The input (B, K, H, W) is consumed as K timesteps of 1-channel images by a
    ConvLSTM cell of hidden width ``base``; the FINAL hidden state feeds a small
    2-conv head and a 1x1 conv to one logit. Intended for k>1 (2.5D) use. With
    ``in_ch == 1`` it degenerates to a sequence of length 1. No downsampling, so
    any H, W passes through unchanged.
    """

    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.steps = in_ch
        self.hidden = base
        self.cell = ConvLSTMCell(1, base)
        self.head = nn.Sequential(_cbr(base, base), _cbr(base, base))
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        b, _, h, w = x.shape
        state = (x.new_zeros(b, self.hidden, h, w),
                 x.new_zeros(b, self.hidden, h, w))
        for t in range(self.steps):
            state = self.cell(x[:, t:t + 1], state)
        return self.outc(self.head(state[0]))


# ---------------------------------------------------------------------------
# Registry.  Each constructor takes in_ch and returns the module with base=32.
# ---------------------------------------------------------------------------
def extra_models():
    """Name -> constructor(in_ch) for every architecture in this module."""
    return {
        "pixel_mlp": lambda in_ch: PixelMLP(in_ch=in_ch, base=32),
        "patch_cnn": lambda in_ch: PatchCNN(in_ch=in_ch, base=32),
        "dilated_cnn": lambda in_ch: DilatedCNN(in_ch=in_ch, base=32),
        "unetpp": lambda in_ch: UNetPP(in_ch=in_ch, base=32),
        "fcn8": lambda in_ch: FCN8(in_ch=in_ch, base=32),
        "segnet": lambda in_ch: SegNet(in_ch=in_ch, base=32),
        "deeplabv3p": lambda in_ch: DeepLabV3P(in_ch=in_ch, base=32),
        "autoencoder": lambda in_ch: SegAutoencoder(in_ch=in_ch, base=32),
        "convlstm": lambda in_ch: ConvLSTMSeg(in_ch=in_ch, base=32),
    }


# ---------------------------------------------------------------------------
# Shape self-test only (no training, no real data).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    for name, make in extra_models().items():
        net = make(1)
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 1, 128, 128))
            assert tuple(y.shape) == (2, 1, 128, 128), (name, tuple(y.shape))
            y_odd = net(torch.randn(2, 1, 100, 140))
            assert tuple(y_odd.shape) == (2, 1, 100, 140), (name, tuple(y_odd.shape))
        if name == "convlstm":
            net5 = make(5)
            net5.eval()
            with torch.no_grad():
                y5 = net5(torch.randn(2, 5, 128, 128))
            assert tuple(y5.shape) == (2, 1, 128, 128), (name, tuple(y5.shape))
        print(f"  OK  {name:12s} -> (2,1,128,128) and (2,1,100,140)")
    print("extra models self-test: OK")
