"""Pretraining as an INSTRUMENT for measuring data starvation, not as a model.

Sky, 2026-08-01:

    "The end goal is not pretrained. The pretrained model having better success
     will just show us we need more data."

That reframing is what this module implements. Pretraining substitutes for data:
an ImageNet encoder arrives already knowing edges, texture and shape statistics
that our 213 training slices would otherwise have to teach from scratch. So the
GAP between a pretrained encoder and the identical architecture randomly
initialised is an estimate of how much data-driven headroom exists here. It
measures the constraint; it is not a candidate for shipping.

[MUST] BOTH ARMS ARE THE SAME ARCHITECTURE
------------------------------------------
The only difference between the arms is the encoder's initial weights:

    pretrained  ResNet-18 encoder, ImageNet weights, + U-Net decoder
    scratch     ResNet-18 encoder, RANDOM init,      + U-Net decoder

Comparing a pretrained ResNet-UNet against the existing plain `UNet2D` would
confound ARCHITECTURE with PRETRAINING and make the gap uninterpretable, which
would defeat the entire purpose. The scratch arm also gives a second, separate
reading: scratch-ResNet18 vs `unet2d_s30000` (0.29409) isolates the architecture
change on its own.

[CORRECTED 2026-08-04] That second reading does NOT isolate architecture on its
own. `ResNetUNet` is BatchNorm throughout (the torchvision ResNet-18 encoder,
plus `_up`'s `nn.BatchNorm2d`) while `UNet2D` uses GroupNorm
(`models_cnn._norm`), so scratch-ResNet18 vs `unet2d_s30000` crosses a
normalisation change as well as an architecture change. The statement above it
is unaffected: the two ARMS of this probe still differ only in the encoder's
initial weights.

Preprocessing is identical in both arms too, including the 1-channel to 3-channel
expansion and ImageNet normalisation. Only `weights=` differs.

[LEAKAGE][MUST] Same frozen split, same 30,000-step budget, same recipe, same
threshold-on-val discipline, test scored once per arm, via the shared
`evaluate.run_and_score` harness.

[HONESTY] What a gap here does and does not prove
-------------------------------------------------
A large pretrained advantage is evidence that the from-scratch model is
data-starved, because pretraining's contribution is precisely the statistics more
data would have supplied. It is NOT proof that more meteorite data specifically
would close it: ImageNet also supplies scale and diversity our domain cannot, and
a domain shift remains. Read the gap as an upper-ish bound on data-driven
headroom, not as a promise.

Also note both arms are n=1 seed, so a small gap here is not interpretable and a
LARGE one is. The pre-registered band is 0.05.

[CORRECTED 2026-08-04] This docstring used to say `data_scaling.py` "warns that
single-seed differences of 0.03-0.06 cannot be separated from run-to-run
scatter". That was a misreading of it, and the paraphrase is withdrawn. What it
actually says is that such differences sit only 4 to 8 SE apart before counting
training-seed variance. That variance has since been MEASURED for the first time:
SD 0.0084 across three seeds at fraction 0.25 (7,200 steps), which is smaller
than the 0.0142 val noise floor, not larger. The 0.05 band was NOT revised on the
strength of it, because it was pre-registered and because an SD from n=3 in a
different budget regime does not transfer.

Run:  python -m src.pretrained_probe --arm=pretrained     (then --arm=scratch)
      python -m src.pretrained_probe --arm=both           (sequential, one at a time)
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import evaluate
from .models_cnn import CNNRunner, dice_loss
from .run_families import best_recipe

OUT = "results/pretrained_probe.json"
# ImageNet statistics; applied in BOTH arms so only the weights differ.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetUNet(nn.Module):
    """ResNet-18 encoder + U-Net decoder, optionally ImageNet-pretrained.

    Accepts `in_ch` channels (1 here) and expands to the 3 the ImageNet stem
    expects by repeating, which keeps conv1's pretrained filters usable. A
    1-channel stem would have to be re-initialised or averaged, discarding the
    very weights the experiment is trying to measure.
    """

    def __init__(self, in_ch=1, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        enc = resnet18(weights=w)
        self.in_ch = in_ch
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # /2, 64
        self.pool = enc.maxpool                                    # /4
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        # channels: l1 64 (/4), l2 128 (/8), l3 256 (/16), l4 512 (/32)
        self.d4 = self._up(512, 256)
        self.d3 = self._up(256 + 256, 128)
        self.d2 = self._up(128 + 128, 64)
        self.d1 = self._up(64 + 64, 64)
        self.d0 = self._up(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        if x.shape[1] != 3:                       # 1 -> 3, keeps the ImageNet stem
            x = x.repeat(1, 3 // x.shape[1] + (3 % x.shape[1] > 0), 1, 1)[:, :3]
        x = (x - self.mean) / self.std
        s0 = self.stem(x)                         # /2
        s1 = self.l1(self.pool(s0))               # /4
        s2 = self.l2(s1)                          # /8
        s3 = self.l3(s2)                          # /16
        s4 = self.l4(s3)                          # /32

        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="bilinear",
                                 align_corners=False)
        y = self.d4(s4)
        y = self.d3(torch.cat([up(y, s3), s3], 1))
        y = self.d2(torch.cat([up(y, s2), s2], 1))
        y = self.d1(torch.cat([up(y, s1), s1], 1))
        y = self.d0(torch.cat([up(y, s0), s0], 1))
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(y)


def build(arm, in_ch=1):
    return ResNetUNet(in_ch=in_ch, pretrained=(arm == "pretrained"))


def run_arm(arm, epochs=60, steps=500, seed=0, root="."):
    rec = best_recipe()
    name = f"resunet18_{arm}_s{epochs * steps}"
    print(f"\n=== {name} | recipe {rec.get('_source')} | "
          f"{epochs * steps} steps ===", flush=True)
    runner = CNNRunner(
        name, lambda ic: build(arm, ic), dice_loss,
        epochs=epochs, steps_per_epoch=steps,
        lr=float(rec.get("lr", 3e-4)), weight_decay=float(rec.get("wd", 0.0)),
        pos_frac=float(rec.get("pos_frac", 0.5)), augment=False,
        val_stride=6, seed=seed, ckpt_dir="results/checkpoints")
    t0 = time.time()
    row = evaluate.run_and_score(
        runner, root=root,
        extra={"recipe": rec.get("_source"), "arm": arm,
               "encoder": "resnet18", "pretrained": arm == "pretrained"},
        save_prob=f"results/probs/{name}.npy")
    row["sec"] = round(time.time() - t0, 1)
    row["val_iou"] = float(getattr(runner, "best_full_val_iou", float("nan")))
    return row


def _prior_rows(arms):
    """Rows already in OUT for arms this invocation is NOT running.

    `main` used to build `rows` from empty and rewrite OUT after every arm, so
    running the arms as two separate processes (the documented workflow) had the
    second one silently delete the first one's result. Keep the other arm's row.
    """
    try:
        with open(OUT) as fh:
            prior = json.load(fh).get("rows", [])
    except (OSError, ValueError):
        return []
    return [r for r in prior
            if r.get("arm") in ("pretrained", "scratch") and r.get("arm") not in arms]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    arm, epochs, steps = "both", 60, 500
    for a in [x for x in argv if x.startswith("--")]:
        k, _, v = a[2:].partition("=")
        if k == "arm":
            arm = v
        elif k == "epochs":
            epochs = int(v)
        elif k == "steps":
            steps = int(v)
        else:
            raise SystemExit(f"unknown option {a}")
    if arm not in ("pretrained", "scratch", "both"):
        raise SystemExit("--arm must be pretrained, scratch or both")

    arms = ["pretrained", "scratch"] if arm == "both" else [arm]
    rows = _prior_rows(arms)
    for a in arms:                       # SEQUENTIAL: one training job at a time
        rows.append(run_arm(a, epochs, steps))
        json.dump({"regime": ("ResNet-18 encoder + U-Net decoder; the arms differ "
                              "ONLY in encoder initialisation (ImageNet vs random). "
                              "Frozen split, threshold on VAL, TEST scored once per "
                              "arm. N=1 meteorite, n=1 seed per arm."),
                   "rows": rows}, open(OUT, "w"), indent=2)
        print(f"  {a}: test IoU={rows[-1].get('iou')}", flush=True)

    if len(rows) == 2:
        p = next(r for r in rows if r.get("arm") == "pretrained")
        s = next(r for r in rows if r.get("arm") == "scratch")
        gap = float(p.get("iou", float("nan"))) - float(s.get("iou", float("nan")))
        print(f"\n=== PRETRAINING GAP (the data-starvation estimate) ===")
        print(f"  pretrained {p.get('iou'):.5f}   scratch {s.get('iou'):.5f}   "
              f"gap {gap:+.5f}")
        print(f"  reference: plain unet2d_s30000 = 0.29409 (different architecture,")
        print(f"             so compare it against the SCRATCH arm, not the pretrained one)")
        print(f"  [HONESTY] n=1 seed per arm. Pre-registered band: a gap under "
              f"0.05 is ambiguous.")
        print(f"            Measured seed SD is 0.0084 (n=3, frac 0.25, 7.2k "
              f"steps), but that is a")
        print(f"            different regime and does not license shrinking "
              f"the band after the fact.")
    print(f"\nwrote {OUT}")
    return rows


if __name__ == "__main__":
    main()
