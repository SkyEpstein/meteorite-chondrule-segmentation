"""Regression tests for the VQ-VAE defects found by review on 2026-08-08.

Each check here pins a bug that was hand-fixed and that nothing otherwise
prevents from returning. Two of the three destroyed real work when they fired:
the shared artifact name overwrote the MSE-only checkpoint, which could not be
recovered, and the factor-blind baseline silently scored the f=4 gate against
the f=8 control and overstated its margin by 2.89 dB.

None of these needs a trained model or the 94 MB patch pool, so they run in
about a second and can gate every commit.

Run:  python -m tests.test_synth_vqvae
"""
from __future__ import annotations
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import synth_vqvae as sv                                  # noqa: E402

FAILURES = []


def check(cond, label, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def main():
    print("=== VQ-VAE regressions ===")

    # 1. Artifacts must be keyed by BOTH factor and grad_w. Keying on factor
    #    alone overwrote the MSE-only checkpoint with the gradient-loss one.
    names = {sv.paths_for(f, g)[0]
             for f in (2, 4, 8) for g in (0.0, 1.0, 0.5)}
    check(len(names) == 9, "paths_for is unique across factor x grad_w",
          f"got {len(names)} distinct names for 9 combinations")
    check(sv.paths_for(8, 0.0)[0] != sv.paths_for(8, 1.0)[0],
          "the two f=8 runs do not share a checkpoint name")

    # 2. The matched-bitrate control must follow the model's own token grid.
    #    A factor-blind baseline made the f=4 gate compare against f=8's.
    # SPATIALLY STRUCTURED input, not white noise. Noise has nothing for a
    # downsample to preserve, so every factor degrades it equally and the
    # baselines converge (measured: 0.3 dB apart on noise, 2.89 dB on real CT).
    # A test that cannot distinguish the factors cannot detect a factor-blind
    # baseline, which is the bug being pinned.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:sv.PATCH, 0:sv.PATCH] / sv.PATCH
    base = (np.sin(6 * np.pi * yy) * np.cos(4 * np.pi * xx) + 1) / 2
    X = np.clip(base[None] + 0.05 * rng.standard_normal(
        (16, sv.PATCH, sv.PATCH)), 0, 1).astype(np.float32)
    b8, b4, b2 = (sv.baseline_psnr(X, 8), sv.baseline_psnr(X, 4),
                  sv.baseline_psnr(X, 2))
    check(b8 < b4 < b2, "baseline_psnr rises as compression falls",
          f"f8 {b8:.2f}  f4 {b4:.2f}  f2 {b2:.2f}")
    check(abs(b8 - b4) > 1.0, "baseline actually differs between factors",
          f"delta {abs(b8 - b4):.3f} dB (real CT: 2.89 dB)")

    # 3. load_model must refuse a checkpoint that does not record its
    #    architecture, rather than silently assuming the default factor.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "no_factor.pt")
        torch.save({"state_dict": {}, "norm_lo": 0.0, "norm_hi": 1.0}, p)
        real = sv.paths_for
        try:
            sv.paths_for = lambda f, g=None: (p, p + ".json")
            try:
                sv.load_model(8, 1.0)
                check(False, "load_model rejects a checkpoint with no `factor`",
                      "it did NOT raise")
            except SystemExit:
                check(True, "load_model rejects a checkpoint with no `factor`")
        finally:
            sv.paths_for = real

    # 3b. The provenance guard must FAIL CLOSED. It originally read
    #     `if "src" in d:`, which silently no-ops on a pool predating that
    #     field, and those are exactly the pools most likely to have been built
    #     by a buggier boundary. A guard that can be bypassed by an old file is
    #     not a guard.
    with tempfile.TemporaryDirectory() as td:
        q = os.path.join(td, "nosrc.npz")
        np.savez_compressed(q, patches=np.zeros((2, 8, 8), np.float32),
                            norm_lo=np.float32(0), norm_hi=np.float32(1))
        try:
            sv.load_patches(q)
            check(False, "load_patches raises when `src` is absent",
                  "it did NOT raise: the guard fails open")
        except SystemExit:
            check(True, "load_patches raises when `src` is absent")

    # 4. Every factor must build and round-trip shape-correctly. The dynamic
    #    encoder/decoder is assembled in a loop, so an off-by-one in the layer
    #    count would surface as a shape error only after training.
    for f in (2, 4, 8):
        m = sv.VQVAE(factor=f)
        x = torch.randn(2, 1, sv.PATCH, sv.PATCH)
        r, _, idx = m(x)
        g = sv.PATCH // f
        check(r.shape == x.shape and tuple(idx.shape[1:]) == (g, g),
              f"factor {f} builds a {g}x{g} grid and round-trips",
              f"recon {tuple(r.shape)} idx {tuple(idx.shape)}")

    # 5. The gradient loss must be zero for identical inputs and positive for a
    #    blurred one, or it is not measuring what its name says.
    a = torch.randn(2, 1, 32, 32)
    blur = torch.nn.functional.avg_pool2d(a, 3, 1, 1)
    check(float(sv.grad_loss(a, a)) < 1e-9, "grad_loss is zero for identity")
    check(float(sv.grad_loss(blur, a)) > 0, "grad_loss penalises blur",
          f"{float(sv.grad_loss(blur, a)):.5f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
