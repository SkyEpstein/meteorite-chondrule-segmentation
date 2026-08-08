"""Regression test for the N-channel auxiliary input path.

Runs on synthetic arrays in about a second. No meteorite data is loaded, so this
is safe to run while a training job holds the machine.

Two halves:

  * FORWARD checks, always run. C channels in gives k+C channels out, channel
    order survives the dihedral augmentation, intensity jitter never touches the
    aux channels (they are probability maps, not intensity), and a malformed aux
    raises instead of silently producing the wrong shape.
  * EQUIVALENCE check against the pre-change code recovered from git. The aux
    path was widened from exactly 1 channel to N on 2026-08-04, and the
    requirement was that a single-channel aux stay BIT-IDENTICAL. This half
    recovers `src/models_cnn.py` at BASELINE_REF and compares draw for draw. It
    skips if that revision is unreachable, so a shallow clone still runs the
    forward checks.

Run:  python -m tests.test_aux_channels     (from the repo root)
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_cnn import CNNRunner, PatchSet            # noqa: E402

# Last commit before the aux path was widened.
BASELINE_REF = "18ab53a"
BASELINE_PATH = "src/models_cnn.py"

N, H, W, P = 5, 64, 64, 16
RNG = np.random.default_rng(1234)

R = RNG.integers(0, 255, size=(N, H, W)).astype(np.uint8)
M = np.zeros((N, H, W), bool)
M[:, 20:34, 18:30] = True
M[:, 44:52, 40:55] = True
V = np.ones((N, H, W), bool)
V[:, :3, :] = False                      # a non-trivial valid mask
AUX1 = RNG.random((N, H, W)).astype(np.float32)


def draw(ds, n):
    """n samples as plain arrays, so two PatchSet versions can be compared."""
    out = []
    for _ in range(n):
        inp, tgt, val = ds.sample()
        out.append((inp.numpy().copy(), tgt.numpy().copy(), val.numpy().copy()))
    return out


def _load_baseline(attr="PatchSet"):
    """`attr` from the pre-widening models_cnn, or None if that rev is unreachable."""
    try:
        src = subprocess.run(["git", "show", f"{BASELINE_REF}:{BASELINE_PATH}"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    # Import it as a package member so its relative imports resolve.
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "src", "_baseline_models_cnn.py")
    tmp = os.path.abspath(tmp)
    try:
        with open(tmp, "w") as fh:
            fh.write(src)
        from importlib import import_module
        return getattr(import_module("src._baseline_models_cnn"), attr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def check_equivalence():
    Old = _load_baseline()
    if Old is None:
        print(f"  SKIP: cannot recover {BASELINE_PATH} at {BASELINE_REF}")
        return 0
    cases = [
        ("aux=None",                dict(aux=None)),
        ("aux=None, augment",       dict(aux=None, augment=True)),
        ("aux=(n,H,W)",             dict(aux=AUX1)),
        ("aux=(n,H,W), geom",       dict(aux=AUX1, augment="geom")),
        ("aux=(n,H,W), full aug",   dict(aux=AUX1, augment=True)),
        ("aux=(n,H,W), k=3",        dict(aux=AUX1, k=3)),
    ]
    for label, kw in cases:
        a = draw(Old(R, M, V, patch=P, seed=7, **kw), 60)
        b = draw(PatchSet(R, M, V, patch=P, seed=7, **kw), 60)
        for i, ((ia, ta, va), (ib, tb, vb)) in enumerate(zip(a, b)):
            assert ia.shape == ib.shape, f"{label}: draw {i} shape moved"
            assert np.array_equal(ia, ib), f"{label}: draw {i} INPUT differs"
            assert np.array_equal(ta, tb), f"{label}: draw {i} TARGET differs"
            assert np.array_equal(va, vb), f"{label}: draw {i} VALID differs"
        print(f"  {label}: 60 draws bit-identical, shape {a[0][0].shape}")
    return len(cases)


def check_promotion():
    """(n,H,W) and (n,1,H,W) must be the same thing to the widened code."""
    a = draw(PatchSet(R, M, V, patch=P, seed=7, aux=AUX1), 40)
    b = draw(PatchSet(R, M, V, patch=P, seed=7, aux=AUX1[:, None]), 40)
    for i, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(x[0], y[0]), f"promotion INPUT differs at draw {i}"
        assert np.array_equal(x[1], y[1]), f"promotion TARGET differs at draw {i}"
        assert np.array_equal(x[2], y[2]), f"promotion VALID differs at draw {i}"
    print("  (n,H,W) and (n,1,H,W) promote identically: 40 draws")


def check_inference_path():
    """The tensor the NET SEES at val and test time, not just the training draws.

    `PatchSet` only covers training. `CNNRunner._channels` builds the input for
    every val and test forward pass and it changed too, so the single-channel
    bit-identity requirement applies here as well. No model is built and `fit`
    is never called: `_channels` needs neither.
    """
    def aux1(sl):
        """Deterministic (H,W) map derived from the slice itself."""
        return sl.astype(np.float32) / 255.0

    def aux3(sl):
        return np.stack([aux1(sl), aux1(sl) * 0.5, aux1(sl) * 0.25])

    Old = _load_baseline("CNNRunner")
    if Old is None:
        print(f"  SKIP: cannot recover {BASELINE_PATH} at {BASELINE_REF}")
    else:
        # `aux_ch` did not exist before the widening, where one channel was the
        # only behaviour, so it is passed to the current class only.
        old = Old("baseline", make_model=lambda ic: None, loss_fn=None,
                  k=1, aux_lookup=aux1)
        new = CNNRunner("current", make_model=lambda ic: None, loss_fn=None,
                        k=1, aux_lookup=aux1, aux_ch=1)
        a, b = old._channels(R), new._channels(R)
        assert a.shape == b.shape, f"_channels shape moved: {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), "_channels differs from the baseline"
        print(f"  _channels bit-identical to the baseline: {a.shape}")

    r3 = CNNRunner("aux3", make_model=lambda ic: None, loss_fn=None,
                   k=1, aux_lookup=aux3, aux_ch=3)
    got = tuple(r3._channels(R).shape)
    assert got == (N, 1 + 3, H, W), got
    print(f"  aux_ch=3 gives (n, k+3, H, W): {got}")

    rbad = CNNRunner("aux_mismatch", make_model=lambda ic: None, loss_fn=None,
                     k=1, aux_lookup=aux3, aux_ch=1)
    try:
        rbad._channels(R)
    except ValueError:
        print("  a lookup disagreeing with aux_ch rejected")
    else:
        raise AssertionError("aux_ch mismatch was NOT rejected")


def check_shapes():
    for C in (1, 2, 3, 5):
        aux = RNG.random((N, C, H, W)).astype(np.float32)
        for k in (1, 3):
            ds = PatchSet(R, M, V, patch=P, seed=7, aux=aux, k=k, augment=True)
            inp, _, _ = ds.sample()
            assert tuple(inp.shape) == (k + C, P, P), \
                f"C={C} k={k}: got {tuple(inp.shape)}"
        print(f"  C={C}: input is (k+{C}, {P}, {P}) for k=1 and k=3")


def check_channel_integrity():
    """Constant-valued channels must stay distinct, ordered, and un-jittered."""
    aux = np.zeros((N, 3, H, W), np.float32)
    aux[:, 0], aux[:, 1], aux[:, 2] = 0.25, 0.50, 0.75
    ds = PatchSet(R, M, V, patch=P, seed=3, aux=aux, k=1, augment=True)
    for _ in range(30):
        inp, _, _ = ds.sample()
        vals = [float(inp[1 + c].min()) for c in range(3)]
        assert vals == [0.25, 0.50, 0.75], f"channel order or content broke: {vals}"
    print("  channel order and content survive flips and rotations: 30 draws")

    ds = PatchSet(R, M, V, patch=P, seed=11, aux=aux, k=1, augment=True)
    for _ in range(30):
        inp, _, _ = ds.sample()
        assert float(inp[1].min()) == float(inp[1].max()) == 0.25, \
            "intensity jitter reached the aux channels"
    print("  intensity jitter leaves aux untouched: 30 draws")


def check_bad_shapes():
    for bad, desc in [(np.zeros((N, H), np.float32), "2D"),
                      (np.zeros((N, 2, 3, H, W), np.float32), "5D")]:
        try:
            PatchSet(R, M, V, patch=P, seed=0, aux=bad)
        except ValueError:
            print(f"  {desc} aux rejected")
        else:
            raise AssertionError(f"{desc} aux was NOT rejected")


def main():
    print("single-channel equivalence against the pre-widening code")
    check_equivalence()
    check_promotion()
    check_inference_path()
    print("\nN-channel behaviour")
    check_shapes()
    check_channel_integrity()
    check_bad_shapes()
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
