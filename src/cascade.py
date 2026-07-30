"""Tier E cascade: superpixel GNN coarse map -> pixel-level CNN refinement.

Stage 1 is the tier-D superpixel GNN: it paints a coarse per-pixel probability
map whose resolution is capped by the superpixel grid (measured oracle ceiling
~0.878). Stage 2 is a pixel-level U-Net that reads the RAW slice plus that
painted map as a SECOND input channel. Because the final output is pixel-level,
the cascade is not bound by the superpixel quantization ceiling at all: the GNN
map is a context feature, not the answer.

Same leakage-free contract as the other tiers, plus one rule specific to a
cascade:

  * Normalization is ALWAYS ``raw / 255.0``, inherited from the two stage
    runners; no statistic is ever estimated from any split. [LEAKAGE][MUST]
  * The aux channel for TRAIN slices is CROSS-FITTED (out-of-fold): the train
    zone is split into two CONTIGUOUS halves (adjacent slices are near
    duplicates, so an interleaved split would leak), a GNN fit on half A
    predicts half B and vice versa. A GNN predicts its own training slices
    near-perfectly, so an in-fold aux channel would teach the refiner to blindly
    copy channel 2 and it would then collapse when channel 2 degrades at test
    time. [LEAKAGE][MUST]
  * VAL and TEST aux both come from a third GNN fit on ALL of train and nothing
    else, so no model ever sees the slices it scores. [LEAKAGE][MUST]

The aux cache is keyed by SLICE CONTENT, mirroring ``GNNRunner._disk_key``:
``predict_proba(raw)`` receives a bare array with no zone identity, so a
(zone, index) key cannot be formed here.

Self-test:  python -m src.cascade      (tiny synthetic zones; no real data)
"""
from __future__ import annotations
import hashlib
import time
import numpy as np

from . import run_families
from .models_cnn import CNNRunner, UNet2D, dice_loss
from .models_gnn import GNNRunner, N_SEGMENTS


# ---------------------------------------------------------------------------
# Runner (matches the tier-A/B/C/D fit / predict_proba API).
# ---------------------------------------------------------------------------
class CascadeRunner:
    """Two-stage coarse-to-fine segmenter; predict_proba -> (n,H,W) in [0,1].

    `.tier` = "E_cascade"; `.name` = "cascade_gnn_cnn". `gnn_epochs` is lower
    than the GNN default because `fit` trains THREE GNNs (two cross-fit halves +
    one full-train). `cnn_epochs` / `cnn_steps` default to the budget of the
    recipe the sweep selected (`run_families.best_recipe()`), which also supplies
    the refiner's lr / weight decay / augmentation / positive fraction.
    `gnn_kwargs` and `cnn_kwargs` override the GNNRunner / CNNRunner arguments
    built here.
    """

    def __init__(self, n_segments=N_SEGMENTS, gnn_epochs=25, cnn_epochs=None,
                 cnn_steps=None, gnn_kwargs=None, cnn_kwargs=None):
        self.name = "cascade_gnn_cnn"
        self.tier = "E_cascade"
        self.n_segments = n_segments
        self.gnn_epochs = gnn_epochs
        self.recipe = run_families.best_recipe()
        self.cnn_epochs = (int(self.recipe.get("epochs", 10))
                           if cnn_epochs is None else cnn_epochs)
        self.cnn_steps = (int(self.recipe.get("steps", 300))
                          if cnn_steps is None else cnn_steps)
        self.gnn_kwargs = gnn_kwargs
        self.cnn_kwargs = cnn_kwargs
        self.gnn_full = None            # fit on ALL of train: aux for val/test
        self.refiner = None             # the pixel-level CNN
        self.best_thr = 0.5
        self._aux_cache = {}            # slice content -> aux map

    # -- aux plumbing --------------------------------------------------------
    def _key(self, sl):
        """Content key for one slice (sha1 of its bytes, as in GNNRunner._disk_key)."""
        h = hashlib.sha1(np.ascontiguousarray(sl).tobytes()).hexdigest()[:16]
        return (sl.shape, h)

    def _make_gnn(self, name):
        """One stage-1 GNN with this cascade's knobs (overridable by gnn_kwargs)."""
        kw = dict(name=name, n_segments=self.n_segments, epochs=self.gnn_epochs)
        kw.update(self.gnn_kwargs or {})
        return GNNRunner(**kw)

    def _aux(self, sl):
        """(H,W) raw slice -> (H,W) float32 GNN map, the CNN's extra channel.

        A cache HIT is a train slice (its OUT-OF-FOLD map) or a val slice (the
        full-train GNN's map). A cache MISS is the TEST path, and is correct by
        construction: test aux is computed by `self.gnn_full`, which was fit on
        train only and has never seen a test slice. [LEAKAGE][MUST]
        """
        key = self._key(sl)
        a = self._aux_cache.get(key)
        if a is None:
            a = self.gnn_full.predict_proba(sl[None])[0]
            self._aux_cache[key] = a
        return a

    # -- API -----------------------------------------------------------------
    def fit(self, train, val):
        """Train on `train` ONLY; use `val` ONLY for thresholds/selection. [LEAKAGE]

        train/val are dicts {"raw":(n,H,W) float32, "mask":bool, "valid":bool}.
        No test set is passed or referenced anywhere in this method.
        """
        t0 = time.time()
        n = train["raw"].shape[0]
        mid = n // 2
        keys = ("raw", "mask", "valid")

        # --- 1. CROSS-FITTED train aux, on CONTIGUOUS halves. [LEAKAGE][MUST] ---
        # Contiguous (not interleaved): adjacent z-slices are near-duplicates, so
        # an interleaved split would put a near-copy of every held-out slice in
        # the fitting half and the "out-of-fold" map would be in-fold in effect.
        half_a = {k: train[k][:mid] for k in keys}
        half_b = {k: train[k][mid:] for k in keys}
        print(f"  [{self.name}] cross-fit halves: A={mid} B={n - mid} slices",
              flush=True)
        # `val` is passed only because GNNRunner.fit wants it for a threshold; the
        # half-GNNs' thresholds are discarded: only their painted maps are used.
        gnn_a = self._make_gnn(f"{self.name}_gnn_A").fit(half_a, val)
        oof_b = gnn_a.predict_proba(half_b["raw"])
        gnn_b = self._make_gnn(f"{self.name}_gnn_B").fit(half_b, val)
        oof_a = gnn_b.predict_proba(half_a["raw"])

        # --- 2. Full-train GNN: the aux source for VAL and TEST. ---
        self.gnn_full = self._make_gnn(f"{self.name}_gnn_full").fit(train, val)

        # --- 3. Aux lookup, keyed by slice CONTENT (no zone identity exists). ---
        self._aux_cache = {}
        for i in range(mid):
            self._aux_cache[self._key(train["raw"][i])] = oof_a[i]
        for j in range(n - mid):
            self._aux_cache[self._key(train["raw"][mid + j])] = oof_b[j]
        val_aux = self.gnn_full.predict_proba(val["raw"])
        for i in range(val["raw"].shape[0]):
            self._aux_cache[self._key(val["raw"][i])] = val_aux[i]
        print(f"  [{self.name}] aux cache: {len(self._aux_cache)} slices "
              f"({n} train out-of-fold + {val['raw'].shape[0]} val full-train) "
              f"in {time.time() - t0:.0f}s", flush=True)

        # --- 4. Pixel-level refiner: raw + GNN map -> pixel probabilities. ---
        rec = self.recipe
        kw = dict(k=1, epochs=self.cnn_epochs, steps_per_epoch=self.cnn_steps,
                  lr=float(rec.get("lr", 3e-4)),
                  weight_decay=float(rec.get("wd", 0.0)),
                  augment=bool(rec.get("aug", True)),
                  pos_frac=float(rec.get("pos_frac", 0.5)),
                  aux_lookup=self._aux)
        kw.update(self.cnn_kwargs or {})
        print(f"  [{self.name}] refiner recipe {rec.get('_source')} "
              f"epochs={kw['epochs']} steps={kw['steps_per_epoch']}", flush=True)
        self.refiner = CNNRunner(
            f"{self.name}_refiner",
            lambda in_ch: UNet2D(in_ch=in_ch, base=32, depth=4),
            dice_loss, **kw)
        self.refiner.fit(train, val)
        self.best_thr = self.refiner.best_thr
        print(f"  [{self.name}] done in {time.time() - t0:.0f}s "
              f"thr={self.best_thr:.2f}", flush=True)
        return self

    def predict_proba(self, raw):
        """(n,H,W) raw -> (n,H,W) float32 probabilities in [0,1].

        Delegated to the refiner, which pulls each slice's GNN map through
        `aux_lookup` (= `self._aux`) as its second input channel.
        """
        assert self.refiner is not None, "call fit() before predict_proba"
        return self.refiner.predict_proba(raw)


# ---------------------------------------------------------------------------
# Self-test: tiny synthetic zones only (no real volume, no leaderboard write).
# ---------------------------------------------------------------------------
def _synthetic_zone(n, H=96, W=120, seed=0):
    """Tiny zone dict: a bright disk (the "chondrule") on noisy background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - 48) ** 2 + (xx - 60) ** 2 <= 22 ** 2
    R = np.zeros((n, H, W), np.float32)
    for i in range(n):
        img = rng.random((H, W)) * 60.0 + 40.0
        img[disk] += 120.0
        R[i] = np.clip(img, 0.0, 255.0)
    M = np.repeat(disk[None], n, axis=0)
    V = np.ones((n, H, W), bool)
    return {"raw": R, "mask": M, "valid": V}


if __name__ == "__main__":
    train = _synthetic_zone(8, seed=0)
    val = _synthetic_zone(4, seed=1)

    runner = CascadeRunner(n_segments=60, gnn_epochs=3, cnn_epochs=2, cnn_steps=5)
    runner.fit(train, val)

    prob = runner.predict_proba(val["raw"])
    assert prob.shape == (4, 96, 120), prob.shape
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0, \
        (float(prob.min()), float(prob.max()))

    in_ch = runner.refiner.model.inc.block[0].in_channels
    assert in_ch == 2, in_ch                       # 1 raw channel + 1 GNN aux

    assert len(runner._aux_cache) > 0, "aux cache empty"
    aux = runner._aux(train["raw"][0])
    assert aux.shape == (96, 120), aux.shape
    assert float(aux.min()) >= 0.0 and float(aux.max()) <= 1.0, \
        (float(aux.min()), float(aux.max()))

    print(f"  train {train['raw'].shape}  val {val['raw'].shape}")
    print(f"  refiner in_ch={in_ch}  aux cache entries={len(runner._aux_cache)}")
    print(f"  aux map {aux.shape} range [{aux.min():.3f}, {aux.max():.3f}]")
    print(f"  predict_proba(val) {prob.shape} range "
          f"[{prob.min():.3f}, {prob.max():.3f}]  thr={runner.best_thr:.2f}")
    print("cascade self-test: OK")
