"""Stage 3/4 driver: sweep every remaining NN family (and the GNN) on the frozen split.

Each family is trained with the BEST regularization recipe found by the
autoresearch search (read from results/phase1/*.json), so architectures are
compared on a level footing rather than against hand-picked settings.

Usage (one family per process; run at most 2 concurrently, see constitution P10):
    python -m src.run_families list
    python -m src.run_families run <index>
"""
from __future__ import annotations
import glob
import json
import os
import sys

from . import evaluate

# Families that cannot run locally are declared here, with the reason, so the
# report never has a silent gap (constitution: no silent caps).
EXCLUDED = {
    "segformer_pretrained": "the PRETRAINED checkpoint is a large download; the "
                            "architecture family is covered from scratch by "
                            "segnext_lite / vit_seg / swin_lite below",
    "transunet_pretrained": "needs pretrained ImageNet ViT weights; the hybrid "
                            "conv+attention family is covered from scratch by "
                            "axial_unet and swin_lite",
    "resnet_unet_pretrained": "torchvision pretrained backbone requires a weights "
                              "download; resunet covers the residual-encoder family "
                              "from scratch",
    "sam": "Segment-Anything foundation model: multi-GB checkpoint, impractical locally",
    "vision_mamba": "state-space vision models need custom CUDA kernels; no usable "
                    "MPS implementation",
    "diffusion_segmentation": "iterative denoising costs orders of magnitude more "
                              "compute than the whole sweep combined",
}
# NOTE: an earlier version of this list excluded transformers outright. That was
# too conservative: only the PRETRAINED variants are impractical here. Attention
# architectures are trained FROM SCRATCH in models_transformer.py.


def best_recipe(default=None):
    """Best config from the search, ranked by full-validation IoU."""
    rows = []
    for f in glob.glob("results/phase1/*.json"):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    rows = [r for r in rows if r.get("val_iou") == r.get("val_iou")]
    if not rows:
        return default or {"lr": 3e-4, "wd": 1e-4, "aug": True, "base": 32,
                           "loss": "dice", "pos_frac": 0.5, "_source": "fallback"}
    best = max(rows, key=lambda r: r["val_iou"])
    best["_source"] = f"phase1:{best['name']} (val_iou={best['val_iou']:.4f})"
    return best


def families():
    """(name, kind, builder) for every runnable family."""
    from .models_cnn import (UNet2D, AttentionUNet2D, CNNRunner, dice_loss,
                             tversky_loss, bce_loss, focal_loss)
    rec = best_recipe()
    losses = {"dice": dice_loss, "tversky": tversky_loss,
              "bce": bce_loss, "focal": focal_loss}
    loss = losses.get(rec.get("loss", "dice"), dice_loss)
    base = int(rec.get("base", 32))

    def mk(ctor, k=1, name=None, epochs=10, steps=300):
        ep, st = globals().get("_BUDGET", (epochs, steps))
        # A budget-override run MUST get its own name. The leaderboard replaces
        # rows by model name, so without a suffix `run 0` (3k steps),
        # `run 0 30 500` (15k) and `run 0 60 500` (30k) would all write the row
        # "unet2d" and silently overwrite each other, destroying the budget
        # curve those runs exist to measure.
        if "_BUDGET" in globals():
            name = f"{name}_s{ep * st}"

        def _build():
            return CNNRunner(name, lambda in_ch: ctor(in_ch), loss, k=k,
                             epochs=ep, steps_per_epoch=st,
                             lr=float(rec.get("lr", 3e-4)),
                             weight_decay=float(rec.get("wd", 0.0)),
                             augment=bool(rec.get("aug", True)),
                             pos_frac=float(rec.get("pos_frac", 0.5)),
                             val_stride=6)
        return _build

    out = []
    # Reference architectures already in models_cnn.
    out.append(("unet2d", mk(lambda c: UNet2D(in_ch=c, base=base, depth=4),
                             name="unet2d")))
    out.append(("attention_unet", mk(lambda c: AttentionUNet2D(in_ch=c, base=base),
                                     name="attention_unet")))
    # 2.5D: adjacent slices as channels (k=5).
    out.append(("unet25d_k5", mk(lambda c: UNet2D(in_ch=c, base=base, depth=4),
                                 k=5, name="unet25d_k5")))
    # The extra families.
    try:
        from .models_extra import extra_models
        for nm, ctor in extra_models().items():
            k = 5 if nm == "convlstm" else 1   # ConvLSTM consumes a z-sequence
            out.append((nm, mk(ctor, k=k, name=nm)))
    except Exception as e:  # keep the sweep alive if that module is broken
        print(f"WARNING: models_extra unavailable ({type(e).__name__}: {e})")
    # From-scratch attention architectures and modern conv families.
    for modname, getter in (("models_transformer", "transformer_models"),
                            ("models_modern", "modern_models")):
        try:
            mod = __import__(f"src.{modname}", fromlist=[getter])
            for nm, ctor in getattr(mod, getter)().items():
                out.append((nm, mk(ctor, k=1, name=nm)))
        except Exception as e:
            print(f"WARNING: {modname} unavailable ({type(e).__name__}: {e})")
    # The GNN is its own runner type.
    try:
        from .models_gnn import GNNRunner
        out.append(("superpixel_gcn", lambda: GNNRunner()))
        # Alternative message-passing schemes on the SAME cached graph, so they
        # share the ~0.878 superpixel ceiling and differ only in aggregation.
        # `name` must be distinct per variant or they overwrite one leaderboard row.
        from .models_gnn_variants import gnn_models
        for nm, cls in gnn_models().items():
            def _mk(cls=cls, nm=nm):         # bind per-iteration, not by closure
                return GNNRunner(name=f"gnn_{nm}", model_cls=cls)
            out.append((f"gnn_{nm}", _mk))
    except Exception as e:
        print(f"WARNING: models_gnn unavailable ({type(e).__name__}: {e})")
    return out


def main():
    fams = families()
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print(f"recipe: {best_recipe().get('_source')}")
        for i, (nm, _) in enumerate(fams):
            print(f"  {i:2d}  {nm}")
        print("\nexcluded (declared, not silent):")
        for k, v in EXCLUDED.items():
            print(f"  - {k}: {v}")
        os.makedirs("results", exist_ok=True)
        json.dump(EXCLUDED, open("results/excluded_models.json", "w"), indent=2)
        return
    if len(sys.argv) > 2 and sys.argv[1] == "run":
        idx = int(sys.argv[2])
        # Optional budget override: `run <idx> <epochs> <steps>`. It drives the
        # budget-scaling curve, where the same config is trained at several
        # budgets to see whether the gains have flattened.
        if len(sys.argv) > 4:
            ep, st = int(sys.argv[3]), int(sys.argv[4])
            globals()["_BUDGET"] = (ep, st)
            fams = families()
        nm, build = fams[idx]
        rec = best_recipe()
        print(f"=== family {idx}: {nm} | recipe {rec.get('_source')} ===", flush=True)
        runner = build()
        row = evaluate.run_and_score(runner, extra={"recipe": rec.get("_source")},
                                     # Use the RUNNER's name, not the family label:
                                     # budget-override runs carry an `_sNNNNN`
                                     # suffix, and writing to the bare family name
                                     # made 3k/15k/30k overwrite one another's
                                     # probability files (same bug as the
                                     # leaderboard row collision, second site).
                                     save_prob=f"results/probs/{runner.name}.npy")
        print(f"RESULT {nm}: IoU={row.get('iou')} err={row.get('error')}", flush=True)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
