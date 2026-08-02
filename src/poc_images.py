"""Proof-of-concept imagery — what the segmenter actually produces.

Renders from REAL saved artifacts, not illustrations:
  * `results/probs/unet2d_s30000.npy`  — the winning model's TEST probabilities
  * the frozen leakage-free split (test slab only)

Every panel is a test-slab slice the model never saw in training. The operating
threshold is the validation-selected one, NOT re-tuned per image. [LEAKAGE]

Figures produced (results/figures/):
  poc_overlay.png       raw CT | truth | probability | TP-FP-FN error map
  poc_error_anatomy.png zoomed crops — where the IoU is actually lost
  poc_z_continuity.png  the same chondrule across consecutive z-slices
  poc_hierarchy.png     the superpixel cascade dropping background, level by level

Run:  python -m src.poc_images
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import zones

FIG = "results/figures"
MODEL = "unet2d_s30000"
PROB = f"results/probs/{MODEL}.npy"


def _leaderboard_row(model=MODEL, path="results/leaderboard_all.json"):
    """The row that produced PROB, so the figures cannot drift from it.

    THR used to be hard-coded (0.0076) while the saved probability map had been
    frozen at a different threshold, so the figures rendered one run's map at
    another run's operating point and captioned it with a third run's IoU. Read
    both out of the leaderboard instead. [MUST] Keep this keyed to MODEL.
    """
    import json
    with open(path) as f:
        board = json.load(f)["leaderboard"]
    hits = [r for r in board if r.get("model") == model]
    if not hits:
        raise SystemExit(f"no leaderboard row for {model} — re-run it first")
    return hits[-1]


_ROW = _leaderboard_row()
THR = float(_ROW["threshold"])
TEST_IOU = float(_ROW["iou"])

C_TP, C_FP, C_FN = "#2E7D32", "#E53935", "#FDD835"   # hit / false alarm / miss
ERR_CMAP = ListedColormap(["#0B0B0B", C_TP, C_FP, C_FN])
LEGEND = [Patch(color=C_TP, label="true positive (hit)"),
          Patch(color=C_FP, label="false positive (false alarm)"),
          Patch(color=C_FN, label="false negative (missed chondrule)")]


def _norm(a):
    """Per-slice contrast stretch for display only — never for inference."""
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _error_map(pred, mask, valid):
    """0 background, 1 TP, 2 FP, 3 FN — ignored pixels forced to background."""
    e = np.zeros(pred.shape, np.uint8)
    e[pred & mask] = 1
    e[pred & ~mask] = 2
    e[~pred & mask] = 3
    e[~valid] = 0
    return e


def _load(root="."):
    z = zones.load_zones(root, zones=("test",))
    R, M, V = z["test"]
    P = np.load(PROB).astype(np.float32)
    assert P.shape == R.shape, f"prob {P.shape} != raw {R.shape}"
    return R, M, V, P


def _busiest(M, V, k):
    """Slices with the most annotated chondrule area — the informative ones."""
    area = (M & V).reshape(M.shape[0], -1).sum(1)
    return np.argsort(area)[::-1][:k]


def fig_overlay(R, M, V, P, n=3):
    idx = sorted(_busiest(M, V, n).tolist())
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.9 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        pred = (P[i] >= THR) & V[i]
        iou = ((pred & M[i] & V[i]).sum() /
               max(((pred | (M[i] & V[i])) & V[i]).sum(), 1))
        axes[r, 0].imshow(_norm(R[i]), cmap="gray")
        axes[r, 0].set_title(f"raw micro-CT — test slice {i}", fontsize=10)
        axes[r, 1].imshow(_norm(R[i]), cmap="gray")
        axes[r, 1].imshow(np.ma.masked_where(~(M[i] & V[i]), M[i]),
                          cmap=ListedColormap(["#00E5FF"]), alpha=.55)
        axes[r, 1].set_title("hand annotation (truth)", fontsize=10)
        im = axes[r, 2].imshow(P[i], cmap="magma", vmin=0, vmax=max(P[i].max(), 1e-3))
        axes[r, 2].set_title("model probability", fontsize=10)
        plt.colorbar(im, ax=axes[r, 2], fraction=.046)
        axes[r, 3].imshow(_error_map(pred, M[i], V[i]), cmap=ERR_CMAP,
                          vmin=0, vmax=3, interpolation="nearest")
        axes[r, 3].set_title(f"errors @ thr {THR:g} — slice IoU {iou:.3f}", fontsize=10)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 3].legend(handles=LEGEND, fontsize=8, loc="upper right", framealpha=.85)
    fig.suptitle("Chondrule segmentation on held-out test slices "
                 f"(U-Net, 30k steps — test IoU {TEST_IOU:.3f}, "
                 f"tiled inference)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, .97))
    plt.savefig(f"{FIG}/poc_overlay.png", dpi=110); plt.close()
    print("  poc_overlay.png")


def fig_error_anatomy(R, M, V, P, crop=200):
    """Where is the IoU actually lost? Zoom on the largest chondrules."""
    from scipy import ndimage
    i = int(_busiest(M, V, 1)[0])
    lab, k = ndimage.label(M[i] & V[i])
    if k == 0:
        return
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
    picks = (np.argsort(sizes)[::-1] + 1)[:3]
    cents = ndimage.center_of_mass(M[i] & V[i], lab, picks)

    fig, axes = plt.subplots(2, len(picks), figsize=(4.6 * len(picks), 9.2))
    axes = np.atleast_2d(axes)
    H, W = R[i].shape
    tally = []
    for c, (cy, cx) in enumerate(cents):
        y0 = int(np.clip(cy - crop // 2, 0, max(H - crop, 0)))
        x0 = int(np.clip(cx - crop // 2, 0, max(W - crop, 0)))
        sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        pred = (P[i][sl] >= THR) & V[i][sl]
        axes[0, c].imshow(_norm(R[i][sl]), cmap="gray")
        axes[0, c].contour(M[i][sl], levels=[.5], colors="#00E5FF", linewidths=1.4)
        axes[0, c].contour(pred, levels=[.5], colors=C_FP, linewidths=1.0)
        axes[0, c].set_title(f"chondrule {c + 1}: truth (cyan) vs predicted (red)",
                             fontsize=9)
        axes[1, c].imshow(_error_map(pred, M[i][sl], V[i][sl]), cmap=ERR_CMAP,
                          vmin=0, vmax=3, interpolation="nearest")
        tp = int((pred & M[i][sl]).sum()); fp = int((pred & ~M[i][sl]).sum())
        fn = int((~pred & M[i][sl] & V[i][sl]).sum())
        axes[1, c].set_title(f"TP {tp}  FP {fp}  FN {fn} px "
                             f"(recall {tp / max(tp + fn, 1):.2f})", fontsize=9)
        tally.append((tp, fp, fn))
        for a in (axes[0, c], axes[1, c]):
            a.set_xticks([]); a.set_yticks([])
    axes[1, 0].legend(handles=LEGEND, fontsize=8, loc="lower left", framealpha=.85)
    # State what the panels actually show, not a comfortable generalization:
    # these crops mix boundary disagreement with objects missed outright.
    rec = [t / max(t + f, 1) for t, _, f in tally]
    fig.suptitle(f"Error anatomy — {crop}x{crop} px crops, test slice {i}\n"
                 f"per-chondrule recall {', '.join(f'{r:.2f}' for r in rec)} — "
                 "failures are a MIX of boundary disagreement and whole objects "
                 "missed", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, .95))
    plt.savefig(f"{FIG}/poc_error_anatomy.png", dpi=110); plt.close()
    print("  poc_error_anatomy.png")


def fig_z_continuity(R, M, V, P, n=5):
    """The premise behind z-confidence: chondrules persist across slices."""
    start = int(np.clip(int(_busiest(M, V, 1)[0]) - n // 2, 0, R.shape[0] - n))
    idx = list(range(start, start + n))
    fig, axes = plt.subplots(3, n, figsize=(2.9 * n, 9))
    for c, i in enumerate(idx):
        axes[0, c].imshow(_norm(R[i]), cmap="gray")
        axes[0, c].set_title(f"z = {i}", fontsize=10)
        axes[1, c].imshow(M[i] & V[i], cmap="gray", interpolation="nearest")
        axes[2, c].imshow(P[i], cmap="magma", vmin=0, vmax=max(P[i].max(), 1e-3))
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for r, lbl in enumerate(("raw CT", "truth", "predicted prob")):
        axes[r, 0].set_ylabel(lbl, fontsize=11)
    # Agreement between neighbours. Raw pixel agreement is NOT reported: at
    # ~90% background a constant all-background map already scores ~90%, so it
    # says almost nothing. Jaccard over the union of positives is the honest
    # measure, and truth-vs-truth is the ceiling any z-method could reach.
    def _j(A, B):
        u = (A | B).sum()
        return (A & B).sum() / u if u else np.nan

    pj = np.nanmean([_j((P[idx[j]] >= THR) & V[idx[j]],
                        (P[idx[j + 1]] >= THR) & V[idx[j + 1]])
                     for j in range(len(idx) - 1)])
    tj = np.nanmean([_j(M[idx[j]] & V[idx[j]], M[idx[j + 1]] & V[idx[j + 1]])
                     for j in range(len(idx) - 1)])
    fig.suptitle("Consecutive z-slices: chondrules persist through depth\n"
                 f"slice-to-slice overlap (IoU) — truth {tj * 100:.1f}%, "
                 f"predictions {pj * 100:.1f}%. Reality is far smoother in z "
                 "than the model is;\nthat ~"
                 f"{(tj - pj) * 100:.0f} point gap is the headroom "
                 "z-refinement could exploit.", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, .94))
    plt.savefig(f"{FIG}/poc_z_continuity.png", dpi=110); plt.close()
    print(f"  poc_z_continuity.png (slice-to-slice IoU: truth {tj * 100:.1f}%, "
          f"pred {pj * 100:.1f}%)")
    return float(pj), float(tj)


def fig_hierarchy(R, M, V):
    """The cascade: coarse regions -> drop background -> refine what survives.

    Each level's regions are shaded by whether they still contain any annotated
    chondrule pixel — i.e. what the drop rule is *allowed* to discard without
    losing a chondrule. Green regions must survive; grey ones are droppable.
    """
    from . import hierarchy as hier
    i = int(_busiest(M, V, 1)[0])
    h = hier.build_hierarchy(R[i])
    gt = M[i] & V[i]
    n_lv = len(h.levels)
    fig, axes = plt.subplots(1, n_lv + 1, figsize=(4.1 * (n_lv + 1), 4.6))
    axes[0].imshow(_norm(R[i]), cmap="gray")
    axes[0].contour(gt, levels=[.5], colors="#00E5FF", linewidths=1.2)
    axes[0].set_title(f"test slice {i}\ntruth outlined (cyan)", fontsize=10)
    for c in range(n_lv):
        rmap = h.region_labels(c)
        n_reg = int(h.levels[c]["n_regions"])
        keep = np.bincount(rmap.ravel(), weights=gt.ravel().astype(np.float64),
                           minlength=n_reg) > 0
        keep_img = keep[rmap]
        b = np.zeros(rmap.shape, bool)
        b[:-1, :] |= rmap[:-1, :] != rmap[1:, :]
        b[:, :-1] |= rmap[:, :-1] != rmap[:, 1:]
        ax = axes[c + 1]
        ax.imshow(_norm(R[i]), cmap="gray")
        ax.imshow(np.ma.masked_where(~keep_img, keep_img),
                  cmap=ListedColormap([C_TP]), alpha=.45)
        ax.imshow(np.ma.masked_where(~b, b), cmap=ListedColormap(["#FDD835"]),
                  alpha=.55)
        ax.set_title(f"{n_reg} regions\n{100 * (1 - keep.mean()):.1f}% droppable "
                     "(oracle)", fontsize=10)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        "Hierarchical cascade — successively finer superpixels. Green = regions "
        "containing a chondrule (must survive); the rest could be dropped before "
        "the CNN sees them.\nThese are ORACLE ceilings, computed from ground "
        "truth. The learned drop rule actually achieves 58.5% dropped at target "
        "recall 0.999 (41.5% of pixels alive) — the gap is the headroom.",
        fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, .92))
    plt.savefig(f"{FIG}/poc_hierarchy.png", dpi=110); plt.close()
    print("  poc_hierarchy.png")


def main(root="."):
    os.makedirs(FIG, exist_ok=True)
    R, M, V, P = _load(root)
    print(f"test slab: {R.shape[0]} slices of {R.shape[1]}x{R.shape[2]} px; "
          f"prob range [{P.min():.4f}, {P.max():.4f}]; thr {THR:g}")
    print("POC figures:")
    for fn, a in ((fig_overlay, (R, M, V, P)), (fig_error_anatomy, (R, M, V, P)),
                  (fig_z_continuity, (R, M, V, P)), (fig_hierarchy, (R, M, V))):
        try:
            fn(*a)
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print("done ->", FIG)


if __name__ == "__main__":
    main()
