"""Data-agnostic pipeline v2 — the validated workflow, run over MORE METEORITES.

The binding constraint on this project is DATA, not method. Measured on the one
meteorite we have (ALH 83010: 531 kept slices, 397 annotated): **142 3D
chondrules in the whole volume, only 19 in validation**. A bootstrap over
validation slices puts the SE of a val-IoU difference at +/-0.0071 IoU, so the
minimum detectable effect is ~0.014 IoU — larger than almost every method effect
measured, which is why nine val/test comparisons flipped sign. Adding meteorites
is the only lever that moves that floor.

[CORRECTED 2026-08-04] That sign-flip count read six when this docstring was
written. It is nine as of 2026-08-04; the pretraining probe was the ninth.

[CORRECTED 2026-08-04] "only 19 in validation" mixes two counting rules. 142 is
the same-id 3D total; 19 is the BINARY connected-components count of the val
zone, and the same-id rule gives 36 there. `recount_chondrules.py` measured both
over identical voxels: train 63/49, val 36/19, test 43/24. `SE_REF_N` was left at
19 deliberately, because the +/-0.0071 it pairs with was bootstrapped over 60
validation SLICES, so neither chondrule count was ever the resampling unit. The
0.0071 and the ~0.014 floor are unchanged, and so is the conclusion.

[CORRECTED 2026-08-04] Adding a meteorite was NOT a config change. Two scans of
different in-plane size crashed `build_multi_split` at the concatenate below.
Fixed by padding to the common in-plane size with the padding marked invalid
(`tests/test_multi_meteorite.py` covers it), but `run_all` with two sources has
still never completed end to end. What the config below does today:

    srcs = [MeteoriteSource("ALH83010", "ALH83010_z_crop_every4",
                            "ALH83010_z_crop_every4_labeled"),
            MeteoriteSource("NWA_xxxx", "NWA_raw", "NWA_labeled")]
    run_all(srcs)

Defaults bake in what actually won the 36-configuration sweep — plain `UNet2D`,
30,000 steps, lr 3e-4 with cosine decay, dice loss, base 32, depth 4, NO
augmentation, pos_frac 0.5. Training budget was the largest measured lever
(3k -> 0.2314, 7.2k -> 0.2834, 15k -> 0.3089, 30k -> 0.3159 test IoU
[WITHDRAWN: all four predate the 2026-07-29 tiler fix; the 30k point re-scores
to 0.29409 post-fix, so read the SHAPE not the heights]), bigger
than any architecture change. The hierarchy is used as a FILTER only (2.71x
compute saving at 99.9% chondrule retention); as a standalone segmenter it
reaches 0.145 IoU. Confidence / prune / z-refine gains were all inside the noise
floor on the best model, so `confidence_sweep` RE-MEASURES them rather than
assuming them.

Leakage contract [LEAKAGE][MUST]:
  * every meteorite is split into its OWN contiguous train/val/test z-slabs with
    its OWN buffer, and the slabs are concatenated — so no meteorite lands only
    in test and no slab seam is shared across meteorites;
  * the hierarchy's drop thresholds are fit on VAL and frozen for train/test;
  * the second-stage architecture winner is picked on VALIDATION;
  * every meta-model in `confidence_sweep` is fit on VAL and touches test only
    through `confidence_bakeoff.evaluate_on_test`.

Self-test:  python -m src.pipeline_v2      (plumbing only — no training)
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi

from . import combine, confidence_bakeoff as cb, data, evaluate, hierarchy
from . import metrics, split as splitmod
from .models_cnn import AttentionUNet2D, CNNRunner, UNet2D, dice_loss
from .models_extra import extra_models
from .models_modern import modern_models

ZONES = ("train", "val", "test")
OUT_DIR = "results_v2"
SPLIT_JSON = os.path.join(OUT_DIR, "split_multi.json")

# The winning recipe (plain UNet2D beat 35 other configurations).
BEST_RECIPE = dict(lr=3e-4, base=32, depth=4, pos_frac=0.5, augment=False,
                   patch=128, batch=16, val_stride=6)
STEPS_PER_EPOCH = 500            # budget 30000 -> 60 epochs x 500 steps
DEFAULT_BUDGET = 30000

# Hierarchy as a FILTER: levels coarsest-first, per-level recall target 0.999
# (relaxing 1.0 -> 0.999 raised the compute saving 1.66x -> 2.70x for 0.05% of
# chondrule pixels).
HIER_LEVELS = (80, 320, 1280, 6000)
HIER_TARGET_RECALL = 0.999

# `confidence_bakeoff.run_bakeoff` opens the base model up to this recall before
# pruning; repeated here so `confidence_sweep` can rebuild the SAME threshold.
BAKEOFF_TARGET_RECALL = 0.95
PRUNE_RECALL_TARGETS = (0.80, 0.90, 0.95, 0.98)

# Measured noise floor: bootstrap SE of a val-IoU difference is 0.0071 IoU at
# 19 validation chondrules. SE scales as 1/sqrt(n).
SE_REF_IOU = 0.0071
SE_REF_N = 19
POWER_REFERENCE_N = (19, 40, 80, 150)


# ---------------------------------------------------------------------------
# 1. A meteorite = one raw/label directory pair.
# ---------------------------------------------------------------------------
@dataclass
class MeteoriteSource:
    """One meteorite's raw + label directory pair.

    `every_n` records the slice subsampling of the ORIGINAL scan (ALH 83010 kept
    every 4th of ~2121 slices), so a buffer of B kept-slices is B * every_n
    original slices of physical separation.

    `voxel_um` is the in-plane micrometres per pixel. It is recorded, not acted
    on. Padding reconciles differing pixel COUNTS but not differing pixel SIZES:
    if two scans differ in resolution, chondrules appear at different pixel
    scales and this model has no scale invariance. Recording it means the
    mismatch is visible the moment a second scan lands instead of being
    discovered later.
    """

    name: str
    raw_dir: str
    label_dir: str
    every_n: int = 4
    voxel_um: float | None = None

    def list_pairs(self):
        """Return (raw_paths, lab_paths), sorted and 1:1 by z-order.

        Mirrors `data.list_pairs`, but reads THIS meteorite's directories rather
        than the module-level ALH 83010 constants.
        """
        raw = sorted(glob.glob(os.path.join(self.raw_dir, "*.tif")))
        lab = sorted(glob.glob(os.path.join(self.label_dir, "*.tif")))
        if not raw or not lab:
            raise FileNotFoundError(
                f"{self.name}: missing tifs under {self.raw_dir}/ or "
                f"{self.label_dir}/")
        if len(raw) != len(lab):
            raise ValueError(f"{self.name}: slice count mismatch: "
                             f"raw={len(raw)} lab={len(lab)}")
        return raw, lab


# ---------------------------------------------------------------------------
# 2. Per-meteorite contiguous z-slab split, concatenated.
# ---------------------------------------------------------------------------
def zone_ranges(N, val_frac, test_frac, buffer):
    """Band-relative (val_zone, test_zone) inclusive ranges for a band of N.

    Layout in z order:  TRAIN | buffer | VAL | buffer | TEST. Slabs are
    CONTIGUOUS — never random slices — because adjacent kept-slices are near
    duplicates (adjacent-slice label IoU ~0.83), so a random split would put a
    slice's near-twin in another zone.
    """
    n_val = int(round(float(val_frac) * N))
    n_test = int(round(float(test_frac) * N))
    assert n_val >= 1 and n_test >= 1, \
        f"val/test fractions too small for band N={N}: n_val={n_val} n_test={n_test}"
    t1 = N - 1
    t0 = t1 - n_test + 1
    v1 = t0 - buffer - 1
    v0 = v1 - n_val + 1
    assert v0 - buffer >= 1, \
        (f"no train slices left: N={N} val_frac={val_frac} test_frac={test_frac} "
         f"buffer={buffer}")
    return (v0, v1), (t0, t1)


def count_chondrules_3d(ids, valid=None):
    """Number of distinct 3D chondrules, by SAME-ID connected components.

    Takes the annotator's instance-ID volume, not a binarized mask. A plain
    binary CC fuses touching chondrules into mega-clusters — the failure that
    once made this project drop 53.6% of positive pixels, fixed in
    `split.instance_label`. Using binary CC here would undercount: on ALH 83010
    it gives 87 where the same-id rule gives 142, a 39% shortfall.

    That matters because this count is the `n` in `power_report`: get it wrong
    and every minimum-detectable-effect figure downstream is wrong too.

    Counted PER METEORITE PER ZONE and then summed, never over the concatenated
    stack — concatenating two meteorites would fuse components across the join.
    """
    ids = np.asarray(ids)
    if valid is not None:
        ids = np.where(valid, ids, 0).astype(ids.dtype)
    _, k = splitmod.instance_label(ids)
    return int(k)


def _straddler_ignore(lab_paths, idxs, slab, name):
    """(N,H,W) bool — voxels of a seam-crossing chondrule OUTSIDE its own slab.

    Identical rule to `split.build_split`, applied per meteorite: same-id 3D
    connected components (a plain binary CC fuses touching chondrules into
    mega-clusters and over-drops), then each straddler is assigned to the slab
    holding most of its voxels and ignored elsewhere. Asserts afterwards that
    no chondrule survives in two core zones. [LEAKAGE][MUST]

    Memory: the int32 instance volume is ~4x the band and is freed as soon as
    `ignore` is derived, because the caller still has to load raw slices.
    """
    vol_ids = data.load_stack(lab_paths, idxs, kind="raw")   # uint8 instance ids
    N = vol_ids.shape[0]
    inst, ninst = splitmod.instance_label(vol_ids)
    spans = splitmod._component_zone_span(inst, slab)
    straddlers = [cid for cid, z in spans.items() if len(z) >= 2]

    zonecode = np.array([{"train": 1, "val": 2, "test": 3}.get(str(slab[i]), 0)
                         for i in range(N)])
    counts = np.zeros((ninst + 1, 4))
    for z in range(N):
        counts[:, zonecode[z]] += np.bincount(inst[z].ravel(), minlength=ninst + 1)
    maj = np.zeros(ninst + 1, dtype=np.int64)
    for cid in straddlers:
        maj[cid] = int(np.argmax(counts[cid, 1:4])) + 1      # 1=train 2=val 3=test

    ignore = np.zeros(vol_ids.shape, dtype=bool)
    for z in range(N):
        zc = zonecode[z]
        if zc == 0:                                          # buffer slice: unused
            continue
        m = maj[inst[z]]
        ignore[z] = (m != 0) & (m != zc)
    del inst

    # Verify, don't assume: recount on the volume with ignored voxels removed.
    inst2, _ = splitmod.instance_label(
        np.where(ignore, 0, vol_ids).astype(vol_ids.dtype))
    remaining = [cid for cid, z
                 in splitmod._component_zone_span(inst2, slab).items()
                 if len(z) >= 2]
    del inst2, vol_ids
    assert not remaining, (f"{name}: leakage remains after resolving — "
                           f"{len(remaining)} chondrule(s) still span 2+ zones")
    print(f"  [{name}] {len(straddlers)} straddler instance(s) resolved to "
          f"majority slab; {int(ignore.sum())} voxels ignored; "
          f"0 chondrules span 2+ zones", flush=True)
    return ignore


def build_multi_split(sources, val_frac=0.15, test_frac=0.2, buffer=12,
                      out_dir=None):
    """Per-meteorite contiguous z-slab split, CONCATENATED across meteorites.

    [LEAKAGE][MUST] Every meteorite contributes its own train/val/test slabs
    with its own buffer, so no chondrule spans a seam and no meteorite appears
    only in test. The slab geometry comes from `zone_ranges` and is realized by
    `split.assign_slabs`, the same function the frozen single-meteorite split
    uses.

    Returns `{"train": zone, "val": zone, "test": zone, "meta": {...}}` where a
    zone is `{"raw": (n,H,W) float32, "mask": bool, "valid": bool,
    "provenance": (n,) object, "abs_slices": (n,) int32}`. `provenance` records
    which meteorite each slice came from. The zone dicts carry exactly the keys
    `evaluate.run_and_score` reads, so the result can be passed straight in as
    its `data` argument.

    Per-zone 3D chondrule counts are printed and stored in
    `results_v2/split_multi.json`.
    """
    acc = {z: {"raw": [], "mask": [], "valid": [], "provenance": [],
               "abs_slices": []} for z in ZONES}
    per_source = []
    chondrules = {z: 0 for z in ZONES}

    for src in sources:
        raw_paths, lab_paths = src.list_pairs()
        data.check_alignment(raw_paths, lab_paths)
        fg = data.fg_fraction(lab_paths)
        b0, b1 = data.annotated_band(fg)
        idxs = list(range(b0, b1 + 1))
        N = len(idxs)
        val_zone, test_zone = zone_ranges(N, val_frac, test_frac, buffer)
        slab = splitmod.assign_slabs(N, val_zone, test_zone, buffer)

        entry = {
            "name": src.name,
            "raw_dir": src.raw_dir,
            "label_dir": src.label_dir,
            "every_n": int(src.every_n),
            "voxel_um": (None if src.voxel_um is None else float(src.voxel_um)),
            "native_shape": None,
            "n_pairs": int(len(raw_paths)),
            "band": [int(b0), int(b1)],
            "N": int(N),
            "buffer": int(buffer),
            "buffer_original_slices": int(buffer * src.every_n),
            "val_zone_band": [int(v) for v in val_zone],
            "test_zone_band": [int(v) for v in test_zone],
            "n_buffer_slices": int(sum(1 for i in range(N)
                                       if slab[i] == "buffer")),
            "zone": {},
        }

        # --- straddler resolution [LEAKAGE][MUST] -------------------------
        # The buffer ALONE is not sufficient. `split.py` measured chondrule
        # z-extents reaching 70 band-slices against a 12-slice buffer, so a
        # chondrule can appear in two core zones despite the gap — the model
        # would then be trained on the same physical object it is tested on.
        # Reuse the audited same-id-CC + assign-to-majority-slab rule rather
        # than reimplementing it: each straddler is kept in the slab holding
        # most of its voxels and IGNORED (via `valid`) in the others.
        ignore = _straddler_ignore(lab_paths, idxs, slab, src.name)

        for z in ZONES:
            zpos = [i for i in range(N) if slab[i] == z]
            sel = [idxs[i] for i in zpos]
            M = data.load_stack(lab_paths, sel, kind="mask")
            V = ~ignore[zpos]
            # Count instances on what actually SURVIVES in this zone, not on
            # the raw mask — an ignored straddler is not a countable chondrule
            # here, and the count is what the power calculation consumes.
            zids = data.load_stack(lab_paths, sel, kind="raw")   # instance ids
            k = count_chondrules_3d(zids, V)
            del zids
            R = data.load_stack(raw_paths, sel, kind="raw").astype(np.float32)
            acc[z]["raw"].append(R)
            acc[z]["mask"].append(M)
            acc[z]["valid"].append(V)
            acc[z]["provenance"].extend([src.name] * len(sel))
            acc[z]["abs_slices"].extend(int(i) for i in sel)
            chondrules[z] += k
            entry["zone"][z] = {
                "n_slices": int(len(sel)),
                "abs_slices": [int(i) for i in sel],
                "n_chondrules_3d": int(k),
                "pos_pixels": int(M.sum()),
                "pos_frac": float(M.mean()),
            }
        h, w = R.shape[1:]
        entry["native_shape"] = [int(h), int(w)]
        per_source.append(entry)

    # Meteorites do not share in-plane dimensions, and concatenating them raw
    # raises a numpy error that says nothing about scans. Pad each up to the
    # common maximum and mark the padding INVALID. `valid` excludes the padding
    # from every loss and every metric, which is what makes the fix sound.
    # It does NOT hide the padding from everything downstream, on two counts.
    # `PatchSet.sample` draws negative patch centres uniformly and never
    # consults `valid`, so padded area costs a proportional share of negative
    # draws. And `_norm` is GroupNorm, which normalises over whatever spatial
    # extent an inference tile covers, so a tile straddling zero padding shifts
    # the predictions for the real pixels inside it. Padding is anchored
    # top-left, so original pixel coordinates are unchanged.
    shapes = sorted({tuple(int(v) for v in a.shape[1:])
                     for z in ZONES for a in acc[z]["raw"]})
    Hm = max(s[0] for s in shapes)
    Wm = max(s[1] for s in shapes)
    padded = len(shapes) > 1

    def _pad(a, fill):
        ph, pw = Hm - a.shape[1], Wm - a.shape[2]
        if not ph and not pw:
            return a
        return np.pad(a, ((0, 0), (0, ph), (0, pw)),
                      mode="constant", constant_values=fill)

    out = {}
    for z in ZONES:
        out[z] = {
            "raw": np.concatenate([_pad(a, 0) for a in acc[z]["raw"]], axis=0),
            "mask": np.concatenate([_pad(a, False) for a in acc[z]["mask"]], axis=0),
            "valid": np.concatenate([_pad(a, False) for a in acc[z]["valid"]], axis=0),
            "provenance": np.array(acc[z]["provenance"], dtype=object),
            "abs_slices": np.asarray(acc[z]["abs_slices"], np.int32),
        }

    meta = {
        "regime": "per-meteorite contiguous z-slab split (train | buffer | val "
                  "| buffer | test), concatenated across meteorites; "
                  "seam-crossing chondrules resolved to their majority slab "
                  "and ignored elsewhere via `valid` (same-id 3D CC, the rule "
                  "audited in split.py) — the buffer alone is NOT sufficient, "
                  "measured chondrule z-extent reaches 70 band-slices; "
                  "3D chondrule counts by scipy.ndimage.label on mask & valid, "
                  "per meteorite per zone",
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "buffer": int(buffer),
        "padded_to": [int(Hm), int(Wm)],
        "native_shapes": [list(s) for s in shapes],
        "padding_applied": bool(padded),
        "n_meteorites": int(len(per_source)),
        "meteorites": per_source,
        # `pos_frac` is over ALL pixels, so padding dilutes it whenever
        # meteorites differ in size. Read `pos_frac_valid` in that case.
        "zone_stats": {
            z: {"n_slices": int(out[z]["raw"].shape[0]),
                "n_chondrules_3d": int(chondrules[z]),
                "pos_pixels": int(out[z]["mask"].sum()),
                "pos_frac": float(out[z]["mask"].mean()),
                "pos_frac_valid": float(out[z]["mask"][out[z]["valid"]].mean())
                                  if out[z]["valid"].any() else float("nan"),
                "provenance": list(out[z]["provenance"])}
            for z in ZONES
        },
        "n_chondrules_3d": {z: int(chondrules[z]) for z in ZONES},
    }
    out["meta"] = meta

    # `out_dir=None` keeps the historical destination. A synthetic or smoke run
    # MUST pass its own, or it silently overwrites the real split record with a
    # split of fake data. Same reason `run_all` threads out_dir everywhere.
    split_dir = OUT_DIR if out_dir is None else out_dir
    split_json = os.path.join(split_dir, os.path.basename(SPLIT_JSON))
    os.makedirs(split_dir, exist_ok=True)
    with open(split_json, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    print(f"=== multi-meteorite split ({len(per_source)} meteorite(s)) ===",
          flush=True)
    if padded:
        print(f"  NOTE: meteorites differ in size {shapes}; padded to "
              f"{Hm}x{Wm} and the padding marked invalid", flush=True)
    for e in per_source:
        print(f"  {e['name']}: {e['n_pairs']} pairs, band {e['band']} "
              f"(N={e['N']}), val_zone={e['val_zone_band']} "
              f"test_zone={e['test_zone_band']}, buffer={e['buffer']} "
              f"(~{e['buffer_original_slices']} original slices)", flush=True)
    print(f"  {'zone':>6} {'slices':>8} {'3D chondrules':>15} {'pos_frac':>10}",
          flush=True)
    for z in ZONES:
        s = meta["zone_stats"][z]
        print(f"  {z:>6} {s['n_slices']:>8} {s['n_chondrules_3d']:>15} "
              f"{s['pos_frac']:>10.5f}", flush=True)
    print(f"  wrote {split_json}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 3. Hierarchical second-stage sweep.
# ---------------------------------------------------------------------------
def default_archs():
    """Name -> constructor(in_ch) for the second-stage candidates."""
    mm, em = modern_models(), extra_models()
    return {
        "unet2d": lambda in_ch: UNet2D(in_ch=in_ch,
                                       base=BEST_RECIPE["base"],
                                       depth=BEST_RECIPE["depth"]),
        "attention_unet": lambda in_ch: AttentionUNet2D(
            in_ch=in_ch, base=BEST_RECIPE["base"]),
        "resunet": mm["resunet"],
        "hrnet_lite": mm["hrnet_lite"],
        "unetpp": em["unetpp"],
    }


def _slice_key(sl):
    """sha1 of one raw slice's bytes — the content key for its alive mask."""
    return hashlib.sha1(np.ascontiguousarray(sl).tobytes()).hexdigest()


def alive_index(zones, alive_masks):
    """sha1(raw slice) -> that slice's (H,W) alive mask, over every zone.

    `CNNRunner.predict_proba` receives only the raw stack — and `fit` calls it on
    a STRIDED val subset — so the alive mask cannot be looked up by zone or by
    position. It is keyed by slice CONTENT instead, the same scheme
    `hierarchy._disk_key` uses for the graph cache.
    """
    idx = {}
    for z in ZONES:
        R, A = zones[z]["raw"], alive_masks[z]
        for i in range(R.shape[0]):
            idx[_slice_key(R[i])] = np.asarray(A[i], bool)
    return idx


class HierMaskedRunner(CNNRunner):
    """`CNNRunner` restricted to the hierarchy's ALIVE pixels.

    Training: the alive mask is intersected into `valid`, so a pixel the cascade
    already dropped contributes exactly zero to the loss and to the threshold
    selection. Inference: the probability map is multiplied by the alive mask, so
    a dropped pixel can never be predicted positive.

    The evaluation `valid` mask is NOT narrowed — `evaluate.run_and_score` scores
    against the zone's own valid mask, so a chondrule the cascade dropped counts
    as a false negative rather than disappearing. [LEAKAGE][MUST]
    """

    def __init__(self, *args, alive_index=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alive_index = alive_index

    def _alive_stack(self, raw):
        return np.stack([self.alive_index[_slice_key(raw[i])]
                         for i in range(raw.shape[0])])

    def _restrict(self, zone):
        return {"raw": zone["raw"], "mask": zone["mask"],
                "valid": zone["valid"] & self._alive_stack(zone["raw"])}

    def fit(self, train, val):
        return super().fit(self._restrict(train), self._restrict(val))

    def predict_proba(self, raw):
        prob = super().predict_proba(raw)
        return (prob * self._alive_stack(raw)).astype(np.float32)


def stage2_sweep(zones, alive_masks, budgets=(DEFAULT_BUDGET,), archs=None,
                 out_dir=OUT_DIR, leaderboard=False, recipe=None,
                 steps_per_epoch=STEPS_PER_EPOCH):
    """Train each candidate architecture INSIDE the hierarchy's alive mask.

    `zones` is a `build_multi_split` result; `alive_masks` maps each zone to its
    (n,H,W) bool `final_alive_mask` from the cascade. `archs` is a name ->
    constructor(in_ch) mapping (default: `default_archs()`), `budgets` a list of
    TOTAL training steps (each becomes `budget // steps_per_epoch` epochs).
    `recipe` overrides individual `BEST_RECIPE` keys, which is what lets a smoke
    run use a patch smaller than the image.

    [CORRECTED 2026-08-08] `leaderboard` now defaults to **False**, and the old
    behaviour was a latent defect rather than a choice. This function scores
    against a `build_multi_split` split, which is NOT the frozen `split.json`
    roster, and `run_and_score`'s own docstring marks `leaderboard=False` a
    [MUST] for exactly that case ("a different split, a data subset, or a
    reduced budget"). It was passing nothing, so the first successful `run_all`
    would have written five rows onto the shared leaderboard under a different
    split, where `fig_leaderboard` plots every row as one comparable ranking.
    Nothing was corrupted because `stage2_sweep` has never completed: there are
    zero `hier_*` rows in `results/leaderboard_all.json` today. Pass
    `leaderboard=True` deliberately if a run really is roster-comparable.

    Returned rows carry `val_iou` (the runner's frozen full-validation IoU) and
    are RANKED BY IT — the winner must be chosen on validation, never on the
    test column. [LEAKAGE][MUST]
    """
    archs = default_archs() if archs is None else archs
    rec = dict(BEST_RECIPE) if recipe is None else {**BEST_RECIPE, **recipe}
    aidx = alive_index(zones, alive_masks)
    rows = []
    for budget in budgets:
        epochs = max(1, int(budget) // steps_per_epoch)
        for name, ctor in archs.items():
            run_name = f"hier_{name}_s{epochs * steps_per_epoch}"
            runner = HierMaskedRunner(
                run_name, ctor, dice_loss, k=1, epochs=epochs,
                steps_per_epoch=steps_per_epoch, batch=rec["batch"],
                patch=rec["patch"], pos_frac=rec["pos_frac"],
                lr=rec["lr"], augment=rec["augment"],
                val_stride=rec["val_stride"], alive_index=aidx)
            row = evaluate.run_and_score(
                runner, data=zones,
                extra={"arch": name, "budget": int(epochs * steps_per_epoch),
                       "stage": "hier_stage2"},
                save_prob=os.path.join(out_dir, "probs", f"{run_name}.npy"),
                leaderboard=leaderboard)
            row["val_iou"] = float(getattr(runner, "best_full_val_iou",
                                           float("nan")))
            print(f"[stage2] {run_name:28s} val IoU={row['val_iou']:.4f} "
                  f"test IoU={row.get('iou', float('nan'))} "
                  f"{row.get('sec')}s", flush=True)
            rows.append(row)
    return sorted(rows, key=lambda r: (r["val_iou"] if r["val_iou"] == r["val_iou"]
                                       else -1.0), reverse=True)


# ---------------------------------------------------------------------------
# 4. Confidence / prune / z-refine — re-measured, never assumed.
# ---------------------------------------------------------------------------
def confidence_sweep(pv, raw_v, Mv, Vv, pt, raw_t, Mt, Vt, thr):
    """Everything below the model: bake-off, free baselines, prune sweep, z-refine.

    Every gain here was inside the noise floor on the best single-meteorite
    model, so all four blocks are RUN and returned together rather than assumed:

      1. `confidence_bakeoff.run_bakeoff` on VAL (10 learned candidates + 4 free
         baselines), plus the val-selected winner's TEST metrics through
         `evaluate_on_test` (with `thr_feat=thr`, so the score map's features are
         built at the same threshold the estimator was fit at);
      2. the free-baseline comparison — the best learned candidate against the
         best zero-training control;
      3. the FP-prune recall sweep over 0.80 / 0.90 / 0.95 / 0.98;
      4. cross-slice z-refinement.

    [LEAKAGE][MUST] Every estimator, cutoff and threshold is fit or selected on
    the VAL arrays; the test arrays are only ever scored with frozen parameters.
    """
    out = {"baseline_test": metrics.metrics_from_pred(pt >= thr, Mt, Vt),
           "threshold": float(thr)}
    base_iou = float(out["baseline_test"]["iou"])
    print(f"[base] test IoU={base_iou:.4f} "
          f"P={out['baseline_test']['precision']:.3f} "
          f"R={out['baseline_test']['recall']:.3f}", flush=True)

    # --- 1. bake-off on VAL, then the winner once on TEST ---
    print("\n=== confidence bake-off (learned candidates + free baselines) ===",
          flush=True)
    board, fitted = cb.run_bakeoff(pv, raw_v, Mv, Vv, thr,
                                   target_recall=BAKEOFF_TARGET_RECALL)
    out["bakeoff"] = board
    thr_rec, rec_m = combine.recall_threshold(pv, Mv, Vv,
                                              target_recall=BAKEOFF_TARGET_RECALL)
    win = board[0]
    est = fitted.get(f"{win['task']}_{win['name']}")
    out["bakeoff_winner"] = {
        "name": win["name"], "task": win["task"],
        "pruned_val_iou": win["pruned_val_iou"],
        "unpruned_val_iou": win["unpruned_val_iou"],
        "val_gain": win["gain"], "cutoff": win["cutoff"],
        "recall_threshold": float(thr_rec),
        "val_recall_at_threshold": float(rec_m["recall"]),
        "test": cb.evaluate_on_test(win["name"], est, win["task"], pt, raw_t,
                                    Mt, Vt, thr_rec, win["cutoff"],
                                    win["higher_is_worse"], thr_feat=thr),
    }
    print(f"[bakeoff] winner {win['name']} ({win['task']}): "
          f"val {win['unpruned_val_iou']:.4f} -> {win['pruned_val_iou']:.4f}, "
          f"test IoU={out['bakeoff_winner']['test']['iou']:.4f}", flush=True)

    # --- 2. free baselines are the control: a learned model that cannot beat
    # |prob - 0.5| has not earned its training time. ---
    learned = [r for r in board if r["task"] in ("trust", "error")]
    free = [r for r in board if r["task"] == "free"]
    best_learned = max(learned, key=lambda r: r["pruned_val_iou"]) if learned else None
    best_free = max(free, key=lambda r: r["pruned_val_iou"]) if free else None
    out["free_baseline_comparison"] = {
        "best_learned": best_learned, "best_free": best_free,
        "learned_minus_free": (float(best_learned["pruned_val_iou"]
                                     - best_free["pruned_val_iou"])
                               if best_learned and best_free else float("nan")),
    }
    if best_learned and best_free:
        print(f"[free] best learned {best_learned['name']} "
              f"{best_learned['pruned_val_iou']:.4f} vs best free "
              f"{best_free['name']} {best_free['pruned_val_iou']:.4f} "
              f"(delta {out['free_baseline_comparison']['learned_minus_free']:+.4f})",
              flush=True)

    # --- 3. over-predict -> FP-prune sweep ---
    print("\n=== over-predict -> FP-prune sweep ===", flush=True)
    sweep = []
    for target in PRUNE_RECALL_TARGETS:
        t, rm = combine.recall_threshold(pv, Mv, Vv, target_recall=target)
        fd = combine.FPDetector().fit(pv, raw_v, Mv, Vv, thr=t)
        sv, st = fd.predict(pv, raw_v, t), fd.predict(pt, raw_t, t)
        fit = combine.fit_prune(pv, Mv, Vv, sv, t, higher_is_worse=True)
        pred = combine.apply_prune(pt, t, st, fit["cutoff"], higher_is_worse=True)
        m = metrics.metrics_from_pred(pred, Mt, Vt)
        unpruned = float(metrics.iou(pt >= t, Mt, Vt))
        sweep.append({"target_recall": float(target), "threshold": float(t),
                      "val_recall": float(rm["recall"]),
                      "val_iou": float(fit["val_iou"]),
                      "unpruned_val_iou": float(fit["unpruned_val_iou"]),
                      "cutoff": fit["cutoff"],
                      "unpruned_test_iou": unpruned, "test": m,
                      "test_gain_vs_unpruned": float(m["iou"] - unpruned)})
        print(f"  R={target:.2f} thr={t:.4f}: unpruned {unpruned:.4f} -> "
              f"PRUNED {m['iou']:.4f} (P={m['precision']:.3f} "
              f"R={m['recall']:.3f})", flush=True)
    out["fp_prune_sweep"] = sweep
    out["best_prune_by_val"] = float(max(sweep, key=lambda e: e["val_iou"])
                                     ["target_recall"])

    # --- 4. cross-slice z-refinement ---
    zr = combine.fit_z_refine(pv, Mv, Vv)
    pt_z = combine.z_smooth(pt, k=zr["k"], mode=zr["mode"])
    zm = metrics.metrics_from_pred(pt_z >= zr["thr"], Mt, Vt)
    out["z_refine"] = {**{k: zr[k] for k in ("mode", "k", "thr", "val_iou")},
                       "test": zm,
                       "test_gain_vs_baseline": float(zm["iou"] - base_iou)}
    print(f"\n[z] {zr['mode']} k={zr['k']}: test IoU={zm['iou']:.4f} "
          f"(delta {out['z_refine']['test_gain_vs_baseline']:+.4f})", flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. Statistical power — the gate on every interpretation.
# ---------------------------------------------------------------------------
def power_report(n_chondrules_val):
    """Print the minimum detectable effect in ABSOLUTE IoU for `n` val chondrules.

    Anchored on the measured bootstrap SE of a val-IoU difference — 0.0071 IoU at
    19 validation chondrules — scaled as SE = 0.0071 * sqrt(19 / n); the MDE is
    2 * SE. Reference rows are printed alongside the actual n.

    Any measured gain smaller than the MDE must be reported as "below the noise
    floor", NOT as a result.
    """
    n = int(n_chondrules_val)
    se = SE_REF_IOU * float(np.sqrt(SE_REF_N / max(n, 1)))
    mde = 2.0 * se
    rows = [{"n_chondrules_val": int(k),
             "se_iou": SE_REF_IOU * float(np.sqrt(SE_REF_N / k)),
             "mde_iou": 2.0 * SE_REF_IOU * float(np.sqrt(SE_REF_N / k))}
            for k in POWER_REFERENCE_N]
    print("=== statistical power (validation chondrules govern it) ===",
          flush=True)
    print(f"  actual n={n}: SE={se:.4f} IoU  MDE={mde:.4f} IoU", flush=True)
    print(f"  {'n':>6} {'SE (IoU)':>10} {'MDE (IoU)':>11}", flush=True)
    for r in rows:
        print(f"  {r['n_chondrules_val']:>6} {r['se_iou']:>10.4f} "
              f"{r['mde_iou']:>11.4f}", flush=True)
    print("  Any gain below the MDE is BELOW THE NOISE FLOOR, not a result.",
          flush=True)
    return {"n_chondrules_val": n, "se_iou": float(se), "mde_iou": float(mde),
            "se_reference_iou": SE_REF_IOU, "se_reference_n": SE_REF_N,
            "reference": rows}


def _gain(value, mde):
    """A reported gain plus its explicit below-the-noise-floor flag."""
    v = float(value)
    return {"gain": v, "below_noise_floor": bool(abs(v) < float(mde))}


# ---------------------------------------------------------------------------
# 6. The whole workflow.
# ---------------------------------------------------------------------------
def run_all(sources, out_dir=OUT_DIR, budgets=None, archs=None,
            leaderboard=False, recipe=None, steps_per_epoch=STEPS_PER_EPOCH,
            hier_levels=HIER_LEVELS):
    """Split -> power -> hierarchy filter -> stage-2 sweep -> confidence sweep.

    Writes `out_dir/pipeline_v2.json` with every number and an explicit
    `below_noise_floor` flag on each reported gain.

    [ADDED 2026-08-08] Everything after `out_dir` exists so this can be proven
    end to end at a tiny budget before real second-meteorite data arrives, which
    the 2026-08-04 handoff listed as the next thing to establish. Defaults
    reproduce the previous behaviour in every respect EXCEPT `leaderboard`,
    which is now False; see `stage2_sweep` for why the old default was a defect.

    `out_dir` is now threaded through the split record, the probability maps and
    the JSON. It previously named only the final JSON while `build_multi_split`,
    `save_prob` and the probability RELOAD all used the module-level `OUT_DIR`,
    so passing a custom `out_dir` wrote the summary to one place and the
    artifacts it describes to another.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    zones = build_multi_split(sources, out_dir=out_dir)
    power = power_report(zones["meta"]["n_chondrules_3d"]["val"])
    mde = power["mde_iou"]

    Rtr, Mtr, Vtr = (zones["train"][k] for k in ("raw", "mask", "valid"))
    Rv, Mv, Vv = (zones["val"][k] for k in ("raw", "mask", "valid"))
    Rt, Mt, Vt = (zones["test"][k] for k in ("raw", "mask", "valid"))

    # --- hierarchy as a FILTER: thresholds fit on VAL, frozen for train/test ---
    print("\n=== hierarchy filter (fit on VAL, frozen for train/test) ===",
          flush=True)
    fit = hierarchy.run_cascade(Rv, Mv, Vv, levels=hier_levels,
                                target_recall=HIER_TARGET_RECALL)
    thr_levels = [lv["threshold"] for lv in fit["levels"]]
    app_tr = hierarchy.run_cascade(Rtr, Mtr, Vtr, thresholds=thr_levels,
                                   levels=hier_levels)
    app_te = hierarchy.run_cascade(Rt, Mt, Vt, thresholds=thr_levels,
                                   levels=hier_levels)
    # Applying the frozen thresholds back to VAL reproduces the fit pass exactly
    # (the drop rule is `feature >= t` with the same t), so it is not re-run.
    alive_masks = {"train": app_tr["final_alive_mask"],
                   "val": fit["final_alive_mask"],
                   "test": app_te["final_alive_mask"]}
    hier_out = {"levels_config": list(hier_levels),
                "target_recall": HIER_TARGET_RECALL,
                "thresholds_from_val": [float(t) for t in thr_levels],
                "val_levels": fit["levels"],
                "train_levels": app_tr["levels"],
                "test_levels": app_te["levels"]}
    for z in ZONES:
        frac = float(alive_masks[z].mean())
        kept = float((zones[z]["mask"] & zones[z]["valid"] & alive_masks[z]).sum()
                     / max(1.0, float((zones[z]["mask"] & zones[z]["valid"]).sum())))
        hier_out[z] = {"frac_alive": frac,
                       "compute_saving": float(1.0 / max(frac, 1e-12)),
                       "chondrule_pixel_retention": kept}
        print(f"  {z:>6} alive={frac:.4f} "
              f"saving={hier_out[z]['compute_saving']:.2f}x "
              f"retention={kept:.4f}", flush=True)

    # --- stage-2 architecture sweep inside the alive mask ---
    print("\n=== stage-2 sweep (inside the alive mask) ===", flush=True)
    ranked = stage2_sweep(zones, alive_masks,
                          budgets=(DEFAULT_BUDGET,) if budgets is None
                          else budgets,
                          archs=archs, out_dir=out_dir,
                          leaderboard=leaderboard, recipe=recipe,
                          steps_per_epoch=steps_per_epoch)
    win = ranked[0]
    print(f"\nWINNER (on VALIDATION): {win['model']} "
          f"val IoU={win['val_iou']:.4f}", flush=True)

    # --- confidence / prune / z-refine on the winner's own probabilities ---
    pv = np.load(os.path.join(out_dir, "probs",
                              f"{win['model']}_val.npy")).astype(np.float32)
    pt = np.load(os.path.join(out_dir, "probs",
                              f"{win['model']}.npy")).astype(np.float32)
    conf = confidence_sweep(pv, Rv, Mv, Vv, pt, Rt, Mt, Vt,
                            float(win["threshold"]))

    base_iou = float(conf["baseline_test"]["iou"])
    gains = {
        "stage2_vs_runner_up": _gain(
            win["val_iou"] - ranked[1]["val_iou"] if len(ranked) > 1 else 0.0,
            mde),
        "bakeoff_winner_val": _gain(conf["bakeoff_winner"]["val_gain"], mde),
        "bakeoff_winner_test": _gain(
            conf["bakeoff_winner"]["test"]["iou"] - base_iou, mde),
        "learned_minus_free_val": _gain(
            conf["free_baseline_comparison"]["learned_minus_free"], mde),
        "z_refine_test": _gain(conf["z_refine"]["test_gain_vs_baseline"], mde),
    }
    for e in conf["fp_prune_sweep"]:
        gains[f"fp_prune_R{e['target_recall']:.2f}_test"] = _gain(
            e["test"]["iou"] - base_iou, mde)

    out = {
        "split": zones["meta"],
        "power": power,
        "hierarchy": hier_out,
        "stage2_ranking": ranked,
        "winner": {"model": win["model"], "selected_on": "validation IoU",
                   "val_iou": win["val_iou"], "threshold": win["threshold"]},
        "confidence_sweep": conf,
        "gains": gains,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = os.path.join(out_dir, "pipeline_v2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n=== reported gains vs the {mde:.4f} IoU noise floor ===", flush=True)
    for k, g in gains.items():
        flag = "BELOW NOISE FLOOR" if g["below_noise_floor"] else "measurable"
        print(f"  {k:32s} {g['gain']:+.4f}  {flag}", flush=True)
    print(f"wrote {path} ({out['runtime_sec']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Self-test: plumbing only — no training, no hierarchy, no meta-models.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p19 = power_report(19)
    assert abs(p19["se_iou"] - 0.0071) < 1e-6, p19
    assert abs(p19["mde_iou"] - 0.0142) < 1e-6, p19
    p76 = power_report(76)
    assert abs(p76["mde_iou"] - 0.0071) < 1e-6, p76

    src = MeteoriteSource("ALH83010", data.RAW_DIR, data.LAB_DIR)
    raw_paths, lab_paths = src.list_pairs()
    assert len(raw_paths) == len(lab_paths) == 531, (len(raw_paths),
                                                     len(lab_paths))
    print(f"\n  MeteoriteSource {src.name}: {len(raw_paths)}/{len(lab_paths)} "
          f"pairs   OK", flush=True)

    z = build_multi_split([src])
    meta = z["meta"]
    for e in meta["meteorites"]:
        tot = sum(e["zone"][k]["n_slices"] for k in ZONES) + e["n_buffer_slices"]
        assert tot == e["N"], (e["name"], tot, e["N"])
    for k in ZONES:
        n = z[k]["raw"].shape[0]
        assert z[k]["mask"].shape[0] == n and z[k]["valid"].shape[0] == n
        assert z[k]["provenance"].shape[0] == n
        assert z[k]["abs_slices"].shape[0] == n
        assert n == meta["zone_stats"][k]["n_slices"], (k, n)
        assert n == sum(e["zone"][k]["n_slices"] for e in meta["meteorites"])

    # No slice may appear in two zones. A slice is identified by (meteorite,
    # absolute z index), so the check survives concatenation across meteorites.
    keys = {k: {(p, int(i)) for p, i in zip(z[k]["provenance"],
                                            z[k]["abs_slices"])}
            for k in ZONES}
    for k in ZONES:
        assert len(keys[k]) == z[k]["raw"].shape[0], k
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (keys[a] & keys[b]), (a, b, sorted(keys[a] & keys[b])[:5])

    print("\n  zone slice counts sum to the band   OK", flush=True)
    print("  no slice index appears in two zones   OK", flush=True)
    print("pipeline_v2 self-test: OK")
