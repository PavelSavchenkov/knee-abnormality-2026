"""Bone bags for the fracture / contusion expert (bone_train.py); adapted from experiments/v1/oct11_synovitis_crops/main/syn_data.py.

A study is a set of bags of 3-slice windows, each cropped around one bone found by the CPU localiser v3 (data/loc_pred_v3:
per-slice presence + box in the side-corrected view; localiser v2 + nnU-Net femur / tibia boxes, scripts/loc_*.py):
  femur    distal femur: the bottom FEM_H mm of the femur box (condyles, trochlea, metaphysis), all planes
  tibia    proximal tibia: the top TIB_H mm of the tibia box (plateaus, spines, fibular head next to it), all planes
  patella  patella box, all planes
  knee     the whole 160 mm view, slices of the joint box (optional global context)
Crop: square of side max(box height (capped) + 2 * MARGIN, box width + 2 * side margin) mm (side margin TIB_XM for the tibia, which
keeps the fibular head, else MARGIN), clipped to [MIN_MM, MAX_MM] * crop_scale, centred on the (capped) box, cut from the 0.4 mm/px cache image (zero padding outside) and resized to `px` on the CPU. Axial crops take the whole
box. Side-flipped series (series_side_final.csv) are mirrored back, so every crop is in the side-corrected view.
Candidate windows of a bag: slices with presence p > P_MIN (not the first / last slice of the series), trimmed to the central
1 - 2 * trim of that range. series: "all", "fs" (fat-suppressed windows only, where the bag has any) or "nofs".
Train: k windows per bag, uniform over the candidates; eval: k_eval per bag, evenly spread.
pair (oct16_experts 10-18, bpair): every window also carries its partner = the nearest window of the same bag and plane from the other
FS class (|d pos| <= PAIR_MM; FS <-> non-FS), cropped from its own series around its own bone box; the dataset returns the partner
crops (x2) and which windows have one (pm)."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.cache import SliceCache
from common.meta import SERIES_TYPES
CACHE = "fov160_px400_q95"   # 10-18 USER U57: the slice cache the crops are cut from; trainers set it from cfg "cache" (e.g. fov160_px640_q95 =
                              # 0.25 mm/px native-resolution copy with the SAME row plan; crop boxes scale by cache.px / mm_per_px)

EXP = Path(__file__).resolve().parents[1]
STRUCT = {"femur": "femur", "tibia": "tibia", "patella": "patella", "knee": "joint"}
PLANES = ("Sagittal", "Coronal", "Axial")
FEM_H, TIB_H, MARGIN, TIB_XM, MIN_MM, MAX_MM = 72.0, 64.0, 8.0, 16.0, 48.0, 120.0   # TIB_XM: side margin of tibia crops (fibular head)
P_MIN = 0.5
TOP_FALLBACK = 3
PAIR_MM = 2.0   # sagittal FS / non-FS series: same range, ~3.3 mm spacing, usually the same start (10-18 check: 4,009 studies)
TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def crop_box(bag, plane, b, px_mm, crop_scale, pat_margin=MARGIN):
    """Series box b = (y0, y1, x0, x1) in cache px -> (cy, cx, side_px) of the square crop. pat_margin: margin (mm) of patella crops
    (poles and tendon insertions)."""
    y0, y1, x0, x1 = b
    m = (pat_margin if bag == "patella" else MARGIN) / px_mm
    xm = (TIB_XM if bag == "tibia" else pat_margin if bag == "patella" else MARGIN) / px_mm
    if plane != "Axial" and bag == "femur":
        y0 = max(y0, y1 - FEM_H / px_mm)
    elif plane != "Axial" and bag == "tibia":
        y1 = min(y1, y0 + TIB_H / px_mm)
    side = float(np.clip(max(y1 - y0 + 2 * m, x1 - x0 + 2 * xm), MIN_MM / px_mm, MAX_MM / px_mm)) * crop_scale
    return (y0 + y1) / 2, (x0 + x1) / 2, side


def build_bags(bags, crop_scale=1.0, trim=0.1, series="all", planes="", loc="data/loc_pred_v3", pat_margin=MARGIN, pair=False):
    """study -> bag -> list of (r0, r1, r2 cache rows of the window, hflip, cy, cx, side_px, type, plane, pos, p[, partner])."""
    sl = pd.read_parquet(EXP / loc / "slices.parquet",
                         columns=["SeriesInstanceUID", "StudyInstanceUID", "plane", "idx", "row", "pos_mm"] + [f"p_{STRUCT[b]}" for b in bags])
    ser = pd.read_parquet(EXP / loc / "series.parquet")
    cache = SliceCache(CACHE)
    meta = cache.series.set_index("SeriesInstanceUID")
    side = pd.read_csv(Path(__file__).resolve().parent / "series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    px_mm = cache.meta["mm_per_px"]
    keep_planes = planes.split(",") if planes else PLANES
    out = {}
    for (study, uid), g in sl.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        g = g.sort_values("idx")
        n, plane = len(g), g.plane.iloc[0]
        stype = meta.at[uid, "series_type"]
        if n < 3 or plane not in keep_planes:
            continue
        rows = g.row.to_numpy()
        flip = uid in flips and plane != "Sagittal"
        for bag in bags:
            st = STRUCT[bag]
            p = g[f"p_{st}"].to_numpy(np.float32)
            if uid not in ser.index or np.isnan(ser.at[uid, f"{st}_y0"]):
                continue
            if bag == "knee":
                cy = cx = cache.px / 2
                side_px = 160 / px_mm
            else:
                b = ser.loc[uid, [f"{st}_y0", f"{st}_y1", f"{st}_x0", f"{st}_x1"]].to_numpy(float) * cache.px
                cy, cx, side_px = crop_box(bag, plane, b, px_mm, crop_scale, pat_margin)
                if flip:
                    cx = cache.px - cx                     # box is in the corrected view: raw-image centre (crop mirrored later)
            keep = [i for i in range(1, n - 1) if p[i] > P_MIN]
            if len(keep) > 3:
                cut = int(round(trim * len(keep)))
                keep = keep[cut:len(keep) - cut]
            cands = [(rows[i - 1], rows[i], rows[i + 1], flip, cy, cx, side_px, TYPE_ID[stype], PLANE_ID[plane],
                      float(g.pos_mm.iloc[i]), float(p[i])) for i in keep]
            out.setdefault(study, {b_: [] for b_ in bags})[bag] += cands
    if pair:                                               # partner = (r0, r1, r2, hflip, cy, cx, side) of the other FS class, or None
        for b in out.values():
            for key, c in b.items():
                fs = np.array([SERIES_TYPES[w[7]].endswith("_FS") for w in c])
                pl, ps = np.array([w[8] for w in c]), np.array([w[9] for w in c])
                new = []
                for j, w in enumerate(c):
                    cand = np.flatnonzero((pl == w[8]) & (fs != fs[j]))
                    d = np.abs(ps[cand] - w[9])
                    new.append(w + (c[cand[d.argmin()]][:7] if len(cand) and d.min() <= PAIR_MM else None,))
                b[key] = new
    if series != "all":                                    # FS (or non-FS) windows only where a bag has any
        want = series == "fs"
        for b in out.values():
            for key, c in b.items():
                b[key] = [w for w in c if SERIES_TYPES[w[7]].endswith("_FS") == want] or c
    return out


def finalize(bags):
    """An empty bag borrows the TOP_FALLBACK highest-presence windows of the study's other bags; studies with all bags empty are
    dropped."""
    out = {}
    for s, b in bags.items():
        if not any(b.values()):
            continue
        for bag in b:
            if not b[bag]:
                donor = [x for k in b if k != bag for x in b[k]]
                b[bag] = sorted(donor, key=lambda w: -w[10])[:TOP_FALLBACK]
        out[s] = b
    return out


class BoneDataset(Dataset):
    def __init__(self, studies, bags, bag_names, targets, weights, k, train, px=256, seed=0, n_out=1, pair=False):
        self.cache = SliceCache(CACHE)
        self.studies, self.bags, self.bag_names, self.targets, self.weights = list(studies), bags, bag_names, targets, weights
        self.k, self.train, self.px, self.seed, self.epoch, self.n_out, self.pair = k, train, px, seed, 0, n_out, pair

    def __len__(self):
        return len(self.studies)

    def choose(self, cands, rng):
        if self.train:
            return [cands[i] for i in rng.choice(len(cands), self.k, replace=len(cands) < self.k)]
        return [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, min(self.k, len(cands))).round().astype(int))]

    def crop(self, img, w):
        """Square crop centred at (cy, cx) of the raw cached image, mirrored afterwards when hflip."""
        _, _, _, hflip, cy, cx, side, *_ = w
        h = int(round(side))
        y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
        out = np.zeros((3, h, h), np.uint8)
        sy, sx = max(y0, 0), max(x0, 0)
        ey, ex = min(y0 + h, img.shape[1]), min(x0 + h, img.shape[2])
        if ey > sy and ex > sx:
            out[:, sy - y0:ey - y0, sx - x0:ex - x0] = img[:, sy:ey, sx:ex]
        if hflip:
            out = out[:, :, ::-1]
        if h == self.px:
            return np.ascontiguousarray(out)
        return np.stack([cv2.resize(np.ascontiguousarray(c), (self.px, self.px), interpolation=cv2.INTER_AREA if h > self.px else cv2.INTER_LINEAR)
                         for c in out])

    def __getitem__(self, i):
        self.epoch, i = divmod(i, len(self.studies))         # EpochSampler passes epoch * n + i (persistent workers)
        sid = self.studies[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        wins, bag_id = [], []
        for j, bag in enumerate(self.bag_names):
            ws = self.choose(self.bags[sid][bag], rng)
            wins += ws
            bag_id += [j] * len(ws)
        kt = self.k * len(self.bag_names)
        x = np.zeros((kt, 3, self.px, self.px), np.uint8)
        x2, pm = (np.zeros_like(x), np.zeros(kt, bool)) if self.pair else (None, None)
        if wins:
            pw = [(j, w[11]) for j, w in enumerate(wins) if w[11] is not None] if self.pair else []
            rows = np.array([w[:3] for w in wins] + [q[:3] for _, q in pw])
            uniq, inv = np.unique(rows.ravel(), return_inverse=True)
            img = self.cache.read(uniq)[inv].reshape(len(rows), 3, self.cache.px, self.cache.px)
            for j, w in enumerate(wins):
                x[j] = self.crop(img[j], w)
            for i_, (j, q) in enumerate(pw):
                x2[j], pm[j] = self.crop(img[len(wins) + i_], q), True
        mask = np.zeros(kt, bool)
        mask[:len(wins)] = True
        bag = np.full(kt, -1, np.int64)
        bag[:len(wins)] = bag_id
        typ, plane, pos = np.zeros(kt, np.int64), np.zeros(kt, np.int64), np.zeros(kt, np.float32)
        for j, w in enumerate(wins):
            typ[j], plane[j], pos[j] = w[7], w[8], w[9]
        y = self.targets[i] if self.targets is not None else np.zeros(self.n_out, np.float32)
        wt = self.weights[i] if self.weights is not None else np.ones(self.n_out, np.float32)
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "bag": torch.from_numpy(bag), "type": torch.from_numpy(typ),
               "plane": torch.from_numpy(plane), "pos": torch.from_numpy(pos), "y": torch.from_numpy(np.asarray(y, np.float32)),
               "w": torch.from_numpy(np.asarray(wt, np.float32)), "idx": torch.tensor(i)}
        return out | ({"x2": torch.from_numpy(x2), "pm": torch.from_numpy(pm)} if self.pair else {})
