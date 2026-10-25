"""Synovitis-region bags for the synovitis crop expert (syn_train.py); adapted from experiments/v1/oct10_oa/main/oa_data.py.

A study is a set of bags of 3-slice windows, each cropped around one synovitis region found by the CPU localiser v2 (loc/pred:
per-slice presence + box in the side-corrected view; copied from the OA session):
  pouch  suprapatellar recess, all planes (axial is the reliable plane: the sagittal / coronal pouch is often cut at the FOV top)
  hoffa  infrapatellar fat pad, sagittal + axial (+ coronal)
  post   popliteal / posterior region (Baker's, posterior recesses), sagittal + axial
  notch  intercondylar notch (cruciate synovium), all planes (optional)
  knee   the whole 160 mm view, all planes, slices of the joint box (optional global context)
  gutter axial slices of the patella, 110 mm crop centred on it: the medial / lateral parapatellar recesses (optional)
plane_bags=True splits every bag into one bag per plane ("pouch:Axial", ...) so each plane gets k windows (the pouch bag was 49%
sagittal / 35% coronal / 16% axial windows although only the axial pouch is not cut by the FOV). clamp=True shifts a crop that
leaves the 400 px image back inside it (same size: sagittal pouch crops were 21% zero padding in the median). planes (comma list)
keeps only those planes; fs_only keeps the fat-suppressed windows of a bag when it has any (the Codex reads saw only sagittal /
axial FS sheets).
Candidate windows of a bag: slices with presence p > P_MIN (not the first / last slice of the series), trimmed to the central
1 - 2 * trim of that range. The crop is a square of CROP_MM[bag, plane] * crop_scale mm centred on the series-level box, cut from the
0.4 mm/px cache image (zero padding outside), resized to `px` on the CPU. No lateral mirroring (midline regions); side-flipped series
(series_side_final.csv) are mirrored back so every crop is in the side-corrected view. Train: k windows per bag, uniform over the
candidates; eval: k_eval per bag, evenly spread."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.cache import SliceCache
from common.meta import SERIES_TYPES

EXP = Path(__file__).resolve().parents[1]
STRUCT = {"pouch": "suprapatellar", "hoffa": "hoffa", "post": "popliteal", "notch": "notch", "knee": "joint", "gutter": "patella"}
PLANES_OF = {"pouch": ("Sagittal", "Coronal", "Axial"), "hoffa": ("Sagittal", "Coronal", "Axial"), "post": ("Sagittal", "Axial"),
             "notch": ("Sagittal", "Coronal", "Axial"), "knee": ("Sagittal", "Coronal", "Axial"), "gutter": ("Axial",)}
CROP_MM = {"pouch": 80, "hoffa": 72, "post": 100, "notch": 64, "knee": 160, "gutter": 110}
P_MIN = 0.5
TOP_FALLBACK = 3
TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def bag_names(bags, plane_bags=False, planes=""):
    keep = lambda p: not planes or p in planes.split(",")
    return [f"{b}:{p}" for b in bags for p in PLANES_OF[b] if keep(p)] if plane_bags else list(bags)


def build_bags(bags, crop_scale=1.0, trim=0.1, fs_only=False, plane_bags=False, clamp=False, planes=""):
    """study -> bag -> list of (r0, r1, r2 cache rows of the window, hflip, cy, cx, side_px, type, plane, pos, p).
    knee bag: centred on the image centre (the cache is centred on the knee), slices of the localiser joint box."""
    sl = pd.read_parquet(EXP / "loc/pred/slices.parquet",
                         columns=["SeriesInstanceUID", "StudyInstanceUID", "plane", "idx", "row", "pos_mm"] + [f"p_{STRUCT[b]}" for b in bags])
    ser = pd.read_parquet(EXP / "loc/pred/series.parquet")
    cache = SliceCache("fov160_px400_q95")
    meta = cache.series.set_index("SeriesInstanceUID")
    side = pd.read_csv(EXP / "main/series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    px_mm = cache.meta["mm_per_px"]
    out = {}
    for (study, uid), g in sl.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        g = g.sort_values("idx")
        n, plane = len(g), g.plane.iloc[0]
        stype = meta.at[uid, "series_type"]
        if n < 3 or (planes and plane not in planes.split(",")):
            continue
        rows = g.row.to_numpy()
        flip = uid in flips and plane != "Sagittal"
        for bag in bags:
            if plane not in PLANES_OF[bag]:
                continue
            st = STRUCT[bag]
            p = g[f"p_{st}"].to_numpy(np.float32)
            if uid not in ser.index or np.isnan(ser.at[uid, f"{st}_y0"]):
                continue
            if bag == "knee":
                cy = cx = cache.px / 2
            else:
                b = ser.loc[uid, [f"{st}_y0", f"{st}_y1", f"{st}_x0", f"{st}_x1"]].to_numpy(float) * cache.px
                cy, cx = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
                if flip:
                    cx = cache.px - cx                     # box is in the corrected view: raw-image centre (crop mirrored later)
            side_px = CROP_MM[bag] * crop_scale / px_mm
            if clamp:                                      # keep the crop inside the image where it fits (raw coordinates)
                lo, hi = min(side_px / 2, cache.px / 2), max(cache.px - side_px / 2, cache.px / 2)
                cy, cx = float(np.clip(cy, lo, hi)), float(np.clip(cx, lo, hi))
            keep = [i for i in range(1, n - 1) if p[i] > P_MIN]
            if len(keep) > 3:
                cut = int(round(trim * len(keep)))
                keep = keep[cut:len(keep) - cut]
            cands = [(rows[i - 1], rows[i], rows[i + 1], flip, cy, cx, side_px, TYPE_ID[stype], PLANE_ID[plane],
                      float(g.pos_mm.iloc[i]), float(p[i])) for i in keep]
            key = f"{bag}:{plane}" if plane_bags else bag
            out.setdefault(study, {b_: [] for b_ in bag_names(bags, plane_bags, planes)})[key] += cands
    if fs_only:                                            # FS windows only where a bag has any (else keep the non-FS ones)
        for b in out.values():
            for key, c in b.items():
                b[key] = [w for w in c if SERIES_TYPES[w[7]].endswith("_FS")] or c
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
                b[bag] = sorted(donor, key=lambda w: -w[-1])[:TOP_FALLBACK]
        out[s] = b
    return out


class SynDataset(Dataset):
    def __init__(self, studies, bags, bag_names, targets, weights, k, train, px=224, seed=0, n_out=1):
        self.cache = SliceCache("fov160_px400_q95")
        self.studies, self.bags, self.bag_names, self.targets, self.weights = list(studies), bags, bag_names, targets, weights
        self.k, self.train, self.px, self.seed, self.epoch, self.n_out = k, train, px, seed, 0, n_out

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
        if wins:                                             # eval ablations may leave a study without windows
            rows = np.array([w[:3] for w in wins])
            uniq, inv = np.unique(rows.ravel(), return_inverse=True)
            img = self.cache.read(uniq)[inv].reshape(len(wins), 3, self.cache.px, self.cache.px)
            for j, w in enumerate(wins):
                x[j] = self.crop(img[j], w)
        mask = np.zeros(kt, bool)
        mask[:len(wins)] = True
        bag = np.full(kt, -1, np.int64)
        bag[:len(wins)] = bag_id
        typ, plane, pos = np.zeros(kt, np.int64), np.zeros(kt, np.int64), np.zeros(kt, np.float32)
        for j, w in enumerate(wins):
            typ[j], plane[j], pos[j] = w[7], w[8], w[9]
        y = self.targets[i] if self.targets is not None else np.zeros(self.n_out, np.float32)
        wt = self.weights[i] if self.weights is not None else np.ones(self.n_out, np.float32)
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "bag": torch.from_numpy(bag), "type": torch.from_numpy(typ),
                "plane": torch.from_numpy(plane), "pos": torch.from_numpy(pos), "y": torch.from_numpy(np.asarray(y, np.float32)),
                "w": torch.from_numpy(np.asarray(wt, np.float32)), "idx": torch.tensor(i)}
