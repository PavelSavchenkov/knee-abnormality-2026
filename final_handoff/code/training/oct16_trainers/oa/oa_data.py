"""Compartment bags for the OA expert (oa_train.py).

A study is three bags of 3-slice windows, each cropped around one compartment found by the CPU localiser (scripts/loc_infer.py
-> data/loc_pred: per slice presence + box in the side-corrected view):
  medial  comp_medial, all planes
  lateral comp_lateral, all planes; coronal / axial crops are mirrored left-right so they look like medial crops (shared features)
  pf      pf_joint, sagittal + axial
Candidate windows of a bag: slices with presence p > P_MIN (not the first / last slice of the series), trimmed to the central
1 - 2 * TRIM of that slice range (edge slices show soft tissue, the notch or anterior coronal anatomy: loc_viz sheets); a bag with none falls back
to the TOP_FALLBACK slices of the highest p over all series. The crop is a square of CROP_MM[bag, plane] mm centred on the
series-level box (presence-weighted mean over its present slices), cut from the side-corrected 0.4 mm/px cache image (zero
padding outside), resized to `px` on the CPU. Train: k windows per bag, uniform over the candidates; eval: k_eval per bag,
evenly spread."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.cache import SliceCache
from common.meta import SERIES_TYPES
CACHE = "fov160_px400_q95"   # 10-18 USER U57: the slice cache the crops are cut from; oa_train.bags_for sets it from cfg "cache"
                              # (fov160_px640_q95 = the 0.25 mm/px native copy, same row plan; boxes scale by cache.px / mm_per_px)

EXP = Path(__file__).resolve().parents[1]
BAGS = ["medial", "lateral", "pf"]
STRUCT = {"medial": "comp_medial", "lateral": "comp_lateral", "pf": "pf_joint"}
STRUCT_V2 = {"medial": "med_tib_cart", "lateral": "lat_tib_cart", "pf": "pat_cart"}   # localiser v2: cartilage-centred bags
PLANES_OF = {"medial": ("Sagittal", "Coronal", "Axial"), "lateral": ("Sagittal", "Coronal", "Axial"), "pf": ("Sagittal", "Axial")}
CROP_MM = {("medial", "Sagittal"): 72, ("medial", "Coronal"): 56, ("medial", "Axial"): 56,
           ("lateral", "Sagittal"): 72, ("lateral", "Coronal"): 56, ("lateral", "Axial"): 56,
           ("pf", "Sagittal"): 64, ("pf", "Axial"): 64}
P_MIN = 0.5
TRIM = 0.15
REL_TARGET = {"medial": "med_tib_cart", "lateral": "lat_tib_cart", "pf": "pat_cart"}
# oct16_experts U47 oa_kh (10-18): an optional whole-knee bag (160 mm on the joint box, all planes) for a knee-view head on the shared backbone;
# used only when the trainer appends "knee" to BAGS (cfg knee_head), so these entries are inert otherwise
STRUCT["knee"] = STRUCT_V2["knee"] = REL_TARGET["knee"] = "joint"
PLANES_OF["knee"] = ("Sagittal", "Coronal", "Axial")
CROP_MM.update({("knee", p): 160 for p in PLANES_OF["knee"]})
TOP_FALLBACK = 3
TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def build_bags(crop_scale=1.0, loc_version="v1", trim=TRIM, tf_axial=True, series="all", loc_dir=None, cache_name=None, type_map=None,
               series_types=None):
    """study -> bag -> list of (r0, r1, r2 cache rows of the window in corrected order, hflip, cy, cx, side_px, type, plane, pos,
    rel_target, rel_fem, p): rel_* = localiser v2 presence of the bag's target cartilage (medial / lateral tibial, patellar) and of
    femoral cartilage on the centre slice (a per-window relevance input); p = presence of the bag structure (last, used for sorting).
    tf_axial=False drops axial windows from the TF bags (A60: 72% of them show no target tibial cartilage).
    hflip = mirror the raw cached image (side correction XOR lateral mirroring of coronal / axial).
    loc_dir / cache_name / type_map (10-18 OAI, USER 'ok, i want to try external data'): an EXTERNAL source's localiser output (same v2 format,
    rows of ITS cache), its slice cache (an absolute path works in SliceCache) and a series_type remap (OAI's water-excitation DESS is fluid
    bright but our TR / TE rule calls it T1: 'SAG_T1_FS:SAG_PD_FS' etc.); defaults = the competition data, unchanged."""
    d = Path(loc_dir) if loc_dir else EXP / ("data/loc_pred" if loc_version == "v1" else "data/loc_pred_v2")
    struct_of = STRUCT_V2 if loc_version == "v2" else STRUCT                     # slice selection (presence)
    centre_of = STRUCT if loc_version == "v1" else STRUCT_V2                      # v2c: v1-style slices, cartilage centres
    # v2cor (A71b): coronal TF bags select slices by tibial-cartilage presence (lateral coronal bags held ~10 of 20 windows
    # outside the lateral tibial cartilage range vs ~8 medial); sagittal / axial / PF as v1
    cor_v2 = loc_version == "v2cor"
    sl = pd.read_parquet(d / "slices.parquet")
    relp = pd.read_parquet((Path(loc_dir) if loc_dir else EXP / "data/loc_pred_v2") / "slices.parquet", columns=["row"] + [f"p_{s}" for s in REL_TARGET.values()] + ["p_fem_cart"]
                           ).set_index("row").astype(np.float32)
    ser = pd.read_parquet(d / "series.parquet")
    cache = SliceCache(cache_name or CACHE)
    meta = cache.series.set_index("SeriesInstanceUID")
    type_map = type_map or {}
    side = pd.read_csv(Path(__file__).resolve().parent / "series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    px_mm = cache.meta["mm_per_px"]
    out = {}
    for (study, uid), g in sl.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        g = g.sort_values("idx")
        n, plane = len(g), g.plane.iloc[0]
        if n < 3:
            continue
        if series_types is not None and meta.at[uid, "series_type"] not in series_types:
            continue
        rows = g.row.to_numpy()
        typ = TYPE_ID[type_map.get(meta.at[uid, "series_type"], meta.at[uid, "series_type"])]
        flip_corr = uid in flips and plane != "Sagittal"
        rv = relp.reindex(rows)
        for bag in BAGS:
            if plane not in PLANES_OF[bag] or (not tf_axial and bag != "pf" and plane == "Axial"):
                continue
            rt, rf = rv[f"p_{REL_TARGET[bag]}"].to_numpy(), rv["p_fem_cart"].to_numpy()
            st = STRUCT_V2[bag] if cor_v2 and plane == "Coronal" and bag != "pf" else struct_of[bag]
            p = g[f"p_{st}"].to_numpy(np.float32)
            if uid not in ser.index or np.isnan(ser.at[uid, f"{st}_y0"]):
                cands = []
            else:
                ct = st if cor_v2 else (centre_of[bag] if not np.isnan(ser.at[uid, f"{centre_of[bag]}_y0"]) else st)
                b = ser.loc[uid, [f"{ct}_y0", f"{ct}_y1", f"{ct}_x0", f"{ct}_x1"]].to_numpy(float) * cache.px
                cy, cx = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
                side_px = CROP_MM[bag, plane] * crop_scale / px_mm
                mirror = bag == "lateral" and plane != "Sagittal"
                if flip_corr:
                    cx = cache.px - cx                     # box is in the corrected view: raw-image centre (crop mirrored later)
                keep = [i for i in range(1, n - 1) if p[i] > P_MIN]
                if len(keep) > 3:
                    cut = int(round(trim * len(keep)))
                    keep = keep[cut:len(keep) - cut]
                cands = [(rows[i - 1], rows[i], rows[i + 1], flip_corr ^ mirror, cy, cx, side_px, typ, PLANE_ID[plane],
                          float(g.pos_mm.iloc[i]), float(rt[i]), float(rf[i]), float(p[i])) for i in keep]
            out.setdefault(study, {b_: [] for b_ in BAGS})[bag] += cands
    if series != "all":                                    # oct16_experts C63 (U35 lesson): FS (or non-FS) windows only where a bag has any
        want = series == "fs"
        for b in out.values():
            for key, c in b.items():
                b[key] = [w for w in c if SERIES_TYPES[w[7]].endswith("_FS") == want] or c
    return out


def finalize(bags):
    """Fallback for empty bags (TOP_FALLBACK highest-p windows of the other planes are not available here, so an empty bag
    borrows the study's medial / lateral bag mirror); returns study -> bag -> list. Studies with all bags empty are dropped."""
    out = {}
    for s, b in bags.items():
        if not any(b.values()):
            continue
        for bag in BAGS:
            if not b[bag]:
                donor = [x for k in BAGS if k != bag for x in b[k]]
                b[bag] = sorted(donor, key=lambda w: -w[-1])[:TOP_FALLBACK]
        out[s] = b
    return out


class CompartmentDataset(Dataset):
    dil = 0.0      # 10-18 (Brainstorm request 17 fallback; public D1, ported from syn_data S-D): train-time window DILATION - with this
                   # probability per window read slices (i-2, i, i+2) instead of (i-1, i, i+1), clipped to the series
    jit = 0.0      # public D2: train-time slice JITTER - with this probability per window shift the whole window by -1 / +1 slice
                   # (clipped to the series). Both off by default (fold / eval behaviour unchanged).
    seq_L = 0      # F1 (Brainstorm FRESH_IDEAS 10-19): per bag, runs of seq_L CONSECUTIVE windows of one series (ordered by slice), around the
                   # bag's highest-localiser-probability window of that series; k // seq_L runs per bag (series by p, eval; random, train).
    seq_shuffle = False   # F1 null: the same windows with the order inside each run shuffled (train and eval)
    seg_dir = None        # U91 aux_seg: dir of nnU-Net label volumes {series uid}.npz 'lab' (n, H, W) in that series' cache grid -> item 'seg'
    seg_px = 56           # label crop size (ids, nearest); 255 = no label volume for the window (competition cache only)

    def __init__(self, studies, bags, targets, weights, k, train, px=224, seed=0, strat=False, keep=None, ext=None):
        """keep (per-class experts, user 10-16): bag names windows are drawn from (None = all three); k windows per kept bag, or a
        dict bag -> k (G20: per-bag window budget inside the shared expert). ext (10-18 OAI) = (cache name, studies): those studies'
        windows are rows of that second cache (None = every study reads the competition cache, unchanged)."""
        self.keep = list(BAGS) if keep is None else list(keep)
        assert set(self.keep) <= set(BAGS) and self.keep, self.keep
        self.caches = [SliceCache(CACHE)] + ([SliceCache(ext[0])] if ext else [])
        self.ext = set(ext[1]) if ext else set()
        self.cache = self.caches[0]
        self.bounds = []
        for c in self.caches:
            sr = c.series.sort_values("start")
            self.bounds.append((sr.start.to_numpy(), (sr.start + sr.n - 1).to_numpy()))
        self.ser_start, self.ser_end = self.bounds[0]
        self.ser_uid = self.caches[0].series.sort_values("start").SeriesInstanceUID.to_numpy()   # aligned with ser_start (U91 label lookup)
        self.studies, self.bags, self.targets, self.weights = list(studies), bags, targets, weights
        self.k, self.train, self.px, self.seed, self.epoch, self.strat = k, train, px, seed, 0, strat

    def __len__(self):
        return len(self.studies)

    def kb(self, bag):
        return self.k[bag] if isinstance(self.k, dict) else self.k

    def choose(self, cands, rng, k):
        if self.train and self.strat and len(cands) > k:   # every series (type, plane) contributes before any repeats
            groups = {}
            for i, w in enumerate(cands):
                groups.setdefault((w[7], w[8]), []).append(i)
            pools = [list(rng.permutation(g)) for g in groups.values()]
            out = []
            while len(out) < k:
                for j in rng.permutation(len(pools)):
                    if pools[j] and len(out) < k:
                        out.append(pools[j].pop())
            return [cands[i] for i in out]
        if self.train:
            return [cands[i] for i in rng.choice(len(cands), k, replace=len(cands) < k)]
        return [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, min(k, len(cands))).round().astype(int))]

    def choose_runs(self, cands, rng, k, i):
        """F1: (windows, valid) for one bag: k // seq_L runs of seq_L ordered consecutive windows (padding slots repeat a window, valid False)."""
        L = self.seq_L
        n_runs = max(1, k // L)
        if not cands:
            return [], []
        centres = np.array([w[1] for w in cands])
        ser = np.searchsorted(self.ser_start, centres, side="right") - 1          # series of each window (competition cache)
        groups = {}
        for j, s in enumerate(ser):
            groups.setdefault(int(s), []).append(j)
        gl = list(groups.values())
        order = list(rng.permutation(len(gl))) if self.train else sorted(range(len(gl)), key=lambda g: -max(cands[j][-1] for j in gl[g]))
        out, valid = [], []
        for g in order[:n_runs]:
            idx = sorted(gl[g], key=lambda j: centres[j])
            best = max(range(len(idx)), key=lambda t: cands[idx[t]][-1])          # the compartment centre in this series
            if self.train:
                best = int(np.clip(best + rng.integers(-2, 3), 0, len(idx) - 1))
            lo = max(0, min(best - L // 2, len(idx) - L))
            run = idx[lo:lo + L]
            if self.seq_shuffle:
                run = list((rng if self.train else np.random.default_rng((self.seed, i, g))).permutation(run))
            out += [cands[j] for j in run] + [cands[run[-1]]] * (L - len(run))
            valid += [True] * len(run) + [False] * (L - len(run))
        while len(out) < n_runs * L:                                             # fewer series than runs: padded runs
            out += [out[-1]] * L
            valid += [False] * L
        return out, valid

    def crop(self, img, w):
        """Square crop centred at (cy, cx) of the raw cached image (cx already in raw coordinates), mirrored afterwards when
        hflip: mirror(crop_raw(px - cx)) = crop of the side-corrected image at cx."""
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
        return np.stack([cv2.resize(np.ascontiguousarray(c), (self.px, self.px), interpolation=cv2.INTER_LINEAR) for c in out])

    def seg_crop(self, lab, w):
        """The label slice cropped with the SAME geometry as crop() (box, mirroring), resized with nearest to seg_px (ids, 0 outside)."""
        _, _, _, hflip, cy, cx, side, *_ = w
        h = int(round(side))
        y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
        out = np.zeros((h, h), np.uint8)
        sy, sx = max(y0, 0), max(x0, 0)
        ey, ex = min(y0 + h, lab.shape[0]), min(x0 + h, lab.shape[1])
        if ey > sy and ex > sx:
            out[sy - y0:ey - y0, sx - x0:ex - x0] = lab[sy:ey, sx:ex]
        if hflip:
            out = out[:, ::-1]
        return cv2.resize(np.ascontiguousarray(out), (self.seg_px, self.seg_px), interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, i):
        self.epoch, i = divmod(i, len(self.studies))         # EpochSampler passes epoch * n + i (persistent workers)
        sid = self.studies[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        wins, bag_id, valid = [], [], []
        for j, bag in enumerate(BAGS):
            if self.seq_L:
                assert not self.ext, "seq_L: competition cache only"
                ws, vs = self.choose_runs(self.bags[sid][bag], rng, self.kb(bag), i) if bag in self.keep else ([], [])
            else:
                ws = self.choose(self.bags[sid][bag], rng, self.kb(bag)) if bag in self.keep else []
                vs = [True] * len(ws)
            wins += ws
            bag_id += [j] * len(ws)
            valid += vs
        kt = sum((max(1, self.kb(b) // self.seq_L) * self.seq_L) if self.seq_L else self.kb(b) for b in self.keep)
        rows = np.array([w[:3] for w in wins])
        ci = 1 if sid in self.ext else 0
        cache = self.caches[ci]
        if self.train and (self.dil > 0 or self.jit > 0) and len(rows):
            ss, se = self.bounds[ci]
            k_ = np.searchsorted(ss, rows[:, 1], side="right") - 1
            lo, hi = ss[k_], se[k_]
            if self.jit > 0:                                 # D2: shift the whole window by one slice
                sh = (rng.random(len(rows)) < self.jit) * rng.choice(np.array([-1, 1]), len(rows))
                rows = np.clip(rows + sh[:, None], lo[:, None], hi[:, None])
            if self.dil > 0:                                 # D1: (i-2, i, i+2)
                d = rng.random(len(rows)) < self.dil
                rows[d, 0] = np.maximum(rows[d, 0] - 1, lo[d])
                rows[d, 2] = np.minimum(rows[d, 2] + 1, hi[d])
        uniq, inv = np.unique(rows.ravel(), return_inverse=True)
        img = cache.read(uniq)[inv].reshape(len(wins), 3, cache.px, cache.px)
        x = np.zeros((kt, 3, self.px, self.px), np.uint8)
        for j, w in enumerate(wins):
            x[j] = self.crop(img[j], w)
        mask = np.zeros(kt, bool)
        mask[:len(wins)] = valid
        bag = np.full(kt, -1, np.int64)
        bag[:len(wins)] = bag_id
        typ, plane, pos = np.zeros(kt, np.int64), np.zeros(kt, np.int64), np.zeros(kt, np.float32)
        rel = np.zeros((kt, 2), np.float32)
        for j, w in enumerate(wins):
            typ[j], plane[j], pos[j] = w[7], w[8], w[9]
            rel[j] = w[10], w[11]
        y = self.targets[i] if self.targets is not None else np.zeros(5, np.float32)
        wt = self.weights[i] if self.weights is not None else np.ones(5, np.float32)
        extra = {}
        if self.seg_dir is not None and self.train:          # U91: label crops of the windows whose series has a label volume (train only)
            seg, vols = np.full((kt, self.seg_px, self.seg_px), 255, np.uint8), {}
            for j, w in enumerate(wins):
                if not valid[j] or ci != 0:
                    continue
                r = int(w[1])
                k_ = int(np.searchsorted(self.ser_start, r, side="right")) - 1
                f = Path(self.seg_dir) / f"{self.ser_uid[k_]}.npz"
                if k_ not in vols:
                    vols[k_] = np.load(f)["lab"] if f.exists() else None
                if vols[k_] is None:
                    continue
                assert len(vols[k_]) == self.ser_end[k_] - self.ser_start[k_] + 1, (f, len(vols[k_]))
                seg[j] = self.seg_crop(vols[k_][r - self.ser_start[k_]], w)
            extra["seg"] = torch.from_numpy(seg)
        return extra | {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "bag": torch.from_numpy(bag), "type": torch.from_numpy(typ),
                "plane": torch.from_numpy(plane), "pos": torch.from_numpy(pos), "rel": torch.from_numpy(np.nan_to_num(rel)), "y": torch.from_numpy(np.asarray(y, np.float32)),
                "w": torch.from_numpy(np.asarray(wt, np.float32)), "idx": torch.tensor(i)}
