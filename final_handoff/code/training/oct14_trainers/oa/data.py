"""Study dataset over the canonical slice cache (common/cache.py).

A study is a bag of 3-slice windows (centre slice +- 1 neighbour of one series). Train: K windows from the valid (series, centre)
pairs of the study, by sample = "random" (uniform over pairs, so long series get more windows), "grid" (one random pair in each of
K equal strata of the ordered pair list: every series and region covered) or "series" (equal number per series, stratified within
each series); eval: K_eval windows spread evenly over the same pairs. With multi-contrast windows the third slice of an FS
series' window is the co-registered non-FS slice (build_index partners). Each window carries its series
type id, plane id and the centre slice position in mm (canonical direction, relative to the series centre).
Pixels leave the worker as uint8, cropped to crop_mm around the image centre; resize, normalisation and augmentation run on GPU."""
import math

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from common.cache import SliceCache
from common.meta import SERIES_TYPES, PLANES
from common.paths import FINDINGS

TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
PLANE_ID = {p: i for i, p in enumerate(PLANES)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


SIDE_TABLE = Path(__file__).resolve().parent / "series_side_final.csv"   # copied from data/build/side_check/out (image-labelling session)
PARTNERS = Path(__file__).resolve().parent / "partners.csv"                # partners.py


def side_flips():
    """SeriesInstanceUIDs whose cached orientation is mirrored (knee side wrong): a mirror / lateral-side detector trained on
    tag- and geometry-consistent series (data/build/side_check/README.md). 387 series in 100 studies, 2 of them gold."""
    t = pd.read_csv(SIDE_TABLE, index_col=0)
    return set(t.index[t.flip_final])


def true_other_series():
    """Series of the opposite knee in bilateral exams: corrected side differs from the study's majority corrected side and the image
    centre is >= 40 mm off-centre (confident geometry). 40 series in 11 studies."""
    t = pd.read_csv(SIDE_TABLE, index_col=0)
    major = t.groupby("StudyInstanceUID").side_final.agg(lambda v: v.value_counts().index[0])
    return set(t.index[(t.side_final != major.reindex(t.StudyInstanceUID).to_numpy()) & (t.x.abs() >= 40)])


STRUCTURES = {"ACL": ["notch"], "MCL": ["mcl"], "Medial Meniscus": ["med_men", "comp_medial"], "Lateral Meniscus": ["lat_men", "comp_lateral"],
              "Medial OA": ["comp_medial", "med_tib_cart"], "Lateral OA": ["comp_lateral", "lat_tib_cart"],
              "PF OA": ["pf_joint", "patella", "pat_cart"], "Effusion": ["suprapatellar"], "Synovitis": ["suprapatellar", "hoffa"],
              "Baker's": ["popliteal"]}                      # contusion / fracture: bone boxes cover most windows -> no prior


def structure_coverage(path, n_rows):
    """(n_rows, 12) float16 per cache row and finding: coverage of the finding's structures. Source: nnU-Net Knee masks of one
    sagittal FS series per study, boxes projected into every series by patient coordinates (Ideas session, work/knee_seg/
    project_boxes.py -> slice_boxes.parquet; frac = share of the structure's 3D box within half a slab of the slice). frac is
    divided by its max over the series' slices (peak slice = 1), then max over the finding's structures; 0 = no box (contusion /
    fracture, studies without a sagittal FS series)."""
    d = pd.read_parquet(path, columns=["SeriesInstanceUID", "row", "structure", "frac"])
    d["c"] = d.frac / d.groupby(["SeriesInstanceUID", "structure"]).frac.transform("max")
    cov = np.zeros((n_rows, len(FINDINGS)), np.float16)
    for f, structs in STRUCTURES.items():
        g = d[d.structure.isin(structs)].groupby("row").c.max()
        cov[g.index.to_numpy(), FINDINGS.index(f)] = g.to_numpy()
    return cov


def contrast_partners():
    """FS series -> co-registered non-FS partner series of the same plane, with the offsets that make cache positions absolute
    (partners.py -> partners.csv)."""
    return pd.read_csv(PARTNERS, index_col=0)


def build_index(cache, span=(0.05, 0.95), drop_other_knee=False, flips=frozenset(), drop=frozenset(), partners=None):
    """study -> list of dicts: rows (cache rows of the series), pos (mm), type, plane, centres (valid window centres), hflip, prow.
    Series in flips get the medial-lateral axis corrected: sagittal slice order reversed and positions negated; coronal / axial
    images flipped left-right when read (hflip). partners (contrast_partners()): an FS series with a kept partner gets prow = the
    partner's cache row nearest in absolute position for each slice (-1 if > 1.5 mm away); its windows then read
    [slice c-1, slice c, partner slice] (multi-contrast windows, pixel-aligned: co-registered series share the image centre)."""
    assert partners is None or not flips, "multi-contrast windows with side flips are not implemented"
    s = cache.series[cache.series.dup_of == ""]
    if drop_other_knee:
        s = s[~s.other_knee]
    s = s[~s.SeriesInstanceUID.isin(drop)]
    pos_all = cache.slices.sort_values("row").pos_mm.to_numpy(np.float32)
    kept = s.set_index("SeriesInstanceUID")
    index = {}
    for r in s.itertuples():
        n = r.n
        if n < 3:
            continue
        lo, hi = max(1, int(np.floor(span[0] * n))), min(n - 2, int(np.ceil(span[1] * n)) - 1)
        if hi < lo:
            lo, hi = 1, n - 2
        rows, pos = r.start + np.arange(n, dtype=np.int64), pos_all[r.start:r.start + n]
        flip = r.SeriesInstanceUID in flips
        if flip and r.plane == "Sagittal":
            rows, pos = rows[::-1].copy(), -pos[::-1]
        prow = None
        if partners is not None and r.SeriesInstanceUID in partners.index and partners.at[r.SeriesInstanceUID, "partner"] in kept.index:
            q = partners.loc[r.SeriesInstanceUID]
            b = kept.loc[q.partner]
            pa, pb = pos + q.offset, pos_all[b.start:b.start + b.n] + q.partner_offset
            j = np.abs(pa[:, None] - pb[None]).argmin(1)
            prow = np.where(np.abs(pa - pb[j]) <= 1.5, b.start + j, -1)
        index.setdefault(r.StudyInstanceUID, []).append({
            "rows": rows, "pos": pos, "type": TYPE_ID[r.series_type], "plane": PLANE_ID[r.plane],
            "centres": np.arange(lo, hi + 1), "hflip": flip and r.plane != "Sagittal", "prow": prow})
    return index


def crop_window(img, w, px):
    """Compartment crop (oa_data.CompartmentDataset.crop) of a window tuple (r0, r1, r2, hflip, cy, cx, side, ...), resized to px."""
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
    return np.stack([cv2.resize(np.ascontiguousarray(c), (px, px), interpolation=cv2.INTER_LINEAR) for c in out])


class StudyDataset(Dataset):
    def __init__(self, cache_name, studies, targets, weights, index, k, train, crop_mm, seed=0, sample="random", cov=None, crops=None,
                 k_crops=0):
        """crops (user 16:45, unified model): oa_data compartment bags {study: {bag: windows}}; k_crops compartment crops (k_crops // 3
        per bag) are appended to the k whole-knee windows, resized to the same crop px, with kind 1 / 2 / 3 = medial / lateral / pf
        (whole-knee windows: kind 0)."""
        self.cache = SliceCache(cache_name)
        self.studies, self.targets, self.weights, self.index, self.cov = list(studies), targets, weights, index, cov
        self.crops, self.k_crops = crops, k_crops if crops is not None else 0
        self.k, self.train, self.seed, self.sample = k, train, seed, sample
        mm_per_px = self.cache.meta["mm_per_px"]
        self.crop = min(self.cache.px, int(round(crop_mm / mm_per_px)))
        self.epoch = 0

    def __len__(self):
        return len(self.studies)

    def pairs(self, sid):
        ser = self.index[sid]
        return [(i, c) for i, s in enumerate(ser) for c in s["centres"]]

    def choose(self, sid, rng):
        p = self.pairs(sid)
        if self.train and self.sample == "random":
            sel = rng.choice(len(p), self.k, replace=len(p) < self.k)
        elif self.train and self.sample == "grid":
            sel = np.minimum(((np.arange(self.k) + rng.random(self.k)) * len(p) / self.k).astype(int), len(p) - 1)
        elif self.train and self.sample == "series":
            ser = self.index[sid]
            per = np.full(len(ser), self.k // len(ser))
            per[rng.choice(len(ser), self.k % len(ser), replace=False)] += 1
            starts = np.cumsum([0] + [len(x["centres"]) for x in ser])
            sel = np.concatenate([starts[j] + np.minimum(((np.arange(m) + rng.random(m)) * len(ser[j]["centres"]) / m).astype(int),
                                                          len(ser[j]["centres"]) - 1) for j, m in enumerate(per) if m > 0])
        else:
            sel = np.unique(np.linspace(0, len(p) - 1, min(self.k, len(p))).round().astype(int))
        return [p[i] for i in sel]

    def __getitem__(self, i):
        sid = sid_ = self.studies[i]
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        win = self.choose(sid, rng)
        ser = self.index[sid]
        rows = np.array([[ser[s]["rows"][c - 1], ser[s]["rows"][c],
                          ser[s]["prow"][c] if ser[s]["prow"] is not None and ser[s]["prow"][c] >= 0 else ser[s]["rows"][c + 1]] for s, c in win])
        uniq, inv = np.unique(rows.ravel(), return_inverse=True)
        o = (self.cache.px - self.crop) // 2
        px = self.cache.read(uniq)[:, o:o + self.crop, o:o + self.crop]
        kt = self.k + self.k_crops
        x = np.zeros((kt, 3, self.crop, self.crop), np.uint8)
        x[: len(win)] = px[inv].reshape(len(win), 3, self.crop, self.crop)
        for j, (s, _) in enumerate(win):
            if ser[s]["hflip"]:
                x[j] = x[j][..., ::-1]
        mask = np.zeros(kt, bool)
        mask[: len(win)] = True
        typ = np.zeros(kt, np.int64)
        plane = np.zeros(kt, np.int64)
        pos = np.zeros(kt, np.float32)
        kind = np.zeros(kt, np.int64)
        sid = np.full(kt, 99, np.int64)                           # series index within the study (99 = padding, 98 = crop)
        for j, (s, c) in enumerate(win):
            typ[j], plane[j], pos[j], sid[j] = ser[s]["type"], ser[s]["plane"], ser[s]["pos"][c], s
        if self.k_crops and sid_ in self.crops:
            j = self.k
            for b_i, bag in enumerate(("medial", "lateral", "pf")):
                cands, m = self.crops[sid_][bag], self.k_crops // 3
                sel = (rng.choice(len(cands), m, replace=len(cands) < m) if self.train else
                       np.unique(np.linspace(0, len(cands) - 1, min(m, len(cands))).round().astype(int)))
                ws = [cands[q] for q in sel]
                img = self.cache.read(np.array([w_[:3] for w_ in ws]).ravel()).reshape(len(ws), 3, self.cache.px, self.cache.px)
                for w_, im in zip(ws, img):
                    x[j] = crop_window(im, w_, self.crop)
                    mask[j], kind[j], sid[j] = True, b_i + 1, 98
                    typ[j], plane[j], pos[j] = w_[7], w_[8], w_[9]
                    j += 1
        cov = np.zeros((kt, len(FINDINGS)), np.float32)    # structure coverage per window (structure_coverage), 0 if unused
        if self.cov is not None:
            cov[: len(win)] = self.cov[[ser[s]["rows"][c] for s, c in win]]
        y = self.targets[i] if self.targets is not None else np.zeros(12, np.float32)
        w = self.weights[i] if self.weights is not None else np.ones(12, np.float32)
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "type": torch.from_numpy(typ), "plane": torch.from_numpy(plane),
                "pos": torch.from_numpy(pos), "sid": torch.from_numpy(sid), "cov": torch.from_numpy(cov), "kind": torch.from_numpy(kind), "y": torch.from_numpy(np.asarray(y, np.float32)), "w": torch.from_numpy(np.asarray(w, np.float32))}


def gpu_augment(x, cfg):
    """x: float (N, 3, S, S) in [0, 1] before normalisation; each window gets its own random transform. All strengths 0 = off.
    aug_rot (deg), aug_scale (+-fraction), aug_shift (+-fraction of size): one affine resample, applied to a share aug_geo_p of windows; aug_bc: brightness +-a and contrast
    x(1 +- a); aug_gamma: gamma in exp(+-a); aug_noise: gaussian sigma; aug_erase: probability of erasing one random box (<= 1/3 side);
    aug_lowres: probability of a down- (to 40-80% size, one factor per batch) and up-sample, mimicking low-resolution scanners."""
    n, dev = x.shape[0], x.device
    lr = cfg.get("aug_lowres", 0)
    if lr:
        m = torch.rand(n, device=dev) < lr
        size = x.shape[-1]
        small = int(size * (0.4 + 0.4 * torch.rand(1).item()))
        low = F.interpolate(F.interpolate(x, size=(small, small), mode="bilinear", antialias=True, align_corners=False), size=(size, size),
                            mode="bilinear", align_corners=False)
        x = torch.where(m.view(-1, 1, 1, 1), low, x)
    rot, sc, sh = cfg.get("aug_rot", 0), cfg.get("aug_scale", 0), cfg.get("aug_shift", 0)
    if rot or sc or sh:
        g = (torch.rand(n, device=dev) < cfg.get("aug_geo_p", 1.0)).float()          # windows that get the affine at all
        a = (torch.rand(n, device=dev) * 2 - 1) * rot * math.pi / 180 * g
        s = 1 + (torch.rand(n, device=dev) * 2 - 1) * sc * g
        t = (torch.rand(n, 2, device=dev) * 2 - 1) * sh * 2 * g[:, None]
        th = torch.stack([torch.stack([a.cos() / s, -a.sin() / s, t[:, 0]], 1), torch.stack([a.sin() / s, a.cos() / s, t[:, 1]], 1)], 1)
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    bc = cfg.get("aug_bc", 0)
    if bc:
        c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * bc
        b = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * bc
        x = (x * c + b)
    g = cfg.get("aug_gamma", 0)
    if g:
        x = x.clamp(0, 1) ** torch.exp((torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * g)
    nz = cfg.get("aug_noise", 0)
    if nz:
        x = x + torch.randn_like(x) * nz
    er = cfg.get("aug_erase", 0)
    if er:
        size = x.shape[-1]
        m = torch.rand(n, device=dev) < er
        hw = (torch.rand(n, 2, device=dev) * size / 3).long() + 1
        xy = (torch.rand(n, 2, device=dev) * (size - hw)).long()
        yy = torch.arange(size, device=dev)
        rmask = (yy[None] >= xy[:, :1]) & (yy[None] < xy[:, :1] + hw[:, :1])
        cmask = (yy[None] >= xy[:, 1:]) & (yy[None] < xy[:, 1:] + hw[:, 1:])
        box = (rmask[:, :, None] & cmask[:, None, :] & m[:, None, None])[:, None]
        x = x.masked_fill(box, 0.0)
    return x


def gpu_prep(x, size, mode="bilinear", cfg=None, mean=MEAN, std=STD, mono=False):
    """uint8 (B, K, 3, H, W) on GPU -> float (B*K, 3, size, size) normalised with the backbone's pretrained mean / std.
    mono: the centre slice in all three channels (backbones pretrained on grayscale copied to RGB, e.g. RadImageNet: forum #737696,
    centre x3 beat 3 adjacent slices by +0.020 for RadImageNet R50)."""
    b, k = x.shape[:2]
    x = x.flatten(0, 1).float().div_(255)
    if mono:
        x = x[:, 1:2].expand(-1, 3, -1, -1)
    if x.shape[-1] != size:
        if mode == "area":
            x = F.interpolate(x, size=(size, size), mode="area")
        else:
            x = F.interpolate(x, size=(size, size), mode=mode, antialias=True, align_corners=False)
    if cfg is not None:
        x = gpu_augment(x, cfg)
    return (x - mean.to(x.device)) / std.to(x.device)
