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
CACHE = "fov160_px400_q95"   # 10-18 USER U57: the slice cache the crops are cut from; trainers set it from cfg "cache" (e.g. fov160_px640_q95 =
                              # 0.25 mm/px native-resolution copy with the SAME row plan; crop boxes scale by cache.px / mm_per_px)

EXP = Path(__file__).resolve().parents[1]
SEG_DIR = EXP.parents[2] / "work/knee_seg/nnunet_knee/out_all/masks"   # 10-18 U44: nnU-Net Knee label volumes, one sagittal FS series per study,
SEG_DIR_COR = SEG_DIR.parent / "masks_cor"                              # + their projection onto every coronal series (work/knee_seg/project_masks.py)
SEG_DROPS = {"bc": np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 1], bool),     # in that series' cache grid; 'bc' drops cartilages 1-4 + bones 7-9 (kept:
             "men": np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], bool),    # background 0 = soft tissue / fluid, menisci 5 / 6); 'men' (U47 loa_mm) drops the menisci
             "bone": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], bool)}   # U44 #2: femur / tibia / patella only (marrow = the Contusion cue of MCL)
STRUCT = {"pouch": "suprapatellar", "hoffa": "hoffa", "post": "popliteal", "notch": "notch", "knee": "joint", "gutter": "patella",
          "mcl": "mcl", "notchsc": "notch",           # oct16_experts: ligament expert bags (mcl: coronal + axial MCL box; notchsc: notch sag + cor)
          "patcart": "pat_cart", "pfj": "pf_joint", "latmen": "lat_men", "medmen": "med_men", "postz": "popliteal",
          "lattib": "lat_tib_cart", "mclz": "mcl",    # oct16_experts W batch (user U8): region zooms for PF / menisci / Baker's / LOA / MCL
          "medc": "comp_medial", "latc": "comp_lateral", "pfl": "pf_joint",   # ME1 merged multi-view (user U11): 100 mm compartment / PF locals
          "medc60": "comp_medial", "latc60": "comp_lateral", "pfl60": "pf_joint", "medc40": "comp_medial", "latc40": "comp_lateral",
          "pfl40": "pf_joint", "notch40": "notch", "knee140": "joint",
          "notch32": "notch", "notch28": "notch", "mcl40": "mcl", "mclza": "mcl", "post48": "popliteal", "pouch48": "suprapatellar",
          "hoffa48": "hoffa", "latc48": "comp_lateral", "medc48": "comp_medial",   # 10-17 U16 zoom double-down (ME4b / Z variants)
          "mclt": "mcl", "notch24": "notch", "notch48": "notch", "medmen40": "med_men", "root40": "notch", "lmfe": "lat_men"}   # 10-17 U17 C1 (Master's crop_ideas): shape-matched MCL crop, a TALL coronal box (see ASPECT)  # ME4 zooms; knee140 = the main's 140 mm field (ME2)
PLANES_OF = {"pouch": ("Sagittal", "Coronal", "Axial"), "hoffa": ("Sagittal", "Coronal", "Axial"), "post": ("Sagittal", "Axial"),
             "notch": ("Sagittal", "Coronal", "Axial"), "knee": ("Sagittal", "Coronal", "Axial"), "gutter": ("Axial",),
             "mcl": ("Coronal", "Axial"), "notchsc": ("Sagittal", "Coronal"),
             "patcart": ("Axial", "Sagittal"), "pfj": ("Axial", "Sagittal"), "latmen": ("Sagittal", "Coronal"), "medmen": ("Sagittal", "Coronal"),
             "postz": ("Axial", "Sagittal"), "lattib": ("Coronal", "Sagittal"), "mclz": ("Coronal",),
             "medc": ("Coronal", "Sagittal"), "latc": ("Coronal", "Sagittal"), "pfl": ("Sagittal", "Axial"),
             "medc60": ("Coronal", "Sagittal"), "latc60": ("Coronal", "Sagittal"), "pfl60": ("Sagittal", "Axial"),
             "medc40": ("Coronal", "Sagittal"), "latc40": ("Coronal", "Sagittal"), "pfl40": ("Sagittal", "Axial"), "notch40": ("Sagittal", "Coronal"),
             "knee140": ("Sagittal", "Coronal", "Axial"),
             "notch32": ("Sagittal", "Coronal"), "notch28": ("Sagittal", "Coronal"), "mcl40": ("Coronal", "Axial"), "mclza": ("Coronal", "Axial"),
             "post48": ("Sagittal", "Axial"), "pouch48": ("Sagittal", "Coronal", "Axial"), "hoffa48": ("Sagittal", "Coronal", "Axial"),
             "latc48": ("Coronal", "Sagittal"), "medc48": ("Coronal", "Sagittal"), "mclt": ("Coronal",), "notch24": ("Sagittal", "Coronal"), "notch48": ("Sagittal", "Coronal"), "medmen40": ("Sagittal", "Coronal"), "root40": ("Sagittal", "Coronal"), "lmfe": ("Sagittal", "Coronal")}
CROP_MM = {"pouch": 80, "hoffa": 72, "post": 100, "notch": 64, "knee": 160, "gutter": 110, "mcl": 70, "notchsc": 64,
           "patcart": 48, "pfj": 72, "latmen": 48, "medmen": 48, "postz": 64, "lattib": 48, "mclz": 48, "medc": 100, "latc": 100, "pfl": 100,
           "medc60": 60, "latc60": 60, "pfl60": 60, "medc40": 40, "latc40": 40, "pfl40": 40, "notch40": 40, "knee140": 140,
           "notch32": 32, "notch28": 28, "mcl40": 40, "mclza": 48, "post48": 48, "pouch48": 48, "hoffa48": 48, "latc48": 48, "medc48": 48,
           "mclt": 48, "notch24": 24, "notch48": 48, "medmen40": 40, "root40": 40, "lmfe": 48}
ANCHOR_Y = {"root40": "joint"}   # 10-18 (U43, Master: LM posterior-root crop): crop height from the joint box centre (~ the joint line,
                                # ~5 mm above the tibial plateau) at the notch box x = the tibial spines / root attachments, below the ACL mid-substance
NOTCH_OUT = {"lmfe": "notch"}   # 10-18 (Brainstorm LM req 5 S4.2, Master): LM crop WITHOUT the ACL notch in view. Coronal: the crop spans the
                                # LM box from its far edge (- 4 mm) to 1 mm short of the notch box's near edge (whole meniscus incl. the free
                                # edge, notch out; latmen's 48 mm crop had the notch in view in 99.9 % of coronal series); sagittal: latmen's
                                # crop on slices with p_notch <= 0.5 only (the notch-adjacent root slices drop out - stated)
ASPECT = {"mclt": 2.0}   # height / width of a non-square view (CROP_MM = its width): mclt = 48 mm wide x 96 mm tall (femoral origin to
                         # tibial insertion of the MCL on coronal slices); the model input is px wide x px * ASPECT tall; one aspect per model
P_MIN = 0.5
TOP_FALLBACK = 3
TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def bag_names(bags, plane_bags=False, planes=""):
    keep = lambda p: not planes or p in planes.split(",")
    return [f"{b}:{p}" for b in bags for p in PLANES_OF[b] if keep(p)] if plane_bags else list(bags)


def build_bags(bags, crop_scale=1.0, trim=0.1, fs_only=False, plane_bags=False, clamp=False, planes="", nofs_only=False, loc_dir=None,
               cache_name=None, type_map=None, series_types=None):
    """study -> bag -> list of (r0, r1, r2 cache rows of the window, hflip, cy, cx, side_px, type, plane, pos, p).
    knee bag: centred on the image centre (the cache is centred on the knee), slices of the localiser joint box.
    loc_dir / cache_name / type_map / series_types (10-18 OAI port of oa_data, USER 'ok, i want to try external data'): an EXTERNAL source's
    localiser output (the same v2 localiser, rows of ITS cache), its slice cache (absolute path), a series_type remap (OAI DESS -> *_PD_FS) and
    the ORIGINAL series types kept (None = all); defaults = the competition data, unchanged."""
    loc = Path(loc_dir) if loc_dir else EXP / "loc/pred"
    sl = pd.read_parquet(loc / "slices.parquet",
                         columns=["SeriesInstanceUID", "StudyInstanceUID", "plane", "idx", "row", "pos_mm"] + list(dict.fromkeys(
                             [f"p_{STRUCT[b]}" for b in bags] + [f"p_{NOTCH_OUT[b]}" for b in bags if b in NOTCH_OUT])))   # zoom variants share a structure
    ser = pd.read_parquet(loc / "series.parquet")
    cache = SliceCache(cache_name or CACHE)
    type_map = type_map or {}
    meta = cache.series.set_index("SeriesInstanceUID")
    side = pd.read_csv(Path(__file__).resolve().parent / "series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    px_mm = cache.meta["mm_per_px"]
    out = {}
    for (study, uid), g in sl.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        g = g.sort_values("idx")
        n, plane = len(g), g.plane.iloc[0]
        stype = meta.at[uid, "series_type"]
        if n < 3 or (planes and plane not in planes.split(",")) or (series_types is not None and stype not in series_types):
            continue
        stype = type_map.get(stype, stype)
        rows = g.row.to_numpy()
        flip = uid in flips and plane != "Sagittal"
        for bag in bags:
            if plane not in PLANES_OF[bag]:
                continue
            st = STRUCT[bag]
            p = g[f"p_{st}"].to_numpy(np.float32)
            if uid not in ser.index or np.isnan(ser.at[uid, f"{st}_y0"]):
                continue
            if bag in ("knee", "knee140"):
                cy = cx = cache.px / 2
            else:
                b = ser.loc[uid, [f"{st}_y0", f"{st}_y1", f"{st}_x0", f"{st}_x1"]].to_numpy(float) * cache.px
                cy, cx = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
                if bag in ANCHOR_Y and not np.isnan(ser.at[uid, f"{ANCHOR_Y[bag]}_y0"]):
                    cy = (ser.at[uid, f"{ANCHOR_Y[bag]}_y0"] + ser.at[uid, f"{ANCHOR_Y[bag]}_y1"]) / 2 * cache.px
                nb = NOTCH_OUT.get(bag)
                if nb and plane == "Coronal":
                    if np.isnan(ser.at[uid, f"{nb}_x0"]):
                        continue                           # no notch box -> the no-notch crop cannot be placed (never silently fall back)
                    n0, n1 = ser.at[uid, f"{nb}_x0"] * cache.px, ser.at[uid, f"{nb}_x1"] * cache.px
                    m, gap = 4 / px_mm, 1 / px_mm
                    a0, a1 = (b[2] - m, n0 - gap) if (n0 + n1) / 2 > cx else (n1 + gap, b[3] + m)
                    cx, crop_side = (a0 + a1) / 2, a1 - a0
                    if crop_side < 16 / px_mm:
                        continue                           # notch box overlapping the LM box: no notch-free view of the meniscus
                if flip:
                    cx = cache.px - cx                     # box is in the corrected view: raw-image centre (crop mirrored later)
            side_px = CROP_MM[bag] * crop_scale / px_mm
            if NOTCH_OUT.get(bag) and plane == "Coronal":
                side_px = crop_side * crop_scale
            if clamp:                                      # keep the crop inside the image where it fits (raw coordinates)
                lo, hi = min(side_px / 2, cache.px / 2), max(cache.px - side_px / 2, cache.px / 2)
                hy = side_px * ASPECT.get(bag, 1.0) / 2    # 10-17: a tall view clamps its height separately (aspect 1 = unchanged)
                lo_y, hi_y = min(hy, cache.px / 2), max(cache.px - hy, cache.px / 2)
                cy, cx = float(np.clip(cy, lo_y, hi_y)), float(np.clip(cx, lo, hi))
            keep = [i for i in range(1, n - 1) if p[i] > P_MIN]
            if NOTCH_OUT.get(bag) and plane == "Sagittal":
                pn = g[f"p_{NOTCH_OUT[bag]}"].to_numpy(np.float32)
                keep = [i for i in keep if pn[i] <= P_MIN]
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
    if nofs_only:                                          # U22 / Master 10-17: coupling-aware MCL crop - non-FS windows only where a bag has any
        for b in out.values():
            for key, c in b.items():
                b[key] = [w for w in c if not SERIES_TYPES[w[7]].endswith("_FS")] or c
    return out


def seg_series_only(bags):
    """U44: keep only windows of series that have a segmentation mask (studies left without any window are dropped)."""
    sr = SliceCache(CACHE).series.sort_values("start")
    start, uid = sr.start.to_numpy(), sr.SeriesInstanceUID.to_numpy()
    have = {f.stem for f in SEG_DIR.glob("*.npz")} | {f.stem for f in SEG_DIR_COR.glob("*.npz")}
    out = {}
    for sid, b in bags.items():
        nb = {k: [w for w in c if uid[np.searchsorted(start, w[1], side="right") - 1] in have] for k, c in b.items()}
        if any(nb.values()):
            out[sid] = nb
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
    dil = 0.0      # oct16_experts S-D (public D1 / D2): train-time window dilation - with this probability per window read slices
                   # (i-2, i, i+2) instead of (i-1, i, i+1), clipped to the series (cache rows of a series are contiguous: start .. start+n-1)
    seg = False    # 10-18 U44: hide the SEG_DROPS[seg_drop] structures (bones + cartilage) of windows whose series has a mask: those pixels take the
                   # median of the crop's kept pixels (per channel; no hard edge); the mask goes through the same crop / resize
    seg_set = "bc"    # which SEG_DROPS set (cfg seg_drop; U47: 'men' = hide the menisci)
    seg_morph = 0     # 10-18 (Master, U44 register): k > 0 dilates / k < 0 erodes the cropped mask by |k| output pixels (cfg seg_morph):
                      # loa_md = a dilated menisci mask (the plain 'men' mask hid 0.1%), mclz_mm = an eroded bone mask sparing the attachments

    def __init__(self, studies, bags, bag_names, targets, weights, k, train, px=224, seed=0, n_out=1, ext=None):
        self.caches = [SliceCache(CACHE)] + ([SliceCache(ext[0])] if ext else [])   # 10-18 OAI: ext = (external cache, its study ids)
        self.cache, self.ext = self.caches[0], set(ext[1]) if ext else set()
        self.bounds = [(c.series.sort_values("start").start.to_numpy(), (c.series.sort_values("start").start + c.series.sort_values("start").n - 1).to_numpy())
                       for c in self.caches]
        sr = self.cache.series.sort_values("start")
        self.ser_start, self.ser_end = sr.start.to_numpy(), (sr.start + sr.n - 1).to_numpy()
        self.ser_uid, self.ser_n, self.seg_cache = sr.SeriesInstanceUID.to_numpy(), sr.n.to_numpy(), {}
        self.studies, self.bags, self.bag_names, self.targets, self.weights = list(studies), bags, bag_names, targets, weights
        self.k, self.train, self.px, self.seed, self.epoch, self.n_out = k, train, px, seed, 0, n_out
        asp = {ASPECT.get(b.split(":")[0], 1.0) for b in bag_names}
        assert len(asp) == 1, f"one view aspect per model: {asp}"
        self.asp = asp.pop()
        self.px_h = int(round(px * self.asp))                # input height (px wide); = px for square views

    def __len__(self):
        return len(self.studies)

    def choose(self, cands, rng):
        if self.train:
            return [cands[i] for i in rng.choice(len(cands), self.k, replace=len(cands) < self.k)]
        return [cands[i] for i in np.unique(np.linspace(0, len(cands) - 1, min(self.k, len(cands))).round().astype(int))]

    def crop(self, img, w):
        """Crop centred at (cy, cx) of the raw cached image (side wide, side x aspect tall; square for aspect 1), mirrored afterwards when
        hflip, resized to px wide x px_h tall."""
        _, _, _, hflip, cy, cx, side, *_ = w
        h, wd = int(round(side * self.asp)), int(round(side))
        y0, x0 = int(round(cy - side * self.asp / 2)), int(round(cx - side / 2))
        out = np.zeros((3, h, wd), np.uint8)
        sy, sx = max(y0, 0), max(x0, 0)
        ey, ex = min(y0 + h, img.shape[1]), min(x0 + wd, img.shape[2])
        if ey > sy and ex > sx:
            out[:, sy - y0:ey - y0, sx - x0:ex - x0] = img[:, sy:ey, sx:ex]
        if hflip:
            out = out[:, :, ::-1]
        if wd == self.px and h == self.px_h:
            return np.ascontiguousarray(out)
        return np.stack([cv2.resize(np.ascontiguousarray(c), (self.px, self.px_h), interpolation=cv2.INTER_AREA if wd > self.px else cv2.INTER_LINEAR)
                         for c in out])

    def seg_drop(self, rows, w):
        """(3, px_h, px) bool: SEG_DROPS[seg_set] pixels of the window's crop, or None when its series has no (matching) mask."""
        k = np.searchsorted(self.ser_start, rows[1], side="right") - 1
        uid = self.ser_uid[k]
        if uid not in self.seg_cache:
            if len(self.seg_cache) > 64:
                self.seg_cache.clear()
            f = SEG_DIR / f"{uid}.npz"
            f = f if f.exists() else SEG_DIR_COR / f"{uid}.npz"
            lab = np.load(f)["lab"] if f.exists() else None
            self.seg_cache[uid] = lab if lab is not None and len(lab) == self.ser_n[k] else None
        lab = self.seg_cache[uid]
        if lab is None:
            return None
        m = self.crop(SEG_DROPS[self.seg_set][lab[rows - self.ser_start[k]]].astype(np.uint8) * 255, w) > 127
        if self.seg_morph:
            ker = np.ones((3, 3), np.uint8)
            op = cv2.dilate if self.seg_morph > 0 else cv2.erode
            m = np.stack([op(c.astype(np.uint8), ker, iterations=abs(self.seg_morph)) > 0 for c in m])
        return m

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
        x = np.zeros((kt, 3, self.px_h, self.px), np.uint8)
        ci = 1 if sid in self.ext else 0                     # 10-18 OAI: the study's own cache
        cache, (s_start, s_end) = self.caches[ci], self.bounds[ci]
        if wins:                                             # eval ablations may leave a study without windows
            rows = np.array([w[:3] for w in wins])
            if self.train and self.dil > 0:
                d = rng.random(len(rows)) < self.dil
                if d.any():
                    k_ = np.searchsorted(s_start, rows[d, 1], side="right") - 1
                    rows[d, 0] = np.maximum(rows[d, 0] - 1, s_start[k_])
                    rows[d, 2] = np.minimum(rows[d, 2] + 1, s_end[k_])
            uniq, inv = np.unique(rows.ravel(), return_inverse=True)
            img = cache.read(uniq)[inv].reshape(len(wins), 3, cache.px, cache.px)
            for j, w in enumerate(wins):
                x[j] = self.crop(img[j], w)
                d = self.seg_drop(rows[j], w) if self.seg and ci == 0 else None
                if d is not None and (~d).any():
                    for c in range(3):
                        if d[c].any():
                            x[j, c][d[c]] = np.median(x[j, c][~d[c]])
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
