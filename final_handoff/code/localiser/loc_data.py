"""Localiser training data: per sampled slice of every (non-duplicate, correctly sided) series of the studies with nnU-Net boxes,
the 3-slice window (c-1, c, c+1) from the canonical cache resized to PX px, and per structure in STRUCTS: presence (frac > 0 in
data/derived/slice_boxes.parquet) and the box (y0, y1, x0, x1) / cache px.

Series in the side-flip table (flip_final) are left out: their cached image is mirrored while the boxes are named by anatomy.
Up to PER_SERIES slices per series, spread evenly over the series with a random offset.
Output: data/loc_{px}/x.npy (uint8 N x 3 x PX x PX), meta.parquet (study, series, row, plane, pos_mm, fold, n_slices, idx),
pres.npy (N x S bool), box.npy (N x S x 4 float32, NaN when absent)."""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from common.cache import SliceCache  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.meta import study_table  # noqa: E402

STRUCTS = ["comp_medial", "comp_lateral", "pf_joint", "patella", "joint"]
STRUCTS_V2 = STRUCTS + ["med_tib_cart", "lat_tib_cart", "pat_cart", "fem_cart", "med_men", "lat_men", "notch", "mcl", "suprapatellar",
                        "hoffa", "popliteal"]
PX = 128
PER_SERIES = 6
CACHE = "fov160_px400_q95"
OUT = HERE.parent / "data" / f"loc_{PX}"
_cache = None


def _read(job):
    global _cache
    if _cache is None:
        _cache = SliceCache(CACHE)
    rows = np.asarray(job)
    u, inv = np.unique(rows.ravel(), return_inverse=True)
    img = _cache.read(u)
    small = np.stack([cv2.resize(a, (PX, PX), interpolation=cv2.INTER_AREA) for a in img])
    return small[inv].reshape(len(rows), 3, PX, PX)


def targets_v2():
    """pres_v2.npy / box_v2.npy / known_v2.npy for STRUCTS_V2 on the existing windows (meta.parquet). known = the series has at
    least one slice with that structure (nnU-Net found it); else the cell is unknown, not negative (one meniscus is missed in ~35%
    of series)."""
    meta = pd.read_parquet(OUT / "meta.parquet")
    px = SliceCache(CACHE).px
    boxes = pd.read_parquet(ROOT / "data/derived/slice_boxes.parquet")
    boxes = boxes[boxes.structure.isin(STRUCTS_V2)]
    found = set(zip(boxes.SeriesInstanceUID, boxes.structure))
    b = boxes.set_index(["row", "structure"])[["y0", "y1", "x0", "x1"]]
    b = b[~b.index.duplicated()]
    pres = np.zeros((len(meta), len(STRUCTS_V2)), bool)
    known = np.zeros((len(meta), len(STRUCTS_V2)), bool)
    box = np.full((len(meta), len(STRUCTS_V2), 4), np.nan, np.float32)
    for j, st in enumerate(STRUCTS_V2):
        key = pd.MultiIndex.from_arrays([meta.row.to_numpy(), np.full(len(meta), st)])
        v = b.reindex(key).to_numpy(np.float32) / px
        pres[:, j] = ~np.isnan(v[:, 0])
        box[:, j] = v
        known[:, j] = [(u, st) in found for u in meta.SeriesInstanceUID]
    np.save(OUT / "pres_v2.npy", pres)
    np.save(OUT / "box_v2.npy", box)
    np.save(OUT / "known_v2.npy", known)
    print("presence:", dict(zip(STRUCTS_V2, pres.mean(0).round(3))), "\nknown:", dict(zip(STRUCTS_V2, known.mean(0).round(3))), flush=True)


def main():
    cache = SliceCache(CACHE)
    px = cache.px
    boxes = pd.read_parquet(ROOT / "data/derived/slice_boxes.parquet")
    boxes = boxes[boxes.structure.isin(STRUCTS)]
    flips = pd.read_csv(ROOT / "data/derived/series_side_final.csv", index_col=0)
    flipped = set(flips.index[flips.flip_final])
    have = set(boxes.StudyInstanceUID)
    folds = get_folds(5, 0, include_gold=True)
    s = cache.series[(cache.series.dup_of == "") & (cache.series.n >= 3) & cache.series.StudyInstanceUID.isin(have)
                     & ~cache.series.SeriesInstanceUID.isin(flipped)]
    pos_all = cache.slices.sort_values("row").pos_mm.to_numpy(np.float32)
    rng = np.random.default_rng(0)
    meta, windows = [], []
    for r in s.itertuples():
        m = min(PER_SERIES, r.n)
        idx = np.minimum(((np.arange(m) + rng.random(m)) * r.n / m).astype(int), r.n - 1)
        for i in idx:
            c = r.start + i
            windows.append([r.start + max(i - 1, 0), c, r.start + min(i + 1, r.n - 1)])
            meta.append((r.StudyInstanceUID, r.SeriesInstanceUID, c, r.plane, pos_all[c], r.n, i))
    meta = pd.DataFrame(meta, columns=["StudyInstanceUID", "SeriesInstanceUID", "row", "plane", "pos_mm", "n_slices", "idx"])
    meta["fold"] = folds.fold.reindex(meta.StudyInstanceUID).fillna(-1).astype(int).to_numpy()
    meta["gold"] = study_table().gold.reindex(meta.StudyInstanceUID).to_numpy()
    print(f"{len(s)} series, {meta.StudyInstanceUID.nunique()} studies, {len(meta)} samples", flush=True)
    b = boxes.set_index(["row", "structure"])[["y0", "y1", "x0", "x1"]]
    b = b[~b.index.duplicated()]
    pres = np.zeros((len(meta), len(STRUCTS)), bool)
    box = np.full((len(meta), len(STRUCTS), 4), np.nan, np.float32)
    for j, st in enumerate(STRUCTS):
        key = pd.MultiIndex.from_arrays([meta.row.to_numpy(), np.full(len(meta), st)])
        v = b.reindex(key).to_numpy(np.float32) / px
        pres[:, j] = ~np.isnan(v[:, 0])
        box[:, j] = v
    print("presence rate per structure:", dict(zip(STRUCTS, pres.mean(0).round(3))), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(OUT / "x.npy", mode="w+", dtype=np.uint8, shape=(len(meta), 3, PX, PX))
    chunk = 2000
    jobs = [windows[a:a + chunk] for a in range(0, len(windows), chunk)]
    with Pool(12) as pool:
        for k, arr in enumerate(pool.imap(_read, jobs)):
            x[k * chunk: k * chunk + len(arr)] = arr
            if k % 10 == 0:
                print(f"{(k + 1) * chunk}/{len(windows)}", flush=True)
    x.flush()
    meta.to_parquet(OUT / "meta.parquet")
    np.save(OUT / "pres.npy", pres)
    np.save(OUT / "box.npy", box)
    print("done", flush=True)


if __name__ == "__main__":
    targets_v2() if sys.argv[1:] == ["v2"] else main()
