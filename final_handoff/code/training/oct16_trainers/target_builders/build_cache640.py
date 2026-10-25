"""U57 native-resolution cache (USER 10-18 ~09:40 via Master: high-resolution readers for MM / Fracture misses; Master's check: the 0.4 mm/px
cache DOWNSAMPLES 74 % of sag / cor series, native median 0.32 mm/px): fov160 at 0.25 mm/px = px 640, JPEG q95, built straight from the
native frames (data/frames/*.b2nd) with common.cache's plan() / apply_inplane / slab rule; the intensity scale 'hi' per series is TAKEN
from fov160_px400_q95 (only the resolution differs). Same row plan as fov160_px400_q95 (asserted: series order, start, n), so every
localiser / window row index carries over; the crop code scales boxes by cache.px and mm_per_px.
Writes data/cache/fov160_px640_q95/{blobs.bin, offsets.npy, series.parquet, slices.parquet, build.json} (SliceCache 'jpeg' format).
--smoke N: the first N studies into the scratch dir given by --out; F3 check = the 640 slices INTER_AREA-downsampled to 400 vs the 400 cache.
taskset -c 16-27 nice -n 19 conda run -n rsna python experiments/v3/oct16_experts/scripts/build_cache640.py [--workers 12] [--smoke 3 --out DIR]"""
import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import blosc2
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from common.cache import SLAB_MM, THIN_MM, HI_PCT, SliceCache, plan  # noqa: E402
from common.geometry import apply_inplane  # noqa: E402
from common.paths import CACHE, FRAMES  # noqa: E402

FOV, PX, Q = 160, 640, 95
SRC = "fov160_px400_q95"


def process_study(args):
    """-> (start row, [jpeg bytes per stored slice]) for one study's series, in plan order."""
    group, hi_of = args
    mm = FOV / PX
    out = []
    for r in group:
        res = []
        for sop in r["sops"]:
            a = blosc2.open(str(FRAMES / f"{r['StudyInstanceUID']}_{r['SeriesInstanceUID']}_{sop}.b2nd"))[:].astype(np.float32)
            a = np.ascontiguousarray(apply_inplane(a, r["ops"]))
            transpose = r["ops"][0]
            sp_r, sp_c = (r["sp_col"], r["sp_row"]) if transpose else (r["sp_row"], r["sp_col"])
            nh, nw = max(1, int(round(a.shape[0] * sp_r / mm))), max(1, int(round(a.shape[1] * sp_c / mm)))
            interp = cv2.INTER_AREA if (nh < a.shape[0] and nw < a.shape[1]) else cv2.INTER_LINEAR
            a = cv2.resize(a, (nw, nh), interpolation=interp)
            o = np.zeros((PX, PX), np.float32)
            y0, x0 = (nh - PX) // 2, (nw - PX) // 2
            sy, sx, dy, dx = max(y0, 0), max(x0, 0), max(-y0, 0), max(-x0, 0)
            hh, ww = min(PX - dy, nh - sy), min(PX - dx, nw - sx)
            o[dy:dy + hh, dx:dx + ww] = a[sy:sy + hh, sx:sx + ww]
            res.append(o)
        vol = np.stack(res)
        k, n = r["k"], r["n"]
        if k > 1:
            vol = vol[: n * k].reshape(n, k, PX, PX).mean(1)
        vol = vol[:n]
        hi = hi_of[r["SeriesInstanceUID"]]
        u8 = np.clip(np.clip(vol, 0, hi) / hi * 255 + 0.5, 0, 255).astype(np.uint8)
        out += [cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, Q])[1].tobytes() for s in u8]
    return group[0]["start"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t = time.time()
    src = SliceCache(SRC)
    p = plan(FOV, PX)
    ref = src.series.set_index("SeriesInstanceUID")
    assert len(p) == len(ref) and (p.SeriesInstanceUID.to_numpy() == src.series.SeriesInstanceUID.to_numpy()).all(), "series order differs"
    assert (p.start.to_numpy() == ref.loc[p.SeriesInstanceUID, "start"].to_numpy()).all() and \
        (p.n.to_numpy() == ref.loc[p.SeriesInstanceUID, "n"].to_numpy()).all(), "row plan differs from the 400 cache"
    hi_of = ref["hi"].to_dict()
    groups = [g.to_dict("records") for _, g in p.groupby("StudyInstanceUID", sort=False)]
    assert all(g[0]["start"] < h[0]["start"] for g, h in zip(groups, groups[1:])), "studies not in row order"
    if a.smoke:
        groups = groups[: a.smoke]
    d = Path(a.out) if a.out else CACHE / f"fov{FOV}_px{PX}_q95"
    assert a.smoke or not (d / "build.json").exists(), f"{d} exists"
    d.mkdir(parents=True, exist_ok=True)
    total = int(sum(r["n"] for g in groups for r in g))
    print(f"plan OK (= {SRC} rows): {len(groups)} studies, {total} slices; {time.time() - t:.0f}s", flush=True)
    offsets, pos, nxt = [0], 0, groups[0][0]["start"]
    with open(d / "blobs.bin", "wb") as fh, Pool(a.workers) as pool:
        for i, (start, blobs) in enumerate(pool.imap(process_study, [(g, hi_of) for g in groups], chunksize=1)):
            assert start == nxt, (start, nxt)
            for b in blobs:
                fh.write(b)
                pos += len(b)
                offsets.append(pos)
            nxt = start + len(blobs)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(groups)} studies, {pos / 1e9:.1f} GB, {time.time() - t:.0f}s", flush=True)
    np.save(d / "offsets.npy", np.asarray(offsets, np.int64))
    shutil.copy(CACHE / SRC / "series.parquet", d / "series.parquet")
    shutil.copy(CACHE / SRC / "slices.parquet", d / "slices.parquet")
    json.dump({"fov_mm": FOV, "px": PX, "mm_per_px": FOV / PX, "hi_pct": HI_PCT, "slab_mm": SLAB_MM, "thin_mm": THIN_MM, "n_slices": total,
               "built": time.strftime("%Y-%m-%d %H:%M"), "format": "jpeg", "quality": Q, "src": "data/frames (native) + hi of " + SRC,
               "bytes": pos, "smoke": a.smoke}, open(d / "build.json", "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s: {total} slices, {pos / 1e9:.2f} GB", flush=True)
    if a.smoke:                                            # F3: downsample 640 -> 400 vs the 400 cache on the same rows
        c6 = SliceCache.__new__(SliceCache)
        c6.meta, c6.px, c6.format = json.load(open(d / "build.json")), PX, "jpeg"
        c6.path, c6.offsets, c6._store = d / "blobs.bin", np.load(d / "offsets.npy"), None
        rows = np.linspace(0, total - 1, 40).astype(int)
        x6, x4 = c6.read(rows), src.read(rows)
        dn = np.stack([cv2.resize(s, (400, 400), interpolation=cv2.INTER_AREA) for s in x6]).astype(np.float32)
        err = np.abs(dn - x4.astype(np.float32))
        print(f"F3 smoke: 640 -> 400 vs the 400 cache on {len(rows)} rows: mean |d| {err.mean():.2f} grey, p99 {np.percentile(err, 99):.1f}, "
              f"corr {np.corrcoef(dn.ravel(), x4.ravel())[0, 1]:.5f}", flush=True)


if __name__ == "__main__":
    main()
