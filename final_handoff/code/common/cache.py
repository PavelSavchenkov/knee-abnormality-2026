"""Canonical slice cache: every frame of every series, in canonical orientation (common/geometry.py), resampled to a fixed mm
grid around the image centre, intensity-scaled per series to uint8. Experiments crop / resize / augment on the GPU from here.

Build (once):   conda run -n rsna python -m common.cache build --fov 160 --px 400
JPEG copy:      conda run -n rsna python -m common.cache jpeg --src fov160_px400 --quality 95      -> fov160_px400_q95
Read:           c = SliceCache("fov160_px400_q95"); c.series (DataFrame), c.slices (DataFrame), c.read(rows) -> uint8 (n, px, px)

Formats: "u8" = one raw uint8 array (115 GB for fov160_px400: larger than the page cache can hold, so random reads thrash RAM and
systemd-oomd kills the session); "jpeg" = one JPEG per slice in blobs.bin + offsets.npy (q95: ~30 GB, mean abs error 0.65 grey levels,
p99 3; decode ~0.35 ms per slice). Train from the JPEG copy.

Per series:
- orientation side: series Laterality tag if L/R, else the study side (common/meta.py); other_knee = series side != study side
- frames sorted by position along the canonical slice direction; pos_mm = position relative to the series centre
- thin series (spacing < 2 mm): consecutive groups of k = round(3 mm / spacing) frames are averaged into one slab (remainder dropped)
- in-plane: stored pixels -> transpose / flips -> resize to mm_per_px (INTER_AREA when shrinking, INTER_LINEAR when enlarging)
  -> centre crop / zero pad to px x px
- intensity: clip(x, 0, hi) / hi * 255 with hi = 99.5th percentile of the series' resampled pixels (RescaleSlope cancels out)
- dup_of: SeriesInstanceUID of an earlier series of the same study with identical middle-frame pixels (else empty)"""
import argparse
import hashlib
import json
import os
import shutil
import time
from multiprocessing import Pool

import blosc2
import cv2
import numpy as np
import pandas as pd

from common.geometry import apply_inplane, inplane_ops, slice_positions
from common.meta import frames, series, series_side, series_table, study_table
from common.paths import CACHE, FRAMES

HI_PCT = 99.5
SLAB_MM = 3.0
THIN_MM = 2.0


def plan(fov, px):
    """One row per series with frame order, slab size and offsets in the slice store."""
    s, st, sr = series(), study_table(), series_table()
    side_tag = s.Laterality.where(s.Laterality.isin(["L", "R"]))
    study_side = st.side.reindex(s.StudyInstanceUID).to_numpy()
    orient_side = side_tag.fillna(pd.Series(study_side, index=s.index))
    fr = frames().sort_values(["SeriesInstanceUID", "InstanceNumber"])
    rows = []
    for uid, f in fr.groupby("SeriesInstanceUID", sort=False):
        x = s.loc[uid]
        r = np.array([x.ImageOrientationPatient_rx, x.ImageOrientationPatient_ry, x.ImageOrientationPatient_rz])
        c = np.array([x.ImageOrientationPatient_cx, x.ImageOrientationPatient_cy, x.ImageOrientationPatient_cz])
        plane, side = sr.at[uid, "plane"], orient_side[uid]
        pos = slice_positions(f[["ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]].to_numpy(), plane, side)
        order = np.argsort(pos, kind="stable")
        spacing = float(np.median(np.diff(pos[order]))) if len(pos) > 1 else 0.0
        k = max(1, int(round(SLAB_MM / spacing))) if 0 < spacing < THIN_MM else 1
        rows.append({"SeriesInstanceUID": uid, "StudyInstanceUID": x.StudyInstanceUID, "plane": plane, "side": side,
                     "other_knee": series_side()[uid] != st.at[x.StudyInstanceUID, "side"], "ops": inplane_ops(r, c, plane, side),
                     "sops": f.SOPInstanceUID.to_numpy()[order].tolist(), "pos": pos[order].tolist(), "frame_spacing": spacing, "k": k,
                     "n_frames": len(pos), "n": len(pos) // k, "sp_row": x.PixelSpacing_row, "sp_col": x.PixelSpacing_col,
                     "rows": x.Rows, "cols": x.Columns, "SeriesNumber": x.SeriesNumber})
    p = pd.DataFrame(rows).sort_values(["StudyInstanceUID", "SeriesNumber", "SeriesInstanceUID"]).reset_index(drop=True)
    p["start"] = np.concatenate([[0], np.cumsum(p.n.to_numpy())[:-1]])
    return p


_store = None


def _init(path, shape):
    global _store
    _store = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)


def _process_study(args):
    group, fov, px = args
    mm_per_px = fov / px
    out = []
    for r in group:
        imgs = []
        for sop in r["sops"]:
            a = blosc2.open(str(FRAMES / f"{r['StudyInstanceUID']}_{r['SeriesInstanceUID']}_{sop}.b2nd"))[:].astype(np.float32)
            imgs.append(a)
        mid = imgs[len(imgs) // 2]
        h = hashlib.md5(mid.tobytes()).hexdigest()
        transpose, fr, fc = r["ops"]
        sp_r, sp_c = (r["sp_col"], r["sp_row"]) if transpose else (r["sp_row"], r["sp_col"])
        res = []
        for a in imgs:
            a = np.ascontiguousarray(apply_inplane(a, r["ops"]))
            nh, nw = max(1, int(round(a.shape[0] * sp_r / mm_per_px))), max(1, int(round(a.shape[1] * sp_c / mm_per_px)))
            interp = cv2.INTER_AREA if (nh < a.shape[0] and nw < a.shape[1]) else cv2.INTER_LINEAR
            a = cv2.resize(a, (nw, nh), interpolation=interp)
            o = np.zeros((px, px), np.float32)
            y0, x0 = (nh - px) // 2, (nw - px) // 2           # crop offset (>0) or pad offset (<0)
            sy, sx = max(y0, 0), max(x0, 0)
            dy, dx = max(-y0, 0), max(-x0, 0)
            hh, ww = min(px - dy, nh - sy), min(px - dx, nw - sx)
            o[dy:dy + hh, dx:dx + ww] = a[sy:sy + hh, sx:sx + ww]
            res.append(o)
        vol = np.stack(res)
        k, n = r["k"], r["n"]
        if k > 1:
            vol = vol[: n * k].reshape(n, k, px, px).mean(1)
        hi = float(np.percentile(vol[:, ::4, ::4], HI_PCT))
        hi = hi if hi > 0 else 1.0
        _store[r["start"]: r["start"] + n] = np.clip(np.clip(vol, 0, hi) / hi * 255 + 0.5, 0, 255).astype(np.uint8)
        out.append({"SeriesInstanceUID": r["SeriesInstanceUID"], "hi": hi, "hash": h, "fov_mm": max(r["rows"] * r["sp_row"], r["cols"] * r["sp_col"])})
    return out


def build(fov=160, px=400, workers=26):
    name = f"fov{fov}_px{px}"
    d = CACHE / name
    d.mkdir(parents=True, exist_ok=True)
    t = time.time()
    p = plan(fov, px)
    total = int(p.n.sum())
    print(f"plan: {len(p)} series, {total} stored slices ({total * px * px / 1e9:.1f} GB), {time.time() - t:.0f}s", flush=True)
    store_path = d / "slices.u8"
    np.memmap(store_path, dtype=np.uint8, mode="w+", shape=(total, px, px)).flush()
    groups = [(g.to_dict("records"), fov, px) for _, g in p.groupby("StudyInstanceUID", sort=False)]
    info = []
    with Pool(workers, initializer=_init, initargs=(str(store_path), (total, px, px))) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_study, groups, chunksize=2)):
            info += res
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(groups)} studies, {time.time() - t:.0f}s", flush=True)
    p = p.merge(pd.DataFrame(info), on="SeriesInstanceUID")
    first = {}
    dup = []
    for r in p.itertuples():
        key = (r.StudyInstanceUID, r.hash)
        dup.append(first.get(key, ""))
        first.setdefault(key, r.SeriesInstanceUID)
    p["dup_of"] = dup
    sr = series_table()
    p = p.join(sr[["series_type", "weighting", "fs", "is_3d", "slice_spacing"]], on="SeriesInstanceUID")
    slices = []
    for r in p.itertuples():
        pos = np.asarray(r.pos)[: r.n * r.k].reshape(r.n, r.k).mean(1)
        slices.append(pd.DataFrame({"SeriesInstanceUID": r.SeriesInstanceUID, "row": r.start + np.arange(r.n), "idx": np.arange(r.n),
                                    "pos_mm": pos - (pos.min() + pos.max()) / 2}))
    slices = pd.concat(slices, ignore_index=True)
    p["slab_spacing"] = p.frame_spacing * p.k
    p["transpose"], p["flip_rows"], p["flip_cols"] = zip(*p.ops)
    p.drop(columns=["sops", "pos", "ops"]).to_parquet(d / "series.parquet")
    slices.to_parquet(d / "slices.parquet")
    json.dump({"fov_mm": fov, "px": px, "mm_per_px": fov / px, "hi_pct": HI_PCT, "slab_mm": SLAB_MM, "thin_mm": THIN_MM, "n_slices": total,
               "built": time.strftime("%Y-%m-%d %H:%M")}, open(d / "build.json", "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s: dup series {int((p.dup_of != '').sum())}, other-knee series {int(p.other_knee.sum())}, "
          f"thin series {int((p.k > 1).sum())}", flush=True)


def _encode_chunk(args):
    src, px, a, b, quality = args
    fd = os.open(src, os.O_RDONLY)
    x = np.frombuffer(os.pread(fd, (b - a) * px * px, a * px * px), np.uint8).reshape(b - a, px, px)
    os.posix_fadvise(fd, a * px * px, (b - a) * px * px, os.POSIX_FADV_DONTNEED)   # keep the raw file out of the page cache
    os.close(fd)
    return [cv2.imencode(".jpg", i, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes() for i in x]


def jpeg(src="fov160_px400", quality=95, workers=26, chunk=1000):
    s, d = CACHE / src, CACHE / f"{src}_q{quality}"
    d.mkdir(parents=True, exist_ok=False)
    meta = json.load(open(s / "build.json"))
    n, px = meta["n_slices"], meta["px"]
    off = np.zeros(n + 1, np.int64)
    t, i = time.time(), 0
    jobs = [(str(s / "slices.u8"), px, a, min(a + chunk, n), quality) for a in range(0, n, chunk)]
    with Pool(workers) as pool, open(d / "blobs.bin", "wb") as f:
        for k, blobs in enumerate(pool.imap(_encode_chunk, jobs)):
            for blob in blobs:
                f.write(blob)
                off[i + 1] = off[i] + len(blob)
                i += 1
            if (k + 1) % 50 == 0:
                print(f"{i}/{n} slices, {off[i] / 1e9:.1f} GB, {time.time() - t:.0f}s", flush=True)
    assert i == n
    np.save(d / "offsets.npy", off)
    for f in ["series.parquet", "slices.parquet"]:
        shutil.copy(s / f, d / f)
    json.dump(meta | {"format": "jpeg", "quality": quality, "src": src, "bytes": int(off[-1]), "built": time.strftime("%Y-%m-%d %H:%M")},
              open(d / "build.json", "w"), indent=1)
    print(f"done in {time.time() - t:.0f}s: {off[-1] / 1e9:.1f} GB", flush=True)


class SliceCache:
    def __init__(self, name="fov160_px400_q95"):
        d = CACHE / name
        self.meta = json.load(open(d / "build.json"))
        self.px = self.meta["px"]
        self.format = self.meta.get("format", "u8")
        self.series = pd.read_parquet(d / "series.parquet")
        self.slices = pd.read_parquet(d / "slices.parquet")
        self.path = d / ("blobs.bin" if self.format == "jpeg" else "slices.u8")
        self.offsets = np.load(d / "offsets.npy") if self.format == "jpeg" else None
        self._store = None

    @property
    def store(self):
        if self._store is None:                               # opened lazily so each DataLoader worker opens its own copy
            if self.format == "jpeg":
                self._store = os.open(self.path, os.O_RDONLY)
            else:
                self._store = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.meta["n_slices"], self.px, self.px))
        return self._store

    def read(self, rows):
        rows = np.asarray(rows)
        if self.format == "u8":
            return np.asarray(self.store[rows])
        out = np.empty((len(rows), self.px, self.px), np.uint8)
        for j, r in enumerate(rows):
            a, b = self.offsets[r], self.offsets[r + 1]
            out[j] = cv2.imdecode(np.frombuffer(os.pread(self.store, int(b - a), int(a)), np.uint8), cv2.IMREAD_GRAYSCALE)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "jpeg"])
    ap.add_argument("--fov", type=int, default=160)
    ap.add_argument("--px", type=int, default=400)
    ap.add_argument("--src", default="fov160_px400")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=26)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.fov, a.px, a.workers)
    else:
        jpeg(a.src, a.quality, a.workers)
