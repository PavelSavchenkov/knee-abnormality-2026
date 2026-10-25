"""Map between our slice cache's pixel grid and DICOM patient coordinates (LPS, mm), per series.

Cache slice s of series u (common/cache.py): frames sorted along the canonical slice direction, consecutive groups of k frames
averaged into one slab; each frame turned into the canonical layout by (transpose, flip_rows, flip_cols), resized to 0.4 mm/px and
centre-cropped / padded to 400 px, so the cache centre (199.5, 199.5) is the image centre of the frame.
    patient(s, y, x) = centre(s) + (y - 199.5) * mm * row_dir + (x - 199.5) * mm * col_dir
row_dir / col_dir = patient directions of increasing canonical row / column (= +-IOP cosines); error <= ~0.4 mm (resize rounding).
Physical coordinates do not depend on the knee-side choice, so this is exact even for the ~2% mis-sided series.

    g = SeriesGeom(uid);  xyz = g.to_patient(s, y, x);  s, y, x, dist = g.to_cache(xyz)   (dist = mm from the nearest slab)

Validation: same-plane co-registered pairs map pixel to pixel (median 0 px; __main__); cross-plane checked visually by projecting
nnU-Net femur / tibia / patella / meniscus voxels of a sagittal series into coronal and axial slices (nnunet_knee/out_all/dots_*.png).
A same-plane test alone cannot catch swapped row / column directions (both series would be wrong the same way): that happened
before the r["transpose"] fix on 2026-10-08 01:48."""
import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.cache import SliceCache  # noqa: E402
from common.geometry import targets  # noqa: E402
from common.meta import frames, series  # noqa: E402


@functools.cache
def _cache(name="fov160_px400_q95"):
    c = SliceCache(name)
    return c.series.set_index("SeriesInstanceUID"), c.meta["mm_per_px"], c.px


@functools.cache
def _frames_by_series():
    f = frames()[["SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]]
    return dict(tuple(f.groupby("SeriesInstanceUID")))


class SeriesGeom:
    def __init__(self, uid, cache="fov160_px400_q95"):
        cs, self.mm, self.px = _cache(cache)
        r = cs.loc[uid]
        m = series().loc[uid]
        rcos = np.array([m.ImageOrientationPatient_rx, m.ImageOrientationPatient_ry, m.ImageOrientationPatient_rz], float)
        ccos = np.array([m.ImageOrientationPatient_cx, m.ImageOrientationPatient_cy, m.ImageOrientationPatient_cz], float)
        f = _frames_by_series()[uid].sort_values("InstanceNumber")
        ipp = f[["ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]].to_numpy(float)
        centre = ipp + rcos * (m.Columns - 1) / 2 * m.PixelSpacing_col + ccos * (m.Rows - 1) / 2 * m.PixelSpacing_row
        normal = targets(r.plane, r.side)[2]
        pos = ipp @ normal
        order = np.argsort(pos, kind="stable")
        k, n = int(r.k), int(r.n)
        self.centres = centre[order][: n * k].reshape(n, k, 3).mean(1)          # (n, 3) slab centres
        # r["transpose"], not r.transpose: the latter is pandas' Series.transpose method (always truthy)
        row_dir, col_dir = (rcos, ccos) if r["transpose"] else (ccos, rcos)    # increasing row / column after the transpose
        self.row_dir = -row_dir if r["flip_rows"] else row_dir
        self.col_dir = -col_dir if r["flip_cols"] else col_dir
        self.normal = np.cross(rcos, ccos)
        self.normal *= np.sign(self.normal @ normal)                          # along the canonical slice direction
        self.plane, self.side, self.n = r.plane, r.side, n

    def to_patient(self, s, y, x):
        s, y, x = np.asarray(s), np.asarray(y, float), np.asarray(x, float)
        c0 = (self.px - 1) / 2
        return self.centres[s] + ((y - c0) * self.mm)[..., None] * self.row_dir + ((x - c0) * self.mm)[..., None] * self.col_dir

    def to_cache(self, xyz):
        """(..., 3) patient points -> nearest slab index, row, col (float), distance (mm) to that slab's plane."""
        xyz = np.asarray(xyz, float)
        d = (xyz[..., None, :] - self.centres) @ self.normal                  # (..., n) signed distance to each slab
        s = np.abs(d).argmin(-1)
        rel = xyz - self.centres[s]
        c0 = (self.px - 1) / 2
        return s, rel @ self.row_dir / self.mm + c0, rel @ self.col_dir / self.mm + c0, np.take_along_axis(np.abs(d), s[..., None], -1)[..., 0]


if __name__ == "__main__":                     # check on co-registered same-plane pairs: a pixel maps to the same pixel
    import pandas as pd
    p = pd.read_csv(ROOT / "experiments/v0/oct7_baseline/ideas_archive/ideas_loop/out/coregistration.csv")
    cs, mm, px = _cache()
    rng = np.random.default_rng(0)
    errs = []
    pairs = p[p.coreg & (p.dz_mm < 0.1)].sample(30, random_state=0)
    for st, plane in zip(pairs.study, pairs.plane):
        u = cs[(cs.StudyInstanceUID == st) & (cs.plane == plane) & (cs.dup_of == "")].index[:2]
        a, b = SeriesGeom(u[0]), SeriesGeom(u[1])
        s = rng.integers(0, a.n, 20)
        y, x = rng.uniform(50, 350, 20), rng.uniform(50, 350, 20)
        sb, yb, xb, d = b.to_cache(a.to_patient(s, y, x))
        errs.append(np.c_[np.abs(yb - y), np.abs(xb - x), d])
    e = np.concatenate(errs)
    print("same-plane co-registered pairs: |dy| px median / q95", np.median(e[:, 0]).round(2), np.quantile(e[:, 0], 0.95).round(2),
          "| |dx|", np.median(e[:, 1]).round(2), np.quantile(e[:, 1], 0.95).round(2), "| slab dist mm median", np.median(e[:, 2]).round(2))
