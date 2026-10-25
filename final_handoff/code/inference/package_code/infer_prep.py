"""Test-time preprocessing straight from DICOM, matching build_cache.py.

scan_study(study_dir, series_info) -> DataFrame (index SeriesInstanceUID) with the cache-meta fields used by select_slots
    (plane, fs, weighting, side, other_knee, n) plus the header data needed to build volumes.
series_volume(row) -> uint8 (D, 320, 320) volume, as in the cache.
series_info: rows of test_series.csv for the study (Fat_Suppression per series)."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom

from common_meta import weighting
from prep import n_stored, process_series, series_geometry


def _read(f, **kw):
    """pydicom.dcmread, then evict the file from the page cache (local runs read hundreds of GB of DICOM; without this they
    flood the page cache and cause memory-pressure stalls for everything else; harmless on Kaggle)."""
    d = pydicom.dcmread(f, **kw)
    try:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass
    return d

TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "Rows", "Columns", "Laterality", "RepetitionTime",
        "EchoTime", "ScanningSequence", "SeriesDescription", "InstanceNumber"]


def _num(v):
    return float(v) if v not in (None, "") else np.nan


def side_of(lat, centres_x):
    """Laterality tag (L / R) if present, else the sign of the mean image-centre x (+x = patient left)."""
    lat = str(lat or "").upper()
    lat = {"LEFT": "L", "RIGHT": "R"}.get(lat, lat)
    if lat in ("L", "R"):
        return lat
    return "L" if np.mean(centres_x) > 0 else "R"


def scan_series(series_dir):
    files = sorted(Path(series_dir).glob("*.dcm"))
    hs = [_read(f, stop_before_pixels=True, specific_tags=TAGS) for f in files]
    h0 = hs[0]
    r, c = np.array(h0.ImageOrientationPatient[:3], float), np.array(h0.ImageOrientationPatient[3:], float)
    ps = [float(v) for v in h0.PixelSpacing]
    ipp = np.array([[float(v) for v in h.ImagePositionPatient] for h in hs])
    centre = ipp + r * (int(h0.Columns) - 1) / 2 * ps[1] + c * (int(h0.Rows) - 1) / 2 * ps[0]
    row = SimpleNamespace(ScanningSequence="\\".join(h0.get("ScanningSequence", "")) if isinstance(h0.get("ScanningSequence"), (list, pydicom.multival.MultiValue)) else str(h0.get("ScanningSequence", "")),
                          RepetitionTime=_num(h0.get("RepetitionTime")), EchoTime=_num(h0.get("EchoTime")),
                          SeriesDescription=str(h0.get("SeriesDescription", "")))
    lat = {"LEFT": "L", "RIGHT": "R"}.get(str(h0.get("Laterality") or "").upper(), str(h0.get("Laterality") or "").upper())
    return dict(files=[str(f) for f in files], r=r, c=c, ipp=ipp, spacing=(ps[0], ps[1]), weighting=weighting(row),
                side=side_of(h0.get("Laterality"), centre[:, 0]), tagged=lat in ("L", "R"), cx=float(np.mean(centre[:, 0])))


CONFIDENT_CX_MM = 40.0  # geometry-only side calls closer to the midline than this follow the study's confident series


def study_sides(hs):
    """Final side per series: the Laterality tag if present; else the image-centre geometry if |centre x| >= 40 mm; else the
    majority side of the study's confident series (tagged or far from the midline), falling back to geometry. On training
    studies this cuts series mis-sided vs series_side_final from 283 to 241 (helper check, all 4,407 studies)."""
    conf = [h["side"] for h in hs.values() if h["tagged"] or abs(h["cx"]) >= CONFIDENT_CX_MM]
    major = max(set(conf), key=conf.count) if conf else None
    return {u: h["side"] if (h["tagged"] or abs(h["cx"]) >= CONFIDENT_CX_MM or major is None) else major for u, h in hs.items()}


def scan_study(study_dir, series_info):
    hs, fss = {}, {}
    for uid, fs in zip(series_info.SeriesInstanceUID, series_info.Fat_Suppression):
        try:  # an unreadable series is skipped (logged), the rest of the study is used
            hs[uid], fss[uid] = scan_series(Path(study_dir) / uid), int(fs)
        except Exception as e:  # noqa: BLE001
            print(f"series {uid}: header scan failed, skipped: {type(e).__name__}: {e}", flush=True)
    assert hs, "no readable series"
    rows = {}
    for uid, side in study_sides(hs).items():
        try:
            h = dict(hs[uid], side=side)
            plane, ops, pos = series_geometry(h["r"], h["c"], h["ipp"], side)
            rows[uid] = dict(h, plane=plane, ops=ops, pos=pos, fs=fss[uid], other_knee=False, n=n_stored(pos))
        except Exception as e:  # noqa: BLE001
            print(f"series {uid}: geometry failed, skipped: {type(e).__name__}: {e}", flush=True)
    assert rows, "no usable series"
    return pd.DataFrame.from_dict(rows, orient="index")


def series_volume(row, log=print, px_mm=0.5, size=320):
    """Volume of one series. A frame that fails to decode (e.g. truncated PixelData) is replaced by the nearest decoded frame
    of the series (logged): on Kaggle one unhandled file would void the whole submission."""
    imgs, bad = [], []
    for k, f in enumerate(row["files"]):
        try:
            d = _read(f)
            assert int(d.get("NumberOfFrames", 1) or 1) == 1, "multi-frame DICOM"
            imgs.append(d.pixel_array)
        except Exception as e:  # noqa: BLE001
            imgs.append(None)
            bad.append(k)
            log(f"decode failed {f}: {type(e).__name__}: {e}")
    ok = [k for k, im in enumerate(imgs) if im is not None]
    assert ok, "no decodable frame in series"
    for k in bad:
        imgs[k] = imgs[min(ok, key=lambda j: abs(j - k))]
    vol, _ = process_series(imgs, row["pos"], row["ops"], row["spacing"], px_mm=px_mm, size=size)
    return vol
