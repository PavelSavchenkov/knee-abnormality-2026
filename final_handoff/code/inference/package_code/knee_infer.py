"""Inference from raw DICOM for oct7_baseline models (experiments/v0/oct7_baseline). Same file runs locally and on Kaggle.

python knee_infer.py --root DIR/test_series --series-csv test_series.csv --studies test.csv --weights DIR --out pred.csv

Preprocessing mirrors data/build/build_data.py + common/meta.py + common/cache.py (fov160_px400) + data.py (eval windows):
- series: plane from ImageOrientationPatient, weighting from ScanningSequence / TR / TE / SeriesDescription, fat suppression from the
  host series csv; series kept in (SeriesNumber, SeriesInstanceUID) order; duplicate series (same middle-frame pixels) dropped
- knee side: study Laterality tag if the series agree on L or R, else majority of series sides (series tag, else sign of the image-centre x),
  tie -> first series; a series is oriented by its own L/R tag, else the study side
- frames sorted along the canonical slice direction, thin series (< 2 mm) averaged into ~3 mm slabs, resized to 0.4 mm/px (cv2 AREA when
  shrinking, else LINEAR), centre crop / pad to 400 px, uint8 with the series' 99.5th percentile, centre 140 mm crop
- eval windows: 96 (series, centre) pairs spread evenly over centres in the 5-95% span of every series
Model: backbone in fp16 (T4 has no bf16), head in fp32; probabilities averaged over the models in --weights.
Image-label head (config synovitis_ext_w = w, needed by 13-output models): column 12 is an extra head trained on image reads of synovitis;
per model, synovitis logit = (1 - w) * main + w * image-head synovitis logit (0.5 = predict.py --set ext_avg=Synovitis, 1 = the image head
alone), before the step below. A package with both kinds of models: Synovitis = mean of the 13-output models only, the other 11
findings = mean of the 12-output models only (e.g. the 0.940 submission's models + an image-label model: only synovitis changes).
Post-processing (config synovitis_effusion_w = w, default 0): per model, synovitis logit = (1 - w) * synovitis + w * effusion logit.
TTA (config "tta", default none): plain windows + `draws` copies with training-style random affine per window (seeded per study), probabilities
averaged per model before the synovitis step; findings in tta["identity_only"] (MCL) keep the plain view (experiments/v0/oct7_baseline/analyses/postproc/README.md).
Runtime (prompts/optimization.md): study threads read each study's files into one pinned buffer and parse headers in place (fast_read);
one thread per GPU uploads the buffer, rebuilds the cache.py volumes on the GPU (cv2-exact resize taps, <= 1 grey level from the CPU path
`study_series`, which is byte-identical to the training cache) and runs the models, backbones replayed as CUDA graphs.
A broken series is skipped; a study that fails gets the prior (mean OOF probability); both are logged. Nothing else is caught."""
import argparse
import collections
import functools
import hashlib
import io
import json
import math
import os
import queue
import re
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import SERIES_TYPES, KneeNet  # noqa: E402

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SYN, EFF = FINDINGS.index("Synovitis"), FINDINGS.index("Effusion")
PLANES = ["Sagittal", "Coronal", "Axial"]
TYPE_ID = {t: i for i, t in enumerate(SERIES_TYPES)}
FOV, PX, HI_PCT, SLAB_MM, THIN_MM = 160.0, 400, 99.5, 3.0, 2.0
MM_PER_PX = FOV / PX
LATERALITY = {"L": "L", "R": "R", "LEFT": "L", "RIGHT": "R", "B": "B"}
RAW = (ExplicitVRLittleEndian, ImplicitVRLittleEndian)
IO_THREADS = 128  # file readers per process; Kaggle /kaggle/input (NFS): 8 readers 113 MB/s, 32 409, 64 614, 128 658
_pool = None
STATS = {}  # last predict(): load_s (fork + models), first_item_s, total_s, steady_s_per_study, main loop: wait_data_s, wait_gpu_s


def text(ds, key):
    """DICOM value as build_data.as_str: multi-values joined by backslash, empty -> None."""
    v = ds.get(key)
    if v is None:
        return None
    s = "\\".join(map(str, v)) if isinstance(v, (list, tuple, pydicom.multival.MultiValue)) else str(v)
    return s.strip() or None


def number(ds, key):
    s = text(ds, key)
    try:
        return float(s)
    except (TypeError, ValueError):
        return math.nan


def weighting(seq, tr, te, desc):
    """common.meta.weighting (missing strings are 'nan' there)."""
    seq = str(seq)
    if "GR" in seq and "SE" not in seq:
        return "GRE"
    if not math.isnan(tr) and not math.isnan(te):
        return "T1" if tr < 800 else "PD" if te < 50 else "T2"
    d = str(desc).lower()
    if re.search(r"t1", d):
        return "T1"
    if re.search(r"pd|proton|dp\b|inter", d):
        return "PD"
    if re.search(r"t2|stir", d):
        return "T2"
    return "UNK"


def targets(plane, side):
    lat = np.array([1.0, 0, 0]) if side == "L" else np.array([-1.0, 0, 0])
    post, sup = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    return {"Sagittal": (-sup, post, lat), "Coronal": (-sup, lat, post), "Axial": (post, lat, sup)}[plane]


def inplane_ops(r, c, plane, side):
    down, right, _ = targets(plane, side)
    transpose = abs(r @ down) > abs(c @ down)
    row_dir, col_dir = (r, c) if transpose else (c, r)
    return bool(transpose), bool(row_dir @ down < 0), bool(col_dir @ right < 0)


def pixels(ds):
    if ds.file_meta.TransferSyntaxUID in RAW and ds.BitsAllocated == 16 and ds.get("NumberOfFrames", 1) in (1, None, ""):
        return np.frombuffer(ds.PixelData, "<i2" if ds.PixelRepresentation else "<u2").reshape(ds.Rows, ds.Columns)
    return ds.pixel_array                                    # compressed: needs a decoder plugin (none seen in train)


def floats(v):
    return None if v is None else tuple(float(x) for x in v)


def header(ds):
    """The header fields the pipeline uses, from a pydicom dataset (None where a tag is absent; ipp is required)."""
    return {"inst": int(ds.get("InstanceNumber") or 0), "ipp": floats(ds.ImagePositionPatient), "iop": floats(ds.get("ImageOrientationPatient")),
            "spacing": floats(ds.get("PixelSpacing")), "rows": ds.get("Rows"), "cols": ds.get("Columns"), "lat": text(ds, "Laterality"),
            "series_number": int(ds.get("SeriesNumber") or 0), "seq": text(ds, "ScanningSequence"), "tr": number(ds, "RepetitionTime"),
            "te": number(ds, "EchoTime"), "desc": text(ds, "SeriesDescription")}


FAST_TAGS = {0x00020010: "ts", 0x0008103E: "desc", 0x00180020: "seq", 0x00180080: "tr", 0x00180081: "te", 0x00200011: "series_number",
             0x00200013: "inst", 0x00200032: "ipp", 0x00200037: "iop", 0x00200060: "lat", 0x00280002: "spp", 0x00280008: "frames",
             0x00280010: "rows", 0x00280011: "cols", 0x00280030: "spacing", 0x00280100: "bits", 0x00280103: "pixrep"}
LONG_VR = frozenset([b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT", b"SV", b"UV"])
EXPLICIT_LE = "1.2.840.10008.1.2.1"


def fast_read(buf):
    """header(), pixel view and pixel byte offset without pydicom (buf: bytes or memoryview), for an uncompressed explicit-VR little-endian file with a flat dataset (all train files);
    None for anything else (the caller then uses pydicom). ~10x faster than pydicom.dcmread. speed/check_fast_read.py: equal to pydicom."""
    if bytes(buf[128:132]) != b"DICM":
        return None
    raw, pos, n, checked = {}, 132, len(buf), False
    while True:
        if pos + 8 > n:
            return None
        g, e = struct.unpack_from("<HH", buf, pos)
        if g != 2 and not checked:                           # leaving the file meta group: the dataset must be explicit VR LE
            if raw.get("ts", b"").decode("latin-1").strip(" \x00") != EXPLICIT_LE:
                return None
            checked = True
        vr = bytes(buf[pos + 4:pos + 6])
        if vr in LONG_VR:
            ln, hdr = struct.unpack_from("<I", buf, pos + 8)[0], 12
        else:
            ln, hdr = struct.unpack_from("<H", buf, pos + 6)[0], 8
        if g == 0x7FE0 and e == 0x0010:
            break
        if ln == 0xFFFFFFFF or vr == b"SQ" or not (vr.isalpha() and vr.isupper()):
            return None
        key = FAST_TAGS.get(g << 16 | e)
        if key:
            raw[key] = bytes(buf[pos + hdr:pos + hdr + ln])
        pos += hdr + ln
    us = {k: struct.unpack("<H", raw[k])[0] for k in ("rows", "cols", "bits", "pixrep", "spp") if len(raw.get(k, b"")) == 2}
    txt = {k: raw[k].decode("latin-1").strip(" \x00") or None for k in raw if k not in us}
    if us.get("bits") != 16 or us.get("spp", 1) != 1 or txt.get("frames") not in (None, "1") or "pixrep" not in us \
            or ln != us["rows"] * us["cols"] * 2 or pos + hdr + ln > n:
        return None
    num = lambda k: float(txt[k]) if txt.get(k) else math.nan
    vec = lambda k: tuple(float(x) for x in txt[k].split("\\")) if txt.get(k) else None
    h = {"inst": int(float(txt["inst"])) if txt.get("inst") else 0, "ipp": vec("ipp"), "iop": vec("iop"), "spacing": vec("spacing"),
         "rows": us["rows"], "cols": us["cols"], "lat": txt.get("lat"), "series_number": int(float(txt["series_number"])) if txt.get("series_number") else 0,
         "seq": txt.get("seq"), "tr": num("tr"), "te": num("te"), "desc": txt.get("desc")}
    if h["ipp"] is None:
        raise ValueError("no ImagePositionPatient")
    img = np.frombuffer(buf, "<i2" if us["pixrep"] else "<u2", us["rows"] * us["cols"], pos + hdr).reshape(us["rows"], us["cols"])
    return h, img, pos + hdr


def io_pool():
    """The process's file-reader threads. Keyed by pid: a forked child inherits the pool object but not its threads (map would hang)."""
    global _pool
    if _pool is None or _pool[0] != os.getpid():
        _pool = (os.getpid(), ThreadPoolExecutor(IO_THREADS))
    return _pool[1]


def read_bytes(paths):
    """File contents, read by the IO threads."""
    return list(io_pool().map(lambda p: open(p, "rb").read(), paths))


def read_into(jobs):
    """Read files into the given memoryview slices (exact file sizes) with the IO thread pool."""
    def one(job):
        path, view = job
        with open(path, "rb", buffering=0) as f:
            got = 0
            while got < len(view):
                k = f.readinto(view[got:])
                if not k:
                    raise EOFError(path)
                got += k
    list(io_pool().map(one, jobs))                           # list(): wait for all and re-raise the first error


def read_frame(b):
    """(header, pixels, pixel byte offset in b or None when pydicom decoded it)."""
    r = fast_read(b)
    if r is None:
        ds = pydicom.dcmread(io.BytesIO(bytes(b)))
        r = header(ds), pixels(ds), None
    return r


def read_series(files, blobs, starts=None):
    """All frames of one series, sorted by InstanceNumber (then file name): headers, pixel arrays, pixel offsets in the study buffer
    (starts = file offsets in it; None entries for frames that are not views of it)."""
    frames = [read_frame(b) for b in blobs]
    order = sorted(range(len(frames)), key=lambda i: (frames[i][0]["inst"], files[i]))
    offs = [None if starts is None or frames[i][2] is None else starts[i] + frames[i][2] for i in order]
    return [frames[i][0] for i in order], [frames[i][1] for i in order], offs


def series_info(uid, hs, fs):
    h = hs[0]
    r, c = np.array(h["iop"][:3]), np.array(h["iop"][3:])
    sp_row, sp_col = h["spacing"]
    plane = PLANES[int(np.argmax(np.abs(np.cross(r, c))))]
    w = weighting(h["seq"], h["tr"], h["te"], h["desc"])
    tag = LATERALITY.get(h["lat"] or "", None)
    x0 = h["ipp"][0] + r[0] * (h["cols"] - 1) / 2 * sp_col + c[0] * (h["rows"] - 1) / 2 * sp_row
    return {"uid": uid, "number": h["series_number"], "plane": plane, "type": TYPE_ID[f"{plane[:3].upper()}_{w}_{'FS' if fs else 'NOFS'}"],
            "tag": tag, "geo": "L" if x0 > 0 else "R", "r": r, "c": c, "sp_row": sp_row, "sp_col": sp_col}


def study_side(infos):
    tags = {i["tag"] for i in infos if i["tag"] is not None}
    if len(tags) == 1 and tags <= {"L", "R"}:
        return tags.pop()
    sides = [i["tag"] if i["tag"] in ("L", "R") else i["geo"] for i in infos]
    n_l, n_r = sides.count("L"), sides.count("R")
    return "L" if n_l > n_r else "R" if n_r > n_l else sides[0]


@functools.lru_cache(maxsize=None)
def resize_matrix(n_in, n_out, area):
    """(n_out, n_in) weights of cv2.resize along one axis, from opencv resize.cpp: INTER_AREA (computeResizeAreaTab) or INTER_LINEAR.
    cv2.resize(img) == resize_matrix(h, nh, area) @ img @ resize_matrix(w, nw, area).T up to float rounding (speed/check_resize_matrix.py)."""
    scale = 1.0 / (n_out / n_in)
    m = np.zeros((n_out, n_in))
    for d in range(n_out):
        if area:
            fs1 = d * scale
            fs2 = fs1 + scale
            cell = min(scale, n_in - fs1)
            s2 = min(math.floor(fs2), n_in - 1)
            s1 = min(math.ceil(fs1), s2)
            if s1 - fs1 > 1e-3:
                m[d, s1 - 1] += np.float32((s1 - fs1) / cell)
            m[d, s1:s2] += np.float32(1.0 / cell)
            if fs2 - s2 > 1e-3:
                m[d, s2] += np.float32(min(min(fs2 - s2, 1.0), cell) / cell)
        else:
            f = np.float32((d + 0.5) * scale - 0.5)
            i = math.floor(f)
            f = np.float32(f - i)
            if i < 0:
                f, i = np.float32(0), 0
            if i >= n_in - 1:
                f, i = np.float32(0), n_in - 1
            m[d, i] += np.float32(1) - f
            if f:
                m[d, i + 1] += f
    return m


def canonical_order(info, hs, side):
    pos = np.array([h["ipp"] for h in hs]) @ targets(info["plane"], side)[2]
    order = np.argsort(pos, kind="stable")
    return order, pos[order]


def volume(info, imgs, side, order, pos):
    """Canonical uint8 volume (n, PX, PX) and slab positions (mm, relative to the series centre)."""
    ops = inplane_ops(info["r"], info["c"], info["plane"], side)
    spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.0
    k = max(1, int(round(SLAB_MM / spacing))) if 0 < spacing < THIN_MM else 1
    n = len(pos) // k
    transpose, flip_r, flip_c = ops
    sp_r, sp_c = (info["sp_col"], info["sp_row"]) if transpose else (info["sp_row"], info["sp_col"])
    out = np.zeros((n * k, PX, PX), np.float32)
    for j, i in enumerate(order[: n * k]):
        a = imgs[i]
        if transpose:
            a = a.T
        if flip_r:
            a = a[::-1]
        if flip_c:
            a = a[:, ::-1]
        a = np.array(a, dtype=np.float32, order="C")       # one copy: oriented view -> contiguous float32
        nh, nw = max(1, int(round(a.shape[0] * sp_r / MM_PER_PX))), max(1, int(round(a.shape[1] * sp_c / MM_PER_PX)))
        a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA if (nh < a.shape[0] and nw < a.shape[1]) else cv2.INTER_LINEAR)
        y0, x0 = (nh - PX) // 2, (nw - PX) // 2
        sy, sx, dy, dx = max(y0, 0), max(x0, 0), max(-y0, 0), max(-x0, 0)
        hh, ww = min(PX - dy, nh - sy), min(PX - dx, nw - sx)
        out[j, dy:dy + hh, dx:dx + ww] = a[sy:sy + hh, sx:sx + ww]
    vol = out.reshape(n, k, PX, PX).mean(1) if k > 1 else out
    hi = float(np.percentile(vol[:, ::4, ::4], HI_PCT))
    hi = hi if hi > 0 else 1.0
    v = np.clip(vol, 0, hi)                                 # in place, same float32 ops as cache.py; values end in [0.5, 255.5]
    v /= hi
    v *= 255
    v += 0.5
    u8 = v.astype(np.uint8)
    slab = pos[: n * k].reshape(n, k).mean(1)
    return u8, (slab - (slab.min() + slab.max()) / 2).astype(np.float32)


def read_study(study_dir, fs_of, buffered=False, pin=False, times=None):
    """Parsed series of one study in (SeriesNumber, uid) order with the study knee side: list of (info, headers, imgs, offs), side,
    skipped, buf. buffered: all files are read into one uint8 tensor buf (pinned if pin) with IO threads, pixels are views of it and offs
    their byte offsets (the GPU path sends buf as is); else files are read as bytes (CPU reference path) and buf is None."""
    times = {} if times is None else times
    t = time.time()
    dirs = sorted((e for e in os.scandir(study_dir) if e.is_dir()), key=lambda e: e.name)
    files = {d.name: sorted((e.path, e.stat().st_size) for e in os.scandir(d.path) if e.is_file()) for d in dirs}
    flat = [f for d in dirs for f in files[d.name]]
    buf = None
    if buffered:
        starts = np.cumsum([0] + [size for _, size in flat])
        buf = torch.empty(int(starts[-1]), dtype=torch.uint8, pin_memory=pin)
        mv = memoryview(buf.numpy())
        read_into([(p, mv[a:b]) for (p, _), a, b in zip(flat, starts[:-1], starts[1:])])
        blobs = iter([mv[a:b] for a, b in zip(starts[:-1], starts[1:])])
        file_starts = iter(starts[:-1].tolist())
    else:
        blobs = iter(read_bytes([p for p, _ in flat]))
    times["read"] = time.time() - t
    t = time.time()
    raw, skipped = [], []
    for d in dirs:
        names = [p for p, _ in files[d.name]]
        b = [next(blobs) for _ in names]
        st = [next(file_starts) for _ in names] if buffered else None
        try:                                                 # a broken series is skipped, the study keeps the others
            hs, imgs, offs = read_series(names, b, st)
            raw.append((series_info(d.name, hs, fs_of.get(d.name, 0)), hs, imgs, offs))
        except Exception as e:
            skipped.append(f"{d.name}: {type(e).__name__}: {e}")
    raw.sort(key=lambda r: (r[0]["number"], r[0]["uid"]))
    times["parse"] = time.time() - t
    return raw, study_side([r[0] for r in raw]), skipped, buf


def kept_series(raw, side, skipped):
    """Yield (raw entry, series side, canonical order, positions) of the non-duplicate series; failures go to skipped."""
    seen = set()
    for r in raw:
        info, hs, imgs = r[:3]
        try:
            s_side = info["tag"] if info["tag"] in ("L", "R") else side
            order, pos = canonical_order(info, hs, s_side)
            mid = imgs[order[len(order) // 2]]
            h = hashlib.md5(mid.dtype.str.encode() + mid.tobytes()).hexdigest()   # = cache.py's md5 of float32 pixels for equal dtypes
            if h in seen:
                continue
            seen.add(h)
            yield r, s_side, order, pos
        except Exception as e:
            skipped.append(f"{info['uid']}: {type(e).__name__}: {e}")


def study_series(study_dir, fs_of):
    """CPU reference (byte-identical to the training cache, check_preprocess.py): list of dicts uid, type, plane, side, vol uint8 (n, PX, PX),
    pos mm; and skipped series."""
    raw, side, skipped, _ = read_study(study_dir, fs_of)
    out = []
    for (info, hs, imgs, _), s_side, order, pos in kept_series(raw, side, skipped):
        try:
            vol, pos = volume(info, imgs, s_side, order, pos)
            out.append({"uid": info["uid"], "type": info["type"], "plane": info["plane"], "side": s_side, "vol": vol, "pos": pos})
        except Exception as e:
            skipped.append(f"{info['uid']}: {type(e).__name__}: {e}")
    return out, skipped


def series_plan(info, imgs, offs, side, order, pos):
    """GPU input of one series: plan for gpu_volume, slab positions (mm, relative to the series centre). The plan holds the byte offsets of
    the kept frames (canonical order) in the study buffer, or, if some frame is not a 16-bit view of it (decoded by pydicom), the stacked
    frames ("frames", int16 bits or float32)."""
    transpose, flip_r, flip_c = inplane_ops(info["r"], info["c"], info["plane"], side)
    spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 0.0
    k = max(1, int(round(SLAB_MM / spacing))) if 0 < spacing < THIN_MM else 1
    n = len(pos) // k
    assert n > 0, "no full slab"
    rows, cols = imgs[0].shape
    assert all(im.shape == (rows, cols) for im in imgs), "frames differ in size"
    h, w = (cols, rows) if transpose else (rows, cols)
    sp_r, sp_c = (info["sp_col"], info["sp_row"]) if transpose else (info["sp_row"], info["sp_col"])
    nh, nw = max(1, int(round(h * sp_r / MM_PER_PX))), max(1, int(round(w * sp_c / MM_PER_PX)))
    slab = pos[: n * k].reshape(n, k).mean(1)
    plan = {"unsigned": imgs[0].dtype == np.uint16, "transpose": transpose, "flip_r": flip_r, "flip_c": flip_c, "nh": nh, "nw": nw,
            "area": nh < h and nw < w, "k": k, "n": n, "rows": rows, "cols": cols}
    kept = order[: n * k]
    if all(offs[i] is not None for i in kept) and imgs[0].dtype in (np.uint16, np.int16):
        plan["offsets"] = [offs[i] for i in kept]
    else:
        bits16 = imgs[0].dtype in (np.uint16, np.int16)
        plan["frames"] = torch.from_numpy(np.stack([imgs[i].view(np.int16) if bits16 else imgs[i].astype(np.float32) for i in kept]))
    return plan, (slab - (slab.min() + slab.max()) / 2).astype(np.float32)


def window_rows(ns, k, span=(0.05, 0.95)):
    """data.StudyDataset eval windows over series with ns slices: unique slice rows (over the concatenated series), idx (k, 3) into them,
    mask, and (series, centre) of each window. Padded windows (mask False) point at row 0."""
    pairs = []
    for si, n in enumerate(ns):
        if n < 3:
            continue
        lo, hi = max(1, int(np.floor(span[0] * n))), min(n - 2, int(np.ceil(span[1] * n)) - 1)
        if hi < lo:
            lo, hi = 1, n - 2
        pairs += [(si, c) for c in range(lo, hi + 1)]
    assert pairs, "no series with >= 3 slices"
    sel = np.unique(np.linspace(0, len(pairs) - 1, min(k, len(pairs))).round().astype(int))
    start = np.cumsum([0] + list(ns))
    rows = np.array([[start[pairs[p][0]] + pairs[p][1] + d for d in (-1, 0, 1)] for p in sel])
    uniq, inv = np.unique(rows, return_inverse=True)
    idx, mask = np.zeros((k, 3), np.int64), np.zeros(k, bool)
    idx[: len(sel)] = inv.reshape(-1, 3)
    mask[: len(sel)] = True
    return uniq, idx, mask, [pairs[p] for p in sel]


def windows(series, k, crop_px, span=(0.05, 0.95)):
    """CPU reference windows from study_series: slices (u, crop, crop) uint8, idx (k, 3) rows of slices per window, mask, type, pos."""
    uniq, idx, mask, wins = window_rows([len(s["vol"]) for s in series], k, span)
    start = np.cumsum([0] + [len(s["vol"]) for s in series])
    o = (PX - crop_px) // 2
    si = np.searchsorted(start, uniq, side="right") - 1
    slices = np.stack([series[a]["vol"][r - start[a], o:o + crop_px, o:o + crop_px] for a, r in zip(si, uniq)])
    typ, pos = np.zeros(k, np.int64), np.zeros(k, np.float32)
    for j, (a, c) in enumerate(wins):
        typ[j], pos[j] = series[a]["type"], series[a]["pos"][c]
    return slices, idx, mask, typ, pos


def study_item(study_dir, fs_of, k, pin=False, times=None):
    """GPU-path input of one study: the study buffer, plans of the kept series, unique slice rows, window idx / mask / type / pos; skipped."""
    times = {} if times is None else times
    raw, side, skipped, buf = read_study(study_dir, fs_of, buffered=True, pin=pin, times=times)
    t = time.time()
    plans, poss, types = [], [], []
    for (info, hs, imgs, offs), s_side, order, pos in kept_series(raw, side, skipped):
        try:
            plan, p = series_plan(info, imgs, offs, s_side, order, pos)
        except Exception as e:
            skipped.append(f"{info['uid']}: {type(e).__name__}: {e}")
            continue
        plans.append(plan), poss.append(p), types.append(info["type"])
    uniq, idx, mask, wins = window_rows([p["n"] for p in plans], k)
    typ, pos = np.zeros(k, np.int64), np.zeros(k, np.float32)
    for j, (a, c) in enumerate(wins):
        typ[j], pos[j] = types[a], poss[a][c]
    small = {n: torch.from_numpy(v) for n, v in (("uniq", uniq), ("idx", idx), ("mask", mask), ("type", typ), ("pos", pos))}
    if pin:
        small = {n: v.pin_memory() for n, v in small.items()}
    times["plan"] = time.time() - t
    return {"buf": buf, "plans": plans} | small, skipped


@functools.lru_cache(maxsize=None)
def resize_taps(n_in, n_out, area, dev):
    """resize_matrix as taps (each output uses <= 4 neighbouring inputs): source indices and weights (T, n_out), sources in increasing
    order as cv2 sums them; unused taps have weight 0. ~100x fewer FLOPs than the dense matmul."""
    m = resize_matrix(n_in, n_out, area)
    nz = [np.flatnonzero(r) for r in m]
    idx, w = np.zeros((max(map(len, nz)), n_out), np.int64), np.zeros((max(map(len, nz)), n_out), np.float32)
    for d, z in enumerate(nz):
        idx[: len(z), d], w[: len(z), d] = z, m[d, z]
    return torch.from_numpy(idx).pin_memory().to(dev, non_blocking=True), torch.from_numpy(w).pin_memory().to(dev, non_blocking=True)


def resize_gpu(x, nh, nw, area):
    """cv2.resize of every frame of x (N, H, W) float32 to (nh, nw): horizontal pass, then vertical, as cv2."""
    idx, w = resize_taps(x.shape[2], nw, area, x.device)
    x = sum(x[:, :, idx[t]] * w[t] for t in range(len(w)))
    idx, w = resize_taps(x.shape[1], nh, area, x.device)
    return sum(x[:, idx[t], :] * w[t][:, None] for t in range(len(w)))


def gpu_volume(frames, plan, dev, crop_px):
    """volume() on the GPU: orientation, resize (cv2 weights as taps), canvas, slabs, 99.5th percentile, uint8; centre crop only."""
    x = frames
    if x.dtype == torch.int16:
        x = x.to(torch.int32)
        x = (x & 0xFFFF if plan["unsigned"] else x).float()
    if plan["transpose"]:
        x = x.transpose(1, 2)
    dims = [d for d, f in ((1, plan["flip_r"]), (2, plan["flip_c"])) if f]
    if dims:
        x = x.flip(dims)
    nh, nw, area = plan["nh"], plan["nw"], plan["area"]
    x = resize_gpu(x, nh, nw, area)
    y0, x0 = (nh - PX) // 2, (nw - PX) // 2
    sy, sx, dy, dx = max(y0, 0), max(x0, 0), max(-y0, 0), max(-x0, 0)
    hh, ww = min(PX - dy, nh - sy), min(PX - dx, nw - sx)
    vol = x.new_zeros((len(x), PX, PX))
    vol[:, dy:dy + hh, dx:dx + ww] = x[:, sy:sy + hh, sx:sx + ww]
    if plan["k"] > 1:
        vol = vol.view(plan["n"], plan["k"], PX, PX).mean(1)
    hi = torch.quantile(vol[:, ::4, ::4].reshape(-1), HI_PCT / 100)
    hi = torch.where(hi > 0, hi, torch.ones_like(hi))
    o = (PX - crop_px) // 2
    v = torch.minimum(vol[:, o:o + crop_px, o:o + crop_px].clamp(min=0), hi)
    return (v / hi * 255 + 0.5).to(torch.uint8)


def gpu_frames(buf, plan, dev):
    """Kept frames (n * k, rows, cols) of one series on dev: int16 views of the study buffer at the plan's offsets, or the stacked frames."""
    if "frames" in plan:
        return plan["frames"].to(dev, non_blocking=True)
    nb = plan["rows"] * plan["cols"] * 2
    return torch.stack([(buf[o:o + nb] if o % 2 == 0 else buf[o:o + nb].clone()).view(torch.int16) for o in plan["offsets"]]) \
        .view(-1, plan["rows"], plan["cols"])


def gpu_slices(item, dev, crop_px):
    """Unique window slices (u, crop, crop) uint8 of one study, on dev."""
    if not item["plans"]:
        return torch.zeros((len(item["uniq"]), crop_px, crop_px), dtype=torch.uint8, device=dev)
    buf = item["buf"].to(dev, non_blocking=True)
    vols = torch.cat([gpu_volume(gpu_frames(buf, p, dev), p, dev, crop_px) for p in item["plans"]])
    return vols[item["uniq"].to(dev, non_blocking=True)]


def collate(items, dev):
    """Batch on dev: window rows shifted to index the concatenated window slices. The items' tensors are pinned, so every copy is async."""
    off = np.cumsum([0] + [len(it["uniq"]) for it in items])[:-1]
    gpu = lambda t: t.to(dev, non_blocking=True)
    return {"idx": torch.stack([gpu(it["idx"]) + int(o) for it, o in zip(items, off)]),
            **{n: torch.stack([gpu(it[n]) for it in items]) for n in ("mask", "type", "pos")},
            "ok": [it["ok"] for it in items], "err": [it["err"] for it in items], "i": [it["i"] for it in items], "sec": [it["sec"] for it in items],
            "sid": [it["sid"] for it in items]}


def load_study(root, sid, fs_of, k, pin):
    """study_item + bookkeeping; a study that fails becomes an empty item (prior) instead of an exception."""
    t, times = time.time(), {}
    try:
        item, skipped = study_item(os.path.join(root, sid), fs_of, k, pin, times)
        ok, err = True, "; ".join(skipped)
    except Exception as e:                                   # one bad study must not cost the whole submission
        item = {"buf": None, "plans": [], "uniq": torch.zeros(1, dtype=torch.int64), "idx": torch.zeros((k, 3), dtype=torch.int64),
                "mask": torch.ones(k, dtype=torch.bool), "type": torch.zeros(k, dtype=torch.int64), "pos": torch.zeros(k)}
        ok, err = False, f"{type(e).__name__}: {e}"
    return item | {"ok": ok, "err": err, "sec": time.time() - t, "times": times}


class GraphedBackbone(torch.nn.Module):
    """A backbone replayed as one CUDA graph for the full-batch input shape (eager for any other shape, e.g. the last batch): one Python
    call instead of ~250 op dispatches per forward, so a GPU thread hardly needs the GIL. Graphs of one device share a memory pool and one
    capture stream `side` of that device (the default capture stream belongs to the first device used: capturing cuda:1 work on it fails;
    a stream per graph keeps the pool from being reused: 5 graphs x 2.7 GB). They replay one after another; each output is cloned."""

    def __init__(self, backbone, shape, dev, pool, side):
        super().__init__()
        self.eager, self.pretrained_cfg = backbone, backbone.pretrained_cfg
        with torch.cuda.device(dev), torch.no_grad():
            self.inp = torch.zeros(shape, dtype=torch.half, device=dev).contiguous(memory_format=torch.channels_last)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):                           # warm-up outside the graph (cudnn handles, workspaces)
                    backbone(self.inp)
            torch.cuda.current_stream(dev).wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=pool, stream=side, capture_error_mode="thread_local"):
                self.out = backbone(self.inp)

    def forward(self, x):
        if x.shape != self.inp.shape:
            return self.eager(x)
        self.inp.copy_(x)
        self.graph.replay()
        return self.out.clone()


def load_models(weights_dir, devices, graph_batch=0):
    """models[i][j] = fold model i on devices[j]: backbone fp16 channels_last, head fp32. graph_batch > 0: backbones as CUDA graphs for
    input (graph_batch, 3, img, img)."""
    cfg = json.load(open(Path(weights_dir) / "config.json"))
    pools = {dev: torch.cuda.graph_pool_handle() for dev in devices} if graph_batch else {}
    sides = {dev: torch.cuda.Stream(dev) for dev in devices} if graph_batch else {}
    models = []
    for f in cfg["weights"]:
        per_dev = []
        sd = torch.load(Path(weights_dir) / f, map_location="cpu")
        for dev in devices:
            m = KneeNet(cfg["backbone"], n_out=sd["cls_b"].shape[0], pretrained=False, hidden=cfg["hidden"], drop=cfg["drop"],
                        use_type=cfg["use_type"], use_pos=cfg["use_pos"], head=cfg["head"])
            m.load_state_dict(sd)
            m.eval().to(dev)
            m.backbone.half().to(memory_format=torch.channels_last)
            if graph_batch:
                m.backbone = GraphedBackbone(m.backbone, (graph_batch, 3, cfg["img"], cfg["img"]), dev, pools[dev], sides[dev])
            per_dev.append(m)
        models.append(per_dev)
    return cfg, models


@functools.lru_cache(maxsize=None)
def norm_stats(mean, std, dev):
    """(1, 3, 1, 1) mean / std on dev, made once: torch.tensor(list, device=dev) is a blocking copy that waits for the GPU queue."""
    return torch.tensor(mean, device=dev).view(1, 3, 1, 1), torch.tensor(std, device=dev).view(1, 3, 1, 1)


def tta_thetas(sids, k, aug):
    """(draws, len(sids) * k, 2, 3) affine matrices drawn as data.gpu_augment draws them in training (rotation, zoom, shift independently
    per window). One CPU RNG per study, seeded by its StudyInstanceUID: the draws do not depend on the GPU model, the batch or the order."""
    th = []
    for sid in sids:
        rng = np.random.default_rng([aug["seed"], zlib.crc32(sid.encode())])
        shape = (aug["draws"], k)
        a = rng.uniform(-1, 1, shape) * aug["rot"] * math.pi / 180
        s = 1 + rng.uniform(-1, 1, shape) * aug["scale"]
        t = rng.uniform(-1, 1, shape + (2,)) * aug["shift"] * 2
        th.append(np.stack([np.stack([np.cos(a) / s, -np.sin(a) / s, t[..., 0]], -1), np.stack([np.sin(a) / s, np.cos(a) / s, t[..., 1]], -1)], -2))
    return np.concatenate(th, 1).astype(np.float32)


@torch.no_grad()
def forward(models_dev, b, dev, img, syn_w=0.0, tta=None, ext_w=0.0):
    """Mean probability of the models. tta (config "tta"): views = the plain windows + `draws` training-style random affine copies;
    per model the probabilities are averaged over views (findings in identity_only keep the plain view). syn_w > 0: each model's synovitis
    logit -> (1 - syn_w) * synovitis + syn_w * effusion logit, applied after the view average. A 13-output model (needs ext_w > 0): synovitis
    logit -> (1 - ext_w) * main + ext_w * image-head (column 12) synovitis logit first."""
    c = models_dev[0].backbone.pretrained_cfg
    mean, std = norm_stats(tuple(c["mean"]), tuple(c["std"]), dev)
    s = b["slices"][:, None].float().div_(255)
    s = F.interpolate(s, size=(img, img), mode="bilinear", antialias=True, align_corners=False)[:, 0]   # per channel, as in data.gpu_prep
    x = s[b["idx"].flatten(0, 1)]  # (b * k, 3, img, img) in [0, 1]
    views = [x]
    if tta:
        th = torch.from_numpy(tta_thetas(b["sid"], b["mask"].shape[1], tta)).pin_memory().to(dev, non_blocking=True)
        views += [F.grid_sample(x, F.affine_grid(t, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
                  for t in th]                                # zero padding on [0, 1] images, as in training
    mask, typ, pos = b["mask"], b["type"], b["pos"]
    probs = [0.0] * len(models_dev)
    plain = [None] * len(models_dev)
    for v in views:
        xv = ((v - mean) / std).half().contiguous(memory_format=torch.channels_last)
        for j, m in enumerate(models_dev):
            p = torch.sigmoid(m(xv, mask, typ, pos).float())
            probs[j] = probs[j] + p / len(views)
            plain[j] = p if plain[j] is None else plain[j]
    out, ext = [], []
    for p, p0 in zip(probs, plain):
        for f in (tta or {}).get("identity_only", []):
            p[:, FINDINGS.index(f)] = p0[:, FINDINGS.index(f)]
        z = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        ext.append(z.shape[1] == 13)
        if ext[-1]:
            assert ext_w > 0, "13-output model without synovitis_ext_w in config.json"
            z[:, SYN] = (1 - ext_w) * z[:, SYN] + ext_w * z[:, 12]
        z = z[:, :12]
        if syn_w:
            z[:, SYN] = (1 - syn_w) * z[:, SYN] + syn_w * z[:, EFF]
        out.append(torch.sigmoid(z))
    if all(ext) or not any(ext):
        return torch.stack(out).mean(0)
    res = torch.stack([o for o, e in zip(out, ext) if not e]).mean(0)           # mixed package: 11 findings from the 12-output models
    res[:, SYN] = torch.stack([o[:, SYN] for o, e in zip(out, ext) if e]).mean(0)  # synovitis from the image-head models
    return res


def gpu_loop(q, models_dev, dev, cfg, crop_px, out, errors):
    """One thread per GPU: batches (lists of studies) from q -> window slices on dev -> models -> out rows. A single thread driving
    2 GPUs serialises them: ~1,500 kernel launches per batch fill the launch queue and block it on one GPU while the other idles.
    PyTorch ops release the GIL, so the threads overlap."""
    st = {"issue_prep_s": 0.0, "issue_model_s": 0.0, "wait_s": 0.0, "idle_s": 0.0}
    ev = []                                                  # CUDA events per batch: GPU time of preprocessing and of the models
    try:
        prev = None                                          # one batch in flight: launch batch n+1 before waiting for batch n
        while True:
            t = time.time()
            items = q.get()
            st["idle_s"] += time.time() - t
            if items is None:
                break
            e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            t = time.time()
            e0.record(torch.cuda.current_stream(dev))
            slices = torch.cat([gpu_slices(it, dev, crop_px) for it in items])
            for it in items:
                it["buf"] = None                             # the pinned block goes back to the caching host allocator
            b = collate(items, dev) | {"slices": slices}
            e1.record(torch.cuda.current_stream(dev))
            st["issue_prep_s"] += time.time() - t
            t = time.time()
            cur = (b["i"], forward(models_dev, b, dev, cfg["img"], cfg.get("synovitis_effusion_w", 0.0), cfg.get("tta"), cfg.get("synovitis_ext_w", 0.0)))
            e2.record(torch.cuda.current_stream(dev))
            st["issue_model_s"] += time.time() - t
            ev.append((e0, e1, e2))
            t = time.time()
            if prev is not None:
                out[prev[0]] = prev[1].cpu().numpy()
            st["wait_s"] += time.time() - t
            prev = cur
        if prev is not None:
            out[prev[0]] = prev[1].cpu().numpy()
        torch.cuda.synchronize(dev)
        st["gpu_prep_s"] = sum(a.elapsed_time(b) for a, b, _ in ev) / 1000
        st["gpu_model_s"] = sum(b.elapsed_time(c) for _, b, c in ev) / 1000
        STATS[dev] = st
    except BaseException as e:                               # a bug, not bad data: re-raised by predict
        errors.append(e)
        while q.get() is not None:                           # keep draining so the producer never blocks
            pass


def predict(root, studies, fs_of, weights_dir, threads=None, ahead=None, bs=4, gpus=None, log=print, pin=True, graphs=True):
    """Probabilities (DataFrame studies x FINDINGS), failed studies, per-study load seconds. One process, no worker processes:
    Models load first (CUDA graphs are captured with nothing else on the GPU); then `threads` study threads (default 2 x CPUs) each read a study's files into one pinned buffer (IO_THREADS file readers shared) and parse
    the headers in place; at most `ahead` studies (default 3 x threads) are loaded ahead of the GPUs. The main thread groups studies into
    batches of bs and hands them to one gpu_loop thread per GPU (queues of 1 batch), which builds the window slices and runs the models."""
    torch.backends.cudnn.benchmark = False                   # T4: autotune costs ~35 s per GPU and batch shape and is 5% slower after
    t_start = time.time()
    threads = 2 * len(os.sched_getaffinity(0)) if threads is None else threads
    ahead = 3 * threads if ahead is None else ahead
    torch.cuda.init()
    cfg = json.load(open(Path(weights_dir) / "config.json"))
    crop_px = min(PX, int(round(cfg["crop_mm"] / MM_PER_PX)))
    devices = [f"cuda:{g}" for g in (gpus if gpus is not None else range(torch.cuda.device_count()))]
    cfg, models = load_models(weights_dir, devices, bs * cfg["k_eval"] if graphs else 0)   # before the loader: no CUDA call (e.g. pinned
    loader = ThreadPoolExecutor(threads)                                                  # allocation) may run while a graph is captured
    futs = collections.deque(loader.submit(load_study, root, sid, fs_of, cfg["k_eval"], pin) for sid in studies[:ahead])
    STATS.clear()
    STATS.update(load_s=time.time() - t_start, n=len(studies))
    out = np.full((len(studies), len(FINDINGS)), np.nan, np.float32)
    failed, secs, errors, t0 = {}, np.zeros(len(studies)), [], time.time()
    queues = [queue.Queue(maxsize=1) for _ in devices]
    gpu_threads = [threading.Thread(target=gpu_loop, args=(q, [m[j] for m in models], d, cfg, crop_px, out, errors), daemon=True)
                   for j, (q, d) in enumerate(zip(queues, devices))]
    for th in gpu_threads:
        th.start()
    items, nb, wait_data, wait_gpu = [], 0, 0.0, 0.0
    for n in range(len(studies)):
        t = time.time()
        it = futs.popleft().result() | {"i": n, "sid": studies[n]}
        wait_data += time.time() - t
        if n + ahead < len(studies):
            futs.append(loader.submit(load_study, root, studies[n + ahead], fs_of, cfg["k_eval"], pin))
        if n == 0:
            STATS["first_item_s"] = time.time() - t_start
        items.append(it)
        secs[n] = it["sec"]
        for k, v in it["times"].items():
            STATS[f"load_{k}_s"] = STATS.get(f"load_{k}_s", 0.0) + v
        if not it["ok"]:
            failed[studies[n]] = it["err"]
            log(f"FAILED {studies[n]}: {it['err']}")
        elif it["err"]:
            log(f"skipped series in {studies[n]}: {it['err']}")
        if len(items) == bs or n == len(studies) - 1:
            tg = time.time()
            queues[nb % len(devices)].put(items)
            wait_gpu += time.time() - tg
            items, nb = [], nb + 1
        if errors:
            raise errors[0]
        if (n + 1) % 200 == 0:
            log(f"{n + 1}/{len(studies)} studies, {time.time() - t0:.0f}s")
    loader.shutdown()
    for q in queues:
        q.put(None)
    for th in gpu_threads:
        th.join()
    if errors:
        raise errors[0]
    STATS.update(wait_data_s=wait_data, wait_gpu_s=wait_gpu)
    STATS.update(total_s=time.time() - t_start, steady_s_per_study=(time.time() - t_start - STATS.get("first_item_s", 0)) / max(1, len(studies) - 1))
    bad = np.isnan(out).any(1)
    bad[[studies.index(sid) for sid in failed]] = True
    out[bad] = np.array([cfg["prior"][f] for f in FINDINGS], np.float32)   # in numpy: pandas 3 refuses float64 into a float32 frame
    pred = pd.DataFrame(out, index=pd.Index(studies, name="StudyInstanceUID"), columns=FINDINGS)
    return pred, failed, pd.Series(secs, index=pred.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--studies", required=True, help="csv with StudyInstanceUID")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    studies = pd.read_csv(a.studies).StudyInstanceUID.tolist()[: a.limit or None]
    sc = pd.read_csv(a.series_csv)
    fs_of = dict(zip(sc.SeriesInstanceUID, sc.Fat_Suppression.fillna(0).astype(int)))
    t = time.time()
    pred, failed, secs = predict(a.root, studies, fs_of, a.weights, a.workers, a.bs)
    pred.to_csv(a.out)
    print(f"{len(pred)} studies in {time.time() - t:.0f}s, failed {len(failed)}, cpu s/study {secs.mean():.2f}")


if __name__ == "__main__":
    main()
