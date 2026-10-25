"""Native-resolution views of one bundle study (by bundle index) for image-labelling agents. Same canonical orientation as the
bundle sheets: every knee shown as a LEFT knee; sagittal slices medial -> lateral, coronal anterior -> posterior, axial
inferior -> superior. Positions are mm from the series centre, as on the bundle tiles.

conda run -n rsna python work/image_labeling/view.py series INDEX
    the study's series: id, plane, sequence, slices, slice spacing, position range, native mm per pixel, flags; then per series
    the scanner's description (if any) and parameters (TR, TE, TI in ms, slice thickness, field strength)
conda run -n rsna python work/image_labeling/view.py show INDEX SERIES FIRST LAST --out FILE.png [--box X0 Y0 X1 Y1] [--every K]
        [--window LO HI]
    PNG grid of slices FIRST..LAST (1-based, canonical order, at most 12) of one series at native resolution, optionally cropped
    to a box given as fractions of the full slice (0 = left / top, 1 = right / bottom); each tile shows the slice number, position,
    orientation letters and a 10 mm bar. --window: grey levels black..white as percentiles of the series' original values
    (default 0 99.5; e.g. 20 99.9 for marrow, 0 97 for dark structures such as menisci and ligaments).
"""
import argparse
import sys
from pathlib import Path

import blosc2
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root, for common/ (no PYTHONPATH needed)
sys.path.insert(0, str(Path(__file__).parent))
from render import annotate, position_text, seq_label, stir_series, text  # noqa: E402

from common.geometry import apply_inplane, slice_positions  # noqa: E402
from common.meta import frames, series  # noqa: E402
from common.paths import CACHE, FRAMES  # noqa: E402

HERE = Path(__file__).parent
TILE, COLS, GAP, HEAD = 512, 3, 6, 30


def study_series(index):
    """Series of the study behind a bundle index, in SeriesNumber order, with ids s1, s2, ..."""
    uid = pd.read_csv(HERE / "out/order.csv", index_col="index").source[index]
    s = pd.read_parquet(CACHE / "fov160_px400_q95/series.parquet", filters=[("StudyInstanceUID", "==", uid)])
    s = s.sort_values(["SeriesNumber", "SeriesInstanceUID"]).reset_index(drop=True)
    s["sid"] = [f"s{i + 1}" for i in range(len(s))]
    return s


def native_frames(row):
    """(canonical images, positions in mm from the series centre, mm per pixel) of every frame of a series, in canonical order."""
    f = frames()
    f = f[f.SeriesInstanceUID == row.SeriesInstanceUID].sort_values("InstanceNumber")
    pos = slice_positions(f[["ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]].to_numpy(), row.plane, row.side)
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    ops = (row["transpose"], row["flip_rows"], row["flip_cols"])           # row["transpose"]: a Series also has a .transpose method
    sp_r, sp_c = (row.sp_col, row.sp_row) if ops[0] else (row.sp_row, row.sp_col)
    mm = min(sp_r, sp_c)
    imgs = []
    for sop in f.SOPInstanceUID.to_numpy()[order]:
        a = blosc2.open(str(FRAMES / f"{row.StudyInstanceUID}_{row.SeriesInstanceUID}_{sop}.b2nd"))[:].astype(np.float32)
        a = np.ascontiguousarray(apply_inplane(a, ops))
        imgs.append(cv2.resize(a, (round(a.shape[1] * sp_c / mm), round(a.shape[0] * sp_r / mm)), interpolation=cv2.INTER_LINEAR))
    return imgs, pos - (pos.min() + pos.max()) / 2, mm


def list_series(index):
    s = study_series(index)
    stir = stir_series()
    by_uid = dict(zip(s.SeriesInstanceUID, s.sid))
    for r in s.itertuples():
        hi = (r.n_frames - 1) * r.frame_spacing / 2
        lo = -hi
        flags = (["other knee: ignore"] if r.other_knee else []) + ([f"duplicate of {by_uid[r.dup_of]}"] if r.dup_of else [])
        mm = min(r.sp_row, r.sp_col)
        print(f"{r.sid:4} {r.plane:9} {seq_label(r, stir):16} {r.n_frames:3} slices, spacing {r.frame_spacing:.1f} mm, "
              f"positions {position_text(r.plane, lo)} .. {position_text(r.plane, hi)}, {mm:.2f} mm/px  {'; '.join(flags)}")
    m = series().loc[s.SeriesInstanceUID]
    print()
    for sid, (_, x) in zip(s.sid, m.iterrows()):
        desc = f'"{x.SeriesDescription}": ' if isinstance(x.SeriesDescription, str) else ""
        print(f"{sid:4} {desc}TR {x.RepetitionTime}, TE {x.EchoTime}, TI {x.InversionTime}, slice thickness {x.SliceThickness} mm, "
              f"{x.MagneticFieldStrength} T")


def show(index, sid, first, last, out, box, every, window):
    s = study_series(index).set_index("sid", drop=False)
    assert sid in s.index, f"{sid} is not a series of study {index}: {list(s.index)}"
    row = s.loc[sid]
    imgs, pos, mm = native_frames(row)
    picks = list(range(first, last + 1, every))
    assert 1 <= first <= last <= len(imgs), f"slices are 1..{len(imgs)}"
    assert len(picks) <= 12, "at most 12 slices per view"
    lo, hi = (float(v) for v in np.percentile(np.stack(imgs)[:, ::4, ::4], window))
    assert hi > lo, "window: LO must be below HI"
    h, w = imgs[0].shape
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "box needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
    c0, c1, r0, r1 = int(x0 * w), int(round(x1 * w)), int(y0 * h), int(round(y1 * h))
    scale = TILE / max(c1 - c0, r1 - r0)
    shown = mm / scale
    assert 10 / shown < TILE - 60, "box too small: make it at least about 15 mm wide"
    label = seq_label(row, stir_series())
    tiles = []
    for i in picks:
        a = np.clip((imgs[i - 1][r0:r1, c0:c1] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        a = cv2.resize(a, (round(a.shape[1] * scale), round(a.shape[0] * scale)),
                       interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        t = np.zeros((TILE, TILE), np.uint8)
        t[:a.shape[0], :a.shape[1]] = a[:TILE, :TILE]
        tiles.append(annotate(t, f"{sid} #{i}", f"{label}  {position_text(row.plane, pos[i - 1])}", row.plane, shown))
    rows = -(-len(tiles) // COLS)
    sheet = np.zeros((HEAD + rows * (TILE + GAP), COLS * (TILE + GAP) - GAP), np.uint8)
    for j, t in enumerate(tiles):
        r, c = divmod(j, COLS)
        sheet[HEAD + r * (TILE + GAP): HEAD + r * (TILE + GAP) + TILE, c * (TILE + GAP): c * (TILE + GAP) + TILE] = t
    text(sheet, f"Study {index} | {sid} {label} | native {mm:.2f} mm/px, shown {shown:.2f} mm/px | LEFT knee view", (6, 20), 0.5)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}: {len(tiles)} slices of {sid} ({len(imgs)} in the series)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("series")
    a.add_argument("index", type=int)
    a = sub.add_parser("show")
    a.add_argument("index", type=int)
    a.add_argument("series")
    a.add_argument("first", type=int)
    a.add_argument("last", type=int)
    a.add_argument("--out", required=True)
    a.add_argument("--box", type=float, nargs=4, default=[0, 0, 1, 1])
    a.add_argument("--every", type=int, default=1)
    a.add_argument("--window", type=float, nargs=2, default=[0, 99.5])
    args = ap.parse_args()
    if args.cmd == "series":
        list_series(args.index)
    else:
        show(args.index, args.series, args.first, args.last, args.out, args.box, args.every, args.window)
