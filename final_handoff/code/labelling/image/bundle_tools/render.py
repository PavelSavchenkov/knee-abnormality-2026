"""Contact sheets of one study for VLM image labelling, from the canonical slice cache (common/cache.py: 0.4 mm/px over 160 mm,
every knee shown as a LEFT knee). Each tile carries its id, series type, slice position, orientation letters and a 10 mm scale bar.

Sheets (tile ids):
  1_sag_fs     S1-S9    sagittal fluid-sensitive, medial -> lateral (spread over the series)
  2_cor_fs     C1-C6    coronal fluid-sensitive, anterior -> posterior
  3_ax_fs      A1-A6    axial fluid-sensitive, inferior -> superior
  4_nonfs      N1-N6    non-fat-suppressed: 3 sagittal (medial, middle, lateral) + 3 coronal (anterior, middle, posterior)
  5_notch      NT1-NT9  9 consecutive sagittal fluid-sensitive slices around the series centre (notch: ACL, PCL), zoom 0.2 mm/px
  6_medial     MM1-MM9  9 consecutive sagittal slices around 18 mm medial (medial meniscus, cartilage), zoom 0.2 mm/px
  7_lateral    LM1-LM9  9 consecutive sagittal slices around 18 mm lateral (lateral meniscus, cartilage), zoom 0.2 mm/px
  8_cor_joint  CJ1-CJ9  9 consecutive coronal slices around the series centre (joint line, both compartments, MCL), zoom 0.24 mm/px
  9_patella    PF1-PF9  9 consecutive axial slices around 14 mm superior (patellofemoral joint), zoom 0.2 mm/px
Consecutive = neighbouring slices of the series (thin 3D series: neighbouring 3 mm slabs), so two neighbouring tiles are two adjacent images.
A slot with no series of the wanted type uses the closest other series of that plane and says so in the sheet title.
"""
import cv2
import numpy as np
import pandas as pd

from common import meta
from common.cache import SliceCache

PX, MM_PER_PX = 400, 0.4
FONT = cv2.FONT_HERSHEY_SIMPLEX
LETTERS = {"Sagittal": ("S", "I", "A", "P"), "Coronal": ("S", "I", "M", "L"), "Axial": ("A", "P", "M", "L")}  # top, bottom, left, right
POS_WORDS = {"Sagittal": ("medial", "lateral"), "Coronal": ("anterior", "posterior"), "Axial": ("inferior", "superior")}
FS_ORDER = {"PD": 0, "T2": 1, "UNK": 2, "GRE": 3, "T1": 4}
NONFS_ORDER = {"T1": 0, "PD": 1, "T2": 2, "UNK": 3, "GRE": 4}
N_RUN = 9
# consecutive-slice zoom sheets: sheet, tile prefix, slot, target slice position (mm), crop centre (x, y) and size in cache px, title
RUNS = [("5_notch", "NT", "sag_fs", 0, (200, 200), 200, "NOTCH (ACL, PCL)"),
        ("6_medial", "MM", "sag_fs", -18, (200, 230), 200, "MEDIAL compartment (meniscus, cartilage)"),
        ("7_lateral", "LM", "sag_fs", 18, (200, 230), 200, "LATERAL compartment (meniscus, cartilage)"),
        ("8_cor_joint", "CJ", "cor_fs", 0, (200, 240), 240, "CORONAL joint line (menisci, cartilage, MCL)"),
        ("9_patella", "PF", "ax_fs", 14, (200, 125), 200, "PATELLOFEMORAL joint")]


def stir_series():
    s = meta.series()
    ti = pd.to_numeric(s.InversionTime, errors="coerce").fillna(0)
    return set(s.index[(ti > 50) | s.ScanningSequence.astype(str).str.contains("IR")])


def seq_label(row, stir):
    plane = {"Sagittal": "SAG", "Coronal": "COR", "Axial": "AX"}[row.plane]
    w = "STIR" if row.SeriesInstanceUID in stir else row.weighting
    fs = " FS" if row.fs and w != "STIR" else ""
    return f"{plane} {w}{fs}{' 3D' if row.is_3d else ''}"


def pick(series, plane, fat_suppressed):
    """Best series of a plane for a slot; falls back to any series of that plane. Returns (row, substituted)."""
    cand = series[(series.plane == plane) & ~series.other_knee & (series.dup_of.fillna("") == "")]
    if cand.empty:
        return None, False
    order = FS_ORDER if fat_suppressed else NONFS_ORDER
    want = cand[cand.fs == int(fat_suppressed)]
    pool = want if len(want) else cand
    pool = pool.assign(rank=pool.weighting.map(order), thin=pool.is_3d.astype(int)).sort_values(["rank", "thin", "n"], ascending=[True, True, False])
    return pool.iloc[0], want.empty


def text(img, s, xy, scale=0.45):
    """White text with a black outline (offset copies: a thicker stroke would widen OpenCV's letter spacing)."""
    x, y = xy
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        cv2.putText(img, s, (x + dx, y + dy), FONT, scale, 0, 1, cv2.LINE_AA)
    cv2.putText(img, s, xy, FONT, scale, 255, 1, cv2.LINE_AA)


def annotate(img, tile_id, label, plane, mm_per_px):
    img = img.copy()
    h, w = img.shape
    text(img, f"{tile_id}  {label}", (6, 16))
    top, bottom, left, right = LETTERS[plane]
    text(img, top, (w // 2 - 5, 34), 0.55)
    text(img, bottom, (w // 2 - 5, h - 8), 0.55)
    text(img, left, (6, h // 2 + 6), 0.55)
    text(img, right, (w - 18, h // 2 + 6), 0.55)
    bar = int(round(10 / mm_per_px))
    x0, y0 = w - bar - 12, h - 14
    cv2.rectangle(img, (x0 - 1, y0 - 3), (x0 + bar + 1, y0 + 3), 0, -1)
    cv2.rectangle(img, (x0, y0 - 2), (x0 + bar, y0 + 2), 255, -1)
    text(img, "10 mm", (x0 + bar // 2 - 22, y0 - 8), 0.4)
    return img


def position_text(plane, pos):
    lo, hi = POS_WORDS[plane]
    return f"{abs(pos):.0f} mm {hi if pos >= 0 else lo}"


def grid(tiles, cols, title):
    rows = -(-len(tiles) // cols)
    gap, head = 6, 30
    sheet = np.zeros((head + rows * (PX + gap), cols * (PX + gap) - gap), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[head + r * (PX + gap): head + r * (PX + gap) + PX, c * (PX + gap): c * (PX + gap) + PX] = t
    text(sheet, title, (6, 20), 0.5)
    return sheet


class Renderer:
    def __init__(self, cache="fov160_px400_q95"):
        self.cache = SliceCache(cache)
        self.series = self.cache.series.set_index("StudyInstanceUID", drop=False)
        self.slices = self.cache.slices.sort_values(["SeriesInstanceUID", "idx"]).groupby("SeriesInstanceUID")
        self.stir = stir_series()

    def frames(self, row, quantiles):
        """Slices of a series at given quantiles of its index range: list of (image, pos_mm)."""
        sl = self.slices.get_group(row.SeriesInstanceUID)
        k = [int(round(q * (len(sl) - 1))) for q in quantiles]
        picked = sl.iloc[k]
        return list(zip(self.cache.read(picked.row.to_numpy()), picked.pos_mm.to_numpy()))

    def run(self, row, target):
        """N_RUN consecutive slices of a series around the slice nearest to target mm: list of (image, pos_mm)."""
        sl = self.slices.get_group(row.SeriesInstanceUID)
        k = int(np.argmin(np.abs(sl.pos_mm.to_numpy() - target)))
        lo = int(np.clip(k - N_RUN // 2, 0, max(len(sl) - N_RUN, 0)))
        picked = sl.iloc[lo:lo + N_RUN]
        return list(zip(self.cache.read(picked.row.to_numpy()), picked.pos_mm.to_numpy()))

    def sheets(self, uid, index):
        """[(sheet name, image, [tile rows])] for one study."""
        series = self.series.loc[[uid]]
        out = []
        slots = {}
        for name, plane, fs in [("sag_fs", "Sagittal", True), ("cor_fs", "Coronal", True), ("ax_fs", "Axial", True),
                                ("sag_nonfs", "Sagittal", False), ("cor_nonfs", "Coronal", False)]:
            slots[name] = pick(series, plane, fs)

        def overview(sheet, slot, prefix, quantiles, cols, what):
            row, sub = slots[slot]
            if row is None:
                return
            label = seq_label(row, self.stir)
            tiles, meta_rows = [], []
            for j, (img, pos) in enumerate(self.frames(row, quantiles), start=1):
                tid = f"{prefix}{j}"
                tiles.append(annotate(img, tid, f"{label}  {position_text(row.plane, pos)}", row.plane, MM_PER_PX))
                meta_rows.append(dict(tile=tid, sheet=sheet, plane=row.plane, sequence=label, pos_mm=round(float(pos), 1),
                                      zoom=1, crop="", series=row.SeriesInstanceUID))
            note = f"  [no {'fat-suppressed' if slot.endswith('_fs') else 'non-fat-suppressed'} {row.plane.lower()} series: showing {label}]" if sub else ""
            out.append((sheet, grid(tiles, cols, f"Study {index} | {what} | {label}{note} | every knee shown as a LEFT knee"), meta_rows))

        overview("1_sag_fs", "sag_fs", "S", np.linspace(0.08, 0.92, 9), 3, "SAGITTAL fluid-sensitive, medial -> lateral")
        overview("2_cor_fs", "cor_fs", "C", np.linspace(0.1, 0.9, 6), 3, "CORONAL fluid-sensitive, anterior -> posterior")
        overview("3_ax_fs", "ax_fs", "A", np.linspace(0.1, 0.9, 6), 3, "AXIAL fluid-sensitive, inferior -> superior")

        tiles, meta_rows = [], []
        for slot, quantiles in [("sag_nonfs", (0.25, 0.5, 0.75)), ("cor_nonfs", (0.25, 0.5, 0.75))]:
            row, sub = slots[slot]
            if row is None:
                continue
            label = seq_label(row, self.stir)
            for img, pos in self.frames(row, quantiles):
                tid = f"N{len(tiles) + 1}"
                tiles.append(annotate(img, tid, f"{label}  {position_text(row.plane, pos)}", row.plane, MM_PER_PX))
                meta_rows.append(dict(tile=tid, sheet="4_nonfs", plane=row.plane, sequence=label, pos_mm=round(float(pos), 1),
                                      zoom=1, crop="", series=row.SeriesInstanceUID))
        if tiles:
            out.append(("4_nonfs", grid(tiles, 3, f"Study {index} | NON-FAT-SUPPRESSED sagittal (medial, middle, lateral) + coronal "
                                                   f"(anterior, middle, posterior) | every knee shown as a LEFT knee"), meta_rows))

        for sheet, prefix, slot, target, (cx, cy), size, what in RUNS:
            row, sub = slots[slot]
            if row is None:
                continue
            label = seq_label(row, self.stir)
            mm = MM_PER_PX * size / PX
            x0, y0 = int(np.clip(cx - size // 2, 0, PX - size)), int(np.clip(cy - size // 2, 0, PX - size))
            tiles, meta_rows = [], []
            for j, (img, pos) in enumerate(self.run(row, target), start=1):
                crop = cv2.resize(img[y0:y0 + size, x0:x0 + size], (PX, PX), interpolation=cv2.INTER_CUBIC)
                tid = f"{prefix}{j}"
                tiles.append(annotate(crop, tid, f"{label}  {position_text(row.plane, pos)}", row.plane, mm))
                meta_rows.append(dict(tile=tid, sheet=sheet, plane=row.plane, sequence=label, pos_mm=round(float(pos), 1),
                                      zoom=round(PX / size, 2), crop=f"{x0},{y0},{size},{size}", series=row.SeriesInstanceUID))
            note = f" [no fat-suppressed series: {label}]" if sub else ""
            out.append((sheet, grid(tiles, 3, f"Study {index} | {what} | {len(tiles)} CONSECUTIVE slices, zoom {mm:.2g} mm/px | "
                                              f"{label}{note} | LEFT knee view"), meta_rows))
        return out
