"""Study- and series-level tables derived from data/meta (train only).

study_table(): one row per study: language, site, sex, knee side, gold flag, duplicate group.
series_table(): one row per series: plane, weighting (T1/PD/T2/GRE/UNK), fat suppression, series type, 3D flag, slices.
All values are computed from DICOM-level fields, so the same functions can run on test meta built the same way."""
import collections
import functools
import re

import numpy as np
import pandas as pd

from common.paths import FINDINGS, META

STOP = {  # stopword heuristic, same rules as experiments/v0/oct7_baseline/analyses/eda/reports_eda.py
    "en": "the of and with is no there are knee normal".split(),
    "es": "de la el con sin se los del rodilla que".split(),
    "de": "der die und mit ohne kein keine im des knie".split(),
    "tr": "ve ile izlendi izlenmedi diz olarak mevcut bulgu".split(),
    "hr": "je i se u na bez koljena nema vidljiv".split(),
    "nl": "de het en met geen van een knie is".split(),
    "fr": "le la les et avec sans des du genou pas".split(),
}
WEIGHTINGS = ["T1", "PD", "T2", "GRE", "UNK"]
PLANES = ["Sagittal", "Coronal", "Axial"]
SERIES_TYPES = [f"{p[:3].upper()}_{w}_{'FS' if fs else 'NOFS'}" for p in PLANES for w in WEIGHTINGS for fs in (0, 1)]
DUPLICATE_EXAMS = [("6623376711", "3531613209")]  # one exam stored under two StudyInstanceUIDs (suffixes), experiments/v0/oct7_baseline/analyses/eda/out/duplicates_check.txt


def language(text):
    if re.search(r"[α-ωΑ-Ω]{3,}", text):
        return "el"
    if re.search(r"[а-яА-Я]{3,}", text):
        return "bg"
    c = collections.Counter(re.findall(r"[a-zçğıöşüäß]+", text.lower()))
    score = {k: sum(c[w] for w in v) for k, v in STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 3 else "unk"


@functools.cache
def studies():
    return pd.read_csv(META / "studies.csv").set_index("StudyInstanceUID")


@functools.cache
def series():
    return pd.read_csv(META / "series.csv", low_memory=False).set_index("SeriesInstanceUID")


@functools.cache
def frames():
    return pd.read_csv(META / "frames.csv", engine="pyarrow")


def first(v):
    """First value of a two-level field ('a|b' -> 'a')."""
    return v.split("|")[0] if isinstance(v, str) else v


def weighting(row):
    """T1 / PD / T2 from TR and TE (gradient echo kept apart); SeriesDescription only when TR/TE are missing."""
    seq = str(row.ScanningSequence)
    if "GR" in seq and "SE" not in seq:
        return "GRE"
    tr, te = row.RepetitionTime, row.EchoTime
    if pd.notna(tr) and pd.notna(te):
        return "T1" if tr < 800 else "PD" if te < 50 else "T2"
    d = str(row.SeriesDescription).lower()
    if re.search(r"t1", d):
        return "T1"
    if re.search(r"pd|proton|dp\b|inter", d):
        return "PD"
    if re.search(r"t2|stir", d):
        return "T2"
    return "UNK"


def plane_of(r, c):
    """Plane from the slice normal (largest component of r x c, LPS)."""
    n = np.cross(r, c)
    return PLANES[int(np.argmax(np.abs(n)))]


@functools.cache
def series_table():
    s = series()
    r = s[["ImageOrientationPatient_rx", "ImageOrientationPatient_ry", "ImageOrientationPatient_rz"]].to_numpy(float)
    c = s[["ImageOrientationPatient_cx", "ImageOrientationPatient_cy", "ImageOrientationPatient_cz"]].to_numpy(float)
    n_slices = frames().groupby("SeriesInstanceUID").size()
    out = pd.DataFrame(index=s.index)
    out["StudyInstanceUID"] = s.StudyInstanceUID
    out["plane"] = [plane_of(a, b) for a, b in zip(r, c)]
    out["weighting"] = [weighting(x) for x in s.itertuples()]
    out["fs"] = s.Fat_Suppression.astype(int)
    out["series_type"] = out.plane.str[:3].str.upper() + "_" + out.weighting + "_" + np.where(out.fs == 1, "FS", "NOFS")
    out["is_3d"] = s.MRAcquisitionType.eq("3D")
    out["n_slices"] = n_slices.reindex(s.index).to_numpy()
    out["slice_spacing"] = pd.to_numeric(s.SpacingBetweenSlices.map(first), errors="coerce").fillna(s.SliceThickness)
    assert (out.plane == s.Anatomical_Plane).mean() > 0.99, "plane from geometry disagrees with Anatomical_Plane"
    return out


def image_centre_x(fr, s):
    """Patient x (mm, LPS) of the image centre of the first frame of each series."""
    f = fr.drop_duplicates("SeriesInstanceUID").set_index("SeriesInstanceUID")
    s = s.loc[f.index]
    half_c = (s.Columns.astype(float) - 1) / 2 * s.PixelSpacing_col
    half_r = (s.Rows.astype(float) - 1) / 2 * s.PixelSpacing_row
    return f.ImagePositionPatient_x + s.ImageOrientationPatient_rx * half_c + s.ImageOrientationPatient_cx * half_r


@functools.cache
def series_side():
    """Knee side per series: series Laterality tag (L/R), else sign of the image-centre x (+x = patient left)."""
    s = series()
    geo = np.where(image_centre_x(frames(), s).reindex(s.index) > 0, "L", "R")
    tag = s.Laterality.where(s.Laterality.isin(["L", "R"]))
    return tag.fillna(pd.Series(geo, index=s.index))


def norm_report(text):
    return " ".join(str(text).lower().split())


@functools.cache
def study_table():
    st = studies()
    out = pd.DataFrame(index=st.index)
    out["language"] = st.Report.fillna("").map(language)
    out["manufacturer"] = st.Manufacturer.map(first)
    out["model"] = st.ManufacturerModelName.map(first).astype(str)
    out["field"] = st.MagneticFieldStrength.map(first).astype(str)
    out["site"] = out.language + "|" + out.manufacturer + "|" + out.model + "|" + out.field
    out["sex"] = st.PatientSex.fillna("U")
    sides = series_side().groupby(series().StudyInstanceUID).agg(lambda v: v.value_counts().index[0])
    tag = st.Laterality.where(st.Laterality.isin(["L", "R"]))
    out["side"] = tag.fillna(sides.reindex(st.index))
    out["side_source"] = np.where(tag.notna(), "tag", "series")
    out["gold"] = st[FINDINGS[0]].notna()
    # duplicate groups: identical normalised report text, plus known duplicated exams
    key = st.Report.map(norm_report)
    group = key.map({k: i for i, k in enumerate(sorted(key.unique()))})
    for a, b in DUPLICATE_EXAMS:
        ia, ib = [st.index[st.index.str.endswith(x)][0] for x in (a, b)]
        group[ib] = group[ia]
    out["dup_group"] = group
    return out
