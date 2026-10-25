"""Build data/meta/{studies,series,frames}.csv and data/frames/*.b2nd from data/original_data (train).
Field levels, normalisation and dropped fields are documented in data/meta/readme.md.

conda run -n rsna python data/build/build_data.py                         # full build into data/
conda run -n rsna python data/build/build_data.py --limit 20 --out /tmp/x  # small test build
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
import pydicom
from pydicom.multival import MultiValue
from pydicom.uid import ExplicitVRLittleEndian

SRC = Path("/home/pavel/Programming/knee-abnormality-2026/data/original_data")
CPARAMS = {"codec": blosc2.Codec.ZSTD, "clevel": 5, "filters": [blosc2.Filter.BITSHUFFLE]}
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IDS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]

# Field levels. "Same within a study/series" was measured on all train files (see readme).
STUDY = ["PatientSex", "PatientPosition", "EchoNumbers", "BitsStored", "PixelRepresentation"]
# Almost always the same within a study: studies.csv gets "a|b" when series disagree, series.csv keeps exact values.
STUDY_AND_SERIES = ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SoftwareVersions", "ReceiveCoilName", "TransmitCoilName",
                    "Laterality", "BodyPartExamined", "NumberOfTemporalPositions", "VariableFlipAngleFlag", "RescaleType", "DiffusionBValue"]
SERIES = ["SeriesNumber", "SeriesDescription", "AcquisitionNumber", "ImageType", "MRAcquisitionType", "ScanningSequence", "SequenceVariant",
          "ScanOptions", "SequenceName", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "EchoTrainLength", "NumberOfAverages",
          "PixelBandwidth", "ImagingFrequency", "ContrastBolusAgent", "Rows", "Columns", "PixelSpacing", "SliceThickness", "AcquisitionMatrix",
          "NumberOfPhaseEncodingSteps", "PercentSampling", "PercentPhaseFieldOfView", "InPlanePhaseEncodingDirection", "ImageOrientationPatient",
          "RescaleSlope", "ParallelAcquisition", "ParallelAcquisitionTechnique", "ParallelReductionFactorInPlane",
          "MRAcquisitionPhaseEncodingStepsInPlane", "RFEchoTrainLength"]
# Almost always the same within a series: series.csv gets "a|b" when frames disagree, frames.csv keeps exact values.
SERIES_AND_FRAME = ["SpacingBetweenSlices"]
FRAME = ["InstanceNumber", "ImagePositionPatient", "SliceLocation", "WindowCenter", "WindowWidth", "InStackPositionNumber"]
# One value wherever present (values listed in readme).
CONSTANT = ["FileMetaInformationGroupLength", "FileMetaInformationVersion", "MediaStorageSOPClassUID", "TransferSyntaxUID", "SOPClassUID",
            "Modality", "SamplesPerPixel", "PhotometricInterpretation", "PlanarConfiguration", "BitsAllocated", "NumberOfFrames", "StackID",
            "PulseSequenceName", "ReceiveCoilType", "TransmitCoilType", "DiffusionDirectionality", "PartialFourier", "PartialFourierDirection",
            "ParallelReductionFactorOutOfPlane", "MRAcquisitionPhaseEncodingStepsOutOfPlane", "TemporalPositionIndex",
            "DiffusionGradientOrientation", "RescaleIntercept"]
REDUNDANT = ["MediaStorageSOPInstanceUID", "HighBit", "PatientID", "FrameOfReferenceUID", "PixelData"]
VECTORS = {"PixelSpacing": ["PixelSpacing_row", "PixelSpacing_col"],
           "ImageOrientationPatient": [f"ImageOrientationPatient_{a}" for a in ["rx", "ry", "rz", "cx", "cy", "cz"]],
           "ImagePositionPatient": ["ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]}
MANUFACTURER = {"Siemens Healthineers": "SIEMENS", "SIEMENS": "SIEMENS", "Siemens": "SIEMENS", "GE MEDICAL SYSTEMS": "GE", "GEHC": "GE",
                "Philips Medical Systems": "PHILIPS", "Philips": "PHILIPS", "Philips Healthcare": "PHILIPS", "TOSHIBA": "TOSHIBA",
                "CANON_MEC": "CANON", "FUJIFILM Healthcare Corporation": "FUJIFILM", "Hitachi Medical Corporation": "HITACHI"}
LATERALITY = {"L": "L", "R": "R", "LEFT": "L", "RIGHT": "R", "B": "B"}
IOP_TOLERANCE = 1e-4


def as_str(v):
    """DICOM value as text; multiple values joined by backslash (DICOM convention); empty -> None."""
    if v is None:
        return None
    s = "\\".join(map(str, v)) if isinstance(v, (MultiValue, list, tuple)) else str(v)
    return s.strip() or None


def convert_series(study, series, frames_dir):
    """Write every frame of a series as .b2nd (exact stored pixels), verify it, return the header of each file as text."""
    blosc2.set_nthreads(1)
    rows = []
    for f in sorted(os.listdir(SRC / "train_series" / study / series)):
        ds = pydicom.dcmread(SRC / "train_series" / study / series / f)
        assert f == ds.SOPInstanceUID + ".dcm" and ds.BitsAllocated == 16 and ds.SamplesPerPixel == 1
        assert ds.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian and len(ds.PixelData) == ds.Rows * ds.Columns * 2
        img = np.frombuffer(ds.PixelData, "<i2" if ds.PixelRepresentation else "<u2").reshape(ds.Rows, ds.Columns)
        assert np.array_equal(img, ds.pixel_array)
        path = str(frames_dir / f"{study}_{series}_{ds.SOPInstanceUID}.b2nd")
        blosc2.asarray(img, urlpath=path, mode="w", cparams=CPARAMS)
        back = blosc2.open(path)[:]
        assert back.dtype == img.dtype and back.tobytes() == ds.PixelData
        row = {e.keyword: as_str(e.value) if e.VR not in ("OB", "OW", "UN") else "<binary>" for e in ds.file_meta}
        row.update({e.keyword: as_str(e.value) for e in ds if e.keyword != "PixelData"})
        assert all(e.keyword and e.VR != "SQ" for e in ds), "private or sequence tag"
        rows.append(row)
    return rows


def convert_job(job):
    return convert_series(*job)


def to_numeric(df):
    """Parse every column whose non-missing values are all numbers; integer-valued columns become Int64."""
    for c in df.columns:
        if c in IDS or df[c].dtype != object and not pd.api.types.is_string_dtype(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() == df[c].notna().sum() and x.notna().any():
            v = x.dropna()
            df[c] = x.astype("Int64") if ((v % 1 == 0) & (v.abs() < 2**53)).all() else x
    return df


def joined(values):
    """Distinct values of a group, most common first, joined by '|'."""
    vc = values.dropna().astype(str).value_counts()
    return "|".join(vc.index) if len(vc) else None


def collapse(df, key, cols, mixed_ok):
    """One row per key. Columns in mixed_ok get 'a|b' where the group disagrees; other columns must agree (missing counts as a value)."""
    g = df.groupby(key, sort=False)
    out = g[cols].first()
    for c in cols:
        n_values = g[c].nunique()
        mixed = n_values > 1
        if c not in mixed_ok:
            n = g[c].count()
            has_missing = (n > 0) & (n < g.size())
            assert not (mixed | has_missing).any(), f"{c} differs inside {key} groups: {int(mixed.sum())} mixed, {int(has_missing.sum())} partly missing"
            continue
        if mixed.any():
            out[c] = out[c].astype(object)
            out.loc[mixed, c] = df[df[key].isin(mixed[mixed].index)].groupby(key)[c].apply(joined)
            print(f"  {c}: {int(mixed.sum())} {key} groups with several values")
    return out.reset_index()


def main(limit, out_root):
    meta_dir, frames_dir = out_root / "meta", out_root / "frames"
    meta_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted((st.name, se.name, frames_dir) for st in os.scandir(SRC / "train_series") for se in os.scandir(st.path))[:limit]
    t0 = time.time()
    rows = []
    with Pool(24) as pool:
        for i, r in enumerate(pool.imap_unordered(convert_job, jobs, chunksize=4)):
            rows.extend(r)
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} series, {time.time() - t0:.0f}s", flush=True)
    print(f"converted {len(jobs)} series, {len(rows)} frames in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    unknown = set(df.columns) - set(IDS + STUDY + STUDY_AND_SERIES + SERIES + SERIES_AND_FRAME + FRAME + CONSTANT + REDUNDANT)
    assert not unknown, f"fields without a level: {unknown}"
    df = df.reindex(columns=IDS + STUDY + STUDY_AND_SERIES + SERIES + SERIES_AND_FRAME + FRAME + CONSTANT + REDUNDANT[:-1])

    # redundant and constant fields
    assert (df.MediaStorageSOPInstanceUID == df.SOPInstanceUID).all()
    assert (df.HighBit.astype(int) == df.BitsStored.astype(int) - 1).all()
    assert (df.groupby("PatientID").StudyInstanceUID.nunique() == 1).all() and (df.groupby("StudyInstanceUID").PatientID.nunique() == 1).all()
    assert (df.groupby("FrameOfReferenceUID").SeriesInstanceUID.nunique() == 1).all() and (df.groupby("SeriesInstanceUID").FrameOfReferenceUID.nunique() == 1).all()
    df = to_numeric(df)
    for c in CONSTANT:
        assert df[c].nunique() <= 1, f"{c} is not constant: {df[c].value_counts().to_dict()}"
        print(f"  constant {c}: {df[c].dropna().iloc[0] if df[c].notna().any() else 'always empty'} in {100 * df[c].notna().mean():.2f}% of files")
    df = df.drop(columns=CONSTANT + REDUNDANT[:-1])

    # normalise values
    assert df.Manufacturer.isin(MANUFACTURER).all(), set(df.Manufacturer) - set(MANUFACTURER)
    df["Manufacturer"] = df.Manufacturer.map(MANUFACTURER)
    assert df.Laterality.dropna().isin(LATERALITY).all(), set(df.Laterality.dropna()) - set(LATERALITY)
    df["Laterality"] = df.Laterality.map(LATERALITY)
    for c, names in VECTORS.items():
        df[names] = df[c].str.split("\\", expand=True).astype(float).to_numpy()
    iop = VECTORS["ImageOrientationPatient"]
    spread = df.groupby("SeriesInstanceUID")[iop].agg(lambda s: s.max() - s.min()).max(axis=1)
    assert spread.max() <= IOP_TOLERANCE, spread.max()
    print(f"  ImageOrientationPatient: max spread inside a series {spread.max():.1e}, first frame's value kept")
    df = df.sort_values(["StudyInstanceUID", "SeriesInstanceUID", "InstanceNumber"]).reset_index(drop=True)
    first_iop = df.groupby("SeriesInstanceUID")[iop].transform("first")
    df[iop] = first_iop
    assert not df.duplicated(["SeriesInstanceUID", "InstanceNumber"]).any()

    def cols(fields):
        return [n for f in fields for n in VECTORS.get(f, [f])]

    # series level (STUDY fields are checked here too: they must not change inside a series)
    print("series level:")
    series_cols = cols(STUDY + STUDY_AND_SERIES + SERIES + SERIES_AND_FRAME)
    series = collapse(df[["SeriesInstanceUID", "StudyInstanceUID"] + series_cols], "SeriesInstanceUID", ["StudyInstanceUID"] + series_cols, SERIES_AND_FRAME)
    print("study level:")
    study_cols = cols(STUDY + STUDY_AND_SERIES)
    studies = collapse(series[["StudyInstanceUID"] + study_cols], "StudyInstanceUID", study_cols, STUDY_AND_SERIES)

    # host tables
    train = pd.read_csv(SRC / "train.csv")
    train[LABELS] = train[LABELS].astype("Int64")
    studies = train.merge(studies, on="StudyInstanceUID", how="right" if limit else "outer", validate="1:1", indicator=True)
    assert (studies._merge == "both").all(), "train.csv and study folders differ"
    studies = studies.drop(columns="_merge").sort_values("StudyInstanceUID")
    train_series = pd.read_csv(SRC / "train_series.csv")
    series = train_series.merge(series, on=["StudyInstanceUID", "SeriesInstanceUID"], how="right" if limit else "outer", validate="1:1", indicator=True)
    assert (series._merge == "both").all(), "train_series.csv and series folders differ"
    series = series.drop(columns="_merge").sort_values(["StudyInstanceUID", "SeriesNumber", "SeriesInstanceUID"])
    series = series[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"] + cols(STUDY_AND_SERIES + SERIES + SERIES_AND_FRAME)]
    frames = df[IDS + cols(FRAME + SERIES_AND_FRAME)]

    for name, table in [("studies", studies), ("series", series), ("frames", frames)]:
        path = meta_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        back = pd.read_csv(path, low_memory=False)
        assert back.shape == table.shape and list(back.columns) == list(table.columns)
        print(f"{name}.csv: {table.shape[0]} rows, {table.shape[1]} columns, {path.stat().st_size / 1e6:.1f} MB")
    n_files = sum(1 for _ in os.scandir(frames_dir))
    assert n_files == len(frames), (n_files, len(frames))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="convert only the first N series (test run)")
    ap.add_argument("--out", type=Path, default=Path("/home/pavel/Programming/knee-abnormality-2026/data"))
    args = ap.parse_args()
    main(args.limit, args.out)
