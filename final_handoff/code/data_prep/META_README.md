# data/meta and data/frames

Built from `data/original_data` (train only) by `data/build/build_data.py`; build log: `data/build/build_data.log`.

| File | Rows | Key | Content |
|---|---|---|---|
| `studies.csv` | 4,407 | `StudyInstanceUID` | Report, 12 labels, study-level DICOM fields |
| `series.csv` | 24,371 | `SeriesInstanceUID` | `train_series.csv` columns, series-level DICOM fields |
| `frames.csv` | 819,078 | `SOPInstanceUID` | Frame-level DICOM fields |
| `../frames/*.b2nd` | 819,078 | file name | Pixels of one frame |

## Pixels: data/frames

- File: `data/frames/{StudyInstanceUID}_{SeriesInstanceUID}_{SOPInstanceUID}.b2nd`, one file per frame (= one original DICOM file).
- Format: blosc2 NDArray, zstd level 5 + bitshuffle. Lossless.
- Content: the exact stored pixel values (`PixelData` bytes), shape `(Rows, Columns)`, dtype `uint16` (`PixelRepresentation` 0)
  or `int16` (`PixelRepresentation` 1: GE, Canon, Toshiba; negative values seen in Toshiba series). No rescale, no windowing, no flips.
- Checked for every file: decoded file == DICOM `PixelData` byte for byte, and == pydicom `pixel_array`.
- Read: `blosc2.open(path)[:]` returns a numpy array (about 0.4 ms per 512x512 frame on one core).
- Slice order in a series: sort `frames.csv` by `InstanceNumber`; it matches the spatial order in every series.

## Reading the CSVs

```python
studies = pd.read_csv("data/meta/studies.csv")
series = pd.read_csv("data/meta/series.csv", low_memory=False)  # ContrastBolusAgent mixes numbers and text
frames = pd.read_csv("data/meta/frames.csv", engine="pyarrow")  # 220 MB: 0.1 s (0.9 s with the default engine)
```

## Conventions

- Empty cell: the tag is absent or empty.
- `\` separates several values of one DICOM field (DICOM convention), e.g. `ImageType` = `ORIGINAL\PRIMARY\M\ND`.
- `|` means the series of a study (or the frames of a series) have different values: distinct values, most common first.
  It occurs only in the fields listed under "Fields stored at two levels". pandas reads such a column as text.
- Numbers are parsed (`1.500000` -> 1.5); integer fields are written without decimals.
- Vectors are split into columns:
  - `PixelSpacing` -> `PixelSpacing_row` (mm between rows), `PixelSpacing_col` (mm between columns).
  - `ImageOrientationPatient` -> `_rx, _ry, _rz` (direction along a row, i.e. increasing column index) and `_cx, _cy, _cz` (direction along a column).
  - `ImagePositionPatient` -> `_x, _y, _z` (mm, centre of the top-left pixel).
  - Patient coordinates are LPS: x to the patient's left, y to the back, z to the head.
- Normalised values:
  - `Manufacturer`: Siemens Healthineers, SIEMENS, Siemens -> SIEMENS; GE MEDICAL SYSTEMS, GEHC -> GE;
    Philips Medical Systems, Philips, Philips Healthcare -> PHILIPS; CANON_MEC -> CANON; FUJIFILM Healthcare Corporation -> FUJIFILM;
    Hitachi Medical Corporation -> HITACHI; TOSHIBA unchanged.
  - `Laterality`: LEFT -> L, RIGHT -> R (B = both kept).
  - `ImageOrientationPatient` differs inside a series only by float rounding (at most 1e-5); `series.csv` keeps the first frame's value.
- Everything else is kept as in DICOM, e.g. `SeriesDescription` (including the placeholder `DummySeriesDesc!`) and `ContrastBolusAgent`
  (which holds values like `right` and `LEFT` in a few series).

## How levels were chosen

For every tag, on all 819,078 files: does the value change between frames of a series, and between series of a study
(a tag present in some series and absent in others counts as a change)?

- Never changes inside a study: study level only.
- Never changes inside a series: series level only.
- Changes inside series: frame level.
- Changes inside very few studies (or series): stored at both levels, see the next section.

## Fields stored at two levels

`studies.csv` holds one value, or `a|b|...` where the series disagree; `series.csv` holds the exact value of each series.
`SpacingBetweenSlices` works the same way between `series.csv` and `frames.csv`.

| Field | Studies with different values between series | Studies where some series lack it | Suggested level | Why |
|---|---|---|---|---|
| `Manufacturer` | 2 | 0 | study | Scanner. The 2 studies (`SIEMENS|PHILIPS`) mix series from two scanners; use series values there |
| `ManufacturerModelName` | 2 | 0 | study | Same 2 studies |
| `MagneticFieldStrength` | 2 | 0 | study | Same 2 studies (`1.5|3.0`) |
| `SoftwareVersions` | 2 | 0 | study | Same 2 studies |
| `ReceiveCoilName` | 8 | 3 | study | Knee coil; a few studies switch coil for some series |
| `TransmitCoilName` | 0 | 73 | study | Values never conflict; some series just omit it |
| `Laterality` | 25 (`L|R` or `R|L`) | 74 | study | Side of the knee. In the 25 studies use series values, or the side from image position |
| `BodyPartExamined` | 98 | 0 | study | Typed at the scanner and unreliable (e.g. `KNEE|WRIST`) |
| `NumberOfTemporalPositions` | 2 | 11 | series | Sequence parameter (1 or 4); vendor default elsewhere |
| `VariableFlipAngleFlag` | 2 | 2 | series | Sequence parameter (N or Y) |
| `RescaleType` | 0 | 3 | series | Belongs with `RescaleSlope`, which is series level |
| `DiffusionBValue` | 0 | 2 | series | Philips placeholder (0 or 7.23e+75); carries no information |
| `SpacingBetweenSlices` | 6 series | 0 | series | In 6 GE series one frame says 1.4, the rest 1.39 (rounding) |

## studies.csv

| Column | Meaning |
|---|---|
| `StudyInstanceUID` | Study ID; folder name in `train_series/` |
| `Report` | Free-text radiology report (9 languages; dates replaced by `[DATE]`) |
| `ACL` ... `Fracture` | 12 labels, 1/0; filled only for 58 studies |
| `PatientSex` | M, F, O (other) |
| `PatientPosition` | FFS (feet first, supine) or, rarely, FFP (feet first, prone) |
| `EchoNumbers` | Echo the image comes from (1; 0 for some vendors) |
| `BitsStored` | 12 or 16 bits used per pixel |
| `PixelRepresentation` | 0 = unsigned (`uint16` frames), 1 = signed (`int16` frames: GE, Canon, Toshiba) |
| Two-level fields | See the table above |

## series.csv

| Column | Meaning |
|---|---|
| `StudyInstanceUID`, `SeriesInstanceUID` | IDs |
| `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` | From `train_series.csv`. `Fluid_Sensitive` == `Fat_Suppression` in all train series; the host says not to rely on this in test |
| Two-level fields | See the table above |
| `SeriesNumber` | Series number in the exam |
| `SeriesDescription` | Protocol name set by the site |
| `AcquisitionNumber` | Acquisition counter on the scanner |
| `ImageType` | Image flags (ORIGINAL, PRIMARY, vendor codes) |
| `MRAcquisitionType` | 2D or 3D |
| `ScanningSequence` | SE spin echo, GR gradient echo, IR inversion recovery, RM research mode |
| `SequenceVariant` | SK segmented k-space, SP spoiled, OSP oversampling phase, MP magnetisation prepared, NONE |
| `ScanOptions` | FS fat saturation, PER phase-encode reordering, SP spatial presaturation, GE `*_GEMS` codes |
| `SequenceName` | Vendor sequence code (Siemens, Toshiba, Canon) |
| `RepetitionTime`, `EchoTime`, `InversionTime` | TR, TE, TI in ms (TI only in inversion-recovery series) |
| `FlipAngle` | Excitation angle, degrees |
| `EchoTrainLength` | Echoes per excitation (turbo spin echo) |
| `NumberOfAverages` | Signal averages |
| `PixelBandwidth` | Hz per pixel |
| `ImagingFrequency` | MHz, about 42.58 x field strength |
| `ContrastBolusAgent` | Contrast agent; set in a few series of 5 studies |
| `Rows`, `Columns` | Frame size in pixels |
| `PixelSpacing_row`, `PixelSpacing_col` | mm |
| `SliceThickness` | mm |
| `AcquisitionMatrix` | Measured matrix: frequency rows \ frequency columns \ phase rows \ phase columns |
| `NumberOfPhaseEncodingSteps`, `PercentSampling`, `PercentPhaseFieldOfView` | Phase-encoding lines, % of lines measured, phase FOV as % of frequency FOV |
| `InPlanePhaseEncodingDirection` | ROW or COL |
| `ImageOrientationPatient_*` | Slice orientation (see Conventions) |
| `RescaleSlope` | Real value = stored value x slope (intercept is always 0); Siemens and Philips only |
| `ParallelAcquisition`, `ParallelAcquisitionTechnique`, `ParallelReductionFactorInPlane` | Parallel imaging (SENSE etc.); rare |
| `MRAcquisitionPhaseEncodingStepsInPlane`, `RFEchoTrainLength` | Rare acquisition details (one Philips scanner type) |
| `SpacingBetweenSlices` | mm between slice centres (two-level field) |

## frames.csv

| Column | Meaning |
|---|---|
| `StudyInstanceUID`, `SeriesInstanceUID`, `SOPInstanceUID` | IDs; together they give the frame file name |
| `InstanceNumber` | Frame number in the series; sorting by it gives spatial order |
| `ImagePositionPatient_x`, `_y`, `_z` | mm, centre of the top-left pixel |
| `SliceLocation` | Slice position along its normal, vendor-defined |
| `WindowCenter`, `WindowWidth` | Display window suggested by the scanner |
| `InStackPositionNumber` | Slice index in the stack (GE, Toshiba, Canon only) |
| `SpacingBetweenSlices` | Two-level field, exact per frame |

## Dropped fields

One value wherever present (share of files that have the tag):

| Field | Value | Present in |
|---|---|---|
| `FileMetaInformationGroupLength` | 148 | 100% |
| `FileMetaInformationVersion` | binary | 100% |
| `MediaStorageSOPClassUID`, `SOPClassUID` | 1.2.840.10008.5.1.4.1.1.4 (MR Image Storage) | 100% |
| `TransferSyntaxUID` | 1.2.840.10008.1.2.1 (explicit VR little endian, uncompressed) | 100% |
| `Modality` | MR | 100% |
| `SamplesPerPixel` | 1 | 100% |
| `PhotometricInterpretation` | MONOCHROME2 (higher value = brighter) | 100% |
| `BitsAllocated` | 16 | 100% |
| `PlanarConfiguration` | 0 | 9.3% |
| `NumberOfFrames` | 1 | 14.9% |
| `StackID` | 1 | 18.4% |
| `DiffusionGradientOrientation` | 0\0\0 | 31.0% |
| `RescaleIntercept` | 0 | 42.8% |
| `TemporalPositionIndex` | 1 | 2.6% |
| `PulseSequenceName` | TSE | 2.4% |
| `ReceiveCoilType` | MULTICOIL | 2.4% |
| `TransmitCoilType` | BODY | 2.4% |
| `DiffusionDirectionality` | NONE | 2.4% |
| `PartialFourier` | NO | 2.4% |
| `ParallelReductionFactorOutOfPlane` | 1 | 2.4% |
| `MRAcquisitionPhaseEncodingStepsOutOfPlane` | 1 | 2.4% |
| `PartialFourierDirection` | always empty | 0% |

Redundant:

| Field | Why |
|---|---|
| `MediaStorageSOPInstanceUID` | Equals `SOPInstanceUID` |
| `HighBit` | Equals `BitsStored` - 1 |
| `PatientID` | One random ID per study (the exam that appears in two studies has two different IDs) |
| `FrameOfReferenceUID` | One random ID per series |
| `PixelData` | Stored in `data/frames` |
| File name | Equals `SOPInstanceUID` + `.dcm` |
