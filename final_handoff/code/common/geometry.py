"""Canonical orientation of a knee MRI series. Pure numpy; works on train meta and on test DICOM headers alike.

Canonical layout (every knee is shown as a LEFT knee; L = patient left for a left knee, patient right for a right knee):
    plane      image rows go       image columns go       slice index goes
    Sagittal   superior->inferior  anterior->posterior    medial->lateral
    Coronal    superior->inferior  medial->lateral        anterior->posterior
    Axial      anterior->posterior medial->lateral        inferior->superior
Patient coordinates are DICOM LPS: +x patient left, +y posterior, +z superior.
r = ImageOrientationPatient row cosine (direction of increasing column index), c = column cosine (increasing row index)."""
import numpy as np

PLANES = ["Sagittal", "Coronal", "Axial"]


def targets(plane, side):
    """(down, right, slice) unit vectors in patient LPS for the canonical layout."""
    lat = np.array([1.0, 0, 0]) if side == "L" else np.array([-1.0, 0, 0])
    post, sup = np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    return {"Sagittal": (-sup, post, lat), "Coronal": (-sup, lat, post), "Axial": (post, lat, sup)}[plane]


def plane_of(r, c):
    return PLANES[int(np.argmax(np.abs(np.cross(r, c))))]


def inplane_ops(r, c, plane, side):
    """How to turn a stored frame into the canonical layout: (transpose, flip_rows, flip_cols), applied in that order."""
    down, right, _ = targets(plane, side)
    transpose = abs(r @ down) > abs(c @ down)
    row_dir, col_dir = (r, c) if transpose else (c, r)       # directions of increasing row / column after the transpose
    return bool(transpose), bool(row_dir @ down < 0), bool(col_dir @ right < 0)


def apply_inplane(img, ops):
    transpose, fr, fc = ops
    if transpose:
        img = img.T
    if fr:
        img = img[::-1]
    if fc:
        img = img[:, ::-1]
    return img


def slice_positions(ipp, plane, side):
    """Position of each frame (mm) along the canonical slice direction; sort ascending to get canonical order."""
    return np.asarray(ipp, float) @ targets(plane, side)[2]
