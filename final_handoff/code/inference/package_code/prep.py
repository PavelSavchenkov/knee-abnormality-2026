"""Series preprocessing shared by the train cache and Kaggle inference.

One series -> uint8 volume (D, S, S) in the canonical layout of common.geometry (every knee shown as a left knee),
in-plane resampled to PX_MM mm/px and centre-cropped / padded to S x S, slices sorted along the canonical slice axis.
Intensity: per-series robust window [p0.5, p99.8] -> [0, 255].
Thin 3D stacks (spacing < MIN_SPACING mm) are averaged in groups so the stored spacing is >= MIN_SPACING."""
import cv2
import numpy as np

from common_geometry import apply_inplane, inplane_ops, plane_of, slice_positions

PX_MM = 0.5
SIZE = 320
MIN_SPACING = 1.5
LO_PCT, HI_PCT = 0.5, 99.8


def series_geometry(r, c, ipps, side):
    """plane, in-plane ops and slice order / positions (mm, ascending canonical) for one series."""
    r, c = np.asarray(r, float), np.asarray(c, float)
    plane = plane_of(r, c)
    ops = inplane_ops(r, c, plane, side)
    pos = slice_positions(ipps, plane, side)
    return plane, ops, pos


def stack_plan(pos):
    """Frame order (ascending position, repeated positions dropped), sorted positions, and the slice-averaging group size k."""
    pos = np.asarray(pos, float)
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    keep = np.r_[True, np.diff(pos) > 1e-3]
    order, pos = order[keep], pos[keep]
    k = 1
    if len(pos) > 1:
        sp = float(np.median(np.diff(pos)))
        k = int(np.ceil(MIN_SPACING / sp - 1e-6)) if sp < MIN_SPACING else 1
    return order, pos, k


def n_stored(pos):
    _, p, k = stack_plan(pos)
    return len(p) // k if k > 1 else len(p)


def process_series(frames, pos, ops, spacing_rc, px_mm=PX_MM, size=SIZE):
    """frames: list of 2D arrays (stored pixel values); pos: slice positions (mm); ops: inplane ops;
    spacing_rc: (row, col) pixel spacing in mm of the stored frames. Returns (uint8 volume D x size x size, positions)."""
    order, pos, k = stack_plan(pos)
    vol = np.stack([np.asarray(frames[i], np.float32) for i in order])
    if k > 1:  # thin stacks: average groups of k slices
        n = len(pos) // k * k
        vol = vol[:n].reshape(-1, k, *vol.shape[1:]).mean(1)
        pos = pos[:n].reshape(-1, k).mean(1)
    sr, sc = spacing_rc
    if ops[0]:
        sr, sc = sc, sr
    sub = vol[:, ::4, ::4]
    lo, hi = np.percentile(sub, [LO_PCT, HI_PCT])
    hi = max(hi, lo + 1e-3)
    out = np.zeros((len(vol), size, size), np.uint8)
    for i, img in enumerate(vol):
        img = np.ascontiguousarray(apply_inplane(img, ops))
        h, w = img.shape
        nh, nw = max(1, round(h * sr / px_mm)), max(1, round(w * sc / px_mm))
        interp = cv2.INTER_AREA if nh * nw < h * w else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
        img = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        # centre crop / pad to size
        y0, x0 = (nh - size) // 2, (nw - size) // 2
        sy, sx = max(y0, 0), max(x0, 0)
        crop = img[sy:sy + size, sx:sx + size]
        dy, dx = max(-y0, 0), max(-x0, 0)
        out[i, dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
    return out, pos.astype(np.float32)
