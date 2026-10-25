"""Series slots and slice sampling (shared by training and inference; numpy / pandas only)."""
import numpy as np
import pandas as pd

# slot -> (plane, fs, weighting preference)
SLOTS = {
    "sag_fs": ("Sagittal", 1, ["PD", "T2", "GRE", "T1", "UNK"]),
    "cor_fs": ("Coronal", 1, ["PD", "T2", "GRE", "T1", "UNK"]),
    "ax_fs": ("Axial", 1, ["PD", "T2", "GRE", "T1", "UNK"]),
    "sag_nofs": ("Sagittal", 0, ["PD", "T2", "T1", "GRE", "UNK"]),
    "cor_nofs": ("Coronal", 0, ["T1", "PD", "T2", "GRE", "UNK"]),
    "ax_nofs": ("Axial", 0, ["T1", "PD", "T2", "GRE", "UNK"]),
    # second series of a plane / FS flag: a different weighting than the slot named in SECOND (no duplicates)
    "sag_fs2": ("Sagittal", 1, ["T2", "PD", "GRE", "T1", "UNK"]),
    "cor_nofs2": ("Coronal", 0, ["T2", "PD", "T1", "GRE", "UNK"]),
    "ax_fs2": ("Axial", 1, ["T2", "PD", "GRE", "T1", "UNK"]),
    "sag_nofs2": ("Sagittal", 0, ["T1", "T2", "PD", "GRE", "UNK"]),
    "cor_fs2": ("Coronal", 1, ["T2", "PD", "GRE", "T1", "UNK"]),
}
SECOND = {"sag_fs2": "sag_fs", "cor_nofs2": "cor_nofs", "ax_fs2": "ax_fs", "sag_nofs2": "sag_nofs", "cor_fs2": "cor_fs"}


def select_slots(meta, slots):
    """DataFrame study x slot -> SeriesInstanceUID (NaN if missing). Other-knee series only as a last resort;
    ties broken by weighting preference, then more slices."""
    m = meta.reset_index()
    out = {}
    for slot in sorted(slots, key=lambda sl: sl in SECOND):  # primary slots first
        plane, fs, pref = SLOTS[slot]
        d = m[(m.plane == plane) & (m.fs == fs)].copy()
        if slot in SECOND:  # exclude the primary slot's series and its weighting
            prim = out[SECOND[slot]] if SECOND[slot] in out else select_slots(meta, [SECOND[slot]])[SECOND[slot]]
            w_prim = d.set_index("SeriesInstanceUID").weighting.reindex(prim.dropna().to_numpy())
            w_prim.index = prim.dropna().index
            d = d[~d.SeriesInstanceUID.isin(prim.dropna().to_numpy()) & (d.weighting != d.StudyInstanceUID.map(w_prim))]
            d = d[~d.other_knee]
        d["w_rank"] = d.weighting.map({w: i for i, w in enumerate(pref)})
        d = d.sort_values(["StudyInstanceUID", "other_knee", "w_rank", "n"], ascending=[True, True, True, False])
        out[slot] = d.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").SeriesInstanceUID
    return pd.DataFrame(out)[list(slots)]


def triplet_indices(n, T, lo, hi, jitter, rng, shift=False):
    """T centres evenly spaced over [lo, hi] fraction of the stack (+-1 slice jitter in training; shift: the whole grid moved
    by a random offset in [-stride/2, stride/2]); returns (T, 3) indices."""
    c = np.linspace(lo * (n - 1), hi * (n - 1), T)
    if shift:
        c = c + rng.uniform(-0.5, 0.5) * (c[1] - c[0] if T > 1 else 0)
    c = np.round(c).astype(int)
    if jitter:
        c = c + rng.integers(-1, 2, size=T)
    c = np.clip(c, 0, n - 1)
    return np.clip(c[:, None] + np.array([-1, 0, 1])[None], 0, n - 1)


def slab_k(spacing):
    """Slices averaged per channel for thin series (stored spacing < 2.5 mm): round(3.3 mm / spacing), so a triplet looks like
    one of a ~3.3 mm series; 1 (no slab) otherwise."""
    return max(1, int(round(3.3 / spacing))) if spacing < 2.5 else 1


def series_spacing(positions, n_stored):
    """Stored slice spacing: span of the raw slice positions / (stored slices - 1); same formula in the cache and DICOM paths."""
    p = np.asarray(positions, float)
    return float((p.max() - p.min()) / max(n_stored - 1, 1)) if len(p) else 3.3


def slab_indices(idx, k, n):
    """(T, 3) triplet indices (c-1, c, c+1) -> (T, 3, k): channel centres c-k, c, c+k, each the k adjacent slices from
    centre - (k-1)//2; k = 1 -> idx[..., None] (unchanged triplets)."""
    if k <= 1:
        return idx[..., None]
    c = idx[:, 1]
    return np.clip((c[:, None] + np.array([-k, 0, k])[None])[..., None] + (np.arange(k) - (k - 1) // 2)[None, None], 0, n - 1)


def slab_mean(stack):
    """(..., k, H, W) uint8 slices -> (..., H, W) uint8 rounded mean (identical in all paths)."""
    k = stack.shape[-3]
    return stack[..., 0, :, :] if k == 1 else ((stack.astype(np.uint16).sum(-3) + k // 2) // k).astype(np.uint8)


WEIGHTINGS = ["T1", "PD", "T2", "GRE", "UNK"]


def slot_alternatives(meta, slots):
    """(study, slot) -> list of same-knee series with the slot's plane and fat-suppression flag (any weighting)."""
    m = meta.reset_index()
    m = m[~m.other_knee]
    out = {}
    for slot in slots:
        plane, fs, _ = SLOTS[slot]
        d = m[(m.plane == plane) & (m.fs == fs)]
        for study, g in d.groupby("StudyInstanceUID"):
            out[(study, slot)] = list(g.SeriesInstanceUID)
    return out
