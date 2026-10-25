"""Project the nnU-Net Knee label volumes (out_all/masks/{sag FS series}.npz, 'lab' (n, 400, 400) in that series' cache grid) onto every
CORONAL series of the same study (10-18, Experts V3 U44 / U47: coronal masked arms - LOA on COR FS with the menisci hidden, the MCL zoom with
marrow hidden). For each target slab, a 200 x 200 grid of cache pixels is mapped to patient coordinates and into the sagittal series
(geom.SeriesGeom); a pixel takes the nearest sagittal voxel's label when its distance to the nearest sagittal slab plane is <= half the
sagittal slab spacing, else 0 (unknown = kept by the masking code); the grid is upsampled x2 (nearest) to 400 x 400. Output
out_all/masks_cor/{coronal series uid}.npz 'lab' (n_target, 400, 400) uint8 + 'src' (the sagittal uid).
conda run -n rsna python work/knee_seg/project_masks.py [--studies N] [--workers 12] [--plot STUDY]"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from common.cache import SliceCache  # noqa: E402
from geom import SeriesGeom  # noqa: E402

MASKS, OUT = HERE / "nnunet_knee/out_all/masks", HERE / "nnunet_knee/out_all/masks_cor"
G = 200


def project(job):
    src_uid, tgt_uid = job
    out = OUT / f"{tgt_uid}.npz"
    if out.exists():
        return tgt_uid, "exists"
    try:
        lab = np.load(MASKS / f"{src_uid}.npz")["lab"]
        S, T = SeriesGeom(src_uid), SeriesGeom(tgt_uid)
        if len(lab) != S.n:
            return tgt_uid, f"mask slabs {len(lab)} != series {S.n}"
        half = np.median(np.abs(np.diff(S.centres @ S.normal))) / 2 if S.n > 1 else 2.0
        yy, xx = np.meshgrid((np.arange(G) + 0.5) * 400 / G - 0.5, (np.arange(G) + 0.5) * 400 / G - 0.5, indexing="ij")
        res = np.zeros((T.n, 400, 400), np.uint8)
        for s in range(T.n):
            ss, y, x, d = S.to_cache(T.to_patient(np.full(yy.size, s), yy.ravel(), xx.ravel()))
            yi, xi = np.rint(y).astype(int), np.rint(x).astype(int)
            ok = (d <= half) & (yi >= 0) & (yi < 400) & (xi >= 0) & (xi < 400)
            v = np.zeros(yy.size, np.uint8)
            v[ok] = lab[ss[ok], yi[ok], xi[ok]]
            res[s] = np.repeat(np.repeat(v.reshape(G, G), 2, 0), 2, 1)
        np.savez_compressed(out, lab=res, src=src_uid)
        return tgt_uid, "ok"
    except Exception as e:                                   # one bad series must not stop the batch; reported in the summary
        return tgt_uid, f"error {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    ser = SliceCache("fov160_px400_q95").series
    have = {f.stem for f in MASKS.glob("*.npz")}
    src = ser[ser.SeriesInstanceUID.isin(have)].set_index("StudyInstanceUID").SeriesInstanceUID
    src = src[~src.index.duplicated()]
    tgt = ser[(ser.plane == "Coronal") & ser.StudyInstanceUID.isin(src.index) & (ser.dup_of == "")]
    jobs = [(src[st], u) for st, u in zip(tgt.StudyInstanceUID, tgt.SeriesInstanceUID)]
    if a.studies:
        keep = set(list(dict.fromkeys(tgt.StudyInstanceUID))[: a.studies])
        jobs = [j for j, st in zip(jobs, tgt.StudyInstanceUID) if st in keep]
    print(f"{len(jobs)} coronal series of {len({j[0] for j in jobs})} studies with a sagittal FS mask")
    with Pool(a.workers) as p:
        res = p.map(project, jobs, chunksize=4)
    st = {}
    for _, r in res:
        st[r.split(":")[0]] = st.get(r.split(":")[0], 0) + 1
    print(st)
    bad = [(u, r) for u, r in res if r not in ("ok", "exists")]
    for u, r in bad[:10]:
        print(u, r)


if __name__ == "__main__":
    main()
