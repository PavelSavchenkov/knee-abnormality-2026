"""Localiser predictions for every slice of every non-duplicate series (all studies incl. gold), in the side-corrected view the
models train on (series in the side table's flip_final: coronal / axial images flipped left-right, sagittal slice order reversed
and positions negated, as main/data.py build_index does with side_fix).

Per slice (corrected order): presence probability and box (y0, y1, x0, x1, fractions of the 400 px cache image, corrected view)
for each localiser structure. Series level (written separately): presence-weighted mean box over slices with p > 0.5 and the
number of such slices.
CLI: python scripts/loc_infer.py [--model out/loc_mnv3.pt] [--threads 20]
Output: data/loc_pred/slices.parquet (SeriesInstanceUID, StudyInstanceUID, plane, idx (corrected), row (cache row), pos_mm
(corrected), p_*, box columns), data/loc_pred/series.parquet"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.cache import SliceCache  # noqa: E402
from loc_data import PX, STRUCTS, STRUCTS_V2, STRUCTS_V3  # noqa: E402
from loc_train import Loc, aux_features  # noqa: E402

_c = None


def load_series(r):
    """(uid, plane, rows corrected order, pos corrected, uint8 (n, PX, PX) corrected view)."""
    global _c
    if _c is None:
        _c = SliceCache("fov160_px400_q95")
    uid, plane, start, n, flip, pos = r
    rows = start + np.arange(n)
    img = np.stack([cv2.resize(a, (PX, PX), interpolation=cv2.INTER_AREA) for a in _c.read(rows)])
    if flip and plane == "Sagittal":
        rows, img, pos = rows[::-1].copy(), img[::-1].copy(), -pos[::-1]
    elif flip:
        img = img[..., ::-1].copy()
    return uid, plane, rows, pos, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(EXP / "out/loc_mnv3.pt"))
    ap.add_argument("--arch", default="mobilenetv3_large_100")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    c = SliceCache("fov160_px400_q95")
    side = pd.read_csv(ROOT / "data/derived/series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    s = c.series[c.series.dup_of == ""]
    pos_all = c.slices.sort_values("row").pos_mm.to_numpy(np.float32)
    jobs = [(r.SeriesInstanceUID, r.plane, r.start, r.n, r.SeriesInstanceUID in flips, pos_all[r.start:r.start + r.n])
            for r in s.itertuples()]
    study = dict(zip(s.SeriesInstanceUID, s.StudyInstanceUID))
    structs = {"v1": STRUCTS, "v2": STRUCTS_V2, "v3": STRUCTS_V3}[a.version]
    model = Loc(a.arch, pretrained=False, structs=structs)
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model = model.to(a.device).eval()
    out = []

    def results(pool):                                    # bounded: imap buffers all finished results, so feed 300 series at a time
        for a in range(0, len(jobs), 300):
            yield from pool.imap(load_series, jobs[a:a + 300], chunksize=4)

    with Pool(6) as pool, torch.no_grad():
        for k, (uid, plane, rows, pos, img) in enumerate(results(pool)):
            n = len(rows)
            w = np.stack([img[np.clip(np.arange(n) + d, 0, n - 1)] for d in (-1, 0, 1)], 1)
            meta = pd.DataFrame({"plane": plane, "pos_mm": pos, "idx": np.arange(n), "n_slices": n})
            p, b = model(torch.from_numpy(w).to(a.device), torch.from_numpy(aux_features(meta)).to(a.device))
            p, b = torch.sigmoid(p).cpu().numpy(), b.cpu().numpy()
            d = {"SeriesInstanceUID": uid, "StudyInstanceUID": study[uid], "plane": plane, "idx": np.arange(n), "row": rows, "pos_mm": pos}
            for j, st in enumerate(structs):
                d[f"p_{st}"] = p[:, j].astype(np.float16)
                for i, col in enumerate(("y0", "y1", "x0", "x1")):
                    d[f"{st}_{col}"] = b[:, j, i].astype(np.float16)
            out.append(pd.DataFrame(d))
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)} series", flush=True)
    sl = pd.concat(out, ignore_index=True)
    dst = EXP / ("data/loc_pred" if a.version == "v1" else f"data/loc_pred_{a.version}")
    dst.mkdir(parents=True, exist_ok=True)
    sl.to_parquet(dst / "slices.parquet")
    ser = []
    for st in structs:
        x = sl[sl[f"p_{st}"] > 0.5]
        w = x[f"p_{st}"].astype(np.float32)
        agg = pd.DataFrame({col: (x[f"{st}_{col}"].astype(np.float32) * w).groupby(x.SeriesInstanceUID).sum() / w.groupby(x.SeriesInstanceUID).sum()
                            for col in ("y0", "y1", "x0", "x1")})
        agg["n_present"] = w.groupby(x.SeriesInstanceUID).size()
        ser.append(agg.add_prefix(f"{st}_"))
    pd.concat(ser, axis=1).to_parquet(dst / "series.parquet")
    print(f"done: {len(sl)} slices, {sl.SeriesInstanceUID.nunique()} series", flush=True)


if __name__ == "__main__":
    main()
