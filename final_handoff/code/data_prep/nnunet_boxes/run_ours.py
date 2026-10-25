"""Run nnU-Net Knee (3d_lowres) on our cached series (fov160_px400_q95: canonical orientation, 0.4 mm/px) and save overlays.
Sagittal goes in as is (slices medial->lateral, rows S->I, cols A->P). Coronal (slices A->P, rows S->I, cols M->L) and axial
(slices I->S, rows A->P, cols M->L) are reformatted into the same (M->L, S->I, A->P) grid. Writes out/<study>_<series_type>.png and
out/summary.csv (voxel counts per class)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402

from common.cache import SliceCache  # noqa: E402
from infer import CLASSES, load, predict  # noqa: E402

torch.set_num_threads(6)
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
COLORS = np.array([[0, 0, 0], [255, 0, 255], [0, 255, 255], [0, 128, 255], [255, 128, 0], [0, 255, 0], [255, 255, 0], [255, 0, 0],
                   [0, 0, 255], [128, 0, 255]], np.uint8)
c = SliceCache("fov160_px400_q95")
s = c.series[c.series.dup_of == ""]
rng = np.random.default_rng(1)
studies = rng.choice(s.StudyInstanceUID.unique(), 10, replace=False)
net, cfg = load("3d_lowres")
rows = []
for st in studies:
    ss = s[s.StudyInstanceUID == st]
    picks = []
    for plane, pref in (("Sagittal", "SAG_PD_FS"), ("Sagittal", "SAG_T1_NOFS")):
        p = ss[ss.plane == plane]
        if len(p):
            r = p[p.series_type == pref].iloc[0] if (p.series_type == pref).any() else p.iloc[0]
            if all(r.SeriesInstanceUID != q.SeriesInstanceUID for q in picks):
                picks.append(r)
    for r in picks:
        v = c.read(np.arange(r.start, r.start + r.n)).astype(np.float32)       # (slice, row, col) canonical
        sp_slice, sp = float(r.slab_spacing), c.meta["mm_per_px"]
        if r.plane == "Sagittal":
            vol, spacing = v, (sp_slice, sp, sp)
        elif r.plane == "Coronal":
            vol, spacing = v.transpose(2, 1, 0), (sp, sp, sp_slice)              # (M->L, S->I, A->P)
        else:
            vol, spacing = v.transpose(2, 0, 1)[:, ::-1], (sp, sp_slice, sp)     # (M->L, S->I, A->P)
        lab = predict(net, cfg, np.ascontiguousarray(vol), spacing)
        counts = np.bincount(lab.ravel(), minlength=len(CLASSES))
        rows.append({"study": st[-12:], "plane": r.plane, "type": r.series_type, **{k: int(n) for k, n in zip(CLASSES, counts)}})
        # overlays on 3 sagittal-grid slices (25 / 50 / 75 % of the M->L axis) + for cor/ax also one slice in the native plane
        tiles = []
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            i = int(q * (vol.shape[0] - 1))
            img = cv2.cvtColor(np.clip(vol[i] / max(np.percentile(vol, 99.5), 1) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            ov = COLORS[lab[i]]
            m = lab[i] > 0
            img[m] = (0.5 * img[m] + 0.5 * ov[m]).astype(np.uint8)
            tiles.append(cv2.resize(img, (260, 260)))
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
        cv2.imwrite(str(OUT / f"{st[-12:]}_{r.series_type}.png"), np.concatenate(tiles, 1))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
