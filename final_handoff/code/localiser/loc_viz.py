"""Visual check of the localiser on held-out studies (fold 0, non-gold).
Sheet 1 (out/loc_viz_boxes_<tag>.png): per study, one series per plane (FS preferred); the slice with the highest presence of
each structure, with the localiser series box (thick) and the nnU-Net box of that slice (thin): medial red, lateral green,
PF blue. Sheet 2 (out/loc_viz_crops_<tag>.png): per study, the crops oa_data would feed for the three bags (4 per bag).
CLI: python scripts/loc_viz.py [--n 6] [--seed 0] [--threads 4] [--tag a]"""
import argparse
import sys
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
from common.folds import get_folds  # noqa: E402
from loc_data import PX, STRUCTS  # noqa: E402
from loc_infer import load_series  # noqa: E402
from loc_train import Loc, aux_features  # noqa: E402

COL = {"comp_medial": (0, 0, 255), "comp_lateral": (0, 200, 0), "pf_joint": (255, 120, 0)}
CROP_MM = {("comp_medial", "Sagittal"): 72, ("comp_medial", "Coronal"): 56, ("comp_medial", "Axial"): 56,
           ("comp_lateral", "Sagittal"): 72, ("comp_lateral", "Coronal"): 56, ("comp_lateral", "Axial"): 56,
           ("pf_joint", "Sagittal"): 64, ("pf_joint", "Axial"): 64}


def run(model, job):
    uid, plane, rows, pos, img = load_series(job)
    n = len(rows)
    w = np.stack([img[np.clip(np.arange(n) + d, 0, n - 1)] for d in (-1, 0, 1)], 1)
    meta = pd.DataFrame({"plane": plane, "pos_mm": pos, "idx": np.arange(n), "n_slices": n})
    with torch.no_grad():
        p, b = model(torch.from_numpy(w), torch.from_numpy(aux_features(meta)))
    return rows, torch.sigmoid(p).numpy(), b.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    c = SliceCache("fov160_px400_q95")
    model = Loc("mobilenetv3_large_100", pretrained=False)
    model.load_state_dict(torch.load(EXP / "out/loc_mnv3.pt", map_location="cpu"))
    model.eval()
    side = pd.read_csv(ROOT / "data/derived/series_side_final.csv", index_col=0)
    flips = set(side.index[side.flip_final])
    boxes = pd.read_parquet(ROOT / "data/derived/slice_boxes.parquet", columns=["row", "structure", "y0", "y1", "x0", "x1"])
    boxes = boxes[boxes.structure.isin(COL)].set_index(["row", "structure"])
    f = get_folds(5, 0)
    studies = pd.Series(f.index[f.fold == 0]).sample(a.n, random_state=a.seed).tolist()
    s = c.series[(c.series.dup_of == "")]
    pos_all = c.slices.sort_values("row").pos_mm.to_numpy(np.float32)
    sheet, crops_sheet = [], []
    for sid in studies:
        row_imgs, crop_row = [], []
        for plane in ("Sagittal", "Coronal", "Axial"):
            cand = s[(s.StudyInstanceUID == sid) & (s.plane == plane)].sort_values("fs", ascending=False)
            if not len(cand):
                row_imgs.append(np.zeros((400, 400, 3), np.uint8))
                continue
            r = cand.iloc[0]
            flip = r.SeriesInstanceUID in flips
            rows, p, b = run(model, (r.SeriesInstanceUID, plane, r.start, r.n, flip, pos_all[r.start:r.start + r.n]))
            full = c.read(rows)
            if flip and plane != "Sagittal":
                full = full[..., ::-1]
            panels = []
            for st in COL:
                j = STRUCTS.index(st)
                if (st, plane) not in CROP_MM and st == "pf_joint":
                    continue
                k = int(p[:, j].argmax())
                img = cv2.cvtColor(np.ascontiguousarray(full[k]), cv2.COLOR_GRAY2BGR)
                pres = p[:, j] > 0.5
                if pres.any():
                    bb = (b[pres, j] * p[pres, j, None]).sum(0) / p[pres, j].sum() * 400
                    cv2.rectangle(img, (int(bb[2]), int(bb[0])), (int(bb[3]), int(bb[1])), COL[st], 3)
                    cy, cx, half = (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2, CROP_MM.get((st, plane), 56) / 0.4 / 2
                    cv2.rectangle(img, (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half)), (0, 255, 255), 1)
                    # crops for sheet 2 (4 present slices)
                    for i in np.linspace(np.flatnonzero(pres)[0], np.flatnonzero(pres)[-1], 4).round().astype(int):
                        im = full[i]
                        if st == "comp_lateral" and plane != "Sagittal":
                            im = im[:, ::-1]
                            ccx = 400 - cx
                        else:
                            ccx = cx
                        pad = np.pad(im, 200)
                        y0, x0 = int(cy - half) + 200, int(ccx - half) + 200
                        crop = cv2.resize(np.ascontiguousarray(pad[y0:y0 + int(2 * half), x0:x0 + int(2 * half)]), (112, 112))
                        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(crop, (0, 0), (111, 111), COL[st], 2)
                        crop_row.append(crop)
                if (rows[k], st) in boxes.index:
                    t = boxes.loc[(rows[k], st)]
                    t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
                    if flip and plane != "Sagittal":
                        x0t, x1t = 400 - t.x1, 400 - t.x0
                    else:
                        x0t, x1t = t.x0, t.x1
                    cv2.rectangle(img, (int(x0t), int(t.y0)), (int(x1t), int(t.y1)), COL[st], 1)
                cv2.putText(img, f"{plane[:3]} {st} p{p[k, j]:.2f}{' FLIP' if flip else ''}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                panels.append(cv2.resize(img, (240, 240)))
            row_imgs.append(np.concatenate(panels, 1) if panels else np.zeros((240, 240, 3), np.uint8))
        w = max(x.shape[1] for x in row_imgs)
        sheet.append(np.concatenate([np.pad(x if x.shape[0] == 240 else cv2.resize(x, (240, 240)), ((0, 0), (0, w - x.shape[1] if x.shape[0] == 240 else w - 240), (0, 0)))
                                     for x in row_imgs], 0))
        crops_sheet.append(np.concatenate(crop_row + [np.zeros((112, 112, 3), np.uint8)] * (32 - len(crop_row)), 1)[:, :112 * 16]
                           if crop_row else np.zeros((112, 112 * 16, 3), np.uint8))
    cv2.imwrite(str(EXP / f"out/loc_viz_boxes_{a.tag}.png"), np.concatenate(sheet[:3], 0))
    if len(sheet) > 3:
        cv2.imwrite(str(EXP / f"out/loc_viz_boxes_{a.tag}2.png"), np.concatenate(sheet[3:], 0))
    cv2.imwrite(str(EXP / f"out/loc_viz_crops_{a.tag}.png"), np.concatenate(crops_sheet, 0))
    print(studies)


if __name__ == "__main__":
    main()
