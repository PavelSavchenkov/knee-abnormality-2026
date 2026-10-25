"""Structure ZOOM crops for VLM reads of focal findings (USER 10-19 ~19:3x: "maybe we need to run VLM labelling on some zoomed images,
because findings are focal???"). Uses the zoom members' own localised crops (experiments/v3/oct16_experts/syn/syn_data.build_bags: the
localiser boxes, 40-48 mm views) instead of the contact sheets: per bag, the middle slice of N_PER windows spread over the structure's
slices, grey, resized to PX x PX (one image per window).
Bags -> findings: notch40 (Sagittal + Coronal) ACL; mclz (Coronal) MCL; medc40 (Coronal + Sagittal) Medial Meniscus / Medial OA;
latc40 (Coronal + Sagittal) Lateral Meniscus / Lateral OA; pfl40 (Sagittal + Axial) PF OA.
Writes zoom/<index>/<bag>_<plane>_<k>.png. Usage: python render_zoom.py STUDIES.csv [--limit N]"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments/v3/oct16_experts/syn"), str(ROOT / "experiments/v3/oct16_experts"), str(ROOT)]
from syn_data import build_bags  # noqa: E402

from common.cache import SliceCache  # noqa: E402

HERE = Path(__file__).parent
BAGS = ["notch40", "mclz", "medc40", "latc40", "pfl40"]
N_PER, PX = 2, 448                                    # windows per (bag, plane), output size


def crop(img, cy, cx, side, flip):
    h = w = int(round(side))
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    out = np.zeros((h, w), np.uint8)
    sy, sx, ey, ex = max(y0, 0), max(x0, 0), min(y0 + h, img.shape[0]), min(x0 + w, img.shape[1])
    if ey > sy and ex > sx:
        out[sy - y0:ey - y0, sx - x0:ex - x0] = img[sy:ey, sx:ex]
    out = out[:, ::-1] if flip else out
    return cv2.resize(np.ascontiguousarray(out), (PX, PX), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("studies")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bags", default=",".join(BAGS))
    a = ap.parse_args()
    st = pd.read_csv(a.studies, index_col="index")
    st = st.iloc[:a.limit] if a.limit else st
    bags = build_bags(a.bags.split(","), clamp=True, plane_bags=True, fs_only=True)     # FS windows where a bag has any
    cache = SliceCache()
    plane_of = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    for i, uid in st.StudyInstanceUID.items():
        d = HERE / "zoom" / str(i)
        if uid not in bags:
            continue
        d.mkdir(parents=True, exist_ok=True)
        for key, cands in bags[uid].items():
            if not cands:
                continue
            bag, plane = key.split(":")
            if (d / f"{bag}_{plane}_0.png").exists():
                continue
            pick = [cands[j] for j in np.unique(np.linspace(0, len(cands) - 1, N_PER + 2)[1:-1].round().astype(int))]
            for k, w in enumerate(pick):
                r0, r1, r2, flip, cy, cx, side, *_ = w
                img = cache.read(np.array([r1]))[0]
                img = img[..., 0] if img.ndim == 3 else img
                cv2.imwrite(str(d / f"{bag}_{plane}_{k}.png"), crop(img, cy, cx, side, flip))
    print("done", len(st))


if __name__ == "__main__":
    main()
