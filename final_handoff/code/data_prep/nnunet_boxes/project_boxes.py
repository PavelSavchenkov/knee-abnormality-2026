"""Per-slice structure boxes for every cached series of a study, from the nnU-Net Knee boxes of its sagittal FS series.

Each 3D box of out_all/boxes.csv (in the sagittal series' cache grid) is filled with a 16 x 24 x 24 point lattice, mapped to patient
coordinates (geom.SeriesGeom.to_patient) and into each target series (to_cache). For every target slab, the lattice points within
half a slab spacing of its plane give that slice's box (y0, y1, x0, x1 in the target's cache grid) and `frac` = share of the
lattice that falls in this slab (a coverage weight). Sagittal series of the same study get the same treatment (no special case).

    conda run -n rsna python work/knee_seg/project_boxes.py [--studies N] [--plot]
Boxes: nnunet_knee/out_all/rows_v2/ (rederive.py: side-table medial / lateral, compartments without both menisci, mcl) when it
exists, else rows/ (mask_job.py) with add_mcl. Output: nnunet_knee/out_all/slice_boxes.parquet: StudyInstanceUID, SeriesInstanceUID, plane, slice, row (cache row), structure,
y0, y1, x0, x1, frac. With --plot: nnunet_knee/out_all/proj_<study>.png (femur / tibia / patella / joint boxes on 3 slices of
the coronal and axial series)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.cache import SliceCache  # noqa: E402
from geom import SeriesGeom  # noqa: E402

OUT = HERE / "nnunet_knee/out_all"
STRUCT = ["femur", "tibia", "patella", "med_men", "lat_men", "fem_cart", "med_tib_cart", "lat_tib_cart", "pat_cart",
          "joint", "comp_medial", "comp_lateral", "notch", "pf_joint", "suprapatellar", "popliteal", "mcl", "hoffa"]


def add_mcl(boxes, meta):
    """Derived MCL box in the sagittal source grid: from 10 mm medial of the medial meniscus to 1 slice inside it, joint row
    -50 mm (femoral attachment) to +60 mm (tibial attachment), middle half of the tibia's A-P extent. Uses the model's medial
    meniscus label on the medial side (meniscus_order 1), else the lateral-named one."""
    out = []
    sp = meta.set_index("SeriesInstanceUID").slab_spacing
    order = meta.set_index("SeriesInstanceUID").get("meniscus_order", pd.Series(dtype=float))
    for uid, g in boxes.groupby("SeriesInstanceUID"):
        b = g.set_index("structure")
        if "mcl" in b.index:                                  # rows_v2 already has an mcl box
            continue
        mm_name = "med_men" if order.get(uid, 1) == 1 else "lat_men"
        if not {mm_name, "joint", "tibia"} <= set(b.index):
            continue
        m, j, t = b.loc[mm_name], b.loc["joint"], b.loc["tibia"]
        jr, w = (j.y0 + j.y1) / 2, t.x1 - t.x0
        out.append({"SeriesInstanceUID": uid, "structure": "mcl", "s0": max(0, int(m.s0 - round(10 / sp[uid]))), "s1": int(m.s0 + 1),
                    "y0": max(0, int(jr - 125)), "y1": min(399, int(jr + 150)), "x0": int(t.x0 + 0.25 * w), "x1": int(t.x0 + 0.75 * w)})
    return pd.concat([boxes, pd.DataFrame(out)], ignore_index=True)


def lattice(b, n=(16, 24, 24)):
    s = np.linspace(b.s0, b.s1, n[0])
    y = np.linspace(b.y0, b.y1, n[1])
    x = np.linspace(b.x0, b.x1, n[2])
    S, Y, X = np.meshgrid(s, y, x, indexing="ij")
    return S.ravel(), Y.ravel(), X.ravel()


def project_study(cs, boxes, series_meta, study):
    src = series_meta[series_meta.StudyInstanceUID == study]
    if not len(src):
        return []
    src = src.sort_values("n", ascending=False).iloc[0]                      # one sagittal FS source per study
    gsrc = SeriesGeom(src.SeriesInstanceUID)
    bs = boxes[(boxes.SeriesInstanceUID == src.SeriesInstanceUID) & boxes.structure.isin(STRUCT)]
    rows = []
    for t in cs[(cs.StudyInstanceUID == study) & (cs.dup_of == "")].itertuples():
        gt = SeriesGeom(t.SeriesInstanceUID)
        half = float(t.slab_spacing) / 2
        for b in bs.itertuples():
            S, Y, X = lattice(b)
            xyz = gsrc.to_patient(np.clip(np.round(S).astype(int), 0, gsrc.n - 1), Y, X)
            s, y, x, d = gt.to_cache(xyz)
            ok = (d <= half) & (y >= 0) & (y < 400) & (x >= 0) & (x < 400)
            for si in np.unique(s[ok]):
                m = ok & (s == si)
                rows.append((study, t.SeriesInstanceUID, t.plane, int(si), int(t.start + si), b.structure, int(y[m].min()), int(y[m].max()),
                             int(x[m].min()), int(x[m].max()), float(m.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", type=int, default=0)
    ap.add_argument("--plot", nargs="?", const="bones", default="", help="bones (default), sides or anterior")
    ap.add_argument("--out", default="")
    ap.add_argument("--v1", action="store_true", help="use mask_job's original rows/ instead of rederive.py's rows_v2/")
    ap.add_argument("--only", default="", help="parquet whose index lists the studies to project (e.g. the attention dump)")
    a = ap.parse_args()
    c = SliceCache("fov160_px400_q95")
    cs = c.series
    import json
    rows_dir = OUT / ("rows_v2" if (OUT / "rows_v2").exists() and not a.v1 else "rows")      # rows_v2: nnunet_knee/rederive.py
    js = [json.load(open(p)) for p in sorted(rows_dir.glob("*.json"))]
    print("boxes from", rows_dir.name, len(js), "series", flush=True)
    boxes = pd.DataFrame([b for j in js for b in j["boxes"]])
    meta = pd.DataFrame([j["meta"] for j in js])
    boxes = add_mcl(boxes, meta)
    meta = meta[meta.ok == 1]
    studies = meta.StudyInstanceUID.unique()
    if a.only:
        studies = [st for st in studies if st in set(pd.read_parquet(a.only).index)]
    if a.studies:
        studies = studies[: a.studies]
    rows = [r for st in studies for r in project_study(cs, boxes, meta, st)]
    d = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "plane", "slice", "row", "structure", "y0", "y1", "x0", "x1", "frac"])
    d.to_parquet(OUT / (a.out or "slice_boxes.parquet"))
    print(len(d), "rows for", d.StudyInstanceUID.nunique(), "studies;", d.groupby("plane").SeriesInstanceUID.nunique().to_dict())
    if a.plot:
        import cv2
        col = {"femur": (255, 0, 0), "tibia": (0, 0, 255), "patella": (255, 0, 255), "joint": (0, 255, 0), "notch": (0, 255, 255)}
        if a.plot == "sides":                              # medial (red) / lateral (blue) compartments and the MCL box (green)
            col = {"comp_medial": (0, 0, 255), "comp_lateral": (255, 0, 0), "mcl": (0, 255, 0)}
        if a.plot == "anterior":                           # suprapatellar (yellow), hoffa (magenta), pf_joint (cyan)
            col = {"suprapatellar": (0, 255, 255), "hoffa": (255, 0, 255), "pf_joint": (255, 255, 0)}
        for st in studies:
            tiles = []
            for plane in ("Sagittal", "Coronal", "Axial"):
                ser = cs[(cs.StudyInstanceUID == st) & (cs.plane == plane) & (cs.dup_of == "")]
                if not len(ser):
                    continue
                t = ser.iloc[0]
                for q in (0.3, 0.5, 0.7):
                    si = int(q * (t.n - 1))
                    img = cv2.cvtColor(c.read([t.start + si])[0], cv2.COLOR_GRAY2BGR)
                    for b in d[(d.SeriesInstanceUID == t.SeriesInstanceUID) & (d.slice == si) & d.structure.isin(col)].itertuples():
                        cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), col[b.structure], 2)
                    cv2.putText(img, f"{plane[:3]} {si}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    tiles.append(cv2.resize(img, (240, 240)))
            cv2.imwrite(str(OUT / f"proj_{st[-12:]}.png"), np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)], 0))


if __name__ == "__main__":
    main()
