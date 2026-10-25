"""Re-derive structure boxes from the saved nnU-Net Knee masks (out_all/masks/*.npz) without re-running the network.

Why: mask_job.derived named medial / lateral by the model's meniscus classes and needed both menisci for the compartment boxes.
- The model misses one meniscus in ~35% of series (mostly the medial one; T2 FS and thick slabs): those series got no
  comp_medial / comp_lateral (and no MCL box).
- The cache is mis-sided for ~2% of series (data/derived/series_side_final.csv); there the canonical medial side is the
  anatomical lateral side.
Rules here:
- medial side = canonical low slice index, swapped where the side table flips the series (flip_final). The model's own meniscus
  names agree with the table on all 32 flipped series checked and disagree on 44 unflipped ones (sides.csv: candidates for missed
  flips; several have p_flip 0.7-0.83 just under the 0.9 sagittal threshold).
- menisci (classes 5 / 6) and tibial cartilage (3 / 4) are named by position: voxels in the medial / lateral half of the tibia's
  slice extent, whatever the model called them.
- comp_medial / comp_lateral: the medial / lateral 40% of the tibia's slice extent, unioned with the meniscus half box (>= 30
  voxels) +- 1 slice when found; rows joint row +- 25 mm, columns the tibia +- 5 mm (as before). notch: middle third between the
  two compartment centres (meniscus centroids when found).
- mcl: 6 mm outside to 6 mm (>= 1 slice) inside the medial-most slice of femur / tibia / medial meniscus, rows joint -50 to +60 mm,
  middle half of the tibia's A-P extent.
- hoffa (new): infrapatellar fat pad, slices between the two compartment centres, rows 25 mm above to 5 mm below the joint row
  (anchoring to the patella's lower edge gave thin strips), columns 20 mm anterior to 15 mm posterior of the tibia's anterior edge (MOAKS Hoffa-synovitis:
  with effusion-synovitis, the non-contrast synovitis signs).
- pf_joint, suprapatellar, popliteal, joint and the bone / patellar / femoral cartilage boxes: unchanged.
Output: out_all/rows_v2/<series>.json ({"meta", "boxes"} like rows/, without the LPS fields; project_boxes.py reads rows_v2 when
it exists) and out_all/sides.csv.
    conda run -n rsna python work/knee_seg/nnunet_knee/rederive.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mask_job import CLASSES, MM, box, clip_box

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "out_all"
SIDES = ROOT / "data/derived/series_side_final.csv"


def derive(lab, sp, medial_low):
    n = lab.shape[0]
    r = lambda mm: int(round(mm / MM))
    rs = lambda mm: int(round(mm / sp))
    fem, tib, pat = lab == 7, lab == 8, lab == 9
    b_f, b_t, b_p = box(fem), box(tib), box(pat)
    out, info = {}, {}
    if b_f is None or b_t is None:
        return out, info
    both = [s for s in range(n) if fem[s].any() and tib[s].any()]
    if not both:
        return out, info
    rows_f = [np.nonzero(fem[s].any(1))[0].max() for s in both]
    rows_t = [np.nonzero(tib[s].any(1))[0].min() for s in both]
    jr = int(round((np.median(rows_f) + np.median(rows_t)) / 2))
    info["joint_row"] = jr
    s_lo, s_hi = min(b_f["s0"], b_t["s0"]), max(b_f["s1"], b_t["s1"])
    out["joint"] = {"s0": s_lo, "s1": s_hi, "y0": jr - r(15), "y1": jr + r(15), "x0": b_t["x0"], "x1": b_t["x1"]}
    tc = (b_t["s0"] + b_t["s1"]) / 2
    sl = np.arange(n)[:, None, None]
    med_half, lat_half = (sl <= tc, sl > tc) if medial_low else (sl >= tc, sl < tc)
    men, tcart = np.isin(lab, (5, 6)), np.isin(lab, (3, 4))
    for name, m in (("med_men", men & med_half), ("lat_men", men & lat_half), ("med_tib_cart", tcart & med_half), ("lat_tib_cart", tcart & lat_half)):
        if (b := box(m)) is not None:
            out[name] = b
    w = b_t["s1"] - b_t["s0"]
    part = {"medial": (b_t["s0"], b_t["s0"] + 0.4 * w) if medial_low else (b_t["s1"] - 0.4 * w, b_t["s1"]),
            "lateral": (b_t["s1"] - 0.4 * w, b_t["s1"]) if medial_low else (b_t["s0"], b_t["s0"] + 0.4 * w)}
    centre, info["men_found"] = {}, []
    for side, key in (("medial", "med_men"), ("lateral", "lat_men")):
        s0, s1 = int(np.floor(part[side][0])), int(np.ceil(part[side][1]))
        c = (s0 + s1) / 2
        m = out.get(key)
        if m is not None and m["voxels"] >= 30:                  # meniscus found: union with its box +- 1 slice
            s0, s1, c = min(s0, m["s0"] - 1), max(s1, m["s1"] + 1), m["cs"]
            info["men_found"].append(side)
        centre[side] = c
        out[f"comp_{side}"] = {"s0": s0, "s1": s1, "y0": jr - r(25), "y1": jr + r(25), "x0": b_t["x0"] - r(5), "x1": b_t["x1"] + r(5)}
    lo_s, hi_s = sorted((int(centre["medial"]), int(centre["lateral"])))
    third = max(1, (hi_s - lo_s) // 3)
    xc = (b_t["x0"] + b_t["x1"]) // 2
    out["notch"] = {"s0": lo_s + third, "s1": hi_s - third, "y0": jr - r(30), "y1": jr + r(10), "x0": xc - r(25), "x1": xc + r(20)}
    ext = [b_f, b_t] + ([out["med_men"]] if "med_men" in out else [])
    edge = min(b["s0"] for b in ext) if medial_low else max(b["s1"] for b in ext)
    a, b_in = rs(6), max(1, rs(6))
    xs = b_t["x1"] - b_t["x0"]
    out["mcl"] = {"s0": edge - a if medial_low else edge - b_in, "s1": edge + b_in if medial_low else edge + a,
                  "y0": jr - r(50), "y1": jr + r(60), "x0": int(b_t["x0"] + 0.25 * xs), "x1": int(b_t["x0"] + 0.75 * xs)}
    if b_p is not None:
        out["pf_joint"] = {"s0": b_p["s0"] - 1, "s1": b_p["s1"] + 1, "y0": b_p["y0"] - r(10), "y1": b_p["y1"] + r(10),
                           "x0": b_p["x0"] - r(5), "x1": b_p["x1"] + r(25)}
        out["suprapatellar"] = {"s0": int(s_lo + (s_hi - s_lo) * 0.2), "s1": int(s_lo + (s_hi - s_lo) * 0.8),
                                "y0": b_p["y0"] - r(55), "y1": b_p["y0"] + r(10), "x0": b_p["x0"] - r(10), "x1": b_p["x1"] + r(20)}
        out["hoffa"] = {"s0": lo_s, "s1": hi_s, "y0": jr - r(25), "y1": jr + r(5), "x0": b_t["x0"] - r(20), "x1": b_t["x0"] + r(15)}
    xpost = max(b_f["x1"], b_t["x1"])
    out["popliteal"] = {"s0": s_lo, "s1": s_hi, "y0": jr - r(40), "y1": jr + r(30), "x0": xpost - r(15), "x1": xpost + r(45)}
    return {k: clip_box(v, n) for k, v in out.items()}, info


def main():
    side = pd.read_csv(SIDES, index_col=0)
    (OUT / "rows_v2").mkdir(exist_ok=True)
    sides = []
    for p in sorted((OUT / "rows").glob("*.json")):
        j = json.load(open(p))
        meta = j["meta"]
        uid = meta["SeriesInstanceUID"]
        flip = bool(side.flip_final.get(uid, False))
        lab = np.load(OUT / "masks" / f"{uid}.npz")["lab"]
        boxes = [{"SeriesInstanceUID": uid, "structure": CLASSES[k], **b} for k in (1, 2, 7, 8, 9) if (b := box(lab == k)) is not None]
        d, info = derive(lab, meta["slab_spacing"], medial_low=not flip)
        boxes += [{"SeriesInstanceUID": uid, "structure": k, **b} for k, b in d.items()]
        meta = {k: v for k, v in meta.items() if k not in ("joint_row", "meniscus_order")} | {"medial_low": int(not flip), **info}
        json.dump({"meta": meta, "boxes": boxes}, open(OUT / "rows_v2" / p.name, "w"), default=float)
        sides.append({"SeriesInstanceUID": uid, "StudyInstanceUID": meta["StudyInstanceUID"], "series_type": meta["series_type"],
                      "meniscus_order": j["meta"].get("meniscus_order"), "flip_final": flip, "p_flip": side.p_flip.get(uid),
                      "men_found": ",".join(info.get("men_found", []))})
    s = pd.DataFrame(sides)
    s.to_csv(OUT / "sides.csv", index=False)
    print(len(s), "series |", pd.crosstab(s.flip_final, s.meniscus_order.fillna(-1)).to_dict(), "| menisci found:", s.men_found.value_counts().to_dict())


if __name__ == "__main__":
    main()
