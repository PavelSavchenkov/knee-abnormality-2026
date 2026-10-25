"""nnU-Net Knee (3d_lowres) masks for every sagittal fat-suppressed series (all studies incl. gold) + structure boxes.

  systemd-run --user --scope -p MemoryMax=20G conda run -n rsna python work/knee_seg/nnunet_knee/mask_job.py [--limit N]
  (needs dynamic_network_architectures on PYTHONPATH; resumable: series with an existing mask file are skipped; GPU <= 5 GB)

Output (out_all/):
- masks/<SeriesInstanceUID>.npz: "lab" uint8 (n, 400, 400) in the cache grid of fov160_px400_q95 (canonical sagittal layout),
  classes 0 bg, 1 patellar cart, 2 femoral cart, 3 medial tibial cart, 4 lateral tibial cart, 5 medial men, 6 lateral men,
  7 femur, 8 tibia, 9 patella (the model's own names; see meniscus_order below). Bones: largest 3D component kept; other
  classes: components < 20 voxels removed.
- boxes.csv: one row per (series, structure): s0, s1 (slice indices), y0, y1, x0, x1 (cache px), voxels, centroid (s, y, x),
  centroid_lps (x, y, z mm), box_lps_min / max (patient LPS, mm, from the 8 corners). Structures = the 9 classes + derived:
  joint (femur-tibia interface), comp_medial, comp_lateral, notch, pf_joint, suprapatellar, popliteal.
  Derived boxes use canonical geometry: slice index runs medial -> lateral in the cache's knee side (wrong for the ~2% mis-sided
  series; apply series_side_final.csv to swap medial / lateral there). meniscus_order: 1 if the model's "medial meniscus"
  centroid is on the medial side (lower slice index), else 0 (then the model's M/L names are swapped for this series).
- series.csv: per series: StudyInstanceUID, series_type, n, slab_spacing, femur / tibia / patella voxels, joint_row, ok flag
  (femur, tibia and patella all found)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.cache import SliceCache  # noqa: E402
from geom import SeriesGeom  # noqa: E402
from infer import CLASSES, gaussian, load  # noqa: E402

OUT = HERE / "out_all"
MM = 0.4


@torch.no_grad()
def predict_gpu(net, cfg, vol, spacing, dev, step=0.5):
    tgt = cfg["spacing"]
    x = torch.from_numpy(np.ascontiguousarray(vol, np.float32)).to(dev)[None, None]
    size = [max(1, int(round(s * sp / t))) for s, sp, t in zip(vol.shape, spacing, tgt)]
    x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    x = (x - x.mean()) / (x.std() + 1e-8)
    patch = cfg["patch_size"]
    pad = [max(0, p - s) for p, s in zip(patch, size)]
    x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    shp = x.shape[2:]
    starts = [sorted({min(i, s - p) for i in range(0, max(1, s - p + 1), max(1, int(p * step)))} | {s - p}) for s, p in zip(shp, patch)]
    acc = torch.zeros((len(CLASSES), *shp), device=dev)
    wsum = torch.zeros(shp, device=dev)
    g = gaussian(patch).to(dev)
    for z in starts[0]:
        for yy in starts[1]:
            for xx in starts[2]:
                sl = (slice(z, z + patch[0]), slice(yy, yy + patch[1]), slice(xx, xx + patch[2]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = net(x[(0, 0) + sl][None, None])[0]
                acc[(slice(None),) + sl] += torch.softmax(o.float(), 0) * g
                wsum[sl] += g
    prob = (acc / wsum)[:, : size[0], : size[1], : size[2]]
    prob = F.interpolate(prob[None], size=vol.shape, mode="trilinear", align_corners=False)[0]
    return prob.argmax(0).to(torch.uint8).cpu().numpy()


def clean(lab):
    out = np.zeros_like(lab)
    for k in range(1, len(CLASSES)):
        m = lab == k
        if not m.any():
            continue
        cc, n = ndimage.label(m)
        sizes = np.bincount(cc.ravel())[1:]
        keep = [int(np.argmax(sizes)) + 1] if k >= 7 else [i + 1 for i, v in enumerate(sizes) if v >= 20]
        out[np.isin(cc, keep)] = k
    return out


def box(mask):
    idx = np.argwhere(mask)
    if not len(idx):
        return None
    lo, hi = idx.min(0), idx.max(0)
    return {"s0": int(lo[0]), "s1": int(hi[0]), "y0": int(lo[1]), "y1": int(hi[1]), "x0": int(lo[2]), "x1": int(hi[2]),
            "voxels": int(len(idx)), "cs": float(idx[:, 0].mean()), "cy": float(idx[:, 1].mean()), "cx": float(idx[:, 2].mean())}


def clip_box(b, n, px=400):
    b = dict(b)
    b["s0"], b["s1"] = int(max(0, b["s0"])), int(min(n - 1, b["s1"]))
    for a in ("y", "x"):
        b[f"{a}0"], b[f"{a}1"] = int(max(0, b[f"{a}0"])), int(min(px - 1, b[f"{a}1"]))
    return b


def derived(lab):
    """Structure regions from the bones / menisci (cache grid; rows S->I, cols A->P, slices medial->lateral)."""
    n = lab.shape[0]
    fem, tib, pat = lab == 7, lab == 8, lab == 9
    out, info = {}, {}
    b_f, b_t, b_p = box(fem), box(tib), box(pat)
    if b_f is None or b_t is None:
        return out, info
    rows_f = [np.nonzero(fem[s].any(1))[0].max() for s in range(n) if fem[s].any() and tib[s].any()]
    rows_t = [np.nonzero(tib[s].any(1))[0].min() for s in range(n) if fem[s].any() and tib[s].any()]
    if not rows_f:
        return out, info
    jr = int(round((np.median(rows_f) + np.median(rows_t)) / 2))
    info["joint_row"] = jr
    r = lambda mm: int(round(mm / MM))
    s_lo, s_hi = min(b_f["s0"], b_t["s0"]), max(b_f["s1"], b_t["s1"])
    out["joint"] = {"s0": s_lo, "s1": s_hi, "y0": jr - r(15), "y1": jr + r(15), "x0": b_t["x0"], "x1": b_t["x1"]}
    men = {k: box(lab == k) for k in (5, 6)}
    if men[5] is not None and men[6] is not None:
        info["meniscus_order"] = int(men[5]["cs"] <= men[6]["cs"])
        med, lat = (men[5], men[6]) if info["meniscus_order"] else (men[6], men[5])
        for name, m in (("comp_medial", med), ("comp_lateral", lat)):
            out[name] = {"s0": m["s0"] - 1, "s1": m["s1"] + 1, "y0": jr - r(25), "y1": jr + r(25), "x0": b_t["x0"] - r(5), "x1": b_t["x1"] + r(5)}
        g0, g1 = (int(med["cs"]), int(lat["cs"]))
    else:
        mid, half = (s_lo + s_hi) / 2, (s_hi - s_lo) / 2
        g0, g1 = int(mid - half / 2), int(mid + half / 2)
    lo_s, hi_s = min(g0, g1), max(g0, g1)
    third = max(1, (hi_s - lo_s) // 3)
    xc = (b_t["x0"] + b_t["x1"]) // 2
    out["notch"] = {"s0": lo_s + third, "s1": hi_s - third, "y0": jr - r(30), "y1": jr + r(10), "x0": xc - r(25), "x1": xc + r(20)}
    if b_p is not None:
        out["pf_joint"] = {"s0": b_p["s0"] - 1, "s1": b_p["s1"] + 1, "y0": b_p["y0"] - r(10), "y1": b_p["y1"] + r(10),
                           "x0": b_p["x0"] - r(5), "x1": b_p["x1"] + r(25)}
        out["suprapatellar"] = {"s0": int(s_lo + (s_hi - s_lo) * 0.2), "s1": int(s_lo + (s_hi - s_lo) * 0.8),
                                "y0": b_p["y0"] - r(55), "y1": b_p["y0"] + r(10), "x0": b_p["x0"] - r(10), "x1": b_p["x1"] + r(20)}
    xpost = max(b_f["x1"], b_t["x1"])
    out["popliteal"] = {"s0": s_lo, "s1": s_hi, "y0": jr - r(40), "y1": jr + r(30), "x0": xpost - r(15), "x1": xpost + r(45)}
    return {k: clip_box(v, n) for k, v in out.items()}, info


def lps_rows(g, uid, name, b):
    corners = np.array([[s, y, x] for s in (b["s0"], b["s1"]) for y in (b["y0"], b["y1"]) for x in (b["x0"], b["x1"])])
    p = g.to_patient(corners[:, 0], corners[:, 1], corners[:, 2])
    c = g.to_patient(int(round(b.get("cs", (b["s0"] + b["s1"]) / 2))), b.get("cy", (b["y0"] + b["y1"]) / 2), b.get("cx", (b["x0"] + b["x1"]) / 2))
    return {"SeriesInstanceUID": uid, "structure": name, **b, "c_lps_x": c[0], "c_lps_y": c[1], "c_lps_z": c[2],
            **{f"min_lps_{a}": v for a, v in zip("xyz", p.min(0))}, **{f"max_lps_{a}": v for a, v in zip("xyz", p.max(0))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(5 / 32)
    dev = "cuda"
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "rows").mkdir(exist_ok=True)
    c = SliceCache("fov160_px400_q95")
    s = c.series[(c.series.dup_of == "") & (c.series.plane == "Sagittal") & (c.series.fs == 1)]
    first = set()
    att = ROOT / "experiments/v0/oct7_baseline/ideas_archive/ideas_loop/out/attention_fold0_pred.parquet"             # studies with saved attention go first, then gold
    if att.exists():
        first |= set(pd.read_parquet(att).index)
    from common.labels import gold_labels
    first |= set(gold_labels().index)
    s = s.assign(_p=~s.StudyInstanceUID.isin(first)).sort_values("_p", kind="stable")
    todo = [r for r in s.itertuples() if not (OUT / "masks" / f"{r.SeriesInstanceUID}.npz").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"sagittal FS series {len(s)} in {s.StudyInstanceUID.nunique()} studies; to do {len(todo)}", flush=True)
    net, cfg = load("3d_lowres")
    net = net.to(dev)
    t0 = time.time()
    for i, r in enumerate(todo):
        vol = c.read(np.arange(r.start, r.start + r.n)).astype(np.float32)
        lab = clean(predict_gpu(net, cfg, vol, (float(r.slab_spacing), MM, MM), dev))
        g = SeriesGeom(r.SeriesInstanceUID)
        rows = [lps_rows(g, r.SeriesInstanceUID, CLASSES[k], b) for k in range(1, len(CLASSES)) if (b := box(lab == k)) is not None]
        dboxes, info = derived(lab)
        rows += [lps_rows(g, r.SeriesInstanceUID, k, b) for k, b in dboxes.items()]
        vox = {CLASSES[k]: int((lab == k).sum()) for k in (7, 8, 9)}
        meta = {"SeriesInstanceUID": r.SeriesInstanceUID, "StudyInstanceUID": r.StudyInstanceUID, "series_type": r.series_type, "n": int(r.n),
                "slab_spacing": float(r.slab_spacing), **vox, **info, "ok": int(all(v > 0 for v in vox.values()))}
        json.dump({"meta": meta, "boxes": rows}, open(OUT / "rows" / f"{r.SeriesInstanceUID}.json", "w"), default=float)
        np.savez_compressed(OUT / "masks" / f"{r.SeriesInstanceUID}.npz", lab=lab)
        if i % 100 == 0:
            print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    js = [json.load(open(p)) for p in sorted((OUT / "rows").glob("*.json"))]
    pd.DataFrame([j["meta"] for j in js]).to_csv(OUT / "series.csv", index=False)
    pd.DataFrame([b for j in js for b in j["boxes"]]).to_csv(OUT / "boxes.csv", index=False)
    print("done", len(js), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
