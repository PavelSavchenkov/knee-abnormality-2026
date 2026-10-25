"""Knee-side check by anatomy: is a series in the slice cache oriented with the wrong knee side?

A wrong side mirrors coronal and axial slices left-right (medial and lateral swap in the image) and reverses the slice order of sagittal
series (the images themselves are unchanged). So two small classifiers, trained on series whose orientation is trusted (confident geometry
|image-centre x| >= 40 mm agrees with the side used, and with the series Laterality tag if there is one):
- mirror: coronal/axial slice from the middle 60% of the series -> was it flipped left-right? (each slice shown as is or flipped at random)
- lateral: sagittal slice at |pos| >= 12 mm -> is it on the lateral side (pos > 0)?
Per series: p_flip = mean P(flipped) over its slices (coronal/axial), or the mean probability that each slice is on the other side than
its position says (sagittal). Held-out studies (20%) measure accuracy on the series as stored and on artificially flipped copies.

systemd-run --user --scope --unit=side-check -p MemoryMax=24G conda run -n rsna python data/build/side_check/side_check.py
Outputs (out/): series_side.csv (one row per cache series), report.txt, mirror.pt, lateral.pt.
"""
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common.cache import SliceCache
from common.meta import frames, image_centre_x, series

OUT = Path(__file__).parent / "out"
PX, CROP, CONF_MM, SAG_MM = 176, 160, 40, 12
EPOCHS, PER_SERIES, BS = 3, 4, 128
DEV = "cuda"


def table(cache):
    cs = cache.series.set_index("SeriesInstanceUID")
    s = series()
    x = image_centre_x(frames(), s).reindex(cs.index)
    tag = s.Laterality.reindex(cs.index).where(lambda t: t.isin(["L", "R"]))
    cs = cs.assign(x=x, geo=np.where(x > 0, "L", "R"), tag=tag, desc=s.SeriesDescription.reindex(cs.index))
    cs["trusted"] = (x.abs() >= CONF_MM) & (cs.side == cs.geo) & (tag.isna() | (tag == cs.side)) & (cs.dup_of.fillna("") == "")
    cs["val"] = cs.StudyInstanceUID.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 5 == 0)
    return cs


def slices_of(cache, cs, kind):
    """Candidate slices per series: (SeriesInstanceUID, row, pos_mm)."""
    sl = cache.slices.merge(cs[["plane"]], left_on="SeriesInstanceUID", right_index=True)
    if kind == "mirror":
        sl = sl[sl.plane.isin(["Coronal", "Axial"])]
        n = sl.groupby("SeriesInstanceUID").idx.transform("max") + 1
        return sl[(sl.idx >= 0.2 * n) & (sl.idx <= 0.8 * n)]
    sl = sl[sl.plane == "Sagittal"]
    return sl[sl.pos_mm.abs() >= SAG_MM]


class Slices(Dataset):
    """rows -> (image, label). mirror: label = flipped (random in training, given in eval); lateral: label = pos > 0."""

    def __init__(self, cache, rows, kind, train, flip=None, pos=None):
        self.cache, self.rows, self.kind, self.train, self.flip, self.pos = cache, np.asarray(rows), kind, train, flip, pos

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a = cv2.resize(self.cache.read([self.rows[i]])[0], (PX, PX), interpolation=cv2.INTER_AREA)
        if self.train:
            y0, x0 = np.random.randint(0, PX - CROP + 1, 2)
            a = a[y0:y0 + CROP, x0:x0 + CROP]
            a = np.clip(a.astype(np.float32) * np.random.uniform(0.8, 1.2) + np.random.uniform(-15, 15), 0, 255)
        else:
            m = (PX - CROP) // 2
            a = a[m:m + CROP, m:m + CROP].astype(np.float32)
        if self.kind == "mirror":
            f = np.random.rand() < 0.5 if self.train else bool(self.flip[i])
            a, y = (a[:, ::-1] if f else a), float(f)
        else:
            y = float(self.pos[i] > 0)
        return torch.from_numpy(np.ascontiguousarray(a) / 255.0)[None].float(), torch.tensor(y)


def model():
    return timm.create_model("resnet18.a1_in1k", pretrained=True, in_chans=1, num_classes=1).to(DEV).to(memory_format=torch.channels_last)


def predict(net, ds):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in DataLoader(ds, BS * 2, num_workers=8):
            out.append(torch.sigmoid(net(x.to(DEV).to(memory_format=torch.channels_last)).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


def train(cache, sl, kind, log):
    net = model()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        pick = sl.sample(frac=1, random_state=int(rng.integers(1e9))).groupby("SeriesInstanceUID").head(PER_SERIES)
        ds = Slices(cache, pick.row.to_numpy(), kind, True, pos=pick.pos_mm.to_numpy())
        net.train()
        t, losses = time.time(), []
        for x, y in DataLoader(ds, BS, shuffle=True, num_workers=8, drop_last=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(net(x.to(DEV).to(memory_format=torch.channels_last)).float().squeeze(1), y.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log(f"{kind} epoch {epoch}: {len(ds)} slices, loss {np.mean(losses):.4f}, {time.time() - t:.0f}s")
    torch.save(net.state_dict(), OUT / f"{kind}.pt")
    return net


def series_scores(net, cache, sl, kind, flipped):
    """p_flip per series for the series as stored (flipped=False) or with every slice flipped / positions reversed (flipped=True)."""
    sl = sl.sort_values(["SeriesInstanceUID", "idx"])
    k = sl.groupby("SeriesInstanceUID").cumcount()
    step = np.ceil(sl.groupby("SeriesInstanceUID").idx.transform("size") / 12)
    sl = sl[k % step == 0]                                               # at most 12 evenly spaced slices per series
    if kind == "mirror":
        p = predict(net, Slices(cache, sl.row.to_numpy(), kind, False, flip=np.full(len(sl), flipped)))
    else:
        pos = -sl.pos_mm.to_numpy() if flipped else sl.pos_mm.to_numpy()
        p_lat = predict(net, Slices(cache, sl.row.to_numpy(), kind, False, pos=pos))
        p = np.where(pos > 0, 1 - p_lat, p_lat)                      # probability that the slice is on the other side
    return pd.Series(p, index=sl.SeriesInstanceUID.to_numpy()).groupby(level=0).mean()


def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True)
        lines.append(m)

    cache = SliceCache("fov160_px400_q95")
    cs = table(cache)
    log(f"series {len(cs)}, trusted {int(cs.trusted.sum())} (val studies 20%)")
    p_flip = {}
    for kind in ["mirror", "lateral"]:
        sl = slices_of(cache, cs, kind)
        tr = sl[sl.SeriesInstanceUID.map(cs.trusted & ~cs.val)]
        va = sl[sl.SeriesInstanceUID.map(cs.trusted & cs.val)]
        net = train(cache, tr, kind, log)
        ok, bad = series_scores(net, cache, va, kind, False), series_scores(net, cache, va, kind, True)
        log(f"{kind} held-out series: as stored p_flip >= 0.5 in {(ok >= 0.5).mean():.4f} of {len(ok)}; "
            f"artificially flipped p_flip < 0.5 in {(bad < 0.5).mean():.4f}; median p_flip {ok.median():.3f} / {bad.median():.3f}")
        p_flip[kind] = series_scores(net, cache, sl, kind, False)
    cs["p_flip"] = pd.concat(p_flip.values())
    cs["flip"] = cs.p_flip >= 0.8
    cs["side_checked"] = np.where(cs.flip, cs.side.map({"L": "R", "R": "L"}), cs.side)
    cols = ["StudyInstanceUID", "plane", "series_type", "side", "tag", "geo", "x", "desc", "other_knee", "dup_of", "trusted", "p_flip",
            "flip", "side_checked"]
    cs[cols].to_csv(OUT / "series_side.csv")
    scored = cs.p_flip.notna()
    log(f"scored series {int(scored.sum())} of {len(cs)} (unscored: sagittal series with no slice at |pos| >= {SAG_MM} mm, or no slices)")
    log(f"flip (p_flip >= 0.8): {int(cs.flip.sum())} series in {cs[cs.flip].StudyInstanceUID.nunique()} studies; "
        f"0.2 < p_flip < 0.8 (unsure): {int(((cs.p_flip > 0.2) & (cs.p_flip < 0.8)).sum())}")
    log("flip by group: " + str(cs[scored].groupby(["trusted", "other_knee"]).flip.agg(["sum", "size"]).to_dict("index")))
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
