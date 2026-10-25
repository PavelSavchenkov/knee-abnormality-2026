"""CPU localiser distilled from the nnU-Net boxes (scripts/loc_data.py): per slice window, for each structure in STRUCTS,
presence (BCE) and the box (y0, y1, x0, x1) / cache px (smooth L1, present slices only).

Model: timm backbone (ImageNet init) on the 3-slice window, pooled features + [plane one-hot, pos_mm / 60, idx / n] -> MLP ->
S x 5 outputs. Train: folds != VAL_FOLD (gold studies left out); validation: VAL_FOLD.
Report (validation): presence AUC; on present slices, box-centre error (mm) and IoU; per series (boxes averaged over slices with
p > 0.5, weighted by p) vs the nnU-Net series box (median over its slices) and vs a fixed box (median per plane x structure over
training series).
CLI: python scripts/loc_train.py [--arch mobilenetv3_large_100] [--epochs 6] [--threads 14] [--name loc_mnv3]
Output: out/<name>.pt, out/<name>_val.parquet, out/<name>_report.txt"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
from loc_data import PX, STRUCTS, STRUCTS_V2, STRUCTS_V3  # noqa: E402

PLANES = ["Sagittal", "Coronal", "Axial"]
MM = 160.0                                     # cache field of view: box units (fraction of the image) x MM = mm
VAL_FOLD = 0


class Loc(nn.Module):
    def __init__(self, arch, pretrained=True, structs=STRUCTS):
        super().__init__()
        self.n_s = len(structs)
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]   # MobileNetV3: head_hidden_size, not num_features
        self.head = nn.Sequential(nn.Linear(d + 5, 256), nn.ReLU(), nn.Linear(256, self.n_s * 5))
        c = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(c["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(c["std"]).view(1, 3, 1, 1) * 255)

    def forward(self, x, aux):
        x = ((x.float() - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        o = self.head(torch.cat([self.backbone(x), aux], 1)).view(-1, self.n_s, 5)
        return o[..., 0], torch.sigmoid(o[..., 1:])            # presence logit, box in [0, 1]


def aux_features(meta):
    plane = np.stack([(meta.plane == p).to_numpy() for p in PLANES], 1).astype(np.float32)
    return np.concatenate([plane, (meta.pos_mm.to_numpy(np.float32) / 60)[:, None],
                           (meta.idx / np.maximum(meta.n_slices - 1, 1)).to_numpy(np.float32)[:, None]], 1)


def predict(model, x, aux, bs=512):
    model.eval()
    ps, bs_ = [], []
    with torch.no_grad():
        for a in range(0, len(x), bs):
            p, b = model(torch.from_numpy(np.asarray(x[a:a + bs])), torch.from_numpy(aux[a:a + bs]))
            ps.append(torch.sigmoid(p).numpy())
            bs_.append(b.numpy())
    return np.concatenate(ps), np.concatenate(bs_)


def centre_err_mm(a, b):
    """a, b (..., 4) boxes y0 y1 x0 x1 in image fractions -> centre distance in mm."""
    return np.hypot((a[..., 0] + a[..., 1] - b[..., 0] - b[..., 1]) / 2, (a[..., 2] + a[..., 3] - b[..., 2] - b[..., 3]) / 2) * MM


def iou(a, b):
    h = np.clip(np.minimum(a[..., 1], b[..., 1]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    w = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 2], b[..., 2]), 0, None)
    inter = h * w
    area = lambda z: (z[..., 1] - z[..., 0]) * (z[..., 3] - z[..., 2])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def q(v, pct):
    return float(np.percentile(v, pct)) if len(v) else np.nan


def report(meta, pres, box, p, pb, train_meta, train_box, train_pres, structs=STRUCTS, known=None):
    lines = []
    for j, s in enumerate(structs):
        for plane in PLANES:
            m = (meta.plane == plane).to_numpy() & (known[:, j] if known is not None else True)
            if pres[m, j].sum() < 20:
                continue
            auc = roc_auc_score(pres[m, j], p[m, j]) if 0 < pres[m, j].mean() < 1 else np.nan
            k = m & pres[:, j]
            e = centre_err_mm(pb[k, j], box[k, j])
            io = iou(pb[k, j], box[k, j])
            # series level
            d = meta[m].assign(p=p[m, j], pres=pres[m, j], **{f"t{i}": box[m, j, i] for i in range(4)},
                               **{f"q{i}": pb[m, j, i] for i in range(4)})
            truth = d[d.pres].groupby("SeriesInstanceUID")[[f"t{i}" for i in range(4)]].median()
            dd = d[d.p > 0.5]
            w = dd.p.to_numpy()[:, None]
            pred = (dd[[f"q{i}" for i in range(4)]] * w).groupby(dd.SeriesInstanceUID).sum().div(
                dd.groupby("SeriesInstanceUID").p.sum(), axis=0)
            common = truth.index.intersection(pred.index)
            tm = train_meta.plane == plane
            fixed = np.nanmedian(pd.DataFrame(train_box[tm.to_numpy() & train_pres[:, j], j]).groupby(
                train_meta[tm & train_pres[:, j]].SeriesInstanceUID.to_numpy()).median().to_numpy(), 0)
            es = centre_err_mm(pred.loc[common].to_numpy(), truth.loc[common].to_numpy())
            ef = centre_err_mm(np.broadcast_to(fixed, (len(truth), 4)), truth.to_numpy())
            lines.append(f"{s:13s} {plane:8s} pres AUC {auc:.3f} | slice centre err mm median {q(e, 50):.1f} p90 "
                         f"{q(e, 90):.1f}, IoU median {q(io, 50):.2f} | series: loc median {q(es, 50):.1f} p90 "
                         f"{q(es, 90):.1f} p95 {q(es, 95):.1f} (n {len(common)}, missed {len(truth) - len(common)}) "
                         f"vs fixed median {q(ef, 50):.1f} p90 {q(ef, 90):.1f} p95 {q(ef, 95):.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mobilenetv3_large_100")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--name", default="loc_mnv3")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    d = EXP / "data" / f"loc_{PX}"
    x = np.load(d / "x.npy", mmap_mode="r")
    meta = pd.read_parquet(d / "meta.parquet")
    if a.version in ("v2", "v3"):
        structs = STRUCTS_V3 if a.version == "v3" else STRUCTS_V2
        v = a.version
        pres, box, known = np.load(d / f"pres_{v}.npy"), np.load(d / f"box_{v}.npy"), np.load(d / f"known_{v}.npy")
    else:
        structs = STRUCTS
        pres, box = np.load(d / "pres.npy"), np.load(d / "box.npy")
        known = np.ones_like(pres)
    aux = aux_features(meta)
    tr = np.flatnonzero((meta.fold != VAL_FOLD).to_numpy() & ~meta.gold.to_numpy())
    va = np.flatnonzero((meta.fold == VAL_FOLD).to_numpy() & ~meta.gold.to_numpy())
    if a.max_train:
        tr = np.random.default_rng(0).choice(tr, min(a.max_train, len(tr)), replace=False)
    print(f"train {len(tr)}, val {len(va)}", flush=True)
    xtr = np.ascontiguousarray(x[np.sort(tr)])                 # RAM copy of the training windows (~5 GB at 128 px)
    tr = np.sort(tr)
    model = Loc(a.arch, structs=structs).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    pres_t, box_t = torch.from_numpy(pres), torch.from_numpy(np.nan_to_num(box))
    aux_t = torch.from_numpy(aux)
    known_t = torch.from_numpy(known).float()
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(tr))
        tl, n = 0.0, 0
        for k in range(len(tr) // a.bs):
            ii = perm[k * a.bs:(k + 1) * a.bs]
            gi = tr[ii]
            xb = torch.from_numpy(xtr[ii]).float()
            g = 1 + (torch.rand(len(ii), 1, 1, 1) - 0.5) * 0.4                        # contrast / brightness jitter
            xb = (xb * g + (torch.rand(len(ii), 1, 1, 1) - 0.5) * 30).clamp(0, 255)
            p, b = model(xb, aux_t[gi])
            y, bt, kn = pres_t[gi].float(), box_t[gi], known_t[gi]
            lp = (F.binary_cross_entropy_with_logits(p, y, reduction="none") * kn).sum() / kn.sum().clamp(min=1)
            lb = (F.smooth_l1_loss(b, bt, reduction="none", beta=0.02).sum(-1) * y).sum() / y.sum().clamp(min=1)
            loss = lp + 5 * lb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
            n += 1
            if k % 200 == 0:
                print(f"ep {ep} step {k}/{len(tr) // a.bs} loss {tl / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), EXP / "out" / f"{a.name}.pt")
        p, pb = predict(model, x[va], aux[va])
        mv = meta.iloc[va].reset_index(drop=True)
        rep = report(mv, pres[va], box[va], p, pb, meta.iloc[tr].reset_index(drop=True), box[tr], pres[tr], structs, known[va])
        print(f"== epoch {ep} loss {tl / n:.4f} ({time.time() - t0:.0f}s)\n{rep}", flush=True)
    open(EXP / "out" / f"{a.name}_report.txt", "w").write(rep + "\n")
    out = mv.copy()
    for j, s in enumerate(structs):
        out[f"p_{s}"] = p[:, j]
        for i, c in enumerate(("y0", "y1", "x0", "x1")):
            out[f"{s}_{c}"] = pb[:, j, i]
    out.to_parquet(EXP / "out" / f"{a.name}_val.parquet")


if __name__ == "__main__":
    main()
