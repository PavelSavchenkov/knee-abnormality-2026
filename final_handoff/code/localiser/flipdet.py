"""Self-supervised mirror detector for test-time laterality. Our cache series are canonical (every knee shown as a left knee,
sides from series_side_final). The test-time side rule mis-sides ~2 % of studies (the volume then comes out mirrored).
Training: canonical series = 0, the same series mirrored = 1 (sagittal: slice order reversed; coronal / axial: horizontal
flip). Input = 3 slices of the canonical volume: sagittal at 25 / 50 / 75 % of the stack (so a mirror is exactly a channel
reversal), coronal / axial the 3 central slices (mirror = horizontal flip). Learns the anatomy that tells medial from
lateral (fibula, LCL / MCL, condyle shapes). Fold `--holdout` is held out for evaluation.
    python flipdet.py [--holdout 1] [--epochs 3]   -> runs/flipdet/flipdet.pt"""
import argparse
import json
import sys
import time
from pathlib import Path

import blosc2
import numpy as np
import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from data import EXP, drop_cache, load_folds, load_meta, worker_init  # noqa: E402

SIZE = 224


def det_slices(n, plane):
    """Slice indices read for the detector input of a stack of n slices. Sagittal: [k, lo, hi, n-1-k] with k ~ 25 % and the
    two central slices (lo, hi) averaged into the middle channel, so reversing the stack is exactly a channel reversal."""
    if plane.startswith("Sag"):
        k = int(round(0.25 * (n - 1)))
        return [k, (n - 1) // 2, n // 2, n - 1 - k]
    c = (n - 1) // 2
    return [max(c - 1, 0), c, min(c + 1, n - 1)]


def combine(sl, plane):
    """Slices read at det_slices -> (3, H, W) uint8."""
    if plane.startswith("Sag"):
        mid = ((sl[1].astype(np.uint16) + sl[2] + 1) // 2).astype(np.uint8)
        return np.stack([sl[0], mid, sl[3]])
    return np.stack(sl)


def det_input(vol, plane):
    """(3, H, W) uint8 detector input from a canonical volume (n, H, W)."""
    return combine([vol[k] for k in det_slices(len(vol), plane)], plane)


def det_mirror(img, plane):
    """Mirror a detector input (torch or numpy, (..., 3, H, W)) the way the whole series would be mirrored."""
    return img.flip(-3) if plane.startswith("Sag") else img.flip(-1)


class FlipNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=pretrained, num_classes=1)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):  # x: (N, 3, H, W) uint8 / float [0, 255] -> logit of "mirrored"
        x = F.interpolate(x.float() / 255, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
        return self.enc((x - self.mean) / self.std)[:, 0]


class Series(torch.utils.data.Dataset):
    def __init__(self, uids, meta, train):
        self.uids, self.meta, self.train = list(uids), meta, train
        self.dir = EXP / "cache/s320"

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        u = self.uids[i]
        r = self.meta.loc[u]
        path = str(self.dir / f"{u}.b2nd")
        arr = blosc2.open(path)
        img = torch.from_numpy(combine([arr[k] for k in det_slices(int(r.n), r.plane)], r.plane))
        drop_cache(path)
        return {"x": img, "sag": torch.tensor(r.plane.startswith("Sag"))}


def mirror_batch(x, sag, m):
    """Mirror the rows of x (N, 3, H, W) where m: sagittal rows reverse the channels, others flip horizontally."""
    x = x.clone()
    x[m & sag] = x[m & sag].flip(1)
    x[m & ~sag] = x[m & ~sag].flip(-1)
    return x


def main(a):
    torch.manual_seed(0)
    meta = load_meta("s320")
    meta = meta[~meta.other_knee & meta.plane.isin(["Sagittal", "Coronal", "Axial"])]
    folds = load_folds()
    fold_of = folds.fold.to_dict()
    meta = meta[meta.StudyInstanceUID.isin(fold_of)]
    tr = meta.index[meta.StudyInstanceUID.map(fold_of) != a.holdout]  # holdout -1: train on every fold (deliverable model)
    ev = meta.index[meta.StudyInstanceUID.map(fold_of) == (a.holdout if a.holdout >= 0 else 1)]  # in-sample if holdout -1
    kw = dict(num_workers=12, worker_init_fn=worker_init, pin_memory=True)
    dl_tr = torch.utils.data.DataLoader(Series(tr, meta, True), batch_size=64, shuffle=True, drop_last=True, **kw)
    dl_ev = torch.utils.data.DataLoader(Series(ev, meta, False), batch_size=64, **kw)
    model = FlipNet().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    print(f"train {len(tr)} series, eval {len(ev)} (fold {a.holdout}), {total} steps", flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in dl_tr:
            x, sag = b["x"].cuda(non_blocking=True), b["sag"].cuda(non_blocking=True)
            y = torch.rand(len(x), device=x.device) < 0.5
            x = mirror_batch(x, sag, y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    model.eval()
    ps, sags = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in dl_ev:
            x, sag = b["x"].cuda(), b["sag"].cuda()
            p0 = torch.sigmoid(model(x).float())
            p1 = torch.sigmoid(model(mirror_batch(x, sag, torch.ones_like(sag))).float())
            ps.append(torch.stack([p0, p1], 1).cpu())
            sags.append(sag.cpu())
    p, s = torch.cat(ps), torch.cat(sags)
    res = {"eval_series": len(p), "acc_canonical": float((p[:, 0] < 0.5).float().mean()), "acc_mirrored": float((p[:, 1] > 0.5).float().mean()),
           "wrong_flip_at_0.9": float((p[:, 0] > 0.9).float().mean()), "caught_at_0.9": float((p[:, 1] > 0.9).float().mean()),
           "acc_by_plane": {pl: float(((p[m, 0] < 0.5).float().mean() + (p[m, 1] > 0.5).float().mean()) / 2)
                            for pl, m in (("sag", s), ("cor/ax", ~s))}}
    print(json.dumps(res, indent=1), flush=True)
    out = EXP / ("runs/flipdet" if a.holdout >= 0 else "runs/flipdet_all")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "flipdet.pt")
    json.dump(dict(vars(a), **res), open(out / "metrics.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    main(ap.parse_args())
