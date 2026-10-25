"""Synovitis crop expert: one shared CNN over synovitis-region crops (syn_data.py), per-output attention pooling over the windows of
the bags that output may read (default: all bags), adapted from experiments/v1/oct10_oa/main/oa_train.py.

h_k = LayerNorm(backbone(crop_k) + type_emb + plane_emb + bag_emb + pos_enc); per output n: a_kn = softmax over its allowed windows of
w_n . tanh(V h_k); logit_n = c_n . sum_k a_kn h_k + b_n.
Outputs (OUTS): img_syn = Codex read synovitis probability (data/img_syn_prob.csv, the m_prob05 target; masked where unread),
img_eff = Codex read effusion probability, syn / eff / bakers = text v1prob Synovitis / Effusion / Baker's, a_* = Codex read
attributes (data/img_attrs.csv: effusion grade, effusion-synovitis grade, Hoffa signal, thickening, debris; soft BCE targets in 0-1).
Loss = masked BCE weighted by w_img (img_syn), w_ieff (img_eff), w_txt (text outputs), w_attr (a_*), normalised by the summed
weights of the batch.
At inference the img_syn output alone is the synovitis score (column img_Synovitis, as the main pipeline's image head).

conda run -n rsna --no-capture-output python experiments/v1/oct11_synovitis_crops/main/syn_train.py --name NAME --folds 1 --set key=value ...
Output: runs/NAME/foldF/{config.json, log.txt, oof.parquet, gold.parquet (FINDINGS + img_Synovitis / img_Effusion, + epoch),
metrics.json, model_ep{E}.pt}
oct14_all_data changes: fold -1 = all-data run (every non-gold study trains, 'val' = gold-58); spe_ref = N runs the schedule
(warmup, cosine, epochs) in units of N steps (the fold runs' steps per epoch: equal step count and LR curve); save_opt writes
opt_ep{E}.pt (optimizer, step) next to every saved model_ep{E}.pt.
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
EXP = HERE.parent
RUNS = EXP / "runs"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.meta import SERIES_TYPES  # noqa: E402
from common.paths import FINDINGS  # noqa: E402
from data import gpu_prep  # noqa: E402
from labels import make_targets  # noqa: E402
from model import sincos  # noqa: E402
from syn_data import SynDataset, bag_names as make_bag_names, build_bags, finalize  # noqa: E402

ATTRS = ["a_effgr", "a_effsyn", "a_hoffa", "a_thick", "a_debris"]       # Codex read attributes (scripts/make_attrs.py), aux outputs
OUT_COL = {"img_syn": "img_Synovitis", "img_eff": "img_Effusion", "syn": "Synovitis", "eff": "Effusion", "bakers": "Baker's"} | {a: a for a in ATTRS}
READS = pd.read_csv(EXP / "data/synovitis_labels_codex.csv").set_index("StudyInstanceUID")
ATTR_T = pd.read_csv(EXP / "data/img_attrs.csv").set_index("StudyInstanceUID")
DEFAULTS = dict(backbone="efficientnet_b0.ra_in1k", px=224, k=8, k_eval=32, bs=8, epochs=20, lr=1e-3, lr_head=1e-3, wd=1e-2,
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, bags="pouch,hoffa,post", outs="img_syn,img_eff,syn,eff,bakers",
                w_img=1.0, w_ieff=0.5, w_txt=0.5, w_attr=0.5, bakers_post=True, img_csv="data/img_syn_prob.csv", labels="v1prob", nm=0.15,
                crop_scale=1.0, trim=0.1, fs_only=False, plane_bags=False, clamp=False, planes="", bagsum=False, init="", aug_rot=15, aug_scale=0.15, aug_shift=0.05, aug_erase=0.3, precision="bf16",
                eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0, save_opt=True, device="cuda", max_eval=0,
                mono=False, compile=False)


def encoder(backbone, pretrained):
    """timm encoder (num_classes=0); 'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (ImageNet mean / std; pair it
    with mono=True)."""
    if backbone != "radimagenet_resnet50":
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    enc = timm.create_model("resnet50", pretrained=False, num_classes=0)
    if pretrained:
        sd = torch.load(HERE.parents[3] / "weights/radimagenet/ResNet50.pt", map_location="cpu")
        names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
        sd = {names[k.split(".")[1]] + k[len("backbone.") + len(k.split(".")[1]):]: v for k, v in sd.items()}
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert not unexpected and not [m for m in missing if not m.startswith("fc")], (missing, unexpected)
    enc.pretrained_cfg = dict(enc.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return enc


class SynNet(nn.Module):
    def __init__(self, backbone, out_bags, n_bags, hidden=256, drop=0.2, pretrained=True, bagsum=False):
        """out_bags: (n_out, n_bags) bool, which bags each output may read. bagsum: attention pooling inside each bag separately and
        one readout per (output, bag), summed (a bag's evidence is not diluted by the windows of the other bags)."""
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        self.backbone = encoder(backbone, pretrained)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.type_emb = nn.Embedding(len(SERIES_TYPES), d)
        self.plane_emb = nn.Embedding(3, d)
        self.bag_emb = nn.Embedding(n_bags, d)
        self.pos_proj = nn.Linear(16, d)
        for m in (self.type_emb, self.plane_emb, self.bag_emb, self.pos_proj):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.pos_proj.bias)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        n_out = out_bags.shape[0]
        self.att_v = nn.Linear(d, hidden)
        self.att_w = nn.Linear(hidden, n_out)
        self.cls_w = nn.Parameter(torch.zeros(n_out, n_bags, d) if bagsum else torch.zeros(n_out, d))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("out_bags", torch.as_tensor(out_bags).T.contiguous(), persistent=False)   # (n_bags, n_out)

    def norm_stats(self):
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, mask, typ, plane, pos, bag):
        b, k = mask.shape
        f = self.backbone(x).float().view(b, k, -1)
        h = self.norm(f + self.type_emb(typ) + self.plane_emb(plane) + self.bag_emb(bag.clamp(min=0)) + self.pos_proj(sincos(pos, scale=240.0)))
        hd = self.drop(h)
        allowed = self.out_bags[bag.clamp(min=0)] & mask[..., None]                 # (b, k, n_out)
        s = self.att_w(torch.tanh(self.att_v(hd))).masked_fill(~allowed, float("-inf"))
        if self.bagsum:
            out = self.cls_b.expand(b, -1)
            for j in range(self.n_bags):
                a = torch.softmax(s.masked_fill((bag != j)[..., None], float("-inf")), 1).nan_to_num(0.0)
                out = out + (torch.einsum("bkn,bkd->bnd", a, hd) * self.cls_w[:, j]).sum(-1)
            return out
        a = torch.softmax(s, 1).nan_to_num(0.0)
        z = torch.einsum("bkn,bkd->bnd", a, hd)
        return (z * self.cls_w).sum(-1) + self.cls_b


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled indices offset by epoch * n, so persistent workers can derive the epoch."""
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch = n, seed, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter((torch.randperm(self.n, generator=g) + self.epoch * self.n).tolist())

    def __len__(self):
        return self.n


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def make_syn_targets(studies, cfg, outs):
    """(n, n_out) targets and loss weights; unread img_* cells get weight 0."""
    yt, _ = make_targets(studies, cfg)
    img = pd.read_csv(EXP / cfg["img_csv"]).set_index("StudyInstanceUID").Synovitis.reindex(studies).to_numpy(np.float32)
    eff = READS.Effusion.reindex(studies).to_numpy(np.float32)
    y, w = np.zeros((len(studies), len(outs)), np.float32), np.zeros((len(studies), len(outs)), np.float32)
    for j, o in enumerate(outs):
        if o in ("img_syn", "img_eff"):
            v = img if o == "img_syn" else eff
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * (cfg["w_img"] if o == "img_syn" else cfg["w_ieff"])
        elif o in ATTRS:
            v = ATTR_T[o].reindex(studies).to_numpy(np.float32)
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_attr"]
        else:
            y[:, j], w[:, j] = yt[:, FINDINGS.index(OUT_COL[o])], cfg["w_txt"]
    return y, w


@torch.no_grad()
def predict(model, loader, cfg, dev, amp):
    model.eval()
    out = []
    for b in loader:
        b = to_dev(b, dev)
        x = gpu_prep(b["x"], cfg["px"], "bilinear", None, *model.norm_stats(), mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            out.append(torch.sigmoid(model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"]).float()).cpu())
    return torch.cat(out).numpy()


def as_frame(p, idx, outs):
    d = pd.DataFrame(np.nan, index=pd.Index(idx, name="StudyInstanceUID"), columns=FINDINGS + ["img_Synovitis", "img_Effusion"])
    for j, o in enumerate(outs):          # attribute outputs become extra columns a_*
        d[OUT_COL[o]] = p[:, j]
    return d


def syn_metrics(pv, pg):
    """Held-out reads AUC (p >= 0.5), clear-cut reads (>= 0.85 vs <= 0.1), gold-58 synovitis of the img_Synovitis column."""
    r = READS.Synovitis.reindex(pv.index)
    has = r.notna()
    y, s = (r[has] >= 0.5).astype(int), pv.img_Synovitis[has]
    conf = (r[has] >= 0.85) | (r[has] <= 0.1)
    g = gold_labels().reindex(pg.index)
    auc = lambda t, x: roc_auc_score(t, x) if 0 < t.sum() < len(t) else float("nan")  # noqa: E731  (all-data runs: val = gold-58)
    return {"r_auc": auc(y, s), "r_conf": auc(y[conf], s[conf]), "g_syn": auc(g.Synovitis, pg.img_Synovitis)}


def load_bags(cfg):
    """finalize(build_bags(...)), cached per bag config in runs/_bags/."""
    key = {k: cfg[k] for k in ("bags", "crop_scale", "trim", "fs_only")} | {k: cfg[k] for k in ("plane_bags", "clamp", "planes") if cfg[k]}
    f = RUNS / "_bags" / (hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10] + ".pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    bags = finalize(build_bags(cfg["bags"].split(","), cfg["crop_scale"], cfg["trim"], cfg["fs_only"], cfg["plane_bags"], cfg["clamp"], cfg["planes"]))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(bags, open(f, "wb"))
    return bags


def run_fold(cfg, fold, out_dir, bags):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev, amp = cfg["device"], {"bf16": torch.bfloat16, "fp32": None}[cfg["precision"]]
    torch.backends.cudnn.benchmark = True
    bag_names, outs = make_bag_names(cfg["bags"].split(","), cfg["plane_bags"], cfg["planes"]), cfg["outs"].split(",")
    out_bags = np.ones((len(outs), len(bag_names)), bool)
    if cfg["bakers_post"] and "bakers" in outs and "post" in cfg["bags"].split(","):
        out_bags[outs.index("bakers")] = [b.split(":")[0] == "post" for b in bag_names]
    folds = get_folds(5, 0)
    folds = folds[folds.index.isin(bags.keys())]
    tr = folds.index[folds.fold != fold].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in bags]
    if fold < 0:                                          # all-data run (fold -1): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    if cfg["max_eval"]:                                   # smoke tests
        va, gold = va[:cfg["max_eval"]], gold[:cfg["max_eval"]]
    y, w = make_syn_targets(tr, cfg, outs)
    log = open(out_dir / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    ds_tr = SynDataset(tr, bags, bag_names, y, w, cfg["k"], True, cfg["px"], cfg["seed"], len(outs))
    sampler = EpochSampler(len(tr), cfg["seed"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(SynDataset(ids, bags, bag_names, None, None, cfg["k_eval"], False, cfg["px"], n_out=len(outs)),
                                batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    model = SynNet(cfg["backbone"], out_bags, len(bag_names), cfg["hidden"], cfg["drop"], bagsum=cfg["bagsum"]).to(dev).to(memory_format=torch.channels_last)
    if cfg["init"]:                      # main-pipeline encoder trained on the same fold split (no leakage), e.g. refs/m_prob05/fold{fold}/model_ep19.pt
        sd = torch.load(EXP / cfg["init"].format(fold=fold), map_location="cpu")
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    if cfg["compile"]:                                    # backbone only; state_dict keys unchanged (the module is wrapped in place)
        model.backbone.compile()
    with torch.no_grad():
        p = np.clip((y * w).sum(0) / np.maximum(w.sum(0), 1e-6), 0.01, 0.99)
        model.cls_b.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    head_ids = {id(q) for q in model.head_params()}
    opt = torch.optim.AdamW([{"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": cfg["lr"]},
                             {"params": model.head_params(), "lr": cfg["lr_head"]}], weight_decay=cfg["wd"])
    spe = cfg["spe_ref"] or len(dl_tr)                   # spe_ref: schedule in the fold runs' epoch units (equal steps)
    total, warm = cfg["epochs"] * spe, int(cfg["warmup"] * spe)
    print(f"steps/epoch {len(dl_tr)} (schedule unit {spe}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]
    mean, std = model.norm_stats()
    step, t0, rows, oof_all, gold_all = 0, time.time(), [], [], []
    for ep in range(cfg["epochs"]):
        model.train()
        sampler.epoch = ep
        tl, n, te = 0.0, 0, time.time()
        for b in dl_tr:
            b = to_dev(b, dev)
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x = gpu_prep(b["x"], cfg["px"], "bilinear", cfg, mean, std, mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                logits = model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"]).float()
            loss = (F.binary_cross_entropy_with_logits(logits, b["y"], reduction="none") * b["w"]).sum() / b["w"].sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tl += loss.detach()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:
                break
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        msg = f"ep {ep} loss {float(tl) / max(n, 1):.4f} time {time.time() - te:.0f}s ({(time.time() - te) / max(n, 1) * 1000:.0f} ms/step)"
        if (ep >= cfg["eval_from"] and (ep + 1) % cfg["eval_every"] == 0) or last:
            pv = as_frame(predict(model, dl_va, cfg, dev, amp), va, outs)
            pg = as_frame(predict(model, dl_gold, cfg, dev, amp), gold, outs)
            m = syn_metrics(pv, pg)
            rows.append({"epoch": ep, **m})
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            msg += " | " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
            if ep >= cfg["save_from"] or last:
                torch.save(model.state_dict(), out_dir / f"model_ep{ep}.pt")
                if cfg["save_opt"]:
                    torch.save({"opt": opt.state_dict(), "step": step, "epoch": ep}, out_dir / f"opt_ep{ep}.pt")
        say(msg)
        if last:
            break
    pd.concat(oof_all).to_parquet(out_dir / "oof.parquet")
    pd.concat(gold_all).to_parquet(out_dir / "gold.parquet")
    json.dump({"curve": rows, "cfg": cfg, "fold": fold, "minutes": (time.time() - t0) / 60}, open(out_dir / "metrics.json", "w"),
              indent=1, default=float)


def parse_value(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v == "True" if v in ("True", "False") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="1")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown key {k}"
        cfg[k] = parse_value(v)
    bags = load_bags(cfg)
    for fold in [int(f) for f in a.folds.split(",")]:
        out = RUNS / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        run_fold(cfg, fold, out, bags)


if __name__ == "__main__":
    main()
