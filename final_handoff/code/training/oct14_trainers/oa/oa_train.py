"""OA expert: one shared CNN over compartment crops (oa_data.py), each output attends only to its own bag:
Medial OA + Medial Meniscus <- medial bag, Lateral OA + Lateral Meniscus <- lateral bag (coronal / axial crops mirrored), PF OA <-
pf bag. So an output cannot read the other compartments ("degenerative knee" shortcut) except through what its own crops show.

h_k = LayerNorm(backbone(crop_k) + type_emb + plane_emb + pos_enc); per output n: a_kn = softmax over its bag's windows of
w_n . tanh(V h_k); logit_n = c_n . sum_k a_kn h_k + b_n. split=True: Medial / Lateral OA share (w, c) with a per-side bias,
and so do the menisci (Split-LR: the lateral bag is mirrored, so one compartment head serves both sides).

conda run -n rsna --no-capture-output python experiments/v1/oct10_oa/main/oa_train.py --name NAME --folds 1 --set key=value ...
Output: runs/NAME/foldF/{config.json, log.txt, oof.parquet, gold.parquet (12 FINDINGS columns, only the 5 outputs filled, + epoch),
metrics.json (OA panel per eval epoch), model_ep{E}.pt}
oct14_all_data changes: fold -1 = all-data run (every non-gold study trains, 'val' = gold-58); spe_ref = N runs the schedule
(warmup, cosine, epochs) in units of N steps (the fold runs' steps per epoch: equal step count and LR curve); save_opt writes
opt_ep{E}.pt (optimizer, step) next to every saved model_ep{E}.pt.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
RUNS = HERE.parent / "runs"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels, text_labels  # noqa: E402
from common.meta import SERIES_TYPES  # noqa: E402
from common.paths import FINDINGS  # noqa: E402
from data import gpu_prep  # noqa: E402
from labels import make_targets  # noqa: E402
from model import sincos  # noqa: E402
from oa_data import BAGS, CompartmentDataset, build_bags, finalize  # noqa: E402
from oa_eval import panel  # noqa: E402

OUTS = ["Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus"]
OUT_BAG = [0, 1, 2, 0, 1]
BML = [("medial", ("medial_femoral_condyle", "medial_tibial_plateau")), ("lateral", ("lateral_femoral_condyle", "lateral_tibial_plateau")),
       ("pf", ("patella",))]                                 # auxiliary marrow-oedema outputs, one per bag (OA analysis A31)


def bml_targets(studies):
    """(n, 3) targets and weights for marrow oedema in the medial / lateral / patellar bones from the v1 contusion cell (any cause,
    degenerative oedema included): 1 where the report places oedema in that compartment, 0 in the other compartments of a
    report that lists oedema elsewhere and in reports with contusion denied / structure_normal; else masked."""
    t = text_labels("v1")
    y, w = np.zeros((len(studies), 3), np.float32), np.zeros((len(studies), 3), np.float32)
    for i, sid in enumerate(studies):
        c = t[sid]["findings"]["contusion"]
        bones = set(c.get("bones") or [])
        if c["status"] in ("denied", "structure_normal"):
            w[i] = 1
        elif bones & {b for _, bs in BML for b in bs}:
            for j, (_, bs) in enumerate(BML):
                y[i, j] = float(bool(bones & set(bs)))
                w[i, j] = 1
    return y, w
DEFAULTS = dict(backbone="efficientnet_b0.ra_in1k", px=224, k=8, k_eval=32, bs=8, epochs=20, lr=1e-3, lr_head=1e-3, wd=1e-2,
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, split=False, men_weight=0.5, bml_weight=0.0, labels="v1prob", nm=0.15,
                side_fix=True, crop_scale=1.0, aug_rot=15, aug_scale=0.15, aug_shift=0.05, aug_erase=0.3, precision="bf16",
                eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0, save_opt=True, device="cuda", max_eval=0,
                crop_norm=False, oa_grade=False, loc_version="v1", poe="", poe_scale=1.0, pair_weight=0.0, pair_bank=1024, iso_weight=1.0, mono=False, lse_r=0.0, trim=0.15, restrict=True, tf_axial=True, rel=False, pos_up=1.0, topk=0, strat=False, init="", hybrid=False, drop_cells="")


def encoder(backbone, pretrained):
    """timm encoder (num_classes=0); 'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (weights/radimagenet, ImageNet
    mean / std; loader copied from experiments/v0/oct10_baseline_3rd/src/model.py; pair it with mono=True)."""
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


class OANet(nn.Module):
    def __init__(self, backbone, hidden=256, drop=0.2, split=False, pretrained=True, bml=False, lse_r=0.0, restrict=True, rel=False, topk=0,
                 hybrid=False):
        super().__init__()
        self.lse_r, self.restrict, self.topk, self.hybrid = lse_r, restrict, topk, hybrid
        self.backbone = encoder(backbone, pretrained)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]   # MobileNetV3: head_hidden_size, not num_features
        self.type_emb = nn.Embedding(len(SERIES_TYPES), d)
        self.plane_emb = nn.Embedding(3, d)
        self.pos_proj = nn.Linear(16, d)
        self.rel_proj = nn.Linear(2, d) if rel else None          # per-window relevance (target / femoral cartilage presence)
        for m in (self.type_emb, self.plane_emb, self.pos_proj) + ((self.rel_proj,) if rel else ()):
            nn.init.zeros_(m.weight)
        nn.init.zeros_(self.pos_proj.bias)
        if rel:
            nn.init.zeros_(self.rel_proj.bias)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        self.split = split
        out_bag = OUT_BAG + ([0, 1, 2] if bml else [])
        n_out = len(out_bag)
        # split: [TF OA, PF OA, meniscus (, TF oedema, PF oedema)] shared over sides
        self.head_of = ([0, 0, 1, 2, 2] + ([3, 3, 4] if bml else [])) if split else list(range(n_out))
        n_head = max(self.head_of) + 1
        self.att_v = nn.Linear(d, hidden)
        self.att_w = nn.Linear(hidden, n_head)
        self.cls_w = nn.Parameter(torch.zeros(n_head, d))
        self.cls_w2 = nn.Parameter(torch.zeros(n_head, d)) if hybrid else None    # lse head of hybrid
        self.cls_b = nn.Parameter(torch.zeros(n_out))           # per output (per side when split)
        self.register_buffer("out_bag", torch.tensor(out_bag), persistent=False)

    def norm_stats(self):
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, mask, typ, plane, pos, bag, rel=None):
        b, k = mask.shape
        f = self.backbone(x).float().view(b, k, -1)
        e = f + self.type_emb(typ) + self.plane_emb(plane) + self.pos_proj(sincos(pos, scale=240.0))
        h = self.norm(e + self.rel_proj(rel) if self.rel_proj is not None else e)
        hd = self.drop(h)
        allowed = ((bag[..., None] == self.out_bag) | (not self.restrict)) & mask[..., None]   # restrict=False: ablation, all bags
        if self.topk:                     # user 10:40: rank windows per output, keep the mean logit of the top-k (window localisation)
            v = self.inst(hd, allowed, self.cls_w).topk(min(self.topk, k), dim=1).values
            fin = torch.isfinite(v)
            return v.where(fin, torch.zeros_like(v)).sum(1) / fin.sum(1).clamp(min=1) + self.cls_b
        if self.lse_r and not self.hybrid:  # I26: per-window logits pooled by log-mean-exp (a few strong windows can carry the study)
            return self.lme(self.inst(hd, allowed, self.cls_w), allowed) + self.cls_b
        s = self.att_w(torch.tanh(self.att_v(hd)))[..., self.head_of]          # (b, k, n_out)
        s = s.masked_fill(~allowed, float("-inf"))
        a = torch.softmax(s, 1)
        z = torch.einsum("bkn,bkd->bnd", a, hd)
        att = (z * self.cls_w[self.head_of]).sum(-1)
        if self.hybrid:                   # attention head + lse head (own weights), averaged: C4's two members in one model
            return 0.5 * (att + self.lme(self.inst(hd, allowed, self.cls_w2), allowed)) + self.cls_b
        return att + self.cls_b

    def inst(self, hd, allowed, w):
        """Per-window logits (b, k, n_out), -inf outside each output's bag."""
        return torch.einsum("bkd,nd->bkn", hd, w[self.head_of]).masked_fill(~allowed, float("-inf"))

    def lme(self, s, allowed):
        """Log-mean-exp over windows with sharpness lse_r."""
        return (torch.logsumexp(self.lse_r * s, 1) - torch.log(allowed.sum(1).clamp(min=1).float())) / self.lse_r


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled indices offset by epoch * n, so persistent workers can derive the epoch (as main/train.py). weights: draw n
    indices per epoch with replacement in proportion to them instead (pos_up)."""
    def __init__(self, n, seed, weights=None):
        self.n, self.seed, self.epoch, self.weights = n, seed, 0, weights

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        idx = (torch.randperm(self.n, generator=g) if self.weights is None
               else torch.multinomial(self.weights, self.n, replacement=True, generator=g))
        return iter((idx + self.epoch * self.n).tolist())

    def __len__(self):
        return self.n


def crop_normalise(x):
    """uint8 (B, K, 3, H, W) -> each window rescaled to its own 1st-99th percentile, back to uint8 (analysis A23: removes the
    series-level brightness = effusion cue)."""
    f = x.float()
    flat = f.flatten(2)[..., ::7]
    q = torch.quantile(flat, torch.tensor([0.01, 0.99], device=x.device), dim=2)            # (2, B, K)
    lo, hi = q[0][..., None, None, None], q[1][..., None, None, None]
    return ((f - lo) / (hi - lo).clamp(min=1.0) * 255).clamp(0, 255).to(torch.uint8)


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


POS, NEG = 0.79, 0.16    # v1prob OA targets: meets >= 0.8; denied / normal 0.01-0.05, general normal 0.08-0.15, not mentioned 0.15


def pair_loss(s, y, bank_s, bank_y):
    """I16 compartment pair ranking: for OA outputs F != G, a knee with F+ G- should outrank a knee with F- G+ on F (what the comp
    test measures). s, y: the batch's OA logits / targets (unknown targets set to 0.5); pairs within the batch and against a bank
    of recent detached scores, at least one side from the batch. Mean softplus(s_F(b) - s_F(a)) over pairs."""
    S, Y, n = torch.cat([s, bank_s]), torch.cat([y, bank_y]), len(s)
    tot, cnt = s.new_zeros(()), 0
    for f in range(3):
        for g in range(3):
            if f == g:
                continue
            ia = ((Y[:, f] > POS) & (Y[:, g] < NEG)).nonzero()[:, 0]
            ib = ((Y[:, f] < NEG) & (Y[:, g] > POS)).nonzero()[:, 0]
            keep = (ia[:, None] < n) | (ib[None, :] < n)
            if keep.any():
                tot = tot + F.softplus(S[ib, f][None, :] - S[ia, f][:, None])[keep].sum()
                cnt += int(keep.sum())
    return tot / max(cnt, 1)


@torch.no_grad()
def predict(model, loader, cfg, dev, amp):
    model.eval()
    out = []
    for b in loader:
        b = to_dev(b, dev)
        x = gpu_prep(crop_normalise(b["x"]) if cfg["crop_norm"] else b["x"], cfg["px"], "bilinear", None,
                     *model.norm_stats(), mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            out.append(torch.sigmoid(model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"], b["rel"]).float()).cpu())
    return torch.cat(out).numpy()


def as12(p, idx):
    d = pd.DataFrame(np.nan, index=idx, columns=FINDINGS)
    d[OUTS] = p[:, :len(OUTS)]
    return d


def run_fold(cfg, fold, out_dir, bags):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev, amp = cfg["device"], {"bf16": torch.bfloat16, "fp32": None}[cfg["precision"]]
    torch.backends.cudnn.benchmark = True
    folds = get_folds(5, 0)
    folds = folds[folds.index.isin(bags.keys())]
    tr = folds.index[folds.fold != fold].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in bags]
    if fold < 0:                                          # all-data run (fold -1): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    if cfg["max_eval"]:                                   # smoke tests
        va, gold = va[:cfg["max_eval"]], gold[:cfg["max_eval"]]
    y, w = make_targets(tr, cfg)
    cols = [FINDINGS.index(f) for f in OUTS]
    y, w = y[:, cols], w[:, cols]
    w[:, 3:] *= cfg["men_weight"]
    if cfg["drop_cells"]:                                  # cleaned labels (confident learning): weight 0 for listed (study, finding) cells
        dc = pd.read_csv(HERE.parent / cfg["drop_cells"])
        pos = {s: i for i, s in enumerate(tr)}
        n_drop = 0
        for s, f in zip(dc.StudyInstanceUID, dc.finding):
            if s in pos and f in OUTS:
                w[pos[s], OUTS.index(f)] = 0
                n_drop += 1
        print(f"drop_cells: {n_drop} training cells set to weight 0", flush=True)
    if cfg["iso_weight"] != 1:                             # I7: isolated OA cells (F+ others-, F- others+) weighted up
        pos, neg = (y[:, :3] > POS) & (w[:, :3] > 0), (y[:, :3] < NEG) & (w[:, :3] > 0)
        for f in range(3):
            o = [g for g in range(3) if g != f]
            iso = (pos[:, f] & neg[:, o].all(1)) | (neg[:, f] & pos[:, o].any(1))
            w[iso, f] *= cfg["iso_weight"]
    if cfg["bml_weight"]:
        yb, wb = bml_targets(tr)
        y, w = np.concatenate([y, yb], 1), np.concatenate([w, wb * cfg["bml_weight"]], 1)
    poe = torch.zeros(len(tr), y.shape[1], device=dev)          # residual training: frozen proxy logits added in the loss only
    if cfg["poe"]:
        t = pd.read_csv(HERE.parent / cfg["poe"]).set_index("StudyInstanceUID").reindex(tr)
        assert not t.isna().any().any(), "poe proxy missing for training studies"
        for c in t.columns:
            poe[:, OUTS.index(c)] = torch.tensor(t[c].to_numpy(np.float32), device=dev) * cfg["poe_scale"]
    log = open(out_dir / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    ds_tr = CompartmentDataset(tr, bags, y, w, cfg["k"], True, cfg["px"], cfg["seed"], cfg["strat"])
    oa_rel = (y[:, :3] * (w[:, :3] > 0)).max(1) >= 0.45                  # pos_up: studies with meets / unsettled / conflict OA
    sampler = EpochSampler(len(tr), cfg["seed"], torch.tensor(np.where(oa_rel, cfg["pos_up"], 1.0)) if cfg["pos_up"] != 1 else None)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(CompartmentDataset(ids, bags, None, None, cfg["k_eval"], False, cfg["px"]), batch_size=max(1, cfg["bs"] // 2),
                                num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    model = OANet(cfg["backbone"], cfg["hidden"], cfg["drop"], cfg["split"], bml=cfg["bml_weight"] > 0, lse_r=cfg["lse_r"],
                  restrict=cfg["restrict"], rel=cfg["rel"], topk=cfg["topk"], hybrid=cfg["hybrid"]).to(dev).to(memory_format=torch.channels_last)
    if cfg["init"]:                         # user 11:00: start from a main-pipeline encoder trained on the same fold split (no leakage)
        sd = torch.load(HERE.parent / cfg["init"].format(fold=fold), map_location="cpu")
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    with torch.no_grad():
        p = np.clip((y * w).sum(0) / np.maximum(w.sum(0), 1e-6), 0.01, 0.99)
        model.cls_b.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        if cfg["poe"]:                                          # the proxy carries the prior: residual outputs start at 0
            model.cls_b[[OUTS.index(c) for c in pd.read_csv(HERE.parent / cfg["poe"], nrows=1).columns[1:]]] = 0.0
    head_ids = {id(q) for q in model.head_params()}
    opt = torch.optim.AdamW([{"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": cfg["lr"]},
                             {"params": model.head_params(), "lr": cfg["lr_head"]}], weight_decay=cfg["wd"])
    spe = cfg["spe_ref"] or len(dl_tr)                   # spe_ref: schedule in the fold runs' epoch units (equal steps)
    total, warm = cfg["epochs"] * spe, int(cfg["warmup"] * spe)
    print(f"steps/epoch {len(dl_tr)} (schedule unit {spe}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]
    mean, std = model.norm_stats()
    step, t0, rows, oof_all, gold_all = 0, time.time(), [], [], []
    bank_s, bank_y = torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev)    # pair_loss bank (newest first)
    for ep in range(cfg["epochs"]):
        model.train()
        sampler.epoch = ep
        tl, n, te = 0.0, 0, time.time()
        for b in dl_tr:
            b = to_dev(b, dev)
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x = gpu_prep(crop_normalise(b["x"]) if cfg["crop_norm"] else b["x"], cfg["px"], "bilinear", cfg, mean, std,
                         mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                logits = model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"], b["rel"]).float()
            loss = (F.binary_cross_entropy_with_logits(logits + poe[b["idx"]], b["y"], reduction="none") * b["w"]).sum() / b["w"].sum().clamp(min=1)
            if cfg["pair_weight"]:
                s, yo = logits[:, :3], torch.where(b["w"][:, :3] > 0, b["y"][:, :3], 0.5)
                loss = loss + cfg["pair_weight"] * pair_loss(s, yo, bank_s, bank_y)
                bank_s, bank_y = torch.cat([s.detach(), bank_s])[:cfg["pair_bank"]], torch.cat([yo, bank_y])[:cfg["pair_bank"]]
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
        msg = f"ep {ep} loss {float(tl) / max(n, 1):.4f} time {time.time() - te:.0f}s"
        if (ep >= cfg["eval_from"] and (ep + 1) % cfg["eval_every"] == 0) or last:
            pv = as12(predict(model, dl_va, cfg, dev, amp), va)
            pg = as12(predict(model, dl_gold, cfg, dev, amp), gold)
            m = panel(pv[OUTS[:3]].pipe(lambda d: d.assign(fold=fold)), pg)
            rows.append({"epoch": ep, **{k: v for k, v in m.items()}})
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            msg += " | " + " ".join(f"{k} {m[k]:.4f}" for k in ("text3", "strat3", "comp3", "thr3", "gold3"))
            msg += " | " + " ".join(f"{k} {m[k]:.4f}" for k in ("text|Medial OA", "text|Lateral OA", "text|PF OA", "comp|Lateral OA"))
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
    assert cfg["side_fix"], "the localiser boxes are in the side-corrected view"
    bags = finalize(build_bags(cfg["crop_scale"], cfg["loc_version"], cfg["trim"], cfg["tf_axial"]))
    for fold in [int(f) for f in a.folds.split(",")]:
        out = RUNS / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        run_fold(cfg, fold, out, bags)


if __name__ == "__main__":
    main()
