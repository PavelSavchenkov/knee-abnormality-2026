"""Bone expert for Fracture / Contusion: one shared CNN over femur / tibia / patella crops (bone_data.py), per-output attention pooling
over the windows of the bags that output may read; adapted from experiments/v1/oct11_synovitis_crops/main/syn_train.py.

h_k = LayerNorm(backbone(crop_k) + type_emb + plane_emb + bag_emb + pos_enc); per output n: a_kn = softmax over its allowed windows of
w_n . tanh(V h_k); logit_n = c_n . sum_k a_kn h_k + b_n.
Outputs (outs): frac / cont = the main-pipeline targets of Fracture / Contusion (labels.make_targets: v1prob, or tgt_csv variants),
read from every bag; frac1 / cont1 = the same with plain v1prob targets (tgt_csv ignored: a v1 head next to a v3 head on one
backbone, saved as Fracture_v1 / Contusion_v1); c_* = contusion location aux targets from the v1 contusion bones (1 = listed bone, 0 = other bones of a
positive or contusion denied / normal, else masked), each read only from its own bone's bag (loc_restrict); optional f_* = fracture
site aux targets from the v2 fracture facts (femur / tibia / fibula / patella; 1 = a stated fracture there, 0 = a fracture elsewhere or
fracture denied / normal, else masked), restricted the same way.
Loss = masked BCE weighted by w_main (frac / cont) and w_loc (c_*), normalised by the summed weights of the batch; hn_w multiplies
the frac weight of fracture-denied / normal knees with a v1 contusion and the cont weight of v1 degenerative oedema (contusion
excluded, cause degeneration): the hardest negatives of D (ledger 14:40).

conda run -n rsna --no-capture-output python experiments/v1/oct11_fracture_contusion/main/bone_train.py --name NAME --folds 1 --set key=value ...
Backbone lr 3e-4 (head 1e-3): at 1e-3 the 20-epoch run damaged the pretrained encoder (loss above the prior for 8 epochs, ledger 02:32).
Output: runs/NAME/foldF/{config.json, log.txt, oof.parquet, gold.parquet (FINDINGS with Fracture / Contusion filled, c_* columns,
epoch), metrics.json, model_ep{E}.pt}
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
from bone_data import BoneDataset, build_bags, finalize  # noqa: E402
from common.labels import status_table, text_labels  # noqa: E402
sys.path.insert(0, str(EXP / "scripts"))
from fc_eval import panel  # noqa: E402

LOC = {"c_lfc": ("lateral_femoral_condyle", "femur"), "c_mfc": ("medial_femoral_condyle", "femur"), "c_fem": ("femur_other", "femur"),
       "c_ltp": ("lateral_tibial_plateau", "tibia"), "c_mtp": ("medial_tibial_plateau", "tibia"), "c_tib": ("tibia_other", "tibia"),
       "c_fib": ("fibula", "tibia"), "c_pat": ("patella", "patella")}
FLOC = {"f_fem": ({"lateral_femoral_condyle", "medial_femoral_condyle", "femur_other", "trochlea"}, "femur"),
        "f_tib": ({"lateral_tibial_plateau", "medial_tibial_plateau", "tibia_other"}, "tibia"), "f_fib": ({"fibula"}, "tibia"),
        "f_pat": ({"patella"}, "patella")}
FR_KINDS = {"fracture", "avulsion_fracture", "osteochondral_fracture", "impaction_injury", "insufficiency_fracture"}
OUT_COL = {"frac": "Fracture", "cont": "Contusion", "frac1": "Fracture_v1", "cont1": "Contusion_v1"} | {c: c for c in LOC} | {c: c for c in FLOC}
DEFAULTS = dict(backbone="efficientnet_b0.ra_in1k", px=256, k=8, k_eval=32, bs=8, epochs=20, lr=3e-4, lr_head=1e-3, wd=1e-2,
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, bags="femur,tibia,patella", outs="frac,cont," + ",".join(LOC),
                w_main=1.0, w_loc=0.5, loc_restrict=True, fs_outs="", bb_pool="avg", labels="v1prob", nm=0.15, sharpen=0.0, hn_w=1.0, tgt_csv="",
                crop_scale=1.0, trim=0.05, series="all", planes="", loc="data/loc_pred_v3", pat_margin=8.0, bagsum=False, init="", aug_rot=15, aug_scale=0.15, aug_shift=0.05,
                aug_erase=0.3, precision="bf16", eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0, save_opt=True,
                device="cuda", max_eval=0, mono=False, compile=False)


def encoder(backbone, pretrained, global_pool="avg"):
    """timm encoder (num_classes=0); global_pool 'avg' (default) or 'avgmax' (0.5 x (avg + max): keeps small focal signals);
    'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (ImageNet mean / std; pair it with mono=True)."""
    if backbone != "radimagenet_resnet50":
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool=global_pool)
    enc = timm.create_model("resnet50", pretrained=False, num_classes=0)
    if pretrained:
        sd = torch.load(HERE.parents[3] / "weights/radimagenet/ResNet50.pt", map_location="cpu")
        names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
        sd = {names[k.split(".")[1]] + k[len("backbone.") + len(k.split(".")[1]):]: v for k, v in sd.items()}
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        assert not unexpected and not [m for m in missing if not m.startswith("fc")], (missing, unexpected)
    enc.pretrained_cfg = dict(enc.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return enc


class BoneNet(nn.Module):
    def __init__(self, backbone, out_bags, n_bags, hidden=256, drop=0.2, pretrained=True, bagsum=False, out_fs=None, bb_pool="avg"):
        """out_bags: (n_out, n_bags) bool, which bags each output may read. bagsum: attention pooling inside each bag separately and
        one readout per (output, bag), summed (a bag's evidence is not diluted by the windows of the other bags). out_fs: (n_out,) bool,
        outputs that read only fat-suppressed windows (a study without any falls back to all windows for that output)."""
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        self.backbone = encoder(backbone, pretrained, bb_pool)
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
        self.register_buffer("out_fs", torch.as_tensor(np.zeros(n_out, bool) if out_fs is None else out_fs), persistent=False)
        self.register_buffer("fs_type", torch.tensor([t.endswith("_FS") for t in SERIES_TYPES]), persistent=False)

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
        if self.out_fs.any():                                                      # FS-only outputs, unless the study has no FS window there
            fs = allowed & self.fs_type[typ][..., None]
            keep = ~self.out_fs | ~fs.any(1)                                       # (b, n_out): output keeps all windows
            allowed = torch.where(keep[:, None, :], allowed, fs)
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


def loc_targets(studies):
    """(n, len(LOC)) contusion location targets from the v1 contusion bones; NaN = unknown."""
    st, t = status_table("v1"), text_labels("v1")
    out = np.full((len(studies), len(LOC)), np.nan, np.float32)
    for i, sid in enumerate(studies):
        s = st.at[sid, "contusion"]
        bones = t[sid]["findings"]["contusion"].get("bones") or []
        if s in ("denied", "structure_normal"):
            out[i] = 0.0
        elif s in ("meets", "unsettled") and bones:
            out[i] = [float(b in bones) for b, _ in LOC.values()]
    return out


def frac_loc_targets(studies):
    """(n, len(FLOC)) fracture site aux targets from the v2 facts of v1 fracture-positive reports; NaN = unknown."""
    st, v2 = status_table("v1"), text_labels("v2")
    out = np.full((len(studies), len(FLOC)), np.nan, np.float32)
    for i, sid in enumerate(studies):
        s = st.at[sid, "fracture"]
        if s in ("denied", "structure_normal", "general_normal"):
            out[i] = 0.0
        elif s in ("meets", "unsettled", "hedged"):
            sites = {f["structure"] for f in v2[sid]["facts"] if f["finding"] in FR_KINDS and f["assertion"] not in ("absent", "absent_qualified")}
            if sites & set().union(*[v[0] for v in FLOC.values()]):
                out[i] = [float(bool(sites & v[0])) for v in FLOC.values()]
    return out


def make_bone_targets(studies, cfg, outs):
    """(n, n_out) targets and loss weights; unknown location cells get weight 0."""
    yt, _ = make_targets(studies, cfg)
    yt1 = make_targets(studies, cfg | {"tgt_csv": ""})[0] if {"frac1", "cont1"} & set(outs) else None
    yl, yf = loc_targets(studies), frac_loc_targets(studies)
    y, w = np.zeros((len(studies), len(outs)), np.float32), np.zeros((len(studies), len(outs)), np.float32)
    for j, o in enumerate(outs):
        if o in LOC or o in FLOC:
            v = yl[:, list(LOC).index(o)] if o in LOC else yf[:, list(FLOC).index(o)]
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_loc"]
        elif o in ("frac1", "cont1"):
            y[:, j], w[:, j] = yt1[:, FINDINGS.index(OUT_COL[o][:-3])], cfg["w_main"]
        else:
            y[:, j], w[:, j] = yt[:, FINDINGS.index(OUT_COL[o])], cfg["w_main"]
    if cfg["hn_w"] != 1.0:                                # hard negatives: fracture denied / normal with a bruise; degenerative oedema
        st, v1 = status_table("v1").loc[studies], text_labels("v1")
        f_hn = (st.fracture.isin(("denied", "structure_normal")) & (st.contusion == "meets")).to_numpy()
        c_hn = np.array([st.at[s, "contusion"] == "excluded" and v1[s]["findings"]["contusion"].get("cause") == "degeneration" for s in studies])
        for j, o in enumerate(outs):
            if o in ("frac", "frac1"):
                w[f_hn, j] *= cfg["hn_w"]
            elif o in ("cont", "cont1"):
                w[c_hn, j] *= cfg["hn_w"]
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
    d = pd.DataFrame(np.nan, index=pd.Index(idx, name="StudyInstanceUID"), columns=FINDINGS)
    for j, o in enumerate(outs):
        d[OUT_COL[o]] = p[:, j]
    return d


def fc_metrics(pv, pg):
    m = panel(pv, pg)
    return {k: m[k] for k in ("Frac_text", "Frac_hard", "Frac_gold", "Cont_text", "Cont_hard", "Cont_gold") if k in m}


def load_bags(cfg):
    """finalize(build_bags(...)), cached per bag config in runs/_bags/."""
    key = {k: cfg[k] for k in ("bags", "crop_scale", "trim", "series", "planes", "loc")} | {"v": 2} | ({"pm": cfg["pat_margin"]} if cfg["pat_margin"] != 8.0 else {})                     # v: bag geometry version (bump when bone_data crop rules change)
    f = RUNS / "_bags" / (hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10] + ".pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    bags = finalize(build_bags(cfg["bags"].split(","), cfg["crop_scale"], cfg["trim"], cfg["series"], cfg["planes"], cfg["loc"], cfg["pat_margin"]))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(bags, open(f, "wb"))
    return bags


def run_fold(cfg, fold, out_dir, bags):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev, amp = cfg["device"], {"bf16": torch.bfloat16, "fp32": None}[cfg["precision"]]
    torch.backends.cudnn.benchmark = True
    bag_names, outs = cfg["bags"].split(","), cfg["outs"].split(",")
    out_bags = np.ones((len(outs), len(bag_names)), bool)
    for j, o in enumerate(outs):                          # location heads read only their own bone's bag (when it is in the bags)
        own = LOC[o][1] if o in LOC else FLOC[o][1] if o in FLOC else None
        if cfg["loc_restrict"] and own in bag_names:
            out_bags[j] = [b == own for b in bag_names]
    folds = get_folds(5, 0)
    folds = folds[folds.index.isin(bags.keys())]
    tr = folds.index[folds.fold != fold].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in bags]
    if fold < 0:                                          # all-data run (fold -1): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    if cfg["max_eval"]:                                   # smoke tests
        va, gold = va[:cfg["max_eval"]], gold[:cfg["max_eval"]]
    y, w = make_bone_targets(tr, cfg, outs)
    log = open(out_dir / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    ds_tr = BoneDataset(tr, bags, bag_names, y, w, cfg["k"], True, cfg["px"], cfg["seed"], len(outs))
    sampler = EpochSampler(len(tr), cfg["seed"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(BoneDataset(ids, bags, bag_names, None, None, cfg["k_eval"], False, cfg["px"], n_out=len(outs)),
                                batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    out_fs = np.array([o in cfg["fs_outs"].split(",") for o in outs])
    model = BoneNet(cfg["backbone"], out_bags, len(bag_names), cfg["hidden"], cfg["drop"], bagsum=cfg["bagsum"], out_fs=out_fs, bb_pool=cfg["bb_pool"]).to(dev).to(memory_format=torch.channels_last)
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
            m = fc_metrics(pv, pg)
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
