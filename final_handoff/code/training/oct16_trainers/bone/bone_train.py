"""Copied from experiments/v1/oct11_fracture_contusion/main/bone_train.py (2026-10-14, backbone sweep): + llrd (timm layer-wise
lr decay of the backbone, biases / norms without weight decay, as main/train.py) and drop_path; fc_eval.py copied next to it.

Bone expert for Fracture / Contusion: one shared CNN over femur / tibia / patella crops (bone_data.py), per-output attention pooling
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
pw (oct16_experts 10-19, overhaul item 4; port of main/train.py's 'pw' arm): pw_weight > 0 adds a RankNet loss on the IMAGE logits of the main
outputs (frac / cont / frac1 / cont1 -> Fracture / Contusion) over TRUSTED cells only (pw_csv: hard 1 / 0, empty = no pair) and, with pw_cross,
only between studies of DIFFERENT sites (in batch + a FIFO memory of pw_queue studies); site_mode 'vague' adds main/train.py's training-only
site term on those outputs (per (site, cell group) offsets, cell group 0 = explicit status, 1 = silent / general_normal, 2 = other non-explicit;
sites with < site_min training studies share 'other'; L2 site_l2; never used at prediction). Both inert at the defaults.
Backbone lr 3e-4 (head 1e-3): at 1e-3 the 20-epoch run damaged the pretrained encoder (loss above the prior for 8 epochs, ledger 02:32).
Output: runs/NAME/foldF/{config.json, log.txt, oof.parquet, gold.parquet (FINDINGS with Fracture / Contusion filled, c_* columns,
epoch), metrics.json, model_ep{E}.pt}"""
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
from timm.optim import param_groups_layer_decay  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.meta import SERIES_TYPES  # noqa: E402
from common.paths import FINDINGS, KEY_OF  # noqa: E402
from data import gpu_prep  # noqa: E402
from labels import make_targets  # noqa: E402
from model import sincos  # noqa: E402
from bone_data import BoneDataset, build_bags, finalize  # noqa: E402
from common.labels import status_table, text_labels  # noqa: E402
from fc_eval import panel  # noqa: E402

LOC = {"c_lfc": ("lateral_femoral_condyle", "femur"), "c_mfc": ("medial_femoral_condyle", "femur"), "c_fem": ("femur_other", "femur"),
       "c_ltp": ("lateral_tibial_plateau", "tibia"), "c_mtp": ("medial_tibial_plateau", "tibia"), "c_tib": ("tibia_other", "tibia"),
       "c_fib": ("fibula", "tibia"), "c_pat": ("patella", "patella")}
FLOC = {"f_fem": ({"lateral_femoral_condyle", "medial_femoral_condyle", "femur_other", "trochlea"}, "femur"),
        "f_tib": ({"lateral_tibial_plateau", "medial_tibial_plateau", "tibia_other"}, "tibia"), "f_fib": ({"fibula"}, "tibia"),
        "f_pat": ({"patella"}, "patella")}
FR_KINDS = {"fracture", "avulsion_fracture", "osteochondral_fracture", "impaction_injury", "insufficiency_fracture"}
OUT_COL = {"frac": "Fracture", "cont": "Contusion", "frac1": "Fracture_v1", "cont1": "Contusion_v1"} | {c: c for c in LOC} | {c: c for c in FLOC} \
    | {"fnc": "frac_not_cont", "cnf": "cont_not_frac"}   # oct16_experts: de-coupling outputs (design.md lever 5), targets yF (1 - yC) / yC (1 - yF)
OUT_COL |= {"img_frac": "img_Fracture", "img_cont": "img_Contusion"}   # 10-20 P2: image-read heads (img_bone_csv targets, masked), saved columns
DEFAULTS = dict(cache="fov160_px400_q95", backbone="efficientnet_b0.ra_in1k", px=256, k=8, k_eval=32, bs=8, epochs=20, lr=3e-4, lr_head=1e-3, wd=1e-2,
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, bags="femur,tibia,patella", outs="frac,cont," + ",".join(LOC),
                img_bone_csv="", w_img=1.0,   # 10-20 P2 image heads (outs img_frac / img_cont); default OFF: no such outs -> recipes unchanged
                w_main=1.0, w_loc=0.5, loc_restrict=True, fs_outs="", bb_pool="avg", labels="v1prob", nm=0.15, sharpen=0.0, hn_w=1.0, w_dec=0.5, tgt_csv="", loc_img_csv="", train_ids="", val_ids="", eval_ckpt="", init_full="", samp_ids="", samp_mult=1.0,
                img_w_csv="", w_img_end=-1.0, rank_img=0.0, lab_per_batch=0, lab_pos=0, opus_csv="data/ext_p2_main.csv", mon100=False, tgt_w=0.0, tgt_gw=0.0, tgt_wmask_csv="",   # 10-20 W4 (default off): Opus-certainty cell weights, w_img ramp, image-head RankNet
                crop_scale=1.0, trim=0.05, series="all", planes="", loc="data/loc_pred_v3", pat_margin=8.0, bagsum=False, init="", aug_rot=15, aug_scale=0.15, aug_shift=0.05,
                aug_erase=0.3, precision="bf16", eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0,
                device="cuda", max_eval=0, mono=False, compile=False, llrd=0.0, drop_path=0.0,
                series_heads="", sd_weight=0.5, sd_agg_weight=0.5, loc_hn=False, sd_outs="", sd_fs=False, sd_prior=False, pair=False,
                pw_weight=0.0, pw_csv="", pw_queue=512, pw_cross=True, pw_site_csv="data/site_train.csv", site_csv="", site_mode="", site_l2=1e-3,
                site_min=20)
MAIN_OUT = {"frac": "Fracture", "cont": "Contusion", "frac1": "Fracture", "cont1": "Contusion"}   # pw / site term act on these outputs only
EXPLICIT = ("meets", "denied", "structure_normal")


def encoder(backbone, pretrained, global_pool="avg", drop_path=0.0):
    """timm encoder (num_classes=0); global_pool 'avg' (default) or 'avgmax' (0.5 x (avg + max): keeps small focal signals);
    'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (ImageNet mean / std; pair it with mono=True)."""
    if backbone != "radimagenet_resnet50":
        kw = {"drop_path_rate": drop_path} if drop_path else {}
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool=global_pool, **kw)
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
    def __init__(self, backbone, out_bags, n_bags, hidden=256, drop=0.2, pretrained=True, bagsum=False, out_fs=None, bb_pool="avg", drop_path=0.0,
                 series_heads="", pair=False):
        """out_bags: (n_out, n_bags) bool, which bags each output may read. bagsum: attention pooling inside each bag separately and
        one readout per (output, bag), summed (a bag's evidence is not diluted by the windows of the other bags). out_fs: (n_out,) bool,
        outputs that read only fat-suppressed windows (a study without any falls back to all windows for that output)."""
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        self.backbone = encoder(backbone, pretrained, bb_pool, drop_path)
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
        # oct16_experts U22 (SD port, as oa_train): per-series-group heads (plane x FS: 6 groups) inside each output's allowed windows,
        # deep-supervised, DETACHED into a per-output aggregator over the present groups; "shared" (main head + group bias) | "sep"
        self.series_heads = series_heads
        assert not series_heads or (series_heads in ("shared", "sep") and not bagsum), series_heads
        if series_heads:
            self.sd_b = nn.Parameter(torch.zeros(6, n_out))
            self.sd_w = nn.Parameter(torch.zeros(6, n_out, d)) if series_heads == "sep" else None
            self.sd_agg, self.sd_agg_b = nn.Parameter(torch.zeros(6, n_out)), nn.Parameter(torch.zeros(n_out))
        # oct16_experts 10-18 bpair: a window token also reads its partner (same bag / plane, other FS class) through a zero-initialised
        # projection of [f, f_partner] (starts as b_v3; learns FS x non-FS conjunctions per slab); x holds k windows, then k partners
        self.pair = pair
        if pair:
            self.pair_proj = nn.Linear(2 * d, d)
            nn.init.zeros_(self.pair_proj.weight)
            nn.init.zeros_(self.pair_proj.bias)
        # oct16_experts C60 (user 20:35: why SD lifts Contusion in the generalist but not here): sd_mask (6, n_out) limits the SD deep
        # supervision to some outputs (sd_outs) and / or the FS groups (sd_fs); set by run_fold, training only
        self.sd_mask, self.sd_masked = None, False

    def norm_stats(self):
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, mask, typ, plane, pos, bag, pm=None):
        b, k = mask.shape
        f = self.backbone(x).float()
        if self.pair:
            f = f.view(b, 2 * k, -1)
            f = f[:, :k] + pm[..., None].float() * self.pair_proj(torch.cat([f[:, :k], f[:, k:]], -1)).float()
        else:
            f = f.view(b, k, -1)
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
        if self.series_heads and self.training:
            self.sd_out = self.series_forward(s, allowed, typ, hd)
        return (z * self.cls_w).sum(-1) + self.cls_b

    def series_forward(self, s, allowed, typ, hd):
        """SD heads (as oa_train.OANet.series_forward): group logits (b, 6, n), present (b, 6, n), aggregator logits (b, n)."""
        grp = (typ // 10) * 2 + (typ % 2)
        logits, present = [], []
        for j in range(6):
            m = allowed & (grp == j)[..., None]
            a = torch.nan_to_num(torch.softmax(s.masked_fill(~m, float("-inf")), 1))
            z = torch.einsum("bkn,bkd->bnd", a, hd)
            w, bias = (self.cls_w, self.cls_b + self.sd_b[j]) if self.series_heads == "shared" else (self.sd_w[j], self.sd_b[j])
            logits.append((z * w).sum(-1) + bias)
            present.append(m.any(1))
        gl, pr = torch.stack(logits, 1), torch.stack(present, 1)
        if self.sd_mask is not None:
            pr = pr & self.sd_mask[None]
        wa = self.sd_agg[None].expand(gl.shape[0], -1, -1).masked_fill(~pr, float("-inf"))
        agg = (torch.nan_to_num(torch.softmax(wa, 1)) * gl.detach()).sum(1) + self.sd_agg_b   # stop-grad: the aggregator cannot shape the heads
        return gl, pr, agg


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled indices offset by epoch * n, so persistent workers can derive the epoch. With weights (n,) (10-20 S3, mainp's draw): study i
    appears about n * w_i / sum(w) times per epoch (floor + random rounding of the expected count), still n draws per epoch.
    10-20 W4: with weights, a study's r-th draw within an epoch gets the dataset key epoch * 16 + r (r < 16), so repeats get fresh crops
    (the datasets seed their window choice from (seed, epoch key, index); the unweighted path keeps the key = epoch).
    W4-8 balanced batches: with lab (bool (n,)) and k, every consecutive group of bs draws (one DataLoader batch) holds exactly k labelled
    studies (cycled through shuffled passes of the labelled pool) and bs - k unlabelled ones (the same), n // bs batches, shuffled inside each batch.
    U100 (USER 10-20: separate positive / negative pools): with pos (bool (n,)) and kpos, the k labelled draws of a batch are kpos from the
    labelled POSITIVES and k - kpos from the labelled negatives."""
    def __init__(self, n, seed, weights=None, lab=None, k=0, bs=0, pos=None, kpos=0):
        self.n, self.seed, self.epoch, self.weights, self.rekey = n, seed, 0, weights, weights is not None or lab is not None
        self.lab, self.k, self.bs, self.pos, self.kpos = lab, k, bs, pos, kpos

    def cycle(self, pool, m, g):
        return torch.cat([pool[torch.randperm(len(pool), generator=g)] for _ in range(-(-m // len(pool)))])[:m]

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        if self.lab is not None:
            nb = self.n // self.bs
            if self.pos is None:
                parts = [self.cycle(torch.where(self.lab)[0], nb * self.k, g).view(nb, self.k)]
            else:
                parts = [self.cycle(torch.where(self.lab & self.pos)[0], nb * self.kpos, g).view(nb, self.kpos),
                         self.cycle(torch.where(self.lab & ~self.pos)[0], nb * (self.k - self.kpos), g).view(nb, self.k - self.kpos)]
            bt = torch.cat(parts + [self.cycle(torch.where(~self.lab)[0], nb * (self.bs - self.k), g).view(nb, self.bs - self.k)], 1)
            idx = torch.gather(bt, 1, torch.argsort(torch.rand(nb, self.bs, generator=g), 1)).reshape(-1)
        elif self.weights is None:
            idx = torch.randperm(self.n, generator=g)
        else:
            c = self.weights / self.weights.sum() * self.n
            idx = torch.repeat_interleave(torch.arange(self.n), (c.floor() + (torch.rand(self.n, generator=g) < c.frac())).long())
            idx = torch.cat([idx[torch.randperm(len(idx), generator=g)], torch.multinomial(self.weights, self.n, True, generator=g)])[: self.n]
        if self.rekey:                                 # 10-20 W4: the r-th repeat of a study in this epoch -> its own crop key
            cnt, rep = {}, []
            for v in idx.tolist():
                rep.append(min(cnt.get(v, 0), 15))
                cnt[v] = cnt.get(v, 0) + 1
            return iter((idx + (self.epoch * 16 + torch.tensor(rep)) * self.n).tolist())
        return iter((idx + self.epoch * self.n).tolist())

    def __len__(self):
        return self.n


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def loc_targets(studies, hn=False):
    """(n, len(LOC)) contusion location targets from the v1 contusion bones; NaN = unknown. hn (oct16_experts C57, user 20:05 transfer):
    contusion 'excluded' knees whose oedema has another cause (degeneration / fracture) become 0 for every per-bone head (they were NaN),
    so the strict per-bone heads learn contusion, not any marrow oedema."""
    st, t = status_table("v1"), text_labels("v1")
    out = np.full((len(studies), len(LOC)), np.nan, np.float32)
    for i, sid in enumerate(studies):
        s = st.at[sid, "contusion"]
        bones = t[sid]["findings"]["contusion"].get("bones") or []
        if s in ("denied", "structure_normal") or (hn and s == "excluded" and t[sid]["findings"]["contusion"].get("cause") in ("degeneration", "fracture")):
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


class PairMemory:
    """'pw' (copy of main/train.py's PairMemory + AUCMemory push, 10-19): RankNet (softplus) pairwise loss over trusted cells, cross-site
    pairs only when cross, in batch + a FIFO memory of recent studies; i == j pairs drop out (same site)."""
    def __init__(self, size, cross):
        self.size, self.cross, self.mem = size, cross, None

    def loss(self, s, y, w, site):
        b, mem = s.shape[0], self.mem
        all_s, all_y, all_w, all_site = (s, y, w, site) if mem is None else (torch.cat([s, mem[0]]), torch.cat([y, mem[1]]), torch.cat([w, mem[2]]), torch.cat([site, mem[3]]))
        keep = (site[:, None] != all_site[None, :]).float()[..., None] if self.cross else None
        if keep is None:
            keep = torch.ones(b, all_s.shape[0], 1, device=s.device)
            keep[:, :b] -= torch.eye(b, device=s.device)[..., None]
        wp = (y * w)[:, None] * ((1 - all_y) * all_w)[None] * keep
        wn = ((1 - y) * w)[:, None] * (all_y * all_w)[None] * keep
        d = all_s[None] - s[:, None]
        num = (wp * F.softplus(d) + wn * F.softplus(-d)).sum((0, 1))
        den = (wp + wn).sum((0, 1))
        has = den > 1e-6
        return (num[has] / den[has]).mean() if has.any() else s.sum() * 0

    def push(self, s, y, w, site):
        new = (s.detach(), y.detach(), w.detach(), site.detach())
        self.mem = new if self.mem is None else tuple(torch.cat([n, m])[: self.size] for n, m in zip(new, self.mem))


def pw_setup(cfg, tr, outs, dev):
    """pw targets / weights (n, n_out) from pw_csv on the main outputs (others weight 0) and the site index per training study."""
    t = pd.read_csv(EXP / cfg["pw_csv"]).set_index("StudyInstanceUID").reindex(tr)
    y, w = np.zeros((len(tr), len(outs)), np.float32), np.zeros((len(tr), len(outs)), np.float32)
    for j, o in enumerate(outs):
        if o in MAIN_OUT:
            v = t[MAIN_OUT[o]].to_numpy(np.float64)
            y[:, j], w[:, j] = np.nan_to_num(v), ~np.isnan(v)
    sv = pd.read_csv(EXP / cfg["pw_site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
    return torch.tensor(y, device=dev), torch.tensor(w, device=dev), torch.tensor(pd.factorize(sv)[0], device=dev), sv.nunique()


def site_setup(cfg, tr, outs, dev):
    """main/train.py's 'vague' site term on the main outputs: (site names, site index per training study, cell group (n, n_out), offsets)."""
    assert cfg["site_mode"] == "vague", cfg["site_mode"]
    s = pd.read_csv(EXP / cfg["site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
    s = s.where(s.map(s.value_counts()) >= cfg["site_min"], "other")
    names = sorted(s.unique())
    st = status_table("v1").loc[tr]
    grp = np.zeros((len(tr), len(outs)), np.int64)
    for j, o in enumerate(outs):
        if o in MAIN_OUT:
            c = st[KEY_OF[MAIN_OUT[o]]]
            grp[:, j] = np.where(c.isin(EXPLICIT), 0, np.where(c.isin(("not_mentioned", "general_normal")), 1, 2))
    off = torch.nn.Parameter(torch.zeros(len(names), 3, len(outs), device=dev))
    return names, torch.tensor(s.map({n: i for i, n in enumerate(names)}).to_numpy(), device=dev), torch.tensor(grp, device=dev), off


def make_bone_targets(studies, cfg, outs):
    """(n, n_out) targets and loss weights; unknown location cells get weight 0."""
    yt, _ = make_targets(studies, cfg)
    yt1 = make_targets(studies, cfg | {"tgt_csv": ""})[0] if {"frac1", "cont1"} & set(outs) else None
    yl, yf = loc_targets(studies, cfg["loc_hn"]), frac_loc_targets(studies)
    if cfg["loc_img_csv"]:                          # 10-20 W2-2: image-read region oedema (Opus p2) replaces the text location targets where set
        t = pd.read_csv(HERE.parents[3] / cfg["loc_img_csv"]).set_index("StudyInstanceUID").reindex(studies)
        for o, (region, _) in LOC.items():
            if region in t:
                v = t[region].to_numpy(np.float32)
                yl[~np.isnan(v), list(LOC).index(o)] = v[~np.isnan(v)]
    y, w = np.zeros((len(studies), len(outs)), np.float32), np.zeros((len(studies), len(outs)), np.float32)
    tw = None
    if cfg.get("tgt_gw"):                                 # Master 10-20 (read-set selection bias): GROUP-PRESERVING up-weight (port of mainp tgt_gw) - per v1
        assert cfg.get("tgt_csv") and not cfg.get("tgt_w"), "tgt_gw needs tgt_csv; exclusive with tgt_w"   # status group of the finding,
        tw = pd.read_csv(cfg["tgt_csv"]).set_index("StudyInstanceUID").reindex(studies)                    # read cells r x the unread multiplier,
        st_gw = status_table("v1").reindex(studies)                                                         # the group TOTAL unchanged
        twm = pd.read_csv(HERE.parents[3] / cfg["tgt_wmask_csv"]).set_index("StudyInstanceUID").reindex(studies) if cfg.get("tgt_wmask_csv") else tw   # Master: only these cells up-weight (v3: the p2-random reads; p3 cells weight 1)
    if cfg.get("tgt_w"):                                  # 10-20 USER 'bigger weight for loss on new labels' (bone analog of mainp tgt_w): the frac / cont
        assert cfg.get("tgt_csv"), "tgt_w needs tgt_csv"   # cells that tgt_csv replaced (image reads) get loss weight tgt_w instead of w_main
        tw = pd.read_csv(cfg["tgt_csv"]).set_index("StudyInstanceUID").reindex(studies)
    for j, o in enumerate(outs):
        if o in LOC or o in FLOC:
            v = yl[:, list(LOC).index(o)] if o in LOC else yf[:, list(FLOC).index(o)]
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_loc"]
        elif o in ("fnc", "cnf"):                         # de-coupling outputs from the main targets (same label set as frac / cont)
            yf_, yc_ = yt[:, FINDINGS.index("Fracture")], yt[:, FINDINGS.index("Contusion")]
            y[:, j], w[:, j] = (yf_ * (1 - yc_)) if o == "fnc" else (yc_ * (1 - yf_)), cfg["w_dec"]
        elif o in ("frac1", "cont1"):
            y[:, j], w[:, j] = yt1[:, FINDINGS.index(OUT_COL[o][:-3])], cfg["w_main"]
        elif o in ("img_frac", "img_cont"):              # 10-20 P2: image-read targets (CSV StudyInstanceUID, Fracture, Contusion; NaN = unread, masked)
            assert cfg.get("img_bone_csv"), "img_frac / img_cont need img_bone_csv"
            v = pd.read_csv(HERE.parents[3] / cfg["img_bone_csv"]).set_index("StudyInstanceUID")[OUT_COL[o][4:]].reindex(studies).to_numpy(np.float32)
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_img"]
            if cfg["img_w_csv"]:                          # W4-3: per-cell loss weight of the image targets (e.g. Opus certainty high 1.0 / moderate 0.6 / low 0.3)
                cw = pd.read_csv(HERE.parents[3] / cfg["img_w_csv"]).set_index("StudyInstanceUID")[OUT_COL[o][4:]].reindex(studies).to_numpy(np.float32)
                assert not np.isnan(cw[~np.isnan(v)]).any(), "img_w_csv lacks a weight for a labelled cell"
                w[:, j] *= np.nan_to_num(cw)
        else:
            y[:, j], w[:, j] = yt[:, FINDINGS.index(OUT_COL[o])], cfg["w_main"]
            if tw is not None and OUT_COL[o] in tw.columns and cfg.get("tgt_w"):
                w[tw[OUT_COL[o]].notna().to_numpy(), j] = cfg["tgt_w"]
            if tw is not None and OUT_COL[o] in tw.columns and cfg.get("tgt_gw"):
                rd, S = twm[OUT_COL[o]].notna().to_numpy() & tw[OUT_COL[o]].notna().to_numpy(), st_gw[KEY_OF[OUT_COL[o]]].to_numpy()
                for g in (("meets",), ("unsettled", "hedged", "conflict", "below"), ("excluded",), ("not_mentioned", "general_normal"), ("denied", "structure_normal")):
                    m = np.isin(S, g)
                    mr, mu = m & rd, m & ~rd
                    if mr.any():
                        f = w[m, j].sum() / (cfg["tgt_gw"] * w[mr, j].sum() + w[mu, j].sum())
                        w[mr, j] *= cfg["tgt_gw"] * f
                        w[mu, j] *= f
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
        xin = torch.cat([b["x"], b["x2"]], 1) if cfg.get("pair") else b["x"]
        x = gpu_prep(xin, cfg["px"], "bilinear", None, *model.norm_stats(), mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            out.append(torch.sigmoid(model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"], b.get("pm")).float()).cpu())
    return torch.cat(out).numpy()


def as_frame(p, idx, outs):
    d = pd.DataFrame(np.nan, index=pd.Index(idx, name="StudyInstanceUID"), columns=FINDINGS)
    for j, o in enumerate(outs):
        d[OUT_COL[o]] = p[:, j]          # extra columns (Fracture_v1, location / de-coupling heads) are appended
    return d


def fc_metrics(pv, pg):
    m = panel(pv, pg)
    return {k: m[k] for k in ("Frac_text", "Frac_hard", "Frac_gold", "Cont_text", "Cont_hard", "Cont_gold") if k in m}


def load_bags(cfg):
    """finalize(build_bags(...)), cached per bag config in runs/_bags/."""
    import bone_data
    bone_data.CACHE = cfg.get("cache") or "fov160_px400_q95"   # U57: bag pixel coordinates live in this cache's grid -> part of the key
    key = {k: cfg[k] for k in ("bags", "crop_scale", "trim", "series", "planes", "loc")} | {"v": 2} | ({"cache": bone_data.CACHE} if bone_data.CACHE != "fov160_px400_q95" else {}) | ({"pm": cfg["pat_margin"]} if cfg["pat_margin"] != 8.0 else {}) \
        | ({"pair": True} if cfg.get("pair") else {})                     # v: bag geometry version (bump when bone_data crop rules change)
    f = RUNS / "_bags" / (hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10] + ".pkl")
    if f.exists():
        return pickle.load(open(f, "rb"))
    bags = finalize(build_bags(cfg["bags"].split(","), cfg["crop_scale"], cfg["trim"], cfg["series"], cfg["planes"], cfg["loc"], cfg["pat_margin"],
                               cfg.get("pair", False)))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(bags, open(f, "wb"))
    return bags



def grow_load(model, sd):
    """10-20 S1 (init_full): load a deployed checkpoint into a model that may have MORE outputs (new image heads appended): equal shapes are
    copied; a tensor that grew along dim 0 gets the checkpoint's rows first and keeps its own init for the new rows; anything else fails.
    Returns (n copied, n grown, keys the checkpoint lacks)."""
    own = model.state_dict()
    out, nc, ng = dict(own), 0, 0
    for k, v in sd.items():
        assert k in own, f"init_full: unexpected checkpoint key {k}"
        if own[k].shape == v.shape:
            out[k], nc = v, nc + 1
        else:
            assert own[k].dim() == v.dim() and own[k].shape[1:] == v.shape[1:] and own[k].shape[0] > v.shape[0], (k, tuple(own[k].shape), tuple(v.shape))
            t = own[k].clone()
            t[:v.shape[0]] = v
            out[k], ng = t, ng + 1
    model.load_state_dict(out)
    return nc, ng, [k for k in own if k not in sd]


def opus_msg(pv, pairs, exp, rel="data/ext_p2_main.csv"):
    """10-20 live curves (USER via Master: read held-out new-label agreement DURING training): AUC x1e3 vs the Opus p2 reads (p >= .5) on this
    fold's labelled validation studies, of each image / ext / aux head next to the main head of the same column ('' when no labels)."""
    from sklearn.metrics import roc_auc_score
    f = Path(exp) / rel                                # 10-21: the run's own opus_csv (default = the wave-1 reads, as before)
    if not f.exists():
        return ""
    op = pd.read_csv(f).set_index("StudyInstanceUID")
    p = pv.set_index("StudyInstanceUID") if "StudyInstanceUID" in pv.columns else pv
    out = []
    for heads, col in pairs:
        y = op[col].reindex(p.index).dropna() if col in op else pd.Series(dtype=float)
        yb = (y >= 0.5).astype(int)
        hs = [h for h in heads if h in p.columns]
        if hs and 0 < yb.sum() < len(yb):
            out.append(f"{col}: " + " ".join(f"{h} {roc_auc_score(yb, p.loc[y.index, h].to_numpy(float)) * 1e3:.1f}" for h in hs) + f" (n {len(yb)}, +{int(yb.sum())})")
    return (" | OPUS held-out " + "; ".join(out)) if out else ""

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
    if cfg["train_ids"]:                                  # 10-20 sanity (USER: overfitting to new labels): train only on these studies (CSV StudyInstanceUID)
        keep = set(pd.read_csv(HERE.parent / cfg["train_ids"]).StudyInstanceUID)
        tr = [s for s in tr if s in keep]
    if cfg["val_ids"]:                                    # evaluate on these studies instead of the fold's val set (in-sample checks)
        va = [s for s in pd.read_csv(HERE.parent / cfg["val_ids"]).StudyInstanceUID if s in bags]
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
    pwm = site = None
    if cfg["pw_weight"]:                                  # 'pw' (overhaul item 4): trusted cross-site pairwise loss on the main outputs
        pw_y, pw_w, pw_site, n_sites = pw_setup(cfg, tr, outs, dev)
        pwm = PairMemory(cfg["pw_queue"], cfg["pw_cross"])
        say(f"pw: RankNet weight {cfg['pw_weight']} over {'CROSS-SITE' if cfg['pw_cross'] else 'all'} pairs of trusted cells "
            f"({int(pw_w.sum())} cells on {[o for o in outs if o in MAIN_OUT]} in {len(tr)} training studies, {n_sites} sites), queue {cfg['pw_queue']}")
    if cfg["site_mode"]:
        site = site_setup(cfg, tr, outs, dev)
        say(f"site term '{cfg['site_mode']}': {len(site[0])} site groups (>= {cfg['site_min']} training studies, rest 'other')")
    ds_tr = BoneDataset(tr, bags, bag_names, y, w, cfg["k"], True, cfg["px"], cfg["seed"], len(outs), pair=cfg["pair"])
    boost = None
    if cfg["samp_ids"]:                                   # 10-20 S3 (USER hparam sweep): oversample these studies (experiment-relative CSV), n draws / epoch
        sd_ = pd.read_csv(HERE.parent / cfg["samp_ids"]).set_index("StudyInstanceUID")
        mult = sd_["mult"] if "mult" in sd_ else pd.Series(float(cfg["samp_mult"]), index=sd_.index)   # W4-2: an optional per-study 'mult' column
        boost = torch.tensor(mult.reindex(tr).fillna(1.0).to_numpy(np.float64))
        assert (boost > 1).any() and (boost >= 1).all(), (cfg["samp_mult"], int((boost > 1).sum()))
        say(f"samp_ids: {int((boost > 1).sum())} / {len(tr)} training studies boosted (mult {sorted(set(boost[boost > 1].tolist()))}; share {float(boost[boost > 1].sum() / boost.sum()):.2f} of draws)")
    opus_ids = set(pd.read_csv(EXP / cfg["opus_csv"]).StudyInstanceUID) if ({"img_frac", "img_cont"} & set(outs) or cfg["mon100"]) else set()   # the read studies (v2 arms: data/v2/...)
    # 10-20 USER ('either don't sample studies with zero gradient from them, or use old labels'): image targets are DENSE from W4 on (Opus where
    # read, old labels elsewhere), so 'labelled' below = has an Opus read (the frozen 559 file), never 'w > 0'
    if cfg["lab_per_batch"]:                              # W4-8 (USER 10-20: 'why do we even have batches without labelled studies?'): k Opus-read per batch
        assert not cfg["samp_ids"] and 0 < cfg["lab_per_batch"] < cfg["bs"] and opus_ids, "lab_per_batch: image heads, exclusive with samp_ids, 0 < k < bs"
        lab_ = torch.tensor([s_ in opus_ids for s_ in tr])
        pos_ = None
        if cfg["lab_pos"]:                                # U100: kpos of the k from the Opus POSITIVES (Fracture or Contusion p >= .5 in opus_csv)
            assert 0 < cfg["lab_pos"] < cfg["lab_per_batch"], "0 < lab_pos < lab_per_batch"
            o_ = pd.read_csv(EXP / cfg["opus_csv"]).set_index("StudyInstanceUID")[["Fracture", "Contusion"]]
            pids = set(o_.index[(o_ >= 0.5).any(axis=1)])
            pos_ = torch.tensor([s_ in pids for s_ in tr])
            assert (lab_ & pos_).any() and (lab_ & ~pos_).any()
            say(f"lab_pos: {cfg['lab_pos']} of the {cfg['lab_per_batch']} from {int((lab_ & pos_).sum())} Opus positives, the rest from "
                f"{int((lab_ & ~pos_).sum())} Opus negatives")
        sampler = EpochSampler(len(tr), cfg["seed"], lab=lab_, k=cfg["lab_per_batch"], bs=cfg["bs"], pos=pos_, kpos=cfg["lab_pos"])
        say(f"lab_per_batch: {cfg['lab_per_batch']} of {cfg['bs']} per batch from {int(lab_.sum())} labelled / {len(tr)} training studies "
            f"(labelled share {cfg['lab_per_batch'] / cfg['bs']:.2f} of draws, ~{cfg['lab_per_batch'] * (len(tr) // cfg['bs']) / int(lab_.sum()):.1f} passes / epoch)")
    else:
        sampler = EpochSampler(len(tr), cfg["seed"], boost)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(BoneDataset(ids, bags, bag_names, None, None, cfg["k_eval"], False, cfg["px"], n_out=len(outs), pair=cfg["pair"]),
                                batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    ji = [j for j, o in enumerate(outs) if o in ("img_frac", "img_cont")]    # 10-20 W4 monitoring + losses on the image heads (only runs with them)
    jm = [j for j, o in enumerate(outs) if o in ("frac", "cont")]
    if ji or cfg["mon100"]:                               # train100: a fixed labelled TRAINING sample (the overfit gap next to the held-out read)
        lab = [s for s in tr if s in opus_ids]
        tr100 = sorted(np.random.default_rng(0).choice(lab, min(100, len(lab)), replace=False).tolist())
        dl_tr100 = mk(tr100)
    assert cfg["w_img_end"] < 0 or (ji and cfg["w_img"] > 0), "w_img_end needs image heads and w_img > 0"
    assert not cfg["rank_img"] or ji, "rank_img needs image heads"
    rkm = None
    if cfg["rank_img"]:                                   # W4-5: RankNet over Opus-labelled image cells (in batch + a FIFO of recent studies; soft targets weight the pairs)
        rk_y = torch.tensor(y, device=dev)
        rk_w = torch.zeros_like(rk_y)
        rk_w[:, ji] = torch.tensor(np.array([s_ in opus_ids for s_ in tr], np.float32), device=dev)[:, None] * torch.tensor((w[:, ji] > 0).astype(np.float32), device=dev)
        rk_site = torch.zeros(len(tr), dtype=torch.long, device=dev)
        rkm = PairMemory(cfg["pw_queue"], False)
        say(f"rank_img: RankNet weight {cfg['rank_img']} on {[outs[j] for j in ji]} ({int(rk_w.sum())} Opus-read cells), queue {cfg['pw_queue']}")
    out_fs = np.array([o in cfg["fs_outs"].split(",") for o in outs])
    model = BoneNet(cfg["backbone"], out_bags, len(bag_names), cfg["hidden"], cfg["drop"], bagsum=cfg["bagsum"], out_fs=out_fs, bb_pool=cfg["bb_pool"], drop_path=cfg["drop_path"],
                    series_heads=cfg["series_heads"], pair=cfg["pair"]).to(dev).to(memory_format=torch.channels_last)
    if cfg["init"]:                      # main-pipeline encoder trained on the same fold split (no leakage), e.g. refs/m_prob05/fold{fold}/model_ep19.pt
        sd = torch.load(EXP / cfg["init"].format(fold=fold), map_location="cpu")
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    if cfg["compile"]:                                    # backbone only; state_dict keys unchanged (the module is wrapped in place)
        model.backbone.compile()
    with torch.no_grad():
        p = np.clip((y * w).sum(0) / np.maximum(w.sum(0), 1e-6), 0.01, 0.99)
        model.cls_b.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        if cfg["sd_prior"] and cfg["series_heads"] == "sep":   # port fix (C60): own group heads start at the class prior, as mainp SD1b
            model.sd_b.copy_(model.cls_b[None].expand_as(model.sd_b))
    if cfg["init_full"]:                                 # 10-20 S1: fine-tune from a deployed checkpoint (after the prior init: old outputs
        nc, ng, miss = grow_load(model, torch.load(EXP / cfg["init_full"].format(fold=fold), map_location="cpu"))   # get the deployed weights, new heads keep the prior)
        print(f"init_full {cfg['init_full']}: {nc} tensors copied, {ng} grown (new outputs), {len(miss)} keys new: {miss[:6]}", flush=True)
    if cfg["eval_ckpt"]:                                  # 10-20 sanity: inference only - a saved checkpoint on va -> eval.parquet, no training
        model.load_state_dict(torch.load(HERE.parent / cfg["eval_ckpt"], map_location="cpu"))
        as_frame(predict(model, dl_va, cfg, dev, amp), va, outs).to_parquet(out_dir / "eval.parquet")
        say(f"eval_ckpt {cfg['eval_ckpt']}: {len(va)} studies -> eval.parquet")
        return
    if cfg["series_heads"] and (cfg["sd_outs"] or cfg["sd_fs"]):
        m = np.ones((6, len(outs)), bool)
        if cfg["sd_outs"]:
            m[:, [o not in cfg["sd_outs"].split(",") for o in outs]] = False
        if cfg["sd_fs"]:                                  # groups = plane x FS (SAG / COR / AX x nonFS, FS): keep the FS groups 1, 3, 5
            m[[0, 2, 4]] = False
        assert m.any(), (cfg["sd_outs"], cfg["sd_fs"])
        model.sd_mask, model.sd_masked = torch.tensor(m, device=dev), True
    head_ids = {id(q) for q in model.head_params()}
    if cfg["llrd"]:                                       # layer-wise lr decay (timm groups; biases / norms get no weight decay)
        bb_groups = [g | {"lr": cfg["lr"] * g.pop("lr_scale", 1.0)} for g in param_groups_layer_decay(model.backbone, cfg["wd"], layer_decay=cfg["llrd"])]
    else:
        bb_groups = [{"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": cfg["lr"]}]
    opt = torch.optim.AdamW(bb_groups + [{"params": model.head_params(), "lr": cfg["lr_head"]}]
                            + ([{"params": [site[3]], "lr": cfg["lr_head"], "weight_decay": 0.0}] if site is not None else []), weight_decay=cfg["wd"])
    spe = cfg["spe_ref"] or len(dl_tr)                   # spe_ref (10-21 port of oct14_all_data): schedule in the fold runs' epoch units (equal steps)
    total, warm = cfg["epochs"] * spe, int(cfg["warmup"] * spe)
    if cfg["spe_ref"]:
        print(f"steps/epoch {len(dl_tr)} (schedule unit {spe}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]
    mean, std = model.norm_stats()
    step, t0, rows, oof_all, gold_all = 0, time.time(), [], [], []
    for ep in range(cfg["epochs"]):
        model.train()
        sampler.epoch = ep
        tl, n, te = 0.0, 0, time.time()
        is_eval = (ep >= cfg["eval_from"] and (ep + 1) % cfg["eval_every"] == 0) or ep == cfg["epochs"] - 1
        wmul, lsum, gshare = None, torch.zeros(4, device=dev), float("nan")    # lsum: image num / den, main num / den (weighted BCE)
        if cfg["w_img_end"] >= 0:                         # W4-4: image-target weight ramps linearly from w_img (epoch 0) to w_img_end (last epoch)
            wmul = torch.ones(len(outs), device=dev)
            wmul[ji] = (cfg["w_img"] + (cfg["w_img_end"] - cfg["w_img"]) * ep / max(1, cfg["epochs"] - 1)) / cfg["w_img"]
        for b in dl_tr:
            b = to_dev(b, dev)
            if wmul is not None:
                b["w"] = b["w"] * wmul
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            xin = torch.cat([b["x"], b["x2"]], 1) if cfg["pair"] else b["x"]
            x = gpu_prep(xin, cfg["px"], "bilinear", cfg, mean, std, mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                logits = model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"], b.get("pm")).float()
            s_img = logits                                # 'pw' ranks the IMAGE logits (before the training-only site offsets)
            if site is not None:                          # training-only site offsets on the main outputs
                g = site[2][b["idx"]]
                logits = logits + site[3][site[1][b["idx"]][:, None].expand_as(g), g, torch.arange(g.shape[1], device=g.device).expand_as(g)] * (g > 0)
            bce = F.binary_cross_entropy_with_logits(logits, b["y"], reduction="none") * b["w"]
            loss = bce.sum() / b["w"].sum().clamp(min=1e-6)
            if ji:
                lsum += torch.stack([bce[:, ji].sum(), b["w"][:, ji].sum(), bce[:, jm].sum(), b["w"][:, jm].sum()]).detach()
            if site is not None:
                loss = loss + cfg["site_l2"] * site[3].pow(2).sum(-1).mean()
            if pwm is not None:
                i = b["idx"]
                loss = loss + cfg["pw_weight"] * pwm.loss(s_img, pw_y[i], pw_w[i], pw_site[i])
            if rkm is not None:
                loss = loss + cfg["rank_img"] * rkm.loss(s_img, rk_y[b["idx"]], rk_w[b["idx"]], rk_site[b["idx"]])
            if cfg["series_heads"]:           # U22: deep supervision of the present group heads + the stop-grad aggregator
                gl, pr, agg = model.sd_out
                wg = b["w"][:, None, :] * pr.float()
                lg_ = F.binary_cross_entropy_with_logits(gl.float(), b["y"][:, None].expand_as(gl), reduction="none")
                loss = loss + cfg["sd_weight"] * (lg_ * wg).sum() / wg.sum().clamp(min=1e-6)
                wa = b["w"] * pr.any(1).float() if model.sd_masked else b["w"]   # masked SD: the aggregator only where a group is kept
                loss = loss + cfg["sd_agg_weight"] * (F.binary_cross_entropy_with_logits(agg.float(), b["y"], reduction="none") * wa).sum() / wa.sum().clamp(min=1e-6)
            if ji and is_eval and math.isnan(gshare) and n >= len(dl_tr) // 2 and b["w"][:, ji].sum() > 0:   # W4 monitoring: the image targets' share of the trunk gradient norm (one labelled batch from mid-epoch on: cls_w starts at 0, so step 0 always reads 0)
                bb = [q for q in model.backbone.parameters() if q.requires_grad]
                gi_ = torch.autograd.grad(bce[:, ji].sum() / b["w"].sum().clamp(min=1e-6), bb, retain_graph=True, allow_unused=True)
                ga_ = torch.autograd.grad(loss, bb, retain_graph=True, allow_unused=True)
                nrm = lambda gs: torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))  # noqa: E731
                gshare = float(nrm(gi_) / nrm(ga_).clamp(min=1e-12))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if pwm is not None:
                pwm.push(s_img, pw_y[b["idx"]], pw_w[b["idx"]], pw_site[b["idx"]])
            if rkm is not None:
                rkm.push(s_img, rk_y[b["idx"]], rk_w[b["idx"]], rk_site[b["idx"]])
            tl += loss.detach()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:      # step >= total: the spe_ref schedule ends (all-data port)
                break
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        msg = f"ep {ep} loss {float(tl) / max(n, 1):.4f} time {time.time() - te:.0f}s ({(time.time() - te) / max(n, 1) * 1000:.0f} ms/step)"
        if ji:                                            # W4 monitoring: per-epoch weighted-BCE means of the image and the main targets (+ the ramp factor)
            L = lsum.cpu().numpy()
            msg += f" | Limg {L[0] / max(L[1], 1e-9):.4f} Lmain {L[2] / max(L[3], 1e-9):.4f}" + (f" wimg x{float(wmul[ji[0]]):.2f}" if wmul is not None else "")
        if (ep >= cfg["eval_from"] and (ep + 1) % cfg["eval_every"] == 0) or last:
            pv = as_frame(predict(model, dl_va, cfg, dev, amp), va, outs)
            pg = as_frame(predict(model, dl_gold, cfg, dev, amp), gold, outs)
            m = fc_metrics(pv, pg)
            rows.append({"epoch": ep, **m})
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            msg += " | " + " ".join(f"{k} {v:.4f}" for k, v in m.items())
            msg += opus_msg(pv, [(["img_Fracture", "Fracture"], "Fracture"), (["img_Contusion", "Contusion"], "Contusion")], EXP, cfg["opus_csv"])
            if ji or cfg["mon100"]:                       # W4 monitoring (10-21 mon100: also main-head runs): the same AUCs on the fixed labelled training sample
                pt = as_frame(predict(model, dl_tr100, cfg, dev, amp), tr100, outs)
                msg += opus_msg(pt, [(["img_Fracture", "Fracture"], "Fracture"), (["img_Contusion", "Contusion"], "Contusion")], EXP, cfg["opus_csv"]).replace(
                    "OPUS held-out", "OPUS train100") + (f" | gshare_img {gshare:.3f}" if ji else "")
            pd.concat(oof_all).to_parquet(out_dir / "oof_partial.parquet")     # 10-20 live curves: readable during training (final oof.parquet unchanged)
            pd.concat(gold_all).to_parquet(out_dir / "gold_partial.parquet")
            if ep >= cfg["save_from"] or last:
                torch.save(model.state_dict(), out_dir / f"model_ep{ep}.pt")
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
