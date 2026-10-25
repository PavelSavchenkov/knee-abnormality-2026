"""oct16_synovitis copy (2026-10-16) of experiments/v3/oct15_labels/syn/syn_train.py; added llrd / drop_path (timm layer-wise lr decay,
biases / norms without weight decay; ported from experiments/v2/oct14_backbone_sweep/bone/bone_train.py) for ConvNeXt-T crop experts;
holdout ('3' or '3+4'): folds left out of training entirely (honest nested PL teachers: fold g model trained outside {g, holdout});
boot_beta < 1: soft bootstrap (Reed et al. 2015) on the img_syn output from epoch boot_from: target = beta x read + (1 - beta) x the
model's own detached probability (online self-distillation against reader noise; no teacher, so OOF reads stay clean).
That file: copied 2026-10-16 from experiments/v2/oct14_all_data/syn/syn_train.py; added tgt_csv / tgt_findings / tgt_partial (text-target
replacement, oct15_labels Synovitis 0.961-like test).
Synovitis crop expert: one shared CNN over synovitis-region crops (syn_data.py), per-output attention pooling over the windows of
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
from timm.optim import param_groups_layer_decay  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels, status_table  # noqa: E402
from common.meta import SERIES_TYPES  # noqa: E402
from common.paths import FINDINGS, KEY_OF, KEYS  # noqa: E402
from data import gpu_prep  # noqa: E402
from labels import make_targets  # noqa: E402
from model import sincos  # noqa: E402
from syn_data import SynDataset, bag_names as make_bag_names, build_bags, finalize, seg_series_only  # noqa: E402

ATTRS = ["a_effgr", "a_effsyn", "a_hoffa", "a_thick", "a_debris"]       # Codex read attributes (scripts/make_attrs.py), aux outputs
OUT_COL = {"img_syn": "img_Synovitis", "img_eff": "img_Effusion", "img_eff_v": "img_Effusion_v", "syn": "Synovitis", "eff": "Effusion", "bakers": "Baker's",
           "acl": "ACL", "mcl": "MCL", "pf": "PF OA", "lm": "Lateral Meniscus", "mm": "Medial Meniscus", "loa": "Lateral OA",
           "moa": "Medial OA", "frac": "Fracture", "cont": "Contusion"} | {a: a for a in ATTRS}   # oct16_experts: any finding as a text output
OUT_COL |= {"img_cont": "img_Contusion", "img_frac": "img_Fracture", "img_pf": "img_PF OA"}   # 10-21 + img_pf (USER "synovitis-like recipes for classes which have new labels"); 10-20 SXC (USER 'new experts on new labels'): bone image-read outputs (ibone_csv)
READS = pd.read_csv(EXP / "data/synovitis_labels_codex.csv").set_index("StudyInstanceUID")
ATTR_T = pd.read_csv(EXP / "data/img_attrs.csv").set_index("StudyInstanceUID")
DEFAULTS = dict(backbone="efficientnet_b0.ra_in1k", px=224, k=8, k_eval=32, bs=8, epochs=20, lr=1e-3, lr_head=1e-3, wd=1e-2,
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, bags="pouch,hoffa,post", outs="img_syn,img_eff,syn,eff,bakers",
                w_img=1.0, w_ieff=0.5, w_txt=0.5, w_attr=0.5, bakers_post=True, img_csv="data/img_syn_prob.csv", labels="v1prob", nm=0.15,
                crop_scale=1.0, trim=0.1, fs_only=False, plane_bags=False, clamp=False, planes="", bagsum=False, init="", aug_rot=15, aug_scale=0.15, aug_shift=0.05, aug_erase=0.3, precision="bf16",
                eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0, save_opt=True, device="cuda", max_eval=0,
                mono=False, compile=False, offset="", offset_alpha=1.0, boost_w="", ft_mask="", ieff_csv="", ieff_v_csv="", clip_mode="global", diag_every=0, ctx="", ctx_mode="film", tgt_csv="", tgt_findings="", tgt_partial=False, llrd=0.0, drop_path=0.0, holdout="", boot_beta=1.0, boot_from=5, aug_bc=0.0, aug_gamma=0.0,
                out_bag="", below_neg="", w_below=2.0, aug_dil=0.0, seg_only=False, seg_mask=False, seg_drop="bc", seg_morph=0, bag_heads=False, w_bag=0.5, w_agg=0.5, grad_ckpt=False, attn_T=1.0, out_stride=0, img_size=0, trunk_by="", cache="fov160_px400_q95",
                xagg=False, w_xagg=0.5, nofs_only=False, knee_ctx="", w_kn=0.5, w_fin=0.5,
                ext_cache="", ext_loc="", ext_labels="", ext_type_map="", ext_map="", ext_types="", ext_off_l2=1e-3, ext_max=0,
                pw_weight=0.0, pw_csv="", pw_queue=512, pw_cross=True, pw_site_csv="data/site_train.csv", site_csv="", site_mode="", site_l2=1e-3,
                site_min=20, ibone_csv="", lab_csv="", lab_per_batch=0, w_out="")
# SXC (10-20, USER "try training new experts on new labels (optionally + old labels) from scratch / look again how we did it in synovitis"):
#           img_cont / img_frac = image-read outputs on ibone_csv's Contusion / Fracture columns (NaN = masked), weight w_img (the img_syn
#           pattern); lab_per_batch k > 0 with lab_csv: every batch holds exactly k studies with a Contusion / Fracture cell in lab_csv (the
#           read pool, cycled) so no batch lacks image gradient (repeats within an epoch get their own crop key, as bone W4-8).
# pw (10-19, overhaul re-triage item 8; port of main/train.py's 'pw' arm, as bone / oa): pw_weight > 0 adds a RankNet loss on the IMAGE
#           logits of the TEXT outputs that are competition findings (syn / eff / bakers / acl / ...; never the Codex image targets img_*) over
#           TRUSTED cells only (pw_csv: hard 1 / 0) and, with pw_cross, only between studies of DIFFERENT sites (in batch + a FIFO memory);
#           site_mode 'vague' = main's training-only site term on the same outputs (0 = explicit status, 1 = silent / general_normal, 2 = other;
#           sites with < site_min training studies share 'other'; L2 site_l2; never at prediction). Inert at the defaults; not with ext rows.
PW_EXPLICIT = ("meets", "denied", "structure_normal")
# ext_*     10-18 OAI (USER 'ok, i want to try external data'; port of oa_train's ext rows): EXTRA TRAINING studies from a second source.
#           ext_cache = its slice cache (absolute), ext_loc = its v2 localiser output dir, ext_labels = its label CSV (StudyInstanceUID + columns;
#           NaN = masked), ext_map = '+'-joined OUT:COLUMN pairs ('_' for spaces; unmapped outputs get weight 0 on external rows; per-output
#           weight = the competition one: w_img / w_ieff / w_attr / w_txt), ext_type_map / ext_types / ext_max as in oa_train. External rows
#           join the TRAINING side only; a training-only per-output SOURCE OFFSET (L2 ext_off_l2, dropped at prediction) on every head's
#           logits (main, bag heads, aggregators, knee context). Inert when ext_cache is ''.
# below_neg oct16_experts W5a: '+'-joined outputs whose v1 'below' cells become target 0 with weight w_txt x w_below
# bag_heads oct16_experts SD2 (user U15): per-(view, output) heads (each output's attention renormalised inside one bag, own readout),
#   deep-supervised with weight w_bag on the study target, and an aggregator over the bags present (per-output softmax weights) on the
#   DETACHED view logits, weight w_agg; the main output is unchanged; predict adds agg_<col> and v<j>_<col> (bag j = bag_names[j]) columns
# attn_T oct16_experts (ME5 passed at T 0.25 on the OA expert): attention softmax(s / T) in training and eval (sharper per-output attention)
# grad_ckpt oct16_experts: timm activation checkpointing in the backbone (same math; ME2 ConvNeXt-T x 9 bags OOMs at 31 GB without it)
#   CAVEAT (Master 10-17 05:52): exact gradients, but on BatchNorm backbones (effb0, NFNet) the recomputed forward runs in train mode and
#   updates BN running stats twice per step (effective momentum ~0.19 vs 0.1) -> eval-time BN stats differ slightly. ConvNeXt (LayerNorm)
#   is unaffected. For any BN model that might ship: no grad_ckpt, or check eval parity against a non-ckpt run.
# out_bag  oct16_experts: '+'-joined OUT:BAG1|BAG2 restricting an output's attention to those bags (e.g. acl:notchsc+mcl:mcl); others read all
# knee_ctx oct16_experts U33 B3 (user: stop-grad so the area crops' gradients are not shaped by the whole-knee view): the main outputs = the
#   AREA branch (out_bag routes must exclude the 'knee' bag); a KNEE branch (own attention + readout per output over the knee bag only,
#   deep-supervised, weight w_kn) and a per-output final aggregator over [area logit, knee logit], both DETACHED (weight w_fin; init =
#   the area logit). 'grad' (B3): the knee branch reads the live features (its loss still shapes the shared backbone); 'detach' (B3b):
#   knee windows run through the backbone in eval mode without grad and the knee head reads detached features (a probe: the backbone,
#   BN stats, norm and embeddings are trained by the area heads only). Gradient norms are clipped separately for the kn_ / fin_ params.
#   predict adds kn_<col> and fin_<col> columns


def encoder(backbone, pretrained, drop_path=0.0, out_stride=0, img_size=0):
    """timm encoder (num_classes=0); 'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (ImageNet mean / std; pair it
    with mono=True). out_stride / img_size (10-18, USER U58 fine-detail backbones): timm output_stride (dilated last stages, e.g. 16) and
    ViT input size (e.g. 384 for a patch-8 ViT); 0 = the backbone's default (unchanged)."""
    if backbone != "radimagenet_resnet50":
        kw = ({"drop_path_rate": drop_path} if drop_path else {}) | ({"output_stride": out_stride} if out_stride else {}) | ({"img_size": img_size} if img_size else {})
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
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
    def __init__(self, backbone, out_bags, n_bags, hidden=256, drop=0.2, pretrained=True, bagsum=False, drop_path=0.0, bag_heads=False, xagg=False,
                 knee_ctx="", knee_bag=-1, out_stride=0, img_size=0, trunk_by="", ctx_n=0, ctx_mode="film"):
        """out_bags: (n_out, n_bags) bool, which bags each output may read. bagsum: attention pooling inside each bag separately and
        one readout per (output, bag), summed (a bag's evidence is not diluted by the windows of the other bags)."""
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        self.backbone = encoder(backbone, pretrained, drop_path, out_stride, img_size)
        # trunk_by (10-18, USER U59 "separate backbones for different series / crops ... different weights"): a 2nd encoder of the same
        # arch with its OWN weights reads part of the windows: "knee" = the knee bag (b3g_split: knee vs area branches fully separate),
        # "fs" = fat-suppressed series windows (FS vs non-FS trunks); the rest of the model (embeddings, norm, attention, heads) is shared.
        self.trunk_by = trunk_by
        if trunk_by:
            assert trunk_by in ("knee", "fs") and (trunk_by != "knee" or knee_bag >= 0) and knee_ctx != "detach", (trunk_by, knee_bag, knee_ctx)
            self.backbone2 = encoder(backbone, pretrained, drop_path, out_stride, img_size)
            self.register_buffer("is_fs", torch.tensor([t.endswith("_FS") for t in SERIES_TYPES]), persistent=False)
            print(f"trunk_by {trunk_by}: 2nd {backbone} encoder (own weights) for the {'knee-bag' if trunk_by == 'knee' else 'FS-series'} windows", flush=True)
        if out_stride or img_size:                         # F2: the setting in effect, in the run's own log
            with torch.no_grad():
                fm = self.backbone.forward_features(torch.zeros(1, 3, img_size or 224, img_size or 224))
            print(f"encoder {backbone}: output_stride {out_stride or 'default'}, img_size {img_size or 'default'} -> feature map {tuple(fm.shape[1:])} at input {img_size or 224}", flush=True)
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
        self.bag_heads, self.sd_out, self.attn_T = bag_heads, None, 1.0
        self.diag, self.diag_ent = False, float("nan")   # 10-19 DINOv3 plateau diagnostic: normalised attention entropy of the last training forward
        # U24 B2: cross-class stop-grad aggregator: every class reads ALL class logits (DETACHED) through a linear map (init identity), so
        # cross-class information enters only after the stop-grad; the class heads (and their routes) stay as trained without it
        self.xagg, self.x_out = xagg, None
        if xagg:
            self.xagg_w = nn.Parameter(torch.eye(out_bags.shape[0]))
            self.xagg_b = nn.Parameter(torch.zeros(out_bags.shape[0]))
        self.knee_ctx, self.knee_bag, self.k_out = knee_ctx, knee_bag, None
        if knee_ctx:
            assert knee_ctx in ("grad", "detach") and knee_bag >= 0 and not out_bags[:, knee_bag].any(), "area routes must exclude the knee bag"
            self.kn_v = nn.Linear(d, hidden)
            self.kn_w = nn.Linear(hidden, n_out)
            self.kn_cls_w = nn.Parameter(torch.zeros(n_out, d))
            self.kn_cls_b = nn.Parameter(torch.zeros(n_out))
            self.fin_w = nn.Parameter(torch.tensor([[1.0, 0.0]]).repeat(n_out, 1))
            self.fin_b = nn.Parameter(torch.zeros(n_out))
        # U80 R-CTX (10-19): the package's study-level OOF logits (ctx_n of them, set per batch in self.ctx) condition the head; zero-init
        # so training starts from the plain model. film: crop features h * (1 + gamma) + beta; concat: + a linear map of ctx on every output.
        self.ctx_n, self.ctx_mode, self.ctx = ctx_n, ctx_mode, None
        if ctx_n:
            assert ctx_mode in ("film", "concat"), ctx_mode
            self.ctx_lin = nn.Linear(ctx_n, 2 * d if ctx_mode == "film" else n_out)
            nn.init.zeros_(self.ctx_lin.weight)
            nn.init.zeros_(self.ctx_lin.bias)
        if bag_heads:
            self.bh_w = nn.Parameter(torch.zeros(n_out, n_bags, d))
            self.bh_b = nn.Parameter(torch.zeros(n_out, n_bags))
            self.agg_w = nn.Parameter(torch.zeros(n_out, n_bags))
            self.agg_b = nn.Parameter(torch.zeros(n_out))

    def norm_stats(self):
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("backbone.", "backbone2."))]

    def forward(self, x, mask, typ, plane, pos, bag):
        b, k = mask.shape
        if self.knee_ctx == "detach" and self.training:          # B3b: knee windows = eval-mode, no-grad features (area BN stats unaffected)
            km = (bag == self.knee_bag).reshape(-1)
            fa = self.backbone(x[~km].contiguous(memory_format=torch.channels_last)).float()
            self.backbone.eval()
            with torch.no_grad():
                fk = self.backbone(x[km].contiguous(memory_format=torch.channels_last)).float()
            self.backbone.train()
            f = fa.new_zeros(b * k, fa.shape[1])
            f[~km], f[km] = fa, fk
            f = f.view(b, k, -1)
        elif self.trunk_by:                                      # U59: route each window to its trunk (empty subsets skipped)
            m2 = ((bag == self.knee_bag) if self.trunk_by == "knee" else self.is_fs[typ]).reshape(-1)
            parts = [(sel, enc) for sel, enc in ((~m2, self.backbone), (m2, self.backbone2)) if sel.any()]
            outs = [(sel, enc(x[sel].contiguous(memory_format=torch.channels_last)).float()) for sel, enc in parts]
            f = outs[0][1].new_zeros(b * k, outs[0][1].shape[1])
            for sel, o in outs:
                f[sel] = o
            f = f.view(b, k, -1)
        else:
            f = self.backbone(x).float().view(b, k, -1)
        h = self.norm(f + self.type_emb(typ) + self.plane_emb(plane) + self.bag_emb(bag.clamp(min=0)) + self.pos_proj(sincos(pos, scale=240.0)))
        if self.ctx_n and self.ctx_mode == "film":
            gm, bt = self.ctx_lin(self.ctx.to(h.dtype)).chunk(2, -1)
            h = h * (1 + gm[:, None]) + bt[:, None]
        hd = self.drop(h)
        allowed = self.out_bags[bag.clamp(min=0)] & mask[..., None]                 # (b, k, n_out)
        s = (self.att_w(torch.tanh(self.att_v(hd))) / self.attn_T).masked_fill(~allowed, float("-inf"))
        if self.bagsum:
            out = self.cls_b.expand(b, -1)
            for j in range(self.n_bags):
                a = torch.softmax(s.masked_fill((bag != j)[..., None], float("-inf")), 1).nan_to_num(0.0)
                out = out + (torch.einsum("bkn,bkd->bnd", a, hd) * self.cls_w[:, j]).sum(-1)
            return out
        a = torch.softmax(s, 1).nan_to_num(0.0)
        if self.diag and self.training:                 # H / log(#allowed windows): 1 = uniform attention, 0 = one window
            al = allowed.float().sum(1)
            ok = al > 1
            ent = -(a.float() * torch.log(a.float().clamp(min=1e-12))).sum(1)
            self.diag_ent = float((ent[ok] / torch.log(al[ok])).mean()) if ok.any() else float("nan")
        z = torch.einsum("bkn,bkd->bnd", a, hd)
        out = (z * self.cls_w).sum(-1) + self.cls_b
        if self.ctx_n and self.ctx_mode == "concat":
            out = out + self.ctx_lin(self.ctx.to(out.dtype))
        if self.bag_heads:
            vl, pres = [], []
            for j in range(self.n_bags):
                sj = s.masked_fill((bag != j)[..., None], float("-inf"))
                a = torch.softmax(sj, 1).nan_to_num(0.0)
                vl.append((torch.einsum("bkn,bkd->bnd", a, hd) * self.bh_w[:, j]).sum(-1) + self.bh_b[:, j])
                pres.append(torch.isfinite(sj).any(1))
            vl, pres = torch.stack(vl, -1), torch.stack(pres, -1)                       # (b, n_out, n_bags)
            wa = torch.softmax(self.agg_w.expand_as(vl).masked_fill(~pres, float("-inf")), -1).nan_to_num(0.0)
            self.sd_out = (vl, pres, (wa * vl.detach()).sum(-1) + self.agg_b)
        if self.xagg:
            self.x_out = out.detach() @ self.xagg_w.T + self.xagg_b
        if self.knee_ctx:
            hk = hd.detach() if self.knee_ctx == "detach" else hd
            sk = self.kn_w(torch.tanh(self.kn_v(hk))).masked_fill(~((bag == self.knee_bag) & mask)[..., None], float("-inf"))
            kl = (torch.einsum("bkn,bkd->bnd", torch.softmax(sk, 1).nan_to_num(0.0), hk) * self.kn_cls_w).sum(-1) + self.kn_cls_b
            self.k_out = (kl, self.fin_w[:, 0] * out.detach() + self.fin_w[:, 1] * kl.detach() + self.fin_b)
        return out


class EpochSampler(torch.utils.data.Sampler):
    """Shuffled indices offset by epoch * n, so persistent workers can derive the epoch.
    SXC (10-20): with lab (bool (n,)) and k, every consecutive group of bs draws (one batch) holds exactly k labelled studies (cycled through
    shuffled passes of the labelled pool) and bs - k unlabelled ones (the same), n // bs batches; the r-th draw of a study within an epoch
    gets the dataset key epoch * 16 + r, so repeats get fresh crops (copy of bone_train's W4-8 path)."""
    def __init__(self, n, seed, lab=None, k=0, bs=0):
        self.n, self.seed, self.epoch, self.lab, self.k, self.bs = n, seed, 0, lab, k, bs

    def cycle(self, pool, m, g):
        return torch.cat([pool[torch.randperm(len(pool), generator=g)] for _ in range(-(-m // len(pool)))])[:m]

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        if self.lab is None:
            return iter((torch.randperm(self.n, generator=g) + self.epoch * self.n).tolist())
        nb = self.n // self.bs
        bt = torch.cat([self.cycle(torch.where(self.lab)[0], nb * self.k, g).view(nb, self.k),
                        self.cycle(torch.where(~self.lab)[0], nb * (self.bs - self.k), g).view(nb, self.bs - self.k)], 1)
        idx = torch.gather(bt, 1, torch.argsort(torch.rand(nb, self.bs, generator=g), 1)).reshape(-1)
        cnt, rep = {}, []
        for v in idx.tolist():
            rep.append(min(cnt.get(v, 0), 15))
            cnt[v] = cnt.get(v, 0) + 1
        return iter((idx + (self.epoch * 16 + torch.tensor(rep)) * self.n).tolist())

    def __len__(self):
        return self.n if self.lab is None else self.n // self.bs * self.bs


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


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


def pw_site_setup(cfg, tr, outs, dev):
    """pw targets / weights (n, n_out) on the text outputs that are FINDINGS (others weight 0) + the pw site index; the 'vague' site term."""
    fo = {j: OUT_COL[o] for j, o in enumerate(outs) if not o.startswith("img_") and OUT_COL.get(o) in FINDINGS}
    pw = site = None
    if cfg["pw_weight"]:
        t = pd.read_csv(EXP / cfg["pw_csv"]).set_index("StudyInstanceUID").reindex(tr)
        y, w = np.zeros((len(tr), len(outs)), np.float32), np.zeros((len(tr), len(outs)), np.float32)
        for j, f in fo.items():
            v = t[f].to_numpy(np.float64)
            y[:, j], w[:, j] = np.nan_to_num(v), ~np.isnan(v)
        sv = pd.read_csv(EXP / cfg["pw_site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
        pw = (torch.tensor(y, device=dev), torch.tensor(w, device=dev), torch.tensor(pd.factorize(sv)[0], device=dev),
              PairMemory(cfg["pw_queue"], cfg["pw_cross"]), sv.nunique(), [outs[j] for j in fo])
    if cfg["site_mode"]:
        assert cfg["site_mode"] == "vague", cfg["site_mode"]
        s = pd.read_csv(EXP / cfg["site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
        s = s.where(s.map(s.value_counts()) >= cfg["site_min"], "other")
        names = sorted(s.unique())
        st = status_table("v1").loc[tr]
        grp = np.zeros((len(tr), len(outs)), np.int64)
        for j, f in fo.items():
            c = st[KEY_OF[f]]
            grp[:, j] = np.where(c.isin(PW_EXPLICIT), 0, np.where(c.isin(("not_mentioned", "general_normal")), 1, 2))
        site = (names, torch.tensor(s.map({n: i for i, n in enumerate(names)}).to_numpy(), device=dev), torch.tensor(grp, device=dev),
                torch.nn.Parameter(torch.zeros(len(names), 3, len(outs), device=dev)))
    return pw, site


def make_syn_targets(studies, cfg, outs):
    """(n, n_out) targets and loss weights; unread img_* cells get weight 0."""
    yt, _ = make_targets(studies, cfg)
    if cfg["tgt_csv"]:                                    # oct15_labels (user 10-16 ~00:30): text targets replaced where the CSV is set
        t = pd.read_csv(HERE.parents[3] / cfg["tgt_csv"]).set_index("StudyInstanceUID").reindex(studies)
        for f in [x.replace("_", " ") for x in cfg["tgt_findings"].split("+") if x] or FINDINGS:
            v = t[f].to_numpy(np.float32)
            ok = np.isfinite(v)
            assert cfg["tgt_partial"] or ok.mean() > 0.9, f"tgt_csv covers < 90% of training studies for {f}"
            yt[ok, FINDINGS.index(f)] = v[ok]
    img = pd.read_csv(EXP / cfg["img_csv"]).set_index("StudyInstanceUID").Synovitis.reindex(studies).to_numpy(np.float32)
    eff = READS.Effusion.reindex(studies).to_numpy(np.float32)
    if cfg["ieff_csv"]:                                   # C217 (10-19, USER re-opened VLM Effusion): img_eff target from a CSV (StudyInstanceUID, Effusion
        eff = pd.read_csv(HERE.parents[3] / cfg["ieff_csv"]).set_index("StudyInstanceUID")["Effusion"].reindex(studies).to_numpy(np.float32)   # prob; NaN = no target)
    y, w = np.zeros((len(studies), len(outs)), np.float32), np.zeros((len(studies), len(outs)), np.float32)
    for j, o in enumerate(outs):
        if o == "img_eff_v":                              # C217b (U93): a 2nd Effusion image head on its own target CSV (e.g. the VLM read), w_ieff
            v = pd.read_csv(HERE.parents[3] / cfg["ieff_v_csv"]).set_index("StudyInstanceUID")["Effusion"].reindex(studies).to_numpy(np.float32)
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_ieff"]
        elif o in ("img_syn", "img_eff"):
            v = img if o == "img_syn" else eff
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * (cfg["w_img"] if o == "img_syn" else cfg["w_ieff"])
        elif o in ("img_cont", "img_frac", "img_pf"):      # SXC / 10-21 syn-like ports: image-read target column of ibone_csv (NaN = no target)
            assert cfg["ibone_csv"], f"{o} needs ibone_csv"
            v = pd.read_csv(HERE.parents[3] / cfg["ibone_csv"]).set_index("StudyInstanceUID")[OUT_COL[o][4:]].reindex(studies).to_numpy(np.float32)
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_img"]
        elif o in ATTRS:
            v = ATTR_T[o].reindex(studies).to_numpy(np.float32)
            y[:, j], w[:, j] = np.nan_to_num(v), (~np.isnan(v)) * cfg["w_attr"]
        else:                                             # 10-21 w_out 'syn:1.0+...': per-output text weight (default w_txt)
            y[:, j], w[:, j] = yt[:, FINDINGS.index(OUT_COL[o])], {k: float(v) for k, v in (x.split(":") for x in cfg["w_out"].split("+") if x)}.get(o, cfg["w_txt"])
            if o in [x for x in cfg["below_neg"].split("+") if x]:      # oct16_experts W5a: v1 'below' cells = explicit negatives (definitional)
                below = text_status()[OUT_COL[o]].reindex(studies).eq("below").to_numpy()
                y[below, j], w[below, j] = 0.0, cfg["w_txt"] * cfg["w_below"]
            if cfg["ft_mask"]:                            # R-TAB twin (Brainstorm 10-19): only explicit v1 cells (meets / denied / structure_normal) train
                assert cfg["ft_mask"] == "x", cfg["ft_mask"]
                w[~status_table("v1")[KEY_OF[OUT_COL[o]]].reindex(studies).isin(PW_EXPLICIT).to_numpy(), j] = 0.0
    if cfg["ft_mask"]:
        print("ft_mask x: trained text cells per output", {o: round(float((w[:, j] > 0).mean()), 3) for j, o in enumerate(outs)}, flush=True)
    return y, w


@torch.no_grad()
def predict(model, loader, cfg, dev, amp):
    model.eval()
    out, sd, xa, kn = [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        x = gpu_prep(b["x"], cfg["px"], "bilinear", None, *model.norm_stats(), mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
        if model.ctx_n:                                   # U80 R-CTX: the loader's per-study context rows
            model.ctx = loader.ctx[b["idx"]]
        with torch.autocast(dev, dtype=amp, enabled=amp is not None):
            out.append(torch.sigmoid(model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"]).float()).cpu())
        if model.bag_heads:
            vl, pres, agg = model.sd_out
            sd.append(torch.cat([torch.sigmoid(agg.float())[..., None], torch.sigmoid(vl.float()).masked_fill(~pres, float("nan"))], -1).cpu())
        if model.xagg:
            xa.append(torch.sigmoid(model.x_out.float()).cpu())
        if model.knee_ctx:
            kn.append(torch.sigmoid(torch.stack(model.k_out, -1).float()).cpu())
    cat = lambda v: torch.cat(v).numpy() if v else None  # noqa: E731
    return torch.cat(out).numpy(), cat(sd), cat(xa), cat(kn)


def as_frame(p, idx, outs):
    p, sd, xa, kn = p
    d = pd.DataFrame(np.nan, index=pd.Index(idx, name="StudyInstanceUID"), columns=FINDINGS + ["img_Synovitis", "img_Effusion"])
    for j, o in enumerate(outs):          # attribute outputs become extra columns a_*
        d[OUT_COL[o]] = p[:, j]
        if sd is not None:                # SD2: agg_<col>, v<bag>_<col>
            d[f"agg_{OUT_COL[o]}"] = sd[:, j, 0]
            for v in range(1, sd.shape[2]):
                d[f"v{v - 1}_{OUT_COL[o]}"] = sd[:, j, v]
        if xa is not None:                # U24 B2: xagg_<col>
            d[f"xagg_{OUT_COL[o]}"] = xa[:, j]
        if kn is not None:                # U33 B3: kn_<col> (knee branch), fin_<col> (final aggregator)
            d[f"kn_{OUT_COL[o]}"], d[f"fin_{OUT_COL[o]}"] = kn[:, j, 0], kn[:, j, 1]
    return d


def syn_metrics(pv, pg):
    """Held-out reads AUC (p >= 0.5), clear-cut reads (>= 0.85 vs <= 0.1), gold-58 synovitis of the img_Synovitis column.
    oct16_experts: runs without an img_Synovitis output (ligament expert) report text (v1 meets vs denied / structure_normal) and
    gold AUC of every scored finding column instead."""
    if pv.img_Synovitis.isna().all():
        st, g = text_status(), gold_labels().reindex(pg.index)
        out = {}
        for c in [c for c in FINDINGS if pv[c].notna().any()]:
            s_ = st[c].reindex(pv.index)
            m = s_.isin(["meets", "denied", "structure_normal"])
            t = s_[m].eq("meets").astype(int)
            out[f"text_{c}"] = roc_auc_score(t, pv.loc[m[m].index, c]) if 0 < t.sum() < len(t) else float("nan")
            out[f"gold_{c}"] = roc_auc_score(g[c], pg[c]) if 0 < g[c].sum() < len(g) else float("nan")
        return out
    r = READS.Synovitis.reindex(pv.index)
    has = r.notna()
    y, s = (r[has] >= 0.5).astype(int), pv.img_Synovitis[has]
    conf = (r[has] >= 0.85) | (r[has] <= 0.1)
    g = gold_labels().reindex(pg.index)
    auc = lambda t, x: roc_auc_score(t, x) if 0 < t.sum() < len(t) else float("nan")  # noqa: E731  (all-data runs: val = gold-58)
    return {"r_auc": auc(y, s), "r_conf": auc(y[conf], s[conf]), "g_syn": auc(g.Synovitis, pg.img_Synovitis)}


def text_status():
    """v1 status per FINDINGS column (StudyInstanceUID index)."""
    st = status_table("v1")
    return st.rename(columns=dict(zip(KEYS, FINDINGS)))


def load_bags(cfg):
    """finalize(build_bags(...)), cached per bag config in runs/_bags/."""
    import syn_data
    syn_data.CACHE = cfg.get("cache") or "fov160_px400_q95"   # U57: bag pixel coordinates live in this cache's grid -> part of the key
    assert syn_data.CACHE == "fov160_px400_q95" or not (cfg["seg_mask"] or cfg["seg_only"]), "seg masks exist on the 400 grid only"
    key = {k: cfg[k] for k in ("bags", "crop_scale", "trim", "fs_only")} | {k: cfg[k] for k in ("plane_bags", "clamp", "planes", "nofs_only") if cfg[k]} \
        | ({"cache": syn_data.CACHE} if syn_data.CACHE != "fov160_px400_q95" else {})
    f = RUNS / "_bags" / (hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10] + ".pkl")
    if f.exists():
        bags = pickle.load(open(f, "rb"))
        return seg_series_only(bags) if cfg["seg_only"] else bags   # 10-18: here (not main) so launch.py applies it too
    bags = finalize(build_bags(cfg["bags"].split(","), cfg["crop_scale"], cfg["trim"], cfg["fs_only"], cfg["plane_bags"], cfg["clamp"], cfg["planes"],
                               cfg["nofs_only"]))
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(bags, open(f, "wb"))
    return seg_series_only(bags) if cfg["seg_only"] else bags


def run_fold(cfg, fold, out_dir, bags):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev, amp = cfg["device"], {"bf16": torch.bfloat16, "fp32": None}[cfg["precision"]]
    torch.backends.cudnn.benchmark = True
    bag_names, outs = make_bag_names(cfg["bags"].split(","), cfg["plane_bags"], cfg["planes"]), cfg["outs"].split(",")
    out_bags = np.ones((len(outs), len(bag_names)), bool)
    for kv in [x for x in cfg["out_bag"].split("+") if x]:        # oct16_experts: per-output bag restriction (de-coupled readers)
        o, bs = kv.split(":")
        out_bags[outs.index(o)] = [b.split(":")[0] in bs.split("|") for b in bag_names]
        assert out_bags[outs.index(o)].any(), kv
    if cfg["bakers_post"] and "bakers" in outs and "post" in cfg["bags"].split(","):
        out_bags[outs.index("bakers")] = [b.split(":")[0] == "post" for b in bag_names]
    folds = get_folds(5, 0)
    folds = folds[folds.index.isin(bags.keys())]
    hold = {int(x) for x in str(cfg["holdout"]).split("+") if x != ""}   # nested teachers: folds never trained on (e.g. "3")
    assert fold not in hold
    tr = folds.index[(folds.fold != fold) & ~folds.fold.isin(hold)].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in bags]
    if fold < 0:                                          # all-data run (fold -1): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    if cfg["max_eval"]:                                   # smoke tests
        va, gold = va[:cfg["max_eval"]], gold[:cfg["max_eval"]]
    y, w = make_syn_targets(tr, cfg, outs)
    ext = None
    if cfg["ext_cache"]:                                  # 10-18 OAI: external TRAINING rows (masked labels) + a training-only source offset
        assert not (cfg["seg_mask"] or cfg["seg_only"] or cfg["boot_beta"] < 1 or cfg["holdout"] or fold < 0), \
            "ext rows: seg / bootstrap / holdout / all-data not supported"
        tmap = dict(kv.split(":") for kv in cfg["ext_type_map"].split("+")) if cfg["ext_type_map"] else None
        eb = finalize(build_bags(cfg["bags"].split(","), cfg["crop_scale"], cfg["trim"], cfg["fs_only"], cfg["plane_bags"], cfg["clamp"],
                                 cfg["planes"], cfg["nofs_only"], loc_dir=HERE.parents[3] / cfg["ext_loc"], cache_name=cfg["ext_cache"],
                                 type_map=tmap, series_types=set(cfg["ext_types"].split("+")) if cfg["ext_types"] else None))
        assert not set(eb) & set(bags), "external study ids collide with competition ids"
        cols = {o: c.replace("_", " ") for o, c in (kv.split(":") for kv in cfg["ext_map"].split("+"))}
        assert cols and set(cols) <= set(outs), f"ext_map outputs {set(cols)} not in outs {outs}"
        lab = pd.read_csv(HERE.parents[3] / cfg["ext_labels"]).set_index("StudyInstanceUID")
        ext_ids = [s_ for s_ in lab.index if s_ in eb and lab.loc[s_, list(cols.values())].notna().any()]
        if cfg["ext_max"]:
            ext_ids = sorted(np.random.default_rng(cfg["seed"]).choice(ext_ids, min(cfg["ext_max"], len(ext_ids)), replace=False).tolist())
        ye, we = np.zeros((len(ext_ids), len(outs)), np.float32), np.zeros((len(ext_ids), len(outs)), np.float32)
        scale = {"img_syn": cfg["w_img"], "img_eff": cfg["w_ieff"]}
        for o, c in cols.items():
            v = lab.loc[ext_ids, c].to_numpy(np.float32)
            j = outs.index(o)
            ye[:, j], we[:, j] = np.nan_to_num(v), np.isfinite(v) * scale.get(o, cfg["w_attr"] if o in ATTRS else cfg["w_txt"])
        bags = {**bags, **{s_: eb[s_] for s_ in ext_ids}}
        tr, y, w = tr + ext_ids, np.concatenate([y, ye]), np.concatenate([w, we])
        ext = (cfg["ext_cache"], ext_ids)
        print(f"ext: {len(ext_ids)} external TRAINING studies ({len(eb)} with bags) from {cfg['ext_cache']} + {cfg['ext_labels']}; competition train "
              f"{len(tr) - len(ext_ids)}; labelled cells / positive rate per mapped output: "
              + ", ".join(f"{o}<-{c} {int((we[:, outs.index(o)] > 0).sum())} / {ye[we[:, outs.index(o)] > 0, outs.index(o)].mean():.3f}" for o, c in cols.items())
              + f"; type remap {tmap}; kept types {cfg['ext_types'] or 'all'}", flush=True)
    rb = None
    if cfg["offset"] or cfg["boost_w"]:                   # U80 correctors (CORRECTORS.md): the SHIPPED OOF logit per training study x output
        assert ext is None and fold >= 0, "offset / boost_w: competition fold runs only"
        assert not (cfg["offset"] and cfg["boost_w"] and cfg["offset"] != cfg["boost_w"]), "one base CSV"
        bdf = pd.read_csv(HERE.parents[3] / (cfg["offset"] or cfg["boost_w"])).set_index("StudyInstanceUID").reindex(tr)
        bcol = {"img_syn": "Synovitis", "img_eff": "Effusion"}   # image-read outputs correct the shipped column of their finding
        bv = bdf[[bcol.get(o, OUT_COL[o]) for o in outs]].to_numpy(np.float32)
        have = np.isfinite(bv)
        w = np.where(have, w, 0.0).astype(np.float32)    # no base -> that output does not train on the study
        if cfg["boost_w"]:                                # R-BOOST: weights ~ |y - p_base|, normalised to mean 1 over the trained cells
            r = np.abs(y - 1 / (1 + np.exp(-np.nan_to_num(bv))))
            r = r / np.maximum((r * (w > 0)).sum(0) / np.maximum((w > 0).sum(0), 1), 1e-6)
            w = (w * np.where(have, r, 0.0)).astype(np.float32)
        if cfg["offset"]:                                 # R-OFF: loss on sigmoid(alpha * base + g); predictions saved = g (the read adds the base)
            rb = torch.tensor(np.nan_to_num(bv) * cfg["offset_alpha"], dtype=torch.float32, device=dev)
        print(f"corrector base {cfg['offset'] or cfg['boost_w']}: offset {bool(cfg['offset'])} (alpha {cfg['offset_alpha']}), boost {bool(cfg['boost_w'])}; "
              f"training studies with a base per output: " + ", ".join(f"{o} {int(have[:, j].sum())} / {len(tr)}" for j, o in enumerate(outs)), flush=True)
    log = open(out_dir / "log.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    SynDataset.dil = cfg["aug_dil"]                       # S-D window dilation (train only)
    SynDataset.seg = cfg["seg_mask"]                      # U44 (10-18): bones + cartilage hidden by the nnU-Net masks (train + eval)
    SynDataset.seg_set = cfg["seg_drop"]                  # U47: "men" = hide the menisci instead (attribute name != the seg_drop method)
    SynDataset.seg_morph = cfg["seg_morph"]               # 10-18: dilate (> 0) / erode (< 0) the cropped mask by |k| pixels
    ds_tr = SynDataset(tr, bags, bag_names, y, w, cfg["k"], True, cfg["px"], cfg["seed"], len(outs), ext=ext)
    lab = None
    if cfg["lab_per_batch"]:                              # SXC: the read pool = studies with a Contusion / Fracture cell (of the img_* bone outputs) in lab_csv
        cols = [OUT_COL[o][4:] for o in outs if o in ("img_cont", "img_frac", "img_pf")]
        assert cfg["lab_csv"] and cols and ext is None, "lab_per_batch needs lab_csv + an img_cont / img_frac output (competition studies only)"
        lab = torch.tensor(pd.read_csv(HERE.parents[3] / cfg["lab_csv"]).set_index("StudyInstanceUID")[cols].reindex(tr).notna().any(axis=1).to_numpy())
        say(f"lab_per_batch {cfg['lab_per_batch']}: read pool {int(lab.sum())} / {len(tr)} training studies ({cols}); "
            f"draws per read study per epoch {len(tr) // cfg['bs'] * cfg['lab_per_batch'] / int(lab.sum()):.2f}")
    sampler = EpochSampler(len(tr), cfg["seed"], lab, cfg["lab_per_batch"], cfg["bs"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(SynDataset(ids, bags, bag_names, None, None, cfg["k_eval"], False, cfg["px"], n_out=len(outs)),
                                batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    model = SynNet(cfg["backbone"], out_bags, len(bag_names), cfg["hidden"], cfg["drop"], bagsum=cfg["bagsum"], drop_path=cfg["drop_path"],
                   bag_heads=cfg["bag_heads"], xagg=cfg["xagg"], knee_ctx=cfg["knee_ctx"],
                   knee_bag=bag_names.index("knee") if (cfg["knee_ctx"] or cfg["trunk_by"] == "knee") else -1, out_stride=cfg["out_stride"], img_size=cfg["img_size"],
                   trunk_by=cfg["trunk_by"], ctx_n=len(FINDINGS) if cfg["ctx"] else 0, ctx_mode=cfg["ctx_mode"]).to(dev).to(memory_format=torch.channels_last)
    if cfg["ctx"]:                                        # U80 R-CTX: base OOF logits per study (gold: the package's expected gold); NaN -> OOF mean
        assert ext is None and fold >= 0, "ctx: competition fold runs only"
        cb = pd.read_csv(HERE.parents[3] / cfg["ctx"]).set_index("StudyInstanceUID")[FINDINGS]
        cg = pd.read_csv(EXP / "refs/pkg961/expected_gold_f5_win.csv").set_index("StudyInstanceUID")[FINDINGS]
        cg = np.log(cg.clip(1e-6, 1 - 1e-6) / (1 - cg.clip(1e-6, 1 - 1e-6)))
        cm = cb.mean()
        ct = lambda ids, src: torch.tensor(src.reindex(ids).fillna(cm).to_numpy(np.float32), device=dev)  # noqa: E731
        ctx_tr, dl_va.ctx, dl_gold.ctx = ct(tr, cb), ct(va, cb), ct(gold, cg)
        print(f"ctx {cfg['ctx']} ({cfg['ctx_mode']}): train studies with a full base row {int(cb.reindex(tr).notna().all(1).sum())} / {len(tr)}, val "
              f"{int(cb.reindex(va).notna().all(1).sum())} / {len(va)}; gold = expected gold", flush=True)
    assert not (cfg["trunk_by"] and cfg["llrd"]), "trunk_by + llrd: backbone2 has no llrd groups"
    if cfg["init"]:                      # main-pipeline encoder trained on the same fold split (no leakage), e.g. refs/m_prob05/fold{fold}/model_ep19.pt
        sd = torch.load(EXP / cfg["init"].format(fold=fold), map_location="cpu")
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
        if cfg["trunk_by"]:
            model.backbone2.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    model.attn_T = cfg["attn_T"]
    model.diag = bool(cfg["diag_every"])
    if cfg["grad_ckpt"]:
        model.backbone.set_grad_checkpointing(True)
        if cfg["trunk_by"]:
            model.backbone2.set_grad_checkpointing(True)
    if cfg["compile"]:                                    # backbone only; state_dict keys unchanged (the module is wrapped in place)
        model.backbone.compile()
        if cfg["trunk_by"]:
            model.backbone2.compile()
    with torch.no_grad():
        p = np.clip((y * w).sum(0) / np.maximum(w.sum(0), 1e-6), 0.01, 0.99)
        model.cls_b.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        if cfg["bag_heads"]:
            model.bh_b.copy_(model.cls_b[:, None].expand_as(model.bh_b))
        if cfg["knee_ctx"]:
            model.kn_cls_b.copy_(model.cls_b)
        if rb is not None:                                # R-OFF: g starts at 0 (zero head)
            assert not (cfg["bag_heads"] or cfg["knee_ctx"] or cfg["xagg"]), "offset: plain attention head only"
            model.cls_w.zero_()
            model.cls_b.zero_()
    head_ids = {id(q) for q in model.head_params()}
    if cfg["llrd"]:                                       # layer-wise lr decay (timm groups; biases / norms get no weight decay)
        bb_groups = [g | {"lr": cfg["lr"] * g.pop("lr_scale", 1.0)} for g in param_groups_layer_decay(model.backbone, cfg["wd"], layer_decay=cfg["llrd"])]
    else:
        bb_groups = [{"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": cfg["lr"]}]
    ext_off = is_ext = None
    if ext is not None:                                   # training-only source offset (dropped at prediction)
        ext_off = torch.nn.Parameter(torch.zeros(len(outs), device=dev))
        is_ext = torch.tensor([s_ in set(ext[1]) for s_ in tr], dtype=torch.float32, device=dev)
    assert not ((cfg["pw_weight"] or cfg["site_mode"]) and ext is not None), "pw / site term: competition rows only (no ext rows)"
    pw, site = pw_site_setup(cfg, tr, outs, dev)
    if pw is not None:
        print(f"pw: RankNet weight {cfg['pw_weight']} over {'CROSS-SITE' if cfg['pw_cross'] else 'all'} pairs of trusted cells "
              f"({int(pw[1].sum())} cells on {pw[5]} in {len(tr)} training studies, {pw[4]} sites), queue {cfg['pw_queue']}", flush=True)
    if site is not None:
        print(f"site term '{cfg['site_mode']}': {len(site[0])} site groups (>= {cfg['site_min']} training studies, rest 'other')", flush=True)
    opt = torch.optim.AdamW(bb_groups + [{"params": model.head_params(), "lr": cfg["lr_head"]}]
                            + ([{"params": [ext_off], "lr": cfg["lr_head"], "weight_decay": 0.0}] if ext_off is not None else [])
                            + ([{"params": [site[3]], "lr": cfg["lr_head"], "weight_decay": 0.0}] if site is not None else []), weight_decay=cfg["wd"])
    spe = cfg["spe_ref"] or len(dl_tr)                   # spe_ref: schedule in the fold runs' epoch units (equal steps)
    total, warm = cfg["epochs"] * spe, int(cfg["warmup"] * spe)
    print(f"steps/epoch {len(dl_tr)} (schedule unit {spe}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]
    mean, std = model.norm_stats()
    step, t0, rows, oof_all, gold_all = 0, time.time(), [], [], []
    diag = []                                             # diag_every: per-step (grad norm bb, head, loss, attn entropy) of the current window
    for ep in range(cfg["epochs"]):
        model.train()
        sampler.epoch = ep
        tl, n, te = 0.0, 0, time.time()
        ext_l, ext_n = 0.0, 0.0
        for b in dl_tr:
            b = to_dev(b, dev)
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x = gpu_prep(b["x"], cfg["px"], "bilinear", cfg, mean, std, mono=cfg["mono"]).contiguous(memory_format=torch.channels_last)
            if cfg["ctx"]:
                model.ctx = ctx_tr[b["idx"]]
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                logits = model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"]).float()
            yb = b["y"]
            if cfg["boot_beta"] < 1 and ep >= cfg["boot_from"] and "img_syn" in outs:   # soft bootstrap on the image target only
                j = outs.index("img_syn")
                yb = yb.clone()
                yb[:, j] = cfg["boot_beta"] * yb[:, j] + (1 - cfg["boot_beta"]) * torch.sigmoid(logits[:, j].detach())
            eo = 0.0 if ext_off is None else is_ext[b["idx"]][:, None] * ext_off   # 10-18 OAI: source offset on external rows (0 on ours)
            so = 0.0
            if site is not None:                          # training-only site offsets on the text finding outputs (main BCE only)
                g = site[2][b["idx"]]
                so = site[3][site[1][b["idx"]][:, None].expand_as(g), g, torch.arange(g.shape[1], device=g.device).expand_as(g)] * (g > 0)
            ro = 0.0 if rb is None else rb[b["idx"]]          # U80 R-OFF: the base enters the loss only
            lm_ = F.binary_cross_entropy_with_logits(logits + eo + so + ro, yb, reduction="none") * b["w"]
            loss = lm_.sum() / b["w"].sum().clamp(min=1e-6)
            if site is not None:
                loss = loss + cfg["site_l2"] * site[3].pow(2).sum(-1).mean()
            if pw is not None:                            # 'pw': rank the IMAGE logits (no offsets)
                loss = loss + cfg["pw_weight"] * pw[3].loss(logits, pw[0][b["idx"]], pw[1][b["idx"]], pw[2][b["idx"]])
            if ext_off is not None:
                loss = loss + cfg["ext_off_l2"] * ext_off.pow(2).sum()
                xe = is_ext[b["idx"]] > 0                   # OAI share of the main loss (log only)
                ext_l, ext_n = ext_l + lm_[xe].sum().detach(), ext_n + lm_.sum().detach()
            if cfg["bag_heads"]:
                vl, pres, agg = model.sd_out
                wv = b["w"][..., None] * pres
                ev = 0.0 if ext_off is None else eo[..., None]
                lv = (F.binary_cross_entropy_with_logits(vl.float() + ev, yb[..., None].expand_as(vl), reduction="none") * wv).sum() / wv.sum().clamp(min=1e-6)
                la = (F.binary_cross_entropy_with_logits(agg.float() + eo, yb, reduction="none") * b["w"]).sum() / b["w"].sum().clamp(min=1e-6)
                loss = loss + cfg["w_bag"] * lv + cfg["w_agg"] * la
            if cfg["xagg"]:
                loss = loss + cfg["w_xagg"] * (F.binary_cross_entropy_with_logits(model.x_out.float() + eo, yb, reduction="none") * b["w"]).sum() / b["w"].sum().clamp(min=1e-6)
            if cfg["knee_ctx"]:
                bce = lambda z: (F.binary_cross_entropy_with_logits(z.float() + eo, yb, reduction="none") * b["w"]).sum() / b["w"].sum().clamp(min=1e-6)  # noqa: E731
                loss = loss + cfg["w_kn"] * bce(model.k_out[0]) + cfg["w_fin"] * bce(model.k_out[1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            ctx = [q for n, q in model.named_parameters() if n.startswith(("kn_", "fin_"))]
            hid = {id(q) for q in model.head_params()}
            if cfg["diag_every"]:                         # 10-19 DINOv3 plateau diagnostic (Master H1: does the GLOBAL clip starve the head?)
                gn = lambda ps: float(torch.sqrt(sum(q.grad.float().pow(2).sum() for q in ps if q.grad is not None)))  # noqa: E731
                diag.append((gn([q for q in model.parameters() if id(q) not in hid]), gn([q for q in model.parameters() if id(q) in hid]),
                             float(loss), model.diag_ent))
                if len(diag) == cfg["diag_every"]:
                    d_ = np.array(diag)
                    print(f"diag step {step + 1}: pre-clip grad norm backbone {d_[:, 0].mean():.3f} (max {d_[:, 0].max():.3f}) head {d_[:, 1].mean():.3f}; "
                          f"loss {d_[:, 2].mean():.4f}; attn entropy (norm.) {np.nanmean(d_[:, 3]):.3f}; lr bb {opt.param_groups[0]['lr']:.2e} head "
                          f"{opt.param_groups[-1]['lr']:.2e}; clip {cfg['clip_mode']} {cfg['grad_clip']}", flush=True)
                    diag.clear()
            clip = [q for q in model.parameters() if all(q is not c for c in ctx)]
            if cfg["clip_mode"] == "bb":                  # clip the backbone only: the head's step is never scaled by backbone grads
                clip = [q for q in clip if id(q) not in hid]
            assert cfg["clip_mode"] in ("global", "bb", "none"), cfg["clip_mode"]
            if cfg["clip_mode"] != "none":
                torch.nn.utils.clip_grad_norm_(clip, cfg["grad_clip"])
            if ctx:
                torch.nn.utils.clip_grad_norm_(ctx, cfg["grad_clip"])
            opt.step()
            if pw is not None:
                pw[3].push(logits, pw[0][b["idx"]], pw[1][b["idx"]], pw[2][b["idx"]])
            tl += loss.detach()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:
                break
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        msg = f"ep {ep} loss {float(tl) / max(n, 1):.4f} time {time.time() - te:.0f}s ({(time.time() - te) / max(n, 1) * 1000:.0f} ms/step)"
        if ext_off is not None:                               # 10-18 OAI model-side log: source offsets per output + OAI share of the main loss
            msg += f" | ext_off {[round(v, 3) for v in ext_off.detach().cpu().tolist()]} OAI loss share {float(ext_l) / max(float(ext_n), 1e-9):.3f}"
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
