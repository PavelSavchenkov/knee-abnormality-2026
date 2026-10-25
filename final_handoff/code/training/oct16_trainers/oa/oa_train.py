"""oct16_experts copy of experiments/v3/oct16_lateral_oa/main/oa_train.py + llrd / drop_path (ported from oct14_backbone_sweep/oa).
OA expert: one shared CNN over compartment crops (oa_data.py), each output attends only to its own bag:
Medial OA + Medial Meniscus <- medial bag, Lateral OA + Lateral Meniscus <- lateral bag (coronal / axial crops mirrored), PF OA <-
pf bag. So an output cannot read the other compartments ("degenerative knee" shortcut) except through what its own crops show.

h_k = LayerNorm(backbone(crop_k) + type_emb + plane_emb + pos_enc); per output n: a_kn = softmax over its bag's windows of
w_n . tanh(V h_k); logit_n = c_n . sum_k a_kn h_k + b_n. split=True: Medial / Lateral OA share (w, c) with a per-side bias,
and so do the menisci (Split-LR: the lateral bag is mirrored, so one compartment head serves both sides).

Pseudo-labels (oct16_lateral_oa): pl = teacher CSV (StudyInstanceUID + OA probabilities, scripts/teacher.py).
  blend    the OA targets become (1 - a) y + a p_T with a = pl_a_exp on explicit v1 cells (meets / denied / structure_normal), pl_a_oth
           elsewhere; pl_a_pos (>= 0) overrides a on meets cells (17:55: teacher-low meets are real focal defects, out/disagree.txt)
  conf     confident-only: on non-explicit cells the blend applies only where p_T <= pl_lo or p_T >= pl_hi (others keep text);
           pl_cw = k up-weights non-explicit cells by 1 + k |2 p_T - 1| (out/calib.txt: confident predictions are right 96-100%)
  tie      pl_tie = eps: cells whose v1 status is in pl_tie_st ("not_mentioned", or "all") keep their level but are spread by the
           teacher's rank r within (finding, status) over the training studies: y + eps (2 r - 1)
  teacher  pl_w > 0: an extra loss term pl_w x L(student OA logits, p_T); on 3 extra OA outputs (same bags; saved as 'T:<finding>')
  term     when pl_head=True, else on the main OA outputs. L = pl_loss: bce (soft BCE), mse (on probabilities), lmse (on logits,
           Ba & Caruana), rank (soft pairwise: BCE of sigmoid(s_i - s_j) vs sigmoid((logit p_i - logit p_j) / pl_tau), batch pairs)

conda run -n rsna --no-capture-output python experiments/v3/oct16_lateral_oa/main/oa_train.py --name NAME --folds 1 --set key=value ...
Output: runs/NAME/foldF/{config.json, log.txt, oof.parquet, gold.parquet (12 FINDINGS columns, only the 5 outputs filled, + epoch),
metrics.json (OA panel per eval epoch), model_ep{E}.pt}"""
import argparse
import json
import math
import re
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels, status_table, text_labels  # noqa: E402
from common.meta import SERIES_TYPES  # noqa: E402
from common.paths import FINDINGS, KEY_OF  # noqa: E402
from data import gpu_prep  # noqa: E402
from labels import make_targets, oa_target_log  # noqa: E402
from model import sincos  # noqa: E402
from oa_data import BAGS, CompartmentDataset, build_bags, finalize  # noqa: E402
from oa_eval import panel  # noqa: E402

OUTS = ["Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus"]
OUT_BAG = [0, 1, 2, 0, 1]
BML = [("medial", ("medial_femoral_condyle", "medial_tibial_plateau")), ("lateral", ("lateral_femoral_condyle", "lateral_tibial_plateau")),
       ("pf", ("patella",))]                                 # auxiliary marrow-oedema outputs, one per bag (OA analysis A31)


def aux_columns(cfg):
    cols = pd.read_csv(HERE.parent / cfg["aux_csv"], nrows=1).columns[1:]
    cols = [c for c in cols if re.search(cfg["aux_cols"], c)] if cfg["aux_cols"] else list(cols)
    assert cols, "aux_cols selects no column"
    return cols


def aux_bag(col):
    """D2 routing of a fact column to its bag: trochlea / patella -> pf, then lateral -> lateral, medial -> medial."""
    for pat, bag in (("trochlea|patella", "pf"), ("lateral", "lateral"), ("medial", "medial")):
        if re.search(pat, col):
            return BAGS.index(bag)
    raise ValueError(f"aux column {col}: no bag")


def aux_bags(cfg):
    return [aux_bag(c) for c in aux_columns(cfg)] if cfg.get("aux_weight", 0) > 0 else []


def aux_targets(cfg, studies):
    """(n, A) fact targets in [0, 1] from aux_csv (NaN = unknown, masked in the loss)."""
    return pd.read_csv(HERE.parent / cfg["aux_csv"]).set_index("StudyInstanceUID")[aux_columns(cfg)].reindex(studies).to_numpy(np.float32)


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
                warmup=1.0, grad_clip=1.0, drop=0.2, hidden=256, split=False, men_weight=0.5, bml_weight=0.0, cache="fov160_px400_q95", labels="v1prob", nm=0.15,
                side_fix=True, crop_scale=1.0, aug_rot=15, aug_scale=0.15, aug_shift=0.05, aug_erase=0.3, precision="bf16",
                eval_every=2, eval_from=11, save_from=11, workers=12, eval_workers=6, seed=0, max_steps=0, spe_ref=0, aug_dil=0.0, aug_jit=0.0, device="cuda", max_eval=0,
                crop_norm=False, oa_grade=False, loc_version="v1", poe="", poe_scale=1.0, pair_weight=0.0, pair_bank=1024, iso_weight=1.0, mono=False, lse_r=0.0, trim=0.15, restrict=True, tf_axial=True, rel=False, pos_up=1.0, topk=0, strat=False, init="", hybrid=False, drop_cells="",
                pl="", pl_a_exp=0.0, pl_a_oth=0.0, pl_a_pos=-1.0, pl_lo=0.5, pl_hi=0.5, pl_cw=0.0, pl_tie=0.0, pl_tie_st="not_mentioned", pl_head=False, pl_w=0.0, pl_loss="bce", knee_head=False, trunk_by="", kn_weight=0.5,
                pl_tau=1.0, hold=-1, tgt_csv="", tgt_cols="", out_w="", keep_bags="", k_bags="", llrd=0.0, drop_path=0.0, init_full="", ft_mask="", aux_seg=0.0, seg_dir="data/seg_sag", seg_px=56, deep_sup=0.0, slice_soft=0.0, attn_T=1.0,
                seq_L=0, seq_shuffle=False, seq_lr=2e-4,
                series_heads="", sd_weight=0.5, sd_agg_weight=0.5, sd_prior=False, series="all",
                ext_cache="", ext_loc="", ext_labels="", ext_type_map="", ext_off_l2=1e-3, ext_max=0, ext_types="", ext_only=False, ext_w=1.0, ext_head=False, ext_bn=False,
                aug_bc=0.0, aug_gamma=0.0, aug_noise=0.0,
                pw_weight=0.0, pw_csv="", pw_queue=512, pw_cross=True, pw_site_csv="data/site_train.csv", site_csv="", site_mode="", site_l2=1e-3,
                site_min=20, aux_csv="", aux_weight=0.0, aux_cols="", train_ids="", val_ids="", eval_ckpt="")
# aux_csv   (10-19 D2, port of main/train.py): CSV (StudyInstanceUID + fact columns in [0, 1], NaN = unknown) -> extra outputs, each routed to
#           its bag (trochlea / patella -> pf, lateral -> lateral, medial -> medial), masked BCE with its own term aux_weight x mean over
#           the batch's studies x facts (main's form; the 5 main outputs' loss is unchanged); aux_cols = a regex on the column names.
# pw (10-19, overhaul re-triage item 5; port of main/train.py's 'pw' arm, as bone/bone_train.py): pw_weight > 0 adds a RankNet loss on the IMAGE
#           logits of the 5 main outputs over TRUSTED cells only (pw_csv: hard 1 / 0, empty = no pair) and, with pw_cross, only between studies
#           of DIFFERENT sites (in batch + a FIFO memory of pw_queue studies); site_mode 'vague' = main's training-only site term on the 5 outputs
#           (per (site, cell group) offsets added to the main BCE logits; 0 = explicit status, 1 = silent / general_normal, 2 = other; sites
#           with < site_min training studies share 'other'; L2 site_l2; never at prediction). Inert at the defaults; not with ext rows.
# oct16_lateral_oa options:
# tgt_csv   CSV (StudyInstanceUID + finding columns) of soft targets, path relative to the REPO ROOT (launch.py rewrites an experiment-relative
#           path; fixed 10-17 after P1b f0 failed twice), '{fold}' -> fold number;
#           replaces the targets of tgt_cols ('+'-joined OUTS names, '_' for spaces; '' = the 3 OA outputs) where the CSV is finite
#           (label-set arms; e.g. copies of experiments/v3/oct15_labels/data/targets/<set>_f{fold}.csv in data/targets/)
# out_w     per-output loss-weight multipliers, '+'-joined NAME:MULT (e.g. Lateral_OA:2): LOA-weighted arms
# keep_bags per-class experts (user 10-16 19:10): '+'-joined bag names ('lateral', 'pf', 'medial+lateral'); windows come only from those
#           bags (k / k_eval per kept bag); outputs whose own bag is not kept attend over the first kept bag (defined, never NaN) and
#           must get loss weight 0 via out_w (asserted): dead heads, only the kept classes train
# init_full oct16_experts (clean-label stage, port of oct12_pseudo main/train.py init + ft_mask): checkpoint path relative to this
#           experiment, '{fold}' filled; FULL state dict incl. heads, no prior init of cls_b
# ft_mask   'x' = only explicit v1 cells (meets / denied / structure_normal) of the 5 outputs keep their loss weight (others 0); '' = all
# k_bags    G20 (training only): per-bag window counts '+'-joined BAG:K (e.g. lateral:12+medial:6+pf:6) instead of k for every bag;
#           eval keeps k_eval per bag
# ext_*     10-18 OAI (USER 'ok, i want to try external data'; Master / Brainstorm data/external/oai/HANDOFF_experts.md (c)): EXTRA TRAINING studies
#           from a second source. ext_cache = its slice cache (absolute path), ext_loc = its localiser output dir (v2 format, rows of ext_cache),
#           ext_labels = its label CSV (StudyInstanceUID + FINDINGS; soft values; NaN = masked = no loss), all repo-root relative except the
#           cache; ext_type_map = '+'-joined FROM:TO series_type remaps (DESS: SAG_T1_FS:SAG_PD_FS+COR_T1_FS:COR_PD_FS+AXI_T1_FS:AXI_PD_FS);
#           ext_max = cap on the external studies (0 = all labelled). External rows join the TRAINING side of every fold only (never val /
#           gold / OOF); a training-only per-finding SOURCE OFFSET (a learned logit offset on external rows, L2 ext_off_l2, dropped at
#           prediction like the D3 site term) absorbs the source's prevalence / contrast shift. Inert when ext_cache is ''.
#           ext_types = '+'-joined ORIGINAL series types of the external source kept (oai_tse: SAG_PD_FS+COR_PD_NOFS = the IW TSE series,
#           no DESS / MPR); ext_only = train on the external rows ONLY (oai_teacher: never sees competition labels; no source offset);
#           ext_w = loss weight of every external cell (dose without dropping knees; Brainstorm lever 6).
#           ext_head = a per-SOURCE readout (Master 10-18, the stronger form of the source offset): external rows are scored by their own
#           final linear head (weights + bias, initialised as the main head, training-only, outside the model state dict) on the SHARED
#           attention-pooled trunk features; our rows keep the main head; no source offset (the head's bias absorbs prevalence). Prediction
#           = the main head only (checkpoints unchanged).
#           ext_bn = per-source BatchNorm statistics (DSBN-lite, Master 10-18): in training our studies and the external studies of a batch
#           go through the model in separate passes (BN batch statistics per source; shared weights / affine); the external pass runs with
#           BN momentum 0, so the running statistics used at test are ours only.
# aug_bc / aug_gamma / aug_noise  10-18: oa/data.py gpu_prep's intensity augmentations (brightness-contrast x(1 +- a), gamma exp(+-a), gaussian
#           sigma), exposed as keys (0 = off = every earlier run); Brainstorm OAI lever 7 (intensity robustness across sources).
EXPLICIT = ("meets", "denied", "structure_normal")
OA_KEY = {"Medial OA": "medial_oa", "Lateral OA": "lateral_oa", "PF OA": "pf_oa"}


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


def pw_site_setup(cfg, tr, dev):
    """pw targets / weights (n, 5) on OUTS from pw_csv + the pw site index; the 'vague' site term (names, site index, cell group, offsets)."""
    pw = site = None
    if cfg["pw_weight"]:
        t = pd.read_csv(HERE.parent / cfg["pw_csv"]).set_index("StudyInstanceUID").reindex(tr)[OUTS]
        sv = pd.read_csv(HERE.parent / cfg["pw_site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
        pw = (torch.tensor(t.fillna(0).to_numpy(np.float32), device=dev), torch.tensor(t.notna().to_numpy(np.float32), device=dev),
              torch.tensor(pd.factorize(sv)[0], device=dev), PairMemory(cfg["pw_queue"], cfg["pw_cross"]), sv.nunique())
    if cfg["site_mode"]:
        assert cfg["site_mode"] == "vague", cfg["site_mode"]
        s = pd.read_csv(HERE.parent / cfg["site_csv"]).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
        s = s.where(s.map(s.value_counts()) >= cfg["site_min"], "other")
        names = sorted(s.unique())
        st = status_table("v1").loc[tr, [KEY_OF[f] for f in OUTS]]
        grp = np.where(st.isin(EXPLICIT), 0, np.where(st.isin(("not_mentioned", "general_normal")), 1, 2))
        site = (names, torch.tensor(s.map({n: i for i, n in enumerate(names)}).to_numpy(), device=dev), torch.tensor(grp, device=dev),
                torch.nn.Parameter(torch.zeros(len(names), 3, len(OUTS), device=dev)))
    return pw, site


class SegDecoder(nn.Module):
    """U91 aux_seg (training only): FPN / UNet-lite decoder over the encoder's stride-8 / 16 / 32 stages (skips; top-down sum of 1x1 laterals,
    a 3x3 mix per level) -> logits of the 9 nnU-Net Knee foreground classes at stride 4 (56 x 56 for 224 px; cartilage is 2-3 mm). Its keys
    ('seg_dec.') never enter a saved checkpoint."""
    def __init__(self, chs, n_cls=9, d=64):
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, d, 1) for c in chs)                 # chs = (c8, c16, c32)
        self.mix = nn.ModuleList(nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True)) for _ in chs)
        self.out = nn.Conv2d(d, n_cls, 1)

    def forward(self, feats):                                                      # feats = [s8, s16, s32]
        z = self.mix[2](self.lat[2](feats[2]))
        for i in (1, 0):
            z = self.mix[i](self.lat[i](feats[i]) + F.interpolate(z, size=feats[i].shape[-2:], mode="bilinear", align_corners=False))
        return self.out(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))


class DeepSupHead(nn.Module):
    """U92 deep supervision (training only): per-window GAP of the stride-16 stage -> a small tanh-attention MIL over the study's valid windows ->
    the main head's outputs (the same targets / weights). Its keys ('ds_head.') never enter a saved checkpoint."""
    def __init__(self, c, n_out, d=128):
        super().__init__()
        self.proj, self.att, self.cls = nn.Linear(c, d), nn.Linear(d, 1), nn.Linear(d, n_out)

    def forward(self, f16, mask):                                                  # f16 (b*k, C, h, w), mask (b, k)
        h = torch.relu(self.proj(f16.mean((-2, -1)).view(*mask.shape, -1)))
        a = torch.softmax(self.att(torch.tanh(h)).squeeze(-1).masked_fill(~mask, float("-inf")), 1)
        return self.cls((a[..., None] * h).sum(1))


def encoder(backbone, pretrained, drop_path=0.0):
    """timm encoder (num_classes=0); 'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (weights/radimagenet, ImageNet
    mean / std; loader copied from experiments/v0/oct10_baseline_3rd/src/model.py; pair it with mono=True)."""
    if backbone != "radimagenet_resnet50":
        return timm.create_model(backbone, pretrained=pretrained, num_classes=0, **({"drop_path_rate": drop_path} if drop_path else {}))
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
                 hybrid=False, pl_head=False, keep_bags="", drop_path=0.0, series_heads="", knee_head=False, trunk_by="", aux_bags=(), seq_L=0):
        super().__init__()
        # oct16_experts U22 (SD1 port): per-series-group heads (plane x FS: 6 groups) on the shared attention scores, each restricted to its
        # group's windows inside every output's bag, deep-supervised; DETACHED into a per-output aggregator over the present groups.
        # "" off | "shared" (main head + per-group bias) | "sep" (own head per group). Inference (the main head) is unchanged.
        self.series_heads = series_heads
        assert not series_heads or (series_heads in ("shared", "sep") and not (lse_r or topk or hybrid)), series_heads
        self.attn_T = 1.0                 # oct16_experts U13: training-time attention temperature (set from cfg)
        self.lse_r, self.restrict, self.topk, self.hybrid = lse_r, restrict, topk, hybrid
        self.backbone = encoder(backbone, pretrained, drop_path)
        # trunk_by (10-18, USER U59 "separate backbones for different series / crops ... different weights", on the LOA expert): a 2nd encoder
        # of the same architecture (own weights) for the FS-series windows; every window goes through exactly one trunk (OANet.encode)
        assert trunk_by in ("", "fs"), trunk_by
        self.trunk_by = trunk_by
        if trunk_by:
            self.backbone2 = encoder(backbone, pretrained, drop_path)
            self.register_buffer("is_fs", torch.tensor([t.endswith("_FS") for t in SERIES_TYPES]), persistent=False)
            print(f"trunk_by {trunk_by}: 2nd {backbone} encoder (own weights) for the FS-series windows", flush=True)
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
        # F1 (10-19): a bi-GRU along each run of seq_L ordered slices (the sampler lays runs out contiguously), as a zero-init residual on
        # the window features: at init the model equals the plain expert on the same windows.
        self.seq_L = seq_L
        if seq_L:
            self.seq_rnn = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
            self.seq_out = nn.Linear(d, d)
            nn.init.zeros_(self.seq_out.weight)
            nn.init.zeros_(self.seq_out.bias)
        self.split = split
        out_bag = OUT_BAG + ([0, 1, 2] if bml else []) + ([0, 1, 2] if pl_head else []) + list(aux_bags)   # D2 fact heads: plain recipe only (asserted)
        if knee_head:                     # U47 oa_kh: kn_ heads of the 5 outputs on the knee bag (b3g mechanism: absorbs the cross-area context)
            out_bag = out_bag + [BAGS.index("knee")] * len(OUT_BAG)
        if keep_bags:                     # per-class experts: outputs of dropped bags attend over the first kept bag (dead heads)
            kept = [BAGS.index(x) for x in keep_bags.split("+")]
            out_bag = [b if b in kept else kept[0] for b in out_bag]
        assert not (split and pl_head)
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
        if series_heads:
            self.sd_b = nn.Parameter(torch.zeros(6, n_out))
            self.sd_w = nn.Parameter(torch.zeros(6, n_head, d)) if series_heads == "sep" else None
            self.sd_agg, self.sd_agg_b = nn.Parameter(torch.zeros(6, n_out)), nn.Parameter(torch.zeros(n_out))

    def norm_stats(self):
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("backbone.", "backbone2.", "seq_"))]

    def seq_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("seq_")]

    def enable_taps(self, seg=False, deep_sup=False, n_cls=9):
        """U91 / U92: hook the backbone's last stride-8 / 16 / 32 stages (timm feature_info) and add the training-only heads: seg_dec (aux_seg,
        FPN over 8 / 16 / 32) and / or ds_head (deep supervision on the stride-16 GAP). The hooks keep the stage outputs of the last TRAINING
        forward in self.seg_feats[8 / 16 / 32] (the encoder sees all b x k windows in order)."""
        assert not self.trunk_by, "aux_seg / deep_sup: one trunk"
        fi = {d["reduction"]: d for d in self.backbone.feature_info}
        self.seg_feats = {}
        for red in (8, 16, 32):
            mod = self.backbone.get_submodule(fi[red]["module"])
            mod.register_forward_hook(lambda m, i, o, red=red: self.seg_feats.__setitem__(red, o) if self.training else None)
        if seg:
            self.seg_dec = SegDecoder([fi[r]["num_chs"] for r in (8, 16, 32)], n_cls)
        if deep_sup:
            self.ds_head = DeepSupHead(fi[16]["num_chs"], len(self.cls_b))   # the main logits' width (outputs, not heads)

    def encode(self, x, typ):
        """Per-window features (n, d) for x (n, 3, H, W) with series-type ids typ (n,): one trunk, or (trunk_by) each window through its
        series' trunk (empty subsets skipped)."""
        if not self.trunk_by:
            return self.backbone(x).float()
        m2 = self.is_fs[typ.reshape(-1)]
        parts = [(sel, enc) for sel, enc in ((~m2, self.backbone), (m2, self.backbone2)) if sel.any()]
        f = None
        for sel, enc in parts:
            y = enc(x[sel]).float()
            f = y.new_zeros((len(x), y.shape[1])) if f is None else f
            f[sel] = y
        return f

    def forward(self, x, mask, typ, plane, pos, bag, rel=None):
        b, k = mask.shape
        f = self.encode(x, typ).view(b, k, -1)
        e = f + self.type_emb(typ) + self.plane_emb(plane) + self.pos_proj(sincos(pos, scale=240.0))
        h = self.norm(e + self.rel_proj(rel) if self.rel_proj is not None else e)
        if self.seq_L:                    # F1: runs of seq_L consecutive slots -> bi-GRU -> zero-init residual (padded slots masked)
            assert k % self.seq_L == 0, (k, self.seq_L)
            r, _ = self.seq_rnn(h.reshape(b * k // self.seq_L, self.seq_L, -1))
            h = h + self.seq_out(r).reshape(b, k, -1) * mask[..., None]
        hd = self.drop(h)
        allowed = ((bag[..., None] == self.out_bag) | (not self.restrict)) & mask[..., None]   # restrict=False: ablation, all bags
        if self.training and getattr(self, "ss_head", None) is not None:   # SLICE-SOFT (C220 D): the training-only per-window head
            self.ss_out, self.ss_allowed = self.ss_head(hd), allowed
        if self.topk:                     # user 10:40: rank windows per output, keep the mean logit of the top-k (window localisation)
            v = self.inst(hd, allowed, self.cls_w).topk(min(self.topk, k), dim=1).values
            fin = torch.isfinite(v)
            return v.where(fin, torch.zeros_like(v)).sum(1) / fin.sum(1).clamp(min=1) + self.cls_b
        if self.lse_r and not self.hybrid:  # I26: per-window logits pooled by log-mean-exp (a few strong windows can carry the study)
            return self.lme(self.inst(hd, allowed, self.cls_w), allowed) + self.cls_b
        s = self.att_w(torch.tanh(self.att_v(hd)))[..., self.head_of] / self.attn_T   # (b, k, n_out); attn_T < 1 = sharper (U13)
        s = s.masked_fill(~allowed, float("-inf"))
        a = torch.softmax(s, 1)
        z = torch.einsum("bkn,bkd->bnd", a, hd)
        if self.training and getattr(self, "keep_z", False):   # 10-18 ext_head: the pooled features for the per-source readout
            self.last_z = z
        att = (z * self.cls_w[self.head_of]).sum(-1)
        if self.series_heads and self.training:
            self.sd_out = self.series_forward(s, allowed, typ, hd)
        if self.hybrid:                   # attention head + lse head (own weights), averaged: C4's two members in one model
            return 0.5 * (att + self.lme(self.inst(hd, allowed, self.cls_w2), allowed)) + self.cls_b
        return att + self.cls_b

    def series_forward(self, s, allowed, typ, hd):
        """SD heads: s (b, k, n) masked attention scores, allowed (b, k, n), typ (b, k) series-type ids (plane block of 10, FS = odd).
        Returns group logits (b, 6, n), present (b, 6, n) (a group counts per output: bags differ), aggregator logits (b, n)."""
        grp = (typ // 10) * 2 + (typ % 2)
        logits, present = [], []
        for j in range(6):
            m = allowed & (grp == j)[..., None]
            a = torch.nan_to_num(torch.softmax(s.masked_fill(~m, float("-inf")), 1))   # groups without windows -> zero weights
            z = torch.einsum("bkn,bkd->bnd", a, hd)
            w, bias = (self.cls_w[self.head_of], self.cls_b + self.sd_b[j]) if self.series_heads == "shared" else (self.sd_w[j][self.head_of], self.sd_b[j])
            logits.append((z * w).sum(-1) + bias)
            present.append(m.any(1))
        gl, pr = torch.stack(logits, 1), torch.stack(present, 1)
        wa = self.sd_agg[None].expand(gl.shape[0], -1, -1).masked_fill(~pr, float("-inf"))
        agg = (torch.nan_to_num(torch.softmax(wa, 1)) * gl.detach()).sum(1) + self.sd_agg_b   # stop-grad: the aggregator cannot shape the heads
        return gl, pr, agg

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


def pl_targets(cfg, tr, y, w):
    """PL targets of the 3 OA outputs (module docstring): -> (y, w, teacher probabilities (n, 3)). y, w: make_targets arrays
    restricted to OUTS (+ bml columns), modified in place."""
    t = pd.read_csv(HERE.parent / cfg["pl"]).set_index("StudyInstanceUID").reindex(tr)[OUTS[:3]]
    assert not t.isna().any().any(), "teacher missing for training studies"
    pt = t.to_numpy(np.float32)
    st = status_table("v1").loc[tr]
    for j, f in enumerate(OUTS[:3]):
        a = np.where(st[OA_KEY[f]].isin(EXPLICIT), cfg["pl_a_exp"], cfg["pl_a_oth"]).astype(np.float32)
        if cfg["pl_a_pos"] >= 0:
            a[st[OA_KEY[f]].eq("meets").to_numpy()] = cfg["pl_a_pos"]
        oth = ~st[OA_KEY[f]].isin(EXPLICIT).to_numpy()
        a[oth & (pt[:, j] > cfg["pl_lo"]) & (pt[:, j] < cfg["pl_hi"])] = 0.0        # confident-only; default band (0.5, 0.5) is empty
        # (18:24 fix: the first defaults -1 / 2 made the band cover every cell, so a_oth was silently 0 in pl_s038 f3 = pl_neg03)
        if cfg["pl_cw"]:
            w[oth, j] *= 1 + cfg["pl_cw"] * np.abs(2 * pt[oth, j] - 1)
        y[:, j] = (1 - a) * y[:, j] + a * pt[:, j]
    if cfg["pl_tie"]:
        for j, f in enumerate(OUTS[:3]):
            s_ = st[OA_KEY[f]]
            m = np.ones(len(tr), bool) if cfg["pl_tie_st"] == "all" else s_.eq(cfg["pl_tie_st"]).to_numpy()
            r = pd.Series(pt[:, j]).groupby(s_.to_numpy()).rank(pct=True).to_numpy()
            y[m, j] = np.clip(y[m, j] + cfg["pl_tie"] * (2 * r[m] - 1), 0.005, 0.995)
    return y, w, pt


def teacher_loss(s, p, kind, tau=1.0):
    """Distillation term: student logits s (B, 3) vs teacher probabilities p (B, 3)."""
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(s, p)
    if kind == "mse":
        return F.mse_loss(torch.sigmoid(s), p)
    lp = torch.logit(p.clamp(1e-4, 1 - 1e-4))
    if kind == "lmse":
        return F.mse_loss(s, lp)
    if kind == "rank":
        i, j = torch.triu_indices(len(s), len(s), 1, device=s.device)
        return F.binary_cross_entropy_with_logits(s[i] - s[j], torch.sigmoid((lp[i] - lp[j]) / tau))
    raise ValueError(kind)


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


def as12(p, idx, pl_head=False, knee_head=False, aux_cols=()):
    d = pd.DataFrame(np.nan, index=idx, columns=FINDINGS)
    d[OUTS] = p[:, :len(OUTS)]
    for j, c in enumerate(aux_cols):                       # 10-20 P2: D2 aux heads are the last len(aux_cols) outputs (plain recipe, asserted)
        d[f"aux:{c}"] = p[:, p.shape[1] - len(aux_cols) + j]
    if knee_head:                                          # U47 oa_kh: knee-head outputs are the last five
        for j, f in enumerate(OUTS):
            d[f"kn_{f}"] = p[:, p.shape[1] - len(OUTS) + j]
    if pl_head:                                            # teacher-head outputs are the last three
        for j, f in enumerate(OUTS[:3]):
            d[f"T:{f}"] = p[:, p.shape[1] - 3 + j]
    return d



def opus_msg(pv, pairs, exp):
    """10-20 live curves (USER via Master: read held-out new-label agreement DURING training): AUC x1e3 vs the Opus p2 reads (p >= .5) on this
    fold's labelled validation studies, of each image / ext / aux head next to the main head of the same column ('' when no labels)."""
    from sklearn.metrics import roc_auc_score
    f = Path(exp) / "data/ext_p2_main.csv"
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
    CompartmentDataset.dil, CompartmentDataset.jit = cfg["aug_dil"], cfg["aug_jit"]   # Brainstorm 17 fallback (D1 / D2); train only
    CompartmentDataset.seq_L, CompartmentDataset.seq_shuffle = cfg["seq_L"], cfg["seq_shuffle"]   # F1 ordered runs (train + eval)
    CompartmentDataset.seg_dir = HERE.parent / cfg["seg_dir"] if cfg["aux_seg"] else None            # U91 label crops (training items only)
    CompartmentDataset.seg_px = cfg["seg_px"]
    folds = get_folds(5, 0)
    folds = folds[folds.index.isin(bags.keys())]
    tr = folds.index[(folds.fold != fold) & (folds.fold != cfg["hold"])].tolist()   # hold: nested teachers (3-fold training)
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in bags]
    if cfg["train_ids"]:                    # 10-20 sanity (USER: overfitting to new labels): train only on these studies (experiment-relative CSV)
        keep = set(pd.read_csv(HERE.parent / cfg["train_ids"]).StudyInstanceUID)
        tr = [s for s in tr if s in keep]
    if cfg["val_ids"]:                      # evaluate on these studies instead of the fold's val set (in-sample checks)
        va = [s for s in pd.read_csv(HERE.parent / cfg["val_ids"]).StudyInstanceUID if s in bags]
    if fold < 0:                                          # all-data run (fold -1; port of oct14_all_data 10-18): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    if cfg["max_eval"]:                                   # smoke tests
        va, gold = va[:cfg["max_eval"]], gold[:cfg["max_eval"]]
    y, w = make_targets(tr, cfg)
    cols = [FINDINGS.index(f) for f in OUTS]
    y, w = y[:, cols], w[:, cols]
    w[:, 3:] *= cfg["men_weight"]
    if cfg["ft_mask"]:                                     # clean-label stage: explicit v1 cells only
        assert cfg["ft_mask"] == "x", cfg["ft_mask"]
        stx = status_table("v1").loc[tr]
        keys = {"Medial OA": "medial_oa", "Lateral OA": "lateral_oa", "PF OA": "pf_oa", "Medial Meniscus": "medial_meniscus",
                "Lateral Meniscus": "lateral_meniscus"}
        for j, c in enumerate(OUTS):
            keep = stx[keys[c]].isin(EXPLICIT).to_numpy()
            w[~keep, j] = 0
            print(f"ft_mask x {c}: {int(keep.sum())} of {len(tr)} cells keep their weight", flush=True)
    if cfg["tgt_csv"]:
        t = pd.read_csv(HERE.parents[3] / cfg["tgt_csv"].format(fold=fold)).set_index("StudyInstanceUID").reindex(tr)   # repo-root relative (launch.py rewrites it; as syn_train)
        for c in ([x.replace("_", " ") for x in cfg["tgt_cols"].split("+")] if cfg["tgt_cols"] else OUTS[:3]):
            v = t[c].to_numpy(np.float32)
            ok = np.isfinite(v)
            j = OUTS.index(c)
            print(f"tgt_csv {c}: {ok.sum()} of {len(tr)} cells replaced, mean target {y[:, j].mean():.4f} -> "
                  f"{np.where(ok, v, y[:, j]).mean():.4f}", flush=True)
            y[:, j] = np.where(ok, v, y[:, j])
    if cfg["out_w"]:
        for kv in cfg["out_w"].split("+"):
            k, m = kv.split(":")
            w[:, OUTS.index(k.replace("_", " "))] *= float(m)
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
    if cfg["knee_head"]:                                   # U47 oa_kh: the knee heads learn the same 5 targets (deep supervision, weight kn_weight)
        y, w = np.concatenate([y, y[:, :len(OUTS)]], 1), np.concatenate([w, w[:, :len(OUTS)] * cfg["kn_weight"]], 1)
    pt = None
    if cfg["pl"]:                                          # pseudo-labels: teacher probabilities for the 3 OA outputs
        y, w, pt = pl_targets(cfg, tr, y, w)
        pt = torch.tensor(pt, device=dev)
    assert not cfg["pl_head"] or cfg["pl_w"] > 0, "pl_head needs pl_w > 0"
    ext, n_comp = None, len(tr)
    if cfg["ext_cache"]:                                   # 10-18 OAI: external TRAINING rows (masked labels) + a training-only source offset
        assert not (cfg["pl"] or cfg["poe"] or cfg["bml_weight"] or cfg["knee_head"] or cfg["pair_weight"] or cfg["pos_up"] != 1), \
            "ext rows: pl / poe / bml / knee_head / pair / pos_up not supported"
        tmap = dict(kv.split(":") for kv in cfg["ext_type_map"].split("+")) if cfg["ext_type_map"] else None
        eb = finalize(build_bags(cfg["crop_scale"], cfg["loc_version"], cfg["trim"], cfg["tf_axial"], cfg["series"],
                                 loc_dir=HERE.parents[3] / cfg["ext_loc"], cache_name=cfg["ext_cache"], type_map=tmap,
                                 series_types=set(cfg["ext_types"].split("+")) if cfg["ext_types"] else None))
        assert not set(eb) & set(bags), "external study ids collide with competition ids"
        lab = pd.read_csv(HERE.parents[3] / cfg["ext_labels"]).set_index("StudyInstanceUID")
        ext_ids = [s for s in lab.index if s in eb and lab.loc[s, OUTS].notna().any()]
        if cfg["ext_max"]:
            ext_ids = sorted(np.random.default_rng(cfg["seed"]).choice(ext_ids, min(cfg["ext_max"], len(ext_ids)), replace=False).tolist())
        ye = lab.loc[ext_ids, OUTS].to_numpy(np.float32)
        we = np.isfinite(ye).astype(np.float32) * cfg["ext_w"]
        we[:, 3:] *= cfg["men_weight"]
        if cfg["out_w"]:
            for kv in cfg["out_w"].split("+"):
                k_, m_ = kv.split(":")
                we[:, OUTS.index(k_.replace("_", " "))] *= float(m_)
        ye = np.nan_to_num(ye, nan=0.0)
        assert y.shape[1] == len(OUTS), "ext rows assume the 5 OA / meniscus outputs only"
        bags = {**bags, **{s: eb[s] for s in ext_ids}}
        tr = tr + ext_ids
        y, w = np.concatenate([y, ye]), np.concatenate([w, we])
        ext = (cfg["ext_cache"], ext_ids)
        print(f"ext: {len(ext_ids)} external TRAINING studies ({len(eb)} with bags) from {cfg['ext_cache']} + {cfg['ext_labels']}; "
              f"competition train {n_comp}; labelled cells / positive rate per output: "
              + ", ".join(f"{c} {int((we[:, j] > 0).sum())} / {ye[we[:, j] > 0, j].mean():.3f}" for j, c in enumerate(OUTS))
              + f"; type remap {tmap}; kept types {cfg['ext_types'] or 'all'}; ext_only {cfg['ext_only']}", flush=True)
        if cfg["ext_only"]:                                # oai_teacher: the external rows are the whole training set
            tr, y, w, n_comp = list(ext_ids), ye, we, 0
    ny = y.shape[1]                                        # the outputs of the main loss term (D2 fact heads come after them)
    if cfg["aux_weight"]:
        assert not (cfg["bml_weight"] or cfg["knee_head"] or cfg["pl_head"] or cfg["split"] or cfg["keep_bags"] or cfg["ext_cache"]), \
            "D2 aux heads: plain or series-heads recipe only"
        yx = aux_targets(cfg, tr)
        y, w = np.concatenate([y, np.nan_to_num(yx)], 1), np.concatenate([w, (~np.isnan(yx)).astype(np.float32)], 1)
        print(f"D2 aux: {yx.shape[1]} fact heads (bags {dict(zip(*np.unique(aux_bags(cfg), return_counts=True)))} of {BAGS}), "
              f"known cells {int((~np.isnan(yx)).sum())}, weight {cfg['aux_weight']}", flush=True)
    poe = torch.zeros(len(tr), ny, device=dev)          # residual training: frozen proxy logits added in the loss only
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
    if n_comp:
        say(oa_target_log(tr[:n_comp], y[:n_comp, :3]))                 # competition rows only (external ids have no v1 status)
    keep = cfg["keep_bags"].split("+") if cfg["keep_bags"] else None
    if keep:                                                 # dead heads (outputs of dropped bags) must not train
        dead = [j for j, b in enumerate(OUT_BAG) if BAGS[b] not in keep]
        assert all((w[:, j] == 0).all() for j in dead), f"keep_bags {keep}: set out_w 0 for {[OUTS[j] for j in dead]}"
        assert not cfg["bml_weight"] and not cfg["pl_head"] and not cfg["split"], "keep_bags with bml / pl_head / split not supported"
    k_tr = {b: int(n) for b, n in (x.split(":") for x in cfg["k_bags"].split("+"))} if cfg["k_bags"] else cfg["k"]
    if isinstance(k_tr, dict):
        assert set(k_tr) == set(keep or BAGS), f"k_bags must name every used bag: {k_tr}"
    ds_tr = CompartmentDataset(tr, bags, y, w, k_tr, True, cfg["px"], cfg["seed"], cfg["strat"], keep=keep, ext=ext)
    oa_rel = (y[:, :3] * (w[:, :3] > 0)).max(1) >= 0.45                  # pos_up: studies with meets / unsettled / conflict OA
    sampler = EpochSampler(len(tr), cfg["seed"], torch.tensor(np.where(oa_rel, cfg["pos_up"], 1.0)) if cfg["pos_up"] != 1 else None)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk = lambda ids: DataLoader(CompartmentDataset(ids, bags, None, None, cfg["k_eval"], False, cfg["px"], keep=keep), batch_size=max(1, cfg["bs"] // 2),
                                num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk(va), mk(gold)
    model = OANet(cfg["backbone"], cfg["hidden"], cfg["drop"], cfg["split"], bml=cfg["bml_weight"] > 0, lse_r=cfg["lse_r"],
                  restrict=cfg["restrict"], rel=cfg["rel"], topk=cfg["topk"], hybrid=cfg["hybrid"], pl_head=cfg["pl_head"],
                  keep_bags=cfg["keep_bags"], drop_path=cfg["drop_path"], series_heads=cfg["series_heads"],
                  knee_head=cfg["knee_head"], trunk_by=cfg["trunk_by"], aux_bags=aux_bags(cfg), seq_L=cfg["seq_L"]).to(dev).to(memory_format=torch.channels_last)
    model.attn_T = cfg["attn_T"]
    if cfg["aux_seg"] or cfg["deep_sup"]:   # U91 / U92: training-only heads (+ hooks); before init_full / the optimizer
        model.enable_taps(seg=bool(cfg["aux_seg"]), deep_sup=bool(cfg["deep_sup"]))
        model.to(dev)
    if cfg["slice_soft"]:                   # SLICE-SOFT (C220 D, COMPETITIONS_REVIEW item 2): per-window soft targets = the study label x the window's
        # target-cartilage presence (rel[..., 0] = localiser-v2 p of the bag's medial / lateral tibial or patellar cartilage on the centre slice,
        # every plane) on a training-only linear head over the window features; its keys ('ss_head.') never enter a saved checkpoint
        assert not (cfg["ext_cache"] or cfg["lse_r"] or cfg["topk"]), "slice_soft: plain attention recipe, competition data only (OAI: no)"
        model.ss_head = nn.Linear(model.norm.normalized_shape[0], len(model.cls_b)).to(dev)
    if cfg["init"]:                         # user 11:00: start from a main-pipeline encoder trained on the same fold split (no leakage)
        sd = torch.load(HERE.parent / cfg["init"].format(fold=fold), map_location="cpu")
        model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
        if cfg["trunk_by"]:
            model.backbone2.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    if cfg["init_full"]:                    # clean-label stage: the whole deployed model (heads included), no prior init
        sd = torch.load(HERE.parent / cfg["init_full"].format(fold=fold), map_location="cpu")
        res = model.load_state_dict(sd, strict=not (cfg["seq_L"] or cfg["aux_seg"] or cfg["deep_sup"] or cfg["slice_soft"]))
        if cfg["seq_L"] or cfg["aux_seg"] or cfg["deep_sup"] or cfg["slice_soft"]:   # F1 seq, U91 seg decoder, U92 ds head, ss head: the only keys allowed missing
            assert not res.unexpected_keys and all(k_.startswith(("seq_", "seg_dec.", "ds_head.", "ss_head.")) for k_ in res.missing_keys), res
    if cfg["eval_ckpt"]:                    # 10-20 sanity: inference only - a saved checkpoint on va -> eval.parquet, no training
        model.load_state_dict(torch.load(HERE.parent / cfg["eval_ckpt"], map_location="cpu"))
        axc = aux_columns(cfg) if cfg["aux_weight"] else ()
        as12(predict(model, dl_va, cfg, dev, amp), va, cfg["pl_head"], cfg["knee_head"], axc).to_parquet(out_dir / "eval.parquet")
        print(f"eval_ckpt {cfg['eval_ckpt']}: {len(va)} studies -> eval.parquet", flush=True)
        return
    with torch.no_grad():
        p = np.clip((y * w).sum(0) / np.maximum(w.sum(0), 1e-6), 0.01, 0.99)
        if cfg["pl_head"]:
            p = np.concatenate([p, pt.mean(0).cpu().numpy()])
        if not cfg["init_full"]:
            model.cls_b.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        if cfg["poe"]:                                          # the proxy carries the prior: residual outputs start at 0
            model.cls_b[[OUTS.index(c) for c in pd.read_csv(HERE.parent / cfg["poe"], nrows=1).columns[1:]]] = 0.0
        if cfg["sd_prior"] and cfg["series_heads"] == "sep":   # port fix (C60, Master 22:05): own group heads start at the class prior, as mainp SD1b
            model.sd_b.copy_(model.cls_b[None].expand_as(model.sd_b))
    head_ids = {id(q) for q in model.head_params()} | {id(q) for q in model.seq_params()}   # F1: seq params get their own group
    assert not (cfg["trunk_by"] and cfg["llrd"]), "trunk_by + llrd: backbone2 has no llrd groups"
    if cfg["llrd"]:                         # oct16_experts (port of oct14_backbone_sweep): timm layer-wise LR decay for LayerNorm encoders
        from timm.optim import param_groups_layer_decay
        enc_groups = [g | {"lr": cfg["lr"] * g.pop("lr_scale", 1.0)} for g in param_groups_layer_decay(model.backbone, cfg["wd"], layer_decay=cfg["llrd"])]
        assert sum(len(g["params"]) for g in enc_groups) == len(list(model.backbone.parameters()))
    else:
        enc_groups = [{"params": [q for q in model.parameters() if id(q) not in head_ids], "lr": cfg["lr"]}]
    ext_off = is_ext = ext_hw = ext_hb = None
    if ext is not None:
        is_ext = torch.tensor([s in set(ext[1]) for s in tr], dtype=torch.float32, device=dev)
    if ext is not None and not cfg["ext_only"] and not cfg["ext_head"]:   # training-only source offset (dropped at prediction)
        ext_off = torch.nn.Parameter(torch.zeros(ny, device=dev))
    bn_mods, bn_mom = [], []
    if ext is not None and cfg["ext_bn"]:                   # per-source BN batch statistics; running statistics = ours only
        assert not (cfg["ext_only"] or cfg["ext_head"] or cfg["series_heads"]), "ext_bn: one change at a time (no ext_only / ext_head / series heads)"
        bn_mods = [m_ for m_ in model.modules() if isinstance(m_, torch.nn.modules.batchnorm._BatchNorm)]
        bn_mom = [m_.momentum for m_ in bn_mods]
        assert bn_mods and all(mo_ is not None for mo_ in bn_mom), "ext_bn needs BatchNorm layers with a momentum"
    if ext is not None and cfg["ext_head"]:                 # per-source readout on the shared pooled features (training-only)
        assert not (cfg["ext_only"] or cfg["series_heads"] or cfg["hybrid"] or cfg["lse_r"] or cfg["topk"] or cfg["split"]), "ext_head: plain attention head only"
        model.keep_z = True
        ext_hw, ext_hb = torch.nn.Parameter(model.cls_w.detach().clone()), torch.nn.Parameter(model.cls_b.detach().clone())
    assert not ((cfg["pw_weight"] or cfg["site_mode"]) and ext is not None), "pw / site term: competition rows only (no ext rows)"
    pw, site = pw_site_setup(cfg, tr, dev)
    if pw is not None:
        say(f"pw: RankNet weight {cfg['pw_weight']} over {'CROSS-SITE' if cfg['pw_cross'] else 'all'} pairs of trusted cells "
            f"({int(pw[1].sum())} cells on {OUTS} in {len(tr)} training studies, {pw[4]} sites), queue {cfg['pw_queue']}")
    if site is not None:
        say(f"site term '{cfg['site_mode']}': {len(site[0])} site groups (>= {cfg['site_min']} training studies, rest 'other')")
    opt = torch.optim.AdamW(enc_groups + [{"params": model.head_params(), "lr": cfg["lr_head"]}]
                            + ([{"params": model.seq_params(), "lr": cfg["seq_lr"]}] if cfg["seq_L"] else [])
                            + ([{"params": [ext_off], "lr": cfg["lr_head"], "weight_decay": 0.0}] if ext_off is not None else [])
                            + ([{"params": [ext_hw, ext_hb], "lr": cfg["lr_head"]}] if ext_hw is not None else [])
                            + ([{"params": [site[3]], "lr": cfg["lr_head"], "weight_decay": 0.0}] if site is not None else []), weight_decay=cfg["wd"])
    spe = cfg["spe_ref"] or len(dl_tr)                   # spe_ref (port of oct14_all_data): schedule in the fold runs' epoch units (equal steps)
    total, warm = cfg["epochs"] * spe, int(cfg["warmup"] * spe)
    print(f"steps/epoch {len(dl_tr)} (schedule unit {spe}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]
    mean, std = model.norm_stats()
    step, t0, rows, oof_all, gold_all = 0, time.time(), [], [], []
    bank_s, bank_y = torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev)    # pair_loss bank (newest first)
    seg_log = {"l": 0.0, "n": 0, "win": 0}                 # U91 aux_seg per-epoch log
    for ep in range(cfg["epochs"]):
        model.train()
        sampler.epoch = ep
        tl, n, te = 0.0, 0, time.time()
        ext_l, ext_n = 0.0, 0.0
        ext_lc = all_lc = 0.0
        tr_log = []                                            # 10-18 OAI sanity log: (logits, y, w, is OAI) per training step
        for b in dl_tr:
            b = to_dev(b, dev)
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            oh = None
            if cfg["aux_seg"]:                            # U91: one-hot label maps (9 classes) get the SAME geometric augmentation
                sid = b["seg"].flatten(0, 1).long()
                seg_has = (sid != 255).flatten(1).any(1) & b["mask"].flatten()
                oh = F.one_hot(sid.masked_fill(sid == 255, 0), 10)[..., 1:].permute(0, 3, 1, 2).float()
            x = gpu_prep(crop_normalise(b["x"]) if cfg["crop_norm"] else b["x"], cfg["px"], "bilinear", cfg, mean, std,
                         mono=cfg["mono"], extra=oh)
            if oh is not None:
                x, oh = x
            x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                if bn_mods:                               # ext_bn: one pass per source present in the batch
                    xe_ = is_ext[b["idx"]] > 0
                    xv = x.view(xe_.shape[0], -1, *x.shape[1:])
                    logits = None
                    for sel, ext_pass in ((~xe_, False), (xe_, True)):
                        if not sel.any():
                            continue
                        if ext_pass:
                            for m_ in bn_mods:
                                m_.momentum = 0.0
                        xs_ = xv[sel].reshape(-1, *x.shape[1:]).contiguous(memory_format=torch.channels_last)
                        lp = model(xs_, b["mask"][sel], b["type"][sel], b["plane"][sel], b["pos"][sel], b["bag"][sel], b["rel"][sel]).float()
                        if ext_pass:
                            for m_, mo_ in zip(bn_mods, bn_mom):
                                m_.momentum = mo_
                        logits = lp.new_zeros((xe_.shape[0], lp.shape[1])) if logits is None else logits
                        logits[sel] = lp
                else:
                    logits = model(x, b["mask"], b["type"], b["plane"], b["pos"], b["bag"], b["rel"]).float()
            eo = 0.0 if ext_off is None else is_ext[b["idx"]][:, None] * ext_off   # 10-18 OAI: source offset on external rows (0 on ours)
            off_b = poe[b["idx"]] + eo
            if site is not None:                          # training-only site offsets on the 5 main outputs (main BCE only)
                g = site[2][b["idx"]]
                so = site[3][site[1][b["idx"]][:, None].expand_as(g), g, torch.arange(g.shape[1], device=g.device).expand_as(g)] * (g > 0)
                off_b = off_b + torch.cat([so, so.new_zeros(so.shape[0], ny - so.shape[1])], 1)
            lg_ = logits[:, :ny]
            if ext_hw is not None:                        # ext_head: external rows read out by their own head
                lx = (model.last_z * ext_hw[model.head_of]).sum(-1)[:, :ny] + ext_hb[:ny]
                lg_ = torch.where(is_ext[b["idx"]][:, None] > 0, lx, lg_)
            lm_ = F.binary_cross_entropy_with_logits(lg_ + off_b, b["y"][:, :ny], reduction="none") * b["w"][:, :ny]
            loss = lm_.sum() / b["w"][:, :ny].sum().clamp(min=1)
            if oh is not None and seg_has.any():          # U91: BCE + soft Dice on the windows with labels, soft area targets at stride 8
                with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                    sl = model.seg_dec([model.seg_feats[r] for r in (8, 16, 32)])
                sl = sl.float()[seg_has]
                tg = F.adaptive_avg_pool2d(oh, sl.shape[-2:])[seg_has].clamp(0, 1)
                pr_ = torch.sigmoid(sl)
                dice = 1 - (2 * (pr_ * tg).sum((0, 2, 3)) + 1) / (pr_.sum((0, 2, 3)) + tg.sum((0, 2, 3)) + 1)
                seg_l = F.binary_cross_entropy_with_logits(sl, tg) + dice.mean()
                loss = loss + cfg["aux_seg"] * seg_l
                seg_log["l"] += float(seg_l.detach()); seg_log["n"] += 1; seg_log["win"] += int(seg_has.sum())
                seg_log["dice"] = seg_log.get("dice", 0.0) + (1 - dice.detach().float().cpu().numpy())   # per-class soft Dice score (tuning log)
                seg_log["main"] = seg_log.get("main", 0.0) + float((lm_.sum() / b["w"][:, :ny].sum().clamp(min=1)).detach())
            if cfg["deep_sup"]:                           # U92: the stride-16 MIL head on the same targets / weights as the main head
                with torch.autocast(dev, dtype=amp, enabled=amp is not None):
                    ds = model.ds_head(model.seg_feats[16], b["mask"]).float()
                lds = F.binary_cross_entropy_with_logits(ds[:, :ny], b["y"][:, :ny], reduction="none") * b["w"][:, :ny]
                ds_l = lds.sum() / b["w"][:, :ny].sum().clamp(min=1)
                loss = loss + cfg["deep_sup"] * ds_l
                seg_log["ds"] = seg_log.get("ds", 0.0) + float(ds_l.detach()); seg_log["dsn"] = seg_log.get("dsn", 0) + 1
            if cfg["slice_soft"]:                         # SLICE-SOFT: BCE of each allowed window vs y_n x presence_k, label weights as the main head
                al = model.ss_allowed[..., :ny].float() * b["w"][:, None, :ny]
                t = b["y"][:, None, :ny] * b["rel"][..., :1].float().clamp(0, 1)
                ls_ = F.binary_cross_entropy_with_logits(model.ss_out.float()[..., :ny], t, reduction="none")
                ss_l = (ls_ * al).sum() / al.sum().clamp(min=1)
                loss = loss + cfg["slice_soft"] * ss_l
                seg_log["ss"] = seg_log.get("ss", 0.0) + float(ss_l.detach()); seg_log["ssn"] = seg_log.get("ssn", 0) + 1
                seg_log["ss_main"] = seg_log.get("ss_main", 0.0) + float((lm_.sum() / b["w"][:, :ny].sum().clamp(min=1)).detach())
                seg_log["ss_t"] = seg_log.get("ss_t", 0.0) + float((t * al).sum() / al.sum().clamp(min=1))
            if cfg["aux_weight"]:                         # D2: the fact heads' own term, mean over the batch's studies x facts (main/train.py form)
                wa = b["w"][:, ny:]
                la = F.binary_cross_entropy_with_logits(logits[:, ny:], b["y"][:, ny:], reduction="none")
                loss = loss + cfg["aux_weight"] * (la * wa).sum() / (wa.shape[0] * wa.shape[1])
            if site is not None:
                loss = loss + cfg["site_l2"] * site[3].pow(2).sum(-1).mean()
            if pw is not None:                            # 'pw': rank the IMAGE logits of the 5 main outputs (no offsets)
                loss = loss + cfg["pw_weight"] * pw[3].loss(logits[:, :len(OUTS)], pw[0][b["idx"]], pw[1][b["idx"]], pw[2][b["idx"]])
            if is_ext is not None and not cfg["ext_only"]:
                if ext_off is not None:
                    loss = loss + cfg["ext_off_l2"] * ext_off.pow(2).sum()
                xe = is_ext[b["idx"]] > 0                   # OAI share of the main loss (log only)
                ext_l, ext_n = ext_l + lm_[xe].sum().detach(), ext_n + lm_.sum().detach()
                ext_lc, all_lc = ext_lc + lm_[xe].sum(0).detach(), all_lc + lm_.sum(0).detach()
                tr_log.append((lg_.detach(), b["y"], b["w"], xe))
            if cfg["series_heads"]:           # U22: deep supervision of the present group heads + the stop-grad aggregator
                gl, pr, agg = model.sd_out
                wg = b["w"][:, None, :ny] * pr[:, :, :ny].float()          # [:ny]: D2 fact heads are supervised by the main head only
                eg = 0.0 if ext_off is None else eo[:, None, :]     # 10-18 OAI: the source offset on the group heads too (was main head only)
                lg_ = F.binary_cross_entropy_with_logits(gl[:, :, :ny].float() + eg, b["y"][:, None, :ny].expand_as(gl[:, :, :ny]), reduction="none")
                loss = loss + cfg["sd_weight"] * (lg_ * wg).sum() / wg.sum().clamp(min=1)
                la_ = F.binary_cross_entropy_with_logits(agg[:, :ny].float() + eo, b["y"][:, :ny], reduction="none") * b["w"][:, :ny]
                loss = loss + cfg["sd_agg_weight"] * la_.sum() / b["w"][:, :ny].sum().clamp(min=1)
            if cfg["pl_w"]:
                s = logits[:, -3:] if cfg["pl_head"] else logits[:, :3]
                loss = loss + cfg["pl_w"] * teacher_loss(s, pt[b["idx"]], cfg["pl_loss"], cfg["pl_tau"])
            if cfg["pair_weight"]:
                s, yo = logits[:, :3], torch.where(b["w"][:, :3] > 0, b["y"][:, :3], 0.5)
                loss = loss + cfg["pair_weight"] * pair_loss(s, yo, bank_s, bank_y)
                bank_s, bank_y = torch.cat([s.detach(), bank_s])[:cfg["pair_bank"]], torch.cat([yo, bank_y])[:cfg["pair_bank"]]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if pw is not None:
                pw[3].push(logits[:, :len(OUTS)], pw[0][b["idx"]], pw[1][b["idx"]], pw[2][b["idx"]])
            tl += loss.detach()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:      # step >= total: the spe_ref schedule ends (all-data port)
                break
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        msg = f"ep {ep} loss {float(tl) / max(n, 1):.4f} time {time.time() - te:.0f}s"
        if cfg["aux_seg"]:                                # tuning log (user 10-19: tune coef / LR / decoder): seg vs main BCE, per-class Dice
            nn_ = max(seg_log["n"], 1)
            dc = np.round(seg_log.get("dice", np.zeros(9)) / nn_, 3).tolist()
            msg += (f" | seg loss {seg_log['l'] / nn_:.4f} (x{cfg['aux_seg']} = {cfg['aux_seg'] * seg_log['l'] / nn_:.4f} vs main BCE "
                    f"{seg_log.get('main', 0.0) / nn_:.4f}) over {seg_log['win']} labelled windows; Dice pat/fem/mt/lt cart, med/lat men, fem/tib/pat {dc}")
        if cfg["deep_sup"]:
            msg += f" | deep_sup loss {seg_log.get('ds', 0.0) / max(seg_log.get('dsn', 0), 1):.4f}"
        if cfg["slice_soft"]:                             # tuning log: ss loss vs main BCE, mean soft target over allowed windows
            ns_ = max(seg_log.get("ssn", 0), 1)
            msg += (f" | slice_soft loss {seg_log.get('ss', 0.0) / ns_:.4f} (x{cfg['slice_soft']} = {cfg['slice_soft'] * seg_log.get('ss', 0.0) / ns_:.4f} vs main "
                    f"BCE {seg_log.get('ss_main', 0.0) / ns_:.4f}), mean target {seg_log.get('ss_t', 0.0) / ns_:.3f}")
        seg_log.update(l=0.0, n=0, win=0, ds=0.0, dsn=0, dice=0.0, main=0.0, ss=0.0, ssn=0, ss_main=0.0, ss_t=0.0)
        if is_ext is not None and not cfg["ext_only"]:        # 10-18 OAI model-side log: source offsets per output + OAI share of the main loss
            msg += (f" | ext_off {[round(v, 3) for v in ext_off.detach().cpu().tolist()]}" if ext_off is not None else " | ext_head")
            msg += f" OAI loss share {float(ext_l) / max(float(ext_n), 1e-9):.3f}"
            msg += f" per column {[round(v, 3) for v in (ext_lc / all_lc.clamp(min=1e-9)).cpu().tolist()]}"
            if tr_log:                                    # SANITY ONLY (USER: never decide on OAI AUCs): train-time AUC by source, hard 0 / 1 cells
                zl, yl, wl, xl = (torch.cat(t).float().cpu().numpy() for t in zip(*tr_log))
                xl = xl > 0
                def _auc(m, j):
                    k = m & (wl[:, j] > 0) & (yl[:, j] != 0.5)          # our soft targets binarised at 0.5
                    yb_ = yl[k, j] > 0.5
                    return roc_auc_score(yb_, zl[k, j]) if k.sum() > 20 and 0 < yb_.sum() < k.sum() else float("nan")
                msg += " | train AUC OAI / ours " + " ".join(f"{OUTS[j].split()[0][0]}{OUTS[j].split()[1][0]} {_auc(xl, j):.3f}/{_auc(~xl, j):.3f}" for j in range(ny))
        if (ep >= cfg["eval_from"] and (ep + 1) % cfg["eval_every"] == 0) or last:
            axc = aux_columns(cfg) if cfg["aux_weight"] else ()
            pv = as12(predict(model, dl_va, cfg, dev, amp), va, cfg["pl_head"], cfg["knee_head"], axc)
            pg = as12(predict(model, dl_gold, cfg, dev, amp), gold, cfg["pl_head"], cfg["knee_head"], axc)
            m = panel(pv[OUTS[:3]].pipe(lambda d: d.assign(fold=fold)), pg)
            rows.append({"epoch": ep, **{k: v for k, v in m.items()}})
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            msg += " | " + " ".join(f"{k} {m[k]:.4f}" for k in ("text3", "strat3", "comp3", "thr3", "gold3"))
            msg += " | " + " ".join(f"{k} {m[k]:.4f}" for k in ("text|Medial OA", "text|Lateral OA", "text|PF OA", "comp|Lateral OA"))
            msg += opus_msg(pv, [(["aux:img_pf_patella", "aux:pl_pf_patella", "PF OA"], "PF OA")], HERE.parent)
            pd.concat(oof_all).to_parquet(out_dir / "oof_partial.parquet")     # 10-20 live curves: readable during training (final oof.parquet unchanged)
            pd.concat(gold_all).to_parquet(out_dir / "gold_partial.parquet")
            if cfg["pl_head"]:
                tc = [f"T:{f}" for f in OUTS[:3]]
                mt = panel(pv[tc].set_axis(OUTS[:3], axis=1).assign(fold=fold), pg[tc].set_axis(OUTS[:3], axis=1))
                msg += " | T-head " + " ".join(f"{k} {mt[k]:.4f}" for k in ("text3", "thr3", "gold3"))
            if ep >= cfg["save_from"]:
                torch.save({k_: v for k_, v in model.state_dict().items() if not k_.startswith(("seg_dec.", "ds_head.", "ss_head."))}, out_dir / f"model_ep{ep}.pt")   # U91 / U92: no training-only heads
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


def bags_for(cfg):
    """The bags of a run (used by main() AND scripts/launch.py - 10-18: launch.py had its own build_bags call without cfg series, so oafs
    (series=fs) trained on ALL series; one helper now). knee_head (U47 oa_kh): a 4th (knee) bag for this process only."""
    if cfg["knee_head"] and "knee" not in BAGS:
        assert not (cfg["pl_head"] or cfg["keep_bags"] or cfg["k_bags"] or cfg["split"] or cfg["series_heads"]), "knee_head: plain recipe only"
        BAGS.append("knee")
    import oa_data
    oa_data.CACHE = cfg.get("cache") or "fov160_px400_q95"   # U57: the crop source (bags are rebuilt per run: no key to update)
    return finalize(build_bags(cfg["crop_scale"], cfg["loc_version"], cfg["trim"], cfg["tf_axial"], cfg["series"]))


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
    bags = bags_for(cfg)
    for fold in [int(f) for f in a.folds.split(",")]:
        out = RUNS / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        run_fold(cfg, fold, out, bags)


if __name__ == "__main__":
    main()
