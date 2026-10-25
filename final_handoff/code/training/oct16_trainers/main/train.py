"""Copied 2026-10-14 from experiments/v0/oct7_baseline (the deployed sub_b0 source; runs go to ../runs). Train one config on one or more folds; write OOF predictions and standard metrics.

conda run -n rsna --no-capture-output python experiments/v0/oct7_baseline/train.py --name NAME --folds 0 --set backbone=resnet34.a1_in1k lr=3e-4
Output: runs/NAME/fold{f}/{config.json, log.txt, oof.parquet (epoch, study x 12), gold.parquet, metrics.json, model.pt (final epoch) and model_ep{E}.pt (each eval epoch >= save_from) if save=True};
a row per fold in runs/results.csv"""
import argparse
import json
import os
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402
from timm.optim import param_groups_layer_decay, param_groups_weight_decay  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels, status_table  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from common.paths import FINDINGS, KEY_OF, KEYS  # noqa: E402
from data import STRUCTURES, contrast_partners, series_not_kept, structure_coverage, StudyDataset, build_index, gpu_prep, side_flips, true_other_series  # noqa: E402
from labels import RECIPES, ATTRS, attribute_targets, make_targets, n_status_classes, status_targets  # noqa: E402
from model import KneeNet  # noqa: E402

DEFAULTS = dict(
    backbone="resnet34.a1_in1k", pretrained=True, cache="fov160_px400_q95", img=256, crop_mm=140, interp="bilinear",
    k=24, k_eval=96, span=(0.05, 0.95), drop_other_knee=False,
    bs=8, epochs=12, lr=3e-4, lr_head=1e-3, wd=1e-2, warmup=1.0, min_lr=0.0, grad_clip=1.0, ema=0.0,
    drop=0.2, hidden=256, head="attn", q_layers=1, q_heads=8, use_type=True, use_pos=True, drop_path=0.0, freeze_bn=False,
    labels="v1prob", nm=0.15, nm_weight=1.0, hard_trusted=False, w_trusted=1.0, site_silence_thr=0.0, site_silence_value=0.03, site_silence_min=20, fs_shrink=5.0, v0_mix=False, syn_proxy=False, eff_below=0.0, cf_cap=0.0, sharpen=0.0, fo_value=0.0, unc_weight=1.0, aux_status=0.0, b_findings="", b_labels="", b_sharpen=0.0, b_v0_mix=False, b_nm=0.0, b_weight=1.0,
    pos_weight=0.0, precision="bf16", channels_last=True, compile=False,
    eval_every=2, eval_from=12, workers=12, seed=0, k_folds=5, prior_init=True, max_steps=0, spe_ref=0, save=False, save_from=-1,
    aug_rot=0.0, aug_scale=0.0, aug_shift=0.0, aug_bc=0.0, aug_gamma=0.0, aug_noise=0.0, aug_erase=0.0, aug_wdrop=0.0,
    aug_lowres=0.0, aug_bagmix=0.0, aug_mixup=0.0, aug_mirror=0.0, aug_geo_p=1.0, sample="random", rank_weight=0.0, rank_margin=0.3,
    kill_epoch=-1, kill_below=0.0, attr_weight=0.0, aux_smooth=0.0, aux_groups="", pos_scale=60.0, side_fix=False, clean_labels=False, drop_true_other=False, series_keep="", ctx_layers=0, gru=False,
    ext_labels="", ext_weight=0.0, samp_rare=0.0, hard_pow=0.0, hard_band=(), hard_mult=2.0, hard_from=2,
    llrd=0.0, nwd_1d=False, freeze_epochs=0.0, bb_pool="avg", lora_r=0, lora_targets="qkv,proj", grad_ckpt=False, sam=0.0, bb_weights="", lr_seq=0.0, syn_mask_rate=0.0, multicontrast=False, auc_weight=0.0, auc_queue=512, bce_weight=1.0, boost_w="", boost_cols="", boost_x=False, pkg_ctx="", struct_cov="", prior="", cov_weight=0.0, syn_tie=False, type_prior=False, eval_workers=6, aug_sdrop=0.0, mono=False, prior_findings="", severity=False, head_recipes="", head_weight=0.5, head_avg=True, img_syn="", ext_avg="", asl_gneg=0.0, asl_m=0.05,
    # oct14_text_labels options (all off by default: a run without them = the copied sub_b0 recipe)
    tgt_csv="", tgt_findings="", aux_csv="", aux_weight=0.0, aux_cols="", site_csv="", site_mode="", site_l2=1e-3, site_min=20, pw_weight=0.0, pw_csv="", pw_queue=512, pw_cross=True, pw_site_csv="data/site_train.csv", pw_balance=0.0,
    pair_weight=0.0, pair_mode="all", pair_csv="", pair_margin=0.1, pair_queue=512, pair_findings="", pair_st_filter="", aucm_weight=0.0,
    aucm_margin=1.0, aucm_lr=0.0,
    init="", ft_mask="", ft_keep="", tgt_nan_w0=False, tgt_partial="", w_csv="",
    xs_cache="", xs_labels="", xs_type_map="", xs_types="", xs_off_l2=1e-3, xs_max=0, xs_only=False, xs_head=False,
)
# xs_*          10-18 OAI (USER 'absolutely consider training on dataset with OAI data added from scratch'; port of oa_train's ext rows; 'xs' =
#               extra source, as 'ext_*' already names the external label heads): xs_cache = the source's slice cache (absolute path, same
#               grid), xs_labels = its label CSV (StudyInstanceUID + the 12 FINDINGS, soft values, NaN = masked: OAI gives MOA / LOA / PF / MM / LM /
#               Effusion / Synovitis proxy, the rest masked), xs_type_map = '+'-joined FROM:TO series_type remaps (DESS), xs_types = ORIGINAL
#               series types kept, xs_max = cap on the source's studies. Its studies join the TRAINING side only (never val / gold / OOF), with a
#               training-only per-finding SOURCE OFFSET on the main 12 logits (L2 xs_off_l2, dropped at prediction). Inert when xs_cache is ''.
#               xs_only = train on the extra-source studies ONLY (main_oaionly: never sees a competition label; no source offset).
#               xs_head = a per-SOURCE readout (port of oa_train ext_head, which beat the offset on the OA expert, oai_srchead): the extra-source
#               rows are scored by their own training-only 12-output head (weights + bias, initialised as the main head, outside the model
#               state dict) on the SHARED attention-pooled features; our rows keep the main head; no source offset; prediction unchanged.
# tgt_csv       CSV (StudyInstanceUID + FINDINGS) of soft targets; '{fold}' in the path -> the fold number ('all' for fold -1);
#               replaces the recipe's targets of tgt_findings ('+'-joined, '_' for spaces; '' = all 12) where the CSV is not NaN
#               (D1 label recipes: data/targets/{nmc,sil,vsite,vclue,v4,v4ns}_f{fold}.csv from scripts/build_targets.py)
# aux_csv       CSV (StudyInstanceUID + any columns, values in [0, 1], NaN = unknown) -> extra attention outputs with masked BCE,
#               weight aux_weight, columns filtered by the regex aux_cols (D2 fact heads: data/aux_facts.csv)
# site_csv      CSV with StudyInstanceUID + site (D3 training-only site term): site_mode 'all' adds a learned per (site, finding)
#               logit offset to the main 12 logits in the training loss only; 'vague' adds it only on non-explicit cells (silent /
#               vague / general_normal / excluded), per (site, finding, silent-or-vague); sites with < site_min training studies
#               share one 'other' offset; L2 penalty site_l2 on the offsets. Never used at prediction (predict() ignores it), so the
#               image pathway need not learn each site's reporting habit. 'fluid' (U68 G2 'tlssf', 10-18): only on Effusion / Synovitis /
#               Baker's, per (site, finding, v1 status 5-way: denied (+ structure / general normal, excluded) / below / unsettled (+ hedged,
#               conflict) / meets / silent) on EVERY cell, plus a site-free per (finding, report-thoroughness tercile) offset on silent cells
#               (tercile of the report's explicit negatives = denied / structure_normal over the 12 keys, all non-gold studies).
# tgt_partial   with tgt_csv: '+'-joined findings whose CSV may cover < 90% of training studies (their NaN cells keep the recipe's
#               targets, full weight; U68 G1: Baker's cell rates on below / unsettled cells only)
# init / ft_mask / ft_keep   clean-label stage as in oct12_pseudo: init = checkpoint path ('{fold}' allowed, relative to the
#               experiment folder); ft_mask 'x' = only explicit cells (meets / denied / structure_normal) train, except the findings
#               in ft_keep (',' or '+'-joined, '_' for spaces) whose cells all train
# w_csv         C223 way C (10-20): CSV (StudyInstanceUID + FINDINGS columns) of per-cell LOSS MULTIPLIERS on the training cells ('{fold}'
#               allowed); NaN / missing cells = 1; targets unchanged (agreement weighting: 1.0 where image read and text agree, 0.3 where not)
# tgt_nan_w0    with tgt_csv: cells where the CSV is NaN get loss weight 0 (masked stages, e.g. data/targets/pl_cstage = the
#               oct12_pseudo teacher-confidence stage L6Fc_ism: agreeing explicit cells + teacher-confident other cells)


class EpochSampler(Sampler):
    """Shuffled indices offset by epoch * n, so workers can derive the epoch without shared state. With weights (n,) set, study i
    appears about n * w_i / sum(w) times per epoch (floor + random rounding of the expected count), still n draws per epoch."""
    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.weights = n, seed, 0, None

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        if self.weights is None:
            idx = torch.randperm(self.n, generator=g)
        else:
            c = self.weights / self.weights.sum() * self.n
            idx = torch.repeat_interleave(torch.arange(self.n), (c.floor() + (torch.rand(self.n, generator=g) < c.frac())).long())
            idx = torch.cat([idx[torch.randperm(len(idx), generator=g)], torch.multinomial(self.weights, self.n, True, generator=g)])[: self.n]
        return iter((idx + self.epoch * self.n).tolist())

    def __len__(self):
        return self.n


class EpochDataset(StudyDataset):
    def __getitem__(self, i):
        self.epoch, j = divmod(i, len(self.studies))
        return super().__getitem__(j) | {"idx": torch.tensor(j)}


class IdxStudyDataset(StudyDataset):
    """Eval dataset whose items carry their position (U80 R-STK2: per-study context rows)."""
    def __getitem__(self, j):
        return super().__getitem__(j) | {"idx": torch.tensor(j)}


def sample_weights(cfg, y, w, study_loss, ep):
    """Per-study sampling weights for the epoch, or None (uniform). samp_rare = gamma: w_i = max(1, max_f y_if (p_f / p_max)^-gamma)
    with p_f the soft prevalence (BirdCLEF class-balanced sampling). From epoch hard_from, using each study's latest training loss:
    hard_pow = beta multiplies by (loss / mean loss)^beta (ratio clipped to [0.1, 10]); hard_band = (lo, hi) multiplies the studies
    whose loss percentile lies in (lo, hi] by hard_mult (e.g. 0.8,1.0 = top 20%; 0.7,0.95 skips the most extreme, likely mislabelled)."""
    s = np.ones(len(y))
    if cfg["samp_rare"]:
        p = (y * w).sum(0) / w.sum(0)
        s = np.maximum(1, (y * (p / p.max()) ** -cfg["samp_rare"]).max(1))
    if (cfg["hard_pow"] or cfg["hard_band"]) and ep >= cfg["hard_from"]:
        assert not np.isnan(study_loss).all()
        loss = np.where(np.isnan(study_loss), np.nanmedian(study_loss), study_loss)
        if cfg["hard_pow"]:
            s = s * np.clip(loss / loss.mean(), 0.1, 10) ** cfg["hard_pow"]
        if cfg["hard_band"]:
            lo, hi = cfg["hard_band"]
            r = pd.Series(loss).rank(pct=True).to_numpy()
            s = s * np.where((r > lo) & (r <= hi), cfg["hard_mult"], 1.0)
    return None if np.all(s == 1) else torch.tensor(s, dtype=torch.float64)


def parse_value(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    if v in ("True", "False"):
        return v == "True"
    if "," in v:
        return tuple(parse_value(x) for x in v.split(","))
    return v


def external_targets(path, studies):
    """(n, E) probabilities from a CSV (StudyInstanceUID + columns named as FINDINGS, e.g. image-read Synovitis); NaN = unlabelled."""
    t = pd.read_csv(path).set_index("StudyInstanceUID")
    assert len(t.columns) and set(t.columns) <= set(FINDINGS), t.columns
    return t.reindex(studies).to_numpy(np.float32)


def mix_batch(x, b, cfg, aux_from):
    """x (B*K, 3, S, S). aug_bagmix: with this probability a study keeps its first K/2 windows and takes the last K/2 from a random
    partner; target 1 - (1 - y_a)(1 - y_b). aug_mixup: with this probability pixels and targets are blended with a random partner,
    lam ~ Beta(0.4, 0.4). Loss weight = min of the two studies' weights; attribute and status targets of mixed studies are ignored.
    b["y"] layout: 12 main targets, attributes, status classes from column aux_from."""
    bsz, k = b["mask"].shape
    x = x.view(bsz, k, *x.shape[1:])
    dev = x.device
    y, w = b["y"][:, :12], b["w"]
    mixed = torch.zeros(bsz, dtype=torch.bool, device=dev)
    if cfg["aug_bagmix"]:
        perm, sel = torch.randperm(bsz, device=dev), torch.rand(bsz, device=dev) < cfg["aug_bagmix"]
        h = k // 2
        for key in ("mask", "type", "pos", "sid", "cov"):
            b[key][sel, h:] = b[key][perm][sel, h:]
        x[sel, h:] = x[perm][sel, h:]
        y = torch.where(sel[:, None], 1 - (1 - y) * (1 - y[perm]), y)
        w = torch.where(sel[:, None], torch.minimum(w, w[perm]), w)
        w[:, 12:] *= ~sel[:, None]
        mixed |= sel
    if cfg["aug_mixup"]:
        perm, sel = torch.randperm(bsz, device=dev), torch.rand(bsz, device=dev) < cfg["aug_mixup"]
        lam = torch.distributions.Beta(0.4, 0.4).sample((bsz,)).to(dev)
        lam = torch.where(sel, lam, torch.ones_like(lam))
        x = lam.view(-1, 1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1, 1)) * x[perm]
        y = lam[:, None] * y + (1 - lam[:, None]) * y[perm]
        w = torch.where(sel[:, None], torch.minimum(w, w[perm]), w)
        w[:, 12:] *= ~sel[:, None]
        b["mask"] = b["mask"] | torch.where(sel[:, None], b["mask"][perm], b["mask"])
        b["cov"] = torch.where(sel[:, None, None], torch.maximum(b["cov"], b["cov"][perm]), b["cov"])
        mixed |= sel
    rest = b["y"][:, 12:].clone()
    rest[mixed, aux_from - 12:] = -100
    b["y"] = torch.cat([y, rest], 1)
    b["w"] = w
    return x.reshape(bsz * k, *x.shape[2:])


MIRROR_SWAP = [FINDINGS.index(f) for f in ["ACL", "MCL", "Lateral Meniscus", "Medial Meniscus", "Lateral OA", "Medial OA", "PF OA",
                                             "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]]


def mirror(x, b, p, mean, std, aux_from):
    """With probability p per study, swap the medial-lateral axis: sagittal windows reverse their 3 slices and negate the slice
    position, coronal / axial windows flip left-right. Labels: medial <-> lateral meniscus and OA swap; MCL (no lateral twin) gets
    loss weight 0. x is already normalised per channel (mean, std (1, 3, 1, 1)), so reversed slices are re-normalised for their
    new channel."""
    bsz, k = b["mask"].shape
    sel = torch.rand(bsz, device=x.device) < p
    if not sel.any():
        return x
    x = x.view(bsz, k, *x.shape[1:])
    sag = (b["plane"] == 0) & sel[:, None]
    other = (b["plane"] != 0) & sel[:, None]
    mean, std = mean.to(x.device).view(1, 1, 3, 1, 1), std.to(x.device).view(1, 1, 3, 1, 1)
    flipped = (x.flip(2) * std.flip(2) + mean.flip(2) - mean) / std
    x = torch.where(sag[..., None, None, None], flipped, x)
    x = torch.where(other[..., None, None, None], x.flip(-1), x)
    b["pos"] = torch.where(sag, -b["pos"], b["pos"])
    b["cov"] = torch.where(sel[:, None, None], b["cov"][..., MIRROR_SWAP], b["cov"])   # medial <-> lateral structures
    y, attr, aux = b["y"][:, :12], b["y"][:, 12:aux_from], b["y"][:, aux_from:]
    if aux.shape[1]:                                       # status classes per finding: same swap, MCL unknown
        aux_sw = aux[:, MIRROR_SWAP].clone()
        aux_sw[:, FINDINGS.index("MCL")] = -100
        aux = torch.where(sel[:, None], aux_sw, aux)
    b["y"] = torch.cat([torch.where(sel[:, None], y[:, MIRROR_SWAP], y), attr, aux], 1)
    w = torch.where(sel[:, None], b["w"][:, MIRROR_SWAP], b["w"][:, :12])
    w[sel, FINDINGS.index("MCL")] = 0
    b["w"] = torch.cat([w, b["w"][:, 12:] * ~sel[:, None]], 1)
    return x.reshape(bsz * k, *x.shape[2:])


def rank_loss(logits, y, w, margin):
    """Pairwise logistic loss over study pairs in the batch whose targets differ by > margin, per finding (an AUC surrogate)."""
    ds = logits[:, None, :] - logits[None, :, :]
    m = ((y[:, None, :] - y[None, :, :]) > margin) & (w[:, None, :] > 0) & (w[None, :, :] > 0)
    return (F.softplus(-ds) * m).sum() / m.sum().clamp(min=1)


class AUCMemory:
    """Soft pairwise AUC loss with a FIFO of recent studies' detached logits / targets / weights (cross-batch memory, Wang et al.
    2020), so rare findings get pairs beyond one batch of 8. Per finding f, over i in the batch and j in batch + memory (j != i):
    sum_ij y_i (1 - y_j) softplus(s_j - s_i) + (1 - y_i) y_j softplus(s_i - s_j), cell weights multiplied in, divided by the pair
    weights; mean over findings with pairs (BirdCLEF 2025 4th / 2026 2nd: soft AUC beat BCE on the LB)."""
    def __init__(self, size):
        self.size, self.mem = size, None

    def loss(self, s, y, w):
        b = s.shape[0]
        all_s, all_y, all_w = (s, y, w) if self.mem is None else (torch.cat([s, self.mem[0]]), torch.cat([y, self.mem[1]]), torch.cat([w, self.mem[2]]))
        not_self = torch.ones(b, all_s.shape[0], 1, device=s.device)
        not_self[:, :b] -= torch.eye(b, device=s.device)[..., None]
        wp = (y * w)[:, None] * ((1 - all_y) * all_w)[None] * not_self          # i positive, j negative
        wn = ((1 - y) * w)[:, None] * (all_y * all_w)[None] * not_self          # i negative, j positive
        d = all_s[None] - s[:, None]                                            # s_j - s_i
        num = (wp * F.softplus(d) + wn * F.softplus(-d)).sum((0, 1))
        den = (wp + wn).sum((0, 1))
        has = den > 1e-6
        return (num[has] / den[has]).mean() if has.any() else s.sum() * 0

    def push(self, s, y, w):
        new = (s.detach(), y.detach(), w.detach())
        self.mem = new if self.mem is None else tuple(torch.cat([n, m])[: self.size] for n, m in zip(new, self.mem))


class PairMemory(AUCMemory):
    """'pw' (oct16_experts 10-18, Brainstorm request 16 / user U66-U67: learn between-site calibration): RankNet (softplus) pairwise loss on
    the IMAGE logits (before any training-only site offsets) over TRUSTED cells only (hard 1 / 0 from pw_csv, empty = no pair) and, with
    pw_cross, only over pairs of studies from DIFFERENT sites (in batch + a FIFO memory of recent studies). The same pair maths as AUCMemory
    with a site mask; i == j pairs drop out automatically (same site)."""
    def __init__(self, size, cross):
        super().__init__(size)
        self.cross = cross

    def loss(self, s, y, w, site):
        b = s.shape[0]
        mem = self.mem
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


PAIR_MODES = ("all", "same_site", "ordinal", "ties_teacher", "matched", "hard")


def pair_mask(mode, yi, yj, ti, tj, sti, stj, si, sj, margin):
    """U90 (10-19, user ranking-loss direction): which pairs say 'i above j' per finding (broadcast tensors / arrays; NaN compares False).
    all / ordinal: y_i - y_j > margin (ordinal = graded y from pair_csv); same_site: + same site; matched: + same stratum (confounder-
    matched, e.g. MCL pairs within one Contusion status); ties_teacher: same stratum st (the tie group; st = y when pair_csv has no st_<F>) AND teacher t_i - t_j >
    margin (the user's 'pseudo labels to break ties'); hard: y_i - y_j > margin AND the teacher (the shipped OOF) does not order them by >
    margin (t_i - t_j < margin: the package mis-orders or ties them)."""
    up = (yi - yj) > margin
    if mode in ("all", "ordinal"):
        return up
    if mode == "same_site":
        return up & (si == sj)
    if mode == "matched":
        return up & (sti == stj)
    if mode == "ties_teacher":
        return (sti == stj) & ((ti - tj) > margin)
    assert mode == "hard", mode
    return up & ((ti - tj) < margin)


class GenPairMemory:
    """U90 generalized pairwise RankNet loss (softplus(s_j - s_i) for every pair 'i above j' of pair_mask) over the batch x (batch + a
    FIFO memory of recent studies), per finding; a pair needs both cells weighted (w > 0); mean over findings with pairs."""
    def __init__(self, size, mode, margin):
        self.size, self.mode, self.margin, self.mem = size, mode, margin, None

    def loss(self, s, y, w, t, st, site):
        b = s.shape[0]
        cur = (s, y, w, t, st, site)
        a = cur if self.mem is None else tuple(torch.cat([c, m]) for c, m in zip(cur, self.mem))
        not_self = torch.ones(b, a[0].shape[0], 1, device=s.device, dtype=torch.bool)
        not_self[:, :b] &= ~torch.eye(b, device=s.device, dtype=torch.bool)[..., None]
        ok = (w[:, None] > 0) & (a[2][None] > 0) & not_self
        si, sj = site[:, None, None], a[5][None, :, None]
        up = pair_mask(self.mode, y[:, None], a[1][None], t[:, None], a[3][None], st[:, None], a[4][None], si, sj, self.margin) & ok
        dn = pair_mask(self.mode, a[1][None], y[:, None], a[3][None], t[:, None], a[4][None], st[:, None], sj, si, self.margin) & ok
        d = a[0][None] - s[:, None]                                   # s_j - s_i
        num = (up * F.softplus(d) + dn * F.softplus(-d)).sum((0, 1))
        den = (up.float() + dn.float()).sum((0, 1))
        has = den > 0
        return (num[has] / den[has]).mean() if has.any() else s.sum() * 0

    def push(self, s, y, w, t, st, site):
        new = (s.detach(), y, w, t, st, site)
        self.mem = new if self.mem is None else tuple(torch.cat([n, m])[: self.size] for n, m in zip(new, self.mem))


def pair_setup(cfg, tr, y, w):
    """U90: the per-study pair inputs (y / w / teacher t / stratum st / site, train order) and a log line with the pair counts per finding over all
    training studies + the share of them the shipped 0.961 OOF already orders correctly (testable on CPU)."""
    assert cfg["pair_mode"] in PAIR_MODES, cfg["pair_mode"]
    pc = pd.read_csv(exp_path(cfg["pair_csv"])).set_index("StudyInstanceUID").reindex(tr) if cfg["pair_csv"] else pd.DataFrame(index=tr)
    cn = lambda pre, f: f"{pre}_{f.replace(' ', '_').replace(chr(39), '')}"  # noqa: E731   (pair_csv names: spaces -> '_', Baker's -> Bakers)
    col = lambda pre, f: pc[cn(pre, f)].to_numpy(np.float32) if cn(pre, f) in pc else np.full(len(tr), np.nan, np.float32)  # noqa: E731

    def strata(f):                                    # st_<F> may be strings ('<status>|<teacher source>'): integer codes; pair_st_filter
        v = pc[cn("st", f)]                           # keeps only strata containing it (e.g. '|codex'); others / missing -> NaN (no pair)
        if not pd.api.types.is_numeric_dtype(v):
            v = v.where(v.astype(str).str.contains(cfg["pair_st_filter"], regex=False) & v.notna()) if cfg["pair_st_filter"] else v
            codes = pd.Series(pd.factorize(v)[0], index=v.index).astype(np.float32)
            return codes.where(codes >= 0).to_numpy(np.float32)
        return v.to_numpy(np.float32)
    ya, wa = np.asarray(y, np.float32), np.asarray(w, np.float32)
    ry = np.stack([col("y", f) if cn("y", f) in pc else ya[:, j] for j, f in enumerate(FINDINGS)], 1)      # y: pair_csv y_<F> else the
    rw = np.stack([np.isfinite(col("y", f)) if cn("y", f) in pc else wa[:, j] > 0 for j, f in enumerate(FINDINGS)], 1).astype(np.float32)  # assigned target
    rt = np.stack([col("t", f) for f in FINDINGS], 1)
    rst = np.stack([strata(f) if cn("st", f) in pc else ry[:, j] for j, f in enumerate(FINDINGS)], 1)   # no st_<F>: the tie group = y
    if cfg["pair_findings"]:                          # restrict the pair loss to these findings ('+'-joined, '_' for spaces)
        keepf = [f.replace("_", " ") for f in cfg["pair_findings"].split("+")]
        assert all(f in FINDINGS for f in keepf), keepf
        rw[:, [j for j, f in enumerate(FINDINGS) if f not in keepf]] = 0
    rsite = pd.factorize(pd.read_csv(exp_path(cfg["pw_site_csv"])).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str))[0]
    bo = pd.read_csv(EXP / "data/corrector_base_0961.csv").set_index("StudyInstanceUID").reindex(tr)[FINDINGS].to_numpy(np.float32)
    cnt = {}
    for j, f in enumerate(FINDINGS):                  # setup log: pairs per finding over all training studies + the share of them the
        v = rw[:, j] > 0                              # shipped 0.961 OOF already orders correctly (where it has both studies)
        m = pair_mask(cfg["pair_mode"], ry[v, j][:, None], ry[v, j][None], rt[v, j][:, None], rt[v, j][None], rst[v, j][:, None],
                      rst[v, j][None], rsite[v][:, None], rsite[v][None], cfg["pair_margin"])
        bb = bo[v, j]
        both = m & np.isfinite(bb)[:, None] & np.isfinite(bb)[None]
        cnt[f] = (int(m.sum()), round(float((both & (bb[:, None] > bb[None])).sum() / max(both.sum(), 1)), 3))
    msg = (f"pair rule {cfg['pair_mode']} (margin {cfg['pair_margin']}, weight {cfg['pair_weight']}, csv {cfg['pair_csv'] or '-'}): pairs / share "
        f"the shipped OOF already orders: {cnt}")
    return ry, rw, rt, rst, rsite, msg


def aucm_loss(logits, yh, wh, p, a, b, alpha, margin):
    """U90 RK1: AUC-M (Yuan et al. 2021, LibAUC AUCMLoss) per finding on sigmoid scores over hard 0 / 1 cells (wh > 0), batch means;
    a / b by descent, alpha by ASCENT (its gradient is negated by a hook) and projected to >= 0 after each step; mean over findings with cells."""
    sc = torch.sigmoid(logits)
    pos, neg = (yh > 0.5) & (wh > 0), (yh < 0.5) & (wh > 0)
    n = (wh > 0).sum(0).clamp(min=1).float()
    L = ((1 - p) * ((sc - a) ** 2 * pos).sum(0) / n + p * ((sc - b) ** 2 * neg).sum(0) / n
         + 2 * alpha * (p * (1 - p) * margin + ((p * sc * neg) - ((1 - p) * sc * pos)).sum(0) / n) - p * (1 - p) * alpha ** 2)
    has = (wh > 0).any(0)
    return L[has].mean() if has.any() else logits.sum() * 0


def ext_avg_pairs(cfg, n_attr):
    """(main finding column, external-label output column) pairs averaged at prediction (ext_avg = findings joined by '+')."""
    if not cfg.get("ext_avg"):
        return []
    cols = pd.read_csv(cfg["ext_labels"], nrows=1).columns[1:].tolist()
    return [(FINDINGS.index(f.replace("_", " ")), 12 + n_attr + cols.index(f.replace("_", " "))) for f in cfg["ext_avg"].split("+")]


def head_names(cfg):
    return cfg["head_recipes"].split("+") if cfg["head_recipes"] else []


def head_slices(cfg, n_attr, n_ext):
    """Output columns of the label-recipe heads averaged with the main 12 at prediction time (head_avg)."""
    lo = 12 + n_attr + n_ext
    return [(lo + 12 * i, lo + 12 * (i + 1)) for i in range(len(head_names(cfg)))] if cfg["head_avg"] else []


def sam_second_pass(model, loss_fn, rho):
    """SAM (Foret et al. 2021), second half of a step: with the gradient g at w in .grad, move to w + rho g / |g|, take the gradient
    there (BatchNorm running statistics frozen for this pass), move back; the optimiser then steps with that gradient."""
    ps = [p for p in model.parameters() if p.grad is not None]
    with torch.no_grad():
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in ps]))
        eps = [p.grad * (rho / (norm + 1e-12)) for p in ps]
        for p, e in zip(ps, eps):
            p.add_(e)
            p.grad = None
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    loss_fn().backward()
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    with torch.no_grad():
        for p, e in zip(ps, eps):
            p.sub_(e)


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def predict(model, loader, cfg, dev, amp, with_aux=False):
    """Sigmoid of the 12 main outputs (n, 12); with_aux=True also returns the status softmax (n, 12, n_aux) or None."""
    model.eval()
    out, aux_out = [], []
    for b in loader:
        b = to_dev(b, dev)
        x = gpu_prep(b["x"], cfg["img"], cfg["interp"], None, *model.norm_stats(), mono=cfg["mono"])
        if cfg["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
        if getattr(model, "pkg_n", 0):                    # U80 R-STK2: the loader's per-study package-logit rows
            model.pkg_x = loader.pkg_ctx[b["idx"]]
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(x, b["mask"], b["type"], b["pos"], b["sid"], cov=b["cov"])
            lg = (o[0] if isinstance(o, tuple) else o).float()
            main = lg[:, :12].clone()
            for fi, ci in getattr(model, "ext_avg_pairs", []):    # main finding logit averaged with its external-label head
                main[:, fi] = (main[:, fi] + lg[:, ci]) / 2
            out.append(torch.sigmoid(torch.stack([main] + [lg[:, lo:hi] for lo, hi in getattr(model, "head_slices", [])]).mean(0)).cpu())
            if isinstance(o, tuple):
                aux_out.append(torch.softmax(o[1][:, :12].float(), -1).cpu())
    probs = torch.cat(out).numpy()
    if with_aux:
        return probs, (torch.cat(aux_out).numpy() if aux_out else None)
    return probs


EXP = HERE.parent
EXPLICIT = ("meets", "denied", "structure_normal")
FLUID = ("Effusion", "Synovitis", "Baker's")                       # site_mode 'fluid' (U68 G2): v1 status -> 5-way group
STATUS5 = {"denied": 1, "structure_normal": 1, "general_normal": 1, "excluded": 1, "below": 2, "unsettled": 3, "hedged": 3, "conflict": 3,
           "meets": 4, "not_mentioned": 5}


def exp_path(p, fold=None):
    """Path relative to the experiment folder; '{fold}' -> the fold number ('all' for the all-data fold -1)."""
    if fold is not None:
        p = p.replace("{fold}", "all" if fold < 0 else str(fold))
    return EXP / p


def finding_list(s):
    s = "+".join(s) if isinstance(s, (tuple, list)) else s          # --set turns 'a,b' into a tuple
    return [f.replace("_", " ") for f in re.split(r"[+,]", s) if f]


def text_label_overrides(cfg, fold, tr, y, w):
    """tgt_csv target replacement, then the ft_mask cell mask (both off by default)."""
    if cfg["tgt_csv"]:
        t = pd.read_csv(exp_path(cfg["tgt_csv"], fold)).set_index("StudyInstanceUID").reindex(tr)
        changed = {}
        for f in finding_list(cfg["tgt_findings"]) or FINDINGS:
            v = t[f].to_numpy(np.float32)
            changed[f] = int((np.isfinite(v) & (np.abs(v - y[:, FINDINGS.index(f)]) > 1e-6)).sum())
            assert cfg["tgt_nan_w0"] or f in finding_list(cfg["tgt_partial"]) or np.isfinite(v).mean() > 0.9, \
                f"tgt_csv covers < 90% of training studies for {f}"
            ok = np.isfinite(v)
            y[ok, FINDINGS.index(f)] = v[ok]
            if cfg["tgt_nan_w0"]:
                w[~ok, FINDINGS.index(f)] = 0.0
        print(f"tgt_csv {cfg['tgt_csv']}: training cells whose target CHANGED (|dy| > 1e-6) per finding: {changed}", flush=True)
    if cfg["w_csv"]:                                     # C223 way C: per-cell loss multipliers (NaN / missing = 1); targets untouched
        t = pd.read_csv(exp_path(cfg["w_csv"], fold)).set_index("StudyInstanceUID").reindex(tr)
        assert set(t.columns) <= set(FINDINGS), set(t.columns) - set(FINDINGS)
        mult = {}
        for f in t.columns:
            m = t[f].to_numpy(np.float32)
            ok = np.isfinite(m)
            assert (m[ok] >= 0).all(), f
            w[ok, FINDINGS.index(f)] *= m[ok]
            mult[f] = {"cells": int(ok.sum()), "down (< 1)": int((m[ok] < 1).sum())}
        print(f"w_csv {cfg['w_csv']}: training cells with a multiplier per finding: {mult}", flush=True)
    if cfg["ft_mask"]:
        assert cfg["ft_mask"] == "x", cfg["ft_mask"]
        keep = status_table("v1").loc[tr, KEYS].isin(EXPLICIT).to_numpy()
        for f in finding_list(cfg["ft_keep"]):
            keep[:, FINDINGS.index(f)] = True
        print("ft_mask x: trained cells per finding", dict(zip(FINDINGS, np.round(keep.mean(0), 3))), flush=True)
        w = w * keep
    return y, w


def aux_columns(cfg):
    cols = pd.read_csv(exp_path(cfg["aux_csv"]), nrows=1).columns[1:]
    cols = [c for c in cols if re.search(cfg["aux_cols"], c)] if cfg["aux_cols"] else list(cols)
    assert cols, "aux_cols selects no column"
    return cols


def n_aux_cols(cfg):
    return len(aux_columns(cfg)) if cfg.get("aux_weight", 0) > 0 else 0


def aux_targets(cfg, studies):
    """(n, A) fact targets in [0, 1] from aux_csv (NaN = unknown, masked in the loss)."""
    t = pd.read_csv(exp_path(cfg["aux_csv"])).set_index("StudyInstanceUID")[aux_columns(cfg)]
    return t.reindex(studies).to_numpy(np.float32)


def site_setup(cfg, tr, dev):
    """D3 training-only site term: (site index per training study, cell group (n, 12): 0 = no offset, 1 / 2 = offset group,
    offset tensor (n_sites, 3, 12) zero-initialised). 'all': every cell group 1; 'vague': explicit cells 0, silent /
    general_normal 1, other non-explicit 2."""
    s = pd.read_csv(exp_path(cfg["site_csv"])).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
    s = s.where(s.map(s.value_counts()) >= cfg["site_min"], "other")
    names = sorted(s.unique())
    sidx = torch.tensor(s.map({n: i for i, n in enumerate(names)}).to_numpy(), device=dev)
    st = status_table("v1").loc[tr, KEYS]
    tgrp = toff = None
    if cfg["site_mode"] == "all":
        grp = np.ones(st.shape, np.int64)
    elif cfg["site_mode"] == "fluid":
        grp = np.zeros(st.shape, np.int64)
        for f in FLUID:
            grp[:, FINDINGS.index(f)] = st[KEY_OF[f]].map(STATUS5).fillna(0).astype(np.int64).to_numpy()
        allst = status_table("v1")
        nneg = allst.loc[~allst.index.isin(gold_labels().index), KEYS].isin(("denied", "structure_normal")).sum(axis=1)
        ter = pd.qcut(nneg.rank(method="first"), 3, labels=False).reindex(tr)
        assert ter.notna().all(), "thoroughness tercile missing for a training study"
        ter = ter.to_numpy(np.int64) + 1                  # 1 terse / 2 middle / 3 thorough
        tgrp = np.where(grp == 5, ter[:, None], 0)
        toff = torch.nn.Parameter(torch.zeros(4, 12, device=dev))
    else:
        assert cfg["site_mode"] == "vague", cfg["site_mode"]
        grp = np.where(st.isin(EXPLICIT), 0, np.where(st.isin(("not_mentioned", "general_normal")), 1, 2))
    off = torch.nn.Parameter(torch.zeros(len(names), int(grp.max()) + 1 if cfg["site_mode"] == "fluid" else 3, 12, device=dev))
    return names, sidx, torch.tensor(grp, device=dev), off, None if tgrp is None else torch.tensor(tgrp, device=dev), toff


def run_fold(cfg, fold, out_dir):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg["precision"]]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    folds = get_folds(cfg["k_folds"], 0, include_gold=False)
    from common.cache import SliceCache
    cache = SliceCache(cfg["cache"])
    cov = structure_coverage(cfg["struct_cov"], len(cache.slices)) if cfg["struct_cov"] else None
    index = build_index(cache, tuple(cfg["span"]), cfg["drop_other_knee"],
                        side_flips() if cfg.get("side_fix") else frozenset(),
                        frozenset((true_other_series() if cfg.get("drop_true_other") else set()) | series_not_kept(cache, cfg["series_keep"])),   # series_keep: m_fs (oct11 port)
                        contrast_partners() if cfg.get("multicontrast") else None)
    folds = folds[folds.index.isin(index.keys())]
    tr = folds.index[folds.fold != fold].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in index]
    if fold < 0:                                          # all-data run (fold -1; 10-21 port of oct14_all_data): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    y, w = make_targets(tr, cfg)
    y, w = text_label_overrides(cfg, fold, tr, y, w)
    if cfg["boost_w"]:                                    # U80 R-BOOST (CORRECTORS.md): weights ~ |y - p_base| on the boost columns (base = the
        bv = pd.read_csv(exp_path(cfg["boost_w"])).set_index("StudyInstanceUID").reindex(tr)[FINDINGS].to_numpy(np.float32)   # shipped OOF)
        cols = [FINDINGS.index(c.replace("_", " ")) for c in cfg["boost_cols"].split(",") if c] or list(range(12))
        w = w.copy()
        for j in cols:
            have = np.isfinite(bv[:, j])
            r = np.abs(y[:, j] - 1 / (1 + np.exp(-np.nan_to_num(bv[:, j]))))
            r = r / max(float((r * (w[:, j] > 0) * have).sum() / max(((w[:, j] > 0) & have).sum(), 1)), 1e-6)
            if cfg["boost_x"]:                            # R-TAB twin (Brainstorm 10-19): boost only explicit v1 cells; the others keep weight 1
                r = np.where(status_table("v1").loc[tr, KEY_OF[FINDINGS[j]]].isin(EXPLICIT).to_numpy(), r, 1.0)
            w[:, j] = np.where(have, w[:, j] * r, 0.0)
        print(f"boost_w {cfg['boost_w']}: columns {[FINDINGS[j] for j in cols]}; studies with a base " +
              ", ".join(f"{FINDINGS[j]} {int(np.isfinite(bv[:, j]).sum())} / {len(tr)}" for j in cols), flush=True)
    n_aux = n_status_classes(cfg["aux_groups"]) if cfg["aux_status"] > 0 else 0
    n_attr = len(ATTRS) if cfg["attr_weight"] > 0 else 0
    ya = attribute_targets(tr) if n_attr else np.zeros((len(tr), 0), np.float32)
    ye = external_targets(cfg["ext_labels"], tr) if cfg["ext_weight"] > 0 else np.zeros((len(tr), 0), np.float32)
    n_ext = ye.shape[1]
    ya = np.concatenate([ya, ye], 1)                  # attributes then external labels: extra outputs, masked BCE where known
    rec = [make_targets(tr, cfg | RECIPES[n]) for n in head_names(cfg)]   # one 12-output head per extra label recipe
    yr = np.concatenate([r[0] for r in rec], 1) if rec else np.zeros((len(tr), 0), np.float32)
    wr = np.concatenate([r[1] for r in rec], 1) if rec else np.zeros((len(tr), 0), np.float32)
    n_rec = yr.shape[1]
    yx = aux_targets(cfg, tr) if cfg["aux_weight"] > 0 else np.zeros((len(tr), 0), np.float32)   # fact heads, after the recipe heads
    n_auxc = yx.shape[1]
    assert not (n_auxc and cfg["aug_mirror"]), "fact heads name medial / lateral structures: mirror augmentation would mislabel them"
    n_extra = n_attr + n_ext + n_rec + n_auxc
    y_ds = np.concatenate([y, np.nan_to_num(ya), yr, np.nan_to_num(yx), status_targets(tr, cfg["aux_groups"]).astype(np.float32) if n_aux else np.zeros((len(tr), 0), np.float32)], 1)
    w_ds = np.concatenate([w, (~np.isnan(ya)).astype(np.float32), wr, (~np.isnan(yx)).astype(np.float32)], 1)
    log = open(out_dir / "log.txt", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    xs = None
    if cfg["xs_cache"]:                                   # 10-18 OAI: extra-source TRAINING studies (masked labels) + a training-only source offset
        assert not (n_extra or n_aux or cfg["site_mode"] or cfg["pw_weight"] or cfg["struct_cov"] or cfg["multicontrast"] or cfg["aug_mirror"]
                    or cfg["aug_bagmix"] or cfg["aug_mixup"] or cfg["auc_weight"] or cfg["rank_weight"] or fold < 0), \
            "xs rows: the plain 12-output recipe only (no extra heads / site / pw / coverage / mirror / mix / rank / all-data)"
        xc = SliceCache(cfg["xs_cache"])
        tmap = dict(kv.split(":") for kv in cfg["xs_type_map"].split("+")) if cfg["xs_type_map"] else None
        xi = build_index(xc, tuple(cfg["span"]), cfg["drop_other_knee"], type_map=tmap,
                         series_types=set(cfg["xs_types"].split("+")) if cfg["xs_types"] else None)
        assert not set(xi) & set(index), "extra-source study ids collide with competition ids"
        lab = pd.read_csv(Path(__file__).resolve().parents[4] / cfg["xs_labels"]).set_index("StudyInstanceUID").reindex(columns=FINDINGS)   # repo-root relative
        xs_ids = [s_ for s_ in lab.index if s_ in xi and lab.loc[s_].notna().any()]
        if cfg["xs_max"]:
            xs_ids = sorted(np.random.default_rng(cfg["seed"]).choice(xs_ids, min(cfg["xs_max"], len(xs_ids)), replace=False).tolist())
        yq = lab.loc[xs_ids].to_numpy(np.float32)
        wq = np.isfinite(yq).astype(np.float32)
        yq = np.nan_to_num(yq)
        index = {**index, **{s_: xi[s_] for s_ in xs_ids}}
        if cfg["xs_only"]:                                # main_oaionly: the extra-source studies are the whole training set
            tr, y, w, y_ds, w_ds = list(xs_ids), yq, wq, yq, wq
        else:
            tr, y, w = tr + xs_ids, np.concatenate([y, yq]), np.concatenate([w, wq])
            y_ds, w_ds = np.concatenate([y_ds, yq]), np.concatenate([w_ds, wq])
        xs = (cfg["xs_cache"], xs_ids)
        say(f"xs: {len(xs_ids)} extra-source TRAINING studies ({len(xi)} indexed) from {cfg['xs_cache']} + {cfg['xs_labels']}; labelled cells / "
            f"positive rate: " + ", ".join(f"{c} {int(wq[:, j].sum())} / {yq[wq[:, j] > 0, j].mean():.3f}" for j, c in enumerate(FINDINGS) if wq[:, j].any())
            + f"; type remap {tmap}; kept types {cfg['xs_types'] or 'all'}; xs_only {cfg['xs_only']}")
    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    ds_tr = EpochDataset(cfg["cache"], tr, y_ds, w_ds, index, cfg["k"], True, cfg["crop_mm"], cfg["seed"], cfg["sample"], cov, ext=xs)
    sampler = EpochSampler(len(tr), cfg["seed"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk_eval = lambda ids: DataLoader((IdxStudyDataset if cfg["pkg_ctx"] else StudyDataset)(cfg["cache"], ids, None, None, index, cfg["k_eval"], False, cfg["crop_mm"], cov=cov),
                                     batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk_eval(va), mk_eval(gold)

    model = KneeNet(cfg["backbone"], pretrained=cfg["pretrained"], hidden=cfg["hidden"], drop=cfg["drop"], use_type=cfg["use_type"],
                    use_pos=cfg["use_pos"], head=cfg["head"], drop_path=cfg["drop_path"], q_layers=cfg["q_layers"],
                    q_heads=cfg["q_heads"], n_aux=n_aux, n_out=12 + n_extra, pos_scale=cfg.get("pos_scale", 60.0),
                    ctx_layers=cfg["ctx_layers"], gru=cfg["gru"], img_size=cfg["img"], bb_pool=cfg["bb_pool"],
                    lora_r=cfg["lora_r"], lora_targets=cfg["lora_targets"], grad_ckpt=cfg["grad_ckpt"], bb_weights=cfg["bb_weights"],
                    prior=cfg["prior"], cov_head=cfg["cov_weight"] > 0, syn_tie=cfg["syn_tie"], type_prior=cfg["type_prior"], prior_findings=cfg["prior_findings"],
                    pkg_n=12 if cfg["pkg_ctx"] else 0).to(dev)
    pkg_tr = None
    if cfg["pkg_ctx"]:                                    # U80 R-STK2: per-study package OOF logits (gold: the expected gold); NaN -> OOF column mean
        cb = pd.read_csv(exp_path(cfg["pkg_ctx"])).set_index("StudyInstanceUID")[FINDINGS]
        cg = pd.read_csv(EXP / "refs/pkg961/expected_gold_f5_win.csv").set_index("StudyInstanceUID")[FINDINGS]
        cg = np.log(cg.clip(1e-6, 1 - 1e-6) / (1 - cg.clip(1e-6, 1 - 1e-6)))
        ct = lambda ids, src: torch.tensor(src.reindex(ids).fillna(cb.mean()).to_numpy(np.float32), device=dev)  # noqa: E731
        pkg_tr, dl_va.pkg_ctx, dl_gold.pkg_ctx = ct(tr, cb), ct(va, cb), ct(gold, cg)
        say(f"pkg_ctx {cfg['pkg_ctx']}: train studies with a full base row {int(cb.reindex(tr).notna().all(1).sum())} / {len(tr)}; gold = expected gold")
    if cfg["prior_init"]:
        with torch.no_grad():
            p = np.clip(np.concatenate([np.where(w.sum(0) > 0, (y * w).sum(0) / np.maximum(w.sum(0), 1e-9), 0.5),   # 0.5 = an unlabelled column (xs_only)
                                        np.nan_to_num(np.nanmean(ya, 0), nan=0.5),
                                        (yr * wr).sum(0) / np.maximum(wr.sum(0), 1e-6), np.nan_to_num(np.nanmean(yx, 0), nan=0.5)]), 0.01, 0.99)
            model.bias().copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    if cfg["init"]:                                       # clean-label stage: start from a trained checkpoint (full state dict)
        model.load_state_dict(torch.load(exp_path(cfg["init"], fold), map_location="cpu"))   # strict: same outputs as the checkpoint
        say(f"init from {exp_path(cfg['init'], fold)}")
    model.head_slices = head_slices(cfg, n_attr, n_ext)
    model.ext_avg_pairs = ext_avg_pairs(cfg, n_attr)
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    if cfg["freeze_bn"]:
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.requires_grad_(False)
    fwd = torch.compile(model) if cfg["compile"] else model
    head_ids = {id(p) for p in model.head_params()}
    bb = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    if cfg["llrd"]:                                       # layer-wise lr decay (timm groups; biases / norms get no weight decay)
        bb_groups = [g | {"lr": cfg["lr"] * g.pop("lr_scale", 1.0)} for g in param_groups_layer_decay(model.backbone, cfg["wd"], layer_decay=cfg["llrd"])]
    elif cfg["nwd_1d"]:                                   # no weight decay on backbone biases / norms
        bb_groups = [g | {"lr": cfg["lr"]} for g in param_groups_weight_decay(model.backbone, cfg["wd"])]
    else:
        bb_groups = [{"params": bb, "lr": cfg["lr"]}]
    seq_ids = {id(p) for p in model.seq_params()} if cfg["lr_seq"] else set()   # lr_seq 0: sequence layers train at lr_head
    head_groups = [{"params": [p for p in model.head_params() if id(p) not in seq_ids], "lr": cfg["lr_head"]}]
    if seq_ids:
        head_groups.append({"params": model.seq_params(), "lr": cfg["lr_seq"]})
    site = site_setup(cfg, tr, dev) if cfg["site_mode"] else None
    assert site is None or not (cfg["aug_mirror"] or cfg["aug_bagmix"] or cfg["aug_mixup"]), "site term assumes unmixed, unmirrored studies"
    if site is not None:
        head_groups.append({"params": [site[3]] + ([site[5]] if site[5] is not None else []), "lr": cfg["lr_head"], "weight_decay": 0.0})
        say(f"site term '{cfg['site_mode']}': {len(site[0])} site groups (>= {cfg['site_min']} training studies, rest 'other')")
    xs_off = is_xs = xh_w = xh_b = None
    if xs is not None and not cfg["xs_only"]:
        is_xs = torch.tensor([s_ in set(xs[1]) for s_ in tr], dtype=torch.float32, device=dev)
    if is_xs is not None and not cfg["xs_head"]:          # training-only source offset (dropped at prediction)
        xs_off = torch.nn.Parameter(torch.zeros(12, device=dev))
        head_groups.append({"params": [xs_off], "lr": cfg["lr_head"], "weight_decay": 0.0})
    if is_xs is not None and cfg["xs_head"]:              # per-source readout on the shared pooled features (training-only)
        assert cfg["head"] == "attn" and not (cfg["compile"] or model.n_aux), "xs_head: the plain attention head, no compile / aux"
        model.keep_z = True
        xh_w, xh_b = torch.nn.Parameter(model.cls_w.detach().clone()), torch.nn.Parameter(model.cls_b.detach().clone())
        head_groups.append({"params": [xh_w, xh_b], "lr": cfg["lr_head"]})
    xs_log = {"l": 0.0, "n": 0.0}
    pwm, pw_last = None, {}
    if cfg["pw_weight"]:                                  # 'pw' arm (Brainstorm request 16): trusted cross-site pairwise loss
        t = pd.read_csv(exp_path(cfg["pw_csv"])).set_index("StudyInstanceUID").reindex(tr)[FINDINGS]
        pw_y = torch.tensor(t.fillna(0).to_numpy(np.float32), device=dev)
        pw_w = torch.tensor(t.notna().to_numpy(np.float32), device=dev)
        sv = pd.read_csv(exp_path(cfg["pw_site_csv"])).set_index("StudyInstanceUID")["site"].reindex(tr).fillna("other").astype(str)
        pw_site = torch.tensor(pd.factorize(sv)[0], device=dev)
        if cfg["pw_balance"]:                         # U75 (c) site-balanced pairs: a study's cells x (its site's study count)^-b, so a pair
            sw = sv.map(sv.value_counts()).to_numpy(np.float64) ** -cfg["pw_balance"]   # weighs (n_i n_j)^-b; mean 1 over studies with cells
            sw = sw / sw[pw_w.sum(1).cpu().numpy() > 0].mean()
            pw_w = pw_w * torch.tensor(sw, dtype=torch.float32, device=dev)[:, None]
            say(f"pw_balance {cfg['pw_balance']}: study weights {sw.min():.2f}..{sw.max():.2f}")
        pwm = PairMemory(cfg["pw_queue"], cfg["pw_cross"])
        say(f"pw: RankNet weight {cfg['pw_weight']} over {'CROSS-SITE' if cfg['pw_cross'] else 'all'} pairs of trusted cells "
            f"({int((pw_w > 0).sum())} cells in {len(tr)} training studies, {sv.nunique()} sites), queue {cfg['pw_queue']}")
    rpm, rk, rpr, am = None, {}, None, None
    if cfg["pair_weight"]:                                # U90 generalized pair rules (pair_mode / pair_csv: y_<F>, t_<F>, st_<F> / pair_margin)
        ry, rw, rt, rst, rsite, msg = pair_setup(cfg, tr, y, w)
        say(msg)
        rpr = {k: torch.tensor(v, device=dev) for k, v in (("y", ry), ("w", rw), ("t", rt), ("st", rst), ("site", rsite))}
        rpm = GenPairMemory(cfg["pair_queue"], cfg["pair_mode"], cfg["pair_margin"])
    if cfg["aucm_weight"]:                                # U90 RK1: AUC-M on hard explicit cells (v1 meets = 1, denied / structure_normal = 0)
        stt = status_table("v1").loc[tr, KEYS]
        yh, wh = stt.eq("meets").to_numpy(np.float32), stt.isin(EXPLICIT).to_numpy(np.float32)
        pp = ((yh * wh).sum(0) / np.maximum(wh.sum(0), 1)).clip(1e-3, 1 - 1e-3)
        am = {"yh": torch.tensor(yh, device=dev), "wh": torch.tensor(wh, device=dev), "p": torch.tensor(pp, dtype=torch.float32, device=dev)}
        am |= {k: torch.nn.Parameter(torch.zeros(12, device=dev)) for k in ("a", "b", "alpha")}
        am["alpha"].register_hook(lambda g: -g)           # gradient ASCENT on alpha (min-max); projected to >= 0 after each step
        head_groups.append({"params": [am["a"], am["b"], am["alpha"]], "lr": cfg["aucm_lr"] or cfg["lr_head"], "weight_decay": 0.0})   # aucm_lr: C214b
        am["acc"] = [0.0, 0]                              # this epoch's AUC-M term (log; C214b: run 1's a / b / alpha never left ~0 at lr 1e-4)
        say(f"aucm: weight {cfg['aucm_weight']}, margin {cfg['aucm_margin']}; explicit cells per finding {dict(zip(FINDINGS, wh.sum(0).astype(int).tolist()))}; "
            f"positive ratio {dict(zip(FINDINGS, np.round(pp, 3).tolist()))}")
    opt = torch.optim.AdamW([g | {"bb": True} for g in bb_groups] + head_groups, weight_decay=cfg["wd"])
    steps_per_epoch = cfg["spe_ref"] or len(dl_tr)       # spe_ref (10-21 port of oct14_all_data): schedule in the fold runs' epoch units (equal steps)
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg["warmup"] * steps_per_epoch)
    if cfg["spe_ref"]:
        print(f"steps/epoch {len(dl_tr)} (schedule unit {steps_per_epoch}), total steps {total}", flush=True)
    base = [g["lr"] for g in opt.param_groups]

    frozen = int(cfg["freeze_epochs"] * steps_per_epoch)  # linear probe first: backbone lr 0, then its own linear warmup

    def set_lr(step):
        f = step / max(1, warm) if step < warm else cfg["min_lr"] + (1 - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * (f if not (frozen and g.get("bb")) else 0.0 if step < frozen else f * min(1.0, (step - frozen) / max(1, warm)))

    scaler = torch.amp.GradScaler(enabled=cfg["precision"] == "fp16")
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True) if cfg["ema"] > 0 else None
    if cfg["pos_weight"] < 0:                             # -1 = per finding clip((1 - p) / p, 1, 10), p = training-fold mean target (Raptor)
        p = (y * w).sum(0) / w.sum(0)
        pw = torch.tensor(np.clip((1 - p) / p, 1, 10), dtype=torch.float32, device=dev)
        say("pos_weight auto:", dict(zip(FINDINGS, np.round(pw.cpu().numpy(), 2).tolist())))
    else:
        pw = torch.full((12,), cfg["pos_weight"], device=dev) if cfg["pos_weight"] > 0 else None
    assert not (cfg["sam"] and cfg["precision"] == "fp16"), "SAM reads raw gradients; use bf16 / fp32"

    auc_mem = AUCMemory(cfg["auc_queue"]) if cfg["auc_weight"] else None
    assert not (cfg["prior"] or cfg["cov_weight"]) or cfg["struct_cov"], "prior / cov_weight need struct_cov"
    cov_cols = torch.tensor([f in STRUCTURES for f in FINDINGS], device=dev)

    def compute_loss(x, b):
        """Total training loss and the per-study main-12 loss (for hard-example sampling)."""
        if pkg_tr is not None:
            model.pkg_x = pkg_tr[b["idx"]]
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = fwd(x, b["mask"], b["type"], b["pos"], b["sid"], cov=b["cov"])
        logits, aux = out if n_aux else (out, None)
        logits = logits.float()
        s12 = logits[:, :12]                              # U90: pair / AUC-M losses read the image logits (before any training-only offsets)
        if pwm is not None:                               # 'pw': rank the IMAGE logits (before the training-only site offsets)
            pw_last["s"] = logits[:, :12]
        if site is not None:                              # training-only site offsets on the main 12 logits
            names, sidx, grp, off, tgrp, toff = site
            g = grp[b["idx"]]
            o = off[sidx[b["idx"]][:, None].expand_as(g), g, torch.arange(12, device=g.device).expand_as(g)] * (g > 0)
            if toff is not None:                          # 'fluid': + the site-free (finding, thoroughness tercile) offset on silent cells
                tg = tgrp[b["idx"]]
                o = o + toff[tg, torch.arange(12, device=tg.device).expand_as(tg)] * (tg > 0)
            logits = torch.cat([logits[:, :12] + o, logits[:, 12:]], 1)
        if xs_off is not None:                            # 10-18 OAI: source offset on the extra-source rows' main 12 logits
            logits = torch.cat([logits[:, :12] + is_xs[b["idx"]][:, None] * xs_off, logits[:, 12:]], 1)
        if xh_w is not None:                              # xs_head: extra-source rows read out by their own head
            lx = (model.last_z.float() * xh_w).sum(-1)[:, :12] + xh_b[:12]
            logits = torch.cat([torch.where(is_xs[b["idx"]][:, None] > 0, lx, logits[:, :12]), logits[:, 12:]], 1)
        w12 = b["w"][:, :12]
        if cfg["asl_gneg"]:                                   # asymmetric loss (Ben-Baruch et al. 2021) with soft targets
            yy, pr = b["y"][:, :12], torch.sigmoid(logits[:, :12])
            pm = (pr - cfg["asl_m"]).clamp(min=0)              # shifted probability: very easy negatives drop out
            bce = -(yy * F.logsigmoid(logits[:, :12]) + (1 - yy) * pm ** cfg["asl_gneg"] * torch.log((1 - pm).clamp(min=1e-8))) * w12
        else:
            bce = F.binary_cross_entropy_with_logits(logits[:, :12], b["y"][:, :12], reduction="none", pos_weight=pw) * w12
        loss = cfg["bce_weight"] * bce.sum() / w12.sum().clamp(min=1)
        if is_xs is not None:
            if xs_off is not None:
                loss = loss + cfg["xs_off_l2"] * xs_off.pow(2).sum()
            xs_log["l"] += float(bce[is_xs[b["idx"]] > 0].sum().detach())     # OAI share of the main loss (log only)
            xs_log["n"] += float(bce.sum().detach())
        if auc_mem is not None:
            loss = loss + cfg["auc_weight"] * auc_mem.loss(logits[:, :12], b["y"][:, :12], w12)
        if pwm is not None:
            i = b["idx"]
            loss = loss + cfg["pw_weight"] * pwm.loss(pw_last["s"], pw_y[i], pw_w[i], pw_site[i])
        if rpm is not None:
            i = b["idx"]
            rk["s"] = s12
            loss = loss + cfg["pair_weight"] * rpm.loss(s12, rpr["y"][i], rpr["w"][i], rpr["t"][i], rpr["st"][i], rpr["site"][i])
        if am is not None:
            i = b["idx"]
            la = aucm_loss(s12, am["yh"][i], am["wh"][i], am["p"], am["a"], am["b"], am["alpha"], cfg["aucm_margin"])
            am["acc"][0], am["acc"][1] = am["acc"][0] + la.detach(), am["acc"][1] + 1
            loss = loss + cfg["aucm_weight"] * la
        if site is not None:
            loss = loss + cfg["site_l2"] * site[3].pow(2).sum(-1).mean()
            if site[5] is not None:
                loss = loss + cfg["site_l2"] * site[5].pow(2).sum(-1).mean()
        for lo, hi, wt in [(12, 12 + n_attr, cfg["attr_weight"]), (12 + n_attr, 12 + n_attr + n_ext, cfg["ext_weight"]),
                           (12 + n_attr + n_ext, 12 + n_attr + n_ext + n_rec, cfg["head_weight"]),
                           (12 + n_attr + n_ext + n_rec, 12 + n_extra, cfg["aux_weight"])]:
            if hi > lo:
                wa = b["w"][:, lo:hi]
                la = F.binary_cross_entropy_with_logits(logits[:, lo:hi], b["y"][:, lo:hi], reduction="none")
                # mean over the batch's studies x outputs, not over labelled cells only: sparse labels (e.g. image reads on ~16%
                # of studies) otherwise give one study the weight of a whole batch and dominate the clipped gradient (12:50 bug)
                loss = loss + wt * (la * wa).sum() / (wa.shape[0] * wa.shape[1])
        if cfg["rank_weight"]:
            loss = loss + cfg["rank_weight"] * rank_loss(logits[:, :12], b["y"][:, :12], w12, cfg["rank_margin"])
        if n_aux:
            loss = loss + cfg["aux_status"] * F.cross_entropy(aux[:, :12].float().reshape(-1, n_aux), b["y"][:, 12 + n_extra:].long().reshape(-1),
                                                            label_smoothing=cfg["aux_smooth"], ignore_index=-100)
        if cfg["cov_weight"]:                  # coverage head: BCE on real windows of studies with masks, findings with a structure
            m = (b["mask"] & (b["cov"].sum((1, 2)) > 0)[:, None])[..., None] & cov_cols
            lc = F.binary_cross_entropy_with_logits(model.cov_logit.float(), b["cov"], reduction="none")
            loss = loss + cfg["cov_weight"] * (lc * m).sum() / m.sum().clamp(min=1)
        return loss, bce.sum(1) / w12.sum(1).clamp(min=1), logits[:, :12]

    step, rows, t0, killed = 0, [], time.time(), False
    study_loss = np.full(len(tr), np.nan)                 # latest training loss per study (main 12), for hard-example sampling
    oof_all, gold_all = [], []
    for ep in range(cfg["epochs"]):
        model.train()
        if cfg["freeze_bn"]:
            for m in model.backbone.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        sampler.epoch = ep
        sampler.weights = sample_weights(cfg, y, w, study_loss, ep)
        tl, n, t_ep, t_wait = 0.0, 0, time.time(), 0.0
        tw = time.time()
        for b in dl_tr:
            t_wait += time.time() - tw
            b = to_dev(b, dev)
            set_lr(step)
            x = gpu_prep(b["x"], cfg["img"], cfg["interp"], cfg, *model.norm_stats(), mono=cfg["mono"])
            if cfg["aug_mirror"]:
                x = mirror(x, b, cfg["aug_mirror"], *model.norm_stats(), 12 + n_extra)
            if cfg["aug_sdrop"]:                          # series dropout: a study with >= 3 series loses all windows of one series
                for i in torch.nonzero(torch.rand(b["mask"].shape[0], device=dev) < cfg["aug_sdrop"]).flatten():
                    present = b["sid"][i][b["mask"][i]].unique()
                    if len(present) >= 3:
                        b["mask"][i] &= b["sid"][i] != present[torch.randint(len(present), (1,), device=dev)]
            if cfg["aug_bagmix"] or cfg["aug_mixup"]:
                x = mix_batch(x, b, cfg, 12 + n_extra)
            if cfg["aug_wdrop"] > 0:
                keep = torch.rand(b["mask"].shape, device=dev) >= cfg["aug_wdrop"]
                keep[:, 0] = True
                b["mask"] = b["mask"] & keep
            if cfg["channels_last"]:
                x = x.contiguous(memory_format=torch.channels_last)
            loss, per_study, main_logits = compute_loss(x, b)
            study_loss[b["idx"].cpu().numpy()] = per_study.detach().cpu().numpy()
            if auc_mem is not None:
                auc_mem.push(main_logits, b["y"][:, :12], b["w"][:, :12])
            if pwm is not None:
                pwm.push(pw_last["s"], pw_y[b["idx"]], pw_w[b["idx"]], pw_site[b["idx"]])
            if rpm is not None:
                i = b["idx"]
                rpm.push(rk["s"], rpr["y"][i], rpr["w"][i], rpr["t"][i], rpr["st"][i], rpr["site"][i])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["sam"]:
                sam_second_pass(model, lambda: compute_loss(x, b)[0], cfg["sam"])
            if cfg["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if am is not None:
                am["alpha"].data.clamp_(min=0)
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:      # step >= total: the spe_ref schedule ends (all-data port)
                break
            tw = time.time()
        msg = f"ep {ep} loss {tl / max(n, 1):.4f} time {time.time() - t_ep:.0f}s (data wait {t_wait:.0f}s)"
        if am is not None:                                # C214b: the AUC-M term and its auxiliary variables (mean over findings)
            msg += (f" | aucm term {float(am['acc'][0]) / max(am['acc'][1], 1):.4f}, a {am['a'].mean().item():.3f} b {am['b'].mean().item():.3f} "
                    f"alpha {am['alpha'].mean().item():.3f}")
            am["acc"] = [0.0, 0]
        if is_xs is not None:                             # 10-18 OAI model-side log: source offsets + OAI share of the main loss (this epoch)
            msg += (f" | xs_off {[round(v, 3) for v in xs_off.detach().cpu().tolist()]}" if xs_off is not None else " | xs_head")
            msg += f" OAI loss share {xs_log['l'] / max(xs_log['n'], 1e-9):.3f}"
            xs_log.update(l=0.0, n=0.0)
        if sampler.weights is not None:
            sw = sampler.weights
            msg += f" | sampling max/mean {(sw.max() / sw.mean()).item():.2f}"
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        if ((ep + 1) % cfg["eval_every"] == 0 and ep >= cfg["eval_from"]) or last or ep == cfg["kill_epoch"]:
            em = ema.module if ema is not None else model
            te = time.time()
            pv, pa = predict(em, dl_va, cfg, dev, amp, with_aux=True)
            pv = pd.DataFrame(pv, index=va, columns=FINDINGS)
            pg = pd.DataFrame(predict(em, dl_gold, cfg, dev, amp), index=gold, columns=FINDINGS)
            m = evaluate(pd.concat([pv, pg[~pg.index.isin(pv.index)]]))   # fold -1: val == gold-58 -> no duplicate rows (oct14 port; fold >= 0: disjoint, unchanged)
            mv = evaluate(pv)
            row = {"epoch": ep, "trusted": mv["trusted_mean"], "lookat": mv["lookat_mean"], "text": mv["text_mean"], "gold": m.get("gold_mean"),
                   "ece": mv["calibration"]["ece"], "loss": tl / max(n, 1), "time": time.time() - t0}
            rows.append(row)
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            msg += f" | trusted {row['trusted']:.4f} lookat {row['lookat']:.4f} text {row['text']:.4f} gold {row['gold']:.4f} ece {row['ece']:.3f} (eval {time.time() - te:.0f}s)"
            final = mv | {"gold_mean": m.get("gold_mean"), "gold_auc": m.get("gold_auc")}
            if pa is not None:
                sv, st_tr = status_targets(va, cfg["aux_groups"]), status_targets(tr, cfg["aux_groups"])
                major = np.array([np.bincount(st_tr[:, j][st_tr[:, j] >= 0], minlength=n_aux).argmax() for j in range(12)])
                known = sv >= 0
                acc = ((pa.argmax(-1) == sv) & known).sum(0) / known.sum(0)
                acc_major = ((sv == major) & known).sum() / known.sum()
                final["aux_status"] = {"acc": float(acc.mean()), "acc_majority": float(acc_major),
                                       "acc_per_finding": dict(zip(FINDINGS, np.round(acc, 3).tolist()))}
                msg += f" | status acc {acc.mean():.3f} (majority {acc_major:.3f})"
                np.save(out_dir / "oof_aux.npy", pa.astype(np.float16))
            if cfg["save"] and 0 <= cfg["save_from"] <= ep:
                torch.save(em.state_dict(), out_dir / f"model_ep{ep}.pt")
        say(msg)
        if ep == cfg["kill_epoch"] and rows[-1]["trusted"] < cfg["kill_below"]:
            say(f"killed at epoch {ep}: trusted {rows[-1]['trusted']:.4f} < {cfg['kill_below']}")
            killed = True
            break
        if last:
            break
    if cfg["save"]:
        torch.save((ema.module if ema is not None else model).state_dict(), out_dir / "model.pt")
    if site is not None:                                  # learned offsets, for reading (never used at prediction)
        gn = ("none", "denied", "below", "unsettled", "meets", "silent") if cfg["site_mode"] == "fluid" else ("none", "g1", "g2")
        pd.DataFrame(site[3].detach().cpu().numpy().reshape(len(site[0]), -1), index=site[0],
                     columns=[f"{g}|{f}" for g in gn for f in FINDINGS]).to_csv(out_dir / "site_offsets.csv")
        if site[5] is not None:
            pd.DataFrame(site[5].detach().cpu().numpy(), index=["none", "terse", "middle", "thorough"],
                         columns=FINDINGS).to_csv(out_dir / "thoroughness_offsets.csv")
    pd.concat(oof_all).to_parquet(out_dir / "oof.parquet")
    pd.concat(gold_all).to_parquet(out_dir / "gold.parquet")
    json.dump({"final": final, "curve": rows, "cfg": cfg, "fold": fold, "minutes": (time.time() - t0) / 60, "killed": killed}, open(out_dir / "metrics.json", "w"), indent=1, default=float)
    return rows[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.set and a.set[0].startswith("script="):     # queue hand-off to another trainer in this folder (e.g. train_vlm.py), with
        script, rest = a.set[0].split("=", 1)[1], a.set[1:]   # an optional pythonpath=DIR for its private packages
        extra = [kv.split("=", 1)[1] for kv in rest if kv.startswith("pythonpath=")]
        rest = [kv for kv in rest if not kv.startswith("pythonpath=")]
        env = os.environ | ({"PYTHONPATH": os.pathsep.join(extra + [os.environ.get("PYTHONPATH", "")])} if extra else {})
        os.execve(sys.executable, [sys.executable, str(HERE / script), "--name", a.name, "--folds", a.folds, "--set", *rest], env)
    cfg = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown key {k}"
        cfg[k] = parse_value(v)
    for fold in [int(f) for f in a.folds.split(",")]:
        out = HERE.parent / "runs" / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        r = run_fold(cfg, fold, out)
        res = HERE.parent / "runs" / "results.csv"
        pd.DataFrame([{"name": a.name, "fold": fold, **r, "overrides": " ".join(a.set)}]).to_csv(res, mode="a", header=not res.exists(), index=False)


if __name__ == "__main__":
    main()
