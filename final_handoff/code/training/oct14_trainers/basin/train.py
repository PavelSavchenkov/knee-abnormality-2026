"""Basin-code pipeline (copy of experiments/v1/oct10_synovitis/basin/train.py = experiments/v0/oct10_baseline_3rd/src/train.py + image
head): train one fold; score the held-out fold and gold-58 every epoch. oct14_all_data changes: --fold -1 = all-data run (every
non-gold study trains, 'oof' rows = gold-58); --spe_ref N runs the schedule (warmup, cosine, epochs) in units of N optimizer steps
(the fold runs' steps per epoch: equal step count and LR curve); --stop_step S stops after S optimizer steps (evaluated + saved
as the partial epoch's checkpoint), e.g. 14 x spe_ref = the fold runs' e14 stop.

Changes vs the source: --img_w (extra output trained on the image-read synovitis target --img_csv, masked BCE normalised per batch study,
as the main pipeline's ext head), --cls_w (per-class weights of the main 12-output loss); preds.parquet also holds img_Synovitis.

    PYTHONPATH=. python experiments/v1/oct10_synovitis/basin/train.py --name n_ref --fold 1 --seed 1 ...
Writes runs/<name>/fold<k>/: config.json, preds.parquet (all epochs; set = oof | gold | train), metrics.json (curve), [model_e<k>.pt]."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from timm.optim import param_groups_layer_decay

sys.path.insert(0, str(Path(__file__).parent))
from data import AUX_CLASSES, EXP, BagDataset, StudyDataset, aux_status, build_targets, load_folds, load_meta, select_slots, worker_init  # noqa: E402
from model import SWAP, BagModel, StudyModel, gpu_views, init_output_bias, mirror, mirror_bag, view_fov  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from common.paths import FINDINGS  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="s320")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--slots", default="sag_fs,cor_fs,ax_fs,sag_nofs,cor_nofs")
    ap.add_argument("--mode", default="grid", help="grid: T triplets per slot; bag: k random windows from all slots")
    ap.add_argument("--k", type=int, default=12, help="bag mode: windows per study in training")
    ap.add_argument("--k_eval", type=int, default=24, help="bag mode: max evenly spaced windows per slot at eval")
    ap.add_argument("--strat", type=int, default=0, help="bag mode: equal share of the k windows per slot")
    ap.add_argument("--bias_init", type=int, default=0, help="output bias = logit(train prevalence)")
    ap.add_argument("--slot_aux", type=float, default=0.0, help="weight of per-slot deep-supervision BCE (grid mode)")
    ap.add_argument("--asym", type=float, default=0.0, help="weight of 2 extra outputs on (MM - LM + 1) / 2 and (MOA - LOA + 1) / 2")
    ap.add_argument("--asym_pf", type=int, default=0, help="with --asym: 2 more outputs on (PF - MOA + 1) / 2 and (PF - LOA + 1) / 2")
    ap.add_argument("--ax_rows", type=int, default=0, help="comp mode: axial slots cropped into anterior / posterior rows")
    ap.add_argument("--pos_weight", type=int, default=0, help="BCE pos_weight = clip((1 - prev) / prev, 1, 10)")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--gshift", type=int, default=0, help="training: random global offset of the triplet grid (+-half stride)")
    ap.add_argument("--sample", default="grid", help="grid: T fixed triplets over [lo, hi]; pos: stratified random centres over the "
                    "whole stack in training, T_eval dense overlapping centres at eval, positions passed to the model")
    ap.add_argument("--T_eval", type=int, default=24)
    ap.add_argument("--pos_window", type=int, default=0, help="sample pos: restrict centres to [lo, hi] of the stack")
    ap.add_argument("--uniform", type=int, default=0, help="sample pos: uniform random centres instead of stratified")
    ap.add_argument("--sub", type=float, default=0.0, help="training: prob. per slot of using another same-knee series of the slot's plane / FS")
    ap.add_argument("--wemb", type=int, default=0, help="series weighting embedding added to its tokens (comp mode)")
    ap.add_argument("--skip_missing", type=int, default=0, help="comp mode: do not encode missing series (bucketed shapes)")
    ap.add_argument("--lo", type=float, default=0.1)
    ap.add_argument("--hi", type=float, default=0.9)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fov", type=float, default=1.0)
    ap.add_argument("--men_reg", type=int, default=0, help="binned comp head: MM / LM logits blind to the opposite side; 1 also blocks the central sagittal third, 2 keeps it")
    ap.add_argument("--fov_sag", type=float, default=0.0, help="per-plane fov (share of the cached FOV kept); 0 = --fov")
    ap.add_argument("--fov_cor", type=float, default=0.0)
    ap.add_argument("--fov_ax", type=float, default=0.0)
    ap.add_argument("--cor_yshift", type=float, default=0.0, help="coronal view centre moved down by this share of the height")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--layer_decay", type=float, default=1.0)
    ap.add_argument("--bn_single", type=int, default=0, help="comp mode: one encoder call for whole images + zero-padded crops (BatchNorm encoders)")
    ap.add_argument("--stage_decay", type=float, default=1.0, help="ConvNeXt per-stage LR decay: last stage x1, earlier stages x d^k, stem x d^4")
    ap.add_argument("--cutout_max", type=float, default=0.3, help="max side of the cutout box (share of the image; min 0.1)")
    ap.add_argument("--cutout_fill", default="zero", help="cutout fill: zero or noise (noise at the image mean level)")
    ap.add_argument("--aux", type=float, default=0.0, help="weight of the auxiliary v1-status head (0 = off)")
    ap.add_argument("--warmup", type=float, default=0.5)
    ap.add_argument("--lp_epochs", type=int, default=0, help="LP-FT: first epochs train the head only (encoder frozen, LR 0); encoder warm-up starts after")
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--head", default="cls")
    ap.add_argument("--pool", default="avg", help="CNN global pool: avg / max / avgmax")
    ap.add_argument("--grid", type=int, default=1, help="G x G spatial tokens per image (grid mode)")
    ap.add_argument("--comp", type=int, default=0, help="medial / lateral column crops for coronal and axial slots")
    ap.add_argument("--crop_frac", type=float, default=0.6)
    ap.add_argument("--bin_all", type=int, default=0, help="binned head: cor / ax slots in 3 slice-position bins too")
    ap.add_argument("--bins_sym", type=int, default=0, help="binned head: symmetric position bins, (3, 2, 3) for T = 8")
    ap.add_argument("--bin_drop", type=float, default=0.3, help="binned head dropout")
    ap.add_argument("--bin_max", type=int, default=0, help="binned head: concat(mean, max) over triplets per bin")
    ap.add_argument("--sag_size", type=int, default=0, help="comp mode: whole sagittal images resized to this (crops keep --size)")
    ap.add_argument("--seq", default="tf", help="tf: transformer over all tokens; gru: bidirectional GRU along slices per series")
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--ckpt_stages", default="", help="ConvNeXt: checkpoint only these stages (e.g. '2' = the 27-block stage of -S); overrides --grad_ckpt")
    ap.add_argument("--mirror", type=float, default=0.0, help="prob. of a left-right mirror with medial/lateral labels swapped")
    ap.add_argument("--slot_drop", type=float, default=0.0, help="prob. of hiding each series in training (>= 1 kept)")
    ap.add_argument("--rot", type=float, default=10.0)
    ap.add_argument("--std_norm", type=int, default=0, help="per-series tissue-anchored intensity standardisation (train + eval)")
    ap.add_argument("--aug_per_image", type=int, default=0, help="draw affine / intensity augmentation per image instead of per series")
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise sigma ~ U(0, noise) per series (train)")
    ap.add_argument("--res_aug", type=float, default=0.0, help="prob. per series of down/up-sampling by f in [0.5, 1] (train)")
    ap.add_argument("--tdrop", type=float, default=0.0, help="prob. per triplet of blanking it (train, grid mode)")
    ap.add_argument("--int_s", type=float, default=1.0, help="strength multiplier of gamma / contrast / brightness aug")
    ap.add_argument("--cutout", type=float, default=0.0, help="prob. per image of one erased rectangle (10-30% side)")
    ap.add_argument("--bias", type=float, default=0.0, help="smooth multiplicative bias field strength per series")
    ap.add_argument("--blur", type=float, default=0.0, help="per series blur / sharpen strength |k| ~ U(0, blur)")
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--shear", type=float, default=0.0, help="train: max affine shear factor")
    ap.add_argument("--aniso", type=float, default=0.0, help="train: max |log aspect ratio| of the affine")
    ap.add_argument("--elastic", type=float, default=0.0, help="train: amplitude of a smooth 4x4 displacement field")
    ap.add_argument("--chan_jit", type=float, default=0.0, help="train: independent gamma jitter per slice of a triplet")
    ap.add_argument("--ghost", type=float, default=0.0, help="train: prob. of an MRI ghosting artefact")
    ap.add_argument("--gibbs", type=float, default=0.0, help="train: prob. of k-space truncation ringing")
    ap.add_argument("--slab", type=int, default=0, help="grid: thin series (< 2.5 mm) as ~3.3 mm slabs (channel = mean of slab_k slices), train and eval")
    ap.add_argument("--tkeep", type=int, default=0, help="train, grid: keep a random tkeep of the T triplets per slot each step (token-masked bag subsampling)")
    ap.add_argument("--mkeep", type=int, default=0, help="train, grid: finding-aware bag: keep mkeep of T triplets via the token mask ...")
    ap.add_argument("--mkeep_p", type=float, default=0.0, help="... on this share of training studies")
    ap.add_argument("--focal", default="PF,LM,MM", help="finding codes whose loss is skipped on bagged studies (--mkeep)")
    ap.add_argument("--wd_head", type=float, default=-1.0, help="weight decay of the head (readout) params; -1 = --wd")
    ap.add_argument("--slice_pv", type=float, default=0.0, help="train, grid: prob. per triplet of partial-volume neighbours (mean with next slice out)")
    ap.add_argument("--slice_dup", type=float, default=0.0, help="train, grid: prob. per triplet of neighbours = centre slice")
    ap.add_argument("--mono", type=int, default=0, help="centre slice in all 3 channels (train + eval): input contract of grayscale-pretrained encoders, e.g. RadImageNet")
    ap.add_argument("--ra_n", type=int, default=0, help="train: RandAugment ops per draw unit (series / triplet), from model.RA_OPS")
    ap.add_argument("--ra_m", type=float, default=1.0, help="train: RandAugment strength multiplier (strength ~ U(0, ra_m) x cap)")
    ap.add_argument("--shift", type=float, default=0.06)
    ap.add_argument("--target", default="v1")
    ap.add_argument("--nm", default="0.1")
    ap.add_argument("--nm_weight", type=float, default=1.0)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--reads", default="", help="comma list of findings taken from the image reads")
    ap.add_argument("--reads_weight", type=float, default=1.0)
    ap.add_argument("--reads_scale", type=float, default=1.0, help="multiply read probabilities (clip 0.02-0.95)")
    ap.add_argument("--unset_mode", default="", help="unsettled PF OA / effusion variants U1 / U2 / U3 / U4 / U5 (data.unsettled_targets)")
    ap.add_argument("--unset_mask", default="", help="comma list of finding codes (MOA,LOA,PF,MM,LM,ACL,...) whose v1 'unsettled' cells get weight 0")
    ap.add_argument("--syn_eff", type=int, default=0, help="synovitis targets of report-silent cells from effusion status")
    ap.add_argument("--img_w", type=float, default=0.0, help="weight of an extra output on the image-read synovitis target (0 = off)")
    ap.add_argument("--img_csv", default="data/img_syn_blockrank.csv", help="image target: StudyInstanceUID, Synovitis (path under EXP)")
    ap.add_argument("--cls_w", default="", help="per-class weights of the main loss, e.g. ALL:0,SYN:1,EFF:1 (ABBR keys; normaliser unchanged)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save", default="", help="'last' or 'all' to keep checkpoints")
    ap.add_argument("--eval_raw", type=int, default=0, help="triage: with --ema, also evaluate the raw (non-EMA) weights at every eval (raw_* metrics, preds_raw.parquet)")
    ap.add_argument("--rare_boost", type=float, default=0.0, help="sampler: studies with a positive rare finding drawn (1 + x) times as often")
    ap.add_argument("--rare", default="MCL,FRA,CON,LOA", help="rare findings for --rare_boost (ABBR keys)")
    ap.add_argument("--freeze_bn", type=int, default=0, help="keep BatchNorm layers in eval mode during training (no bn_single needed)")
    ap.add_argument("--sam", type=float, default=0.0, help="SAM rho (0 = off); doubles the cost per step")
    ap.add_argument("--sched", default="cos", help="cos: cosine over --epochs after warmup; const: constant LR after warmup (any checkpoint is a valid stop / resume point)")
    ap.add_argument("--resume", default="", help="RUN_DIR:EPOCH (a fold dir saved with --save): continue from its raw weights, EMA weights and optimizer state")
    ap.add_argument("--spe_ref", type=int, default=0, help="schedule unit in optimizer steps (0 = this run's steps per epoch)")
    ap.add_argument("--stop_step", type=int, default=0, help="stop after this many optimizer steps (evaluate + save there); 0 = off")
    ap.add_argument("--stop_epoch", type=int, default=0, help="triage: stop after evaluating this epoch (prefix screen of the --epochs schedule)")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_from", type=int, default=1, help="first epoch to evaluate (the last epoch is always evaluated)")
    ap.add_argument("--limit", type=int, default=0, help="debug: train studies cap")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights used for evaluation (0 = off)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--deterministic", type=int, default=1)
    a = ap.parse_args(argv)
    assert not (a.layers == 0 and a.head == "cls"), "CLS head needs a transformer (layers > 0)"
    assert a.head != "binned" or a.mode == "grid", "binned head: grid mode"
    assert a.sample == "grid" or (a.comp and a.head in ("binned", "battn", "gattn", "hattn")), "position-aware sampling: comp + binned / battn / gattn head"
    assert not a.wemb or a.comp, "weighting embedding: comp mode"
    assert a.mode == "grid" or a.head in ("gattn", "attn"), "bag mode needs --head gattn or attn"
    return a


def collapse(prob):
    """Finding-specificity diagnostics of OOF predictions: top-2 PC variance share of z-scored logits, corr(MM, LM),
    corr(MOA, LOA). v1 soft targets give 0.39 / 0.12 / 0.49; prediction collapse pushes them towards 1."""
    z = np.log(prob.clip(1e-6, 1 - 1e-6) / (1 - prob.clip(1e-6, 1 - 1e-6)))
    z = (z - z.mean()) / z.std()
    ev = np.sort(np.linalg.eigvalsh(np.cov(z.to_numpy().T)))[::-1]
    return {"pc12": float(ev[:2].sum() / ev.sum()), "c_mlm": float(z["Medial Meniscus"].corr(z["Lateral Meniscus"])),
            "c_moa_loa": float(z["Medial OA"].corr(z["Lateral OA"]))}


def scores(pred):
    e = evaluate(pred)
    text_core = np.nanmean([v for k, v in e["text_auc"].items() if k not in ("Synovitis", "MCL")])
    gold11 = np.nanmean([v for k, v in e.get("gold_auc", {}).items() if k != "Synovitis"]) if "gold_auc" in e else np.nan
    return {"trusted": e["trusted_mean"], "lookat": e["lookat_mean"], "text": e["text_mean"], "text_core": float(text_core),
            "gold": e.get("gold_mean", np.nan), "gold11": float(gold11)}, e


def mono(v, on):
    """(B, S, T, 3, H, W) views -> the centre slice of each triplet copied to all 3 channels when on."""
    return v[:, :, :, 1:2].expand_as(v).contiguous() if on else v


@torch.no_grad()
def predict(model, loader, a):
    model.eval()
    out = []
    for b in loader:
        x, m = b["x"].cuda(non_blocking=True), b["mask"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.mode == "bag":
                v = gpu_views(x[:, :, None], a.size, False, fov=a.fov)[:, :, 0]
                logit = model(v, b["slot"].cuda(non_blocking=True), b["pos"].cuda(non_blocking=True), m)
            else:
                pt = (b["pos"].cuda(non_blocking=True), b["tmask"].cuda(non_blocking=True)) if a.sample == "pos" else (None, None)
                fv, ys = view_fov(vars(a))
                logit = model(mono(gpu_views(x, a.size, False, fov=fv, yshift=ys, norm=bool(a.std_norm)), a.mono), m, *pt,
                              widx=b["widx"].cuda(non_blocking=True) if a.wemb else None)
        out.append(logit.float().cpu())
    return torch.cat(out).numpy()


ABBR = {"ACL": "ACL", "MCL": "MCL", "MM": "Medial Meniscus", "LM": "Lateral Meniscus", "MOA": "Medial OA", "LOA": "Lateral OA",
        "PF": "PF OA", "EFF": "Effusion", "SYN": "Synovitis", "BAK": "Baker's", "CON": "Contusion", "FRA": "Fracture"}


def sam_second_pass(model, loss_fn, rho):
    """SAM, second half of a step (copied from experiments/v0/oct7_baseline/train.py): with the gradient g at w in .grad, move to
    w + rho g / |g|, take the gradient there (BatchNorm running statistics frozen for this pass), move back; the optimiser then
    steps with that gradient."""
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


def main(a):
    torch.manual_seed(a.seed), np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = not a.deterministic
    if a.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = EXP / "runs" / a.name / f"fold{a.fold}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)

    meta = load_meta(a.cache)
    slots = a.slots.split(",")
    table = select_slots(meta, slots)
    Y, W = build_targets(a.target, nm=None if a.nm == "mask" else float(a.nm), nm_weight=a.nm_weight, hard=a.hard,
                         reads=[r for r in a.reads.split(",") if r], reads_weight=a.reads_weight, syn_eff=bool(a.syn_eff),
                         reads_scale=a.reads_scale, unset_mask=[ABBR[c] for c in a.unset_mask.split(",") if c], unset_mode=a.unset_mode)
    if a.img_w:  # image-read target as a 13th column of Y / W (W = 1 where read)
        img = pd.read_csv(EXP / a.img_csv).set_index("StudyInstanceUID").Synovitis.dropna()
        Y, W = Y.copy(), W.copy()
        Y["img"], W["img"] = img.reindex(Y.index).fillna(0.0), img.reindex(Y.index).notna().astype(float)
    folds = load_folds()
    tr = folds.index[folds.fold != a.fold]
    va = folds.index[folds.fold == a.fold]
    gold = gold_labels().index
    if a.fold < 0:  # all-data run (fold -1): every non-gold study trains; the 'oof' rows are gold-58
        va = gold
    if a.limit:
        tr = tr[: a.limit]
    tr_probe = list(np.random.default_rng(0).choice(tr, min(400, len(tr)), replace=False))
    aux = aux_status() if a.aux else None
    if a.mode == "bag":
        kw = dict(slot_table=table, meta=meta, k=a.k, k_eval=a.k_eval, cache=a.cache)
        ds_tr = BagDataset(tr, Y=Y, W=W, train=True, aux=aux, seed=a.seed, strat=bool(a.strat), **kw)
        ds_ev = BagDataset(list(va) + list(gold) + tr_probe, Y=None, W=None, train=False, **kw)
    else:
        kw = dict(slot_table=table, meta=meta, T=a.T, lo=a.lo, hi=a.hi, cache=a.cache, sample=a.sample, T_eval=a.T_eval, slab=bool(a.slab),
                  pos_window=bool(a.pos_window))
        ds_tr = StudyDataset(tr, Y=Y, W=W, train=True, aux=aux, seed=a.seed, shift=bool(a.gshift), uniform=bool(a.uniform), sub=a.sub,
                             slice_pv=a.slice_pv, slice_dup=a.slice_dup, tkeep=a.tkeep, mkeep=a.mkeep, mkeep_p=a.mkeep_p, **kw)
        assert not (a.tkeep and a.bins_sym), "tkeep strata follow the non-symmetric bins"
        ds_ev = StudyDataset(list(va) + list(gold) + tr_probe, Y=None, W=None, train=False, **kw)
    prev = ((Y * W)[FINDINGS].reindex(tr).sum() / W[FINDINGS].reindex(tr).sum()).to_numpy()
    pos_weight = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device="cuda") if a.pos_weight else None
    dl_kw = dict(num_workers=a.workers, pin_memory=True, worker_init_fn=worker_init, prefetch_factor=2)
    g = torch.Generator()
    g.manual_seed(a.seed)
    dl_ev = torch.utils.data.DataLoader(ds_ev, batch_size=4 if a.mode == "bag" or a.sample == "pos" else a.bs * 2, shuffle=False, **dl_kw)

    n_aux = len(set(AUX_CLASSES.values())) if a.aux else 0
    n_asym = (2 if a.asym else 0) + (2 if a.asym and a.asym_pf else 0)
    n_out = 12 + n_asym + (1 if a.img_w else 0)
    img_col = 12 + n_asym  # the image head's output column
    cw = torch.ones(12, device="cuda")
    kv = dict(t.split(":") for t in a.cls_w.split(",") if t)
    cw[:] = float(kv.pop("ALL", 1))
    for k, v in kv.items():
        cw[FINDINGS.index(ABBR[k])] = float(v)
    if a.mode == "bag":
        model = BagModel(a.backbone, len(slots), n_out=n_out, d=a.d, layers=a.layers, drop_path=a.drop_path, head=a.head, n_aux=n_aux, pool=a.pool,
                         grad_ckpt=bool(a.grad_ckpt)).cuda()
    else:
        model = StudyModel(a.backbone, len(slots), a.T, n_out=n_out, d=a.d, layers=a.layers, drop_path=a.drop_path, head=a.head,
                           grad_ckpt=bool(a.grad_ckpt), n_aux=n_aux, seq=a.seq, pool=a.pool, slot_aux=bool(a.slot_aux),
                           grid=a.grid, comp=[not sl.startswith("sag") for sl in slots] if a.comp else None, crop_frac=a.crop_frac,
                           sag_size=a.sag_size, img_size=a.size, sag=[sl.startswith("sag") for sl in slots],
                           bin_all=bool(a.bin_all), rows=[sl.startswith("ax") for sl in slots] if a.comp and a.ax_rows else None,
                           bins_sym=bool(a.bins_sym), bin_drop=a.bin_drop, bin_max=bool(a.bin_max),
                           posenc="fourier" if a.sample == "pos" else "index", wemb=bool(a.wemb),
                           skip_missing=bool(a.skip_missing), men_reg=a.men_reg, bn_single=bool(a.bn_single)).cuda()
    if a.bias_init:
        init_output_bias(model, np.r_[prev, [0.5] * (n_out - 12)])
    if a.ckpt_stages:  # selective activation checkpointing (before compile)
        model.enc.set_grad_checkpointing(False)
        for i in a.ckpt_stages.split(","):
            model.enc.stages[int(i)].grad_checkpointing = True
    model.enc.to(memory_format=torch.channels_last)
    enc_ids = {id(p) for p in model.enc.parameters()}
    head_named = [(n, p) for n, p in model.named_parameters() if id(p) not in enc_ids]
    # timm groups: layer-wise LR scale (1.0 everywhere when layer_decay = 1); no weight decay on 1D params / embeddings
    if a.stage_decay < 1:  # per-stage decay (ConvNeXt names: stem.*, stages.<k>.*, head.*); no wd on 1-D params
        def depth(n):
            return 4 if n.startswith("stem") else (3 - int(n.split(".")[1])) if n.startswith("stages.") else 0
        buckets = {}
        for n, p_ in model.enc.named_parameters():
            buckets.setdefault((depth(n), p_.ndim <= 1), []).append(p_)
        enc_groups = [{"params": ps, "lr": a.lr * a.stage_decay ** k, "weight_decay": 0.0 if nd else a.wd} for (k, nd), ps in sorted(buckets.items())]
        print("stage decay groups:", [(k, nd, len(ps), round(a.stage_decay ** k, 3)) for (k, nd), ps in sorted(buckets.items())], flush=True)
    else:
        enc_groups = [dict(g, lr=a.lr * g.pop("lr_scale")) for g in param_groups_layer_decay(model.enc, a.wd, layer_decay=a.layer_decay)]
    head_decay = [p for n, p in head_named if p.ndim > 1 and not n.endswith("_emb") and n not in ("cls", "q")]
    head_nodecay = [p for n, p in head_named if not (p.ndim > 1 and not n.endswith("_emb") and n not in ("cls", "q"))]
    opt = torch.optim.AdamW(enc_groups + [{"params": head_decay, "lr": a.lr_head, "weight_decay": a.wd if a.wd_head < 0 else a.wd_head},
                                          {"params": head_nodecay, "lr": a.lr_head, "weight_decay": 0.0}])
    if a.compile:
        torch._dynamo.config.cache_size_limit = 64  # skip_missing / eval shapes
        if getattr(model, "fmap", None) is not None:
            model.fmap = torch.compile(model.fmap)
        else:
            model.enc = torch.compile(model.enc)
    focal_idx = torch.tensor([FINDINGS.index(ABBR[c]) for c in a.focal.split(",") if c], device="cuda")
    steps_per_epoch = a.spe_ref or len(ds_tr) // a.bs // a.accum  # spe_ref: schedule in the fold runs' epoch units (equal steps)
    total, warm = steps_per_epoch * a.epochs, int(steps_per_epoch * a.warmup)
    stop_at = a.stop_step or total
    lp = steps_per_epoch * a.lp_epochs
    cos = (lambda s: 1.0) if a.sched == "const" else (lambda s: 0.5 * (1 + math.cos(math.pi * min(1, s / total))))  # noqa: E731
    warm_h = int(steps_per_epoch * 0.3) if lp else warm  # LP-FT: the head is at full LR during the probe epoch(s)
    head_l = lambda s: min(1, (s + 1) / max(1, warm_h)) * cos(s)  # noqa: E731
    enc_l = lambda s: 0.0 if s < lp else min(1, (s - lp + 1) / max(1, warm)) * cos(s)  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [enc_l] * len(enc_groups) + [head_l, head_l])

    sag = torch.tensor([sl.startswith("sag") for sl in slots], device="cuda")
    params = [p.data for p in model.parameters()] + [b for b in model.buffers() if b.dtype.is_floating_point]  # deduplicated
    ema = [p.detach().clone() for p in params] if a.ema else None
    step = 0
    curve, preds, raw_preds = [], [], []
    start_ep = 0
    if a.resume:  # raw weights -> model, EMA weights -> ema, optimizer + step; earlier curve / predictions carried over
        rdir, rep = a.resume.rsplit(":", 1)
        rdir, rep = Path(rdir), int(rep)
        if ema is not None:
            model.load_state_dict(torch.load(rdir / f"model_e{rep}.pt", map_location="cuda"))
            ema = [p.detach().clone() for p in params]
        model.load_state_dict(torch.load(rdir / (f"model_raw_e{rep}.pt" if ema is not None else f"model_e{rep}.pt"), map_location="cuda"))
        st = torch.load(rdir / f"opt_e{rep}.pt", map_location="cuda")
        opt.load_state_dict(st["opt"])
        step, start_ep = st["step"], rep
        sched.last_epoch = step - 1
        sched.step()
        curve = [r for r in json.load(open(rdir / "metrics.json"))["curve"] if r["epoch"] <= rep]
        old_p = pd.read_parquet(rdir / "preds.parquet")
        preds = [old_p[old_p.epoch <= rep]]
        if (rdir / "preds_raw.parquet").exists():
            old_r = pd.read_parquet(rdir / "preds_raw.parquet")
            raw_preds = [old_r[old_r.epoch <= rep]]
        print(f"resumed from {rdir} epoch {rep} (step {step}, lr {sched.get_last_lr()[0]:.2e})", flush=True)
    print(f"train {len(ds_tr)} eval {len(ds_ev)} steps/epoch {len(ds_tr) // a.bs // a.accum} (schedule unit {steps_per_epoch}), total {total}, stop at {stop_at}", flush=True)
    for ep in range(start_ep, a.epochs):
        if a.lp_epochs and ep in (0, a.lp_epochs):  # LP-FT: no encoder gradients while it is frozen
            for p in model.enc.parameters():
                p.requires_grad_(ep >= a.lp_epochs)
        model.train()
        if a.freeze_bn:  # BatchNorm layers in eval mode (pretrained running statistics), so BN encoders can use the normal crop path
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        ds_tr.epoch = ep
        if a.rare_boost:  # positive-enriched epochs: studies with a rare positive drawn (1 + rare_boost) x as often, same epoch size
            rare = [FINDINGS.index(ABBR[c]) for c in a.rare.split(",")]
            wts = 1 + a.rare_boost * ((ds_tr.Y[:, rare] * (ds_tr.W[:, rare] > 0)) >= 0.5).any(1)
            smp = torch.utils.data.WeightedRandomSampler(torch.as_tensor(wts, dtype=torch.double), len(ds_tr), replacement=True, generator=g)
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, sampler=smp, drop_last=True, **dl_kw)
        else:
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, shuffle=True, drop_last=True, generator=g, **dl_kw)
        t0, tl, nb, t_data, gn = time.time(), torch.zeros((), device="cuda"), 0, 0.0, []  # loss / grad norms stay on the GPU (no sync per step)
        opt.zero_grad(set_to_none=True)
        tq = time.time()
        for b in dl_tr:
            t_data += time.time() - tq
            x, m = b["x"].cuda(non_blocking=True), b["mask"].cuda(non_blocking=True)
            y, w = b["y"].cuda(non_blocking=True), b["w"].cuda(non_blocking=True)
            y, w, yi, wi = y[:, :12], w[:, :12], y[:, 12:], w[:, 12:]  # main 12 | image target (mirror-invariant)
            if "bagged" in b:  # finding-aware bag: no focal-finding loss on studies whose triplets were subsampled
                w = w.clone()
                w[b["bagged"].cuda()[:, None] & torch.isin(torch.arange(w.shape[1], device=w.device), focal_idx)[None]] = 0
            if a.slot_drop:
                drop = torch.rand(m.shape, device=m.device) < a.slot_drop
                if "bagged" in b:  # finding-aware bag: series dropout only on bagged studies (their focal loss is off)
                    drop = drop & b["bagged"].cuda()[:, None]
                keep = m & ~drop
                m = torch.where(keep.any(1, keepdim=True), keep, m)
            if a.mode == "bag":
                slot_b, pos_b = b["slot"].cuda(non_blocking=True), b["pos"].cuda(non_blocking=True)
            pt = (b["pos"].cuda(non_blocking=True), b["tmask"].cuda(non_blocking=True)) if a.sample == "pos" else \
                (None, b["tmask"].cuda(non_blocking=True) if "tmask" in b else None)
            if a.mirror:
                assert not a.tkeep, "mirror with triplet subsets would need reversed tidx"
                flip = torch.rand(len(x), device=x.device) < a.mirror
                if a.mode == "bag":
                    x, pos_b = mirror_bag(x, pos_b, slot_b, sag, flip)
                else:
                    x = mirror(x, sag, flip)
                    if a.sample == "pos":  # mirror() reversed the sagittal token order: reverse positions / masks too, p -> 1 - p
                        fs = flip[:, None] & sag[None]
                        pos_t, tm_t = pt[0].clone(), pt[1].clone()
                        pos_t[fs] = 1 - pt[0][fs].flip(-1)
                        tm_t[fs] = pt[1][fs].flip(-1)
                        pt = (pos_t, tm_t)
                y = torch.where(flip[:, None], y[:, SWAP], y)
                w = torch.where(flip[:, None], w[:, SWAP], w)
                w[flip, 1] = 0.0  # MCL becomes the (unlabelled) LCL
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.mode == "bag":
                    v = gpu_views(x[:, :, None], a.size, True, fov=a.fov, rot=a.rot, scale=a.scale, shift=a.shift)[:, :, 0]
                    fwd = lambda: model(v, slot_b, pos_b, m)  # noqa: E731
                else:
                    if a.tdrop:
                        x = x * (torch.rand(x.shape[:3], device=x.device) >= a.tdrop).to(x.dtype)[..., None, None, None]
                    fv, ys = view_fov(vars(a))
                    v = gpu_views(x, a.size, True, fov=fv, yshift=ys, rot=a.rot, scale=a.scale, shift=a.shift,
                                            noise=a.noise, res_aug=a.res_aug, int_s=a.int_s, cutout=a.cutout, bias=a.bias, blur=a.blur,
                                            norm=bool(a.std_norm), per_image=bool(a.aug_per_image), shear=a.shear, aniso=a.aniso,
                                            elastic=a.elastic, chan_jit=a.chan_jit, ghost=a.ghost, gibbs=a.gibbs, ra_n=a.ra_n, ra_m=a.ra_m,
                                            cutout_max=a.cutout_max, cutout_fill=a.cutout_fill)
                    v = mono(v, a.mono)
                    widx_b = b["widx"].cuda(non_blocking=True) if a.wemb else None
                    tidx_b = b["tidx"].cuda(non_blocking=True) if "tidx" in b else None
                    fwd = lambda: model(v, m, *pt, widx=widx_b, tidx=tidx_b)  # noqa: E731

            def loss_of():  # forward + loss on the same augmented views (called twice per step with SAM)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = fwd()
                extra = {}
                if isinstance(logit, tuple):
                    logit, extra = logit
                loss = (F.binary_cross_entropy_with_logits(logit[:, :12].float(), y, reduction="none", pos_weight=pos_weight) * w * cw).sum() / w.sum().clamp(min=1)
                if a.img_w:  # mean over the batch's studies, not over read studies (as the main pipeline's ext head)
                    li = F.binary_cross_entropy_with_logits(logit[:, img_col:img_col + 1].float(), yi, reduction="none")
                    loss = loss + a.img_w * (li * wi).sum() / len(yi)
                if a.asym:  # medial-vs-lateral contrast targets (FINDINGS: 2 MM, 3 LM, 4 MOA, 5 LOA)
                    pairs = [(2, 3), (4, 5)] + ([(6, 4), (6, 5)] if a.asym_pf else [])  # (MM, LM), (MOA, LOA), (PF, MOA), (PF, LOA)
                    d = torch.stack([(y[:, i] - y[:, j] + 1) / 2 for i, j in pairs], 1)
                    dw = torch.stack([torch.minimum(w[:, i], w[:, j]) for i, j in pairs], 1)
                    loss = loss + a.asym * (F.binary_cross_entropy_with_logits(logit[:, 12:12 + len(pairs)].float(), d, reduction="none") * dw).sum() / dw.sum().clamp(min=1)
                if "aux" in extra:
                    tgt = b["aux"].cuda(non_blocking=True)
                    if a.mirror:
                        tgt = torch.where(flip[:, None], tgt[:, SWAP], tgt)
                        tgt[flip, 1] = -1
                    loss = loss + a.aux * F.cross_entropy(extra["aux"][:, :12].float().flatten(0, 1), tgt.flatten(), ignore_index=-1)
                if "slot" in extra:  # per-slot BCE against the study targets, valid slots only
                    so = extra["slot"][..., :12].float()
                    sl = F.binary_cross_entropy_with_logits(so, y[:, None].expand_as(so), reduction="none")
                    sw = w[:, None] * m[..., None].float()
                    loss = loss + a.slot_aux * (sl * sw).sum() / sw.sum().clamp(min=1)
                return loss

            loss = loss_of()
            (loss / a.accum).backward()
            if a.sam:  # sharpness-aware minimisation: gradient at w + rho g / |g|, same views (Foret et al. 2021)
                assert a.accum == 1, "SAM with gradient accumulation is not implemented"
                sam_second_pass(model, loss_of, a.sam)
            nb += 1
            tl += loss.detach()
            if nb % a.accum == 0:
                gn.append(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if ema is not None:
                    d = min(a.ema, (1 + step) / (10 + step))
                    torch._foreach_lerp_(ema, params, 1 - d)
                if step >= stop_at:
                    break
            tq = time.time()
        t_train = time.time() - t0
        tl, gn = tl.item(), torch.stack(gn).float().cpu().numpy() if gn else np.zeros(1)
        last = ep == a.epochs - 1 or ep + 1 == a.stop_epoch or step >= stop_at
        if not last and (ep + 1 < a.eval_from or (ep + 1) % a.eval_every):
            print(json.dumps({"epoch": ep + 1, "loss": round(tl / nb, 4), "t_train": round(t_train)}), flush=True)
            continue
        if ema is not None:  # evaluate (and save) the EMA weights
            backup = [p.detach().clone() for p in params]
            with torch.no_grad():
                torch._foreach_copy_(params, ema)
        p = predict(model, dl_ev, a)
        prob = pd.DataFrame(1 / (1 + np.exp(-p[:, :12])), index=ds_ev.studies, columns=FINDINGS)
        if a.img_w:
            prob["img_Synovitis"] = 1 / (1 + np.exp(-p[:, img_col]))
        prob["set"] = ["oof"] * len(va) + ["gold"] * len(gold) + ["train"] * len(tr_probe)
        prob["epoch"] = ep + 1
        preds.append(prob)
        sc_oof, _ = scores(prob[prob.set == "oof"][FINDINGS])
        sc_gold, _ = scores(prob[prob.set == "gold"][FINDINGS])
        sc_tr, _ = scores(prob[prob.set == "train"][FINDINGS])
        row = {"epoch": ep + 1, "loss": tl / nb, "trusted": sc_oof["trusted"], "lookat": sc_oof["lookat"], "text": sc_oof["text"],
               "text_core": sc_oof["text_core"], "gold": sc_gold["gold"], "gold11": sc_gold["gold11"],
               "train_trusted": sc_tr["trusted"], "train_lookat": sc_tr["lookat"], **collapse(prob[prob.set == "oof"][FINDINGS]),
               "gnorm_mean": float(np.mean(gn)), "gnorm_max": float(np.max(gn)), "t_train": round(t_train), "t_data_wait": round(t_data), "t_eval": round(time.time() - t0 - t_train)}
        curve.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if a.save == "all" or (a.save == "last" and last):
            torch.save(model.state_dict(), out / f"model_e{ep + 1}.pt")
            torch.save({"opt": opt.state_dict(), "step": step}, out / f"opt_e{ep + 1}.pt")  # for --resume
        if ema is not None:
            with torch.no_grad():
                torch._foreach_copy_(params, backup)
            if a.eval_raw:  # the same evaluation on the raw weights
                pr = predict(model, dl_ev, a)
                prob_r = pd.DataFrame(1 / (1 + np.exp(-pr[:, :12])), index=ds_ev.studies, columns=FINDINGS)
                if a.img_w:
                    prob_r["img_Synovitis"] = 1 / (1 + np.exp(-pr[:, img_col]))
                prob_r["set"], prob_r["epoch"] = prob["set"].to_numpy(), ep + 1
                raw_preds.append(prob_r)
                so, _ = scores(prob_r[prob_r.set == "oof"][FINDINGS])
                sg, _ = scores(prob_r[prob_r.set == "gold"][FINDINGS])
                row.update({"raw_trusted": so["trusted"], "raw_lookat": so["lookat"], "raw_text_core": so["text_core"], "raw_gold": sg["gold"]})
                print(json.dumps({"epoch": ep + 1, **{k: round(v, 4) for k, v in row.items() if k.startswith("raw_")}}), flush=True)
                pd.concat(raw_preds).to_parquet(out / "preds_raw.parquet")
            if a.save == "all" or (a.save == "last" and last):  # EMA runs: the raw weights too (model_e* holds the EMA weights)
                torch.save(model.state_dict(), out / f"model_raw_e{ep + 1}.pt")
        pd.concat(preds).to_parquet(out / "preds.parquet")
        json.dump({"curve": curve}, open(out / "metrics.json", "w"), indent=1)
        if ep + 1 == a.stop_epoch or step >= stop_at:
            break
    print("peak GPU mem GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main(get_args())
