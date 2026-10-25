"""Main pipeline (copy of experiments/v1/oct10_synovitis/main/train.py, itself a copy of experiments/v0/oct7_baseline/train.py):
train one config on one or more folds. oct14_all_data changes: fold -1 = all-data run (every non-gold study trains, 'val' = gold-58);
spe_ref = N runs the schedule (warmup, cosine, epochs) in units of N steps (the fold runs' steps per epoch), so an all-data run
takes the fold runs' step count and LR curve; save_opt also writes opt_ep{E}.pt (optimizer, scaler, step) at every saved epoch.

Changes vs the source: DEFAULTS = the R2 recipe (effb0, 20 ep, aug_ge, save every eval epoch from 12); cls_w (per-class weights of the
main 12-output loss); oof / gold parquets also hold the external-label heads as img_<column> probabilities.

conda run -n rsna --no-capture-output python experiments/v1/oct10_synovitis/main/train.py --name m_NAME --folds 0 --set ext_labels=... ext_weight=0.25
Output: runs/NAME/fold{f}/{config.json, log.txt, oof.parquet (epoch, study x 12 [+ img_*]), gold.parquet, metrics.json, model.pt (final epoch) and
model_ep{E}.pt (each eval epoch >= save_from) if save=True}; a row per fold in runs/results.csv"""
import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
RUNS = HERE.parent / "runs"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Sampler  # noqa: E402
from timm.optim import param_groups_layer_decay, param_groups_weight_decay  # noqa: E402

from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from common.paths import FINDINGS  # noqa: E402
from data import STRUCTURES, contrast_partners, structure_coverage, StudyDataset, build_index, gpu_prep, side_flips, true_other_series  # noqa: E402
from labels import RECIPES, ATTRS, attribute_targets, make_targets, n_status_classes, status_targets  # noqa: E402
from model import KneeNet  # noqa: E402

DEFAULTS = dict(
    backbone="efficientnet_b0.ra_in1k", pretrained=True, cache="fov160_px400_q95", img=256, crop_mm=140, interp="bilinear",
    k=24, k_eval=96, span=(0.05, 0.95), drop_other_knee=False,
    bs=8, epochs=20, lr=1e-3, lr_head=1e-3, wd=1e-2, warmup=1.0, min_lr=0.0, grad_clip=1.0, ema=0.0,
    drop=0.2, hidden=256, head="attn", q_layers=1, q_heads=8, use_type=True, use_pos=True, drop_path=0.0, freeze_bn=False,
    labels="v1prob", nm=0.15, nm_weight=1.0, hard_trusted=False, w_trusted=1.0, site_silence_thr=0.0, site_silence_value=0.03, site_silence_min=20, fs_shrink=5.0, v0_mix=False, syn_proxy=False, eff_below=0.0, cf_cap=0.0, sharpen=0.0, fo_value=0.0, unc_weight=1.0, aux_status=0.0, b_findings="", b_labels="", b_sharpen=0.0, b_v0_mix=False, b_nm=0.0, b_weight=1.0,
    pos_weight=0.0, precision="bf16", channels_last=True, compile=False,
    eval_every=2, eval_from=12, workers=12, seed=0, k_folds=5, prior_init=True, max_steps=0, save=True, save_from=12,
    aug_rot=15, aug_scale=0.15, aug_shift=0.05, aug_bc=0.0, aug_gamma=0.0, aug_noise=0.0, aug_erase=0.3, aug_wdrop=0.0,
    aug_lowres=0.0, aug_bagmix=0.0, aug_mixup=0.0, aug_mirror=0.0, aug_geo_p=1.0, sample="random", rank_weight=0.0, rank_margin=0.3,
    kill_epoch=-1, kill_below=0.0, attr_weight=0.0, aux_smooth=0.0, aux_groups="", pos_scale=60.0, side_fix=False, clean_labels=False, drop_true_other=False, ctx_layers=0, gru=False,
    ext_labels="", ext_weight=0.0, samp_rare=0.0, hard_pow=0.0, hard_band=(), hard_mult=2.0, hard_from=2,
    llrd=0.0, nwd_1d=False, freeze_epochs=0.0, bb_pool="avg", lora_r=0, lora_targets="qkv,proj", grad_ckpt=False, sam=0.0, bb_weights="", lr_seq=0.0, syn_mask_rate=0.0, multicontrast=False, auc_weight=0.0, auc_queue=512, bce_weight=1.0, struct_cov="", prior="", cov_weight=0.0, syn_tie=False, type_prior=False, eval_workers=6, aug_sdrop=0.0, mono=False, prior_findings="", severity=False, head_recipes="", head_weight=0.5, head_avg=True, img_syn="", ext_avg="", asl_gneg=0.0, asl_m=0.05, cls_w="", spe_ref=0, save_opt=True,
)


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
    """(n, E) probabilities from a CSV (StudyInstanceUID + columns named as FINDINGS, e.g. image-read Synovitis; NaN = unlabelled)
    and (n,) per-study loss weights (optional column _w, e.g. the export's post-stratification weight; default 1)."""
    t = pd.read_csv(path).set_index("StudyInstanceUID").reindex(studies)
    w = t.pop("_w").fillna(0).to_numpy(np.float32) if "_w" in t else np.ones(len(t), np.float32)
    assert len(t.columns) and set(t.columns) <= set(FINDINGS), t.columns
    return t.to_numpy(np.float32), w


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


ABBR = {"ACL": "ACL", "MCL": "MCL", "MM": "Medial Meniscus", "LM": "Lateral Meniscus", "MOA": "Medial OA", "LOA": "Lateral OA",
        "PF": "PF OA", "EFF": "Effusion", "SYN": "Synovitis", "BAK": "Baker's", "CON": "Contusion", "FRA": "Fracture"}

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
    """Sigmoid of the 12 main outputs, then of the external-label heads (n, 12 + n_ext); with_aux=True also returns the status
    softmax (n, 12, n_aux) or None."""
    model.eval()
    out, aux_out = [], []
    lo_e, hi_e = model.ext_slice
    for b in loader:
        b = to_dev(b, dev)
        x = gpu_prep(b["x"], cfg["img"], cfg["interp"], None, *model.norm_stats(), mono=cfg["mono"])
        if cfg["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            o = model(x, b["mask"], b["type"], b["pos"], b["sid"], cov=b["cov"])
            lg = (o[0] if isinstance(o, tuple) else o).float()
            main = lg[:, :12].clone()
            for fi, ci in getattr(model, "ext_avg_pairs", []):    # main finding logit averaged with its external-label head
                main[:, fi] = (main[:, fi] + lg[:, ci]) / 2
            main = torch.stack([main] + [lg[:, lo:hi] for lo, hi in getattr(model, "head_slices", [])]).mean(0)
            out.append(torch.sigmoid(torch.cat([main, lg[:, lo_e:hi_e]], 1)).cpu())
            if isinstance(o, tuple):
                aux_out.append(torch.softmax(o[1][:, :12].float(), -1).cpu())
    probs = torch.cat(out).numpy()
    if with_aux:
        return probs, (torch.cat(aux_out).numpy() if aux_out else None)
    return probs


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
                        side_flips() if cfg.get("side_fix") else frozenset(), true_other_series() if cfg.get("drop_true_other") else frozenset(),
                        contrast_partners() if cfg.get("multicontrast") else None)
    folds = folds[folds.index.isin(index.keys())]
    tr = folds.index[folds.fold != fold].tolist()
    va = folds.index[folds.fold == fold].tolist()
    gold = [s for s in gold_labels().index if s in index]
    if fold < 0:                                          # all-data run (fold -1): every non-gold study trains; 'val' = gold-58
        va = list(gold)
    y, w = make_targets(tr, cfg)
    n_aux = n_status_classes(cfg["aux_groups"]) if cfg["aux_status"] > 0 else 0
    n_attr = len(ATTRS) if cfg["attr_weight"] > 0 else 0
    ya = attribute_targets(tr) if n_attr else np.zeros((len(tr), 0), np.float32)
    ye, we = external_targets(cfg["ext_labels"], tr) if cfg["ext_weight"] > 0 else (np.zeros((len(tr), 0), np.float32), np.ones(len(tr), np.float32))
    n_ext = ye.shape[1]
    ya = np.concatenate([ya, ye], 1)                  # attributes then external labels: extra outputs, masked BCE where known
    rec = [make_targets(tr, cfg | RECIPES[n]) for n in head_names(cfg)]   # one 12-output head per extra label recipe
    yr = np.concatenate([r[0] for r in rec], 1) if rec else np.zeros((len(tr), 0), np.float32)
    wr = np.concatenate([r[1] for r in rec], 1) if rec else np.zeros((len(tr), 0), np.float32)
    n_rec = yr.shape[1]
    n_extra = n_attr + n_ext + n_rec
    y_ds = np.concatenate([y, np.nan_to_num(ya), yr, status_targets(tr, cfg["aux_groups"]).astype(np.float32) if n_aux else np.zeros((len(tr), 0), np.float32)], 1)
    known = (~np.isnan(ya)).astype(np.float32)        # extra outputs: masked where unlabelled; ext columns x per-study weight (_w)
    known[:, n_attr:] *= we[:, None]
    w_ds = np.concatenate([w, known, wr], 1)
    log = open(out_dir / "log.txt", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"fold {fold}: train {len(tr)}, val {len(va)}, gold {len(gold)} | cfg {json.dumps(cfg)}")
    ds_tr = EpochDataset(cfg["cache"], tr, y_ds, w_ds, index, cfg["k"], True, cfg["crop_mm"], cfg["seed"], cfg["sample"], cov)
    sampler = EpochSampler(len(tr), cfg["seed"])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["bs"], sampler=sampler, num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=True, prefetch_factor=4)
    mk_eval = lambda ids: DataLoader(StudyDataset(cfg["cache"], ids, None, None, index, cfg["k_eval"], False, cfg["crop_mm"], cov=cov),
                                     batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["eval_workers"], pin_memory=True)
    dl_va, dl_gold = mk_eval(va), mk_eval(gold)

    model = KneeNet(cfg["backbone"], pretrained=cfg["pretrained"], hidden=cfg["hidden"], drop=cfg["drop"], use_type=cfg["use_type"],
                    use_pos=cfg["use_pos"], head=cfg["head"], drop_path=cfg["drop_path"], q_layers=cfg["q_layers"],
                    q_heads=cfg["q_heads"], n_aux=n_aux, n_out=12 + n_extra, pos_scale=cfg.get("pos_scale", 60.0),
                    ctx_layers=cfg["ctx_layers"], gru=cfg["gru"], img_size=cfg["img"], bb_pool=cfg["bb_pool"],
                    lora_r=cfg["lora_r"], lora_targets=cfg["lora_targets"], grad_ckpt=cfg["grad_ckpt"], bb_weights=cfg["bb_weights"],
                    prior=cfg["prior"], cov_head=cfg["cov_weight"] > 0, syn_tie=cfg["syn_tie"], type_prior=cfg["type_prior"], prior_findings=cfg["prior_findings"]).to(dev)
    if cfg["prior_init"]:
        with torch.no_grad():
            p = np.clip(np.concatenate([(y * w).sum(0) / w.sum(0), np.nan_to_num(np.nanmean(ya, 0), nan=0.5),
                                        (yr * wr).sum(0) / np.maximum(wr.sum(0), 1e-6)]), 0.01, 0.99)
            model.bias().copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
    model.head_slices = head_slices(cfg, n_attr, n_ext)
    model.ext_avg_pairs = ext_avg_pairs(cfg, n_attr)
    model.ext_slice = (12 + n_attr, 12 + n_attr + n_ext)
    cols = FINDINGS + (["img_" + c for c in pd.read_csv(cfg["ext_labels"], nrows=1).columns[1:] if c != "_w"] if n_ext else [])
    cw = torch.ones(12, device=dev)                    # cls_w "ALL:0,SYN:1,EFF:1": per-class weights of the main loss (ABBR keys)
    kv = dict(t.split(":") for t in ([cfg["cls_w"]] if isinstance(cfg["cls_w"], str) else list(cfg["cls_w"])) if t)
    cw[:] = float(kv.pop("ALL", 1))
    for k, v in kv.items():
        cw[FINDINGS.index(ABBR[k])] = float(v)
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
    opt = torch.optim.AdamW([g | {"bb": True} for g in bb_groups] + head_groups, weight_decay=cfg["wd"])
    steps_per_epoch = cfg["spe_ref"] or len(dl_tr)       # spe_ref: schedule in the fold runs' epoch units (equal steps)
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg["warmup"] * steps_per_epoch)
    base = [g["lr"] for g in opt.param_groups]
    print(f"steps/epoch {len(dl_tr)} (schedule unit {steps_per_epoch}), total steps {total}", flush=True)

    frozen = int(cfg["freeze_epochs"] * steps_per_epoch)  # linear probe first: backbone lr 0, then its own linear warmup

    def set_lr(step):
        f = step / max(1, warm) if step < warm else cfg["min_lr"] + (1 - cfg["min_lr"]) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * (f if not (frozen and g.get("bb")) else 0.0 if step < frozen else f * min(1.0, (step - frozen) / max(1, warm)))

    scaler = torch.amp.GradScaler(enabled=cfg["precision"] == "fp16")
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]), use_buffers=True) if cfg["ema"] > 0 else None
    pw = torch.full((12,), cfg["pos_weight"], device=dev) if cfg["pos_weight"] > 0 else None
    assert not (cfg["sam"] and cfg["precision"] == "fp16"), "SAM reads raw gradients; use bf16 / fp32"

    auc_mem = AUCMemory(cfg["auc_queue"]) if cfg["auc_weight"] else None
    assert not (cfg["prior"] or cfg["cov_weight"]) or cfg["struct_cov"], "prior / cov_weight need struct_cov"
    cov_cols = torch.tensor([f in STRUCTURES for f in FINDINGS], device=dev)

    def compute_loss(x, b):
        """Total training loss and the per-study main-12 loss (for hard-example sampling)."""
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = fwd(x, b["mask"], b["type"], b["pos"], b["sid"], cov=b["cov"])
        logits, aux = out if n_aux else (out, None)
        logits = logits.float()
        w12 = b["w"][:, :12]
        w12n = w12.sum().clamp(min=1)                       # cls_w scales a class's share; the normaliser stays the unweighted sum
        w12 = w12 * cw
        if cfg["asl_gneg"]:                                   # asymmetric loss (Ben-Baruch et al. 2021) with soft targets
            yy, pr = b["y"][:, :12], torch.sigmoid(logits[:, :12])
            pm = (pr - cfg["asl_m"]).clamp(min=0)              # shifted probability: very easy negatives drop out
            bce = -(yy * F.logsigmoid(logits[:, :12]) + (1 - yy) * pm ** cfg["asl_gneg"] * torch.log((1 - pm).clamp(min=1e-8))) * w12
        else:
            bce = F.binary_cross_entropy_with_logits(logits[:, :12], b["y"][:, :12], reduction="none", pos_weight=pw) * w12
        loss = cfg["bce_weight"] * bce.sum() / w12n
        if auc_mem is not None:
            loss = loss + cfg["auc_weight"] * auc_mem.loss(logits[:, :12], b["y"][:, :12], w12)
        for lo, hi, wt in [(12, 12 + n_attr, cfg["attr_weight"]), (12 + n_attr, 12 + n_attr + n_ext, cfg["ext_weight"]),
                           (12 + n_attr + n_ext, 12 + n_extra, cfg["head_weight"])]:
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
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["sam"]:
                sam_second_pass(model, lambda: compute_loss(x, b)[0], cfg["sam"])
            if cfg["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item()
            n += 1
            step += 1
            if (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total:
                break
            tw = time.time()
        msg = f"ep {ep} loss {tl / max(n, 1):.4f} time {time.time() - t_ep:.0f}s (data wait {t_wait:.0f}s)"
        if sampler.weights is not None:
            sw = sampler.weights
            msg += f" | sampling max/mean {(sw.max() / sw.mean()).item():.2f}"
        last = ep == cfg["epochs"] - 1 or (cfg["max_steps"] and step >= cfg["max_steps"]) or step >= total
        if ((ep + 1) % cfg["eval_every"] == 0 and ep >= cfg["eval_from"]) or last or ep == cfg["kill_epoch"]:
            em = ema.module if ema is not None else model
            te = time.time()
            pv, pa = predict(em, dl_va, cfg, dev, amp, with_aux=True)
            pv = pd.DataFrame(pv, index=va, columns=cols)
            pg = pd.DataFrame(predict(em, dl_gold, cfg, dev, amp), index=gold, columns=cols)
            m = evaluate(pd.concat([pv, pg[~pg.index.isin(pv.index)]])[FINDINGS])
            mv = evaluate(pv[FINDINGS])
            row = {"epoch": ep, "trusted": mv["trusted_mean"], "lookat": mv["lookat_mean"], "text": mv["text_mean"], "gold": m.get("gold_mean"),
                   "ece": mv["calibration"]["ece"], "loss": tl / max(n, 1), "time": time.time() - t0}
            rows.append(row)
            oof_all.append(pv.assign(epoch=ep))
            gold_all.append(pg.assign(epoch=ep))
            pd.concat(oof_all).to_parquet(out_dir / "oof.parquet")     # after every eval, so running arms can be scored
            pd.concat(gold_all).to_parquet(out_dir / "gold.parquet")
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
            if cfg["save"] and (0 <= cfg["save_from"] <= ep or last):
                torch.save(em.state_dict(), out_dir / f"model_ep{ep}.pt")
                if cfg["save_opt"]:                       # fine-tune / resume state: raw weights (if EMA), optimizer, scaler, step
                    torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": ep,
                                "raw": model.state_dict() if ema is not None else None}, out_dir / f"opt_ep{ep}.pt")
        say(msg)
        if ep == cfg["kill_epoch"] and rows[-1]["trusted"] < cfg["kill_below"]:
            say(f"killed at epoch {ep}: trusted {rows[-1]['trusted']:.4f} < {cfg['kill_below']}")
            killed = True
            break
        if last:
            break
    if cfg["save"]:
        torch.save((ema.module if ema is not None else model).state_dict(), out_dir / "model.pt")
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
        out = RUNS / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        r = run_fold(cfg, fold, out)
        res = RUNS / "results.csv"
        pd.DataFrame([{"name": a.name, "fold": fold, **r, "overrides": " ".join(a.set)}]).to_csv(res, mode="a", header=not res.exists(), index=False)


if __name__ == "__main__":
    main()
