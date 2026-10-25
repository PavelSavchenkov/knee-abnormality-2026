"""Copied 2026-10-14 from experiments/v0/oct7_baseline (the deployed sub_b0 source; runs go to ../runs). Re-score a saved fold model (train.py save=True) with other inference settings, without training.

conda run -n rsna --no-capture-output python experiments/v0/oct7_baseline/predict.py RUN FOLD --set k_eval=192 bs=4
Prints the standard metrics for the validation fold and gold-58 and writes runs/RUN/foldF/pred_<overrides>.parquet.
--adabn: first re-estimate the backbone's BatchNorm statistics on the unlabelled windows being scored (validation fold + gold,
as the private test would be at scoring time), then predict (AdaBN, Li et al. 2016)."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common.cache import SliceCache  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from common.paths import FINDINGS  # noqa: E402
from data import contrast_partners, structure_coverage, StudyDataset, build_index, side_flips, true_other_series  # noqa: E402
from labels import ATTRS, n_status_classes  # noqa: E402
from model import KneeNet  # noqa: E402
from data import gpu_prep  # noqa: E402
from train import DEFAULTS, ext_avg_pairs, head_names, head_slices, n_aux_cols, parse_value, predict, to_dev  # noqa: E402


@torch.no_grad()
def adabn(model, loaders, cfg, amp):
    """Reset BatchNorm running statistics and re-estimate them (cumulative average) from one pass over the real windows
    (padding windows skipped) of the given loaders; heads use LayerNorm, so only the backbone is run."""
    bns = [m for m in model.backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns, "backbone has no BatchNorm"
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.eval()
    for m in bns:
        m.train()
    for dl in loaders:
        for b in dl:
            b = to_dev(b, "cuda")
            x = gpu_prep(b["x"], cfg["img"], cfg["interp"], None, *model.norm_stats(), mono=cfg["mono"])[b["mask"].flatten()]
            if cfg["channels_last"]:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                model.backbone(x)
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("fold", type=int)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--adabn", action="store_true")
    ap.add_argument("--no_cov", action="store_true", help="structure coverage = 0 at eval (a mask-free deployment)")
    a = ap.parse_args()
    d = HERE.parent / "runs" / a.run / f"fold{a.fold}"
    cfg = DEFAULTS | json.load(open(d / "config.json"))           # older runs lack newer keys (e.g. side_fix, pos_scale)
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown key {k}"
        cfg[k] = parse_value(v)
    torch.backends.cudnn.benchmark = True
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg["precision"]]
    cache = SliceCache(cfg["cache"])
    cov = structure_coverage(cfg["struct_cov"], len(cache.slices)) if cfg["struct_cov"] and not a.no_cov else None
    index = build_index(cache, tuple(cfg["span"]), cfg["drop_other_knee"],
                        side_flips() if cfg.get("side_fix") else frozenset(), true_other_series() if cfg.get("drop_true_other") else frozenset(),
                        contrast_partners() if cfg.get("multicontrast") else None)
    folds = get_folds(cfg["k_folds"], 0, include_gold=False)
    va = [s for s in folds.index[folds.fold == a.fold] if s in index]
    gold = [s for s in gold_labels().index if s in index]
    model = KneeNet(cfg["backbone"], pretrained=False, hidden=cfg["hidden"], drop=cfg["drop"], use_type=cfg["use_type"], use_pos=cfg["use_pos"],
                    head=cfg["head"], drop_path=cfg["drop_path"], q_layers=cfg["q_layers"], q_heads=cfg["q_heads"],
                    n_aux=n_status_classes(cfg.get("aux_groups", "")) if cfg.get("aux_status", 0) > 0 else 0, n_out=12 + (len(ATTRS) if cfg.get("attr_weight", 0) > 0 else 0)
                    + (len(pd.read_csv(cfg["ext_labels"], nrows=1).columns) - 1 if cfg.get("ext_weight", 0) > 0 else 0)
                    + 12 * len(head_names(cfg)) + n_aux_cols(cfg), pos_scale=cfg.get("pos_scale", 60.0),
                    ctx_layers=cfg.get("ctx_layers", 0), gru=cfg.get("gru", False), img_size=cfg["img"], bb_pool=cfg["bb_pool"],
                    lora_r=cfg["lora_r"], lora_targets=cfg["lora_targets"], grad_ckpt=cfg["grad_ckpt"], bb_weights=cfg["bb_weights"],
                    prior=cfg["prior"], cov_head=cfg["cov_weight"] > 0, syn_tie=cfg["syn_tie"], type_prior=cfg["type_prior"], prior_findings=cfg["prior_findings"])
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    model.ext_avg_pairs = ext_avg_pairs(cfg, len(ATTRS) if cfg.get("attr_weight", 0) > 0 else 0)
    model.head_slices = head_slices(cfg, len(ATTRS) if cfg.get("attr_weight", 0) > 0 else 0,
                                    len(pd.read_csv(cfg["ext_labels"], nrows=1).columns) - 1 if cfg.get("ext_weight", 0) > 0 else 0)
    model = model.cuda()
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    mk = lambda ids: DataLoader(StudyDataset(cfg["cache"], ids, None, None, index, cfg["k_eval"], False, cfg["crop_mm"], cov=cov),
                                batch_size=max(1, cfg["bs"] // 2), num_workers=cfg["workers"], pin_memory=True)
    if a.adabn:
        adabn(model, [mk(va), mk(gold)], cfg, amp)
    pv = pd.DataFrame(predict(model, mk(va), cfg, "cuda", amp), index=va, columns=FINDINGS)
    pg = pd.DataFrame(predict(model, mk(gold), cfg, "cuda", amp), index=gold, columns=FINDINGS)
    mv, mg = evaluate(pv), evaluate(pg)
    print(f"{a.run} fold {a.fold} {' '.join(a.set)}{' adabn' if a.adabn else ''} | trusted {mv['trusted_mean']:.4f} lookat {mv['lookat_mean']:.4f} "
          f"text {mv['text_mean']:.4f} gold {mg['gold_mean']:.4f}")
    tag = "_".join([kv.replace("=", "") for kv in a.set] + (["adabn"] if a.adabn else []) + (["nocov"] if a.no_cov else [])) or "same"
    pd.concat([pv, pg]).to_parquet(d / f"pred_{tag}.parquet")


if __name__ == "__main__":
    main()
