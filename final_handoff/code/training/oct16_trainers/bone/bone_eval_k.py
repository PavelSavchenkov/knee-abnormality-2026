"""Copied from experiments/v1/oct11_fracture_contusion/scripts/bone_eval_k.py; lives in bone/ (EXP = this experiment) and
imports this folder's bone/ copy. Run as: python experiments/v2/oct14_backbone_sweep/bone/bone_eval_k.py runs/NAME 1 --k 64

Eval-time window budget of a trained bone expert (no training): re-score val + gold of saved fold models with another k_eval per
bag (999 = every candidate window), evenly spread as in training-time eval. Writes out/eval_k/{run}_f{fold}_k{K}_{val,gold}.parquet
(FINDINGS + extra head columns) and prints the F / C panel per k.
CLI: python scripts/bone_eval_k.py runs/b_v3 0 --k 32,64,999 [--epoch 19] [--device cuda]"""
import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[2]))
sys.path.insert(0, str(EXP / "bone"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bone_data import BoneDataset  # noqa: E402
from bone_train import DEFAULTS, FLOC, LOC, BoneNet, as_frame, fc_metrics, load_bags, predict  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("fold", type=int)
    ap.add_argument("--k", default="32,64,999")
    ap.add_argument("--epoch", type=int, default=19)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=2)
    a = ap.parse_args()
    d = EXP / a.run / f"fold{a.fold}"
    cfg = DEFAULTS | json.load(open(d / "config.json"))
    bags = load_bags(cfg)
    names, outs = cfg["bags"].split(","), cfg["outs"].split(",")
    out_bags = np.ones((len(outs), len(names)), bool)
    for j, o in enumerate(outs):                          # as bone_train.run_fold
        own = LOC[o][1] if o in LOC else FLOC[o][1] if o in FLOC else None
        if cfg["loc_restrict"] and own in names:
            out_bags[j] = [b == own for b in names]
    out_fs = np.array([o in cfg["fs_outs"].split(",") for o in outs])
    model = BoneNet(cfg["backbone"], out_bags, len(names), cfg["hidden"], cfg["drop"], bagsum=cfg["bagsum"], out_fs=out_fs,
                    bb_pool=cfg["bb_pool"], pretrained=False)
    model.load_state_dict(torch.load(d / f"model_ep{a.epoch}.pt", map_location="cpu"))
    model = model.to(a.device).to(memory_format=torch.channels_last)
    folds = get_folds(5, 0)
    va = [s for s in folds.index[folds.fold == a.fold] if s in bags]
    gold = [s for s in gold_labels().index if s in bags]
    amp = {"bf16": torch.bfloat16, "fp32": None}[cfg["precision"]]
    (EXP / "out/eval_k").mkdir(parents=True, exist_ok=True)
    for k in [int(x) for x in a.k.split(",")]:
        mk = lambda ids: DataLoader(BoneDataset(ids, bags, names, None, None, k, False, cfg["px"], n_out=len(outs)), batch_size=a.bs,
                                    num_workers=cfg["eval_workers"], pin_memory=True)
        pv, pg = (as_frame(predict(model, mk(ids), cfg, a.device, amp), ids, outs) for ids in (va, gold))
        for part, t in (("val", pv), ("gold", pg)):
            t.to_parquet(EXP / "out/eval_k" / f"{Path(a.run).name}_f{a.fold}_k{k}_{part}.parquet")
        print(f"k_eval {k}: " + " ".join(f"{m} {v:.4f}" for m, v in fc_metrics(pv, pg).items()), flush=True)


if __name__ == "__main__":
    main()
