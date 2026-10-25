"""oct16_experts copy of experiments/v3/oct15_labels/scripts/launch.py (only the syn kind has a trainer here: syn/syn_train.py).
Train one arm from a deployed model's exact source config (adapted from oct14_all_data/scripts/launch.py): DEFAULTS of the copied
trainer <- the source run's config.json <- overrides. Writes runs/NAME/fold{F}/done.txt when the trainer returns.

    PYTHONPATH=. python experiments/v3/oct16_experts/scripts/launch.py --kind main --src SRC --name NAME --fold F --set k=v ...
Kinds: main = main/train.py (src: refs/sub_b0/fold0 = the deployed main effb0), basin = basin/train.py (src: refs/basin_nfnet_P_final
or refs/basin_r50_F20 = the deployed NFNet / R50 configs), oa = oa/oa_train.py (refs/exp_oa_crop_v2cor[_nfnet]), bone =
bone/bone_train.py (refs/exp_b_lr3 / exp_b_v3 / exp_b_nf), syn = syn/syn_train.py (refs/exp_s_fs / exp_s_fc125); the expert trainers
are copies of oct14_all_data's self-contained copies (localiser outputs in data/loc_pred*, loc/pred; bag caches in runs/_bags).
Paths in overrides (tgt_csv, aux_csv, site_csv, init) are relative to this experiment folder; '{fold}' is filled here (expert
trainers read tgt_csv relative to the repo root, so it is rewritten)."""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
FILES = {"mainp": "mainp/train.py", "main": "main/train.py", "basin": "basin/train.py", "oa": "oa/oa_train.py", "bone": "bone/bone_train.py", "syn": "syn/syn_train.py", "vol": "src/vol3d.py",
         "djob": "src/djob.py"}   # djob = an inference-only dump job (src/djob.py; no training)
REMAP = {"experiments/v1/oct10_synovitis/data/img_syn_prob.csv": "experiments/v3/oct16_experts/data/img_syn_prob.csv"}
DROP = {"name", "fold", "out"}


def parse_value(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v == "True" if v in ("True", "False") else v


def log_inputs(bags, cfg, kind):
    """Checklist F2 (10-18, Master): the EFFECTIVE inputs in the run's own log, printed before run_fold: bags (names, windows each), the
    filter flags, studies, windows per study, FS share, the configured heads, and for seg_mask runs the hidden-pixel share of a window
    sample (SynDataset.seg_drop, as training applies it)."""
    import numpy as np
    from common.meta import SERIES_TYPES
    ws = [w for b in bags.values() for v in b.values() for w in v]
    per = Counter({k: 0 for b in bags.values() for k in b})
    for b in bags.values():
        for k, v in b.items():
            per[k] += len(v)
    fs = sum(SERIES_TYPES[w[7]].endswith("_FS") for w in ws) / max(len(ws), 1)
    filt = {k: cfg[k] for k in ("series", "planes", "fs_only", "nofs_only", "seg_only", "seg_mask", "seg_drop", "seg_morph", "pair", "series_drop") if k in cfg}
    heads = {k: cfg[k] for k in ("outs", "out_bag", "series_heads", "knee_head", "bag_heads", "pl_head", "sd_outs", "sd_fs") if cfg.get(k)}
    print(f"inputs: {len(bags)} studies, {len(ws)} windows ({len(ws) / max(len(bags), 1):.1f} per study), FS share {fs:.3f}; bags "
          f"{dict(sorted(per.items()))}; filters {filt}; heads {heads}; cache {cfg.get('cache') or 'fov160_px400_q95'}", flush=True)
    if kind == "syn" and cfg.get("seg_mask"):
        from syn_data import SynDataset
        ds = SynDataset([], bags, sorted(per), None, None, 1, False, cfg["px"])
        ds.seg_set, ds.seg_morph = cfg["seg_drop"], cfg.get("seg_morph", 0)
        sample = ws[:: max(1, len(ws) // 300)]
        m = [ds.seg_drop(np.array(w[:3]), w) for w in sample]
        frac = [x.mean() for x in m if x is not None]
        print(f"inputs seg_mask ({cfg['seg_drop']}, morph {cfg.get('seg_morph', 0)}): {len(frac)}/{len(sample)} sampled windows have a mask; hidden pixel share mean "
              f"{np.mean(frac) if frac else 0:.4f}", flush=True)


def load(kind):
    spec = importlib.util.spec_from_file_location(f"{kind}_trainer", EXP / FILES[kind])
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((EXP / FILES[kind]).parent))
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=FILES)
    ap.add_argument("--src", required=True, help="source config dir relative to the experiment (its config.json is the recipe)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[])    # repeated --set groups MERGE (10-17 01:10: last-group-only ran the wrong recipe)
    ap.add_argument("--dry", action="store_true", help="resolve and print the config, write nothing, do not train")
    a = ap.parse_args()
    src = {k: REMAP.get(v, v) if isinstance(v, str) else v for k, v in json.load(open(EXP / a.src / "config.json")).items() if k not in DROP}
    over = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    over = {k: v.replace("{fold}", str(a.fold)) if isinstance(v, str) else v for k, v in over.items()}
    if a.kind in ("oa", "bone", "syn") and over.get("tgt_csv"):
        over["tgt_csv"] = str((EXP / over["tgt_csv"]).relative_to(ROOT))
    mod = load(a.kind)
    out = EXP / "runs" / a.name / f"fold{a.fold}"
    if not a.dry:
        out.mkdir(parents=True, exist_ok=True)
    if a.kind == "basin":
        ns = mod.get_args(["--name", a.name, f"--fold={a.fold}"])
        for k, v in (src | over).items():
            assert hasattr(ns, k), f"unknown basin arg {k}"
            setattr(ns, k, v)
        ns.name, ns.fold = a.name, a.fold
        print("launch basin", json.dumps({k: v for k, v in vars(ns).items() if k in src | over}), flush=True)
        mod.main(ns)
    else:
        cfg = dict(mod.DEFAULTS)
        unknown = [k for k in src | over if k not in cfg]
        assert not unknown, f"keys not in {FILES[a.kind]} DEFAULTS: {unknown}"
        cfg.update(src | over)
        print("launch resolved config", json.dumps(cfg, sort_keys=True), flush=True)
        if a.dry:
            return
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        if a.kind in ("main", "mainp"):
            r = mod.run_fold(cfg, a.fold, out)
            import pandas as pd
            res = EXP / "runs" / "results.csv"
            pd.DataFrame([{"name": a.name, "fold": a.fold, **r, "overrides": " ".join(a.set)}]).to_csv(res, mode="a", header=not res.exists(), index=False)
        elif a.kind in ("oa", "vol"):                          # vol = src/vol3d.py (F2 ordered-slice OA expert; configures oa_train)
            assert cfg["side_fix"], "the localiser boxes are in the side-corrected view"
            bags = mod.bags_for(cfg)                             # 10-18: oa_train.bags_for (series + knee bag); the old call dropped cfg series
            log_inputs(bags, cfg, a.kind)
            mod.run_fold(cfg, a.fold, out, bags)
        else:
            bags = mod.load_bags(cfg)
            log_inputs(bags, cfg, a.kind)
            mod.run_fold(cfg, a.fold, out, bags)
    (out / "done.txt").write_text(json.dumps({"src": a.src, "kind": a.kind, "set": a.set}) + "\n")


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    main()
