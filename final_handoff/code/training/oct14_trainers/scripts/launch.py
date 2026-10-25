"""Train one run from a deployed model's exact source config (oct14_all_data): DEFAULTS of the copied trainer <- the source run's
config.json <- path remaps (data copied into this folder) <- overrides. Writes runs/NAME/fold{F}/done.txt when the trainer returns.

    PYTHONPATH=. python experiments/v2/oct14_all_data/scripts/launch.py --kind main|basin|oa|bone|syn \
        --src experiments/v0/oct7_baseline/runs/sub_b0/fold0 --name A1_sub_b0 --fold -1 --set spe_ref=434 seed=0 ...
Kinds: main = main/train.py (sub_b0, m_prob05), basin = basin/train.py (NFNet, R50, n_prob05), oa = oa/oa_train.py,
bone = bone/bone_train.py, syn = syn/syn_train.py. Override values are parsed like the trainers' --set (int, float, bool, str)."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
FILES = {"main": "main/train.py", "basin": "basin/train.py", "oa": "oa/oa_train.py", "bone": "bone/bone_train.py", "syn": "syn/syn_train.py"}
REMAP = {"experiments/v1/oct10_synovitis/data/img_syn_prob.csv": "experiments/v2/oct14_all_data/data/img_syn_prob.csv",
         "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv": "experiments/v2/oct14_all_data/data/tgt_v3.csv"}
DROP = {"name", "fold", "out"}                       # set by this launcher


def parse_value(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v == "True" if v in ("True", "False") else v


def load(kind):
    spec = importlib.util.spec_from_file_location(f"{kind}_trainer", EXP / FILES[kind])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=FILES)
    ap.add_argument("--src", required=True, help="source run fold dir (its config.json is the recipe)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    src = json.load(open(ROOT / a.src / "config.json"))
    src = {k: REMAP.get(v, v) if isinstance(v, str) else v for k, v in src.items() if k not in DROP}
    over = {k: parse_value(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    mod = load(a.kind)
    out = EXP / "runs" / a.name / f"fold{a.fold}"
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
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        if a.kind == "main":
            mod.run_fold(cfg, a.fold, out)
        else:
            if a.kind == "oa":
                assert cfg["side_fix"], "the localiser boxes are in the side-corrected view"
                bags = mod.finalize(mod.build_bags(cfg["crop_scale"], cfg["loc_version"], cfg["trim"], cfg["tf_axial"]))
            else:
                bags = mod.load_bags(cfg)
            mod.run_fold(cfg, a.fold, out, bags)
    (out / "done.txt").write_text(json.dumps({"src": a.src, "kind": a.kind, "set": a.set}) + "\n")


if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve())
    main()
