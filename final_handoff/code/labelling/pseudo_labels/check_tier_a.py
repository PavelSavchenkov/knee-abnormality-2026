"""Proof for the tier-A PL lines (Master 17:0x item 3). OAI: no. CPU only; imports the oct14 code read-only, edits nothing.
1. DRY: the oct14 launcher's own resolution (DEFAULTS / get_args <- src config <- --set) for the plain a14 line vs the PL line ->
   the only differing keys must be seed + labels (main) / target (basin).
2. TARGET DUMP through the trainer's own target code with the resolved PL config, on the fold -1 training set (every non-gold study):
   Y_pl must equal the CSV (main: after its ext: logit round trip), W_pl must equal W_plain; cells that differ from the plain targets
   are counted per column; nfnet's Synovitis must equal the plain Synovitis (ctns + syn_eff).
  conda run -n rsna python work/pl_alldata_2026-10-21/check_tier_a.py --teacher v0 --recipe sub_b0|nfnet"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
A14 = REPO / "experiments/v2/oct14_all_data"
sys.path[:0] = [str(REPO)]
from common.folds import get_folds  # noqa: E402
from common.labels import external  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

LINES = {"sub_b0": ("main", "experiments/v0/oct7_baseline/runs/sub_b0/fold0", "seed=104 eval_every=2 eval_from=1 save_from=1 spe_ref=434",
                    "labels=ext:pl_ct_sub_b0_{tv}"),
         "nfnet": ("basin", "experiments/v0/oct9_baseline_2nd/runs/P_final/fold0", "seed=104 spe_ref=434 stop_step=6076 stop_epoch=0",
                   "target=pl_ctns_nfnet_{tv}")}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve(L, mod, kind, src, sets, name):
    """The launcher's resolution, without running: returns the resolved config dict."""
    s = __import__("json").load(open(REPO / src / "config.json"))
    s = {k: L.REMAP.get(v, v) if isinstance(v, str) else v for k, v in s.items() if k not in L.DROP}
    over = {k: L.parse_value(v) for k, v in (kv.split("=", 1) for kv in sets.split())}
    if kind == "basin":
        ns = mod.get_args(["--name", name, "--fold=-1"])
        for k, v in (s | over).items():
            assert hasattr(ns, k), f"unknown basin arg {k}"
            setattr(ns, k, v)
        return vars(ns) | {"name": None}
    cfg = dict(mod.DEFAULTS)
    assert not [k for k in s | over if k not in cfg]
    return cfg | s | over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--recipe", required=True, choices=LINES, help="one per process: main and basin both have a module named 'data'")
    a = ap.parse_args()
    L = load(A14 / "scripts/launch.py", "o14_launch")
    tr = get_folds(5, 0, include_gold=False).index.tolist()
    ok = True
    for recipe, (kind, src, plain_sets, pl_key) in [(a.recipe, LINES[a.recipe])]:
        pl_key = pl_key.format(tv=a.teacher)
        pl_sets = plain_sets.replace("seed=104", "seed=201") + " " + pl_key
        sys.path.insert(0, str(A14 / kind))
        mod = L.load(kind)
        c0, c1 = resolve(L, mod, kind, src, plain_sets, "x"), resolve(L, mod, kind, src, pl_sets, "x")
        diff = {k: (c0.get(k), c1.get(k)) for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
        want = {"seed", "labels"} if kind == "main" else {"seed", "target"}
        print(f"\n== {recipe} ({kind}) DRY diff plain a14 line vs PL line: {diff}")
        ok &= set(diff) == want
        csv = external(pl_key.split("=", 1)[1].removeprefix("ext:")).loc[tr, FINDINGS]
        if kind == "main":
            lab = load(A14 / "main/labels.py", "o14_main_labels_chk")
            y0, w0 = lab.make_targets(tr, c0)
            y1, w1 = lab.make_targets(tr, c1)
            ok &= c1["head_recipes"] == "" and c1["ext_labels"] == "" and not c1["img_syn"]
        else:
            dat = load(A14 / "basin/data.py", "o14_basin_data_chk")
            kw = lambda c: dict(nm=None if c["nm"] == "mask" else float(c["nm"]), nm_weight=c["nm_weight"], hard=c["hard"],  # noqa: E731
                                reads=[r for r in c["reads"].split(",") if r], reads_weight=c["reads_weight"], syn_eff=bool(c["syn_eff"]),
                                reads_scale=c["reads_scale"], unset_mask=[], unset_mode=c["unset_mode"])
            assert not c1["unset_mask"] and not c1["img_w"]
            Y0, W0 = dat.build_targets(c0["target"], **kw(c0))
            Y1, W1 = dat.build_targets(c1["target"], **kw(c1))
            y0, w0, y1, w1 = (x.loc[tr, FINDINGS].to_numpy(float) for x in (Y0, W0, Y1, W1))
        y0, w0, y1, w1 = (np.asarray(x, float) for x in (y0, w0, y1, w1))
        dcsv = np.abs(y1 - csv.to_numpy(float)).max(0)
        dw = np.abs(w1 - w0).max()
        changed = (np.abs(y1 - y0) > 1e-6).mean(0)
        print(f"  W_pl == W_plain: {dw == 0} (max |dW| {dw}) | W mean {w1.mean():.4f}")
        print("  max |Y_pl - CSV| per column: " + ", ".join(f"{c} {v:.1e}" for c, v in zip(FINDINGS, dcsv)))
        print("  share of training cells changed vs plain: " + ", ".join(f"{c} {v:.3f}" for c, v in zip(FINDINGS, changed)))
        ok &= dw == 0 and dcsv.max() < 1e-4
        if recipe == "nfnet":
            s = FINDINGS.index("Synovitis")
            print(f"  nfnet Synovitis == plain: {np.abs(y1[:, s] - y0[:, s]).max() == 0}")
            ok &= np.abs(y1[:, s] - y0[:, s]).max() == 0
        sys.path.remove(str(A14 / kind))
    print(f"\nALL CHECKS {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
