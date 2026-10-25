"""Proof for the tier-C PL lines. OAI: no. CPU; edits nothing. One recipe per process (each kind has its own 'labels' / 'data' modules).
1. CONFIG: the oct16 launcher's resolution of the plain all-data line vs the PL line (syn / oa / mainp: launch.py --dry; basin: resolved
   in-process, because launch.py --dry still TRAINS basin) -> the only differing keys must be seed + tgt_csv (+ tgt_findings / tgt_cols).
2. TARGETS with the resolved PL config: syn = syn_train.make_syn_targets, mainp = labels.make_targets (the trainers' own functions);
   oa / basin = their make_targets / build_targets + a verbatim replica of the trainer's inline tgt_csv block (oa_train.py 589-597,
   basin/train.py 336-343). PL columns == the CSV, every other output and every weight == plain. Prints the log line the trainer
   itself writes at start (oa: 'tgt_csv <col>: N of M cells replaced, mean target a -> b'), for the run audit.
  conda run -n rsna python work/pl_alldata_2026-10-21/check_tier_c.py --teacher v0 --recipe z1_acl"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(REPO), str(D)]
import targets as T  # noqa: E402
from targets_c import SPEC  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

LINE = {"z1_acl": ["spe_ref=434"], "w5b_mcl": ["spe_ref=434"], "b3g": ["spe_ref=434"], "sd1b": ["spe_ref=434"],
        "bsd_nf": ["spe_ref=434", "stop_epoch=14", "save=last"], "oasep": ["spe_ref=434"], "oasep_nf": ["spe_ref=435"]}   # = the all-data lines


def pl_sets(r, tv):
    kind, cols = SPEC[r]
    extra = {"syn": [f"tgt_findings={'+'.join(c.replace(' ', '_') for c in cols)}"], "basin": [f"tgt_findings={'+'.join(cols)}"],
             "oa": [f"tgt_cols={'+'.join(c.replace(' ', '_') for c in cols)}"], "mainp": []}[kind]
    return LINE[r] + ["seed=201", f"tgt_csv=data/pl/pl_ct_{r}_{tv}.csv"] + extra


def dry(r, sets):
    out = subprocess.run([sys.executable, str(EXP / "scripts/launch.py"), "--kind", SPEC[r][0], "--src", f"runs/{r}/fold0", "--name", "dry_x",
                          "--fold", "-1", "--set", *sets, "--dry"], capture_output=True, text=True, cwd=REPO, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
    line = [x for x in out.stdout.splitlines() if x.startswith("launch resolved config ")]
    assert out.returncode == 0 and len(line) == 1, out.stderr[-2000:]
    return json.loads(line[0][len("launch resolved config "):])


def basin_resolve(sets):
    L = T.load(EXP / "scripts/launch.py", "o16_launch")
    src = {k: L.REMAP.get(v, v) if isinstance(v, str) else v for k, v in json.load(open(EXP / "runs/bsd_nf/fold0/config.json")).items() if k not in L.DROP}
    over = {k: L.parse_value(v) for k, v in (kv.split("=", 1) for kv in sets)}
    mod = L.load("basin")
    ns = mod.get_args(["--name", "dry_x", "--fold=-1"])
    for k, v in (src | over).items():
        assert hasattr(ns, k), k
        setattr(ns, k, v)
    return vars(ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--recipe", required=True, choices=SPEC)
    a = ap.parse_args()
    os.chdir(REPO)
    r, (kind, cols) = a.recipe, SPEC[a.recipe]
    tr = get_folds(5, 0, include_gold=False).index.tolist()
    csv = pd.read_csv(EXP / f"data/pl/pl_ct_{r}_{a.teacher}.csv").set_index("StudyInstanceUID").loc[tr, cols]
    plain_sets = LINE[r] + ["seed=101"]
    if kind == "basin":
        sys.path.insert(0, str(EXP / "basin"))
        c0, c1 = basin_resolve(plain_sets), basin_resolve(pl_sets(r, a.teacher))
    else:
        sys.path.insert(0, str(EXP / kind))
        c0, c1 = dry(r, plain_sets), dry(r, pl_sets(r, a.teacher))
    diff = {k: (c0.get(k), c1.get(k)) for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
    want = {"seed", "tgt_csv"} | {"syn": {"tgt_findings"}, "basin": {"tgt_findings"}, "oa": {"tgt_cols"}, "mainp": set()}[kind]
    print(f"== {r} ({kind}) CONFIG diff plain vs PL: {diff}")
    ok = set(diff) == want
    j = [FINDINGS.index(c) for c in cols]
    if kind == "syn":
        st = T.load(EXP / "syn/syn_train.py", "o16_syn_train")
        outs = c1["outs"].split(",")
        y0, w0 = (np.asarray(x, float) for x in st.make_syn_targets(tr, c0, outs))
        y1, w1 = (np.asarray(x, float) for x in st.make_syn_targets(tr, c1, outs))
        dcols = [o for i, o in enumerate(outs) if np.abs(y1[:, i] - y0[:, i]).max() > 0]
        hit = {o: min(np.abs(y1[:, outs.index(o)] - csv[c].to_numpy()).max() for c in cols) for o in dcols}
        print(f"  outputs changed: {dcols} (max |Y - CSV| {hit}); W == plain: {np.abs(w1 - w0).max() == 0}")
        ok &= len(dcols) == len(cols) and max(hit.values()) < 1e-6 and np.abs(w1 - w0).max() == 0
    elif kind == "mainp":
        lab = T.load(EXP / "mainp/labels.py", "o16_mainp_labels_chk")
        y0, w0 = (np.asarray(x, float) for x in lab.make_targets(tr, c0))
        y1, w1 = (np.asarray(x, float) for x in lab.make_targets(tr, c1))
        oth = [i for i in range(12) if i not in j]
        d_csv, d_oth, d_w = np.abs(y1[:, j] - csv.to_numpy()).max(), np.abs(y1[:, oth] - y0[:, oth]).max(), np.abs(w1 - w0).max()
        print(f"  max |Y_pl - CSV| {d_csv:.1e} | other columns vs plain {d_oth:.1e} | W vs plain {d_w:.1e}")
        ok &= d_csv < 1e-6 and d_oth == 0 and d_w == 0
    elif kind == "oa":
        lab = T.load(EXP / "oa/labels.py", "o16_oa_labels_chk")
        OUTS = ["Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus"]
        y, w = (np.asarray(x, float)[:, [FINDINGS.index(f) for f in OUTS]] for x in lab.make_targets(tr, c1))
        y0 = y.copy()
        t = pd.read_csv(REPO / c1["tgt_csv"].format(fold=-1)).set_index("StudyInstanceUID").reindex(tr)          # oa_train.py 589-597, verbatim
        for c in ([x.replace("_", " ") for x in c1["tgt_cols"].split("+")] if c1["tgt_cols"] else OUTS[:3]):
            v = t[c].to_numpy(np.float32)
            ok_ = np.isfinite(v)
            jj = OUTS.index(c)
            print(f"  expected trainer log: tgt_csv {c}: {ok_.sum()} of {len(tr)} cells replaced, mean target {y[:, jj].mean():.4f} -> "
                  f"{np.where(ok_, v, y[:, jj]).mean():.4f}")
            y[:, jj] = np.where(ok_, v, y[:, jj])
        jj = [OUTS.index(c) for c in cols]
        oth = [i for i in range(5) if i not in jj]
        d_csv, d_oth = np.abs(y[:, jj] - csv.to_numpy()).max(), np.abs(y[:, oth] - y0[:, oth]).max()
        yp, wp = (np.asarray(x, float)[:, [FINDINGS.index(f) for f in OUTS]] for x in lab.make_targets(tr, c0))
        print(f"  max |Y_pl - CSV| {d_csv:.1e} | other outputs vs plain {np.abs(y[:, oth] - yp[:, oth]).max():.1e} | W vs plain {np.abs(w - wp).max():.1e}")
        ok &= d_csv < 1e-6 and d_oth == 0 and np.abs(y[:, oth] - yp[:, oth]).max() == 0 and np.abs(w - wp).max() == 0
    else:
        dat = T.load(EXP / "basin/data.py", "o16_basin_data_chk")
        kw = lambda c: dict(nm=None if c["nm"] == "mask" else float(c["nm"]), nm_weight=c["nm_weight"], hard=c["hard"], reads=[],  # noqa: E731
                            reads_weight=c["reads_weight"], syn_eff=bool(c["syn_eff"]), reads_scale=c["reads_scale"], unset_mask=[], unset_mode=c["unset_mode"])
        Y0, W0 = dat.build_targets(c0["target"], **kw(c0))
        Y, W = dat.build_targets(c1["target"], **kw(c1))
        t = pd.read_csv(EXP / c1["tgt_csv"].replace("{fold}", str(-1))).set_index("StudyInstanceUID").reindex(Y.index)   # basin/train.py 336-343
        for f in [x.replace("_", " ") for x in c1["tgt_findings"].split("+") if x] or FINDINGS:
            ok_ = t[f].notna()
            Y.loc[ok_, f] = t.loc[ok_, f]
            W.loc[ok_, f] = W.loc[ok_, f].where(W.loc[ok_, f] > 0, 1.0)
        oth = [c for c in FINDINGS if c not in cols]
        d_csv = float((Y.loc[tr, cols] - csv).abs().max().max())
        d_oth, d_w = float((Y.loc[tr, oth] - Y0.loc[tr, oth]).abs().max().max()), float((W.loc[tr] - W0.loc[tr]).abs().max().max())
        print(f"  max |Y_pl - CSV| {d_csv:.1e} | other columns vs plain {d_oth:.1e} | W vs plain {d_w:.1e}")
        ok &= d_csv < 1e-6 and d_oth == 0 and d_w == 0
    print(f"ALL CHECKS {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
