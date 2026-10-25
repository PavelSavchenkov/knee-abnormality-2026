"""Proof for one v2 PL recipe (targets_v2.py). OAI: no. CPU; imports trainer code read-only (one recipe per process: the families share
module names). PASS only if ALL hold:
1. CONFIG: the launcher's resolution of the plain all-data line vs the v2 line differs ONLY in the expected keys (seed + the file keys;
   oct14 basin also syn_eff 1 -> 0, so the blended Synovitis is not overwritten by the effusion rule). oct16 syn / oa / bone / mainp:
   launch.py --dry; oct16 basin + every oct14 kind: resolved in-process (launch.py --dry trains basin).
2. TARGETS through the trainer's own target code with the resolved v2 config (inline blocks replicated verbatim where the trainer has no
   function: oa_train 589-597, oct16 basin 336-343, oct14 basin 261-264): every trained text column == the v2 text CSV, untrained
   columns == plain, text weights == plain; every image head == its v2 image CSV on all training studies with weight == its plain read
   weight everywhere (unread cells now trained).
  conda run -n rsna python work/pl_alldata_2026-10-21/check_v2.py --recipe b3c_rep --tag v0"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
sys.path[:0] = [str(REPO), str(D)]
import targets as T  # noqa: E402
from targets_v2 import E14, E16, OUTS5, R  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

LINE16 = {"bsd_nf": ["spe_ref=434", "stop_epoch=14", "save=last"], "silsep": ["spe_ref=434", "tgt_csv=data/targets/v4b_fall.csv"], **{r: ["spe_ref=435"] for r in ("b3n_rep", "oasep_nf", "cs_fs", "cs_fc125", "ef3",
          "ef3fc", "ef3nf", "ef3nffc", "sxm")}}
SRC16 = {"cs_fs": "refs/syn/x_cnx5e-5/fold0", "cs_fc125": "refs/syn/x_cnx_fc125/fold0"}


def line14(r):
    """(src, sets) of the recipe's plain oct14 all-data line, from its first run's done.txt (the launcher records src + --set there)."""
    d = json.load(open(E14 / f"runs/A1_{r}/fold-1/done.txt"))
    return d["src"], [s for s in d["set"] if not s.startswith(("gb=", "seed="))]


def v2_sets(r, tv):
    fam, cols, imgs = R[r]
    if fam.startswith("o14"):
        s = {"o14_main": [f"labels=ext:pl2_{r}_{tv}"], "o14_oa": [f"labels=ext:pl2_{r}_{tv}"], "o14_syn": [f"labels=ext:pl2_{r}_{tv}"],
             "o14_basin": [f"target=pl2_{r}_{tv}", "syn_eff=0"]}[fam]
        if "ext" in imgs:
            s.append(f"ext_labels=experiments/v2/oct14_all_data/data/pl/pl2_{r}_ext_{tv}.csv")
        if "img" in imgs:
            s.append(f"img_csv=data/pl/pl2_{r}_img_{tv}.csv")
        if "img_syn" in imgs:
            s.append(f"img_csv=data/pl/pl2_{r}_img_syn_{tv}.csv")
        return s
    s = [f"tgt_csv=data/pl/pl2_{r}_{tv}.csv"]
    if fam == "o16_syn":                                     # tgt_findings empty = all 12 (the CSV holds all 12; untrained = plain)
        if "img_syn" in imgs:
            s.append(f"img_csv=data/pl/pl2_{r}_img_syn_{tv}.csv")
        if "img_eff" in imgs:
            s.append(f"ieff_csv=experiments/v3/oct16_experts/data/pl/pl2_{r}_img_eff_{tv}.csv")
        if "img_cont" in imgs:
            s.append(f"ibone_csv=experiments/v3/oct16_experts/data/pl/pl2_{r}_img_bone_{tv}.csv")
    if fam == "o16_oa":
        s.append(f"tgt_cols={'+'.join(c.replace(' ', '_') for c in OUTS5)}")
    return s


def resolve(r, sets):
    fam = R[r][0]
    if fam.startswith("o16") and fam != "o16_basin":
        out = subprocess.run([sys.executable, str(E16 / "scripts/launch.py"), "--kind", fam[4:], "--src", SRC16.get(r, f"runs/{r}/fold0"), "--name",
                              "dry_x", "--fold", "-1", "--set", *sets, "--dry"], capture_output=True, text=True, cwd=REPO, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
        line = [x for x in out.stdout.splitlines() if x.startswith("launch resolved config ")]
        assert out.returncode == 0 and len(line) == 1, out.stderr[-2000:]
        return json.loads(line[0][len("launch resolved config "):])
    exp = E16 if fam.startswith("o16") else E14
    L = T.load(exp / "scripts/launch.py", "launch_x")
    kind = fam[4:]
    src_dir = REPO / line14(r)[0] if fam.startswith("o14") else E16 / f"runs/{r}/fold0"
    src = {k: L.REMAP.get(v, v) if isinstance(v, str) else v for k, v in json.load(open(src_dir / "config.json")).items() if k not in L.DROP}
    over = {k: L.parse_value(v) for k, v in (kv.split("=", 1) for kv in sets)}
    sys.path.insert(0, str(exp / kind))
    mod = L.load(kind)
    if kind == "basin":
        ns = mod.get_args(["--name", "dry_x", "--fold=-1"])
        for k, v in (src | over).items():
            assert hasattr(ns, k), k
            setattr(ns, k, v)
        return vars(ns)
    cfg = dict(mod.DEFAULTS)
    assert not [k for k in src | over if k not in cfg]
    return cfg | src | over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=R)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.chdir(REPO)
    r, tv = a.recipe, a.tag
    fam, cols, imgs = R[r]
    tr = get_folds(5, 0, include_gold=False).index.tolist()
    plain_sets = (line14(r)[1] if fam.startswith("o14") else LINE16.get(r, ["spe_ref=434"])) + ["seed=101"]
    pl = v2_sets(r, tv)
    c0, c1 = resolve(r, plain_sets), resolve(r, [s for s in plain_sets if not s.startswith("seed=")] + ["seed=201"] + pl)
    diff = {k: (c0.get(k), c1.get(k)) for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
    want = {"seed"} | {s.split("=", 1)[0] for s in pl}
    ok = set(diff) == want
    print(f"== {r} ({fam}) CONFIG diff: {diff}\n   expected keys {sorted(want)}: {'OK' if ok else 'MISMATCH'}")
    text = (pd.read_csv(REPO / f"data/text_labels/external/pl2_{r}_{tv}.csv") if fam.startswith("o14") else
            pd.read_csv(E16 / f"data/pl/pl2_{r}_{tv}.csv")).set_index("StudyInstanceUID").loc[tr]
    res = []                                                  # (check, max |d| or bool)
    exp = E14 if fam.startswith("o14") else E16
    if fam in ("o14_main", "o14_oa", "o16_bone", "o16_mainp", "o16_oa"):
        lab = T.load(exp / f"{fam[4:]}/labels.py", "lab_chk")
        y0, w0 = (np.asarray(x, float) for x in lab.make_targets(tr, c0))
        y1, w1 = (np.asarray(x, float) for x in lab.make_targets(tr, c1))
        Y0, Y1 = pd.DataFrame(y0, tr, FINDINGS), pd.DataFrame(y1, tr, FINDINGS)
        if fam == "o16_oa":                                   # oa_train.py 589-597 (both configs)
            for c_, Y in ((c0, Y0), (c1, Y1)):
                if c_["tgt_csv"]:
                    t = pd.read_csv(REPO / c_["tgt_csv"].format(fold=-1)).set_index("StudyInstanceUID").reindex(tr)
                    for c in ([x.replace("_", " ") for x in c_["tgt_cols"].split("+")] if c_["tgt_cols"] else OUTS5[:3]):
                        v = t[c].to_numpy(np.float32)
                        Y[c] = np.where(np.isfinite(v), v, Y[c].to_numpy())
        oth = [c for c in (OUTS5 if "oa" in fam else FINDINGS) if c not in cols]
        res += [("text == CSV", float((Y1[cols] - text[cols]).abs().max().max())), ("untrained == plain", float((Y1[oth] - Y0[oth]).abs().max().max()) if oth else 0.0),
                ("W == plain", float(np.abs(w1 - w0).max()))]
        if "ext" in imgs:
            mt = T.load(E14 / "main/train.py", "o14_main_train_chk")
            ye, we = mt.external_targets(c1["ext_labels"], tr)
            ex = pd.read_csv(REPO / c1["ext_labels"]).set_index("StudyInstanceUID").loc[tr, "Synovitis"].to_numpy()
            res += [("ext image == CSV", float(np.nanmax(np.abs(np.asarray(ye, float)[:, 0] - ex)))), ("ext image all trained", bool(np.isfinite(np.asarray(ye, float)).all()))]
    elif fam in ("o16_basin", "o14_basin"):
        dat = T.load(exp / "basin/data.py", "basin_chk")
        kw = lambda c: dict(nm=None if c["nm"] == "mask" else float(c["nm"]), nm_weight=c["nm_weight"], hard=c["hard"], reads=[],  # noqa: E731
                            reads_weight=c["reads_weight"], syn_eff=bool(c["syn_eff"]), reads_scale=c["reads_scale"], unset_mask=[], unset_mode=c["unset_mode"])
        Y0, W0 = dat.build_targets(c0["target"], **kw(c0))
        Y1, W1 = dat.build_targets(c1["target"], **kw(c1))
        if fam == "o16_basin":                                # oct16 basin/train.py 336-343
            t = pd.read_csv(E16 / c1["tgt_csv"].replace("{fold}", "-1")).set_index("StudyInstanceUID").reindex(Y1.index)
            for f in [x.replace("_", " ") for x in c1["tgt_findings"].split("+") if x] or FINDINGS:
                okc = t[f].notna()
                Y1.loc[okc, f] = t.loc[okc, f]
                W1.loc[okc, f] = W1.loc[okc, f].where(W1.loc[okc, f] > 0, 1.0)
        res += [("text == CSV", float((Y1.loc[tr, cols] - text[cols]).abs().max().max())), ("W == plain", float((W1.loc[tr] - W0.loc[tr]).abs().max().max()))]
        if "img" in imgs:                                     # oct14 basin/train.py 261-264
            img = pd.read_csv(E14 / c1["img_csv"]).set_index("StudyInstanceUID").Synovitis.dropna()
            csv = pd.read_csv(E14 / f"data/pl/pl2_{r}_img_{tv}.csv").set_index("StudyInstanceUID").Synovitis
            res += [("img == CSV", float((img.reindex(tr).fillna(0.0) - csv.loc[tr]).abs().max())), ("img all trained", bool(img.reindex(tr).notna().all()))]
    else:                                                     # o16_syn / o14_syn: make_syn_targets
        sys.path.insert(0, str(exp / "syn"))
        st = T.load(exp / "syn/syn_train.py", "syn_chk")
        outs = c1["outs"].split(",")
        y0, w0 = (np.asarray(x, float) for x in st.make_syn_targets(tr, c0, outs))
        y1, w1 = (np.asarray(x, float) for x in st.make_syn_targets(tr, c1, outs))
        oc = st.OUT_COL if hasattr(st, "OUT_COL") else {}
        for j, o in enumerate(outs):
            col = oc.get(o, o)
            if col in cols:
                res += [(f"{o} == CSV", float(np.abs(y1[:, j] - text[col].to_numpy()).max())), (f"{o} W == plain", float(np.abs(w1[:, j] - w0[:, j]).max()))]
            elif o in ("img_syn", "img_eff", "img_cont", "img_frac") and o in imgs:
                f = {"img_syn": E14 / c1["img_csv"] if fam == "o14_syn" else E16 / c1["img_csv"], "img_eff": REPO / c1.get("ieff_csv", ""),
                     "img_cont": REPO / c1.get("ibone_csv", ""), "img_frac": REPO / c1.get("ibone_csv", "")}[o]
                cname = {"img_syn": "Synovitis", "img_eff": "Effusion", "img_cont": "Contusion", "img_frac": "Fracture"}[o]
                v = pd.read_csv(f).set_index("StudyInstanceUID")[cname].loc[tr].to_numpy()
                wr = c1["w_img"] if o != "img_eff" else c1["w_ieff"]
                res += [(f"{o} == CSV", float(np.abs(y1[:, j] - v).max())), (f"{o} W == {wr} everywhere", bool(np.all(w1[:, j] == wr)))]
            else:
                res += [(f"{o} unchanged", float(max(np.abs(y1[:, j] - y0[:, j]).max(), np.abs(w1[:, j] - w0[:, j]).max())))]
    for name, v in res:
        good = v if isinstance(v, bool) else (v < 1e-6 if "CSV" in name else v == 0)
        ok &= bool(good)
        print(f"   {name}: {v if isinstance(v, bool) else f'{v:.1e}'} {'OK' if good else 'FAIL'}")
    print(f"ALL CHECKS {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
