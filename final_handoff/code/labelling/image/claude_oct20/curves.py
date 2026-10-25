"""Per-epoch held-out agreement with the new Opus labels (USER 10-20: "reading training curves through training runs on different epochs,
including held out metrics"). For an ARM run vs a CONTROL run, pooled over the folds both have: per eval epoch, AUC vs Opus p >= .5 and
clear-cut AUC (Opus >= .85 vs <= .1) on held-out labelled studies, for every head of the arm that predicts the finding (main column,
img_<col>, aux) and the control's main column. Reads oof_partial.parquet (live) when oof.parquet is not there yet. OAI: no; no gold.
  conda run -n rsna python work/claude_labels_oct20/curves.py ARM_RUN CTRL_RUN [--cols Contusion,Fracture,PF OA] [--target CSV]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

E = Path(__file__).resolve().parents[2] / "experiments/v3/oct16_experts"
ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("ctrl")
ap.add_argument("--cols", default="Contusion,Fracture,PF OA")
ap.add_argument("--target", default=str(E / "data/ext_p2_main.csv"), help="label CSV (StudyInstanceUID + columns); default = Opus alone")
a = ap.parse_args()
lab = pd.read_csv(a.target).set_index("StudyInstanceUID")


def run_dir(name):
    for base in (E / "runs", E / "refs/main", E / "refs"):
        if (base / name).exists():
            return base / name
    raise SystemExit(f"no run {name}")


def load(name, f):
    d = run_dir(name) / f"fold{f}"
    for fn in ("oof.parquet", "oof_partial.parquet"):
        if (d / fn).exists():
            o = pd.read_parquet(d / fn)
            return o.set_index("StudyInstanceUID") if "StudyInstanceUID" in o.columns else o
    return None


folds = [f for f in range(5) if load(a.arm, f) is not None and load(a.ctrl, f) is not None]
print(f"ARM {a.arm} vs CTRL {a.ctrl}; folds {folds}; labels {Path(a.target).name}")
for col in a.cols.split(","):
    arms = {f: load(a.arm, f) for f in folds}
    ctls = {f: load(a.ctrl, f) for f in folds}
    heads = [h for h in (col, f"img_{col}", f"aux:{col}") if all(h in arms[f].columns for f in folds)]
    eps = sorted(set.intersection(*[set(arms[f].epoch) for f in folds])) if folds else []
    print(f"== {col}: heads {heads}")
    for ep in eps:
        line, base = f"  ep {ep:2d}:", None
        for h in heads + ["CTRL"]:
            ys, ps = [], []
            for f in folds:
                src = ctls[f] if h == "CTRL" else arms[f]
                ce = ep if h != "CTRL" else (ep if ep in set(src.epoch) else src.epoch.max())
                d = src[src.epoch == ce]
                ids = [s for s in d.index if s in lab.index and not np.isnan(lab.at[s, col])]
                ys.append(lab.loc[ids, col]); ps.append(d.loc[ids, col if h == "CTRL" else h])
            y, p = pd.concat(ys), pd.concat(ps)
            yb = (y >= .5).astype(int)
            cc = (y >= .85) | (y <= .1)
            auc = roc_auc_score(yb, p) * 1e3
            clr = roc_auc_score(yb[cc], p[cc]) * 1e3 if 0 < yb[cc].sum() < cc.sum() else np.nan
            line += f" {h} {auc:.0f} (cc {clr:.0f}) |"
        print(line + f" n {len(y)} pos {int(yb.sum())}")
