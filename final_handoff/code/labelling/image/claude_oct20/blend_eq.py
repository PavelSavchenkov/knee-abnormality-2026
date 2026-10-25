"""Is the Opus-only equal-N model (eqm_new / eqb_new: ~440 read studies) complementary to the full text-trained control? Blend
z(logit) of the control with w * z(logit) of the eq model, per fold, then pooled held-out Opus agreement (AUC, Opus p >= .5) with a paired
bootstrap SE vs the control alone. eqm_old (same studies, text labels) is the null blend partner. OAI: no; no gold.
  python work/claude_labels_oct20/blend_eq.py [--target CSV] [--ctrl RUN --arms A,B --cols C1,C2 --folds 0,1,2]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[2]
E = R / "experiments/v3/oct16_experts"
ap = argparse.ArgumentParser()
ap.add_argument("--target", default=str(R / "work/claude_labels_oct20/opus_v2_all.csv"))
ap.add_argument("--ctrl")
ap.add_argument("--arms", default="")
ap.add_argument("--cols", default="Contusion,Fracture,PF OA")
ap.add_argument("--folds", default="0,1,2")
a = ap.parse_args()
FOLDS = [int(f) for f in a.folds.split(",")]
lab = pd.read_csv(a.target).set_index("StudyInstanceUID")


def last(name, f):
    d = next(b / name for b in (E / "runs", E / "refs/main", E / "refs/bone") if (b / name).exists()) / f"fold{f}"
    o = pd.read_parquet(d / "oof.parquet")
    o = o.set_index("StudyInstanceUID") if "StudyInstanceUID" in o.columns else o
    return o[o.epoch == o.epoch.max()] if "epoch" in o.columns else o


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
rng = np.random.default_rng(0)
SETS = [(a.ctrl, tuple(a.arms.split(",")), tuple(a.cols.split(",")))] if a.ctrl else \
    [("sub_b0_s1", ("eqm_new", "eqm_old"), ("Contusion", "Fracture", "PF OA")), ("b_v3_s1", ("eqb_new", "eqb_old"), ("Contusion", "Fracture"))]
for ctrl, arms, cols in SETS:
    for c in cols:
        for arm in arms:
            for w in (.25, .5, 1.0):
                ys, bs, cs = [], [], []
                for f in FOLDS:
                    q, p = last(ctrl, f)[c], last(arm, f)[c]
                    ids = q.index.intersection(p.index)
                    blend = z(lg(q[ids])) + w * z(lg(p[ids]))     # z over the whole held-out fold, not only read studies
                    ids_l = ids.intersection(lab.index[lab[c].notna()])
                    ys.append(lab.loc[ids_l, c]); bs.append(blend[ids_l]); cs.append(q[ids_l])
                y = (pd.concat(ys) >= .5).astype(int).to_numpy(); b = pd.concat(bs).to_numpy(); q = pd.concat(cs).to_numpy()
                d = roc_auc_score(y, b) - roc_auc_score(y, q)
                bt = [roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], q[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if 0 < y[i].sum() < len(i)]
                print(f"{ctrl} + {w} x {arm:8} {c:9}: d AUC {d * 1e3:+6.1f} (SE {np.std(bt) * 1e3:4.1f}) n {len(y)} pos {y.sum()}")
