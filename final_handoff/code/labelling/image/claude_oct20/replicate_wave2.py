"""Replication of the image-label arm reads on reads no decision has used: the wave-2 Opus reads (saved after the 10-20 16:01 resume; no
training target holds them). For ARM vs CTRL, pooled over the folds both have, at each run's last eval epoch, held-out studies only:
AUC vs Opus p >= .5 and clear-cut AUC (Opus >= .85 vs <= .1), arm - ctrl with a paired study bootstrap SE. OAI: no; no gold.
  python work/claude_labels_oct20/replicate_wave2.py ARM:HEAD CTRL:HEAD COL [--target CSV]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[2]
E = R / "experiments/v3/oct16_experts"
ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("ctrl")
ap.add_argument("col")
ap.add_argument("--target", default=str(R / "work/claude_labels_oct20/opus_v2_wave2.csv"))
a = ap.parse_args()
lab = pd.read_csv(a.target).set_index("StudyInstanceUID")[a.col].dropna()


def run_dir(name):
    return next(b / name for b in (E / "runs", E / "refs/main", E / "refs/bone", E / "refs") if (b / name).exists())


def last(name, head, f):
    d = run_dir(name) / f"fold{f}"
    fn = next((d / x for x in ("oof.parquet", "oof_partial.parquet") if (d / x).exists()), None)
    if fn is None:
        return None
    o = pd.read_parquet(fn)
    o = o.set_index("StudyInstanceUID") if "StudyInstanceUID" in o.columns else o
    if "epoch" in o.columns:
        o = o[o.epoch == o.epoch.max()]
    return o[head]


(an, ah), (cn, ch) = (x.split(":", 1) for x in (a.arm, a.ctrl))
ps, qs, folds = [], [], []
for f in range(5):
    p, q = last(an, ah, f), last(cn, ch, f)
    if p is None or q is None:
        continue
    ids = p.index.intersection(q.index).intersection(lab.index)
    ps.append(p[ids]); qs.append(q[ids]); folds.append(f)
p, q = pd.concat(ps), pd.concat(qs)
y = lab[p.index]
yb = (y >= .5).astype(int).to_numpy()
cc = ((y >= .85) | (y <= .1)).to_numpy()
rng = np.random.default_rng(0)


def stats(m):
    yy, pp, qq = yb[m], p.to_numpy()[m], q.to_numpy()[m]
    d0 = roc_auc_score(yy, pp) - roc_auc_score(yy, qq)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yy), len(yy))
        if 0 < yy[i].sum() < len(i):
            bs.append(roc_auc_score(yy[i], pp[i]) - roc_auc_score(yy[i], qq[i]))
    return roc_auc_score(yy, pp), roc_auc_score(yy, qq), d0, np.std(bs), int(yy.sum()), len(yy)


line = f"{an}:{ah} vs {cn}:{ch} {a.col} | folds {folds} | {Path(a.target).name}"
for name, m in (("AUC", np.ones(len(yb), bool)), ("clear-cut", cc)):
    if 0 < yb[m].sum() < m.sum():
        aa, cc_, d0, se, npos, n = stats(m)
        line += f" | {name} arm {aa * 1e3:.0f} ctrl {cc_ * 1e3:.0f} d {d0 * 1e3:+.1f} (SE {se * 1e3:.1f}) n {n} pos {npos}"
print(line)
