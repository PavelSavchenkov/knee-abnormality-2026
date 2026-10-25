"""Fracture step 2-4 read (prompts/validation.md, CPU): the v3c bone members swapped into the shipped Fracture column (b_v3 / b_lr3 / b_nf),
plus the FS bone member ADDED to the Fracture column at weight w (z(column) + w z(bfs Fracture logit), constants label-free per fold);
bfs is not in the shipped Fracture column. ARM = new-label members + bfs_rep, CTRL = seed-1 old-label members + bfs_s1, DEP = shipped column.
Read-only Opus v3f yardstick decides, dense printed as the error catcher; paired bootstrap SE; folds 0-2 (b_nf dumps). x1e-3. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/frac_add_bfs.py"""
import sys

import numpy as np
import pandas as pd

from common.labels import gold_labels
from common.paths import REPO

EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(EXP / "scripts"), str(EXP / "bone")]
import swap_read as sr  # noqa: E402
from lib import soft_auc  # noqa: E402

c, FOLDS, B = "Fracture", [0, 1, 2], 500
gi = set(gold_labels().index)
ARM = {"b_v3": "b3c_rep", "b_lr3": "b3l_rep", "b_nf": "b3n_rep"}
CTRL = {"b_v3": "b_v3_s1", "b_lr3": "b3l_ctl", "b_nf": "b3n_ctl"}
ro = pd.read_csv(EXP / "data/v3f/opus_raw.csv").set_index("StudyInstanceUID")[c].dropna()
dense = pd.read_csv(EXP / "data/v3f/tgt_bone_v3m.csv").set_index("StudyInstanceUID")[c].dropna()


def bfs(run, f):
    d = pd.read_parquet(EXP / "runs" / run / f"fold{f}" / "oof.parquet")
    d = d[d.epoch == d.epoch.max()] if "epoch" in d else d
    d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d else d
    p = d[c].astype(float).clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).groupby(level=0).mean()


def col(sw, brun, w):
    out = []
    for f in FOLDS:
        x = sr.column(f, "oof", sw)[c]
        z = (x - x.mean()) / x.std()
        if brun and w:
            b = bfs(brun, f).reindex(x.index)
            z = z + w * ((b - b.mean()) / b.std()).fillna(0)
        out.append(z)
    o = pd.concat(out)
    return o[~o.index.isin(gi)]


rng = np.random.default_rng(0)


def read(a, b, y):
    ids = a.index.intersection(b.index).intersection(y.index)
    pv, xa, xb = y[ids].to_numpy(), a[ids].to_numpy(), b[ids].to_numpy()
    d = (soft_auc(xa, pv) - soft_auc(xb, pv)) * 1e3
    bs = [(soft_auc(xa[i], pv[i]) - soft_auc(xb[i], pv[i])) * 1e3 for i in (rng.integers(0, len(pv), len(pv)) for _ in range(B))]
    return f"{d:+.1f} ({np.std(bs):.1f})"


dep = col({}, None, 0)
rows = {}
for w in (0, 0.1, 0.25, 0.5):
    A, C = col(ARM, "bfs_rep", w), col(CTRL, "bfs_s1", w)
    rows[f"w bfs {w}"] = {"ARM vs CTRL read-only": read(A, C, ro), "ARM vs SHIPPED read-only": read(A, dep, ro),
                          "CTRL vs SHIPPED read-only": read(C, dep, ro), "ARM vs SHIPPED dense": read(A, dep, dense)}
pd.set_option("display.width", 220)
print(f"OAI: no. FRACTURE column: v3c members swapped + bfs added at w; folds {FOLDS}; Opus v3f read-only n {len(ro.index.intersection(dep.index))}; x1e-3\n"
      + pd.DataFrame(rows).T.to_string())

# STEP 3: cross-fitted bfs weight - for each held-out fold, w picked on the other two folds (read-only Opus, ARM column), applied to it
GRID = (0, 0.1, 0.25, 0.5, 1.0)


def col_f(sw, brun, w, f):
    x = sr.column(f, "oof", sw)[c]
    z = (x - x.mean()) / x.std()
    if brun and w:
        b = bfs(brun, f).reindex(x.index)
        z = z + w * ((b - b.mean()) / b.std()).fillna(0)
    return z[~z.index.isin(gi)]


cache = {(k, w, f): col_f(sw, br, w, f) for k, sw, br in (("A", ARM, "bfs_rep"), ("C", CTRL, "bfs_s1")) for w in GRID for f in FOLDS}
pick, A, C = {}, [], []
for f in FOLDS:
    def score(w, k):
        s = pd.concat([cache[(k, w, g)] for g in FOLDS if g != f])
        i = s.index.intersection(ro.index)
        return soft_auc(s[i].to_numpy(), ro[i].to_numpy())
    pick[f] = (max(GRID, key=lambda w: score(w, "A")), max(GRID, key=lambda w: score(w, "C")))
    A.append(cache[("A", pick[f][0], f)])
    C.append(cache[("C", pick[f][1], f)])
A, C = pd.concat(A), pd.concat(C)
print(f"\nSTEP 3 cross-fitted w (ARM, CTRL picks per held-out fold {pick}):")
print(f"  ARM vs SHIPPED read-only {read(A, dep, ro)} | dense {read(A, dep, dense)} | ARM vs CTRL (each re-fit) read-only {read(A, C, ro)} | "
      f"CTRL vs SHIPPED read-only {read(C, dep, ro)}")
