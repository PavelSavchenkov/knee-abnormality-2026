"""Where does a new-label bone member lose on the DENSE Contusion yardstick? Single model (own crop fusion, k64 dumps) vs its old-label
counterpart, held-out fold(s): AUC of each POSITIVE group (dense v3e target >= .5) against ALL dense negatives, split by Opus-read or not
and by text v1 status group. x1e-3. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/dense_loss_split.py ARM [ARM ...] --ctrl b_v3_s1 --folds 1"""
import argparse
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, status_table
from common.paths import REPO

EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(EXP / "scripts"), str(EXP / "bone")]
from c179_b_v3floc import BW, fused  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+")
ap.add_argument("--ctrl", default="b_v3_s1")
ap.add_argument("--folds", default="1")
ap.add_argument("--col", default="Contusion")
a = ap.parse_args()
c, folds = a.col, [int(f) for f in a.folds.split(",")]
gi = set(gold_labels().index)
t = pd.read_csv(EXP / "data/v3e/tgt_bone_v3m.csv").set_index("StudyInstanceUID")[c]
read = set(pd.read_csv(EXP / "data/v3e/opus_raw.csv").set_index("StudyInstanceUID")[c].dropna().index)
st = status_table("v1")[{"Contusion": "contusion", "Fracture": "fracture"}[c]]


def score(run):
    s = pd.concat([fused([BW / f"{run}_f{f}_oof.npz"])[c] for f in folds])
    return s[~s.index.isin(gi)]


ctrl = score(a.ctrl)
ids = ctrl.index.intersection(t.index)
y = (t[ids] >= .5)
neg = ids[~y.to_numpy()]
grp = pd.Series(["read" if s in read else "unread" for s in ids], index=ids) + ":" + st.reindex(ids).fillna("none").replace(
    {"structure_normal": "denied", "below": "denied"})
groups = {"all positives": ids[y.to_numpy()]}
for g in sorted(grp[y].unique()):
    groups[g] = ids[y.to_numpy() & (grp == g).to_numpy()]
rows = {}
for arm in a.arms:
    x = score(arm).reindex(ids)
    r = {}
    for g, pos in groups.items():
        if len(pos) < 3:
            continue
        yy = [1] * len(pos) + [0] * len(neg)
        r[f"{g} (n {len(pos)})"] = f"{(roc_auc_score(yy, x[pos.append(neg)]) - roc_auc_score(yy, ctrl[pos.append(neg)])) * 1e3:+.1f}"
    rows[arm] = r
pd.set_option("display.width", 250)
print(f"OAI: no. DENSE {c} loss split, single models vs {a.ctrl}, folds {folds}, {len(neg)} dense negatives; positive group vs all negatives, x1e-3")
print(pd.DataFrame(rows).to_string())
