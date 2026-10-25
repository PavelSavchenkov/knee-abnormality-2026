"""K-fold split of the 4,407 train studies. Fully defined by (k, seed); the inputs are frozen files.

    from common.folds import get_folds
    folds = get_folds(k=5, seed=0, include_gold=False)   # DataFrame index StudyInstanceUID, column "fold" (0..k-1)

- Units: studies with the same normalised report text, and the one exam stored twice, always share a fold.
- iterstrat MultilabelStratifiedKFold over units, labels: site (language|manufacturer|model|field), the 12 v1 text positives
  (status meets or unsettled) and the gold flag; a unit has a label if any of its studies has it.
- The split is always computed over all studies; include_gold=False only drops the 58 gold rows afterwards, so the fold of a
  non-gold study never depends on the flag."""
import functools
import json

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from common.meta import study_table
from common.paths import KEYS, TEXT_LABELS

POSITIVE = ("meets", "unsettled")


def strat_matrix():
    st = study_table()
    v1 = json.load(open(TEXT_LABELS / "v1.json"))
    pos = pd.DataFrame([[v1[s]["findings"][k]["status"] in POSITIVE for k in KEYS] for s in st.index], index=st.index, columns=KEYS)
    site = pd.get_dummies(st.site)
    return pd.concat([site, pos, st.gold.rename("gold")], axis=1).astype(int)


@functools.cache
def _split(k, seed):
    st = study_table()
    units = st.dup_group.to_numpy()
    Y = strat_matrix().groupby(units).max()
    fold = np.empty(len(Y), int)
    for f, (_, test) in enumerate(MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(Y)), Y.to_numpy())):
        fold[test] = f
    return pd.Series(fold, index=Y.index).reindex(units).set_axis(st.index).rename("fold")


def get_folds(k=5, seed=0, include_gold=False):
    f = _split(k, seed).to_frame()
    if not include_gold:
        f = f[~study_table().gold.reindex(f.index)]
    return f


def check(k=5, seed=0):
    """Print fold sizes, label balance and duplicate-group integrity."""
    f = get_folds(k, seed, include_gold=True)
    st = study_table().join(f)
    X = strat_matrix().join(f)
    assert st.groupby("dup_group").fold.nunique().max() == 1, "a duplicate group spans folds"
    assert f.equals(_split.__wrapped__(k, seed).to_frame()), "split is not deterministic"
    print("studies per fold:", st.fold.value_counts().sort_index().to_dict(), "| gold per fold:", st[st.gold].fold.value_counts().sort_index().to_dict())
    print("positives per fold (v1 meets|unsettled):")
    print(X.groupby("fold")[KEYS].sum().to_string())
    top = st.site.value_counts().index[:8]
    print("top sites per fold:")
    print(pd.crosstab(st.site, st.fold).loc[top].to_string())


if __name__ == "__main__":
    check()
