"""Does a VLM read add to the 0.961 OOF? AUC vs a reference of z(OOF) + w * z(logit VLM) for w on a grid, per finding, on the pilot.
Reference: Codex (Effusion / Synovitis) or trusted text cells. Info only (the grid is read on the same studies; no selection).
Usage: python addon_check.py SCORES.csv"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import eval_cells
from common.paths import REPO

oof = pd.read_parquet(REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet").loc["f5_win"]
codex = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")[["Effusion", "Synovitis"]] >= 0.5
cells = eval_cells()
trusted = cells[cells.verdict == "trusted"].pivot(index="StudyInstanceUID", columns="finding", values="label")
r = pd.read_csv(sys.argv[1]).pivot_table(index="StudyInstanceUID", columns="finding", values="p_yes")
z = lambda s: (s - s.mean()) / s.std()
for f in r.columns:
    ref = codex if f in codex else trusted
    if f not in ref:
        continue
    y = ref[f].reindex(r.index)
    v = np.log(r[f] / (1 - r[f]))
    m = y.notna() & v.notna()
    if not 0 < y[m].sum() < m.sum():
        continue
    aucs = {w: roc_auc_score(y[m].astype(int), (z(oof[f].reindex(r.index)) + w * z(v))[m]) for w in (0, 0.1, 0.25, 0.5, 1)}
    print(f"{f:16s} ref {'codex' if f in codex else 'trusted'} n+ {int(y[m].sum())}/{int(m.sum())}  " + "  ".join(f"w{w}: {a:.4f}" for w, a in aucs.items()))
