"""Held-out agreement of the shipped 0.961 package OOF (pkg961 oof_f5_win, folds 0-2, logits) with the Opus p2 reads: AUC (Opus p >= .5) and
clear-cut AUC (Opus >= .85 vs <= .1). The agreement ceiling reference for image-label arms. OAI: no; no gold.
  conda run -n rsna python work/claude_labels_oct20/pkg_opus_ceiling.py"""
import pandas as pd
from sklearn.metrics import roc_auc_score
lab = pd.read_csv("work/claude_labels_oct20/opus_v2_all.csv").set_index("StudyInstanceUID")
o = pd.read_parquet("experiments/v3/oct16_experts/refs/pkg961/oof_f5_win.parquet")
for fm in o.index.get_level_values(0).unique():
    x = o.xs(fm, level=0)
    line = f"{fm:12}"
    for c in ("Contusion", "Fracture", "PF OA"):
        ids = x.index.intersection(lab.index[lab[c].notna()])
        y = (lab.loc[ids, c] >= .5).astype(int); cc = ((lab.loc[ids, c] >= .85) | (lab.loc[ids, c] <= .1)).to_numpy()
        p = x.loc[ids, c]
        line += f" | {c} AUC {roc_auc_score(y, p) * 1e3:.0f} cc {roc_auc_score(y[cc], p[cc]) * 1e3:.0f} n {len(ids)}"
    print(line)
