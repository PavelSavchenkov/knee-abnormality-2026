"""One compact line per running pass for a monitor: studies read so far and the VLM AUC per finding vs text v1 (and vs Codex for
Effusion / Synovitis), next to the 0.961 OOF on the same studies. Usage: python live_check.py PART_OR_CSV"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import probability_table
from common.paths import FINDINGS, KEY_OF, REPO

oof = pd.read_parquet(REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet").loc["f5_win"]
text = (probability_table("v1").fillna(0) >= 0.5).astype(int).rename(columns={KEY_OF[f]: f for f in FINDINGS})
codex = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")[["Effusion", "Synovitis"]] >= 0.5
codex["Baker's"] = pd.read_csv(Path(__file__).parent / "out/codex_bakers.csv", index_col="StudyInstanceUID")["Baker's"] >= 0.5
r = pd.read_csv(sys.argv[1]).pivot_table(index="StudyInstanceUID", columns="finding", values="p_yes")
SHORT = {"ACL": "ACL", "MCL": "MCL", "Medial Meniscus": "MM", "Lateral Meniscus": "LM", "Medial OA": "MOA", "Lateral OA": "LOA",
         "PF OA": "PF", "Effusion": "Eff", "Synovitis": "Syn", "Baker's": "Bak", "Contusion": "Con", "Fracture": "Fx"}


def auc(y, s):
    m = y.notna() & s.notna()
    return roc_auc_score(y[m].astype(int), s[m]) if 0 < y[m].sum() < m.sum() else float("nan")


out = []
for f in [f for f in FINDINGS if f in r]:
    v, o = r[f], oof[f].reindex(r.index)
    s = f"{SHORT[f]} {auc(text[f].reindex(r.index), v):.2f}/{auc(text[f].reindex(r.index), o):.2f}"
    if f in codex:
        s += f" cx {auc(codex[f].reindex(r.index), v):.2f}/{auc(codex[f].reindex(r.index), o):.2f}"
    out.append(s)
print(f"n={len(r)} vlm/oof text AUC: " + " | ".join(out))
