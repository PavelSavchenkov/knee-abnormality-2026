"""Per-finding quality of open-VLM reads (score.py output) on the pilot studies, next to the 0.961 OOF on the same studies.
References:
- text = text v1 probability >= 0.5 (not mentioned -> 0);
- trusted = hand-checked text cells (common.labels.eval_cells, verdict trusted);
- codex = Codex image reads (Synovitis / Effusion from data/derived/synovitis_labels_codex.csv, >= 0.5);
- gold = official labels (only when the read covers gold studies).
Usage: python evaluate.py SCORES.csv [SCORES2.csv ...]"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common.labels import eval_cells, gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

OOF = REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet"
oof = pd.read_parquet(OOF).loc["f5_win"]   # the file holds f5_eff and f5_win; f5_win = shipped 0.961
text = (probability_table("v1").fillna(0) >= 0.5).astype(int).rename(columns={KEY_OF[f]: f for f in FINDINGS})
codex = (pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")[["Effusion", "Synovitis"]] >= 0.5)
codex["Baker's"] = pd.read_csv(Path(__file__).parent / "out/codex_bakers.csv", index_col="StudyInstanceUID")["Baker's"] >= 0.5   # 516 studies
cells = eval_cells()
trusted = cells[cells.verdict == "trusted"].pivot(index="StudyInstanceUID", columns="finding", values="label")
gold = gold_labels()


def auc(y, s):
    m = y.notna() & s.notna()
    y, s = y[m].astype(int), s[m]
    return (round(roc_auc_score(y, s), 3), int(y.sum()), int(m.sum())) if 0 < y.sum() < len(y) else (None, int(y.sum()), int(m.sum()))


for path in sys.argv[1:]:
    r = pd.read_csv(path).pivot(index="StudyInstanceUID", columns="finding", values="p_yes")
    print(f"\n{path}: {len(r)} studies")
    rows = []
    for f in r.columns:
        v, o = r[f], oof[f].reindex(r.index)
        row = {"finding": f, "p_mean": round(v.mean(), 3), "p_sd": round(v.std(), 3), "rho_oof": round(spearmanr(v, o, nan_policy="omit")[0], 3)}
        refs = {"text": text, "trusted": trusted} | ({"codex": codex} if f in codex else {}) | ({"gold": gold} if r.index.isin(gold.index).any() else {})
        for name, ref in refs.items():
            if f not in ref:
                continue
            y = ref[f].reindex(r.index)
            row[f"vlm_{name}"], npos, n = auc(y, v)
            row[f"oof_{name}"] = auc(y, o)[0]
            row[f"n+/n_{name}"] = f"{npos}/{n}"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("finding").to_string())
