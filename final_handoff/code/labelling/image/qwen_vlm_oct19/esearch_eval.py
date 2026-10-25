"""Effusion set-up search read (pre-registered 19:4x, before any e_* data). Per config file (score.py output on esearch_studies.csv):
- pilot part (index < 5000): VLM AUC vs Codex, vs trusted cells, vs text v1;
- Opus part (index >= 5000; 354 hard studies, text below / unsettled): VLM AUC vs Opus (>= .5) and vs Codex; the TWO-READER AUC =
  AUC(z(logit Codex) + z(logit VLM)) vs Opus, and Codex alone vs Opus.
CHOICE RULE: the config with the highest two-reader AUC vs Opus (the label route = Codex + VLM as a training target; Opus = an independent
image reader); ties within 0.005 -> the higher AUC vs trusted, then the cheaper config. The chosen config (+ the e_Ag baseline) then gets
ONE gold-58 read (gold_check.py).
Usage: python esearch_eval.py e_Ag.csv e_Ag_grade.csv ..."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import eval_cells, probability_table
from common.paths import REPO

HERE = Path(__file__).parent
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
z = lambda s: (s - s.mean()) / s.std()
codex = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")["Effusion"]
op = json.load(open(REPO / "data/image_labels/opus_effusion.json"))
opus = pd.Series({k: v["findings"]["effusion"]["probability"] for k, v in op.items()}).dropna()
cells = eval_cells()
trusted = cells[(cells.verdict == "trusted") & (cells.finding == "Effusion")].set_index("StudyInstanceUID")["label"]
text = (probability_table("v1")["effusion"].fillna(0) >= 0.5).astype(int)


def auc(y, s):
    d = pd.concat([y.rename("y"), s.rename("s")], axis=1, join="inner").dropna()
    return roc_auc_score(d.y.astype(int), d.s) if 0 < d.y.sum() < len(d) else float("nan")


rows = []
for path in sys.argv[1:]:
    r = pd.read_csv(path).query("finding == 'Effusion'")
    v = r.set_index("StudyInstanceUID")["p_yes"]
    pil, ops = r[r["index"] < 5000].StudyInstanceUID, r[r["index"] >= 5000].StudyInstanceUID
    vo = v[ops]
    both = vo.index.intersection(codex.dropna().index).intersection(opus.index)
    yo = (opus[both] >= 0.5).astype(int)
    two = z(lg(codex[both])) + z(lg(vo[both]))
    rows.append(dict(config=Path(path).stem, pil_codex=auc(codex[pil] >= 0.5, v[pil]), pil_trusted=auc(trusted, v[pil]), pil_text=auc(text[pil], v[pil]),
                     opus_vlm=auc(yo, vo[both]), opus_codex_alone=auc(yo, codex[both]), opus_two_reader=auc(yo, two), n_opus=f"{yo.sum()}/{len(yo)}"))
t = pd.DataFrame(rows).set_index("config").round(3)
print(t.to_string())
best = t.sort_values(["opus_two_reader", "pil_trusted"], ascending=False).index[0]
print(f"rule pick (highest two-reader vs Opus; ties within 0.005 -> trusted, cheaper): {best}")
