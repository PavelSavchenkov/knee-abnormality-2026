"""Baker's yardstick check (Experts ask 10-21 11:0x: is Codex cleared as the Baker's yardstick?). Codex read the 516 non-gold studies whose
text v1 Baker's status is below / unsettled; Opus p4 (images + report) re-read 160 of them, most uncertain Codex read first. Compares the two
readers on the overlap (agreement, positive rate, rank) and both vs the text v1 probability. No gold here (gold-58 numbers live in
yardstick_check.txt). OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/bakers_yardstick.py"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common.labels import probability_table, status_table
from common.paths import REPO

L = REPO / "data/image_labels"


def probs(f):
    d = json.load(open(L / f))
    return pd.Series({s: v["findings"]["bakers_cyst"].get("probability") for s, v in d.items()}, dtype=float).dropna()


cx, op = probs("codex_bakers.json"), probs("opus_bakers_p4.json")
tp, ts = probability_table("v1")["bakers_cyst"], status_table("v1")["bakers_cyst"]
ids = cx.index.intersection(op.index)
c, o, t = cx[ids], op[ids], tp.reindex(ids)
print(f"Codex pool {len(cx)}, Opus p4 {len(op)}, overlap {len(ids)}")
print(f"positive rate (p >= .5): Codex {(c >= .5).mean():.2f} | Opus {(o >= .5).mean():.2f}; Codex on the whole pool {(cx >= .5).mean():.2f}")
print(f"same side of .5: {((c >= .5) == (o >= .5)).mean():.2f}; Codex+ Opus- {int(((c >= .5) & (o < .5)).sum())}, Codex- Opus+ {int(((c < .5) & (o >= .5)).sum())}")
print(f"rank agreement: Spearman Codex-Opus {spearmanr(c, o)[0]:.2f}; Codex-text {spearmanr(c, t)[0]:.2f}; Opus-text {spearmanr(o, t)[0]:.2f}")
yo = (o >= .5).astype(int)
if 0 < yo.sum() < len(yo):
    print(f"AUC vs Opus binary: Codex {roc_auc_score(yo, c) * 1e3:.0f} | text v1 {roc_auc_score(yo, t.fillna(t.median())) * 1e3:.0f}")
yc = (c >= .5).astype(int)
if 0 < yc.sum() < len(yc):
    print(f"AUC vs Codex binary: Opus {roc_auc_score(yc, o) * 1e3:.0f} | text v1 {roc_auc_score(yc, t.fillna(t.median())) * 1e3:.0f}")
print("by text v1 status (n / Codex+ / Opus+):", ", ".join(f"{g}: {len(d)} / {int((d.c >= .5).sum())} / {int((d.o >= .5).sum())}"
      for g, d in pd.DataFrame({"c": c, "o": o, "g": ts.reindex(ids).fillna("none")}).groupby("g")))
print("Codex probability bins vs Opus+ rate:", ", ".join(f"{b}: {len(d)} / {d.o.ge(.5).mean():.2f}"
      for b, d in pd.DataFrame({"c": c, "o": o}).assign(b=lambda x: pd.cut(x.c, [0, .2, .4, .6, .8, 1.01], right=False)).groupby("b", observed=True)))
