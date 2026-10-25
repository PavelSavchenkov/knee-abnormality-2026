"""In-context examples for few-shot reads: per finding (Effusion, Synovitis) 2 clear Codex positives (>= 0.85) and 2 clear negatives
(<= 0.1), from non-gold, non-pilot studies with a 0.961 OOF row. Indexes 2000+. Writes out/shots_studies.csv (for render_views.py)
and out/shots.csv (index, finding, answer)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels

HERE = Path(__file__).parent
codex = pd.read_csv("data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")
pilot = pd.read_csv(HERE / "out/pilot_studies.csv").StudyInstanceUID
pool = codex.index.difference(gold_labels().index).difference(pilot)
rng = np.random.default_rng(1)
rows, uids = [], []
for f in ["Effusion", "Synovitis"]:
    for ans, m in [("Yes", codex.loc[pool, f] >= 0.85), ("No", codex.loc[pool, f] <= 0.1)]:
        for u in rng.choice(pool[m.to_numpy()], 2, replace=False):
            uids.append(u)
            rows.append((2000 + len(uids) - 1, f, ans))
pd.DataFrame({"StudyInstanceUID": uids, "stratum": "shot"}, index=pd.RangeIndex(2000, 2000 + len(uids), name="index")).to_csv(HERE / "out/shots_studies.csv")
sh = pd.DataFrame(rows, columns=["index", "finding", "answer"])
sh = pd.concat([d.iloc[[0, 2, 1, 3]] for _, d in sh.groupby("finding", sort=False)])     # alternate Yes / No
sh.to_csv(HERE / "out/shots.csv", index=False)
print(sh.to_string())
