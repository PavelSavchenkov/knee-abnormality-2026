"""In-context examples for the ZOOM reads of focal findings: per finding (ACL, MCL, menisci, OA x3) 2 report-explicit positives (text v1
status 'meets', probability >= 0.85) and 2 report-explicit negatives (status 'denied' or 'structure_normal', probability <= 0.1), from non-gold, non-pilot,
non-shot studies. Indexes 4000+. Writes out/zshots_studies.csv (for render_zoom.py) and out/zshots.csv (index, finding, answer).
No gold and no image reader is used to pick them (text labels only)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels, probability_table, status_table
from common.paths import KEY_OF

HERE = Path(__file__).parent
FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA"]
st, pr = status_table("v1"), probability_table("v1")
used = set(pd.read_csv(HERE / "out/pilot_studies.csv").StudyInstanceUID) | set(pd.read_csv(HERE / "out/shots_studies.csv").StudyInstanceUID)
pool = st.index.difference(gold_labels().index).difference(list(used))
rng = np.random.default_rng(7)
rows, uids = [], []
for f in FIND:
    k = KEY_OF[f]
    pos = pool[((st.loc[pool, k] == "meets") & (pr.loc[pool, k] >= 0.85)).to_numpy()]
    neg = pool[(st.loc[pool, k].isin(["denied", "structure_normal"]) & (pr.loc[pool, k] <= 0.1)).to_numpy()]
    for ans, cand in [("Yes", pos), ("No", neg)]:
        for u in rng.choice(cand, 2, replace=False):
            uids.append(u)
            rows.append((4000 + len(uids) - 1, f, ans))
pd.DataFrame({"StudyInstanceUID": uids, "stratum": "zshot"}, index=pd.RangeIndex(4000, 4000 + len(uids), name="index")).to_csv(HERE / "out/zshots_studies.csv")
sh = pd.DataFrame(rows, columns=["index", "finding", "answer"])
sh = pd.concat([d.iloc[[0, 2, 1, 3]] for _, d in sh.groupby("finding", sort=False)])
sh.to_csv(HERE / "out/zshots.csv", index=False)
print(sh.groupby("finding").answer.value_counts().unstack().to_string())
