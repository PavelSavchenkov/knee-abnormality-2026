"""Phase 2, session 7 (USER 10-20: "If fracture matters, a continuation list should add report-vague fracture studies" -> "ok, prepare everything").
The 720 p2_s1..s6 studies hold few fracture positives (~5 %: they were picked for vague contusion / PF wording). This list = the studies whose
text v1 Fracture status is vague (unsettled / hedged / conflict / below), not gold / pilot / anchor, not already in p2_s1..s6, in random
order (seed 21), so any prefix is a random sample of the stratum. Same panel p2. OAI: no.
Appends session 7 rows to _key/p2s.csv (stratum frac_vague), writes _batches/p2_s7.txt and _reports/<index>.txt.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase2_frac.py [--n 200]"""
import argparse

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
st_csv = pd.read_csv(D / "_prompts/studies.csv")
full = st_csv[st_csv.sheets.str.count(";") >= 7].drop_duplicates("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
key = pd.read_csv(D / "_key/p2s.csv")
assert set(key.session) == set(range(1, 7)), "session 7 already built"
used = set(pd.read_csv(D / "_key/pilot.csv")["index"]) | {2507, 3297, 3831} | set(key["index"])
pool = full[~full.study_id.isin(gold) & ~full["index"].isin(used) & full.study_id.isin(meta.index[meta.Report.fillna("").str.strip() != ""])]
st = status_table("v1").reindex(pool.study_id)
vague = st["fracture"].isin({"below", "unsettled", "hedged", "conflict"}).to_numpy()
idx = np.flatnonzero(vague)
print(f"pool {len(pool)}: fracture-vague {len(idx)} ({st['fracture'][vague].value_counts().to_dict()})")
rng = np.random.default_rng(21)
rng.shuffle(idx)
idx = idx[:a.n]
new = pd.DataFrame({"index": pool["index"].to_numpy()[idx], "study_id": pool.study_id.to_numpy()[idx], "stratum": "frac_vague", "session": 7,
                    "pos": np.arange(len(idx))})
assert not new.study_id.isin(gold).any() and not new["index"].isin(key["index"]).any()
pd.concat([key, new]).to_csv(D / "_key/p2s.csv", index=False)
(D / "_batches/p2_s7.txt").write_text("\n".join(str(i) for i in new["index"]) + "\n")
for i, sid in zip(new["index"], new.study_id):
    (D / f"_reports/{i:04d}.txt").write_text(meta.Report[sid].strip() + "\n")
print(f"p2_s7: {len(new)} studies")
