"""Effusion set-up search sets (USER 10-19 ~19:4x: find the best VLM labelling set-up first, then gold).
- out/esearch_studies.csv: the 299 pilot studies (indexes 0-298) + the Opus Effusion second-read studies (384 non-gold, text v1 Effusion
  below / unsettled; indexes 5000+; Opus = an image reader independent of Codex and of the VLM).
- out/eshots.csv + out/eshots_studies.csv: text-explicit in-context examples for Effusion (2 v1 'meets' p >= .85, 2 'structure_normal' /
  'denied' / 'general_normal' p <= .1; non-gold, not in either set; indexes 6000+). No gold, no image reader used to pick them."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels, probability_table, status_table

HERE = Path(__file__).parent
REPO = HERE.parents[1]
pilot = pd.read_csv(HERE / "out/pilot_studies.csv", index_col="index")
op = json.load(open(REPO / "data/image_labels/opus_effusion.json"))
opus = [u for u in op if u not in set(pilot.StudyInstanceUID)]
oset = pd.DataFrame({"StudyInstanceUID": opus, "stratum": "opus"}, index=pd.RangeIndex(5000, 5000 + len(opus), name="index"))
pd.concat([pilot, oset]).to_csv(HERE / "out/esearch_studies.csv")
st, pr = status_table("v1")["effusion"], probability_table("v1")["effusion"]
used = set(pilot.StudyInstanceUID) | set(opus) | set(pd.read_csv(HERE / "out/shots_studies.csv").StudyInstanceUID)
pool = st.index.difference(gold_labels().index).difference(list(used))
rng = np.random.default_rng(11)
pos = rng.choice(pool[((st[pool] == "meets") & (pr[pool] >= 0.85)).to_numpy()], 2, replace=False)
neg = rng.choice(pool[(st[pool].isin(["structure_normal", "denied", "general_normal"]) & (pr[pool].fillna(0) <= 0.1)).to_numpy()], 2, replace=False)
uids = [pos[0], neg[0], pos[1], neg[1]]
pd.DataFrame({"StudyInstanceUID": uids, "stratum": "eshot"}, index=pd.RangeIndex(6000, 6004, name="index")).to_csv(HERE / "out/eshots_studies.csv")
pd.DataFrame({"index": range(6000, 6004), "finding": "Effusion", "answer": ["Yes", "No", "Yes", "No"]}).to_csv(HERE / "out/eshots.csv", index=False)
print(f"esearch: {len(pilot)} pilot + {len(oset)} opus studies; eshots 4")
