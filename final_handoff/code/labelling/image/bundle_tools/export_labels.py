"""Export image labels as training targets for experiments/v0/oct7_baseline (masked loss / image head).

PYTHONPATH=.:work/image_labeling conda run -n rsna python work/image_labeling/export_labels.py RUN [RUN ...] --out FILE.csv
e.g.  ... export_labels.py codex_synovitis codex_synovitis_b1 codex_synovitis_b2 codex_synovitis_b3 codex_synovitis_b4 \
          --out work/image_labeling/out/synovitis_labels_codex.csv

Rows: one per labelled non-gold study (catch items and gold dropped). Columns:
- StudyInstanceUID; one column per labelled finding named as common.paths.FINDINGS ("Synovitis", "Effusion") = probability from mapping.py
  (blank = not assessable); several RUNs labelling the same study are averaged;
- <Finding>_blockrank: percentile rank (0-1, ties averaged) within blocks of --block studies in labelling order inside each folder;
  a second read (synovitis_rK) is ranked within the block of the study's first read; ranks of several reads are averaged;
  Within a folder the order is random with respect to the studies, so this removes drift in how a labeller words its certainty over
  a long chat (Codex says "possible" more often late in a chat);
- reads (number of synovitis reads), folder (source label folder(s)) and sampling: "queue" (the random 3:1 synovitis queue) or "batch" (targeted draws within high-value
  text cells, folders synovitis_b*);
- stratum (queue stratum: priority / control), cell (text v1 effusion status / synovitis status);
- weight: post-stratification weight per (stratum, cell) group = (non-gold studies in the group / labelled in the group), scaled to
  mean 1. The queue sampled at random within strata and the 2026-10-08 batch at random within cells, so within each group the
  labelled studies are a random sample and the weights make the labelled set represent all non-gold studies (groups with no labelled
  study cannot be represented)."""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import FINDING_OF
from evaluate import HERE, sources
from mapping import probability

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=75)
a = ap.parse_args()
order = pd.read_csv(HERE / "out/order.csv", index_col="index")
rows = []
for run, folder, _, members in sources():
    if run not in a.runs:
        continue
    for pos, (_, data, t) in enumerate(sorted(members, key=lambda m: m[2])):
        x = json.loads(data)
        i = x["index"]
        if order.section[i] == "gold" or order.catch[i]:
            continue
        for k, f in x["findings"].items():
            rows.append(dict(StudyInstanceUID=order.source[i], finding=FINDING_OF[k], p=probability(f["status"], f["certainty"]),
                             block=f"{folder}#{pos // a.block}", folder=folder.split("/")[-1]))
if not rows:
    sys.exit(f"no labels for runs {a.runs}")
d = pd.DataFrame(rows)
# a second read (folders synovitis_rK) is ranked within the block of the study's first read: re-read blocks hold only uncertain first
# reads, so ranks inside them would not be comparable with the first reads' ranks
reread = d.folder.str.fullmatch(r"synovitis_r\d+")
first = d[~reread]
d.loc[~reread, "rank"] = first.groupby(["block", "finding"]).p.rank(pct=True)
ref = first.groupby(["block", "finding"]).p.apply(lambda v: np.sort(v.dropna().to_numpy()))
home = first.drop_duplicates(["StudyInstanceUID", "finding"]).set_index(["StudyInstanceUID", "finding"]).block
for i, x in d[reread].iterrows():
    v, b = x.p, home.get((x.StudyInstanceUID, x.finding))
    if b is None or v is None or np.isnan(v):
        continue
    pr = ref[(b, x.finding)]
    d.loc[i, "rank"] = (np.sum(pr < v) + (np.sum(pr == v) + 1) / 2) / (len(pr) + 1)   # as pandas pct rank with v inserted
t = d.pivot_table(index="StudyInstanceUID", columns="finding", values="p", aggfunc="mean")
r = d.pivot_table(index="StudyInstanceUID", columns="finding", values="rank", aggfunc="mean").add_suffix("_blockrank")
t = t.join(r.round(4))
queue = order[(order.section != "gold") & ~order.catch].drop_duplicates("source").set_index("source")
st = status_table("v1")
group = queue.stratum + " | " + (st.effusion + " / " + st.synovitis).reindex(queue.index)
folders = d.groupby("StudyInstanceUID").folder.agg(lambda f: ";".join(sorted(set(f))))
t["folder"] = folders.reindex(t.index)
t["sampling"] = t.folder.map(lambda f: "batch" if all(re.fullmatch(r"synovitis_b\d+", x) for x in f.split(";")
                                                       if not re.fullmatch(r"synovitis_r\d+", x)) else "queue")
t["reads"] = d[d.finding == "Synovitis"].groupby("StudyInstanceUID").size().reindex(t.index)
t["stratum"] = queue.stratum.reindex(t.index)
t["cell"] = (st.effusion + " / " + st.synovitis).reindex(t.index)
g = group.reindex(t.index)
w = g.map(group.value_counts() / g.value_counts())
t["weight"] = (w / w.mean()).round(4)
t.reset_index().to_csv(a.out, index=False)
print(f"{len(t)} studies -> {a.out} | strata {t.stratum.value_counts().to_dict()} | groups with labels {g.nunique()} of {group.nunique()}"
      f" | weight range {t.weight.min():.2f}-{t.weight.max():.2f}")
