"""Phase 2 of the Claude (Opus 5.5) image + report labelling (Master 10-20; prepared for the USER, who starts the chats; OAI: no).
Pilot evidence (eval_deep.txt, gold-58): Contusion = the one class where Opus image + report beats the target in use (tgt_v3 +3.0, text v1
+8.5); PF = the one class where package + Opus improves gold (+0.024, 90 % [+0.006, +0.049]) = the distillation candidate.
Panel 'p2' = contusion + fracture + pf_oa (USER 10-20: 'why not fracture as well?': tgt_v3's gold read is leak-flattered; gold sides with Opus on 10 / 15 package disagreements; near-zero marginal cost). ~800 non-gold bundle studies, 3 chats; never gold, never a pilot / anchor index.
Selection, priority first (each study once):
 (a) 60 %: text v1 Contusion OR PF status vague (selection unchanged; Fracture rides along) (below / unsettled / hedged / conflict);
 (b) 20 %: silent on either finding with the 3x3 generalist OOF probability of that finding in [0.15, 0.85] (models unsure);
 (c) 20 %: a random fill of the remaining pool (seed 20).
Writes _batches/p2_c1..c3.txt, _key/p2.csv (stratum), _reports/<index>.txt, and the panel schema _prompts/schema_p2.json registered in the
_prompts copy of validate.py.
  conda run -n rsna python work/claude_labels_oct20/build_phase2.py [--n 800]"""
import argparse
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=800)
a = ap.parse_args()
st_csv = pd.read_csv(D / "_prompts/studies.csv")
full = st_csv[st_csv.sheets.str.count(";") >= 7].drop_duplicates("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
used = set(pd.read_csv(D / "_key/pilot.csv")["index"]) | {2507, 3297, 3831}
pool = full[~full.study_id.isin(gold) & ~full["index"].isin(used) & full.study_id.isin(meta.index[meta.Report.fillna("").str.strip() != ""])]
st = status_table("v1").reindex(pool.study_id)
mem = pd.read_parquet(REPO / "experiments/v3/oct16_experts/refs/pkg961/members.parquet")
mo = mem[mem.set == "oof"]
oof = {c: (sum(1 / (1 + np.exp(-mo[mo.member == m].groupby("StudyInstanceUID")[c].mean())) for m in ("main", "nfnet", "r50")) / 3).reindex(pool.study_id)
       for c in ("Contusion", "PF OA")}
VAGUE = {"below", "unsettled", "hedged", "conflict"}
sid = pool.study_id.to_numpy()
vague = st["contusion"].isin(VAGUE).to_numpy() | st["pf_oa"].isin(VAGUE).to_numpy()
silent_mid = ((st["contusion"].isna() | (st["contusion"] == "not_mentioned")).to_numpy() & oof["Contusion"].between(.15, .85).to_numpy()) | \
             ((st["pf_oa"].isna() | (st["pf_oa"] == "not_mentioned")).to_numpy() & oof["PF OA"].between(.15, .85).to_numpy())
rng = np.random.default_rng(20)
order = []
for name, m, cap in (("vague", vague, int(a.n * 0.6)), ("silent_mid", silent_mid & ~vague, int(a.n * 0.2))):
    idx = np.flatnonzero(m)
    rng.shuffle(idx)
    order += [(i, name) for i in idx[:cap]]   # caps keep a broad sample for the PF distillation member (60 / 20 / 20 %)
rest = np.setdiff1d(np.arange(len(pool)), [i for i, _ in order])
rng.shuffle(rest)
order += [(i, "random") for i in rest]
order = order[:a.n]
key = pd.DataFrame({"index": pool["index"].to_numpy()[[i for i, _ in order]], "study_id": sid[[i for i, _ in order]], "stratum": [s for _, s in order]})
key.to_csv(D / "_key/p2.csv", index=False)
for k in range(3):
    (D / f"_batches/p2_c{k + 1}.txt").write_text("\n".join(str(i) for i in key["index"].iloc[k::3]) + "\n")
for i, s in zip(key["index"], key.study_id):
    (D / f"_reports/{i:04d}.txt").write_text(meta.Report[s].strip() + "\n")
P = D / "_prompts"
os.chmod(P, os.stat(P).st_mode | stat.S_IWUSR)
sch = json.load(open(P / "schema_pilot5.json"))
sch["title"] = sch["title"].split(" (pilot5")[0] + " (p2 panel: contusion, pf_oa)"
sch["properties"]["panel"] = {"enum": ["p2"]}
f = sch["properties"]["findings"]
f["properties"] = {k: v for k, v in f["properties"].items() if k in ("contusion", "pf_oa")}
f["required"] = ["pf_oa", "contusion"]
if not (P / "schema_p2.json").exists():
    (P / "schema_p2.json").write_text(json.dumps(sch, indent=1))
v = (P / "validate.py").read_text()
if '"p2":' not in v:
    os.chmod(P / "validate.py", stat.S_IRUSR | stat.S_IWUSR)
    v = v.replace('PANELS = {"pilot5":', 'PANELS = {"p2": (json.loads((HERE / "schema_p2.json").read_text(encoding="utf-8")), None),   # oct20 phase 2: contusion + pf_oa\n          "pilot5":')
    (P / "validate.py").write_text(v)
for p in (P / "validate.py", P / "schema_p2.json"):
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
os.chmod(P, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
print(f"phase 2: {len(key)} studies {key.stratum.value_counts().to_dict()} -> p2_c1..c3 ({[len(key) // 3 + (k < len(key) % 3) for k in range(3)]})")
