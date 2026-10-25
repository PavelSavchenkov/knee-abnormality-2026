"""Phase 2 for 6 parallel Claude chats (USER 10-20: "i'll run 6 labelling claude sessions ... maintain needed split between report vague,
report silent and random images, so i can stop at any time and test"). OAI: no.
Same selection as build_phase2.py (panel p2 = contusion + fracture + pf_oa; non-gold, non-pilot, non-anchor studies with a report):
 vague 60 % (text v1 Contusion OR PF status vague) / silent_mid 20 % (silent and 3x3 OOF p in [0.15, 0.85]) / random 20 %.
Each chat's list is interleaved V V S V R, so ANY prefix of every list keeps ~60 / 20 / 20.
Writes _batches/p2_s1..s6.txt, _key/p2s.csv (index, study_id, stratum, session, pos), _reports/<index>.txt.
  conda run -n rsna python work/claude_labels_oct20/build_phase2_6.py [--n 1500]"""
import argparse

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--chats", type=int, default=6)
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
vague = st["contusion"].isin(VAGUE).to_numpy() | st["pf_oa"].isin(VAGUE).to_numpy()
silent_mid = ((st["contusion"].isna() | (st["contusion"] == "not_mentioned")).to_numpy() & oof["Contusion"].between(.15, .85).to_numpy()) | \
             ((st["pf_oa"].isna() | (st["pf_oa"] == "not_mentioned")).to_numpy() & oof["PF OA"].between(.15, .85).to_numpy())
print(f"pool {len(pool)}: vague {vague.sum()}, silent_mid (not vague) {(silent_mid & ~vague).sum()}")
rng = np.random.default_rng(20)
strata = {}
for name, m, cap in (("vague", vague, int(a.n * 0.6)), ("silent_mid", silent_mid & ~vague, int(a.n * 0.2))):
    idx = np.flatnonzero(m)
    rng.shuffle(idx)
    assert len(idx) >= cap, f"{name}: only {len(idx)} studies for {cap}"
    strata[name] = idx[:cap]
rest = np.setdiff1d(np.arange(len(pool)), np.concatenate(list(strata.values())))
rng.shuffle(rest)
strata["random"] = rest[:a.n - sum(len(v) for v in strata.values())]
rows = []
for s in range(a.chats):
    q = {k: list(v[s::a.chats]) for k, v in strata.items()}
    pos = 0
    while any(q.values()):
        for name in ("vague", "vague", "silent_mid", "vague", "random"):
            if q[name]:
                i = q[name].pop(0)
                rows.append((pool["index"].iat[i], pool.study_id.iat[i], name, s + 1, pos))
                pos += 1
key = pd.DataFrame(rows, columns=["index", "study_id", "stratum", "session", "pos"])
assert key["index"].is_unique and not key.study_id.isin(gold).any()
key.to_csv(D / "_key/p2s.csv", index=False)
for s, g in key.groupby("session"):
    (D / f"_batches/p2_s{s}.txt").write_text("\n".join(str(i) for i in g.sort_values("pos")["index"]) + "\n")
for i, sid in zip(key["index"], key.study_id):
    (D / f"_reports/{i:04d}.txt").write_text(meta.Report[sid].strip() + "\n")
for s, g in key.groupby("session"):
    g = g.sort_values("pos")
    first = {n: (g.stratum.iloc[:n] == "vague").mean() for n in (10, 50, len(g))}
    print(f"p2_s{s}: {len(g)} studies {g.stratum.value_counts().to_dict()}; vague share in the first 10 / 50 / all: "
          + " / ".join(f"{v:.2f}" for v in first.values()))
