"""Phase 3: one-finding, per-study panels for 6 new Claude chats (USER 10-20 17:2x: "give me 6 prompts for 6 new labelling sessions";
plan = the cheapest tokens per new positive, positive_yield.py). OAI: no. No gold.
Selection from positive_yield_pool.csv (P(Opus positive) per unread study and finding, text v1 group):
- contusion: report not "meets" (vague / excluded / silent / negative), the top N_C by P  -> new information;
- fracture : report "meets" (all, incl. 8 outside the scored pool) + any study with P >= .1 -> the uncertain pool is used up;
- pf_oa    : report "meets", the top N_PF_MEETS by P + vague, the top N_PF_VAGUE by P (Opus corrects ~half of PF "meets").
A study is read only for its selected findings (p3 panel; usually one). USER 10-20 17:2x: "build lists in such a way that when they run in
parallel we arrive to at least one class having 200+ positives as soon as possible": every list runs in BLOCKS - first every study with
contusion (it needs ~+62 for 200; + the top N_C_MEETS report-"meets" confirmations as margin), then pf_oa (reaches ~210), then fracture
(~190: its uncertain pool is used up). Inside a block studies are best-first by P and dealt round-robin, so all chats reach the
same block milestones together. Each chat also gets 4 blind RE-READS of studies already read in
phase 2 (one finding each; per finding 4 Opus-positive + 4 negative; fracture negatives from report-vague studies) to calibrate p3 (fewer sheets) against p2.
Writes _batches/p3_s1..s6.txt ("INDEX finding[,finding]"), _key/p3.csv, _reports/<index>.txt.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3.py"""
import json

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
N_C, N_C_MEETS, N_PF_MEETS, N_PF_VAGUE, F_MIN_P, CHATS, REREADS = 70, 30, 150, 62, .1, 8, 3   # 10-20 18:1x: 8 chats, fracture to 200+
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
p2 = json.load(open(REPO / "data/image_labels/opus_p2.json"))
pilot = set(json.load(open(REPO / "data/image_labels/opus_pilot5_gold.json")))
anchors = {2507, 3297, 3831}
assert not set(pool.index) & (gold | pilot | set(p2)), "pool must be unread non-gold studies"
pool = pool[~studies.loc[pool.index, "index"].isin(anchors).to_numpy()]   # the 3 text anchors are shown to every chat
# USER 10-20 18:1x "200+ positives for each of 3 classes": Fracture needs nearly every candidate. Add the unread report-"meets" fracture
# studies outside the scored pool (fewer sheets or no package OOF; report present), at the pooled meets rate 0.82.
st0 = status_table("v1")
extra = [s for s in studies.index if s not in pool.index and s not in gold | pilot | set(p2) and s in st0.index and st0.at[s, "fracture"] == "meets"
         and str(meta.Report.get(s, "") or "").strip() != "" and studies.at[s, "index"] not in anchors]
pool = pd.concat([pool, pd.DataFrame({"p_contusion": 0.0, "p_fracture": .82, "p_pf_oa": 0.0, "grp_contusion": "extra", "grp_fracture": "meets",
                                      "grp_pf_oa": "extra"}, index=pd.Index(extra, name=pool.index.name))])
print(f"extra report-meets fracture studies outside the pool: {len(extra)}")

sel = {"contusion": pool[pool.grp_contusion != "meets"].p_contusion.nlargest(N_C).index.union(
           pool[pool.grp_contusion == "meets"].p_contusion.nlargest(N_C_MEETS).index),
       "fracture": pool[(pool.grp_fracture == "meets") | (pool.p_fracture >= F_MIN_P)].index,
       "pf_oa": pool[pool.grp_pf_oa == "meets"].p_pf_oa.nlargest(N_PF_MEETS).index.union(
           pool[pool.grp_pf_oa == "vague"].p_pf_oa.nlargest(N_PF_VAGUE).index)}
rows = {}
for f, ids in sel.items():
    for s in ids:
        rows.setdefault(s, []).append(f)
new = pd.DataFrame({"study_id": list(rows), "findings": [",".join(sorted(v)) for v in rows.values()]})
new["expected_pos"] = [sum(pool.at[s, f"p_{f}"] for f in v.split(",")) for s, v in zip(new.study_id, new.findings)]
new["kind"] = "new"
BLOCKS = ("contusion", "fracture", "pf_oa")   # 10-20 18:2x: fracture before PF (Opus PF strictness; PF last)
new["block"] = [next(f for f in BLOCKS if f in v.split(",")) for v in new.findings]
new["block_p"] = [pool.at[s, f"p_{b}"] for s, b in zip(new.study_id, new.block)]
new = new.sort_values(["block", "block_p"], key=lambda c: c.map(BLOCKS.index) if c.name == "block" else -c).reset_index(drop=True)

# blind re-reads of phase-2 studies (calibration of the p3 protocol): per finding, Opus-positive and negative, from the strata p3 samples
rng = np.random.default_rng(30)
rr = []
st = status_table("v1")
GRP = {"meets": {"meets"}, "vague": {"unsettled", "hedged", "below", "conflict"}}
for f, gpos, gneg in (("contusion", "vague", "vague"), ("fracture", "meets", "vague"), ("pf_oa", "meets", "meets")):   # report-"meets" fracture is ~95 % positive
    cand = [s for s, r in p2.items() if r["findings"][f]["probability"] is not None and s in studies.index and s in st.index]
    pos = sorted(s for s in cand if st.at[s, f] in GRP[gpos] and p2[s]["findings"][f]["probability"] >= .5)
    neg = sorted(s for s in cand if st.at[s, f] in GRP[gneg] and p2[s]["findings"][f]["probability"] < .5)
    k = CHATS * REREADS // 3
    for s in list(rng.choice(pos, k // 2, replace=False)) + list(rng.choice(neg, k - k // 2, replace=False)):
        rr.append({"study_id": s, "findings": f, "expected_pos": np.nan, "kind": "reread", "block": f})
rr = pd.DataFrame(rr).sample(frac=1, random_state=31).sort_values("block", key=lambda c: c.map(BLOCKS.index), kind="stable").reset_index(drop=True)
assert rr.study_id.is_unique and not set(rr.study_id) & set(new.study_id)

out = []
for c in range(CHATS):
    part = []
    for b in BLOCKS:                                  # per block: its studies dealt round-robin, its re-reads inside the block (3rd, 9th item)
        blk = new[new.block == b].iloc[c::CHATS].to_dict("records")
        for j, r in enumerate(rr[rr.block == b].iloc[c::CHATS].to_dict("records")):
            blk.insert(min(len(blk), 2 + 6 * j), r)
        part += blk
    for pos, r in enumerate(part):
        out.append(r | {"session": c + 1, "pos": pos})
key = pd.DataFrame(out)
key["index"] = studies.loc[key.study_id, "index"].to_numpy().astype(int)
assert key.study_id.is_unique and not key["index"].isin(anchors).any() and not key.study_id.isin(gold | pilot).any()
assert (key.kind == "reread").sum() == len(rr) and key[key.kind == "new"].study_id.isin(set(p2)).sum() == 0
assert (key.study_id.map(lambda s: str(meta.Report.get(s, "") or "").strip() != "")).all(), "every study needs a report"
assert (studies.loc[key.study_id, "sheets"].str.count(";") >= 3).all()   # extras may lack a plane; validate.py asks only for sheets the study has
key[["index", "study_id", "findings", "kind", "block", "session", "pos", "expected_pos"]].to_csv(D / "_key/p3.csv", index=False)
for c, g in key.groupby("session"):
    (D / f"_batches/p3_s{c}.txt").write_text("".join(f"{i} {f}\n" for i, f in zip(g["index"], g.findings)))
for i, s in zip(key["index"], key.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[s].strip() + "\n")
n = key[key.kind == "new"]
print(f"new studies {len(n)} (finding-reads {n.findings.str.count(',').add(1).sum()}), re-reads {len(rr)}")
for f in ("contusion", "fracture", "pf_oa"):
    m = n.findings.str.contains(f)
    print(f"  {f}: {m.sum()} studies, expected new positives {sum(pool.at[s, f'p_{f}'] for s in n.study_id[m]):.0f}")
for c, g in key.groupby("session"):
    print(f"  p3_s{c}: {len(g)} lines ({(g.kind == 'reread').sum()} re-reads); blocks " + ", ".join(f"{b} {(g.block == b).sum()}" for b in BLOCKS))
# race: expected positives per class after the first k lines of EVERY chat (all chats in parallel), from 811 reads
base = {"contusion": 138, "fracture": 72, "pf_oa": 72}
for k in (5, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80):
    g = n[n.pos < k]
    cum = {f: base[f] + sum(pool.at[s, f"p_{f}"] for s, v in zip(g.study_id, g.findings) if f in v.split(",")) for f in base}
    print(f"  after {k:2d} lines per chat ({CHATS * k} reads): " + ", ".join(f"{f} {v:.0f}" for f, v in cum.items()))
