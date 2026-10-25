"""Phase 3, contusion batch (USER 10-21 01:2x: 20 sessions x 20 studies; picked "Contusion + Baker's"). 12 sessions (24-35) of 20:
every remaining unread report-"meets" Contusion study (~177) + the best report-not-"meets" ones by P to fill 240, from
positive_yield_pool.csv. Riders: pf_oa if its P >= .3, fracture if >= .1 (the study is open anyway). Best-first, dealt round-robin. No
re-reads. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_cont12.py"""
import json

import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
FIRST, CHATS, MAX = 24, 12, 20
RIDE = {"pf_oa": .3, "fracture": .1}
key = pd.read_csv(D / "_key/p3.csv")
assert key.session.max() < FIRST, "sessions >= 24 already exist"
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
read = set(json.load(open(REPO / "data/image_labels/opus_p2.json"))) | set(json.load(open(REPO / "data/image_labels/opus_pilot5_gold.json")))
anchors = {2507, 3297, 3831}
c = pool[~pool.index.isin(set(key.study_id) | read | gold)]
c = c[~studies.loc[c.index, "index"].isin(anchors).to_numpy() & [str(meta.Report.get(s, "") or "").strip() != "" for s in c.index]]
meets = c[c.grp_contusion == "meets"].p_contusion.sort_values(ascending=False)
inf = c[c.grp_contusion != "meets"].p_contusion.nlargest(CHATS * MAX - len(meets))
sel = pd.concat([meets, inf]).sort_values(ascending=False)
rows = []
for s in sel.index:
    fs = {"contusion"} | {f for f, t in RIDE.items() if pool.at[s, f"p_{f}"] >= t}
    rows.append({"study_id": s, "findings": ",".join(sorted(fs)), "kind": "new", "block": "contusion",
                 "expected_pos": sum(pool.at[s, f"p_{f}"] for f in fs), "index": int(studies.at[s, "index"])})
lines = pd.DataFrame(rows)
lines["session"] = FIRST + lines.index % CHATS
lines["pos"] = lines.groupby("session").cumcount()
assert len(lines) <= CHATS * MAX and lines.study_id.is_unique
new_key = pd.concat([key, lines[key.columns]]).sort_values(["session", "pos"])
assert new_key["index"].is_unique and not new_key.study_id.isin(gold).any()
kf = D / "_key/p3.csv"
kf.chmod(0o644)
new_key.to_csv(kf, index=False)
kf.chmod(0o444)
for s, g in lines.groupby("session"):
    f = D / f"_batches/p3_s{s}.txt"
    assert not f.exists(), f
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
for i, sid in zip(lines["index"], lines.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[sid].strip() + "\n")
for f in ("contusion", "pf_oa", "fracture"):
    m = lines[lines.findings.str.contains(f)]
    print(f"{f}: {len(m)} reads, model-expected +{sum(pool.at[s, f'p_{f}'] for s in m.study_id):.0f}")
print(f"{len(lines)} studies ({len(meets)} report-meets + {len(inf)} others); sessions {FIRST}..{FIRST + CHATS - 1}")
