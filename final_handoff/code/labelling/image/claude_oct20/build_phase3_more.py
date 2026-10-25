"""Phase 3 extension (USER 10-20 23:4x: "give me 20 prompts to create 20 sessions with 20 studies each so i maximise probability of
reaching 200+ labelled positives for each of 3 classes"). Contusion is done (231). This rebuilds the never-started PF sessions 14-24
together with NEW studies into 20 sessions of 20 lines (sessions 14..33):
- every unsaved line of sessions 14-24 (the PF block + its re-reads);
- the next N_PF report-"meets" PF OA studies by P (positive_yield_pool.csv), not listed anywhere yet;
- the best N_F remaining fracture candidates by P (the uncertain fracture pool is used up: ~+3 expected).
Order in every session: PF lines best-first by P (old and new merged), then the fracture lines; dealt round-robin so every session
reaches the same milestones together. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_more.py"""
import json

import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
N_PF, N_F, FIRST, CHATS, MAX = 150, 33, 14, 20, 20
key = pd.read_csv(D / "_key/p3.csv")
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
read = set(json.load(open(REPO / "data/image_labels/opus_p2.json"))) | set(json.load(open(REPO / "data/image_labels/opus_pilot5_gold.json")))
saved = {json.load(open(p))["index"] for p in D.glob("p3_s*/*.json")}
anchors = {2507, 3297, 3831}
old = key[key.session.between(FIRST, 24)]
assert not old["index"].isin(saved).any(), "a PF-block session has started; do not rebuild"
cand = pool[~pool.index.isin(set(key.study_id) | read | gold) & ~studies.loc[pool.index, "index"].isin(anchors).to_numpy()]
cand = cand[[str(meta.Report.get(s, "") or "").strip() != "" for s in cand.index]]
new_pf = cand[cand.grp_pf_oa == "meets"].p_pf_oa.nlargest(N_PF).index
new_f = cand.drop(new_pf).p_fracture.nlargest(N_F).index
new = pd.DataFrame({"study_id": list(new_pf) + list(new_f), "findings": ["pf_oa"] * len(new_pf) + ["fracture"] * len(new_f),
                    "kind": "new", "block": ["pf_oa"] * len(new_pf) + ["fracture"] * len(new_f)})
new["expected_pos"] = [pool.at[s, f"p_{b}"] for s, b in zip(new.study_id, new.block)]
new["index"] = studies.loc[new.study_id, "index"].to_numpy().astype(int)
lines = pd.concat([old.drop(columns=["session", "pos"]), new], ignore_index=True)
lines["bp"] = [pool.at[s, f"p_{b}"] if s in pool.index else 0.0 for s, b in zip(lines.study_id, lines.block)]
assert len(lines) == CHATS * MAX and lines.study_id.is_unique, (len(lines), lines.study_id.duplicated().sum())
out = []
for b in ("pf_oa", "fracture"):
    g = lines[lines.block == b]
    nw, rr = g[g.kind == "new"].sort_values("bp", ascending=False), g[g.kind == "reread"]
    for c in range(CHATS):
        part = nw.iloc[c::CHATS].to_dict("records")
        for j, r in enumerate(rr.iloc[c::CHATS].to_dict("records")):
            part.insert(min(len(part), 3 + 7 * j), r)
        out += [r | {"session": FIRST + c} for r in part]
moved = pd.DataFrame(out)
moved["pos"] = moved.groupby("session").cumcount()
assert moved.groupby("session").size().max() <= MAX + 1 and set(moved["index"]) == set(lines["index"])
key = pd.concat([key[~key.session.between(FIRST, 24)], moved.drop(columns="bp")])[key.columns].sort_values(["session", "pos"])
assert key["index"].is_unique and not key.study_id.isin(gold).any()
kf = D / "_key/p3.csv"
kf.chmod(0o644)
key.to_csv(kf, index=False)
kf.chmod(0o444)
for s in range(FIRST, FIRST + CHATS):
    f = D / f"_batches/p3_s{s}.txt"
    if f.exists():
        f.chmod(0o644)
    g = key[key.session == s]
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
for i, sid in zip(new["index"], new.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[sid].strip() + "\n")
n = key[key.session >= FIRST]
for f in ("pf_oa", "fracture"):
    m = n[(n.kind == "new") & n.findings.str.contains(f)]
    e = sum(pool.at[s, f"p_{f}"] if s in pool.index else .82 for s in m.study_id)
    print(f"{f}: {len(m)} lines, model-expected +{e:.0f} (x0.8 calibration +{0.8 * e:.0f})")
print("sessions", FIRST, "..", FIRST + CHATS - 1, "lines per session", n.groupby("session").size().to_dict())
