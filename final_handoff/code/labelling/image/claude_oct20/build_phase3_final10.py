"""Phase 3, last batch (USER 10-20 23:5x: "just give me prompts to copy paste. up to 20 studies in each session ... we can make 10
sessions, with the best value for each of 3 classes in question"). Replaces the never-started sessions 14-33 with 10 sessions (14-23)
of 20 studies, the best value per class from positive_yield_pool.csv (unread, unlisted, non-gold studies with a report):
- contusion: the next report-not-"meets" studies by P (informative; the column with the best new-label lead);
- pf_oa: the top report-"meets" studies by P (each adds ~0.9 positives under the 0.5 Opus + 0.5 report blend);
- fracture: the best remaining candidates by P (the uncertain pool is used up).
Riders: a selected study also gets another finding when its P for it is >= RIDE (cheap, as the study is open anyway). Classes are
interleaved best-first and dealt round-robin, so every prefix of every list has the same class mix. No re-reads (calibration is
done: 16, same side 0.94). No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_final10.py"""
import json

import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
N = {"contusion": 80, "pf_oa": 80, "fracture": 40}
RIDE = {"contusion": .3, "pf_oa": .3, "fracture": .1}
FIRST, CHATS, MAX = 14, 10, 20
key = pd.read_csv(D / "_key/p3.csv")
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
read = set(json.load(open(REPO / "data/image_labels/opus_p2.json"))) | set(json.load(open(REPO / "data/image_labels/opus_pilot5_gold.json")))
saved = {json.load(open(p))["index"] for p in D.glob("p3_s*/*.json")}
assert not key[key.session >= FIRST]["index"].isin(saved).any(), "a session >= 14 has started; do not rebuild"
kept = key[key.session < FIRST]
anchors = {2507, 3297, 3831}
c = pool[~pool.index.isin(set(kept.study_id) | read | gold)]
c = c[~studies.loc[c.index, "index"].isin(anchors).to_numpy() & [str(meta.Report.get(s, "") or "").strip() != "" for s in c.index]]
sel = {"contusion": c[c.grp_contusion != "meets"].p_contusion.nlargest(N["contusion"]),
       "pf_oa": c[c.grp_pf_oa == "meets"].p_pf_oa.nlargest(N["pf_oa"])}
sel["fracture"] = c.drop(sel["contusion"].index.union(sel["pf_oa"].index)).p_fracture.nlargest(N["fracture"])
rows = {}
for f in ("contusion", "pf_oa", "fracture"):               # class order inside a study only fixes the primary class
    for rank, s in enumerate(sel[f].index):
        rows.setdefault(s, {"primary": f, "rank": rank, "findings": set()})["findings"].add(f)
for s, r in rows.items():                                    # riders
    for f, t in RIDE.items():
        if pool.at[s, f"p_{f}"] >= t:
            r["findings"].add(f)
lines = pd.DataFrame([{"study_id": s, "findings": ",".join(sorted(r["findings"])), "block": r["primary"], "rank": r["rank"]}
                      for s, r in rows.items()])
lines["kind"] = "new"
lines["expected_pos"] = [sum(pool.at[s, f"p_{f}"] for f in fs.split(",")) for s, fs in zip(lines.study_id, lines.findings)]
lines["index"] = studies.loc[lines.study_id, "index"].to_numpy().astype(int)
assert lines.study_id.is_unique and len(lines) <= CHATS * MAX, len(lines)
lines = lines.sort_values(["rank", "block"], key=lambda col: col.map({"contusion": 0, "pf_oa": 1, "fracture": 2}) if col.name == "block" else col)
lines = lines.reset_index(drop=True)
lines["session"] = FIRST + lines.index % CHATS                # interleaved best-first, dealt round-robin
lines["pos"] = lines.groupby("session").cumcount()
assert lines.groupby("session").size().max() <= MAX
new_key = pd.concat([kept, lines[key.columns]]).sort_values(["session", "pos"])
assert new_key["index"].is_unique and not new_key.study_id.isin(gold).any()
kf = D / "_key/p3.csv"
kf.chmod(0o644)
new_key.to_csv(kf, index=False)
kf.chmod(0o444)
for s in range(FIRST, 34):                                    # old never-started lists 14-33
    f = D / f"_batches/p3_s{s}.txt"
    if f.exists():
        f.chmod(0o644)
        f.unlink()
for s, g in lines.groupby("session"):
    f = D / f"_batches/p3_s{s}.txt"
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
for i, sid in zip(lines["index"], lines.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[sid].strip() + "\n")
for f in ("contusion", "pf_oa", "fracture"):
    m = lines[lines.findings.str.contains(f)]
    e = sum(pool.at[s, f"p_{f}"] for s in m.study_id)
    print(f"{f}: {len(m)} finding-reads ({(lines.block == f).sum()} primary), model-expected Opus positives +{e:.0f} (x0.8 +{0.8 * e:.0f})")
print(f"{len(lines)} studies, sessions {FIRST}..{FIRST + CHATS - 1}: {lines.groupby('session').size().to_dict()}")
