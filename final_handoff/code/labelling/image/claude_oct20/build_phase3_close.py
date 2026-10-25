"""Phase 3 'close the yardstick' batch (USER 10-21 02:3x: the dense referee = new label where read, old label elsewhere is valid only if
almost every study where the new label matters is read). Every non-gold study whose dense v3e bone target (Opus where read, tgt_v3 elsewhere)
is >= .3 for Contusion or Fracture but has no Opus read of that finding: 152 studies (Contusion 143, Fracture 11 at the 10-21 02:3x count).
Only those finding(s) per line, highest target first, dealt round-robin to sessions 36-43 (chats 68-75). No gold, no anchors. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_close.py"""
import pandas as pd

from common.labels import gold_labels
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
V3E = REPO / "experiments/v3/oct16_experts/data/v3e"
FIRST, CHATS, TH = 36, 8, .3
key = pd.read_csv(D / "_key/p3.csv")
assert key.session.max() < FIRST, "sessions >= 36 already exist"
gi = set(gold_labels().index)
t = pd.read_csv(V3E / "tgt_bone_v3m.csv").set_index("StudyInstanceUID")
t = t[~t.index.isin(gi)]
op = pd.read_csv(V3E / "opus_raw.csv").set_index("StudyInstanceUID")
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
anchors = {2507, 3297, 3831}
rows = []
for s in t.index:
    fs = [f for f, c in (("contusion", "Contusion"), ("fracture", "Fracture")) if t.at[s, c] >= TH and pd.isna(op[c].get(s))]
    if not fs or s not in studies.index or studies.at[s, "index"] in anchors or not str(meta.Report.get(s, "") or "").strip():
        continue
    rows.append({"study_id": s, "findings": ",".join(fs), "kind": "new", "block": "close", "expected_pos": float(t.loc[s, ["Contusion", "Fracture"]].max()),
                 "index": int(studies.at[s, "index"])})
lines = pd.DataFrame(rows).sort_values(["expected_pos", "index"], ascending=[False, True]).reset_index(drop=True)
lines["session"] = FIRST + lines.index % CHATS
lines["pos"] = lines.groupby("session").cumcount()
new_key = pd.concat([key, lines[key.columns]]).sort_values(["session", "pos"])
cells = new_key.assign(f=new_key.findings.str.split(",")).explode("f")
assert not cells[cells.kind == "new"].duplicated(["study_id", "f"]).any(), "a (study, finding) listed twice"
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
print(f"{len(lines)} studies; contusion {lines.findings.str.contains('contusion').sum()}, fracture {lines.findings.str.contains('fracture').sum()}; "
      f"sessions {FIRST}..{FIRST + CHATS - 1}: " + " ".join(f"s{s}:{n}" for s, n in lines.groupby('session').size().items()))
