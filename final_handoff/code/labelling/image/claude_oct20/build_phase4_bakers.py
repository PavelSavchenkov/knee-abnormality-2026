"""Phase 4: Opus second reads of the Codex Baker's pool (USER 10-21 01:2x picked "Contusion + Baker's"; the Effusion EF3 multi-reader
target had the strongest stats on 10-20, so Baker's gets a 2-reader target). Pool = data/image_labels/codex_bakers.json (516 non-gold
studies whose text v1 Baker's status is below / unsettled). Most uncertain Codex read first (|Codex p - 0.5| ascending), 8 sessions x 20,
dealt round-robin. Panel p4 = bakers_cyst only (schema_p4.json; validate.py p4 sheets 1_sag_fs / 3_ax_fs / 6_medial). Images + report,
like Phase 3. No gold, no Codex values shown to the labeller. OAI: no.
Writes _batches/p4_s1..s8.txt, _key/p4.csv, _reports/<index>.txt.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase4_bakers.py"""
import json

import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
CHATS, MAX = 8, 20
cod = json.load(open(REPO / "data/image_labels/codex_bakers.json"))
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
anchors = {2507, 3297, 3831}
rows = []
for s, r in cod.items():
    p = r["findings"]["bakers_cyst"]["probability"]
    if p is None or s in gold or s not in studies.index or studies.at[s, "index"] in anchors or not str(meta.Report.get(s, "") or "").strip():
        continue
    rows.append({"study_id": s, "index": int(studies.at[s, "index"]), "codex_p": p, "unc": abs(p - .5)})
pool = pd.DataFrame(rows).sort_values(["unc", "index"]).reset_index(drop=True)
sel = pool.head(CHATS * MAX).copy()
sel["findings"], sel["kind"], sel["block"] = "bakers_cyst", "new", "bakers_cyst"
sel["session"] = 1 + sel.index % CHATS
sel["pos"] = sel.groupby("session").cumcount()
assert sel.study_id.is_unique and not sel.study_id.isin(gold).any()
kf = D / "_key/p4.csv"
assert not kf.exists()
sel[["index", "study_id", "findings", "kind", "block", "session", "pos", "codex_p"]].to_csv(kf, index=False)
kf.chmod(0o444)
for s, g in sel.groupby("session"):
    f = D / f"_batches/p4_s{s}.txt"
    f.write_text("".join(f"{i} bakers_cyst\n" for i in g["index"]))
    f.chmod(0o444)
for i, sid in zip(sel["index"], sel.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[sid].strip() + "\n")
print(f"pool {len(pool)} -> {len(sel)} studies, Codex p range {sel.codex_p.min():.2f}..{sel.codex_p.max():.2f}; "
      f"Codex >= .5 {int((sel.codex_p >= .5).sum())}; sessions 1..{CHATS}")
