"""Phase 3 PF batch (USER 10-21 11:3x: "lets try more labelling. i can copy paste 16 sessions with up to 20 studies now"). PF OA is the one
image-label class blocked by coverage: 914 of the 1,297 non-gold old-label PF positives (text v1 p >= .5) have no Opus PF read. Where Opus
moves the label (read studies): text p in [.6, .8) -> Opus+ ~16 %, [.8, .9) -> ~54 %, [.9, 1] -> ~95 % (little change). So this batch
draws 160 + 160 unread studies at random (seed 0) from [.6, .8) and [.8, .9): a representative sample of the two strata where the label
changes, so a later depth calibration is not fitted to a hand-picked subset. Only pf_oa per line, read with the g6 depth record
("d~N% e~M mm;"). Interleaved strata, dealt round-robin to sessions 44-59 (chats 80-95), 20 lines each. No gold, no anchors. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_pf.py"""
import numpy as np
import pandas as pd

from common.labels import gold_labels, probability_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
FIRST, CHATS, PER, STRATA = 44, 16, 20, ((.6, .8), (.8, .9))
key = pd.read_csv(D / "_key/p3.csv")
assert key.session.max() < FIRST, "sessions >= 44 already exist"
gi = set(gold_labels().index)
tp = probability_table("v1")["pf_oa"]
reads = pd.read_csv(REPO / "data/image_labels/reads.csv")
read_pf = set(reads[reads.run.str.startswith("opus_") & (reads.finding == "pf_oa")].study_id)
listed_pf = set(key[key.findings.str.contains("pf_oa")].study_id)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
anchors = {2507, 3297, 3831}
ok = [s for s in tp.index if s not in gi and s not in read_pf and s not in listed_pf and s in studies.index
      and studies.at[s, "index"] not in anchors and str(meta.Report.get(s, "") or "").strip()]
cand = tp[ok]
rng = np.random.default_rng(0)
n = CHATS * PER // len(STRATA)
draws = []
for lo, hi in STRATA:
    pool = cand[(cand >= lo) & (cand < hi)].sort_index()
    assert len(pool) >= n, (lo, hi, len(pool))
    draws.append(list(rng.choice(pool.index, n, replace=False)))
order = [s for pair in zip(*draws) for s in pair]          # alternate the strata
lines = pd.DataFrame({"study_id": order, "findings": "pf_oa", "kind": "new", "block": "pf_depth",
                      "expected_pos": cand[order].to_numpy(), "index": [int(studies.at[s, "index"]) for s in order]})
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
print(f"candidates (unread, unlisted, non-gold old-label PF): " + ", ".join(f"[{lo}, {hi}) {int(((cand >= lo) & (cand < hi)).sum())}" for lo, hi in STRATA)
      + f"; total p >= .5: {int((cand >= .5).sum())}")
print(f"{len(lines)} studies, sessions {FIRST}..{FIRST + CHATS - 1}: " + " ".join(f"s{s}:{k}" for s, k in lines.groupby('session').size().items()))
