"""Claude (Opus 5.5) IMAGE + REPORT pilot (Master 10-19; USER: "i could label some with claude (about 1k-2k) ... research which ones are
the most promising"; "the best one was to give text as reference"). Plan: work/claude_label_plan_2026-10-19/PLAN.md. OAI: no.
Builds work/claude_labels_oct20/:
- _prompts/ = a copy of work/claude_labels_oct16/_prompts (all12 prompt.txt, schema.json, validate.py, studies.csv, tiles.csv,
  definitions.md, host_figures/), read-only for the labeller;
- _batches/pilot_c1.txt, pilot_c2.txt: 50 indexes each = the 58 gold studies + 42 random non-gold decoys, shuffled (seed 20), so the
  labeller cannot tell gold apart;
- _reports/<index>.txt: the report text of every pilot / anchor index (no labels, no study ids);
- _prompts_pilot/anchors.md: 3 anchor studies (non-gold, not in the pilot) with report-derived reference statuses per finding (text v1
  'meets' -> present per report, 'denied' / 'structure_normal' -> absent per report; others not given): the text reference;
- _key/pilot.csv (index, study_id, stratum): for evaluation only, forbidden to the labeller.
conda run -n rsna python work/claude_labels_oct20/build_pilot.py"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import status_table
from common.paths import FINDINGS, KEY_OF, REPO

D = REPO / "work/claude_labels_oct20"
SRC = REPO / "work/claude_labels_oct16/_prompts"
for sub in ("_batches", "_reports", "_prompts_pilot", "_key"):
    (D / sub).mkdir(parents=True, exist_ok=True)
if not (D / "_prompts").exists():
    shutil.copytree(SRC, D / "_prompts")
st_csv = pd.read_csv(D / "_prompts/studies.csv")
full = st_csv[st_csv.sheets.str.count(";") >= 7]                       # >= 8 sheets
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
first = st_csv.drop_duplicates("study_id").set_index("study_id")["index"]
g_idx = sorted(first[first.index.isin(gold)].tolist())
assert len(g_idx) == 58, len(g_idx)
rng = np.random.default_rng(20)
has_rep = set(meta.index[meta.Report.fillna("").str.strip() != ""])
pool = full[~full.study_id.isin(gold) & full.study_id.isin(has_rep)].drop_duplicates("study_id")
st = status_table("v1")
used = set(g_idx)


def explicit(sid):
    s = st.loc[sid] if sid in st.index else None
    return {f: ("present per report" if s[KEY_OF[f]] == "meets" else "absent per report" if s[KEY_OF[f]] in ("denied", "structure_normal")
                else None) for f in FINDINGS} if s is not None else {}


cand = pool[pool.study_id.isin(st.index)].copy()
cand["ex"] = cand.study_id.map(explicit)
prof = {"trauma knee": ["ACL", "Contusion", "MCL", "Lateral Meniscus"], "degenerative knee": ["Medial OA", "PF OA", "Medial Meniscus", "Effusion"],
        "normal knee": []}
anchors = []
for name, pos in prof.items():
    def score(ex):
        if pos:
            return sum(ex.get(f) == "present per report" for f in pos) * 10 + sum(v is not None for v in ex.values())
        return sum(v == "absent per report" for v in ex.values())
    c = cand[~cand["index"].isin([a[1] for a in anchors])]
    best = c.loc[c.ex.map(score).sort_values(ascending=False).index[0]]
    anchors.append((name, int(best["index"]), best.study_id, best.ex))
used |= {a[1] for a in anchors}
dec = []   # USER 10-20: gold only, no decoys (the access rules keep it blind)
key = pd.DataFrame({"index": g_idx + dec, "stratum": ["gold"] * 58 + ["decoy"] * len(dec)})
key["study_id"] = key["index"].map(st_csv.set_index("index").study_id)
key = key.sample(frac=1, random_state=20).reset_index(drop=True)
gk = key[key.stratum == "gold"].index.tolist()
c1 = key.iloc[::2]; c2 = key.iloc[1::2]
for n, c in (("c1", c1), ("c2", c2)):
    (D / f"_batches/pilot_{n}.txt").write_text("\n".join(str(i) for i in c["index"]) + "\n")
key.to_csv(D / "_key/pilot.csv", index=False)
for i, sid in list(zip(key["index"], key.study_id)) + [(a[1], a[2]) for a in anchors]:
    rep = meta.Report.get(sid)
    rep = rep.strip() if isinstance(rep, str) and rep.strip() else "(no report available for this study: label from the images only)"
    (D / f"_reports/{i:04d}.txt").write_text(rep + "\n")
lines = ["# Anchor studies: the text reference (read before the first study)", "",
         "Three studies from the same dataset, not in your list. For each, the finding statuses that its radiology report states clearly",
         "('present per report' / 'absent per report'). Findings the report leaves vague or silent are not listed. Look at every sheet of each",
         "anchor (and its report, _reports/<index>.txt) once before your first study, to see how clearly reported findings look on these",
         "sheets. Do not save labels for anchors.", ""]
for name, i, _, ex in anchors:
    lines += [f"## Anchor {i:04d} ({name})", ""] + [f"- {f}: {v}" for f, v in ex.items() if v] + [""]
(D / "_prompts_pilot/anchors.md").write_text("\n".join(lines))
print(f"pilot: {len(key)} indexes (gold only) -> c1 {len(c1)} (gold {int((c1.stratum == 'gold').sum())}) / c2 {len(c2)} "
      f"(gold {int((c2.stratum == 'gold').sum())}); anchors {[a[1] for a in anchors]}; reports written {len(list((D / '_reports').glob('*.txt')))}")
