"""Quality of v1 labels (returned_v1/): agreement with v0 on the same studies, probability use per status, whether the
wording rules for the probability were followed (checked from attributes and English quote words), identical-report
consistency, and per-batch drift. Writes out/check_v1.txt."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from collect_v0 import load                                        # noqa: E402
from score_returned import GROUP                                    # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
v1 = load("returned_v1")
L = {"acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus", "medial_oa": "Medial OA",
     "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion", "synovitis": "Synovitis", "bakers_cyst": "Baker's", "contusion": "Contusion", "fracture": "Fracture"}
v2 = pd.read_csv(REPO / "work/eda/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv").set_index("StudyInstanceUID")
rep = pd.read_csv(REPO / "text_labeling_bundle/reports.csv").set_index("study_id")
rows = []
for s, (b, x) in v1.items():
    for k, f in x["findings"].items():
        rows.append({"sid": s, "batch": b, "finding": k, "status": f["status"], "p": f["probability"], "quote": f["quote"], "v0": v0[s]["findings"][k]["status"],
                     "size_word": f.get("size"), "longest_mm": f.get("longest_mm"), "steven_v2": v2.at[s, L[k]], "lang": rep.at[s, "language"]})
df = pd.DataFrame(rows)
out = [f"== {df.sid.nunique()} studies in {df.batch.nunique()} archives"]

out.append("\n== v1 vs v0 on the same studies (two independent runs)")
out.append(f"   exact status {(df.status == df.v0).mean():.3f}; group (P/N/U/S) {(df.status.map(GROUP) == df.v0.map(GROUP)).mean():.3f}; "
           f"meets vs not: {((df.status == 'meets') == (df.v0 == 'meets')).mean():.3f}")
out.append("   per finding exact: " + ", ".join(f"{k} {(d.status == d.v0).mean():.3f}" for k, d in df.groupby("finding")))
out.append("   status counts v1 / v0: " + ", ".join(f"{s} {int((df.status == s).sum())}/{int((df.v0 == s).sum())}" for s in df.v0.value_counts().index))

out.append("\n== probability values used, per status (counts)")
out.append(pd.crosstab(df.status, df.p.fillna(-1).round(2)).replace(0, "").to_string())

out.append("\n== rule checks")
e = df[df.finding == "effusion"]
out.append("   effusion meets, size word x probability:\n" + pd.crosstab(e[e.status == "meets"].size_word.fillna("null"), e[e.status == "meets"].p).to_string())
out.append("   effusion below, size word x probability:\n" + pd.crosstab(e[e.status == "below"].size_word.fillna("null"), e[e.status == "below"].p).to_string())
bk = df[(df.finding == "bakers_cyst") & (df.status == "unsettled")].copy()
bk["mm"] = pd.cut(bk.longest_mm.fillna(-1), [-2, 0, 19, 34, 49, 1000], labels=["no size", "<20", "20-34", "35-49", ">=50"])
out.append("   Baker's unsettled, measured size x probability (rule: 20-34 -> 0.45, 35-49 -> 0.75, no size -> 0.60):\n" + pd.crosstab(bk.mm, bk.p).to_string())
h = df[(df.status == "hedged") & (df.lang == "English")].copy()
word = lambda q: ("probable/likely" if re.search(r"\bprobabl|\blikely", q, re.I) else "possible/suspected" if re.search(r"possibl|may represent|might|could represent|suspect|suspicious", q, re.I)
                  else "R/O, ?, questionable" if re.search(r"r/o|rule out|\?|questionab|cannot be (completely )?excluded|differential", q, re.I) else "other")
h["word"] = h.quote.map(word)
out.append("   hedged (English quotes), firmest hedge word x probability (rule 0.50 / 0.40 / 0.30):\n" + pd.crosstab(h.word, h.p).to_string())
dn = df[(df.status == "denied") & (df.lang == "English")].copy()
dn["qualified"] = dn.quote.str.contains(r"significant|definite|obvious|evidence|signs of|appreciable|discrete", case=False, regex=True)
out.append("   denied (English quotes), qualified wording x probability (rule: qualified 0.03, categorical 0.01):\n" + pd.crosstab(dn.qualified, dn.p).to_string())

out.append("\n== within-status order: Spearman of v1 probability with steven v2, cells pooled over findings")
for st in ["meets", "below", "unsettled", "hedged"]:
    d = df[df.status == st]
    out.append(f"   {st:10s} n={len(d):5d}  rho={d.p.rank().corr(d.steven_v2.rank()):.3f}")

r = rep.loc[list(v1)]
r = r.assign(key=r.report.str.split().str.join(" ") + "|" + r.knee_side)
groups = [g.index.tolist() for _, g in r.groupby("key") if len(g) > 1]
pairs = [(v1[a][1]["findings"][k], v1[b][1]["findings"][k]) for g in groups for a, b in zip(g, g[1:]) for k in L]
out.append(f"\n== identical report and knee side: {len(groups)} groups; same status {np.mean([x['status'] == y['status'] for x, y in pairs]):.3f}, "
           f"same status and probability {np.mean([x['status'] == y['status'] and x['probability'] == y['probability'] for x, y in pairs]):.3f} ({len(pairs)} finding pairs)")

bt = df.groupby("batch").apply(lambda d: pd.Series({"mean_p_meets": d.p[d.status == "meets"].mean(), "mean_p_below": d.p[d.status == "below"].mean(),
                                                    "mean_p_unsettled": d.p[d.status == "unsettled"].mean(), "exact_vs_v0": (d.status == d.v0).mean()}), include_groups=False)
out.append("\n== per batch: mean probability within a status and agreement with v0 (mean, sd, min, max)")
out.append(bt.agg(["mean", "std", "min", "max"]).round(3).to_string())
(HERE / "out/check_v1.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
