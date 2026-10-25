"""PF protocol shift check (10-21 11:3x): Opus+ rate of the new p3pf reads (PF-only, native zoom, depth record) vs the older Opus PF reads
(p2 / p3 / p3c) inside the same text v1 groups and p bins; and the new reads status vs the recorded depth rule. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/pf_protocol_shift.py"""
import glob, json, re, sys
import pandas as pd
from common.labels import status_table, probability_table
from common.paths import REPO
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability
D = REPO / "work/claude_labels_oct20"
ts, tp = status_table("v1")["pf_oa"], probability_table("v1")["pf_oa"]
new = {}
for f in glob.glob(str(D / "p3_s4[4-9]/*.json")) + glob.glob(str(D / "p3_s5[0-9]/*.json")):
    x = json.load(open(f)); v = x["findings"]["pf_oa"]
    m = re.match(r"^\s*d~(\d+)%\s*e~(\d+)\s*mm;", v["note"])
    new[x["study_id"]] = (probability(v["status"], v["certainty"]), v["status"], int(m[1]), int(m[2]))
n = pd.DataFrame(new, index=["p", "status", "d", "e"]).T
r = pd.read_csv(REPO / "data/image_labels/reads.csv")
old = r[r.run.isin(["opus_p2", "opus_p3", "opus_p3c"]) & (r.finding == "pf_oa") & r.repeat.isna()].drop_duplicates("study_id").set_index("study_id").rename(columns={"probability": "p"})
for name, d in (("NEW", n), ("OLD", old)):
    d = d.assign(g=ts.reindex(d.index).fillna("none").values, t=tp.reindex(d.index).values)
    d = d[(d.t >= .6) & (d.t < .9)]
    print(name, "t in [.6,.9):", ", ".join(f"{g} n {len(x)} Opus+ {(x.p.astype(float) >= .5).mean():.2f}" for g, x in d.groupby("g")),
          "| by t:", ", ".join(f"{b} n {len(x)} {(x.p.astype(float) >= .5).mean():.2f}" for b, x in d.groupby(pd.cut(d.t, [.6, .7, .8, .9], right=False), observed=True)))
print("NEW status vs depth rule d>50 & e>=10:", pd.crosstab(n.status, (n.d.astype(int) > 50) & (n.e.astype(int) >= 10)).to_dict())
print("old runs per t-bin [.6,.8):", old.assign(t=tp.reindex(old.index).values).query("t >= .6 and t < .8").run.value_counts().to_dict())
