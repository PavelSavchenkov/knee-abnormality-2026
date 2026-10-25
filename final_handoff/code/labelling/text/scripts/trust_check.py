"""Which v1 text labels can be trusted as image-level labels? Error of status groups against official gold-58, per finding,
with optional filters (v0 == v1 status, steven v4 agrees), and how much of the non-gold data each group covers.
conda run -n rsna python work/text_labeling/trust_check.py   # saved copy: work/text_labeling/out/trust_check.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/pavel/Programming/knee-abnormality-2026")
KEYS = {"acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus", "medial_oa": "Medial OA",
        "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion", "synovitis": "Synovitis", "bakers_cyst": "Baker's",
        "contusion": "Contusion", "fracture": "Fracture"}
GROUP = {"denied": "trusted negative", "structure_normal": "trusted negative", "meets": "trusted positive"}
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 110)


def cells(version):
    rows = [(sid, KEYS[k], f["status"], f.get("probability"), f["quote"]) for sid, s in json.load(open(ROOT / f"data/text_labels/{version}.json")).items()
            for k, f in s["findings"].items()]
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "Finding", "status", "probability", "quote"]).set_index(["StudyInstanceUID", "Finding"])


def wilson(k, n):
    if n == 0:
        return "-"
    p, z = k / n, 1.96
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * p:.0f}% [{100 * (centre - half):.0f}-{100 * (centre + half):.0f}]"


c = cells("v1").join(cells("v0")[["status"]].rename(columns={"status": "status_v0"}))
gold = pd.read_csv(ROOT / "data/meta/studies.csv").set_index("StudyInstanceUID")[list(KEYS.values())]
c = c.join(gold.stack(future_stack=True).rename_axis(["StudyInstanceUID", "Finding"]).rename("gold"))
v4 = pd.read_csv(ROOT / "work/eda/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
c = c.join(v4[list(KEYS.values())].stack(future_stack=True).rename_axis(["StudyInstanceUID", "Finding"]).rename("steven_v4"))
c["group"] = c.status.map(GROUP).fillna("untrusted")
c["stable"] = c.status == c.status_v0
c["steven_agrees"] = np.where(c.group == "trusted positive", c.steven_v4 > 0.5, c.steven_v4 < 0.5)

assert c.index.is_unique and len(c) == 4407 * 12, len(c)
g, ng = c[c.gold.notna()], c[c.gold.isna()]
print(f"gold-58: {len(g)} cells, {int(g.gold.sum())} officially positive; non-gold: {len(ng)} cells\n")

print("== official positive rate by v1 status (gold-58) [Wilson 95% CI]")
t = g.groupby("status").gold.agg(["size", "sum"])
t["official positive"] = [wilson(k, n) for k, n in zip(t["sum"], t["size"])]
print(t.sort_values("sum").to_string(), "\n")

print("== trust groups with filters (gold-58): cells, wrong vs official, error [CI]")
for grp, wrong_if in [("trusted negative", 1), ("trusted positive", 0)]:
    for name, mask in [("v1 status only", True), ("+ v0 == v1", g.stable), ("+ v0 == v1 + steven v4 agrees", g.stable & g.steven_agrees)]:
        x = g[(g.group == grp) & mask]
        k = int((x.gold == wrong_if).sum())
        print(f"   {grp:16s} {name:32s} n={len(x):4d} wrong={k:3d} error {wilson(k, len(x))}")
print()

print("== per finding (gold-58): trusted negative wrong/total | trusted positive wrong/total | gold positives left in untrusted")
rows = {}
for f in KEYS.values():
    x = g.xs(f, level="Finding")
    neg, pos, un = x[x.group == "trusted negative"], x[x.group == "trusted positive"], x[x.group == "untrusted"]
    rows[f] = {"neg wrong/total": f"{int(neg.gold.sum())}/{len(neg)}", "pos wrong/total": f"{int((pos.gold == 0).sum())}/{len(pos)}",
               "untrusted cells": len(un), "untrusted official positive": f"{int(un.gold.sum())}/{len(un)}"}
print(pd.DataFrame(rows).T.to_string(), "\n")

print("== trusted cells that official gold contradicts (gold-58)")
bad = g[((g.group == "trusted negative") & (g.gold == 1)) | ((g.group == "trusted positive") & (g.gold == 0))]
print(bad.reset_index()[["Finding", "status", "probability", "status_v0", "steven_v4", "quote"]].assign(quote=lambda d: d.quote.str[:100]).to_string(index=False), "\n")

print("== non-gold coverage per finding: % of studies in each group (and % after v0 == v1 + steven v4 filter)")
cov = {}
for f in KEYS.values():
    x = ng.xs(f, level="Finding")
    strict = x.stable & x.steven_agrees
    cov[f] = {"trusted neg %": round(100 * (x.group == "trusted negative").mean(), 1), "trusted pos %": round(100 * (x.group == "trusted positive").mean(), 1),
              "untrusted %": round(100 * (x.group == "untrusted").mean(), 1),
              "strict neg %": round(100 * ((x.group == "trusted negative") & strict).mean(), 1), "strict pos %": round(100 * ((x.group == "trusted positive") & strict).mean(), 1),
              "trusted pos n": int((x.group == "trusted positive").sum())}
cov = pd.DataFrame(cov).T
cov.loc["ALL (mean)"] = cov.mean().round(1)
print(cov.to_string())
print("\n== untrusted non-gold cells by status (all findings)")
print(ng[ng.group == "untrusted"].status.value_counts().to_string())
