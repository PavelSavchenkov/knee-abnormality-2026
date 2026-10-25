"""Consistency of attributes with status in returned label files, and side handling for two-knee reports."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
reports = pd.read_csv("/home/pavel/Programming/knee-abnormality-2026/text_labeling_bundle/reports.csv").set_index("index")
docs = [json.loads(f.read_text(encoding="utf-8")) for d in (HERE / "returned_v0").iterdir() if d.is_dir() for f in d.glob("*.json")]
docs = sorted(docs, key=lambda x: x["index"])
bad = []


def flag(x, k, msg):
    bad.append(f"{x['index']:3d} {k}: {msg} | status={x['findings'][k]['status']} | {x['findings'][k]['quote'][:120]}")


for x in docs:
    F = x["findings"]
    e = F["effusion"]
    ok = {"meets": {"moderate", "large"}, "below": {"trace", "small", "mild"}, "denied": {"none"}}
    if e["status"] in ok and e["size"] is not None and e["size"] not in ok[e["status"]]:
        flag(x, "effusion", f"size {e['size']}")
    b = F["bakers_cyst"]
    if b["status"] == "meets" and not (b["size"] in ("moderate", "large") or (b["longest_mm"] or 0) >= 50):
        flag(x, "bakers_cyst", f"meets with size {b['size']} {b['longest_mm']}")
    if b["status"] == "below" and not (b["size"] == "small" or (b["longest_mm"] is not None and b["longest_mm"] < 20)):
        flag(x, "bakers_cyst", f"below with size {b['size']} {b['longest_mm']}")
    if b["status"] == "unsettled" and (b["size"] is not None or (b["longest_mm"] is not None and not 20 <= b["longest_mm"] < 50)):
        flag(x, "bakers_cyst", f"unsettled with size {b['size']} {b['longest_mm']}")
    for k in ("acl", "mcl"):
        g, s = F[k]["grade"], F[k]["status"]
        if (s == "meets" and g in ("degeneration", "low_grade", "low_to_partial", "partial")) or (s == "below" and g in ("partial", "high_grade", "complete")):
            flag(x, k, f"grade {g}")
    for k in ("medial_oa", "lateral_oa", "pf_oa"):
        o = F[k]
        if o["status"] == "meets" and o["depth"] in ("superficial", "partial") and o["severity_word"] != "severe":
            flag(x, k, f"meets with depth {o['depth']}")
        if o["status"] == "below" and o["depth"] in ("deep", "full") and not (o["size_mm"] and o["size_mm"] < 10):
            flag(x, k, f"below with depth {o['depth']} size {o['size_mm']}")
    c = F["contusion"]
    if c["status"] == "meets" and c["cause"] not in ("trauma", None):
        flag(x, "contusion", f"meets with cause {c['cause']}")
    if c["status"] == "excluded" and c["cause"] in ("trauma", None):
        flag(x, "contusion", f"excluded with cause {c['cause']}")
    if c["status"] in ("meets", "excluded", "unsettled", "hedged") and not c["bones"]:
        flag(x, "contusion", "no bones listed")
    for k in ("medial_meniscus", "lateral_meniscus"):
        if F[k]["displaced_fragment"] is True and F[k]["status"] != "meets":
            flag(x, k, "displaced fragment but not meets")

print("attribute / status inconsistencies:", len(bad))
print("\n".join(bad))
print("\nreport_side counts:", pd.Series([x["study"]["report_side"] for x in docs]).value_counts().to_dict())
for x in docs:
    rep = reports.at[x["index"], "report"]
    if "BILATERAL NOTE" in rep or x["study"]["report_side"] in ("both", "other"):
        print(f"  index {x['index']}: knee_side {reports.at[x['index'], 'knee_side']}, report_side {x['study']['report_side']}, two reports: {'BILATERAL NOTE' in rep}")
print("report_type:", pd.Series([x["study"]["report_type"] for x in docs]).value_counts().to_dict())
print("refers_to_prior true:", sum(x["study"]["refers_to_prior"] for x in docs), "| hoffa_edema:", pd.Series([str(x["study"]["hoffa_edema"]) for x in docs]).value_counts().to_dict())
dup = reports.loc[[x["index"] for x in docs], "report"]
print("duplicate report texts inside 0-84:", int(dup.duplicated(keep=False).sum()))
