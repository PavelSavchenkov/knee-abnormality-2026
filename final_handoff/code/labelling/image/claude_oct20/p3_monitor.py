"""Monitor of the 8 Phase 3 Claude chats (p3_s1..s8: only the listed finding(s) per study). Per chat: saved / listed, validate.py problems
(incl. the listed findings), pace, notes compliance (native series + slice range, source tag). Pooled per finding: Opus positive rate vs
the yield model's expectation (positive_yield.py), and the blind RE-READS vs their phase-2 read (p3 protocol calibration). No gold.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/p3_monitor.py"""
import importlib.util
import json
import re
import sys
import time

import numpy as np
import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
spec = importlib.util.spec_from_file_location("validate", D / "_prompts/validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

SERIES = re.compile(r"\bs\d+\b[^.;]*?#\s*\d+", re.I)
TAG = re.compile(r"\b(images|both|report)\b(\s*:|\W*$)")
key = pd.read_csv(D / "_key/p3.csv").set_index("index")
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
p2 = json.load(open(REPO / "data/image_labels/opus_p2.json"))
now, alerts, rows = time.time(), [], []
print(f"p3 monitor {time.strftime('%H:%M')}")
for s in sorted(key.session.unique()):   # 10-20 18:3x: sessions 9+ of at most 20 lines
    lst = key[key.session == s]
    folder = D / f"p3_s{s}"
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) if folder.exists() else []
    if not files:
        print(f"  s{s}: 0 / {len(lst)}")
        continue
    t = np.array([p.stat().st_mtime for p in files])
    bad = {}
    for p in files:
        x = json.load(open(p))
        i = x["index"]
        if i not in lst.index:
            bad[p.name] = ["not in the list"]
            continue
        m = validate.check_file(p, i, set(lst.at[i, "findings"].split(",")))
        if m:
            bad[p.name] = m
        for f, v in x["findings"].items():
            rows.append({"s": s, "index": i, "study_id": x["study_id"], "f": f, "kind": lst.at[i, "kind"],
                         "p": None if v["status"] == "not_assessable" else probability(v["status"], v["certainty"]),
                         "ok": bool(SERIES.search(v.get("note", ""))) and bool(TAG.search(v.get("note", ""))), "t": p.stat().st_mtime})
    gaps = np.diff(t) / 60
    idle = (now - t.max()) / 60
    print(f"  s{s}: {len(files)} / {len(lst)} | last save {idle:.0f} min ago | {(t > now - 3600).sum()} in the last hour"
          + (f" | median {np.median(gaps):.1f} min / study" if len(gaps) else ""))
    if bad:
        alerts.append(f"s{s}: {len(bad)} file(s) fail validate.py, e.g. {next(iter(bad))}: {next(iter(bad.values()))[:2]}")
    if len(gaps) >= 3 and np.median(gaps[-5:]) < 0.3:
        alerts.append(f"s{s}: last saves < 0.3 min / study")
    if idle > 20 and len(files) < len(lst):
        alerts.append(f"s{s}: no save for {idle:.0f} min (stopped / stuck?)")
r = pd.DataFrame(rows)
if rows:
    n = r[r.kind == "new"]
    for f, g in n.groupby("f"):
        exp = np.mean([pool.at[sid, f"p_{f}"] if sid in pool.index else .82 for sid in g.study_id])   # .82 = the 8 report-meets fracture extras outside the pool
        print(f"  {f}: {len(g)} new reads | Opus positive {(g.p >= .5).mean():.2f} (yield model {exp:.2f}) | n/a {g.p.isna().mean():.2f} | "
              f"notes OK {g.ok.mean():.2f} (last 10 {g.ok.iloc[-10:].mean():.1f})")
        if len(g) >= 20 and (g.p >= .5).mean() < exp - .2:
            alerts.append(f"{f}: positive rate {(g.p >= .5).mean():.2f} far below the yield model ({exp:.2f})")
        if len(g) >= 10 and g.ok.iloc[-10:].mean() < .7:
            alerts.append(f"{f}: notes lack series + slices or the source tag in the last 10")
    rr = r[r.kind == "reread"]
    if len(rr):
        a = pd.DataFrame({"f": rr.f, "p3": rr.p.to_numpy(), "p2": [p2[s]["findings"][f]["probability"] for s, f in zip(rr.study_id, rr.f)]})
        print(f"  re-reads vs phase 2: {len(a)} | same side of .5 {((a.p3 >= .5) == (a.p2 >= .5)).mean():.2f} | mean p3 - p2 {np.nanmean(a.p3 - a.p2):+.2f}")
        for x in a.itertuples():
            print(f"    {x.f}: p2 {x.p2:.2f} -> p3 {x.p3 if x.p3 is not None else float('nan'):.2f}")
base = {f: sum((r["findings"][f]["probability"] or 0) >= .5 for r in p2.values()) for f in ("contusion", "fracture", "pf_oa")}
newpos = {f: int(((r.kind == "new") & (r.f == f) & (r.p >= .5)).sum()) if rows else 0 for f in base} if rows else {f: 0 for f in base}
tot = {f: base[f] + newpos[f] for f in base}
print("  positives with the 811 p2 reads: " + ", ".join(f"{f} {tot[f]} (+{newpos[f]})" for f in base))
for f in base:
    if tot[f] >= 200:
        print(f"MILESTONE {f} {tot[f]} positives (>= 200)")
cells = [(x["study_id"], f) for x in (json.load(open(p)) for p in D.glob("p3_s*/*.json")) for f in x["findings"]]   # closing batch 36-43 re-opens studies for OTHER findings
ids = [s for s, _ in cells]
if len(cells) != len(set(cells)):
    alerts.append(f"{len(cells) - len(set(cells))} (study, finding) read twice across p3_s*/ folders")
dup = set(ids) & set(json.load(open(p))["study_id"] for p in D.glob("p2_s*/*.json")) - set(key[key.kind == "reread"].study_id)
if dup:
    alerts.append(f"{len(dup)} p3 study also read in phase 2 outside the re-read set (an old chat resumed?)")
print("\n".join(f"ALERT {x}" for x in alerts) if alerts else "no alerts")
