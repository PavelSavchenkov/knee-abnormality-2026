"""Monitor of the 4 gold feasibility chats (g5_s1..s4, panel p5: acl / mcl / menisci / medial + lateral OA). Per chat: saved / listed,
validate.py problems, pace; pooled notes compliance (native series + slice range and the source tag in every finding's note) and the
Opus positive rate per finding. No gold numbers (labeller blindness). OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/g5_monitor.py"""
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

FS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa"]
SERIES = re.compile(r"\bs\d+\b[^.;]*?#\s*\d+", re.I)
TAG = re.compile(r"\b(images|both|report)\b(\s*:|\W*$)")
key = pd.read_csv(D / "_key/g5.csv").set_index("index")
now, alerts, rows = time.time(), [], []
print(f"g5 monitor {time.strftime('%H:%M')}")
for s in sorted(key.session.unique()):
    lst = key[key.session == s]
    files = sorted((D / f"g5_s{s}").glob("*.json"), key=lambda p: p.stat().st_mtime) if (D / f"g5_s{s}").exists() else []
    if not files:
        print(f"  s{s}: 0 / {len(lst)}")
        continue
    t = np.array([p.stat().st_mtime for p in files])
    bad = {}
    for p in files:
        x = json.load(open(p))
        if x["index"] not in lst.index:
            bad[p.name] = ["not in the list"]
            continue
        if m := validate.check_file(p, x["index"]):
            bad[p.name] = m
        for f in FS:
            v = x["findings"].get(f) or {}
            note = v.get("note", "")
            rows.append({"f": f, "pos": v.get("status") != "not_assessable" and probability(v["status"], v["certainty"]) >= .5,
                         "ok": bool(SERIES.search(note)) and bool(TAG.search(note))})
    gaps = np.diff(t) / 60
    idle = (now - t.max()) / 60
    print(f"  s{s}: {len(files)} / {len(lst)} | last save {idle:.0f} min ago" + (f" | median {np.median(gaps):.1f} min / study" if len(gaps) else ""))
    if bad:
        alerts.append(f"s{s}: {len(bad)} file(s) fail validate.py, e.g. {next(iter(bad))}: {next(iter(bad.values()))[:2]}")
    if len(gaps) >= 3 and np.median(gaps[-5:]) < 1:
        alerts.append(f"s{s}: last saves < 1 min / study (6 findings with zoom crops)")
    if idle > 30 and len(files) < len(lst):
        alerts.append(f"s{s}: no save for {idle:.0f} min")
if rows:
    r = pd.DataFrame(rows)
    print("  Opus positive rate: " + " ".join(f"{f} {r[r.f == f].pos.mean():.2f}" for f in FS) + f" | notes OK {r.ok.mean():.2f}")
    if len(r) >= 30 and r.ok.iloc[-30:].mean() < .7:
        alerts.append("notes lack series + slices or the source tag in the last 5 studies")
print("\n".join(f"ALERT {x}" for x in alerts) if alerts else "no alerts")
