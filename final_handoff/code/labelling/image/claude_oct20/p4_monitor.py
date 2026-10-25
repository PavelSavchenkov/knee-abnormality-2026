"""Monitor of the 8 Phase 4 chats (p4_s1..s8: Opus second reads of the Codex Baker's pool, bakers_cyst only). Per chat: saved / listed,
validate.py problems, pace, notes compliance (native series + slice range, a diameter, the source tag). Pooled: Opus positive rate and
agreement with the Codex first read (same side of .5; the 2-reader target's consistency). No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/p4_monitor.py"""
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
MM = re.compile(r"\d+\s*(x\s*\d+\s*)*mm", re.I)
key = pd.read_csv(D / "_key/p4.csv").set_index("index")
now, alerts, rows = time.time(), [], []
print(f"p4 monitor {time.strftime('%H:%M')}")
for s in sorted(key.session.unique()):
    lst = key[key.session == s]
    files = sorted((D / f"p4_s{s}").glob("*.json"), key=lambda p: p.stat().st_mtime) if (D / f"p4_s{s}").exists() else []
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
        m = validate.check_file(p, x["index"], {"bakers_cyst"})
        if m:
            bad[p.name] = m
        v = x["findings"]["bakers_cyst"]
        note = v.get("note", "")
        rows.append({"p": None if v["status"] == "not_assessable" else probability(v["status"], v["certainty"]), "codex": lst.at[x["index"], "codex_p"],
                     "ok": bool(SERIES.search(note)) and bool(TAG.search(note)) and bool(MM.search(note) or v.get("diameter_mm") is not None)})
    gaps = np.diff(t) / 60
    idle = (now - t.max()) / 60
    print(f"  s{s}: {len(files)} / {len(lst)} | last save {idle:.0f} min ago" + (f" | median {np.median(gaps):.1f} min / study" if len(gaps) else ""))
    if bad:
        alerts.append(f"s{s}: {len(bad)} file(s) fail validate.py, e.g. {next(iter(bad))}: {next(iter(bad.values()))[:2]}")
    if len(gaps) >= 3 and np.median(gaps[-5:]) < 0.3:
        alerts.append(f"s{s}: last saves < 0.3 min / study")
    if idle > 20 and len(files) < len(lst):
        alerts.append(f"s{s}: no save for {idle:.0f} min")
if rows:
    r = pd.DataFrame(rows).dropna(subset=["p"])
    agree = ((r.p >= .5) == (r.codex >= .5)).mean()
    print(f"  bakers: {len(r)} reads | Opus positive {(r.p >= .5).mean():.2f} (Codex on the same {(r.codex >= .5).mean():.2f}) | same side as Codex {agree:.2f} "
          f"| notes OK {pd.DataFrame(rows).ok.mean():.2f}")
    if len(rows) >= 10 and pd.DataFrame(rows).ok.iloc[-10:].mean() < .7:
        alerts.append("notes lack series + slices, a diameter or the source tag in the last 10")
print("\n".join(f"ALERT {x}" for x in alerts) if alerts else "no alerts")
