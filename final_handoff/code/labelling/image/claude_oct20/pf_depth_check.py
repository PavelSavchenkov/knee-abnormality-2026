"""Phase 3 PF batch check (sessions 44-59, build_phase3_pf.py): saved / listed per session, the depth-record prefix "d~N% e~M mm;" on every
pf_oa note, Opus+ rate (p >= .5) per text stratum ([.6, .8) / [.8, .9)) vs the read-pool rates the batch was drawn on (~0.16 / ~0.54), and
the recorded depth per status. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/pf_depth_check.py"""
import json
import re

import sys

import pandas as pd

from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402   the saved JSONs hold status + certainty; p comes from the shared mapping (as p3_monitor)
DEPTH = re.compile(r"^\s*d~(\d+)%\s*e~(\d+)\s*mm;")
LO, HI = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "44-59").split("-"))   # session range; 60-72 = the PF re-read batch
key = pd.read_csv(D / "_key/p3.csv")
key = key[key.session.between(LO, HI)]
rows, per = [], []
for s, g in key.groupby("session"):
    got = {p.stem: json.load(open(p)) for p in (D / f"p3_s{s}").glob("*.json")} if (D / f"p3_s{s}").exists() else {}
    per.append(f"s{s}:{len(got)}/{len(g)}")
    for sid, t in zip(g.study_id, g.expected_pos):
        if sid in got:
            f = got[sid]["findings"]["pf_oa"]
            m = DEPTH.match(f.get("note") or "")
            rows.append({"t": t, "p": None if f["status"] == "not_assessable" else probability(f["status"], f["certainty"]), "status": f.get("status"), "depth": int(m[1]) if m else None, "ext": int(m[2]) if m else None})
print("saved: " + " ".join(per))
if rows:
    r = pd.DataFrame(rows)
    r["stratum"] = pd.cut(r.t, [.6, .8, .9], right=False)
    print(f"reads {len(r)} | depth prefix {r.depth.notna().sum()} / {len(r)}" + (" ALERT missing depth prefix" if r.depth.isna().any() else ""))
    print("Opus+ by stratum (n / rate; pool ~0.16 / ~0.54): " + ", ".join(f"{k}: {len(d)} / {(d.p >= .5).mean():.2f}" for k, d in r.groupby("stratum", observed=True)))
    print("depth by status (n / median d / median e): " + ", ".join(f"{k}: {len(d)} / {d.depth.median():.0f} / {d.ext.median():.0f}" for k, d in r.groupby("status")))
