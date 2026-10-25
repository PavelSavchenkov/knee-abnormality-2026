"""Hand-look aid for the Fracture host-like drop of an Opus-led main arm (mixed-evidence protocol): on one fold's held-out studies whose
text v1 Fracture status is "meets" (the report-positive side a host-like reader rewards), rank change arm vs control, with the Opus read
(status / type / note, if read) and the report's fracture sentences. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/frac_movers.py ARM CTRL FOLD [--n 15]"""
import argparse
import json
import re

import pandas as pd

from common.labels import status_table
from common.paths import REPO

E = REPO / "experiments/v3/oct16_experts"
ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("ctrl")
ap.add_argument("fold", type=int)
ap.add_argument("--n", type=int, default=15)
a = ap.parse_args()


def last(name):
    d = next(b / name for b in (E / "runs", E / "refs/main", E / "refs/bone") if (b / name).exists()) / f"fold{a.fold}"
    o = pd.read_parquet(d / "oof.parquet")
    o = o.set_index("StudyInstanceUID") if "StudyInstanceUID" in o.columns else o
    return (o[o.epoch == o.epoch.max()] if "epoch" in o.columns else o)["Fracture"]


p, q = last(a.arm), last(a.ctrl)
ids = p.index.intersection(q.index)
r = pd.DataFrame({"arm": p[ids].rank(pct=True), "ctrl": q[ids].rank(pct=True)})
r["d"] = r.arm - r.ctrl
st = status_table("v1").reindex(ids)["fracture"]
reads = json.load(open(REPO / "data/image_labels/opus_p2.json"))
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
m = r[st == "meets"].sort_values("d")
print(f"{a.arm} vs {a.ctrl} fold {a.fold}: {len(ids)} held-out, text Fracture meets {len(m)}; mean rank change on meets {m.d.mean():+.3f}")
for sid, x in m.head(a.n).iterrows():
    o = reads.get(sid, {}).get("findings", {}).get("fracture")
    ot = f"Opus {o['status']} / {o.get('type')} / line {o.get('line_seen')}" if o else "Opus: not read"
    rep = str(meta.Report.get(sid, "") or "")
    sents = [s.strip() for s in re.split(r"(?<=[.;])\s+|\n", rep) if re.search(r"fractur|insufficien|stress|avuls|osteochond|subchondral", s, re.I)]
    print(f"\n{sid[-12:]}  rank {x.ctrl:.2f} -> {x.arm:.2f} ({x.d:+.2f}) | {ot}")
    print("   report: " + " | ".join(sents)[:400])
    if o:
        print("   Opus note: " + o.get("note", "")[:250])
