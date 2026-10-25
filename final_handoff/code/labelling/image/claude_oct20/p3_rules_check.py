"""Consistency rules of prompt.txt that validate.py runs only for the full 12-finding panel (consistency -> full_rules needs 'acl'), applied
to the p2 / p3 panel reads. Per session: PF OA meets needs cartilage_loss partial_ge_50 / full_thickness (and extent_ge_10mm true); PF
cartilage_loss none -> normal, partial_lt_50 -> below / borderline / possible; contusion meets needs an edema region > 0, all-zero edema
-> normal, excluded needs a fracture / degenerative / SONK pattern; fracture meets needs line_seen true and a type other than
old_or_healed, old_or_healed -> excluded. Prints the violation count per session and the first examples. No gold. OAI: no.
  python work/claude_labels_oct20/p3_rules_check.py [--sessions 14-23]"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--sessions", default="1-40")
a = ap.parse_args()
lo, hi = map(int, a.sessions.split("-"))
HIGH = {"partial_ge_50", "full_thickness"}


def rules(f):
    out = []
    if "pf_oa" in f:
        o = f["pf_oa"]
        if o["status"] == "meets" and (o.get("cartilage_loss") not in HIGH or o.get("extent_ge_10mm") is not True):
            out.append(f"pf_oa meets with cartilage_loss {o.get('cartilage_loss')} / extent {o.get('extent_ge_10mm')}")
        if o.get("cartilage_loss") == "none" and o["status"] != "normal":
            out.append(f"pf_oa cartilage_loss none but {o['status']}")
        if o.get("cartilage_loss") == "partial_lt_50" and o["status"] not in {"below", "borderline", "possible"}:
            out.append(f"pf_oa partial_lt_50 but {o['status']}")
    if "contusion" in f:
        c = f["contusion"]
        ed = c.get("edema") or {}
        if c["status"] == "meets" and not any((v or 0) > 0 for v in ed.values()):
            out.append("contusion meets with no edema region > 0")
        if ed and all((v or 0) == 0 for v in ed.values()) and c["status"] not in {"normal", "not_assessable"}:
            out.append(f"contusion all-zero edema but {c['status']}")
        if c["status"] == "excluded" and not set(c.get("pattern") or []) & {"around_fracture", "degenerative_subchondral", "insufficiency_or_sonk"}:
            out.append("contusion excluded without a fracture / degenerative / SONK pattern")
    if "fracture" in f:
        r = f["fracture"]
        if r["status"] == "meets" and not (r.get("line_seen") is True and r.get("type") not in (None, "old_or_healed")):
            out.append(f"fracture meets with line_seen {r.get('line_seen')} / type {r.get('type')}")
        if r.get("type") == "old_or_healed" and r["status"] != "excluded":
            out.append(f"fracture old_or_healed but {r['status']}")
    return out


bad, n = defaultdict(list), defaultdict(int)
for s in range(lo, hi + 1):
    for p in sorted((D / f"p3_s{s}").glob("*.json")):
        x = json.load(open(p))
        n[s] += 1
        for m in rules(x["findings"]):
            bad[s].append(f"{x['index']}: {m}")
for s in sorted(n):
    print(f"s{s}: {n[s]} reads, {len(bad[s])} rule violations" + (f" e.g. {bad[s][:3]}" if bad[s] else ""))
print(f"total {sum(n.values())} reads, {sum(len(v) for v in bad.values())} violations")
