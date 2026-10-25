"""Host-alignment gold check g6 (fracture site checklist + PF depth / extent record) vs the strict pilot reads (pilot_c1 / c2) on gold-58:
AUC, log-loss, FP / FN at .5 by text report group; PF: the recorded 'd~N% e~M mm' against host rules (d > 50 and e >= 10 = the definition).
Output eval_g6.txt (work only: never memory, never labellers). OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/eval_g6.py"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table, status_table
from common.paths import REPO

D = Path(__file__).parent
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

G = gold_labels()
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(1 - np.clip(p, .01, .99))))  # noqa: E731


def load(dirs):
    r = {}
    for d in dirs:
        for p in (D / d).glob("*.json"):
            x = json.load(open(p))
            r[x["study_id"]] = x["findings"]
    return r


new, old = load([f"g6_s{k}" for k in range(1, 5)]), load(["pilot_c1", "pilot_c2"])
out = [f"g6 reads on gold: {len(new)} of 58"]
for c, k in (("Fracture", "fracture"), ("PF OA", "pf_oa")):
    ids = [s for s in G.index if s in new and s in old]
    if len(ids) < 10:
        continue
    y = G.loc[ids, c].astype(int)
    pn = pd.Series({s: probability(new[s][k]["status"], new[s][k]["certainty"]) for s in ids})
    po = pd.Series({s: probability(old[s][k]["status"], old[s][k]["certainty"]) for s in ids})
    pt = probability_table("v1")[k].reindex(ids).fillna(0.15)
    grp = status_table("v1")[k].reindex(ids).fillna("none")
    out.append(f"\n{c} (n {len(ids)}, host+ {int(y.sum())}): AUC g6 {roc_auc_score(y, pn) * 1e3:.0f} vs pilot {roc_auc_score(y, po) * 1e3:.0f} vs text "
               f"{roc_auc_score(y, pt) * 1e3:.0f} | log-loss g6 {ll(y, pn):.3f} vs pilot {ll(y, po):.3f}")
    if c == "PF OA":
        for w in (.25, .5):
            out.append(f"  blend w_opus {w}: g6 AUC {roc_auc_score(y, w * pn + (1 - w) * pt) * 1e3:.0f} vs pilot {roc_auc_score(y, w * po + (1 - w) * pt) * 1e3:.0f}")
    for g in sorted(grp.unique()):
        m = (grp == g).to_numpy()
        yy, a, b = y[m], pn[m] >= .5, po[m] >= .5
        out.append(f"  {g:16s} n {m.sum():2d} host+ {int(yy.sum()):2d} | g6 FP {int((a & (yy == 0)).sum())} FN {int((~a & (yy == 1)).sum())} | "
                   f"pilot FP {int((b & (yy == 0)).sum())} FN {int((~b & (yy == 1)).sum())}")
    if c == "PF OA":
        rec = {}
        for s in ids:
            mm = re.search(r"d~\s*(\d+)\s*%\s*e~\s*(\d+)", new[s][k].get("note", ""))
            if mm:
                rec[s] = (int(mm.group(1)), int(mm.group(2)))
        if rec:
            r = pd.DataFrame(rec, index=["d", "e"]).T
            r["y"] = y[r.index]
            for name, rule in (("definition d>50 & e>=10", (r.d > 50) & (r.e >= 10)), ("d>=50", r.d >= 50), ("d>=30", r.d >= 30), ("d>=30 & e>=10", (r.d >= 30) & (r.e >= 10))):
                out.append(f"  recorded depth rule {name:24s}: host+ among rule+ {r.y[rule].mean():.2f} (n {int(rule.sum())}) | host+ among rule- {r.y[~rule].mean():.2f} (n {int((~rule).sum())})")
            out.append(f"  recorded depth as a score: AUC {roc_auc_score(r.y, r.d + r.e / 100) * 1e3:.0f} (n {len(r)})")
    if c == "Fracture":
        chk = pd.Series({s: (re.search(r"chk ([^;]*);", new[s][k].get("note", "")) or [None, ""])[1] for s in ids})
        out.append(f"  checklist recorded on {int((chk != '').sum())} / {len(ids)} reads; host+ with a '+' site: {int(((chk.str.contains(r'\+')) & (y == 1)).sum())} / {int(y.sum())}")
(D / "eval_g6.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
