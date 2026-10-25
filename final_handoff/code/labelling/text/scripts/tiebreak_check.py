"""Does anything order cells *within* a v0 status consistently with the official gold-58? Compares tie-breakers:
steven v4, its two parts (steven v2 verbal probability, lixin hard 0/1), and v0's own attributes ordered by fixed rules.
Scores: within-status AUC pooled over findings, and macro AUC of 'status rate + tie-breaker' minus 'status rate' (paired bootstrap).
Writes out/tiebreak_check.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_official_and_duplicates import L, auc                   # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
EXT = REPO / "work/eda/ext_labels"
train = pd.read_csv(REPO / "data/original_data/train.csv").set_index("StudyInstanceUID")
gold = train[train.ACL.notna()]
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
tab = {"steven_v4": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv", "steven_v2": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv",
       "lixin_hard": "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"}
tab = {k: pd.read_csv(EXT / v).set_index("StudyInstanceUID") for k, v in tab.items()}

# v0 attributes -> one ordinal per cell, by fixed rules (higher = more of the finding); 0 when nothing is stated
EFF = {"none": 0, "trace": 1, "small": 2, "mild": 3, "mild_to_moderate": 4, "moderate": 5, "large": 6}
LIG = {"degeneration": 1, "low_grade": 2, "low_to_partial": 3, "partial": 4, "high_grade": 5, "complete": 6}
DEPTH = {"superficial": 1, "partial": 2, "deep": 3, "full": 4}
SEV = {"mild": 1, "mild_to_moderate": 2, "moderate": 3, "moderate_to_severe": 4, "severe": 5}
BAK = {"small": 1, "moderate": 2, "large": 3}


def attr(x, k):
    f, st = x["findings"][k], x["study"]
    if k in ("acl", "mcl"):
        return LIG.get(f["grade"], 0) - (1 if f["acuity"] == "chronic" else 0)
    if k in ("medial_meniscus", "lateral_meniscus"):
        return 2 * bool(f["displaced_fragment"]) + len(f["location"]) / 5
    if k in ("medial_oa", "lateral_oa", "pf_oa"):
        return DEPTH.get(f["depth"], 0) + SEV.get(f["severity_word"], 0) / 10 + {"diffuse": 0.3, "multifocal": 0.2, "focal": 0.1}.get(f["extent"], 0)
    if k == "effusion":
        return EFF.get(f["size"], 0)
    if k == "synovitis":                                       # MOAKS-style proxies: effusion amount and Hoffa edema
        return EFF.get(x["findings"]["effusion"]["size"], 0) + 3 * bool(st["hoffa_edema"])
    if k == "bakers_cyst":
        return BAK.get(f["size"], 0) + (f["longest_mm"] or 0) / 100
    if k == "contusion":
        return {"trauma": 2, None: 1}.get(f["cause"], 0) + len(f["bones"]) / 10
    if k == "fracture":
        return {"acute": 2, None: 1, "subacute": 1}.get(f["acuity"], 0)


cells = pd.DataFrame([{"sid": s, "finding": L[k], "status": v0[s]["findings"][k]["status"], "y": int(gold.at[s, L[k]]),
                       "v0_attributes": attr(v0[s], k), **{t: tab[t].at[s, L[k]] for t in tab}} for s in gold.index for k in L])
cells["rate"] = cells.status.map(cells.groupby("status").y.mean())
breakers = ["steven_v4", "steven_v2", "lixin_hard", "v0_attributes"]
macro = lambda d, col: np.mean([auc(g.y, g[col]) for _, g in d.groupby("finding")])
for b in breakers:
    cells[f"rate+{b}"] = cells.rate + 0.001 * cells[b].rank(pct=True)          # rank inside 0..1, so it never crosses statuses
def main():
    out = ["== within-status AUC (cells pooled over findings; only statuses with both outcomes and >= 5 positives)"]
    rows = {}
    for st, d in cells.groupby("status"):
        if d.y.sum() >= 5 and (1 - d.y).sum() >= 5:
            rows[st] = {"cells": len(d), "positives": int(d.y.sum()), **{b: round(auc(d.y, d[b]), 3) for b in breakers}}
    out.append(pd.DataFrame(rows).T.to_string())

    rng = np.random.default_rng(0)
    by_sid = {s: d for s, d in cells.groupby("sid")}
    draws = [pd.concat([by_sid[s] for s in rng.choice(list(by_sid), len(by_sid))], ignore_index=True) for _ in range(1000)]
    draws = [d for d in draws if d.groupby("finding").y.nunique().min() == 2]
    base = np.array([macro(d, "rate") for d in draws])
    out.append(f"\n== macro AUC on gold-58: status rate alone {macro(cells, 'rate'):.4f}; adding a tie-breaker inside each status (difference, paired bootstrap 95% CI)")
    for b in breakers:
        diff = np.array([macro(d, f"rate+{b}") for d in draws]) - base
        out.append(f"   + {b:14s} {macro(cells, f'rate+{b}'):.4f}   {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    out.append("\n== distinct values steven v2 uses (all 4,407 studies, all findings): " + str(tab["steven_v2"].drop(columns=[c for c in tab["steven_v2"].columns if c not in L.values()]).stack().round(3).value_counts().head(12).to_dict()))
    (HERE / "out/tiebreak_check.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
