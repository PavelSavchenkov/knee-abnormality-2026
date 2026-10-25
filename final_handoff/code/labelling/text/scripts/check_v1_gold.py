"""v1 probability scored on the official gold-58 (macro AUC, paired bootstrap vs steven v4), next to v0 options.
not_mentioned has no probability in v1; it gets 0.15, a fixed value between general_normal and below (not fitted on gold).
Writes out/check_v1_gold.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_official_and_duplicates import L, auc                   # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
NOT_MENTIONED = 0.15
train = pd.read_csv(REPO / "data/original_data/train.csv").set_index("StudyInstanceUID")
gold = train[train.ACL.notna()]
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
v1 = json.load(open(REPO / "data/text_labels/v1.json"))
EXT = REPO / "work/eda/ext_labels/stevenleehans_rsna-knee-llm-report-labels"
v4 = pd.read_csv(EXT / "llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
v2 = pd.read_csv(EXT / "llm_labels_v2.csv").set_index("StudyInstanceUID")
assert all(s in v1 for s in gold.index), "every gold study must be in v1"
cells = pd.DataFrame([{"sid": s, "finding": L[k], "y": int(gold.at[s, L[k]]), "v0_status": v0[s]["findings"][k]["status"],
                       "v1_status": v1[s]["findings"][k]["status"], "v1_p": v1[s]["findings"][k]["probability"],
                       "steven_v4": v4.at[s, L[k]], "steven_v2": v2.at[s, L[k]]} for s in gold.index for k in L])
cells["v1_probability"] = cells.v1_p.fillna(NOT_MENTIONED)
cells["v0_rate_steven_v2"] = cells.v0_status.map(cells.groupby("v0_status").y.mean()) + 0.001 * cells.steven_v2.rank(pct=True)
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
cells["logit_mean_v1_steven_v4"] = (lg(cells.v1_probability) + lg(cells.steven_v4)) / 2
cells["v1_probability_steven_v2_inside"] = cells.v1_probability + 0.001 * cells.steven_v2.rank(pct=True)
methods = ["v1_probability", "v1_probability_steven_v2_inside", "logit_mean_v1_steven_v4", "v0_rate_steven_v2", "steven_v4"]
macro = lambda d, m: np.mean([auc(g.y, g[m]) for _, g in d.groupby("finding")])
rng = np.random.default_rng(0)
by = {s: d for s, d in cells.groupby("sid")}
draws = [pd.concat([by[s] for s in rng.choice(list(by), len(by))], ignore_index=True) for _ in range(1000)]
draws = [d for d in draws if d.groupby("finding").y.nunique().min() == 2]
base = np.array([macro(d, "steven_v4") for d in draws])
out = ["== official gold-58, macro AUC; difference vs steven v4 (paired bootstrap 95% CI)"]
for m in methods:
    diff = np.array([macro(d, m) for d in draws]) - base
    out.append(f"   {m:34s} {macro(cells, m):.4f}   {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
out.append("   (v0_rate_steven_v2 orders statuses by their gold-58 positive rate, so it has seen the gold; v1_probability has not)")
out.append("\n== per finding: v1_probability / steven_v4")
out.append("   " + ", ".join(f"{f} {auc(g.y, g.v1_probability):.3f}/{auc(g.y, g.steven_v4):.3f}" for f, g in cells.groupby("finding")))
out.append("\n== v1 probability vs official positive rate (calibration on gold-58)")
cal = cells.groupby(cells.v1_p.fillna(-1).round(2)).y.agg(["count", "mean"]).round(2)
cal.index = ["not_mentioned" if i == -1 else i for i in cal.index]
out.append(cal.T.to_string())
(HERE / "out/check_v1_gold.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
