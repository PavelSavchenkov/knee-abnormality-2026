"""Turn v0 statuses into soft targets and score them on the official gold-58 (macro AUC, paired bootstrap vs steven v4).
The gold-58 is used only to order the 10 statuses (pooled over findings), not per finding. Leave-one-out rates are avoided on purpose:
removing a study lowers its own status rate whenever it is positive, which ranks positives below negatives inside a status.
Writes out/calibrate_status.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_official_and_duplicates import L, ORDER, auc            # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
train = pd.read_csv(REPO / "data/original_data/train.csv").set_index("StudyInstanceUID")
gold = train[train.ACL.notna()]
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
v4 = pd.read_csv(REPO / "work/eda/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
cells = pd.DataFrame([{"sid": s, "finding": L[k], "status": v0[s]["findings"][k]["status"], "y": int(gold.at[s, L[k]]), "v4": v4.at[s, L[k]]}
                      for s in gold.index for k in L])
rate = cells.groupby("status").y.mean()
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
cells["fixed_order"] = cells.status.map(ORDER)
cells["rate_order"] = cells.status.map(rate)
cells["rate_order_v4_inside"] = cells.rate_order + 0.001 * cells.v4            # status gaps are >= 0.02, so v4 only breaks ties
cells["logit_mean_rate_v4"] = (lg(cells.rate_order) + lg(cells.v4)) / 2
methods = ["fixed_order", "rate_order", "rate_order_v4_inside", "logit_mean_rate_v4", "v4"]
macro = lambda d, m: np.mean([auc(g.y, g[m]) for _, g in d.groupby("finding")])
rng = np.random.default_rng(0)
by_sid = {s: d for s, d in cells.groupby("sid")}
draws = []
for _ in range(1000):
    d = pd.concat([by_sid[s] for s in rng.choice(list(by_sid), len(by_sid))], ignore_index=True)
    if d.groupby("finding").y.nunique().min() == 2:
        draws.append(d)
base = np.array([macro(d, "v4") for d in draws])
out = ["== official gold-58, macro AUC; paired bootstrap difference vs steven v4"]
for m in methods:
    diff = np.array([macro(d, m) for d in draws]) - base
    out.append(f"   {m:22s} {macro(cells, m):.4f}   vs v4 {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
out.append("\n== official positive rate by v0 status (pooled over findings)")
out.append(cells.groupby("status").y.agg(["count", "sum", "mean"]).round(2).sort_values("mean").to_string())
out.append("\n== per finding: rate_order / rate_order_v4_inside / v4")
out.append("   " + ", ".join(f"{f} {auc(g.y, g.rate_order):.3f}/{auc(g.y, g.rate_order_v4_inside):.3f}/{auc(g.y, g.v4):.3f}" for f, g in cells.groupby("finding")))
(HERE / "out/calibrate_status.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
