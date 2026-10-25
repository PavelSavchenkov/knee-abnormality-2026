"""All label sources scored against my hand labels (text-based) on the 150 random non-gold studies, which no public table could tune on,
and on the gold-58 for reference. Target: my status, positive = meets; negative = below / excluded / denied / structure_normal / hedged;
cells I left unsettled, conflict or not_mentioned are skipped. Macro AUC over findings, paired bootstrap over studies vs steven v4.
Writes out/compare_on_hand_labels.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_official_and_duplicates import L, ORDER, auc            # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
EXT = REPO / "work/eda/ext_labels"
hand = json.load(open(REPO / "data/text_labels/gold_text_labels.json"))
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
v1 = json.load(open(REPO / "data/text_labels/v1.json"))
tabs = {"steven v1": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv", "steven v4": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "lixin hard": "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv", "pilkwang": "pilkwang_rsna-knee-llm-labels/report_labels_v2.csv"}
tabs = {k: pd.read_csv(EXT / v).set_index("StudyInstanceUID") for k, v in tabs.items()}
POS, NEG = {"meets"}, {"below", "excluded", "denied", "structure_normal", "hedged"}
rows = []
for s, h in hand.items():
    for k, f in h["findings"].items():
        if f["status"] in POS | NEG:
            p1 = v1[s]["findings"][k]["probability"]
            rows.append({"sid": s, "set": h["set"], "finding": L[k], "y": int(f["status"] in POS),
                         "our v1 probability": 0.15 if p1 is None else p1, "our v0 status order": ORDER[v0[s]["findings"][k]["status"]],
                         **{t: (tab.at[s, L[k]] if s in tab.index else np.nan) for t, tab in tabs.items()}})
df = pd.DataFrame(rows)
df["pilkwang"] = df.pilkwang.fillna(0.28)
methods = ["our v1 probability", "our v0 status order", "steven v1", "steven v4", "lixin hard", "pilkwang"]
macro = lambda d, m: np.nanmean([auc(g.y, g[m]) for _, g in d.groupby("finding") if g.y.nunique() == 2])
out = []
for name in ["random150", "gold58"]:
    d = df[df.set == name]
    by = {s: g for s, g in d.groupby("sid")}
    rng = np.random.default_rng(0)
    draws = [pd.concat([by[s] for s in rng.choice(list(by), len(by))], ignore_index=True) for _ in range(500)]
    base = np.array([macro(x, "steven v4") for x in draws])
    out.append(f"== {name}: {d.sid.nunique()} studies, {len(d)} cells my labels settle ({int(d.y.sum())} positive); macro AUC, difference vs steven v4 [95% CI]")
    for m in methods:
        diff = np.array([macro(x, m) for x in draws]) - base
        out.append(f"   {m:22s} {macro(d, m):.4f}   {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    out.append("")
(HERE / "out/compare_on_hand_labels.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
