"""Why steven v4 scores higher than v1 on the official gold-58. Compares steven's versions (v1 = before the gold-tested synovitis fill,
v2 = with it, v4 = v2 blended with lixin) with our v1 probability, per finding and without synovitis, and tests v1 with the same
a-priori synovitis rule (synovitis not mentioned -> use the effusion probability, the MOAKS effusion-synovitis idea).
Writes out/steven_gap.txt."""
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
v1 = json.load(open(REPO / "data/text_labels/v1.json"))
tabs = {"steven v1 (before synovitis fill)": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_full.csv",
        "steven v2 (synovitis fill, chosen on gold)": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v2.csv",
        "steven v4 (v2 + lixin blend)": "stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv",
        "lixin hard": "lixin73_rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"}
tabs = {k: pd.read_csv(EXT / v).set_index("StudyInstanceUID") for k, v in tabs.items()}
p = lambda s, k: v1[s]["findings"][k]["probability"]
ours = pd.DataFrame({L[k]: [p(s, k) if p(s, k) is not None else 0.15 for s in gold.index] for k in L}, index=gold.index)
syn_fill = ours.copy()
for s in gold.index:
    if v1[s]["findings"]["synovitis"]["status"] == "not_mentioned":
        syn_fill.at[s, "Synovitis"] = ours.at[s, "Effusion"]
scores = {"our v1 probability": ours, "our v1 + synovitis from effusion (a priori)": syn_fill, **{k: t.loc[gold.index, list(L.values())] for k, t in tabs.items()}}
fs = list(L.values())
macro = lambda sc, idx, use: np.mean([auc(gold.loc[idx, f].astype(int), sc.loc[idx, f]) for f in use])
rng = np.random.default_rng(0)
draws = [rng.choice(gold.index, len(gold)) for _ in range(1000)]
draws = [d for d in draws if all(gold.loc[d, f].nunique() == 2 for f in fs)]
ref = "steven v4 (v2 + lixin blend)"
base = np.array([macro(scores[ref], d, fs) for d in draws])
out = ["== official gold-58 macro AUC: all 12 findings / without synovitis; all-12 difference vs steven v4 (paired bootstrap 95% CI)"]
for k, sc in scores.items():
    diff = np.array([macro(sc, d, fs) for d in draws]) - base
    out.append(f"   {k:44s} {macro(sc, gold.index, fs):.4f} / {macro(sc, gold.index, [f for f in fs if f != 'Synovitis']):.4f}   "
               f"{diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
out.append("\n== per finding AUC")
out.append(pd.DataFrame({k: {f: round(auc(gold[f].astype(int), sc.loc[gold.index, f]), 3) for f in fs} for k, sc in scores.items()}).to_string())
(HERE / "out/steven_gap.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
