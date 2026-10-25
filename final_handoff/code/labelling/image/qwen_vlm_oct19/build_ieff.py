"""Two-reader Effusion image target for Experts' C217 (img_eff2r): where a study has a Codex Effusion read AND a VLM read (pool_e_Ag_tshot:
Qwen3.5-9B, A + guide + 2 + 2 text-explicit examples = the pre-registered pick of the 10-19 Effusion set-up search), target = sigmoid(mean of the Codex logit and the VLM logit z-aligned to the Codex logit scale); Codex-
only studies keep Codex; no Codex read -> NaN (no image target, as the shipped head). Alignment = mean / sd of logit(Codex) and
logit(VLM) over the studies with both (label-free).
Checks (printed): AUC of Codex alone vs the two-reader target against the independent Opus Effusion reads (data/image_labels/
opus_effusion.json, 384 studies, >= 0.5), plus the Codex-vs-VLM rank correlation.
Writes to experiments/v3/oct16_experts/data/ (StudyInstanceUID, Effusion), USER 10-19 ~20:3x "try different targets -- 2 heads for codex
and for VLM, their blending, etc":
- ieff_codex_vlm.csv: T1, the two-reader average (Codex-only studies keep Codex; no Codex -> NaN);
- ieff_vlm.csv: the aligned VLM read alone (T3 target, and the 2nd head's target in T2);
- ieff_union.csv: T4, Codex where read, else the aligned VLM (coverage expansion)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
REPO = HERE.parents[1]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
codex = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")["Effusion"]
vlm = pd.read_csv(HERE / "out/pool_e_Ag_tshot.csv").query("finding == 'Effusion'").set_index("StudyInstanceUID")["p_yes"]   # the pre-registered pick
both = codex.dropna().index.intersection(vlm.dropna().index)
lc, lv = lg(codex[both]), lg(vlm[both])
lv_al = (lv - lv.mean()) / lv.std() * lc.std() + lc.mean()
two = 1 / (1 + np.exp(-(lc + lv_al) / 2))
tgt = codex.copy()
tgt[both] = two
OUT = REPO / "experiments/v3/oct16_experts/data"
out = OUT / "ieff_codex_vlm.csv"                                      # T1: two-reader average where both, else Codex
tgt.rename("Effusion").rename_axis("StudyInstanceUID").to_csv(out)
lv_all = lg(vlm.dropna())                                             # the VLM on the Codex logit scale (same alignment constants)
v_al = 1 / (1 + np.exp(-((lv_all - lv.mean()) / lv.std() * lc.std() + lc.mean())))
v_al.rename("Effusion").rename_axis("StudyInstanceUID").to_csv(OUT / "ieff_vlm.csv")       # T2 2nd head / T3 VLM-only target
union = codex.dropna().combine_first(v_al)                            # T4: Codex where read, else the aligned VLM (coverage)
union.rename("Effusion").rename_axis("StudyInstanceUID").to_csv(OUT / "ieff_union.csv")
print(f"T3 vlm-only {v_al.notna().sum()}; T4 union {union.notna().sum()} (VLM-only fill {union.notna().sum() - codex.notna().sum()})")
print(f"codex {codex.notna().sum()}, vlm {vlm.notna().sum()}, both {len(both)}; Spearman codex-vlm {spearmanr(lc, lv)[0]:.3f}; -> {out}")
op = json.load(open(REPO / "data/image_labels/opus_effusion.json"))
opus = pd.Series({k: v["findings"]["effusion"]["probability"] for k, v in op.items()}).dropna()
m = opus.index.intersection(both)
y = (opus[m] >= 0.5).astype(int)
print(f"vs Opus (n {len(m)}, n+ {y.sum()}): Codex {roc_auc_score(y, codex[m]):.3f}  VLM {roc_auc_score(y, vlm[m]):.3f}  two-reader {roc_auc_score(y, two[m]):.3f}")
