"""EF3 target (C221 / C217 extension, Master 10-19 23:3x 'EF3 = C217's T1 with Opus added on the 384 (a three-reader target)'; OAI: no; no gold):
data/ieff_codex_vlm_opus.csv = T1 (data/ieff_codex_vlm.csv: the Codex + VLM img_eff target) everywhere; on the Opus-read studies (384 non-gold
vague Effusion cells, data/image_labels/opus_effusion.json) sigmoid((2 logit(T1) + Opus logit z-aligned to T1's logit on those studies) / 3)
= Opus as one of three readers (Master's two-reader z-alignment, build.py, extended). Checks as the C217 queue waiter (unique ids, [0, 1],
>= 3000 targets, no gold study carries a target).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_ieff_ef3.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def main():
    t1 = pd.read_csv(EXP / "data/ieff_codex_vlm.csv", index_col="StudyInstanceUID")["Effusion"].dropna()
    op = pd.Series({s: r["findings"]["effusion"]["probability"] for s, r in json.load(open(REPO / "data/image_labels/opus_effusion.json")).items()},
                   dtype=float).dropna()
    gold = set(pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "ACL"]).dropna().StudyInstanceUID)
    both = op.index.intersection(t1.index)
    lt, lo = lg(t1[both]), lg(op[both])
    lo_al = (lo - lo.mean()) / lo.std() * lt.std() + lt.mean()
    out = t1.copy()
    out[both] = sg((2 * lt + lo_al) / 3)
    v = out.rename("Effusion")
    assert v.index.is_unique and v.between(0, 1).all() and len(v) >= 3000, (len(v), v.min(), v.max())
    assert not set(v.index) & gold, "gold studies carry targets"
    v.rename_axis("StudyInstanceUID").to_csv(EXP / "data/ieff_codex_vlm_opus.csv")
    d = (out[both] - t1[both]).abs()
    print(f"EF3: {len(v)} targets (T1 {len(t1)}), Opus on {len(both)} of {len(op)} reads; on them mean T1 {t1[both].mean():.3f} -> {out[both].mean():.3f}, "
          f"mean |change| {d.mean():.3f}, >= .5 flips {int(((t1[both] >= .5) != (out[both] >= .5)).sum())}; Spearman T1-Opus "
          f"{t1[both].corr(op[both], method='spearman'):.3f} -> data/ieff_codex_vlm_opus.csv")


if __name__ == "__main__":
    main()
