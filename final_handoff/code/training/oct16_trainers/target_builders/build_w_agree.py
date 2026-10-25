"""C223 way C target-free weights (USER 10-20 'at least 3 diverse ways to use image-based labels'; Master's design; OAI: no; no gold):
data/w_agree.csv = per-cell LOSS MULTIPLIERS for main/train.py w_csv: 1.0 where the image read and the text v1 probability agree on the side of
0.5, 0.3 where they disagree; NaN (= 1) where no image read exists. Image reads: Effusion = the three-reader EF3 target (data/ieff_codex_vlm_opus
.csv: Codex + VLM, + Opus on the 384), Baker's = Codex (data/image_labels/codex_bakers.json, 516 vague-report studies), Synovitis = Codex
(data/derived/synovitis_labels_codex.csv; INFO column). Gold-58 studies carry no weight.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_w_agree.py"""
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path.insert(0, str(REPO))
from common.labels import probability_table  # noqa: E402

DOWN = 0.3


def main():
    pr = probability_table("v1")
    gold = set(pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "ACL"]).dropna().StudyInstanceUID)
    img = {"Effusion": pd.read_csv(EXP / "data/ieff_codex_vlm_opus.csv", index_col="StudyInstanceUID")["Effusion"],
           "Baker's": pd.Series({s: r["findings"]["bakers_cyst"]["probability"]
                                 for s, r in json.load(open(REPO / "data/image_labels/codex_bakers.json")).items()}, dtype=float),
           "Synovitis": pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")["Synovitis"]}
    key = {"Effusion": "effusion", "Baker's": "bakers_cyst", "Synovitis": "synovitis"}
    out = {}
    for c, s in img.items():
        s = s.dropna()
        s = s[~s.index.isin(gold)]
        t = pr[key[c]].reindex(s.index)
        s, t = s[t.notna()], t.dropna()
        out[c] = ((s >= 0.5) == (t >= 0.5)).map({True: 1.0, False: DOWN})
        print(f"{c}: {len(s)} cells with an image read; disagree (x{DOWN}) {int((out[c] < 1).sum())} ({(out[c] < 1).mean():.1%}); "
              f"image >= .5 {(s >= .5).mean():.2f}, text >= .5 {(t >= .5).mean():.2f}")
    w = pd.DataFrame(out)
    assert not set(w.index) & gold and w.index.is_unique
    w.rename_axis("StudyInstanceUID").to_csv(EXP / "data/w_agree.csv")
    print(f"-> data/w_agree.csv: {len(w)} studies, columns {list(w.columns)}")


if __name__ == "__main__":
    main()
