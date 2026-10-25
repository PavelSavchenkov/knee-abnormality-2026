"""W2-3 m_sif target (Experts V3 10-20 11:4x, Master's wave 2: 17 of 41 Opus fracture positives are subchondral insufficiency fractures that text
calls 'unsettled'; the host definition counts them as fracture; OAI: no; CPU). data/tgt_p2cal_sif.csv = data/tgt_p2cal.csv (arm E: Opus-calibrated
text targets) + a report rule on ALL non-gold studies: Report matching SIF_RE (case-insensitive) -> Fracture target 0.85 (overrides the calibrated
value; NaN elsewhere = the trainer's own text target). Prints the matched count and their current Fracture text label distribution.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_tgt_sif.py"""
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from lib import GOLD  # noqa: E402

SIF_RE = r"(subchondral|insufficiency)[^.]{0,40}fracture|fracture[^.]{0,40}(subchondral|insufficiency)|spontaneous osteonecrosis|sonk"


def main():
    s = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "Fracture"]).set_index("StudyInstanceUID")
    s = s[~s.index.isin(GOLD.index)]
    hit = s.Report.fillna("").str.contains(SIF_RE, case=False, regex=True)
    t = pd.read_csv(EXP / "data/tgt_p2cal.csv").set_index("StudyInstanceUID")
    t = t.reindex(t.index.union(s.index[hit]))
    t.loc[s.index[hit], "Fracture"] = 0.85
    assert not t.index.isin(GOLD.index).any()
    t.to_csv(EXP / "data/tgt_p2cal_sif.csv")
    print(f"SIF rule: {int(hit.sum())} non-gold studies matched; their host-CSV Fracture field: {s.loc[hit, 'Fracture'].value_counts(dropna=False).to_dict()}; "
          f"tgt_p2cal_sif {len(t)} studies, set {t.notna().sum().to_dict()}")


if __name__ == "__main__":
    main()
