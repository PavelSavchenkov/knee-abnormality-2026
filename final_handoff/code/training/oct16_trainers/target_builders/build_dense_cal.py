"""W5-3 dense NON-PL image target (Master 10-20 16:4x; USER: "try to achieve solid gains w/o pseudo labels as well -- e.g. by giving more
signal from new label studies"; OAI: no; CPU). data/tgt_p2_bone_dcal.csv (Fracture / Contusion, every non-gold study), no model prediction
anywhere: Opus where read (frozen 559) > the Opus-CALIBRATED text (data/tgt_p2cal.csv: per text status, the Opus mean shrunk to text; set on
2,107 studies) > the main recipe's text v1 target (mainp labels.make_targets, m_prob05 config) elsewhere. The fill differs from the bone main
target (tgt_v3) - tests whether a non-PL fill that is not the main head's own target keeps the image head distinct (as the PL fill did).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_dense_cal.py"""
from pathlib import Path

import pandas as pd

from build_dense_fill import recipe_targets
from common.folds import get_folds
from common.labels import gold_labels

EXP = Path(__file__).resolve().parents[1]


def main():
    cols = ["Fracture", "Contusion"]
    gi = set(gold_labels().index)
    studies = [s for s in get_folds().index if s not in gi]
    out = recipe_targets("mainp", "refs/m_prob05/config.json", studies)[cols].astype("float64").rename_axis("StudyInstanceUID")
    cal = pd.read_csv(EXP / "data/tgt_p2cal.csv").set_index("StudyInstanceUID")[cols].reindex(out.index)
    n_cal = {}
    for c in cols:
        m = cal[c].notna()
        out.loc[m, c], n_cal[c] = cal.loc[m, c], int(m.sum())
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")[cols]
    assert op.index.isin(out.index).all()
    out.loc[op.index] = op.to_numpy()
    assert out.notna().all().all()
    out.to_csv(EXP / "data/tgt_p2_bone_dcal.csv")
    t3 = pd.read_csv(EXP.parents[1] / "v1/oct11_fracture_contusion/data/tgt_v3.csv").set_index("StudyInstanceUID")[cols].reindex(out.index)
    off = ~out.index.isin(op.index)
    print(f"OAI: no. tgt_p2_bone_dcal.csv: {len(out)} studies (Opus {len(op)}, calibrated text {n_cal}, text v1 elsewhere); mean "
          f"{out.mean().round(3).to_dict()}; fill vs the bone main target tgt_v3 off the read studies: Spearman "
          + ", ".join(f"{c} {out.loc[off, c].corr(t3.loc[off, c], method='spearman'):.3f}" for c in cols))


if __name__ == "__main__":
    main()
