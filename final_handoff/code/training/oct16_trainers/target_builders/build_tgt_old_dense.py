"""Fine-tune CONTROL target (Master 10-20 15:0x, the C223 A 'ftc' design: same init / schedule as S1, image heads on the OLD targets only;
OAI: no; CPU). data/tgt_p2_bone_old.csv = tgt_v3 Fracture / Contusion on every non-gold study = tgt_p2_bone_dv3.csv with NO Opus cells
(asserted: identical index, identical values off the 559 Opus-read studies). ARM (dv3) - CONTROL (old) = the pure label effect.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_tgt_old_dense.py"""
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]


def main():
    cols = ["Fracture", "Contusion"]
    dv3 = pd.read_csv(EXP / "data/tgt_p2_bone_dv3.csv").set_index("StudyInstanceUID")[cols]
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    t3 = pd.read_csv(REPO / "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv").set_index("StudyInstanceUID")[cols].reindex(dv3.index)
    assert t3.notna().all().all()
    off = ~dv3.index.isin(op.index)
    assert (t3[off] == dv3[off]).all().all(), "dv3's fill != tgt_v3"
    t3.to_csv(EXP / "data/tgt_p2_bone_old.csv")
    d = (t3 - dv3).abs()
    print(f"OAI: no. tgt_p2_bone_old.csv: {len(t3)} studies = dv3 without Opus cells; cells that differ from dv3: "
          + ", ".join(f"{c} {int((d[c] > 1e-9).sum())}" for c in cols) + " (all on the 559 Opus-read studies)")


if __name__ == "__main__":
    main()
