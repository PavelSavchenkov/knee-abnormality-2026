"""Dense image target with UP-WEIGHTED Opus cells (Experts V3 10-20 15:1x; S1 read: on the dense target the image head converges to a copy of
the main head - tgt_v3 fills 87 % of its cells; OAI: no; CPU). data/img_w_opus4.csv: StudyInstanceUID, Fracture, Contusion over every study of
tgt_p2_bone_dv3.csv: Opus-read cells (the frozen 559) weight K, old-label cells 1.0 (bone img_w_csv); K 4 = the in-loss twin of w4_samp4 (x4
draws), K 8 = the dose's second point (Master 15:2x: ctl 1x / 4x / 8x). Writes data/img_w_opus{K}.csv.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_opus_w.py [K]"""
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    dv3 = pd.read_csv(EXP / "data/tgt_p2_bone_dv3.csv").set_index("StudyInstanceUID")
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    w = pd.DataFrame(1.0, index=dv3.index, columns=["Fracture", "Contusion"])
    for c in w.columns:
        w.loc[op.index[op[c].notna()], c] = k
    w.to_csv(EXP / f"data/img_w_opus{k:g}.csv")
    print(f"OAI: no. img_w_opus{k:g}.csv: {len(w)} studies; " + "; ".join(f"{c} {w[c].value_counts().sort_index().to_dict()}" for c in w.columns)
          + f"; Opus share of the image-loss weight: " + ", ".join(f"{c} {w[c][w[c] > 1].sum() / w[c].sum():.2f}" for c in w.columns))


if __name__ == "__main__":
    main()
