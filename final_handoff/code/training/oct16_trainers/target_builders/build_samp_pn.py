"""U100 (USER 10-20: "separate positive and negative pools and coeffs, so claude-labelled positives are sample meaningfully, negatives are less
critical"; OAI: no; CPU): per-study sampling multipliers for samp_ids ('mult' column; studies not listed = 1):
 data/samp_pn_bone.csv  Opus-read studies: x8 if Opus Fracture or Contusion p >= .5, else x2   (bone recipes)
 data/samp_pn_main.csv  the same with Fracture, Contusion or PF OA (ext_p2_main_r: PF = 0.5 Opus + 0.5 text)   (main recipe m_p2part)
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_samp_pn.py"""
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
POS, NEG = 8.0, 2.0


def main():
    for name, src, cols in (("bone", "tgt_p2_bone.csv", ["Fracture", "Contusion"]), ("main", "ext_p2_main_r.csv", ["Fracture", "Contusion", "PF OA"])):
        t = pd.read_csv(EXP / "data" / src).set_index("StudyInstanceUID")[cols]
        pos = (t >= 0.5).any(axis=1)
        pd.DataFrame({"mult": np.where(pos, POS, NEG)}, index=t.index).to_csv(EXP / f"data/samp_pn_{name}.csv")
        print(f"OAI: no. samp_pn_{name}.csv: {len(t)} read studies, positive ({' or '.join(cols)} >= .5) {int(pos.sum())} -> x{POS:g}, negative "
              f"{int((~pos).sum())} -> x{NEG:g}")


if __name__ == "__main__":
    main()
