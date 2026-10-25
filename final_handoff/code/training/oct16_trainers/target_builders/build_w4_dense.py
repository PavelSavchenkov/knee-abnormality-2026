"""Wave 4 on DENSE image targets (USER 10-20: "either don't sample studies with zero gradient from them, or use old labels"; OAI: no; CPU).
 data/img_w_cert_dense.csv  StudyInstanceUID, Fracture, Contusion over every study of data/tgt_p2_bone_dv3.csv (Opus where read, tgt_v3
                            elsewhere): Opus-read cells = the frozen certainty weights (data/img_w_cert.csv: 1.0 / 0.6 / 0.3), old-label cells 1.0
                            -> vs the dense control the ONLY change is the down-weighting of uncertain Opus reads (W4-3 dense).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_w4_dense.py"""
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    dv3 = pd.read_csv(EXP / "data/tgt_p2_bone_dv3.csv").set_index("StudyInstanceUID")
    cw = pd.read_csv(EXP / "data/img_w_cert.csv").set_index("StudyInstanceUID")
    assert dv3.notna().all().all() and cw.index.isin(dv3.index).all()
    out = pd.DataFrame(1.0, index=dv3.index, columns=["Fracture", "Contusion"])
    out.loc[cw.index] = cw[["Fracture", "Contusion"]].to_numpy()
    assert out.notna().all().all()
    out.to_csv(EXP / "data/img_w_cert_dense.csv")
    print(f"OAI: no. img_w_cert_dense.csv: {len(out)} studies ({len(cw)} Opus-read with certainty weights, {len(out) - len(cw)} old-label cells at 1.0); "
          + "; ".join(f"{c} {out[c].value_counts().sort_index().to_dict()}" for c in out.columns))


if __name__ == "__main__":
    main()
