"""P2 wave training inputs (Experts V3 10-20 11:3x; USER go via Master: 'at least 5 diverse high ROI ways' to use the Opus p2 image labels;
OAI: no; CPU). Reads data/tgt_p2_bone.csv + data/tgt_p2_pf.csv (scripts/build_tgt_p2.py, Opus p2 reads -> mapping.probability; NaN =
unlabelled) and writes:
 (C) data/ext_p2_main.csv      StudyInstanceUID, Contusion, Fracture, PF OA = the p2 probabilities (mainp ext_labels: one masked image head per
                               column, img_Contusion / img_Fracture / img_PF OA; the Synovitis m_prob05 mechanism);
 (D) data/tgt_v3_p2blend.csv   StudyInstanceUID, Fracture, Contusion = tgt_v3 (the b_v3 bone recipe's target, all 4,407 studies) with
                               0.5 tgt_v3 + 0.5 Opus p on the p2-labelled studies (bone_train tgt_csv);
 (E) data/tgt_p2cal.csv        copy of Master's Opus-calibrated text targets (work/claude_labels_oct20/tgt_p2cal.csv; mainp tgt_csv: NaN =
                               the trainer's own text target), md5 printed for provenance.
Asserts: no gold study carries a p2 value; (D) differs from tgt_v3 only on labelled studies.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_wave.py"""
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from lib import GOLD  # noqa: E402

TGT_V3 = REPO / "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv"
P2CAL = REPO / "work/claude_labels_oct20/tgt_p2cal.csv"


def main():
    b = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    pf = pd.read_csv(EXP / "data/tgt_p2_pf.csv").set_index("StudyInstanceUID")
    ext = b[["Contusion", "Fracture"]].join(pf.rename(columns={"img_pf_patella": "PF OA"}), how="outer")
    assert not ext.index.isin(GOLD.index).any(), "gold study in p2"
    ext.to_csv(EXP / "data/ext_p2_main.csv")
    t = pd.read_csv(TGT_V3).set_index("StudyInstanceUID")
    out = t.copy()
    for c in ("Fracture", "Contusion"):
        v = b[c].dropna()
        assert v.index.isin(t.index).all(), c
        out.loc[v.index, c] = 0.5 * t.loc[v.index, c] + 0.5 * v
    changed = (out != t).any(axis=1)
    assert changed[changed].index.isin(b.index).all()
    out.to_csv(EXP / "data/tgt_v3_p2blend.csv")
    shutil.copy(P2CAL, EXP / "data/tgt_p2cal.csv")
    cal = pd.read_csv(EXP / "data/tgt_p2cal.csv").set_index("StudyInstanceUID")
    assert not cal.index.isin(GOLD.index).any(), "gold study in tgt_p2cal"
    md5 = hashlib.md5(open(EXP / "data/tgt_p2cal.csv", "rb").read()).hexdigest()
    print(f"(C) data/ext_p2_main.csv: {len(ext)} studies, labelled {ext.notna().sum().to_dict()}, p >= 0.5 {(ext >= 0.5).sum().to_dict()}")
    print(f"(D) data/tgt_v3_p2blend.csv: {len(out)} studies, changed {int(changed.sum())}; mean |d| on changed "
          f"{(out - t)[changed].abs().mean().round(4).to_dict()}; mean target tgt_v3 {t.mean().round(4).to_dict()} -> {out.mean().round(4).to_dict()}")
    print(f"(E) data/tgt_p2cal.csv = copy of {P2CAL.relative_to(REPO)} (md5 {md5}): {len(cal)} studies, set {cal.notna().sum().to_dict()}")


if __name__ == "__main__":
    main()
