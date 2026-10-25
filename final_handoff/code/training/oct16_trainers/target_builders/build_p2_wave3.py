"""P2 WAVE 3 targets (Experts V3 10-20 ~12:0x; USER via Master: "you may want to train on pseudo labels + image-based labels, with different
heads, or blends, etc. keep experimenting in diverse directions and analysing if we are measuring right things"; label-form quality on gold-58 in
work/claude_labels_oct20/eval_target_forms.txt: PF best = 0.5 Opus + 0.5 text; Fracture best = tgt_v3 / tgt_v3 + Opus; OAI: no; CPU). Text v1 =
common.labels.probability_table('v1') (the same source as eval_target_forms); Opus = data/tgt_p2_bone.csv / tgt_p2_pf.csv (NaN = unlabelled).
 W3-1 data/tgt_p2_pf_blend.csv    img_pf_patella = 0.5 Opus + 0.5 text v1 on the labelled studies (masked elsewhere)  -> oa aux_csv
 W3-2 data/tgt_p2_pf_pl.csv       img_pf_patella (as W3-1) + pl_pf_patella = the shipped PF OOF probability (pseudo label) on every study
                                  with a shipped OOF: pkg961 oof_f5_win (folds 0-2) + data/clean_f34/PF_OA.csv (folds 3-4, C213 clean
                                  rebuild of the 0.961 PF form; PF passed the C213 parity). Fold 1 trains on folds 0 / 2 / 3 / 4 -> its PL
                                  rows are f0 + f2 (pkg961) + f3 + f4 (clean); fold-1 rows exist in the file but are validation there.
                                  CAVEAT: the f0 / f2 / f3 / f4 OOF models trained on fold 1's labels (standard PL leak path).
 W3-3 data/tgt_pf_plonly.csv      pl_pf_patella only (the PL control)
 W3-4 data/tgt_p2_bone_v3o.csv    Fracture / Contusion = 0.5 tgt_v3 + 0.5 Opus on the labelled studies (masked elsewhere) -> bone img_bone_csv
The 'patella' in the column names routes the aux heads to the PF bag (oa_train aux_bag).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_wave3.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from common.labels import probability_table  # noqa: E402
from lib import GOLD  # noqa: E402

TGT_V3 = REPO / "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv"


def main():
    text = probability_table("v1")
    pf = pd.read_csv(EXP / "data/tgt_p2_pf.csv").set_index("StudyInstanceUID").img_pf_patella.dropna()
    tp = text["pf_oa"].reindex(pf.index)
    print(f"text v1 PF missing on {int(tp.isna().sum())} of {len(pf)} labelled studies -> Opus alone there (as eval_target_forms: o.fillna(t))")
    blend = 0.5 * pf + 0.5 * tp.fillna(pf)
    blend.rename("img_pf_patella").to_frame().to_csv(EXP / "data/tgt_p2_pf_blend.csv")
    pk = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)["PF OA"]
    c34 = pd.read_csv(EXP / "data/clean_f34/PF_OA.csv").set_index("StudyInstanceUID").logit
    assert not pk.index.intersection(c34.index).size, "pkg961 f0-2 and clean f3-4 overlap"
    pl = (1 / (1 + np.exp(-pd.concat([pk, c34])))).rename("pl_pf_patella")
    pl = pl[~pl.index.isin(GOLD.index)]
    pd.concat([blend.rename("img_pf_patella"), pl], axis=1).rename_axis("StudyInstanceUID").to_csv(EXP / "data/tgt_p2_pf_pl.csv")
    pl.to_frame().rename_axis("StudyInstanceUID").to_csv(EXP / "data/tgt_pf_plonly.csv")
    b = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    t3 = pd.read_csv(TGT_V3).set_index("StudyInstanceUID")
    v3o = pd.DataFrame({c: 0.5 * t3[c].reindex(b.index) + 0.5 * b[c] for c in ("Fracture", "Contusion")}).dropna(how="all")
    assert v3o.notna().all().all() and not v3o.index.isin(GOLD.index).any()
    v3o.rename_axis("StudyInstanceUID").to_csv(EXP / "data/tgt_p2_bone_v3o.csv")
    print(f"W3-1 tgt_p2_pf_blend: {len(blend)} labelled, mean {blend.mean():.3f} (Opus {pf.mean():.3f}, text {text["pf_oa"].reindex(pf.index).mean():.3f}), "
          f">= .5: {int((blend >= .5).sum())}")
    print(f"W3-2 / 3 pseudo label pl_pf_patella: {len(pl)} studies (pkg961 f0-2 {len(pk)}, clean f3-4 {len(c34)}; gold dropped), mean {pl.mean():.3f}")
    print(f"W3-4 tgt_p2_bone_v3o: {len(v3o)} labelled, mean {v3o.mean().round(3).to_dict()}, >= .5 {(v3o >= .5).sum().to_dict()}")


if __name__ == "__main__":
    main()
