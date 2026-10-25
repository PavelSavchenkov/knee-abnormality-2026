"""LEAK-FREE dense pseudo-label target for fold F (default 1) (Master 10-20 16:4x: size the p2b_dpl leak; OAI: no; CPU). p2b_dpl's fill on a fold-F
student's training studies = the package OOF from fold models trained on folds != G (INCLUDING fold F) -> the teacher saw the held-out fold.
Here the fill on fold-1 TRAINING studies = the deployed fold-1 b_v3 bone model (trained on folds != 1) IN-SAMPLE on those studies
(runs/dplt_f{F}/fold{F}/eval.parquet, main heads Fracture / Contusion, probabilities); Opus where read (frozen 559) on top; fold-1 studies NaN
(never trained on). In-sample fills are sharper (closer to the text targets the teacher fit) - part of any dpl_lf loss is that, not the leak.
data/tgt_p2_bone_dpl_lf_f{F}.csv. conda run -n rsna python experiments/v3/oct16_experts/scripts/build_dpl_lf.py [F]"""
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    F = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cols = ["Fracture", "Contusion"]
    ev = pd.read_parquet(EXP / f"runs/dplt_f{F}/fold{F}/eval.parquet")
    ev = (ev.set_index("StudyInstanceUID") if "StudyInstanceUID" in ev else ev)[cols].groupby(level=0).mean()
    ids = pd.read_csv(EXP / f"data/f{F}_train_ids.csv").StudyInstanceUID
    assert ev.index.isin(ids).all() and ev.notna().all().all(), f"teacher rows outside the fold-{F} training set / NaN"
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")[cols]
    dpl = pd.read_csv(EXP / "data/tgt_p2_bone_dpl.csv").set_index("StudyInstanceUID")[cols]
    out = pd.DataFrame(index=dpl.index, columns=cols, dtype=float).rename_axis("StudyInstanceUID")
    out.loc[ev.index.intersection(out.index)] = ev.loc[ev.index.intersection(out.index)].to_numpy()
    rd = op.index[op.index.isin(ids)]
    out.loc[rd] = op.loc[rd].to_numpy()
    out.to_csv(EXP / f"data/tgt_p2_bone_dpl_lf_f{F}.csv")
    tr = out.index.isin(ids)
    print(f"OAI: no. tgt_p2_bone_dpl_lf_f{F}.csv: {int(tr.sum())} fold-{F} training studies filled ({len(rd)} Opus-read on top), "
          f"{int((~tr).sum())} others NaN; teacher in-sample mean {ev.mean().round(3).to_dict()} vs dpl fill mean on the same studies "
          f"{dpl.loc[ev.index.intersection(dpl.index)].mean().round(3).to_dict()}; corr(teacher, dpl fill) "
          + ", ".join(f"{c} {ev[c].corr(dpl.loc[ev.index.intersection(dpl.index), c], method='spearman'):.3f}" for c in cols))


if __name__ == "__main__":
    main()
