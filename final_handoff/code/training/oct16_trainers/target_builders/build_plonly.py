"""PL-ONLY control target for p2b_dpl (Master 10-20 16:4x: dpl's image head may be a distillation of the 0.961 package, not the new labels;
OAI: no; CPU). data/tgt_p2_bone_plonly.csv = the SAME dense package-PL fill as tgt_p2_bone_dpl.csv on EVERY study, the Opus-read cells ALSO
the package PL (no Opus anywhere; build_p2_dense.pkg_prob, the rule dpl's fill used). dpl - plonly isolates the Opus contribution and
cancels the shared fold leak of the PL fill. Asserted: identical to dpl off the 559 read studies.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_plonly.py"""
from pathlib import Path

import pandas as pd

from build_p2_dense import COLS, pkg_prob
from lib import GOLD

EXP = Path(__file__).resolve().parents[1]


def main():
    pl = pd.DataFrame({c: pkg_prob(c) for c in COLS})
    pl = pl[~pl.index.isin(GOLD.index)].rename_axis("StudyInstanceUID")
    dpl = pd.read_csv(EXP / "data/tgt_p2_bone_dpl.csv").set_index("StudyInstanceUID")[COLS]
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    miss = dpl.index.difference(pl.index)
    assert miss.isin(op.index).all(), "a non-read dpl study lacks PL"
    off = dpl.index[~dpl.index.isin(op.index)]
    assert (pl.loc[off, COLS] - dpl.loc[off]).abs().max().max() < 1e-9, "fill != dpl's fill"
    out = pl.reindex(dpl.index)
    print(f"OAI: no. read studies without a package PL value (kept NaN = masked): {len(miss)}")
    out.to_csv(EXP / "data/tgt_p2_bone_plonly.csv")
    print(f"OAI: no. tgt_p2_bone_plonly.csv: {len(out)} studies, NaN {out.isna().sum().to_dict()}; == dpl off the read studies; on the 559 read "
          f"studies PL instead of Opus (mean PL {out.loc[op.index.intersection(out.index)].mean().round(3).to_dict()} vs Opus {op[COLS].mean().round(3).to_dict()})")


if __name__ == "__main__":
    main()
