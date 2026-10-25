"""DENSE image-head targets for the bone expert (Experts V3 10-20 13:4x; USER via Master: "models trained on new labels should not lose to older
models ... keep investigating where you are losing gains"; Master's diagnosis of p2b's Fracture loss: the control's frac head trains on DENSE
tgt_v3 (3,478 studies), the image head on SPARSE Opus (442, 32 positives) and overfits after ep 15; OAI: no; CPU). Fracture / Contusion:
 data/tgt_p2_bone_dv3.csv  Opus where labelled, tgt_v3 elsewhere (dense partial: every non-gold study has a target)
 data/tgt_p2_bone_dpl.csv  Opus where labelled, the shipped column's OOF probability elsewhere (pkg961 oof_f5_win f0-2 + data/clean_f34 f3-4)
                           = PL + image labels. CAVEAT: a training study's OOF comes from fold models that trained on the other folds' labels,
                           including the validation fold of the run (the standard PL leak path).
Gold studies are excluded. Prints coverage and how many targets come from each source.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_dense.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from lib import GOLD  # noqa: E402

COLS = ["Fracture", "Contusion"]


def pkg_prob(c):
    pk = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)[c]
    cl = pd.read_csv(EXP / f"data/clean_f34/{c}.csv").set_index("StudyInstanceUID").logit
    assert not pk.index.intersection(cl.index).size
    return 1 / (1 + np.exp(-pd.concat([pk, cl])))


def main():
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")[COLS]
    t3 = pd.read_csv(REPO / "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv").set_index("StudyInstanceUID")[COLS]
    t3 = t3[~t3.index.isin(GOLD.index)]
    dv3 = t3.copy()
    dv3.loc[op.index] = op.values
    pl = pd.DataFrame({c: pkg_prob(c) for c in COLS})
    pl = pl[~pl.index.isin(GOLD.index)]
    dpl = pl.copy()
    miss = op.index.difference(dpl.index)
    dpl = dpl.reindex(dpl.index.union(op.index))
    dpl.loc[op.index] = op.values
    for name, d in (("dv3", dv3), ("dpl", dpl)):
        assert not d.index.isin(GOLD.index).any() and d.notna().all().all(), name
        d.rename_axis("StudyInstanceUID").to_csv(EXP / f"data/tgt_p2_bone_{name}.csv")
    print(f"tgt_p2_bone_dv3: {len(dv3)} studies (Opus {len(op)}, tgt_v3 {len(dv3) - len(op)}); mean {dv3.mean().round(3).to_dict()}, >= .5 {(dv3 >= .5).sum().to_dict()}")
    print(f"tgt_p2_bone_dpl: {len(dpl)} studies (Opus {len(op)}, package OOF {len(dpl) - len(op)}; labelled without OOF {len(miss)}); mean "
          f"{dpl.mean().round(3).to_dict()}, >= .5 {(dpl >= .5).sum().to_dict()}")


if __name__ == "__main__":
    main()
