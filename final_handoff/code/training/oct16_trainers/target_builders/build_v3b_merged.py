"""Bone main-target file for members whose deployed recipe trains on tgt_v3 (b_v3, b_nf, bfs_fs): tgt_v3 (dense, all studies) with the
v3b Opus reads replacing Fracture / Contusion cells where read (OAI: no; CPU). A tgt_csv override REPLACES the ref's tgt_csv, so passing
the sparse data/v3b/tgt_p2_bone.csv on refs/exp_b_v3 drops tgt_v3 to v1prob on every unread study (the 10-20 b_p2part_w8 confound).
b_lr3 trains on v1prob (tgt_csv '') and keeps the sparse file. Writes data/<SNAP>/tgt_bone_v3m.csv (SNAP default v3b; v3c = 10-20 18:57).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_v3b_merged.py [SNAP]"""
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    base = pd.read_csv(EXP / "data/tgt_v3.csv").set_index("StudyInstanceUID")
    snap = sys.argv[1] if len(sys.argv) > 1 else "v3b"
    op = pd.read_csv(EXP / f"data/{snap}/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    assert op.index.isin(base.index).all() and base.notna().all().all(), "v3b read outside tgt_v3 / tgt_v3 not dense"
    m = base.copy()
    for c in ("Fracture", "Contusion"):
        v = op[c].dropna()
        m.loc[v.index, c] = v
    m.to_csv(EXP / f"data/{snap}/tgt_bone_v3m.csv")
    ch = (m != base).sum()
    print(f"OAI: no. {snap} tgt_bone_v3m: {len(m)} studies (dense), Opus-replaced cells F {int(op.Fracture.notna().sum())} / C {int(op.Contusion.notna().sum())}, "
          f"changed values F {int(ch.Fracture)} / C {int(ch.Contusion)}; >= .5 F {int((m.Fracture >= .5).sum())} (tgt_v3 {int((base.Fracture >= .5).sum())}) "
          f"/ C {int((m.Contusion >= .5).sum())} (tgt_v3 {int((base.Contusion >= .5).sum())})")


if __name__ == "__main__":
    main()
