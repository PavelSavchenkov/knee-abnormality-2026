"""U80 correctors (CORRECTORS.md protocol): the base logits per non-gold study x finding = the SHIPPED 0.961 fold-package OOF, each study from its
own fold models (out-of-sample for training and eval studies alike) -> data/corrector_base_0961.csv (StudyInstanceUID + 12 FINDINGS logits).
- folds 0-2: refs/pkg961/oof_f5_win (the shipped OOF) for every column;
- folds 3-4 (no shipped OOF): only the columns whose 0.961 form is rebuildable exactly from stored members: ACL / MCL (the 3x3 pipeline mean)
  and Fracture (the crop-fused bone column in the package's no-NFNet form there, c179.column); the rest stay NaN (no base -> that output does
  not train on the study).
F6: on folds 0-2 the rebuilt ACL / MCL / Fracture equal oof_f5_win (asserted). OAI: no.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_corrector_base.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP.parents[2]), str(EXP / "scripts"), str(EXP / "bone")]
from c179_b_v3floc import column  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

lgt = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))  # noqa: E731


def main():
    G = gold_labels()
    pk = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)
    m = pd.read_parquet(EXP / "refs/pkg961/members.parquet")
    m = m[m.set == "oof"]

    def pipe_mean(f, cols):
        parts = []
        for p in ("main", "nfnet", "r50"):
            d = m[(m.member == p) & (m.fold == f)].groupby("StudyInstanceUID")[cols].mean()
            parts.append(d if d.min().min() < 0 else pd.DataFrame(lgt(d.to_numpy(float)), index=d.index, columns=cols))
        return sum(parts) / 3

    out = [pk.loc[~pk.index.isin(G.index), FINDINGS]]
    for f in (0, 1, 2):                                     # F6: the rebuild equals the shipped OOF where both exist
        r = pipe_mean(f, ["ACL", "MCL"])
        i = r.index.intersection(pk.index[pk.fold == f])
        d1 = float(np.abs(r.loc[i] - pk.loc[i, ["ACL", "MCL"]]).max().max())
        fr = column(f, "oof", None)["Fracture"]
        j = fr.index.intersection(pk.index[pk.fold == f])
        d2 = float(np.abs(fr[j] - pk.loc[j, "Fracture"]).max())
        assert d1 < 1e-4 and d2 < 1e-4, (f, d1, d2)
        print(f"F6 f{f}: ACL / MCL 3x3 mean vs shipped {d1:.1e}; Fracture crop-fused vs shipped {d2:.1e}")
    for f in (3, 4):
        r = pipe_mean(f, ["ACL", "MCL"])
        r["Fracture"] = column(f, "oof", None)["Fracture"].reindex(r.index)
        r = r[~r.index.isin(G.index)].reindex(columns=FINDINGS)
        out.append(r)
    base = pd.concat(out)
    assert base.index.is_unique, "a study appears twice"
    base.index.name = "StudyInstanceUID"
    base.to_csv(EXP / "data/corrector_base_0961.csv")
    print(f"data/corrector_base_0961.csv: {len(base)} studies; non-NaN per column: " + ", ".join(f"{c} {int(base[c].notna().sum())}" for c in FINDINGS))


if __name__ == "__main__":
    main()
