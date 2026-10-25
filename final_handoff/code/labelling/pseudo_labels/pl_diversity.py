"""Does PL collapse cross-recipe diversity? Per column: mean pairwise corr (gold-58 logits, no labels) between DIFFERENT recipes'
seed-mean outputs, plain vs PL (same seed count per recipe), and each recipe's corr to the rest of the column's members. OAI: no. CPU.
  conda run -n rsna python work/pl_alldata_2026-10-21/pl_diversity.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
sys.path.insert(0, str(D))
import pl_stability as S  # noqa: E402

COLS = {"Lateral OA": ["oasep", "oasep_nf", "silsep", "oa_v2cor", "oa_v2cor_nf", "sub_b0", "nfnet"],
        "PF OA": ["oa_v2cor", "oa_v2cor_nf", "sub_b0", "nfnet"],
        "Medial OA": ["oa_v2cor", "oa_v2cor_nf", "sub_b0", "nfnet"],
        "Fracture": ["b3c_rep", "b3l_rep", "b3n_rep", "bfs_rep", "sub_b0", "nfnet"],
        "Contusion": ["b3c_rep", "b3l_rep", "b3n_rep", "bfs_rep", "sxm"],
        "MCL": ["w5b_mcl", "sd1b", "bsd_nf", "b3g", "sub_b0", "nfnet"],
        "Effusion": ["ef3", "ef3fc", "ef3nf", "ef3nffc", "sub_b0", "nfnet"],
        "Baker's": ["sub_b0", "nfnet"]}


def seed_mean(r, pl, c, k):
    oc = S.shipped(r, W)
    oc = dict(oc).get(c, c)
    Z = [S.gold_logits(d)[oc].to_numpy(float) for d in S.runs(r, pl)[:k]]
    return np.mean(Z, 0) if Z else None


def main():
    global W
    s = pd.read_csv(S.REPO / "work/alldata_final_2026-10-21/shares.csv")
    s = s[(s.form == "effc_off") & (s.w.abs() > 1e-4)]
    W = s.pivot_table(index="recipe", columns="col", values="w", aggfunc="sum").reindex(columns=S.FINDINGS).fillna(0.0)
    for c, rs in COLS.items():
        k = {r: min(len(S.runs(r, False)), len(S.runs(r, True)), 3) for r in rs}
        rs = [r for r in rs if k[r] >= 1]
        P = {r: seed_mean(r, False, c, k[r]) for r in rs}
        Q = {r: seed_mean(r, True, c, k[r]) for r in rs}
        cp = np.mean([np.corrcoef(P[a], P[b])[0, 1] for a, b in itertools.combinations(rs, 2)])
        cq = np.mean([np.corrcoef(Q[a], Q[b])[0, 1] for a, b in itertools.combinations(rs, 2)])
        per = []
        for r in rs:
            rest_p = np.mean([P[x] for x in rs if x != r], 0)
            per.append(f"{r} {np.corrcoef(P[r], rest_p)[0, 1]:.3f}->{np.corrcoef(Q[r], rest_p)[0, 1]:.3f}")
        print(f"{c:10s} pairwise corr between recipes plain {cp:.3f} -> PL {cq:.3f} | corr to rest (plain rest): " + ", ".join(per))


if __name__ == "__main__":
    main()
