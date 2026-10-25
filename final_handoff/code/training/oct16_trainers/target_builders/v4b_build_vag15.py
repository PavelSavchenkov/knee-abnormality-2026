"""vag15 = v4b with every silent (not_mentioned) cell back at v1's trainer fill 0.15: v4b's vague-cell calibration alone. Splits v4b's
gain into 'vague calibration' vs 'silent level / ordering' (Labels Work 10-15 ~19:20; silence-only sets nm05 / sil failed on GPU f3).
-> data/targets/vag15_f{0-4,all}.csv"""
import pandas as pd

from base import FINDINGS, S1
from build_variants import OUTD

for fold in list(range(5)) + ["all"]:
    T = pd.read_csv(OUTD / f"v4b_f{fold}.csv", index_col=0)
    for f in FINDINGS:
        T.loc[S1.reindex(T.index)[f] == "not_mentioned", f] = 0.15
    T.to_csv(OUTD / f"vag15_f{fold}.csv")
    print(fold, T.shape, flush=True)
