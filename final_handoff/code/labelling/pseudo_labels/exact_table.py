"""Exact gold-58 per-column table over every exact batch (Experts' local_ref --ids gold): PL_m / AVG_m / seedrow vs PLAIN_m, and the
dp4 group attributions (pl_m - x<group> = that group's PL effect inside the PL package). OAI: no. CPU, read-only.
  conda run -n rsna python work/pl_alldata_2026-10-21/exact_table.py"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))
from common.labels import gold_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

FA = REPO / "submissions/oct21_final_alldata"
G = gold_labels()


def auc(name):
    p = pd.read_csv(FA / f"local_gold_{name}.csv").set_index("StudyInstanceUID").reindex(G.index)
    assert p.notna().all().all(), name
    return pd.Series({c: roc_auc_score(G[c], p[c]) for c in FINDINGS})


def main():
    rows = {}
    for t in ("pl_d1", "pl_dp2", "pl_dp3", "pl_dp4"):
        b = auc(f"{t}_plain_m")
        rows[f"{t} PL-PLAIN"] = auc(f"{t}_pl_m") - b
        rows[f"{t} AVG-PLAIN"] = auc(f"{t}_avg_m") - b
        if (FA / f"local_gold_{t}_seedrow.csv").exists():
            rows[f"{t} null"] = auc(f"{t}_seedrow") - b
    pm = auc("pl_dp4_pl_m")
    for g in ("bone", "loax", "silsep", "oa", "cfgA"):
        rows[f"dp4 PL eff {g}"] = pm - auc(f"pl_dp4_x{g}")
    T = pd.DataFrame(rows).T
    T["MACRO"] = T.mean(1)
    pd.set_option("display.width", 250)
    print((T * 1e3).round(1).to_string())


if __name__ == "__main__":
    main()
