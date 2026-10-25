"""INFO (Master 10-21 23:4x: PL variant reads; OAI: no; CPU): gold-58 per-column AUC + 12-column mean of local gold runs
(local_gold_<tag>.csv from local_ref.py --ids gold) and a PAIRED bootstrap over the gold studies of every tag vs REF (the same resampled
studies for both; 5000 resamples): mean difference, sd, 95% CI, P(diff <= 0), per-column sd. The bootstrap measures gold-58 study sampling,
which model averaging does not remove; the seed / fold noise of big packages is a small extra term (USER 10-21). Numbers go to the asker
only (labeller blindness).
  conda run -n rsna python submissions/oct21_final_alldata/gold_table.py REF_TAG TAG [TAG ...] [--pairs A:B,C:D]
--pairs: bootstrap exactly these differences (A - B) instead of every TAG vs REF_TAG (8b 10-22: pl_m - plain_m, avg_m - plain_m, avg_m - pl_m)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.labels import gold_labels  # noqa: E402

FIND = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def auc(p, t):
    """Mann-Whitney AUC per column (ties count half)."""
    out = np.empty(p.shape[1])
    for j in range(p.shape[1]):
        r = rankdata(p[:, j])
        n1 = t[:, j].sum()
        out[j] = (r[t[:, j] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(t) - n1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--pairs", default="")
    a = ap.parse_args()
    tags = a.tags
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")] if a.pairs else [(t, tags[0]) for t in tags[1:]]
    assert pairs and all(len(q) == 2 and set(q) <= set(tags) for q in pairs), (tags, pairs)
    g = gold_labels()
    y = g[FIND].astype(int).to_numpy()
    P = {}
    for t in tags:
        d = pd.read_csv(HERE / f"local_gold_{t}.csv", index_col=0).reindex(g.index)
        assert d[FIND].notna().all().all() and np.isfinite(d[FIND].to_numpy()).all(), f"{t}: every gold study, finite"
        P[t] = d[FIND].to_numpy()
    T = pd.DataFrame({t: auc(p, y) for t, p in P.items()}, index=FIND)
    T.loc["MEAN (12)"] = T.mean()
    pd.set_option("display.width", 250)
    print(f"gold-58 AUC (n {len(y)}; INFO)\n" + T.round(4).to_string())
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(5000)]
    for t, ref in pairs:
        D = np.array([auc(P[t][i], y[i]) - auc(P[ref][i], y[i]) for i in idx])
        m = D.mean(axis=1)
        d0 = T.loc["MEAN (12)", t] - T.loc["MEAN (12)", ref]
        print(f"\n{t} - {ref}: mean {d0:+.4f}; paired bootstrap sd {m.std():.4f}; 95% [{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]; "
              f"P(<= 0) {np.mean(m <= 0):.3f}")
        print("  per column: " + "; ".join(f"{c} {T.loc[c, t] - T.loc[c, ref]:+.4f} (sd {s:.4f})" for c, s in zip(FIND, D.std(axis=0))))


if __name__ == "__main__":
    main()
