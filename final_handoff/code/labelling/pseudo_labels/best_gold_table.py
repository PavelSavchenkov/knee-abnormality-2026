"""Best-candidate gold hold table (Master 09:1x): exact gold-58 outputs (Experts' local_ref --ids gold) of each candidate vs safe3w9.
Per column + macro deltas (AUC x1e-3), the fresh seed-swap null (pl_f_seedrow - safe3w9; all twins frozen -> plain_m == safe3w9's
seeds), a paired study bootstrap (90%) of the macro delta, and SANITY flags (8b 09:3x to Master): macro < -SD (seed-row sd, 0.0014
established) or a column < -3 x its seed-noise sd (RMS over every seed-swap null: d1 / dp2 / dp3 / dp4 + the fresh full swap; Master 09:3x:
catch breakage, ~1-2 % false flags). NOTES only: columns in (-3, -2] sd, and columns worse than max(|own null move|, one rank swap) (that
one-draw rule flagged 8 / 12 null-vs-null pairs). Gold = sanity, not the selector. OAI: no. CPU, read-only.
  conda run -n rsna python work/pl_alldata_2026-10-21/best_gold_table.py [--base safe3w9] [--null pl_f_seedrow] [--cands a,b]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(__file__).parent
REPO = D.parents[1]
sys.path.insert(0, str(REPO))
from common.labels import gold_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

FA = REPO / "submissions/oct21_final_alldata"
NULLS = [("pl_d1_seedrow", "pl_d1_plain_m"), ("pl_dp2_seedrow", "pl_dp2_plain_m"), ("pl_dp3_seedrow", "pl_dp3_plain_m"),
         ("pl_dp4_seedrow", "pl_dp4_plain_m")]                     # + (--null, --base)
CANDS = ["pl_f_pl_m", "best_b1", "best_b2", "best_b6", "best_b6a", "best_b6c", "best_b6ca"]
SD = 0.0014
G = gold_labels()


def preds(name):
    p = pd.read_csv(FA / f"local_gold_{name}.csv").set_index("StudyInstanceUID").reindex(G.index)[FINDINGS]
    assert p.notna().all().all() and ((p >= 0) & (p <= 1)).all().all(), f"{name}: missing / out-of-range gold rows"
    return p


def aucs(p, idx=None):
    g, q = (G, p) if idx is None else (G.iloc[idx], p.iloc[idx])
    return pd.Series({c: roc_auc_score(g[c], q[c]) if g[c].nunique() == 2 else np.nan for c in FINDINGS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="safe3w9")
    ap.add_argument("--null", default="pl_f_seedrow")
    ap.add_argument("--cands", default=",".join(CANDS))
    a = ap.parse_args()
    base = preds(a.base)
    a0 = aucs(base)
    null = aucs(preds(a.null)) - a0
    moves = [aucs(preds(n)) - aucs(preds(b)) for n, b in NULLS] + [null]
    swap = pd.Series({c: 1 / (G[c].sum() * (len(G) - G[c].sum())) for c in FINDINGS})
    sdc = np.maximum(np.sqrt((pd.DataFrame(moves) ** 2).mean()), swap)   # floor: one rank swap (5 draws under-read quiet columns)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(G), len(G)) for _ in range(1000)]
    rows, flags = {"null (seedrow)": null}, {}
    for name in [c for c in a.cands.split(",") if (FA / f"local_gold_{c}.csv").exists()]:
        p = preds(name)
        d = aucs(p) - a0
        rows[name] = d
        bd = [np.nanmean(aucs(p, i) - aucs(base, i)) for i in boots]
        lo, hi = np.percentile(bd, [5, 95])
        f = [f"MACRO {d.mean() * 1e3:+.1f} < -{SD * 1e3:.1f}"] if d.mean() < -SD else []
        z = d / sdc
        f += [f"{c} {d[c] * 1e3:+.1f} ({z[c]:+.1f} sd)" for c in FINDINGS if z[c] < -3]
        nt = [f"{c} {d[c] * 1e3:+.1f} ({z[c]:+.1f} sd)" for c in FINDINGS if -3 <= z[c] < -2]
        nt += [f"{c} {d[c] * 1e3:+.1f} < own seedrow" for c in FINDINGS if z[c] >= -2 and d[c] < -max(abs(null[c]), swap[c]) - 1e-12]
        flags[name] = (f"{'FLAG' if f else 'PASS'} macro {d.mean() * 1e3:+.1f} boot90 [{lo * 1e3:+.1f}, {hi * 1e3:+.1f}] | flags: {', '.join(f) or 'none'}"
                       f" | notes: {', '.join(nt) or 'none'}")
    T = pd.DataFrame(rows).T
    T.insert(0, "MACRO", T.mean(axis=1))
    pd.set_option("display.width", 250)
    print(f"Gold-58 exact, delta vs {a.base} (AUC x1e-3; {a.base} macro {a0.mean():.4f}). Seed-row sd {SD} established; this null's macro "
          f"move {null.mean() * 1e3:+.1f}. OAI: no.")
    print((T * 1e3).round(1).to_string())
    print("seed-noise sd per column x1e-3 (RMS of", len(moves), "nulls):", (sdc * 1e3).round(1).to_dict())
    print("\n".join(f"{k}: {v}" for k, v in flags.items()))
    if {"best_b1", "best_b2"} <= set(rows):
        print(f"xgen (b2 - b1, generalists PL vs plain on the final twin set): macro {(rows['best_b2'] - rows['best_b1']).mean() * 1e3:+.1f}")


if __name__ == "__main__":
    main()
