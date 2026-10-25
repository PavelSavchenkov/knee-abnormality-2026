"""Seed stability of PL vs plain all-data models (USER 08:4x: "test stability of PL models. are they as stable as their counterparts? maybe
some are more or less stable and we should train different number of seeds"). OAI: no. CPU, read-only, NO labels.
Same definition as work/alldata_final_2026-10-21/seed_noise.py: per recipe and SHIPPED output (image heads for the syn / prob05 / sxm
recipes), on gold-58 (never trained on by any all-data model), from each seed's deployed (last) checkpoint:
  rho = mean over studies of the across-seed logit variance / the between-study variance of the seed-mean logit (noise-to-signal of ONE
        seed; n seeds leave rho / n), plus the mean seed-seed logit correlation.
Plain = every finished plain all-data seed (oct14 A<k>_<r>, oct16 <r>_all_s1xx incl. null spares); PL = every finished PL seed.
ratio = rho_PL / rho_plain (bootstrap 90% over studies): seeds of PL for the same residual noise as n plain seeds = n x ratio.
SEEDS: noise left with the base's count n (safe3w9) for plain vs PL, and the PL count that matches plain's noise.
  conda run -n rsna python work/pl_alldata_2026-10-21/pl_stability.py        (writes pl_stability.txt / .csv)"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "submissions/oct21_final_alldata")]
import build as B  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

E16, E14 = REPO / "experiments/v3/oct16_experts/runs", REPO / "experiments/v2/oct14_all_data/runs"
SYN = {"cs_fs", "cs_fc125", "s_fs_nf", "s_fc125_nf", "ef3", "ef3fc", "ef3nf", "ef3nffc"}
O14 = {"sub_b0", "m_prob05", "nfnet", "r50", "n_prob05", "oa_v2cor", "oa_v2cor_nf", "s_fs_nf", "s_fc125_nf"}
G = gold_labels().index
BASE = "safe3w9"
lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
rng = np.random.default_rng(0)


def gold_logits(d):
    if (d / "gold.parquet").exists():
        g = pd.read_parquet(d / "gold.parquet")
    else:
        g = pd.read_parquet(d / "preds.parquet")
        g = g[g["set"] == "gold"]
    g = g[g.epoch == g.epoch.max()] if "epoch" in g else g
    g = g.set_index("StudyInstanceUID") if "StudyInstanceUID" in g else g
    num = g.select_dtypes("number").drop(columns=[c for c in ("epoch", "fold") if c in g], errors="ignore").groupby(level=0).mean()
    return num.apply(lambda s: pd.Series(lgt(s.to_numpy(float)), index=s.index) if s.min() >= 0 else s).reindex(G)


def runs(r, pl):
    if r in O14:
        pat, rx = (f"PL2[0-9][0-9]_{r}", rf"PL2\d\d_{re.escape(r)}") if pl else (f"A*_{r}", rf"A\d+_{re.escape(r)}")
        root = E14
    else:
        pat, rx = (f"{r}_pl2_s2[0-9][0-9]", rf"{re.escape(r)}_pl2_s2\d\d") if pl else (f"{r}_all_s1[0-9][0-9]", rf"{re.escape(r)}_all_s1\d\d")
        root = E16
    return [Path(p).parent for p in sorted(glob.glob(str(root / pat / "fold-1/done.txt"))) if re.fullmatch(rx, Path(p).parent.parent.name)]


def shipped(r, W):
    """(package column, recipe output) pairs this recipe ships directly (|share| >= 0.05; DEC-only shares excluded)."""
    out = []
    for c in FINDINGS:
        if r in W.index and abs(W.loc[r, c]) >= 0.05:
            img = (r in SYN and c in ("Synovitis", "Effusion")) or (r in ("m_prob05", "n_prob05") and c == "Synovitis") or (r == "sxm" and c == "Contusion")
            out.append((c, "img_" + c if img else c))
    return out


def rho(Z):
    return Z.var(0, ddof=1).mean() / Z.mean(0).var(ddof=1)


def main():
    S = pd.read_csv(REPO / "work/alldata_final_2026-10-21/shares.csv")
    S = S[(S.form == "effc_off") & (S.w.abs() > 1e-4)]
    W = S.pivot_table(index="recipe", columns="col", values="w", aggfunc="sum").reindex(columns=FINDINGS).fillna(0.0)
    base = B.VARIANTS[BASE]["seeds"]
    rows = []
    for r in W.index:
        P, Q = [gold_logits(d) for d in runs(r, False)], [gold_logits(d) for d in runs(r, True)]
        if len(P) < 2 or len(Q) < 2:
            continue
        for c, oc in shipped(r, W):
            Zp = np.stack([p[oc].to_numpy(float) for p in P])
            Zq = np.stack([q[oc].to_numpy(float) for q in Q])
            if np.isnan(Zp).any() or np.isnan(Zq).any():
                continue
            rp, rq = rho(Zp), rho(Zq)
            bs = []
            for _ in range(300):
                i = rng.integers(0, Zp.shape[1], Zp.shape[1])
                bs.append(rho(Zq[:, i]) / rho(Zp[:, i]))
            n = len(base.get(r, [])) or 3
            cp = np.corrcoef(Zp)[np.triu_indices(len(P), 1)].mean()
            cq = np.corrcoef(Zq)[np.triu_indices(len(Q), 1)].mean()
            rows.append({"recipe": r, "col": c, "share": round(W.loc[r, c], 3), "n_plain": len(P), "n_PL": len(Q), "rho_plain": rp, "rho_PL": rq,
                         "ratio": rq / rp, "ratio90": f"[{np.percentile(bs, 5):.2f}, {np.percentile(bs, 95):.2f}]", "corr_plain": cp, "corr_PL": cq,
                         "n_base": n, "noise_plain@n": rp / n, "noise_PL@n": rq / n, "PL_seeds_for_equal_noise": int(np.ceil(n * rq / rp))})
    R = pd.DataFrame(rows).sort_values(["ratio"], ascending=False)
    R.to_csv(D / "pl_stability.csv", index=False)
    pd.set_option("display.width", 240)
    txt = (f"PL vs plain seed stability on gold-58 (no labels; rho = one seed's noise / signal, lower = more stable; ratio = rho_PL / rho_plain;\n"
           f"n_base = the {BASE} seed count; PL_seeds_for_equal_noise = PL seeds giving the same residual noise as n_base plain seeds). OAI: no.\n"
           + R.round(4).to_string(index=False) + "\n")
    (D / "pl_stability.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
