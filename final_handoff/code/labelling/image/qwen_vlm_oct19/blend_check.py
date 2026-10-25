"""Blend open-VLM reads with the 0.961 OOF: is there a usable signal out of sample?
VLM = mean of z(logit P(yes)) over the given score files (one per prompt / view / model config). Blend = z(OOF) + w * VLM.
For each finding and reference:
- w is picked on two OOF folds (grid W, AUC) and applied to the third; the held-out AUC gain is pooled over the 3 folds;
- null: the same cross-fit with the VLM vector permuted across studies (N_PERM times); p = share of nulls >= the real gain;
- fixed w = 0.25 (pre-set from the pass-1 look, no selection) with a bootstrap 90 % interval.
References: codex (Effusion / Synovitis, Codex image reads >= 0.5), trusted (hand-checked text cells), text (v1 >= 0.5).
Usage: python blend_check.py SCORES.csv [SCORES2.csv ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import eval_cells, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

W = (0, 0.1, 0.25, 0.5, 1.0)
N_PERM, N_BOOT, W_FIX = 200, 1000, 0.25
oof_all = pd.read_parquet(REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet").loc["f5_win"]
codex = (pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")[["Effusion", "Synovitis"]] >= 0.5).astype(float)
codex["Baker's"] = (pd.read_csv(Path(__file__).parent / "out/codex_bakers.csv", index_col="StudyInstanceUID")["Baker's"] >= 0.5).astype(float)   # 516 studies, one text cell
cells = eval_cells()
trusted = cells[cells.verdict == "trusted"].pivot(index="StudyInstanceUID", columns="finding", values="label")
text = (probability_table("v1").fillna(0) >= 0.5).astype(float).rename(columns={KEY_OF[f]: f for f in FINDINGS})
z = lambda s: (s - s.mean()) / s.std()


def vlm_read(paths):
    reads = []
    for p in paths:
        r = pd.read_csv(p).pivot_table(index="StudyInstanceUID", columns="finding", values="p_yes").clip(1e-6, 1 - 1e-6)
        reads.append(z(np.log(r / (1 - r))))
    return pd.concat(reads).groupby(level=0).mean(), len(reads)


def crossfit_gain(y, o, v, fold):
    """Held-out AUC gain of o + w v over o, w picked on the other folds; pooled over folds (AUC on the concatenated held-out scores)."""
    s = np.empty(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        w = max(W, key=lambda w: roc_auc_score(y[tr], o[tr] + w * v[tr]) if 0 < y[tr].sum() < tr.sum() else -w)
        s[te] = o[te] + w * v[te]
    return roc_auc_score(y, s) - roc_auc_score(y, o)


def main():
    v_all, n_cfg = vlm_read(sys.argv[1:])
    rng = np.random.default_rng(0)
    rows = []
    for f in [f for f in FINDINGS if f in v_all]:
        for name, ref in [("codex", codex), ("trusted", trusted), ("text", text)]:
            if f not in ref:
                continue
            d = pd.DataFrame({"y": ref[f], "o": oof_all[f], "fold": oof_all["fold"], "v": v_all[f]}).dropna()
            if not 5 <= d.y.sum() <= len(d) - 5:
                continue
            y, o, v, fold = d.y.to_numpy(int), z(d.o).to_numpy(), z(d.v).to_numpy(), d.fold.to_numpy()
            g = crossfit_gain(y, o, v, fold)
            null = np.array([crossfit_gain(y, o, rng.permutation(v), fold) for _ in range(N_PERM)])
            base = roc_auc_score(y, o)
            fix = roc_auc_score(y, o + W_FIX * v) - base
            bs = []
            for _ in range(N_BOOT):
                b = rng.integers(0, len(y), len(y))
                if 0 < y[b].sum() < len(b):
                    bs.append(roc_auc_score(y[b], o[b] + W_FIX * v[b]) - roc_auc_score(y[b], o[b]))
            rows.append(dict(finding=f, ref=name, n=f"{y.sum()}/{len(y)}", vlm_auc=round(roc_auc_score(y, v), 3), oof_auc=round(base, 4),
                             cv_gain=round(1e3 * g, 1), null_p95=round(1e3 * np.quantile(null, 0.95), 1), p=round((null >= g).mean(), 3),
                             fix_gain=round(1e3 * fix, 1), fix_90=f"[{1e3 * np.quantile(bs, 0.05):+.1f}, {1e3 * np.quantile(bs, 0.95):+.1f}]"))
    print(f"VLM = mean of {n_cfg} config(s): {', '.join(sys.argv[1:])}; gains x1e-3 AUC")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
