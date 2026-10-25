"""Gold-58 monitor of the PL counterpart (USER 22:1x: compare every PL model with its individual counterpart - sanity, PL should be better
- and each class's full prediction of the original vs the new package on gold; try replacing models / ensembling PL + original; propose up to
5 probes). OAI: no. CPU, read-only. Gold-58 = 58 studies, a few positives per class: a trend, never a decision rule alone.
TWINS: the final candidate's k-th plain seed of recipe r (build.VARIANTS['final']) <-> PL seed 200+k (oct16 <r>_pl2_s2xx, oct14 PL2xx_<r>).
A PAIRS     per recipe x shipped column (its shipped output: image heads for the syn / prob05 / sxm recipes): AUC plain_k vs PL_k over the
            pairs that exist; mean delta and wins; ALL12 = mean delta over the 12 text outputs for 12-output models (sanity).
B PACKAGE   LINEARISED: class logit = sum_r w_rc x (recipe r's seed-mean logit), w_rc = Experts' shares.csv 'effc_off' (= the final form; d final
            logit / d recipe seed-mean logit through the package's own functions; DEC-only shares routed through the context output). Fusions /
            constants are linearised: compare variants with each other. Variants: PLAIN (final seeds) | SWAP (each existing twin replaces its
            plain seed = the all-PL counterpart when complete) | HALF (per recipe the first ceil(n/2) seeds plain, the rest PL twins: runtime-
            neutral mix) | BLEND (mean of the plain and the PL seed-means: plain + PL ensemble, double the models). Bootstrap over gold studies.
C MARGINALS swap ONE recipe to its twins (others plain): macro / per-class delta (which recipes help or hurt).
D TREND     one line per run in pl_gold_trend.csv.
  conda run -n rsna python work/pl_alldata_2026-10-21/pl_gold.py        (writes pl_gold.txt)"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "submissions/oct21_final_alldata")]
import build as B  # noqa: E402  (VARIANTS['final'], RECIPES; read-only)
from common.labels import gold_labels  # noqa: E402
from common.metrics import auc  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

E16, E14 = REPO / "experiments/v3/oct16_experts/runs", REPO / "experiments/v2/oct14_all_data/runs"
SYN = {"cs_fs", "cs_fc125", "s_fs_nf", "s_fc125_nf", "ef3", "ef3fc", "ef3nf", "ef3nffc"}
DEC_CTX = {"MCL": "Contusion", "Lateral Meniscus": "ACL"}
G = gold_labels()
NBOOT = 400
BASE = "safe3w9"                         # Master 08:0x: the submission candidate (sxm OFF); twins pair with its k-th seed
GROUPS = (("LOA extras", ["oasep", "oasep_nf", "silsep"], ["Lateral OA"]), ("bone", ["b3c_rep", "b3l_rep", "b3n_rep", "bfs_rep", "sxm"], ["Fracture", "Contusion"]),
          ("OA experts", ["oa_v2cor", "oa_v2cor_nf"], ["Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus"]),
          ("Synovitis", ["cs_fs", "cs_fc125", "s_fs_nf", "s_fc125_nf", "m_prob05", "n_prob05"], ["Synovitis"]), ("EF3", ["ef3", "ef3fc", "ef3nf", "ef3nffc"], ["Effusion"]),
          ("ACL experts", ["z1_acl", "b3g"], ["ACL"]), ("MCL experts", ["w5b_mcl", "b3g", "sd1b", "bsd_nf"], ["MCL"]), ("generalists", ["sub_b0", "nfnet", "r50"], FINDINGS))
lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731


def out_col(r, c, W):
    if c in DEC_CTX and W.loc[r, DEC_CTX[c]] != 0 and abs(W.loc[r, c]) < 0.05:
        return out_col(r, DEC_CTX[c], W)
    if (r in SYN and c in ("Synovitis", "Effusion")) or (r in ("m_prob05", "n_prob05") and c == "Synovitis") or (r == "sxm" and c == "Contusion"):
        return "img_" + c
    return c


def gold_logits(d):
    if (d / "gold.parquet").exists():
        g = pd.read_parquet(d / "gold.parquet")
    else:
        g = pd.read_parquet(d / "preds.parquet")
        g = g[g["set"] == "gold"]
    g = g[g.epoch == g.epoch.max()] if "epoch" in g else g
    g = g.set_index("StudyInstanceUID") if "StudyInstanceUID" in g else g
    num = g.select_dtypes("number").drop(columns=[c for c in ("epoch", "fold") if c in g], errors="ignore")
    return num.groupby(level=0).mean().apply(lambda s: pd.Series(lgt(s.to_numpy(float)), index=s.index) if s.min() >= 0 else s).reindex(G.index)


def twins(r):
    """[(plain run dir, PL run dir or None)] in the final's seed order."""
    kind, src, name = B.RECIPES[r]
    out = []
    for k, s in enumerate(B.VARIANTS[BASE]["seeds"][r], 1):
        p = (B.A14 / f"{s}_{name}" / "fold-1") if src == "a14" else (B.EXP / "runs" / f"{name}_all_s{s}" / "fold-1")
        q = (E14 / f"PL{200 + k}_{r}" / "fold-1") if src == "a14" else (E16 / f"{r}_pl2_s{200 + k}" / "fold-1")
        if (p / "done.txt").exists():                   # a planned plain seed still training is left out of both sides (count-matched)
            out.append((p, q if (q / "done.txt").exists() else None))
    return out


def col_auc(y, z):
    return auc(y, z) if len(np.unique(y)) == 2 else np.nan


def main():
    S = pd.read_csv(REPO / "work/alldata_final_2026-10-21/shares.csv")
    S = S[(S.form == "effc_off") & (S.w.abs() > 1e-4)]
    W = S.pivot_table(index="recipe", columns="col", values="w", aggfunc="sum").reindex(columns=FINDINGS).fillna(0.0)
    if not B.VARIANTS[BASE]["form"]["sxm"]:
        W = W.drop(index="sxm", errors="ignore")    # NOTE: shares are the effc_off (sxm ON) form's; with sxm off its Contusion share is approximate
    T = {r: [(gold_logits(p), gold_logits(q) if q else None) for p, q in twins(r)] for r in W.index}
    ntw = {r: sum(q is not None for _, q in T[r]) for r in W.index}
    # A: pairs
    rows = []
    for r in W.index:
        pairs = [(p, q) for p, q in T[r] if q is not None]
        if not pairs:
            continue
        for c in [c for c in FINDINGS if W.loc[r, c] != 0]:
            oc = out_col(r, c, W)
            d = [col_auc(G[c].to_numpy(int), q[oc]) - col_auc(G[c].to_numpy(int), p[oc]) for p, q in pairs]
            rows.append({"recipe": r, "col": c, "w": round(W.loc[r, c], 3), "pairs": len(pairs),
                         "plain": np.nanmean([col_auc(G[c].to_numpy(int), p[oc]) for p, _ in pairs]),
                         "pl": np.nanmean([col_auc(G[c].to_numpy(int), q[oc]) for _, q in pairs]), "d_mean": np.nanmean(d),
                         "wins": f"{sum(x > 0 for x in d)}/{len(d)}"})
        t12 = [c for c in FINDINGS if c in pairs[0][0] and pairs[0][0][c].notna().all() and pairs[0][1][c].notna().all()]
        if len(t12) == 12:
            rows.append({"recipe": r, "col": "ALL12", "pairs": len(pairs), "d_mean": np.nanmean(
                [col_auc(G[c].to_numpy(int), q[c]) - col_auc(G[c].to_numpy(int), p[c]) for p, q in pairs for c in t12])})
    A = pd.DataFrame(rows)

    # B / C: linearised package
    def pkg(choice):
        """choice: recipe -> 'plain' | 'swap' | 'half' | 'blend' -> study x class logits."""
        z = pd.DataFrame(0.0, index=G.index, columns=FINDINGS)
        for r in W.index:
            n = len(T[r])
            for c in [c for c in FINDINGS if W.loc[r, c] != 0]:
                oc = out_col(r, c, W)
                pl = [p[oc] for p, _ in T[r]]
                sw = [q[oc] if q is not None else p[oc] for p, q in T[r]]
                h = int(np.ceil(n / 2))
                hf = [p[oc] if i < h or q is None else q[oc] for i, (p, q) in enumerate(T[r])]
                pp = [p[oc] for p, q in T[r] if q is not None]   # PAIRED: only seeds with a twin, on both sides (A+B+C vs A'+B'+C')
                qq = [q[oc] for p, q in T[r] if q is not None]
                m = {"plain": sum(pl) / n, "swap": sum(sw) / n, "half": sum(hf) / n, "blend": 0.5 * sum(pl) / n + 0.5 * sum(sw) / n,
                     "pp": sum(pp) / len(pp) if pp else sum(pl) / n, "pq": sum(qq) / len(qq) if qq else sum(pl) / n,
                     "avg": 0.5 * (sum(pp) / len(pp) if pp else sum(pl) / n) + 0.5 * (sum(qq) / len(qq) if qq else sum(pl) / n)}[choice.get(r, "plain")]
                # AVG = mean of the paired plain seeds and their PL twins (count-matched: compare with PAIRED_plain / PAIRED_PL, never PLAIN)
                z[c] += W.loc[r, c] * m
        return z

    def macro(z, idx=None):
        idx = G.index if idx is None else idx
        return np.nanmean([col_auc(G.loc[idx, c].to_numpy(int), z.loc[idx, c]) for c in FINDINGS])

    V = {"PLAIN": pkg({}), "SWAP": pkg({r: "swap" for r in W.index}), "HALF": pkg({r: "half" for r in W.index}),
         "BLEND": pkg({r: "blend" for r in W.index}), "PAIRED_plain": pkg({r: "pp" for r in W.index}), "PAIRED_PL": pkg({r: "pq" for r in W.index}),
         "AVG": pkg({r: "avg" for r in W.index})}
    V = {"PLAIN_m": V.pop("PAIRED_plain"), "PL_m": V.pop("PAIRED_PL"), "AVG_m": V.pop("AVG"), **V}   # USER 00:0x: the apples-to-apples trio first
    P = pd.DataFrame({k: {c: col_auc(G[c].to_numpy(int), z[c]) for c in FINDINGS} for k, z in V.items()})
    P["n_pos"] = G[FINDINGS].sum().astype(int)
    P.loc["MACRO"] = [macro(V[k]) for k in V] + [np.nan]
    rng = np.random.default_rng(0)
    PAIRS = {"PL_m - PLAIN_m": ("PL_m", "PLAIN_m"), "AVG_m - PLAIN_m": ("AVG_m", "PLAIN_m"), "AVG_m - PL_m": ("AVG_m", "PL_m"),
             "(SWAP - PLAIN)": ("SWAP", "PLAIN"), "(HALF - PLAIN)": ("HALF", "PLAIN")}
    boots = {k: [] for k in PAIRS}
    for _ in range(NBOOT):
        idx = G.index[rng.integers(0, len(G), len(G))]
        for k, (a_, b_) in PAIRS.items():
            boots[k].append(macro(V[a_], idx) - macro(V[b_], idx))
    ci = {k: (macro(V[PAIRS[k][0]]) - macro(V[PAIRS[k][1]]), *np.percentile(v, [5, 95])) for k, v in boots.items()}
    base = macro(V["PLAIN"])
    M = []
    for r in W.index:
        if ntw[r]:
            z = pkg({r: "swap"})
            M.append({"recipe": r, "twins": f"{ntw[r]}/{len(T[r])}", "macro_d": macro(z) - base,
                      **{c: col_auc(G[c].to_numpy(int), z[c]) - P.loc[c, "PLAIN"] for c in FINDINGS if W.loc[r, c] != 0}})
    M = pd.DataFrame(M)
    GR = []                                              # sub-ensembles: the group's paired plain members vs their PL twins, on its classes
    for gname, rs, cols in GROUPS:
        have = [r for r in rs if r in W.index and ntw[r]]
        for c in cols:
            zp = [sum(p[out_col(r, c, W)] for p, q in T[r] if q is not None) / ntw[r] for r in have]
            zq = [sum(q[out_col(r, c, W)] for p, q in T[r] if q is not None) / ntw[r] for r in have]
            if have:
                y = G[c].to_numpy(int)
                GR.append({"group": gname, "recipes_with_twins": "+".join(have), "col": c, "plain": col_auc(y, sum(zp) / len(zp)),
                           "PL": col_auc(y, sum(zq) / len(zq))})
    GR = pd.DataFrame(GR)
    if len(GR):
        GR["d"] = GR.PL - GR.plain
    # saved for pl_cases.py (reliability / manual case review): variant logits, group sub-ensemble logits, plain seed sets for the null
    pd.concat({k: z for k, z in V.items()}, axis=1).to_parquet(D / "pl_gold_variants.parquet")
    gz, nulls = {}, {}
    for gname, rs, cols in GROUPS:
        have = [r for r in rs if r in W.index and ntw[r]]
        for c in cols:
            if not have:
                continue
            k = {r: ntw[r] for r in have}
            gz[(gname, c, "plain")] = sum(sum(p[out_col(r, c, W)] for p, q in T[r] if q is not None) / k[r] for r in have) / len(have)
            gz[(gname, c, "PL")] = sum(sum(q[out_col(r, c, W)] for p, q in T[r] if q is not None) / k[r] for r in have) / len(have)
            for j in range(1, 7):                       # seed-swap nulls: the same group from OTHER plain seed windows of the same size
                if all(len(T[r]) >= j + k[r] for r in have):
                    gz[(gname, c, f"null{j}")] = sum(sum(T[r][i][0][out_col(r, c, W)] for i in range(j, j + k[r])) / k[r] for r in have) / len(have)
    if gz:
        pd.DataFrame(gz).to_parquet(D / "pl_gold_groups.parquet")
    tot = sum(len(T[r]) for r in W.index)
    have = sum(ntw.values())
    stamp = datetime.datetime.now().strftime("%m-%d %H:%M")
    txt = [f"PL gold-58 monitor {stamp} (OAI: no; 58 studies -> trend only). PL twins {have} / {tot}: " + ", ".join(f"{r} {ntw[r]}/{len(T[r])}" for r in W.index if ntw[r]),
           "\nA PAIRS (PL twin vs its plain seed; shipped output; d = PL - plain AUC)", A.round(4).to_string(index=False) if len(A) else "(no twins yet)",
           "\nB PACKAGE (linearised, effc_off shares; AUC per class). APPLES TO APPLES: PLAIN_m / PL_m / AVG_m use the SAME seed indices",
           "  (each recipe's twinned plain seeds / their PL twins / both averaged = more models, by design); recipes without twins = plain in all.",
           "  When the counterpart is complete: PLAIN_m = the final, PL_m = the full PL package, AVG_m = their average. Secondary (vs PLAIN = all",
           "  plain seeds, same total count): SWAP / HALF / BLEND.", P[["PLAIN_m", "PL_m", "AVG_m", "PLAIN", "SWAP", "HALF", "BLEND", "n_pos"]].round(4).to_string(),
           "\n  macro deltas, observed [90% bootstrap]: " + " | ".join(f"{k} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for k, (m, lo, hi) in ci.items()),
           "\nC MARGINALS (swap one recipe to its twins; deltas vs PLAIN)", M.round(4).to_string(index=False) if len(M) else "(none)",
           "\nE GROUPS (sub-ensemble of a class's members: paired plain seeds vs their PL twins, equal size)", GR.round(4).to_string(index=False) if len(GR) else "(none)"]
    out = "\n".join(txt) + "\n"
    (D / "pl_gold.txt").write_text(out)
    tr = D / "pl_gold_trend.csv"
    pd.DataFrame([{"time": stamp, "twins": have, "of": tot, "PLAIN_m": macro(V["PLAIN_m"]), "PL_m": macro(V["PL_m"]), "AVG_m": macro(V["AVG_m"])}]).to_csv(
        tr, mode="a", header=not tr.exists(), index=False)
    print(out)


if __name__ == "__main__":
    main()
