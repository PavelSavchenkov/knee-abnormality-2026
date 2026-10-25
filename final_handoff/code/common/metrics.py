"""Standard scoring of out-of-fold predictions. Every experiment scores with these functions so results are comparable.

pred: DataFrame indexed by StudyInstanceUID with the 12 FINDINGS columns (probabilities or logits; AUC uses order only).

evaluate(pred) -> dict
    trusted_mean   mean AUC over TRUSTED_FINDINGS on hand-checked trusted cells (the primary metric)
    lookat_mean    mean AUC over LOOKAT_FINDINGS on trusted + look-at cells (tie-breaker)
    text_mean      mean AUC over all 12 findings, v1 'meets' vs 'denied'/'structure_normal' (covers findings without trusted positives)
    gold_mean      macro AUC on the official gold-58 labels (only studies present in pred)
    gold_calibration  ECE / Brier / mean prediction vs mean label over all 12 x gold cells
    per-finding dicts for each of the above, calibration (ECE, Brier, mean prediction for trusted pos/neg), score quantiles
compare(pred_a, pred_b, metric) -> paired bootstrap over studies: mean difference b - a, 95% CI, share of resamples with b <= a
by_site(pred) -> trusted_mean per site (sites with enough studies)"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import RULES, eval_cells, gold_labels, status_table
from common.meta import study_table
from common.paths import FINDINGS, KEYS

MIN_CLASS = 5                                          # a finding is scored only with at least this many positives and negatives
_r = RULES[RULES.verdict == "trusted"]
TRUSTED_FINDINGS = [f for f in FINDINGS if set(_r[_r.finding == f].label) == {0, 1}]
_r = RULES
LOOKAT_FINDINGS = [f for f in FINDINGS if set(_r[_r.finding == f].label) == {0, 1}]
TEXT_POS, TEXT_NEG = ("meets",), ("denied", "structure_normal")


def _auc_weighted(y, s, w):
    """AUC with sample weights (bootstrap counts); NaN when a class is missing."""
    keep = w > 0
    y, s, w = y[keep], s[keep], w[keep]
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, s, sample_weight=w))


def _auc_boot(y, s, W):
    """Weighted AUC (ties count half) for every row of W (n_boot x n); equals roc_auc_score(y, s, sample_weight=w) per row."""
    o = np.argsort(s, kind="mergesort")
    y, s, W = y[o], s[o], W[:, o]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    gn = np.add.reduceat(W * (y == 0), start, axis=1)
    gp = np.add.reduceat(W * (y == 1), start, axis=1)
    below = np.cumsum(gn, axis=1) - gn
    den = gp.sum(1) * gn.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (gp * (below + 0.5 * gn)).sum(1) / den, np.nan)


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    if y.sum() < MIN_CLASS or (1 - y).sum() < MIN_CLASS:
        return np.nan
    return _auc_weighted(y, s, np.ones(len(y)))


def _cells(kind):
    """Long table StudyInstanceUID, finding, label for 'trusted', 'lookat' (trusted + look at) or 'text'."""
    if kind == "trusted":
        c = eval_cells()
        return c[c.verdict == "trusted"][["StudyInstanceUID", "finding", "label"]]
    if kind == "lookat":
        return eval_cells()[["StudyInstanceUID", "finding", "label"]]
    st = status_table("v1")
    long = st.rename(columns=dict(zip(KEYS, FINDINGS))).stack().rename("status").reset_index()
    long.columns = ["StudyInstanceUID", "finding", "status"]
    long = long[long.status.isin(TEXT_POS + TEXT_NEG)]
    long["label"] = long.status.isin(TEXT_POS).astype(int)
    return long[["StudyInstanceUID", "finding", "label"]]


def _join(pred, kind):
    long = pred[FINDINGS].stack().rename("score").reset_index()
    long.columns = ["StudyInstanceUID", "finding", "score"]
    return _cells(kind).merge(long, on=["StudyInstanceUID", "finding"])


def _per_finding(j, findings):
    return {f: auc(d.label, d.score) for f, d in j.groupby("finding") if f in findings}


def _mean(d, findings):
    v = [d.get(f, np.nan) for f in findings]
    return float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan


def ece_brier(p, y):
    """ECE (10 equal-width bins) and Brier score for probabilities p and 0/1 labels y."""
    p, y = np.clip(np.asarray(p, float), 0, 1), np.asarray(y, float)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return float(ece), float(((p - y) ** 2).mean())


def calibration(j):
    """ECE and Brier on trusted cells, assuming scores are probabilities."""
    ece, brier = ece_brier(j.score, j.label)
    return {"ece": ece, "brier": brier,
            "mean_pos": j[j.label == 1].groupby("finding").score.mean().round(4).to_dict(),
            "mean_neg": j[j.label == 0].groupby("finding").score.mean().round(4).to_dict()}


def evaluate(pred):
    pred = pred[FINDINGS].astype(float)
    out = {"n_studies": int(len(pred))}
    for kind, findings in [("trusted", TRUSTED_FINDINGS), ("lookat", LOOKAT_FINDINGS), ("text", FINDINGS)]:
        j = _join(pred, kind)
        d = _per_finding(j, findings)
        out[f"{kind}_auc"] = {k: round(v, 4) for k, v in d.items()}
        out[f"{kind}_mean"] = _mean(d, findings)
        if kind == "trusted":
            out["calibration"] = calibration(j)
            out["quantiles"] = {f"{f}|{lab}": np.round(np.quantile(d_.score, [0.1, 0.5, 0.9]), 4).tolist()
                                for (f, lab), d_ in j.groupby(["finding", "label"])}
    g = gold_labels()
    gi = pred.index.intersection(g.index)
    if len(gi):
        d = {f: auc(g.loc[gi, f], pred.loc[gi, f]) for f in FINDINGS}
        out["gold_auc"] = {k: round(v, 4) for k, v in d.items()}
        out["gold_mean"] = _mean(d, FINDINGS)
        out["gold_n"] = int(len(gi))
        ece, brier = ece_brier(pred.loc[gi, FINDINGS].to_numpy().ravel(), g.loc[gi, FINDINGS].to_numpy().ravel())
        out["gold_calibration"] = {"ece": ece, "brier": brier, "mean_pred": float(pred.loc[gi, FINDINGS].to_numpy().mean()),
                                   "mean_label": float(g.loc[gi, FINDINGS].to_numpy().mean())}
    return out


def _metric_long(pred, metric):
    if metric == "gold":
        g = gold_labels()
        gi = pred.index.intersection(g.index)
        lab = g.loc[gi].stack().rename("label").reset_index()
        lab.columns = ["StudyInstanceUID", "finding", "label"]
        long = pred.loc[gi, FINDINGS].stack().rename("score").reset_index()
        long.columns = ["StudyInstanceUID", "finding", "score"]
        return lab.merge(long, on=["StudyInstanceUID", "finding"]), FINDINGS
    findings = {"trusted": TRUSTED_FINDINGS, "lookat": LOOKAT_FINDINGS, "text": FINDINGS}[metric]
    return _join(pred, metric), findings


def compare(pred_a, pred_b, metric="trusted", n_boot=2000, seed=0):
    """Paired bootstrap over studies (common to both). Returns dict: a, b, diff, ci, p_le0, per-finding diffs."""
    idx = pred_a.index.intersection(pred_b.index)
    ja, findings = _metric_long(pred_a.loc[idx], metric)
    jb, _ = _metric_long(pred_b.loc[idx], metric)
    j = ja.merge(jb[["StudyInstanceUID", "finding", "score"]], on=["StudyInstanceUID", "finding"], suffixes=("_a", "_b"))
    sid_pos = {s: i for i, s in enumerate(idx)}
    groups = [(f, d) for f, d in j.groupby("finding") if f in findings and d.label.sum() >= MIN_CLASS and (1 - d.label).sum() >= MIN_CLASS]
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot).astype(float)
    diffs, per = np.zeros(n_boot), {}
    base_a, base_b = [], []
    for f, d in groups:
        si = d.StudyInstanceUID.map(sid_pos).to_numpy()
        y, sa, sb = d.label.to_numpy(), d.score_a.to_numpy(), d.score_b.to_numpy()
        a0, b0 = _auc_weighted(y, sa, np.ones(len(y))), _auc_weighted(y, sb, np.ones(len(y)))
        base_a.append(a0), base_b.append(b0)
        fd = _auc_boot(y, sb, w[:, si]) - _auc_boot(y, sa, w[:, si])
        per[f] = {"a": round(a0, 4), "b": round(b0, 4), "diff": round(b0 - a0, 4), "ci": np.round(np.nanpercentile(fd, [2.5, 97.5]), 4).tolist()}
        diffs += np.nan_to_num(fd) / len(groups)
    return {"metric": metric, "a": float(np.mean(base_a)), "b": float(np.mean(base_b)), "diff": float(np.mean(base_b) - np.mean(base_a)),
            "ci": np.round(np.percentile(diffs, [2.5, 97.5]), 4).tolist(), "p_le0": float((diffs <= 0).mean()), "n_studies": int(len(idx)),
            "per_finding": per}


def by_site(pred, min_studies=60):
    st = study_table().reindex(pred.index)
    out = {}
    for site, idx in st.groupby("site").groups.items():
        if len(idx) >= min_studies:
            out[site] = round(evaluate(pred.loc[idx])["trusted_mean"], 4)
    return out
