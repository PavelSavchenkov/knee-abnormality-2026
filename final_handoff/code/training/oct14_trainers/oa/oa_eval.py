"""OA evaluation panel: how well do scores rank the three OA findings, and do they know WHICH compartment has the OA?

Scores: OOF predictions (study x 12 FINDINGS, probabilities or logits; AUC uses order only) with a `fold` column, and gold-58
predictions (58 x 12, mean logit over the fold models, like the test).

Per OA finding F (v1 text statuses; pos = meets, neg = denied / structure_normal):
  text     AUC pos vs neg
  trusted / lookat   common.metrics cells (Medial OA, PF OA only; Lateral OA has none)
  strat    mean of the AUCs inside the strata of the other compartments' status (LOA: MOA pos / MOA neg; MOA: LOA pos / neg;
           PF: any tibiofemoral pos / both neg). Removes the "degenerative knee" shortcut from the comparison.
  comp     compartment test: F's score on "F pos, other neg" vs "F neg, other pos" (PF: mean of vs MOA and vs LOA).
           0.5 = the score cannot tell the compartments apart.
  thr      meets vs below (the > 50% depth threshold)
  gold     gold-58 AUC (veto only: 11 / 15 / 21 positives)
Summary columns: text3 / strat3 / comp3 / thr3 / gold3 = means over the three findings.

CLI: python scripts/oa_eval.py RUN [RUN ...]  (runs in ../runs or ../refs; epoch = last, or RUN@EPOCH)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from common.labels import eval_cells, gold_labels, status_table  # noqa: E402
from common.paths import FINDINGS, KEYS  # noqa: E402

EXP = HERE.parent
OA = ["Medial OA", "Lateral OA", "PF OA"]
KEY = dict(zip(FINDINGS, KEYS))
POS, NEG = ("meets",), ("denied", "structure_normal")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def to_logit(df):
    """Probabilities -> logits; frames that already hold logits (values outside [0, 1]) are returned as they are."""
    v = df[FINDINGS].to_numpy(float)
    return df[FINDINGS] if (v.min() < 0 or v.max() > 1) else pd.DataFrame(logit(v), index=df.index, columns=FINDINGS)


def run_dir(name):
    for base in (EXP / "runs", EXP / "refs"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(name)


def load_run(name, epoch=None, folds=None):
    """OOF (study x 12 logits + fold) and gold (58 x 12 mean logit over the fold models) of a main-pipeline run at `epoch`
    (default: each fold's last evaluated epoch)."""
    d = run_dir(name)
    oof, gold = [], []
    for fd in sorted(d.glob("fold*")):
        f = int(fd.name[4:])
        if folds is not None and f not in folds or not (fd / "oof.parquet").exists():
            continue
        o, g = pd.read_parquet(fd / "oof.parquet"), pd.read_parquet(fd / "gold.parquet")
        e = o.epoch.max() if epoch is None else epoch
        if e not in set(o.epoch):
            continue
        oof.append(to_logit(o[o.epoch == e]).assign(fold=f))
        gold.append(to_logit(g[g.epoch == e]))
    assert oof, f"no folds for {name} at epoch {epoch}"
    return pd.concat(oof), pd.concat(gold).groupby(level=0).mean()


def load_basin(path, folds=None):
    """Basin-code preds parquet (set oof / gold, fold column) -> same format as load_run."""
    d = pd.read_parquet(path)
    if folds is not None:
        d = d[d.fold.isin(folds)]
    o = d[d.set == "oof"]
    g = d[d.set == "gold"]
    return to_logit(o).assign(fold=o.fold), to_logit(g).groupby(level=0).mean()


def _auc(y, s):
    y = np.asarray(y, bool)
    return roc_auc_score(y, s) if 3 <= y.sum() <= len(y) - 3 else np.nan


def panel(oof, gold=None):
    """dict metric -> value for one score set."""
    st = status_table("v1").loc[oof.index]
    pos = {f: st[KEY[f]].isin(POS).to_numpy() for f in OA}
    neg = {f: st[KEY[f]].isin(NEG).to_numpy() for f in OA}
    below = {f: (st[KEY[f]] == "below").to_numpy() for f in OA}
    s = {f: oof[f].to_numpy(float) for f in OA}
    out = {}
    cells = eval_cells()
    cells = cells[cells.StudyInstanceUID.isin(set(oof.index))]
    for f in OA:
        m = pos[f] | neg[f]
        out[f"text|{f}"] = _auc(pos[f][m], s[f][m])
        for kind, c in (("trusted", cells[cells.verdict == "trusted"]), ("lookat", cells)):
            c = c[c.finding == f]
            if c.label.nunique() == 2:
                out[f"{kind}|{f}"] = _auc(c.label.to_numpy(), oof.loc[c.StudyInstanceUID, f].to_numpy())
        if f == "PF OA":
            other_pos = pos["Medial OA"] | pos["Lateral OA"]
            other_neg = neg["Medial OA"] & neg["Lateral OA"]
            comps = [_auc(pos[f][(pos[f] & neg[g]) | (neg[f] & pos[g])], s[f][(pos[f] & neg[g]) | (neg[f] & pos[g])])
                     for g in ("Medial OA", "Lateral OA")]
        else:
            g = "Lateral OA" if f == "Medial OA" else "Medial OA"
            other_pos, other_neg = pos[g], neg[g]
            mm = (pos[f] & neg[g]) | (neg[f] & pos[g])
            comps = [_auc(pos[f][mm], s[f][mm])]
        strata = [_auc(pos[f][m & k], s[f][m & k]) for k in (other_pos, other_neg)]
        out[f"strat|{f}"] = float(np.nanmean(strata))
        out[f"comp|{f}"] = float(np.nanmean(comps))
        mt = pos[f] | below[f]
        out[f"thr|{f}"] = _auc(pos[f][mt], s[f][mt])
    if gold is not None:
        g = gold_labels()
        gi = gold.index.intersection(g.index)
        for f in OA:
            out[f"gold|{f}"] = _auc(g.loc[gi, f].to_numpy(), gold.loc[gi, f].to_numpy())
    for k in ("text", "strat", "comp", "thr", "gold"):
        v = [out.get(f"{k}|{f}", np.nan) for f in OA]
        out[f"{k}3"] = float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan
    return out


SUMMARY = ["text3", "strat3", "comp3", "thr3", "gold3"]
PER = [f"{k}|{f}" for f in OA for k in ("text", "trusted", "lookat", "strat", "comp", "thr", "gold")]


def table(rows):
    """rows: dict name -> panel dict -> DataFrame (summary columns first)."""
    t = pd.DataFrame(rows).T
    return t[[c for c in SUMMARY + PER if c in t.columns]]


def short(t):
    """Compact column names for printing."""
    ab = {"Medial OA": "M", "Lateral OA": "L", "PF OA": "P"}
    return t.rename(columns=lambda c: c if "|" not in c else f"{c.split('|')[0][:5]}{ab[c.split('|')[1]]}")


def boot_diff(oof_a, oof_b, metric, n=1000, seed=0):
    """Paired study bootstrap of panel(b)[metric] - panel(a)[metric] on common studies: (mean diff, 2.5%, 97.5%)."""
    idx = oof_a.index.intersection(oof_b.index)
    a, b = oof_a.loc[idx], oof_b.loc[idx]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(idx), len(idx))
        ra, rb = a.iloc[i], b.iloc[i]
        d.append(panel(rb)[metric] - panel(ra)[metric])
    d = np.array(d)
    return float(panel(b)[metric] - panel(a)[metric]), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    rows = {}
    for arg in sys.argv[1:]:
        name, _, ep = arg.partition("@")
        o, g = load_run(name, int(ep) if ep else None)
        rows[f"{arg} ({o.fold.nunique()}f)"] = panel(o, g)
    print(short(table(rows)).round(4).to_string())
