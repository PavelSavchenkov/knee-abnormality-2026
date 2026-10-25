"""Fracture / contusion evaluation panel (pooled out-of-fold predictions + gold-58).

Cell sets per finding (label 1 vs 0; studies outside a set are ignored):
- text:  v1 meets vs denied / structure_normal (the common/metrics text metric).
- tn:    v1 meets vs every trusted negative status (common/trusted_rules.csv: fracture adds general_normal, contusion adds
         excluded = degenerative / fracture oedema).
- hard:  contusion meets vs excluded oedema only (traumatic vs degenerative oedema); fracture meets vs contusion-meets studies
         with a fracture denied / normal (oedema without a fracture).
- plain: fracture only: v1 meets with a firm v2 fracture / avulsion fact, not chronic, vs trusted negatives.
- v3:    v3 qualifying / likely_qualifying vs negative / covered_normal (a definition-aware text read).
- gold:  gold-58, test-like (mean of the fold models' logits).

    from fc_eval import load_run, panel, paired
    oof, gold = load_run(RUN_DIR, epoch=19)        # RUN_DIR/fold*/oof.parquet, gold.parquet (columns FINDINGS + epoch)
    panel(oof, gold) -> dict metric -> AUC;  paired(a, b, metric) -> (diff, lo, hi)
CLI: python scripts/fc_eval.py RUN_DIR [RUN_DIR ...] [--epoch 19]"""
import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from common.labels import gold_labels, status_table, text_labels  # noqa: E402

FC = ["Fracture", "Contusion"]
KEY = {"Fracture": "fracture", "Contusion": "contusion"}
TRUSTED_NEG = {"fracture": ("denied", "structure_normal", "general_normal"), "contusion": ("denied", "structure_normal", "excluded")}
FIRM = ("present", "present_indirect")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@functools.cache
def cells():
    """Long table StudyInstanceUID, finding, set, label."""
    st = status_table("v1")
    v1, v2, v3 = text_labels("v1"), text_labels("v2"), text_labels("v3")
    rows = []
    for sid in st.index:
        fs, cs = st.at[sid, "fracture"], st.at[sid, "contusion"]
        ccause = v1[sid]["findings"]["contusion"].get("cause")
        for f, s in (("Fracture", fs), ("Contusion", cs)):
            k = KEY[f]
            if s == "meets" or s in ("denied", "structure_normal"):
                rows.append((sid, f, "text", int(s == "meets")))
            neg = s in TRUSTED_NEG[k] and not (k == "contusion" and s == "excluded" and ccause not in ("degeneration", "fracture"))
            if s == "meets" or neg:
                rows.append((sid, f, "tn", int(s == "meets")))
            ev = v3[sid]["findings"][k]["evidence"]
            if ev in ("qualifying", "likely_qualifying", "negative", "covered_normal"):
                rows.append((sid, f, "v3", int(ev in ("qualifying", "likely_qualifying"))))
        if cs == "meets" or (cs == "excluded" and ccause in ("degeneration", "fracture")):
            rows.append((sid, "Contusion", "hard", int(cs == "meets")))
        if fs == "meets" or (cs == "meets" and fs in ("denied", "structure_normal")):
            rows.append((sid, "Fracture", "hard", int(fs == "meets")))
        plain = fs == "meets" and any(x["finding"] in ("fracture", "avulsion_fracture") and x["assertion"] in FIRM and x.get("acuity") != "chronic"
                                      for x in v2[sid]["facts"])
        if plain or fs in TRUSTED_NEG["fracture"]:
            rows.append((sid, "Fracture", "plain", int(plain)))
    return pd.DataFrame(rows, columns=["StudyInstanceUID", "finding", "set", "label"])


def load_run(run, epoch=19, folds=None):
    """Pooled OOF (rows of `epoch`, column fold) and test-like gold (mean logit over the fold models, back to probability)."""
    run = Path(run)
    oofs, golds = [], []
    for d in sorted(run.glob("fold*")):
        f = int(d.name[4:])
        if folds is not None and f not in folds or not (d / "oof.parquet").exists():
            continue
        o = pd.read_parquet(d / "oof.parquet")
        if "epoch" in o:
            ep = epoch if epoch in set(o.epoch) else o.epoch.max()
            if ep != epoch:
                print(f"WARNING {d}: epoch {epoch} missing, using {ep} (unfinished run?)", file=sys.stderr)
            o = o[o.epoch == ep].drop(columns="epoch")
        oofs.append(o.assign(fold=f))
        if (d / "gold.parquet").exists():
            g = pd.read_parquet(d / "gold.parquet")
            if "epoch" in g:
                ep = epoch if epoch in set(g.epoch) else g.epoch.max()
                g = g[g.epoch == ep].drop(columns="epoch")
            golds.append(g[[c for c in g.columns if c != "fold"]])
    oof = pd.concat(oofs)
    gold = None
    if golds:
        lg = sum(pd.DataFrame(logit(g.to_numpy(float)), index=g.index, columns=g.columns) for g in golds) / len(golds)
        gold = 1 / (1 + np.exp(-lg))
    return oof, gold


def _auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def table(oof, gold, metric, finding):
    """(labels, scores) for one metric x finding."""
    if metric == "gold":
        g = gold_labels()
        idx = gold.index.intersection(g.index)
        return g.loc[idx, finding].to_numpy(), gold.loc[idx, finding].to_numpy()
    c = cells()
    c = c[(c.finding == finding) & (c.set == metric)]
    c = c[c.StudyInstanceUID.isin(oof.index)]
    return c.label.to_numpy(), oof.loc[c.StudyInstanceUID, finding].to_numpy()


METRICS = ["text", "tn", "hard", "plain", "v3", "gold"]


def panel(oof, gold=None, findings=FC):
    out = {}
    for f in findings:
        for m in METRICS:
            if m == "gold" and gold is None or (m == "plain" and f != "Fracture"):
                continue
            y, s = table(oof, gold, m, f)
            out[f"{f[:4]}_{m}"] = _auc(y, s)
    return out


def counts(oof, gold=None):
    out = {}
    for f in FC:
        for m in METRICS:
            if m == "gold" and gold is None or (m == "plain" and f != "Fracture"):
                continue
            y, _ = table(oof, gold, m, f)
            out[f"{f[:4]}_{m}"] = f"{int(y.sum())}/{len(y)}"
    return out


def paired(a, b, metric, finding, a_gold=None, b_gold=None, n_boot=2000, seed=0):
    """b - a AUC on the common studies of one cell set, paired study bootstrap: (diff, lo, hi)."""
    if metric == "gold":
        g = gold_labels()
        idx = a_gold.index.intersection(b_gold.index).intersection(g.index)
        y, sa, sb = g.loc[idx, finding].to_numpy(), a_gold.loc[idx, finding].to_numpy(), b_gold.loc[idx, finding].to_numpy()
    else:
        c = cells()
        c = c[(c.finding == finding) & (c.set == metric)]
        c = c[c.StudyInstanceUID.isin(a.index.intersection(b.index))]
        y, sa, sb = c.label.to_numpy(), a.loc[c.StudyInstanceUID, finding].to_numpy(), b.loc[c.StudyInstanceUID, finding].to_numpy()
    d0 = _auc(y, sb) - _auc(y, sa)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ds = []
    for _ in range(n_boot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ds.append(_auc(y[i], sb[i]) - _auc(y[i], sa[i]))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return d0, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--epoch", type=int, default=19)
    ap.add_argument("--per_fold", action="store_true")
    a = ap.parse_args()
    rows = {}
    for r in a.runs:
        oof, gold = load_run(r, a.epoch)
        rows[Path(r).name] = panel(oof, gold)
        if a.per_fold:
            for f in sorted(oof.fold.unique()):
                rows[f"{Path(r).name}/f{f}"] = panel(oof[oof.fold == f].drop(columns="fold"))
    print(pd.DataFrame(rows).T.round(4).to_string())
    print(json.dumps(counts(oof, gold)))


if __name__ == "__main__":
    main()
