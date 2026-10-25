"""PL targets for the all-data students, tier A (oct14 trainers, NO code change: they read external label sets). OAI: no. CPU.
  sub_b0 (main):  labels=ext:pl_ct_sub_b0_<tv>   -> common.labels.external -> data/text_labels/external/<name>.csv
  nfnet (basin):  target=pl_ctns_nfnet_<tv>      (same file kind; basin weights = notna, so every cell is filled)
Target per cell = (1 - a) y_plain + a p_T (oct12 'ct' blend, the K6r LB students):
  y_plain = the recipe's OWN plain targets from its own oct14 code (main/labels.make_targets, basin/data.build_targets) and resolved
            all-data config (runs/A1_<recipe>/fold-1/config.json), so only the blend differs from a plain seed;
  p_T     = teacher OOF logits quantile-mapped onto the student recipe's pooled OOF marginal (sub_b0 for both tier-A students, as
            oct12 pl_lib.to_prob; their v1 targets share one scale), per fold group
            (f0-2 / f3-4: the groups can come from different forms);
  a       = by v1 report status: meets / denied / structure_normal 0.3, other stated statuses 0.7, not_mentioned 1.0;
            nfnet 'ctns': a = 0 on Synovitis (its Synovitis stays the plain v1 + syn_eff target).
Gold-58 rows (never trained on at fold -1; 'val') get y_plain. Writes the CSV read-only + an md5 line in manifest.txt + a report.
  conda run -n rsna python work/pl_alldata_2026-10-21/targets.py --recipe sub_b0 --teacher v0 [--force]"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
sys.path.insert(0, str(REPO))
from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels, status_table  # noqa: E402
from common.metrics import auc  # noqa: E402
from common.paths import FINDINGS, KEYS, TEXT_LABELS  # noqa: E402

A14 = REPO / "experiments/v2/oct14_all_data"
EXPLICIT, SILENT = ("meets", "denied", "structure_normal"), "not_mentioned"
NAME = {"sub_b0": "pl_ct_sub_b0", "nfnet": "pl_ctns_nfnet"}
NO_TEACHER = {"sub_b0": (), "nfnet": ("Synovitis",)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plain_targets(recipe, studies):
    """(Y, W) DataFrames (studies x FINDINGS) exactly as the recipe's plain all-data run builds them."""
    cfg = json.load(open(A14 / f"runs/A1_{recipe}/fold-1/config.json"))
    if recipe == "sub_b0":
        lab = load(A14 / "main/labels.py", "o14_main_labels")
        y, w = lab.make_targets(list(studies), cfg)
        return pd.DataFrame(y, studies, FINDINGS, dtype=float), pd.DataFrame(w, studies, FINDINGS, dtype=float)
    sys.path.insert(0, str(A14 / "basin"))
    dat = load(A14 / "basin/data.py", "o14_basin_data")
    assert cfg["target"] == "v1" and not cfg["img_w"] and not cfg["unset_mask"], "plain nfnet = v1 targets, no image head / unset mask"
    Y, W = dat.build_targets(cfg["target"], nm=None if cfg["nm"] == "mask" else float(cfg["nm"]), nm_weight=cfg["nm_weight"],
                             hard=cfg["hard"], reads=[r for r in cfg["reads"].split(",") if r], reads_weight=cfg["reads_weight"],
                             syn_eff=bool(cfg["syn_eff"]), reads_scale=cfg["reads_scale"], unset_mask=[], unset_mode=cfg["unset_mode"])
    return Y.loc[studies, FINDINGS].astype(float), W.loc[studies, FINDINGS].astype(float)


def ref_oof(run_dir, folds=range(5)):
    """A run family's pooled OOF probabilities at its last epoch (the student recipe's own output scale)."""
    out = []
    for k in folds:
        d = pd.read_parquet(run_dir / f"fold{k}" / "oof.parquet")
        out.append(d[d.epoch == d.epoch.max()] if "epoch" in d else d)
    return pd.concat(out)


def teacher_prob(t, ref=None, cols=FINDINGS):
    """Teacher logits -> probabilities on the student recipe's pooled OOF marginal (default sub_b0, epoch 19, folds 0-4; oct12
    pl_lib.to_prob), per fold group and column."""
    ref = ref_oof(REPO / "experiments/v0/oct7_baseline/runs/sub_b0") if ref is None else ref
    q = np.linspace(0, 1, 1001)
    p = pd.DataFrame(index=t.index, columns=cols, dtype=float)
    for grp in ([0, 1, 2], [3, 4]):
        m = t.fold.isin(grp)
        for c in cols:
            p.loc[m, c] = np.interp(t.loc[m, c], np.quantile(t.loc[m, c], q), np.quantile(ref[c], q))
    return p


def alpha(recipe, studies):
    st = status_table("v1").loc[studies, KEYS]
    st.columns = FINDINGS
    a = pd.DataFrame(0.7, index=studies, columns=FINDINGS)
    a[st.isin(EXPLICIT)] = 0.3
    a[st == SILENT] = 1.0
    assert st.notna().all().all()
    for c in NO_TEACHER[recipe]:
        a[c] = 0.0
    return a, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=NAME)
    ap.add_argument("--teacher", required=True, help="teacher version: reads teacher_<tv>.parquet")
    ap.add_argument("--force", action="store_true", help="replace an existing file (plumbing versions only)")
    a = ap.parse_args()
    name = f"{NAME[a.recipe]}_{a.teacher}"
    out = TEXT_LABELS / "external" / f"{name}.csv"
    t = pd.read_parquet(D / f"teacher_{a.teacher}.parquet")
    folds = get_folds(5, 0, include_gold=False).fold
    gold = gold_labels().index
    assert set(t.index) == set(folds.index) and not set(t.index) & set(gold), "teacher must cover exactly the non-gold studies"
    assert (t.fold == folds.reindex(t.index)).all()
    tr = folds.index.tolist()
    studies = tr + [s for s in gold if s not in set(tr)]
    Y, W = plain_targets(a.recipe, studies)
    need = [c for c in FINDINGS if c not in NO_TEACHER[a.recipe]]
    assert t[need].notna().all().all(), f"teacher columns incomplete for {a.recipe}: {t[need].columns[t[need].isna().any()].tolist()}"
    p = teacher_prob(t, cols=need).loc[tr].reindex(columns=FINDINGS).fillna(0.0)   # columns without a teacher have a = 0
    al, st = alpha(a.recipe, tr)
    y = Y.copy()
    y.loc[tr] = (1 - al) * Y.loc[tr] + al * p
    assert y.notna().all().all() and ((y >= 0) & (y <= 1)).all().all()
    rep = []
    for c in FINDINGS:
        s, yp, yy, tc = st[c], Y.loc[tr, c], y.loc[tr, c], t.loc[tr, c]
        ex = s.isin(("meets", "denied", "structure_normal"))
        lab = (s[ex] == "meets").astype(int)
        rep.append({"col": c, "a_mean": al[c].mean(), "silent": (s == SILENT).mean(), "y_plain": yp.mean(), "y_pl": yy.mean(),
                    "mean|d|": (yy - yp).abs().mean(), "teacher_auc_explicit": auc(lab, tc[ex]) if lab.nunique() == 2 else np.nan,
                    "spearman_pl_plain": yy.corr(yp, method="spearman")})
    rep = pd.DataFrame(rep).set_index("col")
    if out.exists():
        assert a.force and a.teacher.startswith("v0"), f"{out} exists (only plumbing versions may be replaced)"
        os.chmod(out, 0o644)
    y.reset_index(names="StudyInstanceUID").to_csv(out, index=False, float_format="%.7f")
    os.chmod(out, 0o444)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    with open(D / "manifest.txt", "a") as f:
        f.write(f"{md5}  {out.relative_to(REPO)}  recipe={a.recipe} teacher={a.teacher} rows={len(y)}\n")
    txt = (f"{name}: {len(tr)} train + {len(y) - len(tr)} gold rows (gold = plain), md5 {md5}\n"
           f"teacher {a.teacher}; W plain mean {W.loc[tr].to_numpy().mean():.4f} (unchanged by the CSV)\n" + rep.round(4).to_string())
    (D / f"targets_{name}.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
