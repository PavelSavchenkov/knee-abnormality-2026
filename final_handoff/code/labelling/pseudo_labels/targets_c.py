"""PL targets for tier C (oct16 trainers, their existing tgt_csv inputs; no code change). OAI: no. CPU. Master 19:07: prepare + prove;
earlier PL evidence on these columns is negative (P2 / P5 MCL, P1c / P4 / P7 / P9 LOA); shipping is evidence-gated (the user decides).
Only the column each recipe SHIPS gets PL (the other heads keep their plain targets):
  z1_acl ACL | w5b_mcl MCL | b3g ACL + MCL (syn: tgt_csv + tgt_findings)  sd1b MCL (mainp: tgt_csv columns)
  bsd_nf MCL (basin: tgt_csv + tgt_findings)  oasep / oasep_nf Lateral OA (oa: tgt_csv + tgt_cols)
Target = the tier-A ct blend (targets.alpha), y_plain = the recipe's own text targets from its own code + resolved all-data config
(runs/<r>_all_s101/fold-1), p_T = the teacher on THIS recipe's pooled OOF marginal of that column. Gold-58 rows = y_plain.
  conda run -n rsna python work/pl_alldata_2026-10-21/targets_c.py --teacher v1 [--recipes z1_acl,oasep] [--force]"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(REPO), str(D)]
import targets as T  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.metrics import auc  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

SPEC = {"z1_acl": ("syn", ["ACL"]), "w5b_mcl": ("syn", ["MCL"]), "b3g": ("syn", ["ACL", "MCL"]), "sd1b": ("mainp", ["MCL"]),
        "bsd_nf": ("basin", ["MCL"]), "oasep": ("oa", ["Lateral OA"]), "oasep_nf": ("oa", ["Lateral OA"])}
OUT = EXP / "data/pl"


def cfg_of(r):
    return json.load(open(EXP / f"runs/{r}_all_s101/fold-1/config.json"))


def plain(r, studies):
    """The recipe's text targets (studies x FINDINGS) before any tgt_csv, from its own target code."""
    kind, cfg = SPEC[r][0], cfg_of(r)
    assert not cfg.get("tgt_csv"), f"{r} already has a tgt_csv"
    if kind == "basin":
        sys.path.insert(0, str(EXP / "basin"))
        dat = T.load(EXP / "basin/data.py", "o16_basin_data")
        assert not cfg["unset_mask"] and not cfg["reads"]
        Y, _ = dat.build_targets(cfg["target"], nm=None if cfg["nm"] == "mask" else float(cfg["nm"]), nm_weight=cfg["nm_weight"],
                                 hard=cfg["hard"], reads=[], reads_weight=cfg["reads_weight"], syn_eff=bool(cfg["syn_eff"]),
                                 reads_scale=cfg["reads_scale"], unset_mask=[], unset_mode=cfg["unset_mode"])
        return Y.loc[studies, FINDINGS].astype(float)
    lab = T.load(EXP / f"{kind}/labels.py", f"o16_{kind}_labels")
    y = lab.make_targets(list(studies), cfg)[0]
    return pd.DataFrame(np.asarray(y, float), index=studies, columns=FINDINGS)


def ref(r, cols):
    """The recipe's pooled OOF probabilities of its PL columns (its deployed epoch / file)."""
    out = []
    for k in range(5):
        d = EXP / f"runs/{r}/fold{k}"
        if r == "bsd_nf":
            if not (d / "preds.parquet").exists():
                continue
            p = pd.read_parquet(d / "preds.parquet")
            out.append(p[(p["set"] == "oof") & (p.epoch == 14)][cols])
        elif (d / "oof.parquet").exists():
            p = pd.read_parquet(d / "oof.parquet")
            out.append((p[p.epoch == p.epoch.max()] if "epoch" in p else p)[cols])
    assert len(out) >= 3, f"{r}: OOF on {len(out)} folds"
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--recipes", default=",".join(SPEC))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    t = pd.read_parquet(D / f"teacher_{a.teacher}.parquet")
    folds = get_folds(5, 0, include_gold=False).fold
    gold = gold_labels().index
    assert set(t.index) == set(folds.index) and (t.fold == folds.reindex(t.index)).all()
    tr = folds.index.tolist()
    studies = tr + [s for s in gold if s not in set(tr)]
    al, st = T.alpha("sub_b0", tr)
    OUT.mkdir(exist_ok=True)
    for r in a.recipes.split(","):
        cols = SPEC[r][1]
        assert t[cols].notna().all().all(), f"{r}: teacher {cols} incomplete"
        Y = plain(r, studies)[cols]
        p = T.teacher_prob(t, ref(r, cols), cols).loc[tr]
        y = Y.copy()
        y.loc[tr] = (1 - al.loc[tr, cols]) * Y.loc[tr] + al.loc[tr, cols] * p
        assert y.notna().all().all() and ((y >= 0) & (y <= 1)).all().all()
        name = f"pl_ct_{r}_{a.teacher}"
        out = OUT / f"{name}.csv"
        if out.exists():
            assert a.force and a.teacher.startswith("v0"), f"{out} exists (only plumbing versions may be replaced)"
            os.chmod(out, 0o644)
        y.reset_index(names="StudyInstanceUID").to_csv(out, index=False, float_format="%.7f")
        os.chmod(out, 0o444)
        md5 = hashlib.md5(out.read_bytes()).hexdigest()
        with open(D / "manifest.txt", "a") as f:
            f.write(f"{md5}  {out.relative_to(REPO)}  recipe={r} teacher={a.teacher} rows={len(y)}\n")
        rep = [{"col": c, "y_plain": Y.loc[tr, c].mean(), "y_pl": y.loc[tr, c].mean(), "mean|d|": (y.loc[tr, c] - Y.loc[tr, c]).abs().mean(),
                "teacher_auc_explicit": auc((st[c][st[c].isin(T.EXPLICIT)] == "meets").astype(int), t.loc[tr, c][st[c].isin(T.EXPLICIT)]),
                "spearman_pl_plain": y.loc[tr, c].corr(Y.loc[tr, c], method="spearman")} for c in cols]
        txt = f"{name} ({SPEC[r][0]}): {len(tr)} train + {len(y) - len(tr)} gold rows, md5 {md5}\n" + pd.DataFrame(rep).set_index("col").round(4).to_string()
        (D / f"targets_{name}.txt").write_text(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()
