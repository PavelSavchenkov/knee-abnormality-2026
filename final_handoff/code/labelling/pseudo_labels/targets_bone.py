"""PL targets for the all-data bone students, tier B (oct16 bone trainer, its existing tgt_csv input; no code change). OAI: no. CPU.
Recipes b3c_rep / b3l_rep / b3n_rep / bfs_rep (Contusion / Fracture columns + crop-fusion members). Per cell of Fracture / Contusion:
  Opus-read cells (data/v3c/tgt_p2_bone.csv not NaN) keep the recipe's own target (the image read; never overwritten);
  every other cell = (1 - a) y_plain + a p_T, the tier-A ct blend (targets.alpha), p_T = the teacher on THIS recipe's pooled OOF
  marginal (targets.teacher_prob; v3m-trained recipes sit lower than v1);
  y_plain = the recipe's own targets from its own code (bone/labels.make_targets) and resolved all-data config (runs/<r>_all_s101/fold-1).
The PL line differs from the plain line only in tgt_csv (this file replaces the recipe's tgt_csv: it holds the same values on read cells)
and the seed. Gold-58 rows get y_plain. Output: experiments/v3/oct16_experts/data/pl/pl_ct_<recipe>_<tv>.csv (read-only) + manifest.txt.
  conda run -n rsna python work/pl_alldata_2026-10-21/targets_bone.py --teacher v1 [--force]"""
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

RECIPES = ("b3c_rep", "b3l_rep", "b3n_rep", "bfs_rep")
COLS = ["Fracture", "Contusion"]
OUT = EXP / "data/pl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)                                       # tgt_csv paths in the configs are repo-relative (as the runner runs them)
    lab = T.load(EXP / "bone/labels.py", "o16_bone_labels")
    t = pd.read_parquet(D / f"teacher_{a.teacher}.parquet")
    folds = get_folds(5, 0, include_gold=False).fold
    gold = gold_labels().index
    assert set(t.index) == set(folds.index) and (t.fold == folds.reindex(t.index)).all()
    assert t[COLS].notna().all().all(), "teacher Fracture / Contusion incomplete"
    tr = folds.index.tolist()
    studies = tr + [s for s in gold if s not in set(tr)]
    al, st = T.alpha("sub_b0", tr)                      # no column excluded (sub_b0's ct table)
    read = pd.read_csv(EXP / "data/v3c/tgt_p2_bone.csv").set_index("StudyInstanceUID")[COLS].reindex(tr).notna()
    OUT.mkdir(exist_ok=True)
    for r in RECIPES:
        cfg = json.load(open(EXP / f"runs/{r}_all_s101/fold-1/config.json"))
        nf = sorted(int(x.name[4:]) for x in (EXP / f"runs/{r}").glob("fold[0-4]") if (x / "oof.parquet").exists())
        p = T.teacher_prob(t, T.ref_oof(EXP / f"runs/{r}", nf), COLS).loc[tr]   # the recipe's own OOF scale (v3m / v1 differ)
        assert set(cfg["outs"].split(",")) >= {"frac", "cont"} and not cfg.get("frac1") and not cfg.get("tgt_w") and not cfg.get("tgt_gw")
        y, w = lab.make_targets(studies, cfg)
        Y = pd.DataFrame(np.asarray(y, float), index=studies, columns=FINDINGS)[COLS]
        y_pl = Y.copy()
        blend = (1 - al.loc[tr, COLS]) * Y.loc[tr] + al.loc[tr, COLS] * p
        y_pl.loc[tr] = blend.where(~read, Y.loc[tr])
        assert y_pl.notna().all().all() and ((y_pl >= 0) & (y_pl <= 1)).all().all()
        assert (y_pl.loc[tr][read] == Y.loc[tr][read]).sum().sum() == read.sum().sum(), "read cells must keep the image read"
        name = f"pl_ct_{r}_{a.teacher}"
        out = OUT / f"{name}.csv"
        if out.exists():
            assert a.force and a.teacher.startswith("v0"), f"{out} exists (only plumbing versions may be replaced)"
            os.chmod(out, 0o644)
        y_pl.reset_index(names="StudyInstanceUID").to_csv(out, index=False, float_format="%.7f")
        os.chmod(out, 0o444)
        md5 = hashlib.md5(out.read_bytes()).hexdigest()
        with open(D / "manifest.txt", "a") as f:
            f.write(f"{md5}  {out.relative_to(REPO)}  recipe={r} teacher={a.teacher} rows={len(y_pl)}\n")
        rep = []
        for c in COLS:
            ex = st[c].isin(("meets", "denied", "structure_normal"))
            rep.append({"col": c, "read": int(read[c].sum()), "changed": int(((y_pl.loc[tr, c] - Y.loc[tr, c]).abs() > 1e-6).sum()),
                        "y_plain": Y.loc[tr, c].mean(), "y_pl": y_pl.loc[tr, c].mean(),
                        "teacher_auc_explicit": auc((st[c][ex] == "meets").astype(int), t.loc[tr, c][ex]),
                        "spearman_pl_plain": y_pl.loc[tr, c].corr(Y.loc[tr, c], method="spearman")})
        txt = f"{name}: {len(tr)} train + {len(y_pl) - len(tr)} gold rows, md5 {md5}, plain tgt_csv {cfg['tgt_csv']}, ref OOF folds {nf}\n" + \
            pd.DataFrame(rep).set_index("col").round(4).to_string()
        (D / f"targets_{name}.txt").write_text(txt + "\n")
        print(txt)


if __name__ == "__main__":
    main()
