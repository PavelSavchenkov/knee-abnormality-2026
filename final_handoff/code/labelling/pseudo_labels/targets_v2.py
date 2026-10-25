"""PL v2 targets (USER 10-21 21:3x): the teacher OOF blended into EVERY training target of EVERY final recipe, text- and image-derived
alike. OAI: no. CPU. No trainer code changes: every target enters through an existing file input of its trainer.
Per cell: target = (1 - a) y_plain + a p_T,
  text cells:   a by v1 report status (0.3 meets / denied / structure_normal, 0.7 other stated, 1.0 not mentioned) = the v1 ct blend;
                bone Opus-read cells (tgt_p2_bone) a = 0.3 (an image read = an explicit label);
  image heads:  read cells a = 0.3; unread cells (masked in the plain run) a = 1.0 -> filled with the teacher and trained (weight w_img /
                w_ieff, as read cells);
  y_plain = the recipe's own plain targets from its own code + resolved all-data config; p_T = the teacher column (Synovitis for img_syn,
  Effusion for img_eff, Contusion / Fracture for img_cont / img_frac) quantile-mapped onto the recipe's OWN OOF marginal of that output
  (oct14 text recipes: sub_b0's, their v1 scale).
Not covered (no file input in the trainer): oct14 syn img_eff head (s_fs_nf / s_fc125_nf; not shipped since EF3); bone location heads c_*
and other auxiliary heads (no teacher column).
Outputs: oct14 text -> data/text_labels/external/pl2_<r>.csv (all 12 columns; columns a recipe does not train = plain); oct14 image ->
experiments/v2/oct14_all_data/data/pl/; oct16 -> experiments/v3/oct16_experts/data/pl/pl2_<r>[_img_*].csv. Read-only + manifest.txt.
  conda run -n rsna python work/pl_alldata_2026-10-21/targets_v2.py --recipes b3c_rep,z1_acl [--teacher v1]"""
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
E16, E14 = REPO / "experiments/v3/oct16_experts", REPO / "experiments/v2/oct14_all_data"
sys.path[:0] = [str(REPO), str(D)]
import targets as T  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.labels import gold_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

OUTS5 = ["Medial OA", "Lateral OA", "PF OA", "Medial Meniscus", "Lateral Meniscus"]
SYN3 = ["Synovitis", "Effusion", "Baker's"]
# recipe: (family, text columns trained, {image head: teacher column})
R = {"b3c_rep": ("o16_bone", ["Fracture", "Contusion"], {}), "b3l_rep": ("o16_bone", ["Fracture", "Contusion"], {}),
     "b3n_rep": ("o16_bone", ["Fracture", "Contusion"], {}), "bfs_rep": ("o16_bone", ["Fracture", "Contusion"], {}),
     "z1_acl": ("o16_syn", ["ACL"], {}), "w5b_mcl": ("o16_syn", ["MCL"], {}),
     "b3g": ("o16_syn", FINDINGS, {"img_syn": "Synovitis", "img_eff": "Effusion"}),
     "cs_fs": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}), "cs_fc125": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}),
     "ef3": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}), "ef3fc": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}),
     "ef3nf": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}), "ef3nffc": ("o16_syn", SYN3, {"img_syn": "Synovitis", "img_eff": "Effusion"}),
     "sxm": ("o16_syn", ["Contusion", "Fracture"], {"img_cont": "Contusion", "img_frac": "Fracture"}),
     "sd1b": ("o16_mainp", FINDINGS, {}), "bsd_nf": ("o16_basin", FINDINGS, {}),
     "oasep": ("o16_oa", OUTS5, {}), "oasep_nf": ("o16_oa", OUTS5, {}), "silsep": ("o16_oa", OUTS5, {}),
     "sub_b0": ("o14_main", FINDINGS, {}), "m_prob05": ("o14_main", FINDINGS, {"ext": "Synovitis"}),
     "nfnet": ("o14_basin", FINDINGS, {}), "r50": ("o14_basin", FINDINGS, {}), "n_prob05": ("o14_basin", FINDINGS, {"img": "Synovitis"}),
     "oa_v2cor": ("o14_oa", OUTS5, {}), "oa_v2cor_nf": ("o14_oa", OUTS5, {}),
     "s_fs_nf": ("o14_syn", SYN3, {"img_syn": "Synovitis"}), "s_fc125_nf": ("o14_syn", SYN3, {"img_syn": "Synovitis"})}
SRC14 = {"sub_b0": "experiments/v0/oct7_baseline/runs/sub_b0", "m_prob05": "experiments/v1/oct10_synovitis/runs/m_prob05",
         "n_prob05": "experiments/v1/oct10_synovitis/runs/n_prob05", "s_fs_nf": "experiments/v1/oct11_synovitis_crops/runs/s_fs_nf",
         "s_fc125_nf": "experiments/v1/oct11_synovitis_crops/runs/s_fc125_nf"}       # fold runs = the image heads' own OOF scale
SRC16 = {"cs_fs": "refs/syn_cnx/x_cnx5e-5", "cs_fc125": "refs/syn_cnx/x_cnx_fc125"}     # their fold runs (f0-4 / f0-2)
IMG_COL = {"img_syn": "img_Synovitis", "img_eff": "img_Effusion", "img_cont": "img_Contusion", "img_frac": "img_Fracture",
           "ext": "img_Synovitis", "img": "img_Synovitis"}


def cfg_of(r):
    return json.load(open(E14 / f"runs/A1_{r}/fold-1/config.json" if R[r][0].startswith("o14") else E16 / f"runs/{r}_all_s101/fold-1/config.json"))


def basin_targets(dat, cfg):
    assert not cfg["unset_mask"] and not cfg["reads"]
    Y, _ = dat.build_targets(cfg["target"], nm=None if cfg["nm"] == "mask" else float(cfg["nm"]), nm_weight=cfg["nm_weight"], hard=cfg["hard"],
                             reads=[], reads_weight=cfg["reads_weight"], syn_eff=bool(cfg["syn_eff"]), reads_scale=cfg["reads_scale"],
                             unset_mask=[], unset_mode=cfg["unset_mode"])
    return Y


def plain_text(r, studies):
    """studies x FINDINGS plain text targets, exactly as the plain all-data run builds them (incl. the recipe's own tgt_csv)."""
    fam, cfg = R[r][0], cfg_of(r)
    exp = E14 if fam.startswith("o14") else E16
    sub = fam.split("_")[1]
    if sub == "basin":
        sys.path.insert(0, str(exp / "basin"))
        Y = basin_targets(T.load(exp / "basin/data.py", f"{fam}_data"), cfg).loc[studies, FINDINGS].astype(float)
        sys.path.remove(str(exp / "basin"))
        return Y
    y = np.asarray(T.load(exp / f"{sub if sub != 'bone' else 'bone'}/labels.py", f"{fam}_labels").make_targets(list(studies), cfg)[0], float)
    Y = pd.DataFrame(y, index=studies, columns=FINDINGS)
    if fam == "o16_oa" and cfg.get("tgt_csv"):              # silsep's v4b: oa_train.py's inline tgt_csv block (589-597)
        t = pd.read_csv(REPO / cfg["tgt_csv"]).set_index("StudyInstanceUID").reindex(studies)
        for c in [x.replace("_", " ") for x in cfg["tgt_cols"].split("+")]:
            Y[c] = np.where(np.isfinite(t[c].to_numpy(float)), t[c].to_numpy(float), Y[c].to_numpy())
    return Y


def plain_image(r, head, studies):
    """The image head's plain target (NaN = unread, weight 0 in the plain run)."""
    fam, cfg = R[r][0], cfg_of(r)
    if head == "img_syn":
        base = E14 if fam.startswith("o14") else E16
        return pd.read_csv(base / cfg["img_csv"]).set_index("StudyInstanceUID").Synovitis.reindex(studies)
    if head == "img_eff":
        f = REPO / cfg["ieff_csv"] if cfg.get("ieff_csv") else E16 / "data/synovitis_labels_codex.csv"
        return pd.read_csv(f).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").Effusion.reindex(studies)
    if head in ("img_cont", "img_frac"):
        return pd.read_csv(REPO / cfg["ibone_csv"]).set_index("StudyInstanceUID")[IMG_COL[head][4:]].reindex(studies)
    if head == "ext":
        return pd.read_csv(REPO / cfg["ext_labels"]).set_index("StudyInstanceUID").Synovitis.reindex(studies)
    if head == "img":
        return pd.read_csv(E14 / cfg["img_csv"]).set_index("StudyInstanceUID").Synovitis.reindex(studies)
    raise ValueError(head)


def own_oof(r, col):
    """The recipe's pooled OOF probabilities of one output (text column or img_* column) at its deployed epoch."""
    fam = R[r][0]
    if fam.startswith("o14") and not col.startswith("img_"):
        return T.ref_oof(REPO / "experiments/v0/oct7_baseline/runs/sub_b0")[[col]]
    root = REPO / SRC14[r] if r in SRC14 else E16 / SRC16.get(r, f"runs/{r}")
    out = []
    for k in range(5):
        d = root / f"fold{k}"
        if r in ("n_prob05", "bsd_nf") and (d / "preds.parquet").exists():
            p = pd.read_parquet(d / "preds.parquet")
            out.append(p[(p["set"] == "oof") & (p.epoch == 14)][[col]])
        elif (d / "oof.parquet").exists():
            p = pd.read_parquet(d / "oof.parquet")
            out.append((p[p.epoch == p.epoch.max()] if "epoch" in p else p)[[col]])
    assert len(out) >= 3 and all(len(x) for x in out), f"{r} {col}: OOF on {len(out)} folds"
    return pd.concat(out)


def write(df, path, r, tag):
    if path.exists():
        raise SystemExit(f"{path} exists: never overwrite a released target file")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(names="StudyInstanceUID").to_csv(path, index=False, float_format="%.7f")
    os.chmod(path, 0o444)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    with open(D / "manifest.txt", "a") as f:
        f.write(f"{md5}  {path.relative_to(REPO)}  recipe={r} {tag} rows={len(df)}\n")
    return md5


def build(r, t, tv):
    fam, cols, imgs = R[r]
    need = sorted(set(cols) | set(imgs.values()))
    assert t[need].notna().all().all(), f"{r}: teacher incomplete for {[c for c in need if t[c].isna().any()]}"
    folds = get_folds(5, 0, include_gold=False).fold
    tr = folds.index.tolist()
    studies = tr + [s for s in gold_labels().index if s not in set(tr)]
    al, st = T.alpha("sub_b0", tr)
    Y = plain_text(r, studies)
    y = Y.copy()
    if fam == "o16_bone":                                    # Opus-read cells = image reads: a 0.3
        read = pd.read_csv(E16 / "data/v3c/tgt_p2_bone.csv").set_index("StudyInstanceUID")[cols].reindex(tr).notna()
        al.loc[tr, cols] = al.loc[tr, cols].mask(read, 0.3)
    for c in cols:
        p = T.teacher_prob(t, own_oof(r, c), [c]).loc[tr, c]
        y.loc[tr, c] = (1 - al.loc[tr, c]) * Y.loc[tr, c] + al.loc[tr, c] * p
    rep = [f"{c}: plain {Y.loc[tr, c].mean():.4f} -> pl {y.loc[tr, c].mean():.4f}, spearman {y.loc[tr, c].corr(Y.loc[tr, c], method='spearman'):.3f}"
           for c in cols]
    files = {}
    if fam.startswith("o14"):
        files["text"] = write(y[FINDINGS], REPO / f"data/text_labels/external/pl2_{r}_{tv}.csv", r, "text(all 12; untrained = plain)")
    else:
        keep = FINDINGS if fam == "o16_syn" else cols            # syn: all 12 columns (untrained = plain) -> tgt_findings stays empty (no "Baker's" token in queue lines)
        files["text"] = write(y[keep], E16 / f"data/pl/pl2_{r}_{tv}.csv", r, f"text {cols} (file columns {len(keep)})")
    for head, tc in imgs.items():
        v = plain_image(r, head, studies)
        read = v.loc[tr].notna()
        p = T.teacher_prob(t, own_oof(r, IMG_COL[head]).rename(columns={IMG_COL[head]: tc}), [tc]).loc[tr, tc]
        out = v.copy()
        out.loc[tr] = np.where(read, 0.7 * v.loc[tr].fillna(0) + 0.3 * p, p)
        col = {"img_syn": "Synovitis", "ext": "Synovitis", "img": "Synovitis", "img_eff": "Effusion", "img_cont": "Contusion",
               "img_frac": "Fracture"}[head]
        df = out.to_frame(col)
        if head in ("img_cont", "img_frac"):                  # sxm: one ibone_csv with both columns
            continue
        base = E14 / "data/pl" if fam.startswith("o14") else E16 / "data/pl"
        files[head] = write(df, base / f"pl2_{r}_{head}_{tv}.csv", r, f"image {head} (read {int(read.sum())} / {len(tr)})")
        rep.append(f"{head}: read {int(read.sum())} of {len(tr)}; read cells plain {v.loc[tr][read].mean():.4f} -> {out.loc[tr][read].mean():.4f};"
                   f" unread filled mean {out.loc[tr][~read].mean():.4f}")
    if "img_cont" in imgs:
        df = pd.DataFrame(index=studies)
        for head in ("img_cont", "img_frac"):
            v = plain_image(r, head, studies)
            read = v.loc[tr].notna()
            tc = imgs[head]
            p = T.teacher_prob(t, own_oof(r, IMG_COL[head]).rename(columns={IMG_COL[head]: tc}), [tc]).loc[tr, tc]
            o = v.copy()
            o.loc[tr] = np.where(read, 0.7 * v.loc[tr].fillna(0) + 0.3 * p, p)
            df[IMG_COL[head][4:]] = o
            rep.append(f"{head}: read {int(read.sum())} of {len(tr)}; unread filled mean {o.loc[tr][~read].mean():.4f}")
        files["ibone"] = write(df, E16 / f"data/pl/pl2_{r}_img_bone_{tv}.csv", r, "image img_cont / img_frac")
    txt = f"pl2 {r} ({fam}) teacher {tv}: " + json.dumps(files) + "\n  " + "\n  ".join(rep)
    (D / f"targets_pl2_{r}_{tv}.txt").write_text(txt + "\n")
    print(txt, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--teacher", default="v1")
    ap.add_argument("--tag", default="v1", help="file version suffix (v0 = plumbing)")
    a = ap.parse_args()
    os.chdir(REPO)
    t = pd.read_parquet(D / f"teacher_{a.teacher}.parquet")
    for r in a.recipes.split(","):
        build(r, t, a.tag)


if __name__ == "__main__":
    main()
