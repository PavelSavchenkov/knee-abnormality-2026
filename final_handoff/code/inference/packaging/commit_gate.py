"""Kaggle COMMIT-CHECK gate for a FINAL variant without a local GPU reference (Master 10-21 22:1x; OAI: no; CPU). Inputs: the variant's
commit outputs (check_pred / check_pred_base / check_parts on the same 4 edge studies) and a REFERENCE commit of another variant whose
seed sets overlap (e.g. safe3 = 81 of final's 118 files). Checks:
 A. SHARED members: every part that is a function of one checkpoint (Basin per-file parts) or of one member whose file list is IDENTICAL in
    both variants (fc_<member>, oa_<member>, synm_<member>[@out], the syn fusion when both designs' members match) must be bit-identical
    (max |dp| <= 1e-6; the safe3 vs safe3-lctest commits were bit-identical);
 B. SPLICE CHAIN recomputed from the variant's OWN Kaggle parts with its config constants (logits): Contusion = (1 - w) cont_sxm_all4 + w
    align(sxm img_cont); Fracture = (1 - w) frac_allmem + w align(bfs_rep Fracture); ACL / MCL splice parts; check_pred = DEC(check_pred_base)
    (<= 1e-5 in probability);
 C. INFO: per-column max |dp| and Spearman of check_pred vs the reference (expected small: more seeds / sxm).
  conda run -n rsna python submissions/oct21_final_alldata/commit_gate.py VARIANT COMMIT_DIR REF_VARIANT REF_COMMIT_DIR"""
import ast
import json
import re
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))   # noqa: E731
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(': ' + str(detail)[:900]) if detail else ''}")


def load(d):
    d = Path(d)
    parts = pd.read_csv(d / "check_parts.csv", index_col=[0, 1])
    return (pd.read_csv(d / "check_pred.csv", index_col=0), pd.read_csv(d / "check_pred_base.csv", index_col=0),
            {k: parts.loc[k].dropna(axis=1, how="all") for k in parts.index.get_level_values(0).unique()})


def member_files(v):
    """{part name: tuple of weight files} of variant v (its config in the registry's cfg dataset)."""
    cfg = json.load(open(HERE / f"pkg_{B.tag(v)}" / "config.json"))
    syn = json.load(open(HERE / f"pkg_{B.tag(v)}" / cfg["syn"]["members"]))
    out = {}
    for n, m in cfg["fc"]["members"].items():
        out[f"fc_{n}"] = tuple(m["files"])
    for n, m in cfg["oa"]["members"].items():
        out[f"oa_{n}"] = tuple(m["files"])
    for n, m in syn.items():
        out[f"synm_{n}"] = tuple(m["files"])
    if cfg.get("cont_sxm"):
        out[f"synm_{cfg['cont_sxm']['name']}"] = tuple(cfg["cont_sxm"]["files"])
    return cfg, syn, out


def main():
    v, cdir, rv, rdir = sys.argv[1:5]
    pred, base, parts = load(cdir)
    rpred, rbase, rparts = load(rdir)
    assert list(pred.index) == list(rpred.index) and len(pred) == 4, "the same 4 check studies"
    cfg, syn, mf = member_files(v)
    _, _, rmf = member_files(rv)
    # A. shared members
    shared = []
    for k in parts:
        if k not in rparts:
            continue
        stem = k if k in mf else k.split("@")[0]        # FC member names contain '@' themselves (runs/b3c_rep@k64): exact name first
        same = (k.endswith(".pt") and k in rparts) or (stem in mf and stem in rmf and mf[stem] == rmf[stem])
        if k == "syn_fused":                             # the fusion over both designs: equal iff every design member's files are equal
            designs = [n for d in cfg["syn"]["fuse"]["designs"].values() for ns in d.values() for n in ns]
            same = all(mf.get(f"synm_{n}") == rmf.get(f"synm_{n}") for n in designs)
        if same:
            a, b = parts[k], rparts[k].reindex(index=parts[k].index, columns=parts[k].columns)
            shared.append((k, float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))))
    worst = max(shared, key=lambda x: x[1]) if shared else ("-", 0.0)
    check(f"A. {len(shared)} shared parts (identical member files) bit-identical vs {rv}", shared and worst[1] <= 1e-6,
          f"worst {worst[0]} max |dp| {worst[1]:.2e}")
    # B. splice chain from the variant's own parts
    z = {k: lg(p.to_numpy(float)) for k, p in [("cont_sxm_all4", parts.get("cont_sxm_all4")), ("cont_sxm", parts.get("cont_sxm"))] if p is not None}
    bz = pd.DataFrame(lg(base.to_numpy(float)), index=base.index, columns=base.columns)
    al = lambda x, k: (x - k["mu_m"]) / k["sd_m"] * k["sd_d"] + k["mu_d"]   # noqa: E731
    if cfg.get("cont_sxm"):
        cs = cfg["cont_sxm"]
        x = lg(parts[f"synm_{cs['name']}@img_cont"].iloc[:, 0].to_numpy(float))
        c_exp = (1 - cs["w"]) * z["cont_sxm_all4"][:, 0] + cs["w"] * al(x, cs)
        d = float(np.max(np.abs(c_exp - bz["Contusion"].to_numpy())))
        check("B1. Contusion = (1 - w) all-4 + w align(sxm img_cont) == check_pred_base (logit)", d <= 1e-4, f"max |d| {d:.2e}")
    if cfg.get("frac_fs"):
        ff = cfg["frac_fs"]
        fa = lg(parts["frac_allmem"]["Fracture"].to_numpy(float))
        fb = lg(parts[f"fc_{ff['name']}"]["Fracture"].to_numpy(float))
        f_exp = (1 - ff["w"]) * fa + ff["w"] * al(fb, ff)
        d = float(np.max(np.abs(f_exp - bz["Fracture"].to_numpy())))
        check("B2. Fracture = (1 - w) frac_allmem + w align(bfs_rep Fracture) == check_pred_base (logit)", d <= 1e-4, f"max |d| {d:.2e}")
    src = nbformat.read(open(next((HERE / f"kernel_{B.tag(v)}").glob("*.ipynb"))), 4).cells[0].source
    dec = ast.literal_eval(re.findall(r"^DEC = (\{.*?\})\s+#", src, re.M)[0])
    out = base.copy()
    for f, dd in dec.items():
        p_, q_ = np.clip(base[f].to_numpy(np.float64), 1e-7, 1 - 1e-7), np.clip(base[dd["context"]].to_numpy(np.float64), 1e-7, 1 - 1e-7)
        out[f] = 1 / (1 + np.exp(-(np.log(p_ / (1 - p_)) - dd["b"] * (dd["c0"] + dd["c"] * np.log(q_ / (1 - q_))))))
    d = float((out - pred).abs().max().max())
    check(f"B3. check_pred == DEC(check_pred_base) ({list(dec)}; the kernel's DEC)", d <= 1e-6, f"max |dp| {d:.2e}")
    # C. info vs the reference variant
    dp = (pred - rpred).abs().max()
    rho = {c: float(pd.Series(pred[c]).corr(pd.Series(rpred[c]), method="spearman")) for c in pred.columns}
    print(f"INFO C. check_pred vs {rv}: max |dp| per column {dp.round(4).to_dict()}; Spearman (4 studies) {({k: round(x, 2) for k, x in rho.items()})}")
    print(f"INFO all finite: {bool(np.isfinite(pred.to_numpy()).all())}; parts {len(parts)}")
    print("COMMIT GATE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
