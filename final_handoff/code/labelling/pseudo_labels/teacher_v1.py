"""PL teacher v1 = the CURRENT global-best FOLD pipeline's OOF logits per column on all 4,349 non-gold studies. OAI: no. CPU. Read-only use of
Experts' column builders (experiments/v3/oct16_experts/scripts, map from Experts 10-21 17:1x / 17:2x). One form per column; FORM records the
source per fold group. f3-4 parts that need the teacher fold-3/4 runs fail loudly until those runs exist.
  Baker's            f0-2 pkg961 (shipped window fusion) | f3-4 fused3 = the same 3x3 window fusion (parity-checked on f0-2)
  Lateral OA         f0-2 c188.bases Q15                 | f3-4 clean_f34 Q15 (c34 NF slot)
  PF / MOA / MM      f0-2 pkg961 (0.961 forms)           | f3-4 clean_f34
  Lateral Meniscus   f0-2 c188.bases Q22 (DEC on Q16 ACL) | f3-4 clean_f34 Q22 (= the v2 kernel DEC constants)
  ACL                f0-2 c157.acl16 (Q16)               | f3-4 c213.acl16_on
  Contusion          all folds swap_read all-4 (b3c / b3l / b3n / bfs_rep); f3-4 NF slot only with --nf_all (b3n_rep f3/f4 dumps);
                     --sxm (only if probe 57480087 wins): + the cont_sxm splice w 0.3 (out/sxm_add_fit.json ARM; f3-4 needs sxm f3/f4 runs)
  Fracture           all folds v3c_fs = all-4 column + bfs_rep splice (frac_fs.json)
  Effusion           all folds EF3 = window fusion base + EF3 img_eff term, v2 recipe + EF3 stats (f3-4 needs ef3* fold 3/4 runs)
  Synovitis          f0-2 c188.bases cnx form            | f3-4 its study-level form 1/3 image heads + 2/3 crop pipelines (cnx, NF) (needs
                     the F3/F4 m_prob05 / n_prob05 / s_*_nf runs + cs_fc125 f3/f4)
  MCL                all folds Q21 chain (SD-2 3x3 -> w5b zoom -> b3g), then the v2 kernel DEC on the teacher's Contusion
  conda run -n rsna python work/pl_alldata_2026-10-21/teacher_v1.py --folds 0,1,2 [--check]     (parts/<col>_f<F>.parquet)
  conda run -n rsna python work/pl_alldata_2026-10-21/teacher_v1.py --assemble [--nf_all] [--sxm] (teacher_v1.parquet + teacher_v1_src.json)"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
EXP = REPO / "experiments/v3/oct16_experts"
A14 = REPO / "experiments/v2/oct14_all_data/runs"
sys.path[:0] = [str(REPO), str(EXP / "scripts"), str(EXP / "bone"), str(REPO / "submissions/oct20_probe_eff_ef3_v2")]
import build as ef3b  # noqa: E402  (the EF3 kit's term() / recipe(); main() not run)
import c188_item_screen as c188  # noqa: E402
import lib  # noqa: E402
import swap_read as sr  # noqa: E402
from c122_allpw_read import fused3  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.metrics import auc  # noqa: E402
from common.labels import status_table  # noqa: E402
from common.paths import FINDINGS, KEYS  # noqa: E402

PARTS = D / "parts"
gi = set(lib.GOLD.index)
FOLDS = get_folds(5, 0, include_gold=False).fold
PK = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)
ALL4 = "b_v3=b3c_rep,b_lr3=b3l_rep,b_nf=b3n_rep,bfs=bfs_rep"
FRAC = json.load(open(REPO / "submissions/oct21_probe_frac_bsif/frac_fs.json"))
EF3_ST = json.load(open(REPO / "submissions/oct20_probe_eff_ef3_v2/dataset/member.json"))["eff_ef3"]["stats"]
_src = "".join(json.load(open(next((REPO / "submissions/oct20_stack_all_v2/kernel").glob("*.ipynb"))))["cells"][0]["source"])
DEC = ast.literal_eval(re.findall(r"^DEC = (\{.*?\})\s+#", _src, re.M)[0])
lgt = lambda p: np.log(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6) / (1 - np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)))  # noqa: E731


def ids_of(F):
    return FOLDS.index[FOLDS == F]


def clean(c, F):
    x = pd.read_csv(EXP / f"data/clean_f34/{c.replace(' ', '_')}.csv").set_index("StudyInstanceUID")
    return x.logit[x.fold == F]


def run_logit(run_dir, col, epoch=None, kind="oof"):
    """Study-level logit of one run's per-epoch parquet (oof.parquet: probabilities; epoch = last unless given)."""
    d = pd.read_parquet(run_dir / f"{kind}.parquet")
    if "epoch" in d:
        d = d[d.epoch == (d.epoch.max() if epoch is None else epoch)]
    d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d else d
    assert len(d), (run_dir, col, epoch)
    return pd.Series(lgt(d[col].to_numpy(float)), index=d.index).groupby(level=0).mean()


def contusion_fracture(F, nf_all):
    sw = sr.spec(ALL4)
    if F > 2 and not nf_all:
        sw.pop("b_nf")
    col = sr.column(F, "oof", sw, nf_all=nf_all) if F > 2 else sr.column(F, "oof", sw)
    x = col["Fracture"]
    b = run_logit(EXP / f"runs/{FRAC['fs_member']}/fold{F}", "Fracture").reindex(x.index)
    assert b.notna().mean() > .99
    fr = (1 - FRAC["w"]) * x + FRAC["w"] * ((b - FRAC["mu_m"]) / FRAC["sd_m"] * FRAC["sd_d"] + FRAC["mu_d"]).fillna(x)
    return col["Contusion"], fr


def sxm_splice(F, x):
    """oct21_probe_cont_sxm: Contusion' = (1 - w) all-4 + w ((m - mu_m) / sd_m * sd_d + mu_d), m = sxm fold-F img_Contusion logit (epoch 19)."""
    k = json.load(open(EXP / "out/sxm_add_fit.json"))["ARM"]
    assert (EXP / f"runs/sxm/fold{F}/done.txt").exists(), f"sxm f{F} not trained yet"
    m = run_logit(EXP / f"runs/sxm/fold{F}", "img_Contusion", 19).reindex(x.index)
    assert m.notna().mean() > .99
    return (1 - k["w"]) * x + k["w"] * ((m - k["mu_m"]) / k["sd_m"] * k["sd_d"] + k["mu_d"])


def effusion(F, ids):
    z = dict(np.load(EXP / f"data/win3x3/win_f{F}.npz", allow_pickle=True))
    base = fused3(z, [None] * 3, F, False)["Effusion"].reindex(ids)
    for s in ("ef3", "ef3fc", "ef3nf", "ef3nffc"):
        assert (EXP / f"runs/{s}/fold{F}/done.txt").exists(), f"Effusion f{F}: runs/{s}/fold{F} not trained yet"
    ee = ef3b.term(ef3b.NEWP, "oof", ids, [F])
    return ef3b.recipe(base, ee, EF3_ST)


def syn_member(name, F):
    """Study-level img_Synovitis logit of a Synovitis member's fold-F model (f3-4: the teacher fold runs)."""
    if F <= 2:
        return {"m_prob05": lambda: lib.oof_member("mprob", F)["img_Synovitis"], "n_prob05": lambda: lib.oof_member("nprob", F)["img_Synovitis"],
                "s_fs_nf": lambda: lib.oof_member("s_fs_nf", F)["img_Synovitis"], "s_fc125_nf": lambda: lib.oof_member("s_fc125_nf", F)["img_Synovitis"],
                "x_cnx5e-5": lambda: run_logit(EXP / f"refs/syn_cnx/x_cnx5e-5/fold{F}", "img_Synovitis", 19),
                "x_cnx_fc125": lambda: run_logit(EXP / f"refs/syn_cnx/x_cnx_fc125/fold{F}", "img_Synovitis", 19)}[name]()
    if name == "x_cnx5e-5":
        return run_logit(EXP / f"refs/syn_cnx/x_cnx5e-5/fold{F}", "img_Synovitis", 19)
    if name == "x_cnx_fc125":
        r = EXP / f"runs/cs_fc125/fold{F}"
        assert (r / "done.txt").exists(), f"Synovitis f{F}: {r} not trained yet"
        return run_logit(r, "img_Synovitis", 19)
    r = A14 / f"F{F}_{name}/fold{F}"
    assert (r / "done.txt").exists(), f"Synovitis f{F}: {r} not trained yet"
    if name == "n_prob05":       # = refs/pkg961 'nprob' exactly on f0 (max |d| 0.0): preds.parquet, set oof, epoch 14
        p = pd.read_parquet(r / "preds.parquet")
        p = p[(p["set"] == "oof") & (p.epoch == 14)]
        assert len(p), f"{r}: no oof rows at epoch 14"
        p = p.set_index("StudyInstanceUID") if "StudyInstanceUID" in p else p
        return pd.Series(lgt(p["img_Synovitis"].to_numpy(float)), index=p.index).groupby(level=0).mean()
    return run_logit(r, "img_Synovitis", 19)   # m_prob05 / s_*_nf = pkg961 'mprob' / 's_*_nf' exactly on f0: oof.parquet epoch 19


def synovitis_study_level(F, ids):
    """lib.package's Synovitis with the cnx crops in the main pipeline: 1/3 image heads + 2/3 mean over {cnx, NF} crop pipelines."""
    ih = (syn_member("m_prob05", F) + syn_member("n_prob05", F)) / 2
    crops = ((syn_member("x_cnx5e-5", F) + syn_member("x_cnx_fc125", F)) / 2 + (syn_member("s_fs_nf", F) + syn_member("s_fc125_nf", F)) / 2) / 2
    return (ih / 3 + 2 * crops / 3).reindex(ids)


def mcl_pre_dec(F):
    """Q21 chain (submissions/oct18_probe_mcl_stack/build.py, its stored constants) on fold F, before the kernel DEC."""
    main = run_logit(EXP / f"runs/sd1b/fold{F}", "MCL")
    p = pd.read_parquet(EXP / f"runs/bsd_nf/fold{F}/preds.parquet")
    p = p[(p.epoch == 14) & (p["set"] == "oof")]
    nf = pd.Series(lgt(p["MCL"].to_numpy(float)), index=p["StudyInstanceUID"] if "StudyInstanceUID" in p else p.index).groupby(level=0).mean()
    r = lib.oof_member("r50", F)["MCL"]
    r = r if r.min() < 0 else pd.Series(lgt(r.to_numpy(float)), index=r.index)
    sd2 = ((main + nf.reindex(main.index) + r.reindex(main.index)) / 3).dropna()
    zm = json.load(open(REPO / "submissions/oct17_probe_mcl_zoom/dataset/member.json"))["mcl_member"]
    rf = json.load(open(REPO / "submissions/oct18_probe_mcl_sd2/dataset/member_sd.json"))["mcl_member_refit"]
    a2 = json.load(open(REPO / "submissions/oct18_probe_mcl_stack/stack.json"))
    w5 = run_logit(EXP / f"runs/w5b_mcl/fold{F}", "MCL", 19)
    bo = run_logit(EXP / f"runs/b3g/fold{F}", "MCL")
    m1 = 0.75 * sd2 + 0.25 * ((w5.reindex(sd2.index) - zm["mu_m"]) / zm["sd_m"] * rf["sd_d"] + rf["mu_d"])
    return (1 - a2["w"]) * m1 + a2["w"] * ((bo.reindex(m1.index) - a2["mu_m"]) / a2["sd_m"] * a2["sd_d"] + a2["mu_d"]), sd2


def eff_ready(F):
    return F <= 2 or all((EXP / f"runs/{s}/fold{F}/done.txt").exists() for s in ("ef3", "ef3fc", "ef3nf", "ef3nffc"))


def syn_ready(F):
    return F <= 2 or ((EXP / f"runs/cs_fc125/fold{F}/done.txt").exists()
                      and all((A14 / f"F{F}_{n}/fold{F}/done.txt").exists() for n in ("m_prob05", "n_prob05", "s_fs_nf", "s_fc125_nf")))


def sfx(F, nf_all, sxm):
    return ("_nfall" if nf_all and F > 2 else "") + ("_sxm" if sxm else "")


def build_fold(F, nf_all, sxm, partial=False):
    """partial: Effusion / Synovitis whose f3-4 teacher runs are not done yet stay NaN ('PENDING'); every other column is complete."""
    ids = ids_of(F)
    out, form = {}, {}
    pend = [c for c, ok in (("Effusion", eff_ready(F)), ("Synovitis", syn_ready(F))) if not ok]
    assert partial or not pend, f"f{F}: teacher runs pending for {pend} (use --partial)"
    if F <= 2:
        b = c188.bases(F, ["Lateral OA", "Lateral Meniscus", "ACL", "Synovitis"])[0]
        for c in ("Baker's", "PF OA", "Medial OA", "Medial Meniscus"):
            out[c], form[c] = PK.loc[ids, c], "pkg961"
        for c, f in (("Lateral OA", "Q15 (c188)"), ("Lateral Meniscus", "Q22 (c188)"), ("ACL", "Q16 acl16"), ("Synovitis", "cnx (c188)")):
            out[c], form[c] = b[c][0], f
    else:
        import c213_clean_forms as c213
        z = dict(np.load(EXP / f"data/win3x3/win_f{F}.npz", allow_pickle=True))
        out["Baker's"], form["Baker's"] = fused3(z, [None] * 3, F, False)["Baker's"], "3x3 window fusion (fused3)"
        for c in ("Lateral OA", "PF OA", "Medial OA", "Medial Meniscus", "Lateral Meniscus"):
            out[c], form[c] = clean(c, F), "clean_f34 (C213)"
        out["ACL"], form["ACL"] = c213.acl16_on(F), "Q16 acl16_on (C213)"
        if "Synovitis" in pend:
            out["Synovitis"], form["Synovitis"] = pd.Series(np.nan, index=ids), "PENDING"
        else:
            out["Synovitis"], form["Synovitis"] = synovitis_study_level(F, ids), "study-level cnx form (1/3 heads + 2/3 cnx / NF crops)"
    out["Contusion"], out["Fracture"] = contusion_fracture(F, nf_all)
    form["Contusion"] = "all-4 swap_read" + (" + NF slot (nf_all)" if F > 2 and nf_all else "")
    form["Fracture"] = form["Contusion"].replace("all-4", "v3c_fs: all-4") + " + bfs_rep splice w 0.75"
    if sxm:
        out["Contusion"] = sxm_splice(F, out["Contusion"])
        form["Contusion"] += " + cont_sxm splice w 0.3"
    if "Effusion" in pend:
        out["Effusion"], form["Effusion"] = pd.Series(np.nan, index=ids), "PENDING"
    else:
        out["Effusion"], form["Effusion"] = effusion(F, ids), "EF3 (window fusion + EF3 img_eff, v2 recipe, EF3 stats)"
    out["MCL"], _ = mcl_pre_dec(F)
    form["MCL"] = "Q21 chain, then the v2 kernel DEC on this teacher's Contusion"
    T = pd.DataFrame({c: out[c].reindex(ids) for c in FINDINGS})
    live = [c for c in FINDINGS if c not in pend]
    miss = T[live].isna().mean()
    assert (miss < 0.01).all(), f"f{F} coverage: {miss[miss >= 0.01].to_dict()}"
    T[live] = T[live].apply(lambda s: s.fillna(s.median()))   # < 1% per column (studies a member never scored); counted in the report
    PARTS.mkdir(exist_ok=True)
    T.to_parquet(PARTS / f"f{F}{sfx(F, nf_all, sxm)}.parquet")
    json.dump({"form": form, "missing_share": miss.round(5).to_dict(), "pending": pend}, open(PARTS / f"f{F}{sfx(F, nf_all, sxm)}.json", "w"), indent=1)
    print(f"f{F}: {len(T)} studies, missing filled {miss[miss > 0].round(4).to_dict()}, pending {pend}")


def dec(T):
    """The v2 kernel's DEC (MCL <- Contusion) on logits; LM's Q22 already carries the same constants (asserted)."""
    q22 = json.load(open(REPO / "submissions/oct19_probe_acl_b3g_q8/dec.json"))["Lateral Meniscus"]
    assert DEC["Lateral Meniscus"] == q22 and set(DEC) == {"MCL", "Lateral Meniscus"}, DEC
    d = DEC["MCL"]
    T = T.copy()
    T["MCL"] = T["MCL"] - d["b"] * (d["c0"] + d["c"] * T[d["context"]])
    return T


def check():
    """Parity of the f3-4 builders on f0-2 where an exact shipped OOF exists."""
    for F in (0, 1, 2):
        ids = ids_of(F)
        z = dict(np.load(EXP / f"data/win3x3/win_f{F}.npz", allow_pickle=True))
        bk = fused3(z, [None] * 3, F, False)["Baker's"].reindex(ids)
        d = float((bk - PK.loc[ids, "Baker's"]).abs().max())
        print(f"f{F} Baker's fused3 vs pkg961: max |d| {d:.2e}")
        s0 = c188.bases(F, ["Synovitis"])[0]["Synovitis"][0]
        s1 = synovitis_study_level(F, s0.index)
        print(f"f{F} Synovitis study-level vs shipped cnx form: spearman {s1.corr(s0, method='spearman'):.4f}, n {len(s0)}")
    _, sd2 = zip(*[mcl_pre_dec(F) for F in (0, 1, 2)])
    s = pd.concat(sd2)
    s = s[~s.index.isin(gi)]
    rf = json.load(open(REPO / "submissions/oct18_probe_mcl_sd2/dataset/member_sd.json"))["mcl_member_refit"]
    print(f"MCL SD-2 part pooled f0-2: mu {s.mean():.6f} sd {s.std():.6f} vs stored refit {rf['mu_d']:.6f} / {rf['sd_d']:.6f}")


def assemble(nf_all, sxm):
    """Columns pending on any fold are NaN everywhere (never half a column). A rebuild must reproduce every column that an earlier
    teacher_v1.parquet already had complete, bit for bit (targets built from it stay consistent)."""
    T = pd.concat([pd.read_parquet(PARTS / f"f{F}{sfx(F, nf_all, sxm)}.parquet") for F in range(5)])
    pend = sorted({c for F in range(5) for c in json.load(open(PARTS / f"f{F}{sfx(F, nf_all, sxm)}.json")).get("pending", [])})
    T[pend] = np.nan
    T = dec(T)
    live = [c for c in FINDINGS if c not in pend]
    assert set(T.index) == set(FOLDS.index) and T[live].notna().all().all()
    T.insert(0, "fold", FOLDS.reindex(T.index))
    old = D / "teacher_v1.parquet"
    if old.exists():
        O = pd.read_parquet(old).reindex(T.index)
        both = [c for c in FINDINGS if O[c].notna().all() and c in live]
        d = float((O[both] - T[both]).abs().max().max()) if both else 0.0
        assert d == 0.0, f"rebuild changed already-complete teacher columns (max |d| {d}): stop, targets built from them would be stale"
        print(f"consistency: {both} identical to the previous teacher_v1.parquet")
    T.to_parquet(D / "teacher_v1.parquet")
    src = {f"f{F}": json.load(open(PARTS / f"f{F}{sfx(F, nf_all, sxm)}.json")) for F in range(5)}
    src["pending_columns"] = pend
    json.dump(src, open(D / "teacher_v1_src.json", "w"), indent=1)
    st = status_table("v1").loc[T.index, KEYS]
    st.columns = FINDINGS
    rows = []
    for F in range(5):
        m = T.fold == F
        row = {"fold": F}
        for c in live:
            ex = st.loc[m, c].isin(("meets", "denied", "structure_normal"))
            row[c] = auc((st.loc[m, c][ex] == "meets").astype(int), T.loc[m, c][ex])
        rows.append(row)
    print("teacher AUC on explicit report cells (meets vs denied / structure_normal), per fold:")
    print(pd.DataFrame(rows).set_index("fold").round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--nf_all", action="store_true", help="f3-4 Contusion / Fracture with the b3n_rep NF slot (needs its f3/f4 dumps)")
    ap.add_argument("--sxm", action="store_true", help="Contusion + the cont_sxm splice (only if probe 57480087 wins)")
    ap.add_argument("--partial", action="store_true", help="leave Effusion / Synovitis NaN on f3-4 while their teacher runs are pending")
    a = ap.parse_args()
    if a.check:
        check()
    for F in [int(x) for x in a.folds.split(",") if x]:
        build_fold(F, a.nf_all, a.sxm, a.partial)
    if a.assemble:
        assemble(a.nf_all, a.sxm)


if __name__ == "__main__":
    main()
