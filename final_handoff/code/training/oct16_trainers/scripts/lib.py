"""Shared helpers for the Lateral OA session: the 0.961 package's OA block rebuilt from member logits, and the LOA read panel.

0.961 package (submissions/oct13_f5_win) OA block, per study:
  base = mean over pipelines (main sub_b0, NFNet, R50) of the pipeline's fold-mean logit
  e    = crop-level fusion geo|pm|g10|wmean of the OA experts (effb0 v2cor f0-4, NFNet v2cor f0-2) on shared crops, aligned to the
         plain expert mean's OOF distribution (refs/oa_fuse_recipe.json)
  OA   = oa_recipe(base, e): z-align e to base, (1 - 0.6) base + 0.6 z, then de-coupling (LOA k 0.4 on MOA / PF)
OOF (folds 0-2): one fold model per member per study (dumps exist for f0-2 only). Gold-58: every fold model (test-like).
Copied from experiments/v2/oct13_deep_ensembling_test/lib.py + w_oa.py (refs/ = copies of its data/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE / "scripts")]
import wfuse  # noqa: E402
from common.labels import gold_labels, status_table, text_labels  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

OA = ["Medial OA", "Lateral OA", "PF OA"]
MEN = ["Medial Meniscus", "Lateral Meniscus"]
OUTS = OA + MEN
PIPES = ["main", "nfnet", "r50"]
LOA = "Lateral OA"
CFG = json.load(open(HERE / "oa_refs/config.json"))
FUSE = json.load(open(HERE / "oa_refs/oa_fuse_recipe.json"))
GOLD = gold_labels()
CONTESTED_LOA = {s for s in GOLD.index if s.endswith("63451855003")}      # Master 10-16: gold 1, report 'cartilages normal', reads ~0.02
assert len(CONTESTED_LOA) == 1
DOMINANT_LOA = CONTESTED_LOA | {s for s in GOLD.index if s.endswith("559395")}   # + report-silent trauma knee, every image read negative
assert len(DOMINANT_LOA) == 2
ST = status_table("v1")
T = pd.read_parquet(HERE / "oa_refs/members.parquet")
OOF = {m: d.drop(columns=["member", "set"]).set_index("StudyInstanceUID") for m, d in T[T.set == "oof"].groupby("member")}
GLD = {m: d.drop(columns=["member", "set"]) for m, d in T[T.set == "gold"].groupby("member")}
FOLD = pd.concat([d.fold for d in OOF.values()]).groupby(level=0).first()
WIN = HERE / "oa_refs/win"


def sig(x):
    return 1 / (1 + np.exp(-x))


def lgt(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def oa_recipe(b, e, rec):
    """= submissions/oct13_f5_win/oa_stage.apply_recipe."""
    o = b.copy()
    for f in rec["stats"]:
        s = rec["stats"][f]
        z = (e[f] - s["mu_e"]) * s["sd_b"] / s["sd_e"] + s["mu_b"]
        o[f] = (1 - rec["w"]) * b[f] + rec["w"] * z.fillna(b[f])
    out = o.copy()
    for f, (k, c0, *c) in rec.get("dec", {}).items():
        out[f] = o[f] - k * (c0 + o[[g for g in OA if g != f]].to_numpy() @ np.asarray(c))
    return out


def fuse_files(files, groups, spec=FUSE["spec"]):
    """Crop-level fusion over dumps with identical crop lists -> DataFrame studies x OUTS (logits)."""
    Z = [dict(np.load(WIN / f)) for f in files]
    ids = [str(s) for s in Z[0]["studies"]]
    for z in Z[1:]:
        assert [str(s) for s in z["studies"]] == ids and (z["meta"] == Z[0]["meta"]).all(), "crop lists differ"
    off = np.r_[Z[0]["offsets"], len(Z[0]["s"])]
    out = []
    for i in range(len(ids)):
        a, b = off[i], off[i + 1]
        c = wfuse.prepare(np.stack([z["s"][a:b] for z in Z]).astype(np.float64), np.stack([z["u"][a:b] for z in Z]).astype(np.float64), groups)
        out.append(wfuse.fuse(c, spec))
    return pd.DataFrame(np.array(out), index=ids, columns=OUTS)


def align(ef):
    al = FUSE["align"]
    return pd.DataFrame({f: (ef[f] - al[f]["mu_f"]) / al[f]["sd_f"] * al[f]["sd_e"] + al[f]["mu_e"] for f in al}, index=ef.index)


def oof_member(m, fold):
    d = OOF[m]
    d = d[d.fold == fold]
    return None if d.empty else d.drop(columns="fold")


def gold_member(m, folds):
    g = GLD[m][GLD[m].fold.isin(folds)]
    return g.drop(columns="fold").groupby("StudyInstanceUID").mean()


def oof_parts(folds=(0, 1, 2)):
    """Per OOF study (gold excluded): base (3x3 mean, 12 findings), e0 (plain expert mean, OUTS), ef (fused + aligned, OUTS), fold."""
    B, E0, EF = [], [], []
    for f in folds:
        fz = fuse_files([f"oa_crop_v2cor_f{f}_oof.npz", f"oa_crop_v2cor_nfnet_f{f}_oof.npz"], [0, 1])
        ids = fz.index[~fz.index.isin(GOLD.index)]
        B.append((sum(oof_member(p, f).reindex(ids)[FINDINGS] for p in PIPES) / 3).assign(fold=f))
        E0.append(((oof_member("oa_b0", f).reindex(ids)[OUTS] + oof_member("oa_nf", f).reindex(ids)[OUTS]) / 2).assign(fold=f))
        EF.append(align(fz.loc[ids]).assign(fold=f))
    return pd.concat(B), pd.concat(E0), pd.concat(EF)


def gold_parts():
    gi = GOLD.index
    base = sum(gold_member(p, [0, 1, 2, 3, 4]).reindex(gi)[FINDINGS] for p in PIPES) / 3
    e0 = (gold_member("oa_b0", [0, 1, 2, 3, 4]).reindex(gi)[OUTS] + gold_member("oa_nf", [0, 1, 2]).reindex(gi)[OUTS]) / 2
    fz = fuse_files([f"oa_crop_v2cor_f{f}_gold.npz" for f in range(5)] + [f"oa_crop_v2cor_nfnet_f{f}_gold.npz" for f in range(3)],
                    [0] * 5 + [1] * 3)
    return base, e0, align(fz.reindex(gi))


def package_oa(base, e, rec=None):
    """OA columns of the package from base (3x3 logits, OA columns at least) and the expert term e (OA columns)."""
    return oa_recipe(base[OA], e[OA], rec or CFG["oa"]["recipe"])


# ---------------------------------------------------------------- LOA read panel
KEY = "lateral_oa"
LST = ST[KEY]
MST = ST["medial_oa"]
LMST = ST["lateral_meniscus"]
_EXT = {s: (r["findings"][KEY].get("extent"), r["findings"][KEY].get("depth")) for s, r in text_labels("v1").items()}
EXTENT = pd.Series({s: v[0] for s, v in _EXT.items()})
DEPTH = pd.Series({s: v[1] for s, v in _EXT.items()})
NEG = {"denied", "structure_normal"}
POSM = {"meets", "unsettled", "conflict"}       # other compartment 'diseased' for strata


def _auc(y, s, mn=5):
    y, s = np.asarray(y, int), np.asarray(s, float)
    return roc_auc_score(y, s) if y.sum() >= mn and (1 - y).sum() >= mn else np.nan


# Status-level soft-gold rates for LOA (gold-58 counts per v1 status shrunk 5 pseudo-cells toward the OA-pooled rate; contested cell
# dropped; meets split by depth). Weakly gold-derived (status level only, never per study): used only for the 'sg' read.
SG_RATE = {"meets_full": 0.95, "meets_deep": 0.67, "meets": 0.80, "unsettled": 0.43, "below": 0.26, "not_mentioned": 0.056,
           "denied": 0.01, "structure_normal": 0.011, "general_normal": 0.03, "conflict": 0.6, "hedged": 0.3, "excluded": 0.05}


def soft_auc(score, rate):
    """AUC against soft labels: every cell is a positive with weight r and a negative with weight 1 - r (pairs weighted r_i (1 - r_j))."""
    sc = np.r_[score, score]
    return roc_auc_score(np.r_[np.ones(len(score)), np.zeros(len(score))], sc, sample_weight=np.r_[rate, 1 - rate])


def _pair(score, pos_mask, neg_mask):
    i = score.index
    p, n = pos_mask.reindex(i).fillna(False).astype(bool), neg_mask.reindex(i).fillna(False).astype(bool)
    keep = p | n
    return _auc(p[keep], score[keep])


def loa_panel(s):
    """s: LOA logits (Series over studies). Report reads (v1 statuses; all are v1-derived, see VALIDATION.md):
    text = meets vs denied / structure_normal (common.metrics text); md = meets vs denied; thr = meets vs below;
    un = unsettled vs explicit negatives; bel = below vs explicit negatives; sil = not_mentioned vs explicit negatives (should be
    ~0.5-0.6: silent ~ negative); hard = meets focal vs explicit negatives; tfull = 'trusted-style' read: meets with full-thickness
    depth (134 non-gold cells; gold 3 / 3 positive, mirrors the MOA / PF trusted meets rule) vs the trusted negatives (= text's negatives); strat = mean text AUC inside MOA status strata
    (diseased / healthy); iso = LOA meets with MOA explicit-negative vs LOA explicit-negative with MOA meets (compartment test);
    lmneg = text AUC among knees with an explicit-negative lateral meniscus, lmpos among knees with LM meets (LM coupling);
    tsev = 'large or full-thickness' meets (depth full or extent diffuse / multifocal) vs explicit negatives and tord = those vs
    focal non-full meets: reads aligned with the host definition ('moderate or large area (~1 cm) of > 50% loss; borderline =
    negative'), which makes small focal defects (the 'hard' read) partly test-NEGATIVE;
    gold / gold_rb (contested ...855003 dropped)."""
    s = s.dropna()
    o = s[~s.index.isin(GOLD.index)]
    st = LST.reindex(o.index)
    neg = st.isin(NEG)
    r = {"n": len(o), "text": _pair(o, st.eq("meets"), neg), "md": _pair(o, st.eq("meets"), st.eq("denied")),
         "thr": _pair(o, st.eq("meets"), st.eq("below")), "un": _pair(o, st.eq("unsettled"), neg),
         "bel": _pair(o, st.eq("below"), neg), "sil": _pair(o, st.eq("not_mentioned"), neg),
         "hard": _pair(o, st.eq("meets") & EXTENT.reindex(o.index).eq("focal"), neg),
         "tfull": _pair(o, st.eq("meets") & DEPTH.reindex(o.index).eq("full"), neg)}
    ext, dep = EXTENT.reindex(o.index), DEPTH.reindex(o.index)
    large = st.eq("meets") & (dep.eq("full") | ext.isin(["diffuse", "multifocal"]))
    r["tsev"] = _pair(o, large, neg)                                             # host definition: >= ~1 cm area of > 50% loss
    r["tord"] = _pair(o, large, st.eq("meets") & ext.eq("focal") & dep.ne("full"))   # large / full meets above focal-deep meets
    key = st.where(~st.eq("meets"), "meets_" + dep.fillna("x")).replace({"meets_partial": "meets", "meets_x": "meets"})
    rate = key.map(SG_RATE)
    ok = rate.notna()
    r["sg"] = soft_auc(o[ok].to_numpy(), rate[ok].to_numpy())
    ms = MST.reindex(o.index)
    v = [_pair(o[g], st[g].eq("meets"), neg[g]) for g in (ms.isin(POSM), ms.isin(NEG))]
    r["strat"] = np.nanmean(v) if len(o) and not np.isnan(v).all() else np.nan
    r["iso"] = _pair(o, st.eq("meets") & ms.isin(NEG), neg & ms.eq("meets"))
    lm = LMST.reindex(o.index)
    r["lmneg"] = _pair(o[lm.isin(NEG)], st[lm.isin(NEG)].eq("meets"), neg[lm.isin(NEG)])
    r["lmpos"] = _pair(o[lm.eq("meets")], st[lm.eq("meets")].eq("meets"), neg[lm.eq("meets")])
    g = s[s.index.isin(GOLD.index)]
    if len(g):
        r["gold"] = _auc(GOLD.loc[g.index, LOA], g, 1)
        k = g.index[~g.index.isin(CONTESTED_LOA)]
        r["gold_rb"] = _auc(GOLD.loc[k, LOA], g[k], 1)
        k = g.index[~g.index.isin(DOMINANT_LOA)]
        r["gold_rb_loa"] = _auc(GOLD.loc[k, LOA], g[k], 1)
    return r


def panel_by_fold(s, fold):
    """loa_panel per fold (fold: Series study -> fold) and the mean over folds."""
    t = pd.DataFrame({f: loa_panel(s[fold.reindex(s.index) == f]) for f in sorted(fold.reindex(s.index).dropna().unique())}).T
    t.loc["mean"] = t.mean()
    return t


def gold_delta(a, b):
    """Gold-58 LOA delta b - a (x1e-3) with the leave-one-gold-study-out jackknife range and the largest single-study effect.
    a, b: LOA logits over gold studies. Returns dict d, d_rb (contested dropped), d_rb_loa (+ 559395 dropped), jk_lo, jk_hi, top_study, top_effect."""
    gi = GOLD.index.intersection(a.dropna().index).intersection(b.dropna().index)
    y = GOLD.loc[gi, LOA]
    d = (_auc(y, b[gi], 1) - _auc(y, a[gi], 1)) * 1e3
    k = gi[~gi.isin(CONTESTED_LOA)]
    d_rb = (_auc(y[k], b[k], 1) - _auc(y[k], a[k], 1)) * 1e3
    k = gi[~gi.isin(DOMINANT_LOA)]
    d_rb_loa = (_auc(y[k], b[k], 1) - _auc(y[k], a[k], 1)) * 1e3
    jk = {}
    for s in gi:
        kk = gi[gi != s]
        jk[s] = (_auc(y[kk], b[kk], 1) - _auc(y[kk], a[kk], 1)) * 1e3
    jk = pd.Series(jk)
    eff = d - jk                                       # how much each study contributes to the delta
    top = eff.abs().idxmax()
    return {"d": d, "d_rb": d_rb, "d_rb_loa": d_rb_loa, "jk_lo": jk.min(), "jk_hi": jk.max(), "top_study": top[-6:], "top_effect": eff[top]}
