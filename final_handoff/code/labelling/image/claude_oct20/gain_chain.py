"""Where does a new-label member's gain go between the single model and the shipped column? (USER 10-21 01:3x: "if each model is better
it is reasonable to expect ensemble to be better ... investigate where we lose gains. maybe we just need to re-fit ensemble coefs/fusion?")
Same ARM / CTRL specs as scripts/swap_read.py (seed-1 controls = same recipe, old targets). Rows, each ARM - CTRL on the held-out folds:
  alone <slot>        the member by itself (bone: its own crop fusion; bfs: its study logit)
  swap <slot>         only that slot swapped in the shipped column (shipped constants)
  sum of swaps        sum of the single-slot rows (additivity check vs 'swap all')
  bone fusion         the 3-expert crop fusion alone (no 3x3 base, no cont_fs)
  swap all            every slot swapped in the shipped column (= swap_read.py's read)
  refit all           cross-fit weights (fit on the other fold(s), score the held-out one; non-negative + intercept, soft BCE on Opus
                      w1-2) over pipelines / each expert's fusion / the 3-expert fusion / bfs, fitted separately for ARM and CTRL
Readers: Opus w1-2 (+ paired bootstrap SE), Opus w1-2 positives split by their report text (v1 meets / not) vs all Opus negatives, text,
host-like, robust gold (fold-matched models). Opus = the arms' training reads (FITTED-TO, held-out fold only). x1e-3. OAI: no. CPU.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/gain_chain.py [--arm SPEC] [--ctrl SPEC] [--folds 1,2] [--col Contusion]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.paths import KEY_OF, REPO

EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(EXP / "scripts"), str(EXP / "bone")]
import c188_item_screen as c188  # noqa: E402
import swap_read as sr  # noqa: E402
from main_swap_read import load as load_main  # noqa: E402
from c179_b_v3floc import BF, BW, CF, M, REC, fused  # noqa: E402
from common.labels import gold_labels, status_table  # noqa: E402
from lib import soft_auc  # noqa: E402
from retune_xfit import fit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="b_v3=b3c_rep,b_lr3=b3l_rep,b_nf=b3n_rep,bfs=bfs_rep")
ap.add_argument("--ctrl", default="b_v3=b_v3_s1,b_lr3=b3l_ctl,b_nf=b3n_ctl,bfs=bfs_s1")
ap.add_argument("--folds", default="1,2")
ap.add_argument("--col", default="Contusion")
ap.add_argument("--opus", default="v3c")
ap.add_argument("--waves", default="1,2", help="Opus waves read (3 = the Phase 3 reads, v3d only)")
ap.add_argument("--main", default="", help="ARM:CTRL main-pipeline runs (e.g. m_p2part_pn:refs/main/sub_b0_s1): rows 'add main w' = z(shipped column) + w z(main Contusion) for ARM vs CTRL (the shipped Contusion gives the generalists weight 0)")
ap.add_argument("--dense", default="", help="dense yardstick CSV (experiment-relative, e.g. data/v3e/tgt_bone_v3m.csv): the referee = this target on ALL held-out studies (new label where read, old elsewhere) instead of the read studies only")
ap.add_argument("--boot", type=int, default=500)
a = ap.parse_args()
FOLDS, c = [int(f) for f in a.folds.split(",")], a.col
SA, SC = sr.spec(a.arm), sr.spec(a.ctrl)
assert set(SA) == set(SC)
G = gold_labels()
gi = set(G.index)
op = pd.read_csv(EXP / f"data/{a.opus}/opus_raw.csv").set_index("StudyInstanceUID")
op = op.loc[op.wave.isin([int(w) for w in a.waves.split(",")]), c]
if a.dense:
    op = pd.read_csv(EXP / a.dense).set_index("StudyInstanceUID")[c]
text = status_table("v1")[KEY_OF[c]]


def bone(paths):
    return fused(paths)[c]


def slot_path(b, run, f, set_):
    return BW / f"{run}_f{f}_{set_}.npz" if run else BF / "win" / f"{b}_f{f}_{set_}.npz"


def alone(b, run, f, set_):
    return sr.study_logit(run, f, set_) if b == "bfs" else bone([slot_path(b, run, f, set_)])


def components(sw, f, set_):
    """Per-study refit components of fold f's models with the slots in sw swapped (others shipped)."""
    d = {f"pipe_{p}": M[(M.member == p) & (M.fold == f) & (M.set == set_)].groupby("StudyInstanceUID")[c].mean() for p in ("main", "nfnet", "r50")}
    mem = ["b_v3", "b_lr3"] + (["b_nf"] if f <= 2 else [])
    paths = [slot_path(b, sw.get(b), f, set_) for b in mem]
    d.update({f"exp_{b}": bone([p]) for b, p in zip(mem, paths)})
    d["exp_fused"] = bone(paths)
    if c == "Contusion":
        d["bfs"] = sr.study_logit(sw.get("bfs", "bfs"), f, set_)
    return pd.DataFrame(d).dropna()


def refit(sw):
    """Cross-fit: weights fitted on the other fold(s)' OOF (Opus w1-2 targets), applied to the held-out fold's OOF and gold outputs."""
    X = {f: components(sw, f, "oof") for f in FOLDS}
    X = {f: x[~x.index.isin(gi)] for f, x in X.items()}
    oof, gold = [], []
    for f in FOLDS:
        trn = pd.concat([X[g] for g in FOLDS if g != f])
        keep = [k for k in trn.columns if trn[k].notna().all() and k in X[f].columns]   # b_nf crops exist on f0-2 only
        trn = trn[keep]
        mu, sd = trn.mean(), trn.std()
        y = op.reindex(trn.index).dropna()
        p = fit(((trn.loc[y.index] - mu) / sd).to_numpy(), y.to_numpy())
        oof.append(pd.Series(((X[f][keep] - mu) / sd).to_numpy() @ p[:-1] + p[-1], index=X[f].index))
        xg = components(sw, f, "gold")[keep]
        gold.append(pd.Series(((xg - mu) / sd).to_numpy() @ p[:-1] + p[-1], index=xg.index).reindex(G.index))
    return pd.concat(oof), sum(gold) / len(FOLDS)


def form(kind, slot=None):
    """(oof, gold) per study for ARM and CTRL under one form."""
    out = []
    for S in (SA, SC):
        if kind == "refit":
            out.append(refit(S))
            continue
        o, g = [], []
        for f in FOLDS:
            if kind == "alone":
                o.append(alone(slot, S[slot], f, "oof")), g.append(alone(slot, S[slot], f, "gold").reindex(G.index))
            elif kind == "bone":
                mem = [b for b in ("b_v3", "b_lr3", "b_nf") if b != "b_nf" or f <= 2]
                o.append(bone([slot_path(b, S.get(b), f, "oof") for b in mem])), g.append(bone([slot_path(b, S.get(b), f, "gold") for b in mem]).reindex(G.index))
            else:
                sw = {slot: S[slot]} if kind == "swap" else S
                o.append(sr.column(f, "oof", sw)[c]), g.append(sr.column(f, "gold", sw)[c].reindex(G.index))
        o = pd.concat(o)
        out.append((o[~o.index.isin(gi)], sum(g) / len(FOLDS)))
    return out


def realigned(sw):
    """The shipped column with its label-free constants re-fitted on this member set's own pooled OOF (held-out folds read): the bone
    fusion's mean / sd (the c179 mu_e / sd_e map) and, for Contusion, the cont_fs member's mu_m / sd_m. Weights unchanged; no labels."""
    E, B, Z = {}, {}, {}
    for f in FOLDS:
        mem = ["b_v3", "b_lr3"] + (["b_nf"] if f <= 2 else [])
        for set_ in ("oof", "gold"):
            B[f, set_] = sum(M[(M.member == q) & (M.fold == f) & (M.set == set_)].groupby("StudyInstanceUID")[c].mean() for q in ("main", "nfnet", "r50")) / 3
            E[f, set_] = bone([slot_path(b, sw.get(b), f, set_) for b in mem]).reindex(B[f, set_].index)
            Z[f, set_] = sr.study_logit(sw.get("bfs", "bfs"), f, set_).reindex(B[f, set_].index) if c == "Contusion" else None
    eo = pd.concat([E[f, "oof"] for f in FOLDS])
    s, w = REC[c]["stats"], REC[c]["w"]
    out = {}
    for set_ in ("oof", "gold"):
        o = []
        for f in FOLDS:
            x = (1 - w) * B[f, set_] + w * ((E[f, set_] - eo.mean()) / eo.std() * s["sd_b"] + s["mu_b"])
            if c == "Contusion":
                zo = pd.concat([Z[g, "oof"] for g in FOLDS])
                zb = (Z[f, set_] - zo.mean()) / zo.std() * CF["sd_d"] + CF["mu_d"]
                x = ((1 - CF["w"]) * x + CF["w"] * zb).fillna(x)
            o.append(x)
        out[set_] = pd.concat(o) if set_ == "oof" else sum(q.reindex(G.index) for q in o) / len(FOLDS)
    return out["oof"][~out["oof"].index.isin(gi)], out["gold"]


rng = np.random.default_rng(0)


def reads(arm, ctrl):
    (oa, ga), (oc, gc) = arm, ctrl
    ids = oa.index.intersection(oc.index)
    s = op.reindex(ids).dropna()
    pv, xa, xc = s.to_numpy(), oa[s.index].to_numpy(), oc[s.index].to_numpy()
    d = (soft_auc(xa, pv) - soft_auc(xc, pv)) * 1e3
    se = np.std([(soft_auc(xa[i], pv[i]) - soft_auc(xc[i], pv[i])) * 1e3 for i in (rng.integers(0, len(pv), len(pv)) for _ in range(a.boot))])
    r = {"Opus (SE)": f"{d:+.1f} ({se:.1f})"}
    neg = s.index[s < .5]
    for name, pos in (("pos text-meets", s.index[(s >= .5) & text.reindex(s.index).eq("meets")]),
                      ("pos text-not", s.index[(s >= .5) & ~text.reindex(s.index).eq("meets")])):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        r[f"Opus {name} (n)"] = f"{(roc_auc_score(y, oa[pos.append(neg)]) - roc_auc_score(y, oc[pos.append(neg)])) * 1e3:+.1f} ({len(pos)})"
    tm = text.reindex(ids)
    m = tm.isin(["meets", "denied", "structure_normal"]).to_numpy()
    yt, ta_, tc_ = tm[m].eq("meets").to_numpy().astype(int), oa[ids].to_numpy()[m], oc[ids].to_numpy()[m]
    gk = ga.dropna().index.intersection(gc.dropna().index)
    yg, ga_, gc_ = G.loc[gk, c].to_numpy().astype(int), ga[gk].to_numpy(), gc[gk].to_numpy()
    for name, y, xa_, xc_ in (("text", yt, ta_, tc_), ("gold", yg, ga_, gc_)):
        bs = []
        for _ in range(a.boot):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                bs.append((roc_auc_score(y[i], xa_[i]) - roc_auc_score(y[i], xc_[i])) * 1e3)
        r[f"{name} (SE)"] = f"{(roc_auc_score(y, xa_) - roc_auc_score(y, xc_)) * 1e3:+.1f} ({np.std(bs):.1f})"
    ta, _ = c188.readers(c, oa[ids], ga)
    tc, _ = c188.readers(c, oc[ids], gc)
    for k in ("host-like", "robust gold"):
        r[k] = f"{ta[k] - tc[k]:+.1f}" if pd.notna(ta.get(k)) else ""
    r["_d"] = d
    return r


rows = {}
for b in SA:
    rows[f"alone {b} ({SA[b]})"] = reads(*form("alone", b))
for b in SA:
    rows[f"swap {b}"] = reads(*form("swap", b))
rows["sum of swaps (Opus)"] = {"Opus (SE)": f"{sum(rows[f'swap {b}']['_d'] for b in SA):+.1f}"}
rows["bone fusion (3 experts)"] = reads(*form("bone"))
rows["swap all (shipped form)"] = reads(*form("swap_all"))
if len(FOLDS) > 1:
    RA, RC = form("refit")
    dep = []
    for f in FOLDS:
        dep.append((sr.column(f, "oof", {})[c], sr.column(f, "gold", {})[c].reindex(G.index)))
    DEP = (pd.concat([o for o, _ in dep]).pipe(lambda s: s[~s.index.isin(gi)]), sum(g for _, g in dep) / len(FOLDS))
    SWA, SWC = form("swap_all")
    rows["refit ARM vs refit CTRL"] = reads(RA, RC)
    rows["seed row: swap CTRL vs shipped"] = reads(SWC, DEP)
    rows["seed row: refit CTRL vs shipped"] = reads(RC, DEP)
    rows["deploy: swap ARM vs shipped"] = reads(SWA, DEP)
    rows["deploy: refit ARM vs shipped"] = reads(RA, DEP)
    LA, LC, LD = realigned(SA), realigned(SC), realigned({})
    rows["realign ARM vs realign CTRL"] = reads(LA, LC)
    rows["realign shipped vs shipped"] = reads(LD, DEP)
    rows["deploy: realign ARM vs shipped"] = reads(LA, DEP)
if a.main:
    zf = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    dep = {f: (sr.column(f, "oof", {})[c], sr.column(f, "gold", {})[c]) for f in FOLDS}
    mo = {r: {f: load_main(r, f) for f in FOLDS} for r in a.main.split(":")}

    def add_main(run, w):
        o, g = [], []
        for f in FOLDS:
            (xo, xg), (do, dg) = mo[run][f], dep[f]
            mu, sd = do.mean(), do.std()
            ids = do.index.intersection(xo[c].index)
            o.append((do[ids] - mu) / sd + w * zf(xo[c][ids]))
            g.append(((dg - mu) / sd + w * (xg[c] - xo[c].mean()) / xo[c].std()).reindex(G.index))
        o = pd.concat(o)
        return o[~o.index.isin(gi)], sum(g) / len(FOLDS)
    ra, rc = a.main.split(":")
    rows[f"alone main ({ra})"] = reads(*[(pd.concat([mo[r][f][0][c] for f in FOLDS]).pipe(lambda s: s[~s.index.isin(gi)]),
                                          sum(mo[r][f][1][c].reindex(G.index) for f in FOLDS) / len(FOLDS)) for r in (ra, rc)])
    DEPM = (pd.concat([dep[f][0] for f in FOLDS]).pipe(lambda s: s[~s.index.isin(gi)]), sum(dep[f][1].reindex(G.index) for f in FOLDS) / len(FOLDS))
    for w in (0.1, 0.25, 0.5, 1.0):
        rows[f"add main w {w}: ARM vs CTRL"] = reads(add_main(ra, w), add_main(rc, w))
        rows[f"add main w {w}: ARM vs SHIPPED"] = reads(add_main(ra, w), DEPM)
t = pd.DataFrame(rows).T.drop(columns="_d").fillna("")
pd.set_option("display.width", 250)
print(f"OAI: no. GAIN CHAIN {c}: ARM {SA} vs CTRL {SC}; folds {FOLDS}; referee {a.dense or f"Opus {a.opus} waves {a.waves}"} (FITTED-TO, held-out); x1e-3\n{t.to_string()}")
