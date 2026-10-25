"""Alternative path (USER 10-21 03:5x: "try ensembling new models with their older counterparts to achieve credible gain on both
image-read-only and dense", CPU). Each swapped slot of the shipped bone crop fusion holds a PAIR (the shipped member + the new-label member,
averaged as one fusion group, so the slot keeps its share); bfs pairs = the mean of the two logits. NULL pair = shipped + the seed-1 old-label
control (the gain from simply averaging two models). Also a column blend: 0.5 z(shipped column) + 0.5 z(swapped column). Everything vs the
SHIPPED column; read-only Opus v3f decides, dense (v3f) = error catcher; paired bootstrap SE. x1e-3. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/pair_ensemble.py"""
import sys

import numpy as np
import pandas as pd

from common.labels import gold_labels
from common.paths import REPO

EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(EXP / "scripts"), str(EXP / "bone")]
import swap_read as sr  # noqa: E402
import wfuse  # noqa: E402
from c179_b_v3floc import BF, BW, CF, FC, FR, M, REC  # noqa: E402
from lib import soft_auc  # noqa: E402

gi = set(gold_labels().index)
YR = {c: pd.read_csv(EXP / "data/v3f/opus_raw.csv").set_index("StudyInstanceUID")[c].dropna() for c in ("Contusion", "Fracture")}
YD = {c: pd.read_csv(EXP / "data/v3f/tgt_bone_v3m.csv").set_index("StudyInstanceUID")[c].dropna() for c in ("Contusion", "Fracture")}
rng = np.random.default_rng(0)


def fused_groups(slots, f, set_):
    """slots: list of lists of npz paths (one list per slot); members of a slot are averaged as one fusion group."""
    paths = [p for s in slots for p in s]
    group = [k for k, s in enumerate(slots) for _ in s]
    Z = [dict(np.load(p)) for p in paths]
    ids = [str(v) for v in Z[0]["studies"]]
    for z in Z[1:]:
        assert [str(v) for v in z["studies"]] == ids and (z["meta"] == Z[0]["meta"]).all()
    off = np.r_[Z[0]["offsets"], len(Z[0]["s"])]
    e = np.array([wfuse.fuse(wfuse.prepare(np.stack([z["s"][a:b] for z in Z]).astype(np.float64), np.stack([z["u"][a:b] for z in Z]).astype(np.float64),
                                           group), FR["spec"]) for a, b in zip(off[:-1], off[1:])])
    e = pd.DataFrame(e, index=ids, columns=FC)
    return pd.DataFrame({c: (e[c] - FR["align"][c]["mu_f"]) / FR["align"][c]["sd_f"] * FR["align"][c]["sd_e"] + FR["align"][c]["mu_e"] for c in FC})


def ship(b, f, set_):
    return BF / "win" / f"{b}_f{f}_{set_}.npz"


def column(c, f, set_, pairs, bfs_pair=None):
    """pairs: {slot: [extra run, ...]} added to that slot's group next to the shipped member; bfs_pair: extra bfs run averaged in."""
    base = sum(M[(M.member == p) & (M.fold == f) & (M.set == set_)].groupby("StudyInstanceUID")[FC].mean() for p in ("main", "nfnet", "r50")) / 3
    mem = ["b_v3", "b_lr3"] + (["b_nf"] if f <= 2 else [])
    e = fused_groups([[ship(b, f, set_)] + [BW / f"{r}_f{f}_{set_}.npz" for r in pairs.get(b, [])] for b in mem], f, set_).reindex(base.index)
    s, w = REC[c]["stats"], REC[c]["w"]
    out = (1 - w) * base[c] + w * ((e[c] - s["mu_e"]) / s["sd_e"] * s["sd_b"] + s["mu_b"])
    if c == "Contusion":
        lb = sr.study_logit("bfs", f, set_)
        if bfs_pair:
            lb = (lb + sr.study_logit(bfs_pair, f, set_).reindex(lb.index)) / 2
        zb = (lb.reindex(out.index) - CF["mu_m"]) / CF["sd_m"] * CF["sd_d"] + CF["mu_d"]
        out = ((1 - CF["w"]) * out + CF["w"] * zb).fillna(out)
    return out


def oof(c, folds, fn):
    o = pd.concat([fn(f) for f in folds])
    return o[~o.index.isin(gi)]


def read(a, b, y, B=400):
    ids = a.index.intersection(b.index).intersection(y.index)
    pv, xa, xb = y[ids].to_numpy(), a[ids].to_numpy(), b[ids].to_numpy()
    d = (soft_auc(xa, pv) - soft_auc(xb, pv)) * 1e3
    bs = [(soft_auc(xa[i], pv[i]) - soft_auc(xb[i], pv[i])) * 1e3 for i in (rng.integers(0, len(pv), len(pv)) for _ in range(B))]
    return f"{d:+.1f} ({np.std(bs):.1f})"


z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
CASES = [   # (name, column, folds, ARM pairs, ARM bfs, NULL pairs, NULL bfs)
    ("C b_v3 pair v3e", "Contusion", [1, 2], {"b_v3": ["b3e_rep"]}, None, {"b_v3": ["b_v3_s1"]}, None),
    ("C b_v3 pair v3c", "Contusion", [0, 1, 2], {"b_v3": ["b3c_rep"]}, None, {"b_v3": ["b_v3_s1"]}, None),
    ("C all-4 pairs v3c", "Contusion", [0, 1, 2], {"b_v3": ["b3c_rep"], "b_lr3": ["b3l_rep"], "b_nf": ["b3n_rep"]}, "bfs_rep",
     {"b_v3": ["b_v3_s1"], "b_lr3": ["b3l_ctl"], "b_nf": ["b3n_ctl"]}, "bfs_s1"),
    ("C b_v3 + b_lr3 pairs v3c", "Contusion", [0, 1, 2], {"b_v3": ["b3c_rep"], "b_lr3": ["b3l_rep"]}, None,
     {"b_v3": ["b_v3_s1"], "b_lr3": ["b3l_ctl"]}, None),
    ("F all-3 pairs v3c", "Fracture", [0, 1, 2], {"b_v3": ["b3c_rep"], "b_lr3": ["b3l_rep"], "b_nf": ["b3n_rep"]}, None,
     {"b_v3": ["b_v3_s1"], "b_lr3": ["b3l_ctl"], "b_nf": ["b3n_ctl"]}, None),
]
rows = {}
for name, c, folds, ap, ab, npairs, nb in CASES:
    dep = oof(c, folds, lambda f: column(c, f, "oof", {}))
    arm = oof(c, folds, lambda f: column(c, f, "oof", ap, ab))
    nul = oof(c, folds, lambda f: column(c, f, "oof", npairs, nb))
    swap = oof(c, folds, lambda f: sr.column(f, "oof", {k: v[0] for k, v in ap.items()} | ({"bfs": ab} if ab else {}))[c])
    blend = z(dep) + z(swap.reindex(dep.index))
    for k, (x, ref) in {"PAIR vs SHIPPED": (arm, dep), "NULL pair vs SHIPPED": (nul, dep), "PAIR vs NULL pair": (arm, nul),
                        "SWAP (no pair) vs SHIPPED": (swap, dep), "COLUMN BLEND 0.5 vs SHIPPED": (blend, dep)}.items():
        rows[(name, k)] = {"read-only (decides)": read(x, ref, YR[c]), "dense (error catcher)": read(x, ref, YD[c])}
pd.set_option("display.width", 220)
print(f"OAI: no. PAIR ENSEMBLES (shipped + new-label member per slot) vs SHIPPED; Opus v3f; x1e-3\n" + pd.DataFrame(rows).T.to_string())
