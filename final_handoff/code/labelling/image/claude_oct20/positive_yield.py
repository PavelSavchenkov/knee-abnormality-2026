"""Where are the cheapest new Opus positives? (USER 10-20 17:1x: "plan to spend the least amount of tokens per X new positives").
Fits, per finding, P(Opus p >= .5) on the 811 p2 reads from the text v1 status (grouped) + the 0.961 package's OOF logit (main / NFNet /
R50 mean), then scores every unread non-gold study with a report. Prints the yield curve: top-k studies by expected positives -> expected
new positives, split into INFORMATIVE ones (text status not "meets": the label adds information) and confirmations. Per-study panels =
label a study only for the findings with P >= a cut. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/positive_yield.py"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from common.labels import status_table
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
F = {"contusion": "Contusion", "fracture": "Fracture", "pf_oa": "PF OA"}
GROUP = {"meets": "meets", "unsettled": "vague", "hedged": "vague", "below": "vague", "conflict": "vague", "excluded": "excluded",
         "not_mentioned": "silent", "denied": "neg", "structure_normal": "neg", "general_normal": "neg"}
st = status_table("v1")
mem = pd.read_parquet(REPO / "experiments/v3/oct16_experts/refs/pkg961/members.parquet")
mo = mem[(mem.set == "oof") & mem.member.isin(["main", "nfnet", "r50"])]
oof = mo.groupby(["StudyInstanceUID", "member"])[list(F.values())].mean().groupby(level=0).mean()   # mean logit over members
reads = json.load(open(REPO / "data/image_labels/opus_p2.json"))
pool = pd.read_csv(D / "_prompts/studies.csv")
pool = pool[pool.sheets.str.count(";") >= 7].drop_duplicates("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report", "ACL"]).set_index("StudyInstanceUID")
gold = set(meta.index[meta.ACL.notna()])
has_rep = set(meta.index[meta.Report.fillna("").str.strip() != ""])
used = set(json.load(open(REPO / "data/image_labels/opus_pilot5_gold.json"))) | set(reads)
unread = [s for s in pool.study_id if s not in gold and s in has_rep and s not in used and s in oof.index]
print(f"read {len(reads)} | unread non-gold with report + OOF: {len(unread)}")


def feats(ids, f):
    g = st.reindex(ids)[f].map(GROUP).fillna("silent")
    X = pd.get_dummies(g).reindex(columns=["meets", "vague", "excluded", "silent", "neg"], fill_value=0).astype(float)
    X["oof"] = oof.loc[ids, F[f]].to_numpy()
    for k in ("meets", "vague", "excluded"):
        X[f"oof_x_{k}"] = X["oof"] * X[k]
    return X, g


P, G = {}, {}
for f, c in F.items():
    ids = [s for s in reads if reads[s]["findings"][f]["probability"] is not None and s in oof.index]
    y = np.array([reads[s]["findings"][f]["probability"] >= .5 for s in ids]).astype(int)
    X, g = feats(ids, f)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    cv = cross_val_predict(lr, X, y, cv=5, method="predict_proba")[:, 1]
    lr.fit(X, y)
    print(f"\n== {c}: {y.sum()} positives in {len(y)} reads; CV AUC of the yield model {roc_auc_score(y, cv):.3f}")
    print("  Opus positive rate by text group (reads):", {k: f"{y[g.to_numpy() == k].sum()}/{(g == k).sum()}" for k in ["meets", "vague", "excluded", "silent", "neg"]})
    Xu, gu = feats(unread, f)
    P[f], G[f] = pd.Series(lr.predict_proba(Xu)[:, 1], index=unread), gu.set_axis(unread)
    p, inf = P[f], G[f] != "meets"
    print("  unread pool by text group: n / expected positives:", {k: f"{(gu == k).sum()} / {p[gu.to_numpy() == k].sum():.0f}" for k in ["meets", "vague", "excluded", "silent", "neg"]})
    o = p[inf].sort_values(ascending=False)
    cum = o.cumsum()
    marks = [(k, cum.iloc[k - 1]) for k in (50, 100, 200, 300, 500) if k <= len(o)]
    print("  INFORMATIVE (text not meets), best-first: studies -> expected positives:", ", ".join(f"{k} -> {v:.0f}" for k, v in marks),
          f"| total available {o.sum():.0f}")
    for X_ in (25, 50, 100):
        k = int(np.searchsorted(cum.to_numpy(), X_)) + 1
        print(f"    +{X_} informative positives need the top {k} studies (yield {X_ / k:.2f} per study)" if cum.iloc[-1] >= X_ else f"    +{X_}: not available")
# per-study panels: one read covers every finding whose P >= cut
print("\n== per-study panels (label only the findings with P >= cut; informative findings only)")
A = pd.DataFrame({f: P[f].where(G[f] != "meets", 0) for f in F})
for cut in (.2, .3, .4):
    sel = A[(A >= cut).any(axis=1)]
    exp = {F[f]: sel[f].where(sel[f] >= cut, 0).sum() for f in F}
    nf = (sel >= cut).sum(axis=1)
    print(f"  cut {cut}: {len(sel)} studies, {nf.sum()} finding-reads (mean {nf.mean():.2f} per study); expected informative positives "
          + ", ".join(f"{k} {v:.0f}" for k, v in exp.items()))
out = pd.DataFrame({f"p_{f}": P[f] for f in F}).assign(**{f"grp_{f}": G[f] for f in F})   # raw P (meets included)
print("\n== text 'meets' studies best-first (confirmations for C / F; for PF half are corrections)")
for f in F:
    m = P[f][G[f] == "meets"].sort_values(ascending=False)
    if len(m):
        print(f"  {F[f]}: " + ", ".join(f"top {k} -> {m.iloc[:k].sum():.0f}" for k in (50, 100, 150, 300, len(m)) if k <= len(m)))
out.to_csv(D / "positive_yield_pool.csv")
print(f"\nwrote {D / 'positive_yield_pool.csv'} ({len(out)} unread studies)")
