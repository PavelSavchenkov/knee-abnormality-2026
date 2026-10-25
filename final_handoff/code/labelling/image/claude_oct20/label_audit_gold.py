"""Do the Claude (Opus) labels need fixing? Audit against gold-58 (USER 10-21 04:0x: "deep research of whether we need to fix
claude-generated labels ... judging from comparison and insights from gold"). Gold facts stay in work/ files and the chat: never memory,
never labelling sessions. OAI: no.
Parts:
  A. per class: label AUC vs gold (Opus, text v1, 0.5 / 0.5 blend); confusion at .5 by text report group.
  B. inside the mixed groups (Contusion 'meets' not high-certainty, PF 'below', Fracture misses): which Opus structured fields separate
     host positives from negatives (edema grades / bones / pattern; osteophytes / subchondral / narrowing / focal / extent).
  C. calibration: log-loss + Brier of the current status x certainty -> p mapping vs a LEAVE-ONE-STUDY-OUT per-bin host rate (an honest
     out-of-sample estimate of what a recalibrated mapping would gain) and vs the text blend.
  D. protocol sensitivity: the fast pilot pass vs the strict pass on the same gold studies; self-agreement of re-reads (repeats.json).
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/label_audit_gold.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table, status_table
from common.paths import REPO

D = Path(__file__).parent
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

G = gold_labels()
TP, TS = probability_table("v1"), status_table("v1")
KEY = {"Contusion": "contusion", "Fracture": "fracture", "PF OA": "pf_oa"}
out = []
say = lambda s="": out.append(str(s))  # noqa: E731


def load(dirs):
    rows = {}
    for d in dirs:
        for p in (D / d).glob("*.json"):
            x = json.load(open(p))
            if x["study_id"] in G.index:
                rows[x["study_id"]] = x["findings"]
    return rows


strict, fast = load(["pilot_c1", "pilot_c2"]), load(["pilot_c1_v1", "pilot_c2_v1"])


def p_of(v):
    return None if not v or v.get("status") in (None, "not_assessable") else probability(v["status"], v["certainty"])


ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(1 - np.clip(p, .01, .99))))  # noqa: E731
pd.set_option("display.width", 220)
for c, k in KEY.items():
    ids = [s for s in G.index if s in strict and k in strict[s]]
    y = G.loc[ids, c].astype(int)
    v = {s: strict[s][k] for s in ids}
    po = pd.Series({s: p_of(v[s]) for s in ids}, dtype=float)
    pt = TP[k].reindex(ids).fillna(0.15)
    grp = TS[k].reindex(ids).fillna("none")
    say(f"\n=== {c}: n {len(ids)}, host+ {int(y.sum())}")
    say(f"A. label AUC vs gold: Opus {roc_auc_score(y, po) * 1e3:.0f} | text {roc_auc_score(y, pt) * 1e3:.0f} | 0.5 / 0.5 blend {roc_auc_score(y, .5 * po + .5 * pt) * 1e3:.0f}")
    t = pd.DataFrame({"y": y, "opus_pos": po >= .5, "grp": grp})
    say("   confusion at .5 by report group (host+ / Opus+ / FP / FN):")
    for g, d in t.groupby("grp"):
        say(f"     {g:16s} n {len(d):2d} | host+ {int(d.y.sum()):2d} | Opus+ {int(d.opus_pos.sum()):2d} | FP {int((d.opus_pos & (d.y == 0)).sum())} | FN {int((~d.opus_pos & (d.y == 1)).sum())}")
    # B. fields inside the mixed groups
    f = pd.DataFrame({s: {kk: (json.dumps(vv) if isinstance(vv, (dict, list)) else vv) for kk, vv in v[s].items() if kk not in ("note", "evidence", "boxes")}
                      for s in ids}).T
    f["y"] = y
    if c == "Contusion":
        ed = pd.DataFrame({s: v[s].get("edema") or {} for s in ids}).T.fillna(0).astype(float)
        f["edema_max"], f["edema_bones"] = ed.max(1), (ed > 0).sum(1)
        m = (f.status == "meets")
        say("B. Opus 'meets': host+ rate by certainty / max oedema grade / bones with oedema:")
        for col_ in ("certainty", "edema_max", "edema_bones"):
            say("   " + col_ + ": " + ", ".join(f"{a}: {int(b['size'])} / {b['mean']:.2f}" for a, b in f[m].groupby(col_).y.agg(["size", "mean"]).iterrows()))
        pat = f[m].pattern.map(lambda x: ",".join(sorted(json.loads(x))) if isinstance(x, str) else "-")
        say("   pattern: " + ", ".join(f"{a}: {int(b['size'])} / {b['mean']:.2f}" for a, b in f[m].assign(pat=pat).groupby("pat").y.agg(["size", "mean"]).iterrows()))
    if c == "PF OA":
        m = (f.status == "below")
        say("B. Opus 'below': host+ rate by secondary signs:")
        for col_ in ("cartilage_loss", "extent_ge_10mm", "osteophytes", "subchondral_marrow", "narrowing", "focal", "certainty"):
            if col_ in f:
                say("   " + col_ + ": " + ", ".join(f"{a}: {int(b['size'])} / {b['mean']:.2f}" for a, b in f[m].groupby(f[m][col_].astype(str)).y.agg(["size", "mean"]).iterrows()))
        say("   ALL statuses by osteophytes: " + ", ".join(f"{a}: {int(b['size'])} / {b['mean']:.2f}" for a, b in f.groupby(f.osteophytes.astype(str)).y.agg(["size", "mean"]).iterrows()))
    if c == "Fracture":
        say("B. host fractures Opus missed (Opus < .5), by report group: " + ", ".join(f"{g}: {n}" for g, n in t[(t.y == 1) & ~t.opus_pos].grp.value_counts().items()))
        say("   Opus fracture 'meets' by type: " + ", ".join(f"{a}: {int(b['size'])} / {b['mean']:.2f}" for a, b in f[f.status == "meets"].groupby(f.type.astype(str)).y.agg(["size", "mean"]).iterrows()))
    # C. calibration, leave-one-study-out per (status, certainty) bin, shrunk to the status rate, then the overall rate
    b = (f.status.astype(str) + "|" + f.certainty.astype(str))
    loo = []
    for s in ids:
        tr = [x for x in ids if x != s]
        yb = y[tr]
        same = [x for x in tr if b[x] == b[s]]
        st = [x for x in tr if f.status[x] == f.status[s]]
        prior = (yb[st].sum() + 1 * yb.mean()) / (len(st) + 1) if st else yb.mean()
        loo.append((yb[same].sum() + 2 * prior) / (len(same) + 2))
    loo = pd.Series(loo, index=ids)
    say(f"C. calibration (log-loss / Brier; lower = better): current mapping {ll(y, po):.3f} / {np.mean((po - y) ** 2):.3f} | "
        f"LOO per-bin host rate {ll(y, loo):.3f} / {np.mean((loo - y) ** 2):.3f} | text {ll(y, pt):.3f} / {np.mean((pt - y) ** 2):.3f} | "
        f"blend {ll(y, .5 * po + .5 * pt):.3f} | LOO AUC {roc_auc_score(y, loo) * 1e3:.0f}")
    # D. fast vs strict pass
    both = [s for s in ids if s in fast and k in fast[s]]
    if both:
        pf_ = pd.Series({s: p_of(fast[s][k]) for s in both}, dtype=float)
        yy = G.loc[both, c].astype(int)
        say(f"D. protocol: fast pass vs strict pass on the same {len(both)} studies: AUC {roc_auc_score(yy, pf_) * 1e3:.0f} vs {roc_auc_score(yy, po[both]) * 1e3:.0f}; "
            f"same side of .5 {float(((pf_ >= .5) == (po[both] >= .5)).mean()):.2f}")
rep = json.load(open(REPO / "data/image_labels/repeats.json"))
main = {n: json.load(open(REPO / f"data/image_labels/{n}.json")) for n in ("opus_p2", "opus_p3")}
pairs = []
for r in rep:
    if r.get("run") != "opus_p3" or r.get("kind") != "protocol_reread":
        continue
    first = main["opus_p2"].get(r["study_id"])
    for k, v in r["findings"].items():
        if first and k in first["findings"] and v.get("probability") is not None and first["findings"][k].get("probability") is not None:
            pairs.append((k, first["findings"][k]["probability"], v["probability"]))
if pairs:
    q = pd.DataFrame(pairs, columns=["f", "p2", "p3"])
    say("\nD. self-agreement, p3 protocol re-reads of p2 studies: " + ", ".join(
        f"{f}: n {len(d)}, same side {((d.p2 >= .5) == (d.p3 >= .5)).mean():.2f}, corr {d.p2.corr(d.p3):.2f}" for f, d in q.groupby("f")))
(D / "label_audit_gold.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
