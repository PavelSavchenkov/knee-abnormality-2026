"""Which form of the Claude (Opus) image + report reads is good on gold-58? (USER 10-20: "look at what setting image-based labels were good on
gold: blended with text labels? blended with our models predictions? anything else?"). OAI: no. Gold numbers stay in this folder.
Pilot strict reads (pilot_c1 / pilot_c2) of the gold studies; per finding (contusion / fracture / pf_oa + synovitis as the precedent check):
AUC on gold of TARGET forms (what a model could be trained on) and of TEST-TIME forms (package + reader; an upper bound if the reader existed
at test). Paired study bootstrap 90 % CI vs text v1; LOSO range.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/eval_target_forms.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table, status_table
from common.paths import REPO

sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

D = REPO / "work/claude_labels_oct20"
G = gold_labels()
text, st = probability_table("v1"), status_table("v1")
pkg = pd.read_csv(REPO / "experiments/v2/oct14_text_labels/data/expected_gold_f5_win.csv", index_col=0)
v3 = pd.read_csv(REPO / "experiments/v3/oct16_experts/data/tgt_v3.csv").set_index("StudyInstanceUID")
cal = pd.read_csv(D / "tgt_p2cal.csv").set_index("StudyInstanceUID")
F = {"contusion": "Contusion", "fracture": "Fracture", "pf_oa": "PF OA", "synovitis": "Synovitis"}
VAGUE = {"below", "unsettled", "hedged", "conflict", "excluded", "not_mentioned"}
lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))  # noqa: E731
z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
reads = {}
for p in list((D / "pilot_c1").glob("*.json")) + list((D / "pilot_c2").glob("*.json")):
    x = json.load(open(p))
    if x["study_id"] in G.index:
        reads[x["study_id"]] = {f: (None if v["status"] == "not_assessable" else probability(v["status"], v["certainty"])) for f, v in x["findings"].items()}
ids = sorted(reads)
rng = np.random.default_rng(0)
B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
lines = [f"Gold-58 studies with a pilot strict read: {len(ids)}. AUC x1e-3; [90 % paired bootstrap of the form minus text v1]; LOSO min..max of the same difference."]
for f, c in F.items():
    y = G.reindex(ids)[c].to_numpy()
    if np.isnan(y).any():
        keep = ~np.isnan(y)
    else:
        keep = np.ones(len(ids), bool)
    o = pd.Series({s: reads[s].get(f) for s in ids}, dtype=float)
    if o.isna().all():
        continue
    t = text.reindex(ids)[f].astype(float).fillna(.05)   # silent text (not_mentioned) -> 0.05 (Opus: ~0 % positive on silent random-stratum studies)
    s_ = st.reindex(ids)[f]
    vague = s_.isin(VAGUE) | s_.isna()
    pk = pkg.reindex(ids)[c].astype(float) if c in pkg.columns else None
    forms = {"text v1": t, "Opus": o.fillna(t), "0.25 Opus + 0.75 text": .25 * o.fillna(t) + .75 * t, "0.5 Opus + 0.5 text": .5 * o.fillna(t) + .5 * t,
             "0.75 Opus + 0.25 text": .75 * o.fillna(t) + .25 * t,
             "PARTIAL (Opus on vague / silent text, text elsewhere) = the p2 training scheme": t.where(~vague, o.fillna(t)),
             "PARTIAL 0.5 blend on vague, text elsewhere": t.where(~vague, .5 * o.fillna(t) + .5 * t)}
    if c in cal.columns:
        forms["calibrated text (tgt_p2cal, arm E)"] = cal.reindex(ids)[c].fillna(t)
    if c in v3.columns:
        tv = v3.reindex(ids)[c].astype(float).fillna(t)
        forms["tgt_v3 (bone experts' target)"] = tv
        forms["0.5 tgt_v3 + 0.5 Opus (arm D)"] = .5 * tv + .5 * o.fillna(tv)
    if pk is not None:
        forms["TEST-TIME: package (v2 / 0.961 expected gold)"] = pk
        for w in (.25, .5):
            forms[f"TEST-TIME: z(package) + {w} z(Opus)  [upper bound: needs a reader at test]"] = z(lg(pk)) + w * z(lg(o.fillna(pk)))
    yk = y[keep].astype(int)
    lines.append(f"\n== {c} (n {keep.sum()}, positives {int(yk.sum())})")
    base = t.to_numpy()[keep]
    for name, v in forms.items():
        v = v.to_numpy()[keep]
        a = roc_auc_score(yk, v)
        d = [roc_auc_score(yk[b], v[b]) - roc_auc_score(yk[b], base[b]) for b in B if 0 < yk[b].sum() < len(b)]
        lo = [roc_auc_score(np.delete(yk, i), np.delete(v, i)) - roc_auc_score(np.delete(yk, i), np.delete(base, i)) for i in range(len(yk))]
        lines.append(f"  {a * 1e3:6.1f}  d vs text {np.mean(d) * 1e3:+6.1f} [{np.percentile(d, 5) * 1e3:+6.1f}, {np.percentile(d, 95) * 1e3:+6.1f}]  LOSO {min(lo) * 1e3:+6.1f}..{max(lo) * 1e3:+6.1f}  {name}")
(D / "eval_target_forms.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
