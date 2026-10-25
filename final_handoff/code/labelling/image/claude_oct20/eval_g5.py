"""Read the gold feasibility batch (g5_s1..s4: Opus images + report + native zoom on ACL / MCL / menisci / MOA / LOA, panel p5) on gold-58:
per finding AUC vs text v1, the 0.961 package (expected gold, all fold models), Codex img+rep, Opus image-only and the pilot (medial
meniscus only), on the same studies; 'fixes' = package-discordant gold pairs the reader orders correctly; in-sample blend z(pkg) +
0.5 z(reader) (INFO). Output: eval_g5.txt (never memory, never labeller messages: labeller blindness). OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/eval_g5.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

D = Path(__file__).parent
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402

lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa: E731
z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
KM = {"Baker's": "bakers_cyst"}
IL = REPO / "data/image_labels"
gold = gold_labels()
pkg = pd.read_csv(REPO / "experiments/v2/oct14_text_labels/data/expected_gold_f5_win.csv", index_col=0).reindex(gold.index)
text = probability_table("v1").reindex(gold.index)


def score(v):
    if not v or v.get("status") == "not_assessable":
        return None
    return probability(v["status"], v["certainty"])


def load_dirs(dirs):
    rows = {}
    for p in sorted(q for d in dirs for q in (D / d).glob("*.json")):
        r = json.loads(p.read_text())
        rows[r["study_id"]] = {f: score(r["findings"].get(KM.get(f, KEY_OF[f]))) for f in FINDINGS}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FINDINGS).astype(float)


def rd(name):
    d = json.load(open(IL / f"{name}.json"))
    return pd.DataFrame.from_dict({s: {f: (r["findings"].get(KM.get(f, KEY_OF[f])) or {}).get("probability") for f in FINDINGS} for s, r in d.items()},
                                  orient="index").reindex(columns=FINDINGS).astype(float)


def auc(y, s):
    m = s.notna() & y.notna()
    return roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else np.nan


me = load_dirs([f"g5_s{k}" for k in range(1, 5)])
pilot = load_dirs(["pilot_c1", "pilot_c2"])
cxr, op12 = rd("codex_report_image_gold"), rd("opus_v4_gold")
out = [f"g5 reads on gold: {int(me.index.isin(gold.index).sum())} of 58"]
for f in [c for c in FINDINGS if me[c].notna().any()]:
    s = me[f].reindex(gold.index)
    m = s.notna() & gold[f].notna()
    ys, ps, ss = gold[f][m], pkg[f][m], s[m]
    bad = [(p, q) for p in ys.index[ys == 1] for q in ys.index[ys == 0] if ps[p] < ps[q]]
    o = cxr[f].reindex(ys.index)
    out.append(f"{f:16s} n {int(m.sum())} n+ {int(ys.sum()):2d} | opus g5 {auc(ys, ss):.3f} | text {auc(ys, text[KEY_OF[f]][m].fillna(0.15)):.3f} | "
               f"pkg {auc(ys, ps):.3f} | blend {auc(ys, z(lg(ps)) + 0.5 * z(lg(ss))):.3f} | fixes {sum(ss[p] > ss[q] for p, q in bad)} / {len(bad)} | "
               f"codex img+rep {auc(ys[o.notna()], o.dropna()):.3f} (n {int(o.notna().sum())}) | opus img-only {auc(ys, op12[f].reindex(ys.index)):.3f} | "
               f"pilot {auc(ys, pilot[f].reindex(ys.index)):.3f}")
(D / "eval_g5.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
