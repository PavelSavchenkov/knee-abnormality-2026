"""Read the Claude image + report pilot (pilot_c1 / pilot_c2) on gold-58: per finding AUC of Opus img+rep vs text v1, the 0.961 package
(expected gold, all fold models), Codex img+rep (39 gold) and Opus image-only (30 gold), on the same studies; 'fixes' = package-discordant
gold pairs the reader orders correctly; in-sample blend z(pkg) + 0.5 z(reader) (INFO). Decoys are not scored. Output: eval_pilot.txt
(never memory: labeller blindness). OAI: no.  conda run -n rsna python work/claude_labels_oct20/eval_pilot.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

D = Path(__file__).parent
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa: E731
z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
KM = {"Baker's": "bakers_cyst"}
gold = gold_labels()
pkg = pd.read_csv(REPO / "experiments/v2/oct14_text_labels/data/expected_gold_f5_win.csv", index_col=0).reindex(gold.index)
text = probability_table("v1").reindex(gold.index)


import sys  # noqa: E402
sys.path.insert(0, str(REPO / "work/image_labeling"))
from mapping import probability  # noqa: E402  (status x certainty -> probability, the table every image-label set uses)


def score(v):
    if not v:
        return None
    if v.get("probability") is not None:
        return v["probability"]
    return None if v.get("status") == "not_assessable" else probability(v["status"], v["certainty"])


def load_dir(paths):
    rows = {}
    for p in paths:
        r = json.loads(p.read_text())
        rows[r["study_id"]] = {f: score(r["findings"].get(KM.get(f, KEY_OF[f]))) for f in FINDINGS}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FINDINGS).astype(float)


TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = the strict re-run (pilot_c1 / pilot_c2), "_v1" = the first fast pass
me = load_dir(sorted(p for c in ("pilot_c1", "pilot_c2") for p in (D / (c + TAG)).glob("*.json")))
IL = REPO / "data/image_labels"


def rd(name):
    d = json.load(open(IL / f"{name}.json"))
    return pd.DataFrame.from_dict({s: {f: (r["findings"].get(KM.get(f, KEY_OF[f])) or {}).get("probability") for f in FINDINGS} for s, r in d.items()},
                                  orient="index").reindex(columns=FINDINGS).astype(float)


cxr, op12 = rd("codex_report_image_gold"), rd("opus_v4_gold")


def auc(y, s):
    m = s.notna() & y.notna()
    return roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else np.nan


out = [f"pilot reads on gold: {int(me.index.isin(gold.index).sum())} of 58"]
for f in [c for c in FINDINGS if me[c].notna().any()]:
    y = gold[f]
    s = me[f].reindex(gold.index)
    m = s.notna()
    ys, ps, ss = y[m], pkg[f][m], s[m]
    pos, neg = ys[ys == 1].index, ys[ys == 0].index
    bad = [(p, q) for p in pos for q in neg if ps[p] < ps[q]]
    fx = sum(ss[p] > ss[q] for p, q in bad)
    bl = z(lg(ps)) + 0.5 * z(lg(ss))
    o = cxr[f].reindex(ys.index)
    out.append(f"{f:16s} n {int(m.sum())} n+ {int(ys.sum()):2d} | opus img+rep {auc(ys, ss):.3f} | text {auc(ys, text[KEY_OF[f]][m].fillna(0.15)):.3f} | "
               f"pkg {auc(ys, ps):.3f} | blend {auc(ys, bl):.3f} | fixes {fx} / {len(bad)} | codex img+rep {auc(ys[o.notna()], o.dropna()):.3f} "
               f"(n {int(o.notna().sum())}) | opus img-only {auc(ys, op12[f].reindex(ys.index)):.3f}")
(D / f"eval_pilot{TAG}.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
