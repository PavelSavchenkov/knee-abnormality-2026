"""Per-class yardstick choice (USER 10-21 02:2x: one label algorithm per class = training target AND held-out referee): every label source
we have, scored AS A LABEL SOURCE against gold-58 (all sources were made blind to gold), next to the report text v1 on the same studies.
Pool coverage = how many non-gold studies carry that source (a yardstick needs training-pool reads). Output yardstick_check.txt (work/
only: never memory, never labellers). OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/yardstick_check.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

D = Path(__file__).parent
IL = REPO / "data/image_labels"
G = gold_labels()
T = probability_table("v1").reindex(G.index)
KM = {"Baker's": "bakers_cyst"}


def src(name, f):
    d = json.load(open(IL / f"{name}.json"))
    k = KM.get(f, KEY_OF[f])
    return pd.Series({s: (r["findings"].get(k) or {}).get("probability") for s, r in d.items()}, dtype=float)


def mean_of(*names):
    return lambda f: pd.concat([src(n, f) for n in names], axis=1).mean(axis=1)


SOURCES = {   # finding -> {source name: loader}; pool = the run(s) that carry the same source on non-gold studies
    "Contusion": {"opus img+rep": lambda f: src("opus_pilot5_gold", f), "codex img+rep": lambda f: src("codex_report_image_gold", f)},
    "Fracture": {"opus img+rep": lambda f: src("opus_pilot5_gold", f), "codex img+rep": lambda f: src("codex_report_image_gold", f)},
    "PF OA": {"opus img+rep": lambda f: src("opus_pilot5_gold", f), "codex img+rep": lambda f: src("codex_report_image_gold", f)},
    "Synovitis": {"opus img+rep": lambda f: src("opus_pilot5_gold", f), "codex fluid": lambda f: src("codex_fluid_gold", f),
                  "opus fluid": lambda f: src("opus_fluid_gold", f), "3-reader fluid": mean_of("codex_fluid_gold", "opus_fluid_gold", "fable_fluid_gold")},
    "Effusion": {"codex fluid": lambda f: src("codex_fluid_gold", f), "opus fluid": lambda f: src("opus_fluid_gold", f),
                 "3-reader fluid (EF3)": mean_of("codex_fluid_gold", "opus_fluid_gold", "fable_fluid_gold")},
    "Baker's": {"codex bakers": lambda f: src("codex_bakers_gold", f), "codex fluid": lambda f: src("codex_fluid_gold", f),
                "opus fluid": lambda f: src("opus_fluid_gold", f), "codex + opus fluid": mean_of("codex_bakers_gold", "opus_fluid_gold")},
    "Lateral OA": {"opus g5 zoom": lambda f: src("opus_p5_gold", f), "codex lateral OA": lambda f: src("codex_lateral_oa_gold", f)},
}
for f in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA"):
    SOURCES[f] = {"opus g5 zoom": lambda f: src("opus_p5_gold", f), "codex img+rep": lambda f: src("codex_report_image_gold", f)}
SOURCES["Medial Meniscus"]["opus pilot"] = lambda f: src("opus_pilot5_gold", f)


def blend(a, b):
    return lambda f: 0.5 * a(f) + 0.5 * T[KEY_OF[f]].fillna(0.15)


SOURCES["PF OA"]["0.5 opus + 0.5 text"] = blend(SOURCES["PF OA"]["opus img+rep"], None)
out = ["label source vs gold-58 (AUC x1e-3; text on the same studies; pairs = gold pos-neg pairs the source orders right and text v1 does not / the reverse)"]
for f in FINDINGS:
    y0 = G[f].dropna()
    t0 = T[KEY_OF[f]].fillna(0.15)
    for name, load in SOURCES.get(f, {}).items():
        s = load(f).reindex(y0.index)
        m = s.notna()
        y, s, t = y0[m], s[m], t0[y0.index][m]
        if not 0 < y.sum() < len(y):
            continue
        pairs = [(p, q) for p in y.index[y == 1] for q in y.index[y == 0]]
        a = sum(s[p] > s[q] and not t[p] > t[q] for p, q in pairs)
        b = sum(t[p] > t[q] and not s[p] > s[q] for p, q in pairs)
        out.append(f"{f:16s} {name:22s} n {len(y):2d} n+ {int(y.sum()):2d} | source {roc_auc_score(y, s) * 1e3:5.0f} | text {roc_auc_score(y, t) * 1e3:5.0f} | "
                   f"pairs {a:3d} / {b:3d}")
(D / "yardstick_check.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
