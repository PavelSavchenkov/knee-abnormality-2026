"""One-time gold-58 check of VLM reads (never used to tune anything). Per finding (Synovitis, Effusion): AUC on the official gold labels
of the VLM read alone, the Codex fluid-panel gold read alone (data/image_labels/codex_fluid_gold.json), the 0.961 package (expected gold,
refs/pkg961), and the blends z(package) + w * z(VLM), w in {0.25, 0.5}, plus Codex + VLM (the two-reader label idea). Bootstrap 90 %
intervals for the VLM-vs-Codex and blend-vs-package differences. z is label-free on the gold set itself.
NOTE: the shipped Synovitis form is the cnx form, not the 0.961 package; the package base here is an approximation for Synovitis.
Usage: python gold_check.py SCORES.csv [SCORES2.csv ...]   (several files = mean of z(logit) over configs)"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import gold_labels
from common.paths import REPO

gold = gold_labels()
pkg = pd.read_csv(REPO / "experiments/v3/oct16_experts/refs/pkg961/expected_gold_f5_win.csv", index_col="StudyInstanceUID")
cx = json.load(open(REPO / "data/image_labels/codex_fluid_gold.json"))
codex = pd.DataFrame({k: {"Synovitis": v["findings"]["synovitis"]["probability"], "Effusion": v["findings"]["effusion"]["probability"]}
                      for k, v in cx.items()}).T.astype(float)
z = lambda s: (s - s.mean()) / s.std()
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
reads = [pd.read_csv(p).pivot_table(index="StudyInstanceUID", columns="finding", values="p_yes") for p in sys.argv[1:]]
rng = np.random.default_rng(0)
for f in [f for f in ["Synovitis", "Effusion"] if all(f in r for r in reads)]:
    v = pd.concat([z(lg(r[f])) for r in reads], axis=1).mean(axis=1)
    d = pd.DataFrame({"y": gold[f], "v": v, "c": lg(codex[f]), "p": lg(pkg[f])}).dropna()
    y = d.y.to_numpy(int)
    sc = {"VLM": z(d.v), "Codex": z(d.c), "package": z(d.p), "pkg+0.25VLM": z(d.p) + 0.25 * z(d.v), "pkg+0.5VLM": z(d.p) + 0.5 * z(d.v),
          "Codex+VLM": z(d.c) + z(d.v)}
    auc = {k: roc_auc_score(y, s) for k, s in sc.items()}
    bs = {k: [] for k in ["VLM-Codex", "pkg+0.25VLM-package", "Codex+VLM-Codex"]}
    for _ in range(2000):
        b = rng.integers(0, len(y), len(y))
        if 0 < y[b].sum() < len(b):
            a = {k: roc_auc_score(y[b], s.to_numpy()[b]) for k, s in sc.items()}
            bs["VLM-Codex"].append(a["VLM"] - a["Codex"])
            bs["pkg+0.25VLM-package"].append(a["pkg+0.25VLM"] - a["package"])
            bs["Codex+VLM-Codex"].append(a["Codex+VLM"] - a["Codex"])
    print(f"{f}: n+ {y.sum()}/{len(y)}  " + "  ".join(f"{k} {a:.3f}" for k, a in auc.items()))
    print("   " + "  ".join(f"{k} 90% [{np.quantile(x, 0.05):+.3f}, {np.quantile(x, 0.95):+.3f}]" for k, x in bs.items()))
