"""Gold-example few-shot, cross-fitted on gold-58 (USER 10-19 ~19:0x). Gold-58 -> 3 folds stratified by Synovitis; for each fold k and
draw d (2 draws), 2 gold positives + 2 gold negatives per finding (Synovitis, Effusion) are drawn from the OTHER folds as in-context
examples, and fold k is read with them. Every gold study is read only with examples that are not itself.
Writes out/gshot_k{k}_d{d}_studies.csv (the fold's studies, gold view indexes 1000+) and out/gshot_k{k}_d{d}_shots.csv (index, finding,
answer), and out/gshot_examples.csv = the log of which gold studies served as examples where (exclude them from any later gold read of
models trained on these labels)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels

HERE = Path(__file__).parent
K, DRAWS, FINDINGS = 3, 2, ["Synovitis", "Effusion"]
g = pd.read_csv(HERE / "out/gold_studies.csv", index_col="index")
lab = gold_labels().loc[g.StudyInstanceUID]
lab.index = g.index
rng = np.random.default_rng(19)
fold = pd.Series(-1, index=g.index)
for y in (0, 1):                                    # stratified by Synovitis
    idx = rng.permutation(lab.index[lab.Synovitis == y])
    fold[idx] = np.arange(len(idx)) % K
log = []
for k in range(K):
    te, tr = fold.index[fold == k], fold.index[fold != k]
    for d in range(DRAWS):
        g.loc[te].to_csv(HERE / f"out/gshot_k{k}_d{d}_studies.csv")
        rows = []
        for f in FINDINGS:
            pos = rng.choice(tr[(lab.loc[tr, f] == 1).to_numpy()], 2, replace=False)
            neg = rng.choice(tr[(lab.loc[tr, f] == 0).to_numpy()], 2, replace=False)
            rows += [(pos[0], f, "Yes"), (neg[0], f, "No"), (pos[1], f, "Yes"), (neg[1], f, "No")]     # alternate Yes / No
        pd.DataFrame(rows, columns=["index", "finding", "answer"]).to_csv(HERE / f"out/gshot_k{k}_d{d}_shots.csv", index=False)
        log += [(k, d, f, i, g.StudyInstanceUID[i]) for i, f, _ in rows]
pd.DataFrame(log, columns=["fold", "draw", "finding", "index", "StudyInstanceUID"]).to_csv(HERE / "out/gshot_examples.csv", index=False)
print("fold sizes", fold.value_counts().sort_index().to_dict(), "| distinct example studies", len({r[3] for r in log}))
