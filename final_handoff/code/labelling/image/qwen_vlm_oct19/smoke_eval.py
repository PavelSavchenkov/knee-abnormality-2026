"""Smoke test of evaluate.py: a fake score file with p_yes = sigmoid(0.961 OOF) (vlm columns should equal oof columns)."""
import numpy as np
import pandas as pd

from common.paths import FINDINGS, REPO

oof = pd.read_parquet(REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet").loc["f5_win"]
st = pd.read_csv("work/vlm_label_2026-10-19/out/pilot_studies.csv")
rows = [(i, u, f, 1 / (1 + np.exp(-oof.loc[u, f])), 0) for i, u in zip(st["index"], st.StudyInstanceUID) for f in FINDINGS]
pd.DataFrame(rows, columns=["index", "StudyInstanceUID", "finding", "p_yes", "tokens"]).to_csv("work/vlm_label_2026-10-19/out/smoke_scores.csv", index=False)
