"""Label tables. All return DataFrames indexed by StudyInstanceUID.

text_labels(version)        raw LLM JSON (dict study -> record), version in {"v0", "v1"}
status_table(version)       study x KEYS, v1/v0 status strings
probability_table(version)  study x KEYS, v1 probability (NaN for not_mentioned; v0 has none)
gold_labels()               58 x FINDINGS, official labels (0/1)
external(name)              study x FINDINGS, public tables: steven_v4_blend, steven_v2, pilkwang_v2, lixin_sol56
eval_cells()                long table of hand-checked text cells: StudyInstanceUID, finding, label (0/1), verdict (trusted / look at)
                            rules: common/trusted_rules.csv (experiments/v0/oct7_baseline/analyses/levers, reviewed 2026-10-07)"""
import functools
import json
from pathlib import Path

import pandas as pd

from common.paths import FINDINGS, KEYS, TEXT_LABELS
from common.meta import studies

RULES = pd.read_csv(Path(__file__).with_name("trusted_rules.csv"), keep_default_na=False)
FILTERS = {
    "": lambda f, s: True,
    "grade low_grade": lambda f, s: f.get("grade") == "low_grade",
    "cause degeneration|fracture": lambda f, s: f.get("cause") in ("degeneration", "fracture"),
    "effusion denied": lambda f, s: s["findings"]["effusion"]["status"] == "denied",
}


@functools.cache
def text_labels(version="v1"):
    return json.load(open(TEXT_LABELS / f"{version}.json"))


@functools.cache
def status_table(version="v1"):
    t = text_labels(version)
    return pd.DataFrame([[t[s]["findings"][k]["status"] for k in KEYS] for s in t], index=list(t), columns=KEYS)


@functools.cache
def probability_table(version="v1"):
    t = text_labels(version)
    return pd.DataFrame([[t[s]["findings"][k].get("probability") for k in KEYS] for s in t], index=list(t), columns=KEYS, dtype=float)


@functools.cache
def gold_labels():
    st = studies()
    return st.loc[st[FINDINGS[0]].notna(), FINDINGS].astype(int)


@functools.cache
def external(name):
    df = pd.read_csv(TEXT_LABELS / "external" / f"{name}.csv").set_index("StudyInstanceUID")
    return df[FINDINGS].astype(float)


@functools.cache
def eval_cells():
    t = text_labels("v1")
    rows = []
    for r in RULES.itertuples():
        for sid, s in t.items():
            f = s["findings"][r.key]
            if f["status"] == r.status and FILTERS[r.filter](f, s):
                rows.append((sid, r.finding, int(r.label), r.verdict))
    cells = pd.DataFrame(rows, columns=["StudyInstanceUID", "finding", "label", "verdict"])
    assert not cells.duplicated(["StudyInstanceUID", "finding"]).any()
    return cells
