"""Undo the floor that labelling session 3 put on its silent values (see v3_unsure.txt).

- 253 values that its script / sed raised to start - 10: restored to the session's own values, read from the session transcript
  (silent_check.py output printed just before the clamp, 2026-10-08 18:35 BST, and before the sed edit at 19:37 BST).
- 115 later silent values at exactly start - 10 (labelled after the clamp, index >= 2528): re-judged by reading each report,
  anchored on what sessions 0, 1, 2 and 4 gave in the same finding / setting / coverage / effusion situation.

Inputs: part3_fix_changes.csv (index, finding, before, after, source, why). Backs up every touched file to part3_before_fix/
before writing, then rewrites text_labeling_bundle_v3/out/<study_id>.json.

    conda run -n rsna python work/text_labeling/v2v3/fix_part3.py"""
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "text_labeling_bundle_v3/out"
BACKUP = HERE / "part3_before_fix"

changes = pd.read_csv(HERE / "part3_fix_changes.csv")
files = {json.loads(f.read_text(encoding="utf-8"))["index"]: f for f in OUT.glob("*.json")}
BACKUP.mkdir(exist_ok=True)
for i, rows in changes.groupby("index"):
    f = files[i]
    if not (BACKUP / f.name).exists():
        shutil.copy2(f, BACKUP / f.name)
    x = json.loads(f.read_text(encoding="utf-8"))
    for r in rows.itertuples():
        cell = x["findings"][r.finding]
        assert cell["evidence"] == "silent" and cell["p"] == r.before, (i, r.finding, cell)
        cell["p"] = int(r.after)
    f.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(changes)} values in {changes['index'].nunique()} files; changed {int((changes.before != changes.after).sum())}")
