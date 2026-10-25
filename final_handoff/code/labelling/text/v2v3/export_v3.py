"""Export the v3 forecast labels: text_labeling_bundle_v3/out/*.json -> data/text_labels/v3.json (keyed by study_id, index order).

Fails unless every part passes the bundle's validate.py and every index of reports.csv has a file. The labellers' objects are
written unchanged.

    conda run -n rsna python work/text_labeling/v2v3/export_v3.py"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "text_labeling_bundle_v3"
OUT = REPO / "data/text_labels/v3.json"

for k in range(5):
    r = subprocess.run([sys.executable, str(BUNDLE / "validate.py"), "--part", str(k)], capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1]
    if r.returncode or not last.endswith(" 0 still to do"):
        sys.exit(f"part {k} not complete or invalid: {last}")
reports = pd.read_csv(BUNDLE / "reports.csv", usecols=["index", "study_id"])
labels = {x["study_id"]: x for x in (json.loads(f.read_text(encoding="utf-8")) for f in (BUNDLE / "out").glob("*.json"))}
assert set(labels) == set(reports.study_id), "out/ does not hold exactly the studies of reports.csv"
ordered = {s: labels[s] for s in reports.sort_values("index").study_id}
OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(ordered)} studies -> {OUT}")
