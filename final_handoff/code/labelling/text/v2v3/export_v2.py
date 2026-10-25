"""Export the v2 fact labels: work/text_labeling/returned_v2/v2_FROM-TO.zip -> data/text_labels/v2.json (keyed by study_id,
index order).

Each archive is extracted into returned_v2/FROM-TO/ (a second run of the same range, "v2_FROM-TO (1).zip", into
FROM-TO_repeat1/; repeats are kept for consistency checks and not exported). Every range must pass the bundle's validate.py.
Missing ranges are listed and left out; rerun after adding their archives. The labellers' objects are written unchanged.

    conda run -n rsna python work/text_labeling/v2v3/export_v2.py"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RETURNED = REPO / "work/text_labeling/returned_v2"
BUNDLE = REPO / "text_labeling_bundle_v2"
OUT = REPO / "data/text_labels/v2.json"

ranges = {}
for z in sorted(RETURNED.glob("v2_*.zip")):
    m = re.fullmatch(r"v2_(\d+)-(\d+)(?: \((\d+)\))?\.zip", z.name)
    folder = RETURNED / (f"{m[1]}-{m[2]}" + (f"_repeat{m[3]}" if m[3] else ""))
    if not folder.exists():
        folder.mkdir()
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                assert re.fullmatch(r"[0-9.]+\.json", name), f"{z.name}: unexpected entry {name}"
                (folder / name).write_bytes(f.read(name))
    r = subprocess.run([sys.executable, str(BUNDLE / "validate.py"), str(folder), m[1], m[2]], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{folder.name} fails validation:\n{r.stdout[-2000:]}")
    if not m[3]:
        ranges[(int(m[1]), int(m[2]))] = folder

reports = pd.read_csv(BUNDLE / "reports.csv", usecols=["index", "study_id"]).sort_values("index")
n = len(reports)
missing = [(lo, min(lo + 49, n - 1)) for lo in range(0, n, 50) if (lo, min(lo + 49, n - 1)) not in ranges]
labels = {}
for folder in ranges.values():
    for f in folder.glob("*.json"):
        x = json.loads(f.read_text(encoding="utf-8"))
        assert x["study_id"] not in labels, f"duplicate study {x['study_id']}"
        labels[x["study_id"]] = x
ordered = {s: labels[s] for s in reports.study_id if s in labels}
OUT.write_text(json.dumps(ordered, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(ordered)} of {n} studies -> {OUT}")
print("missing ranges:", ", ".join(f"{lo}-{hi}" for lo, hi in missing) or "none")
