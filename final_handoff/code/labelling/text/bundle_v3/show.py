"""Print report rows by index. Pure Python, no extra packages.

Usage: python3 show.py INDEX [INDEX ...]"""
import csv
import sys
from pathlib import Path

csv.field_size_limit(10**8)
with open(Path(__file__).resolve().parent / "reports.csv", encoding="utf-8", newline="") as fh:
    ROWS = {int(r["index"]): r for r in csv.DictReader(fh)}
if len(sys.argv) < 2:
    sys.exit(__doc__)
for i in map(int, sys.argv[1:]):
    r = ROWS[i]
    print(f"=== index {i} | study_id {r['study_id']} | knee_side {r['knee_side']} | language {r['language']}\n{r['report']}\n")
