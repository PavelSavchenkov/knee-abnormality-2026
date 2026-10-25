"""Viewing helper for the image + report pilot (display only).
conda run -n rsna python work/claude_labels_oct20/views.py INDEX --chat K
Empties work/claude_labels_oct20/_tmp_c<K>/, extracts the study's sheets (all it has) into _tmp_c<K>/<INDEX>/ and copies its report to
_tmp_c<K>/<INDEX>/report.txt. Prints the study_id and the sheets. Native full-resolution slices: work/image_labeling/view.py (--help),
written into the same temp folder. Run from the repo root."""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

D = Path("work/claude_labels_oct20")
p = argparse.ArgumentParser()
p.add_argument("index", type=int)
p.add_argument("--chat", type=int, required=True)
a = p.parse_args()
tmp = D / f"_tmp_c{a.chat}"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir()
with open(D / "_prompts/studies.csv", newline="") as fh:
    row = next(r for r in csv.DictReader(fh) if int(r["index"]) == a.index)
folder = f"{a.index:04d}"
sheets = row["sheets"].split(";")
subprocess.run(["unzip", "-o", "-q", f"image_labeling_bundle/{row['images_zip']}"] + [f"{folder}/{s}.jpg" for s in sheets] + ["-d", str(tmp)],
               check=True)
shutil.copyfile(D / f"_reports/{folder}.txt", tmp / folder / "report.txt")
print("study_id", row["study_id"], "| sheets", sheets, "| report", tmp / folder / "report.txt")
