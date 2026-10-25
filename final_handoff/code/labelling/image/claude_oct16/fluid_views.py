"""Fluid-panel viewing helper (display only): extract a study's fluid sheets and render the fixed native protocol with view.py.

conda run -n rsna python work/claude_labels_oct16/fluid_views.py INDEX --chat K [--sag SID] [--ax SID]
Writes _tmp_c<K>/<idx>/ sheets and _tmp_c<K>/v_sag.png (middle sagittal), v_pm.png (medial sagittal, posterior half), v_ax1.png / v_ax2.png
(axial, joint line to top of recess), v_axpm.png (axial at and below the joint line, posteromedial crop). Run from repo root.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

D = Path("work/claude_labels_oct16")
SHEETS = ["1_sag_fs", "3_ax_fs", "5_notch", "6_medial", "9_patella"]
VIEW = ["work/image_labeling/view.py"]
WIN = ["--window", "5", "99.9"]

p = argparse.ArgumentParser()
p.add_argument("index", type=int)
p.add_argument("--sag")
p.add_argument("--ax")
p.add_argument("--chat", type=int, required=True, help="chat number 1-5: temp folder work/claude_labels_oct16/_tmp_c<chat>/")
a = p.parse_args()
TMP = D / f"_tmp_c{a.chat}"

shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir()
with open(D / "_prompts/studies.csv", newline="") as fh:
    row = next(r for r in csv.DictReader(fh) if int(r["index"]) == a.index)
folder = f"{a.index:04d}"
have = [s for s in SHEETS if s in row["sheets"].split(";")]
subprocess.run(["unzip", "-o", "-q", f"image_labeling_bundle/{row['images_zip']}"] + [f"{folder}/{s}.jpg" for s in have]
               + ["-d", str(TMP)], check=True)
print("study_id", row["study_id"], "sheets", have)

out = subprocess.run([sys.executable] + VIEW + ["series", str(a.index)], capture_output=True, text=True, check=True).stdout
series = {}
for line in out.splitlines():
    m = re.match(r"(s\d+)\s+(\w+)\s+(.+?)\s+(\d+) slices, spacing ([\d.]+) mm, positions ([\d.]+) mm (\w+) \.\. ([\d.]+) mm (\w+)", line)
    if m:
        sid, plane, seq, n, sp, p0, _, _, _ = m.groups()
        series[sid] = dict(plane=plane, seq=seq.strip(), n=int(n), sp=float(sp), p0=-float(p0), flags=line[m.end():].strip())
        print(line)


def fluid(s):
    return any(k in s["seq"] for k in ("FS", "STIR")) and "other knee" not in s["flags"]


def pick(plane, forced):
    if forced:
        return forced
    cands = [k for k, s in series.items() if s["plane"] == plane and fluid(s)]
    return min(cands, key=lambda k: (" 3D" in series[k]["seq"], -round(series[k]["n"] * series[k]["sp"] / 20), series[k]["sp"])) if cands else None


def idx(s, pos):
    return min(max(round((pos - s["p0"]) / s["sp"]) + 1, 1), s["n"])


def show(sid, lo_mm, hi_mm, name, box=None, max_n=12):
    s = series[sid]
    first, last = idx(s, lo_mm), idx(s, hi_mm)
    every = max(1, -(-(last - first + 1) // max_n))
    if s["sp"] < 2.5:
        every = max(every, round(3.5 / s["sp"]))
    cmd = [sys.executable] + VIEW + ["show", str(a.index), sid, str(first), str(last), "--every", str(every), "--out", str(TMP / name)] + WIN
    if box:
        cmd += ["--box"] + [str(v) for v in box]
    print(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip(), f"(#{first}-{last} every {every})")


sag, ax = pick("Sagittal", a.sag), pick("Axial", a.ax)
print("sag", sag, "ax", ax)
if sag:
    s = series[sag]
    half = 5 * max(s["sp"], 3.5) if s["sp"] < 2.5 else 5 * s["sp"]
    show(sag, -half, half, "v_sag.png", max_n=11)
    show(sag, -38, -12, "v_pm.png", box=[0.35, 0.05, 1.0, 0.85])
if ax:
    show(ax, -8, 26, "v_ax1.png", box=[0.0, 0.0, 1.0, 1.0])
    show(ax, 26, 70, "v_ax2.png", box=[0.0, 0.0, 1.0, 1.0])
    show(ax, -36, 0, "v_axpm.png", box=[0.0, 0.35, 0.75, 1.0])
