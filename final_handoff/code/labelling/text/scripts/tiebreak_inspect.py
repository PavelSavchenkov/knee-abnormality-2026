"""Is steven v2's within-status order better than v0's attributes (paired bootstrap), and what does it pick up?
Lists gold-58 cells inside the main statuses sorted by steven v2, with the official label and v0's quote. Writes out/tiebreak_inspect.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from check_official_and_duplicates import L, auc                   # noqa: E402
from tiebreak_check import attr, cells                             # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
v0 = json.load(open(REPO / "data/text_labels/v0.json"))
inv = {v: k for k, v in L.items()}
cells["quote"] = [v0[s]["findings"][inv[f]]["quote"] for s, f in zip(cells.sid, cells.finding)]
macro = lambda d, col: np.mean([auc(g.y, g[col]) for _, g in d.groupby("finding")])
rng = np.random.default_rng(1)
by_sid = {s: d for s, d in cells.groupby("sid")}
diffs = []
for _ in range(1000):
    d = pd.concat([by_sid[s] for s in rng.choice(list(by_sid), len(by_sid))], ignore_index=True)
    if d.groupby("finding").y.nunique().min() == 2:
        diffs.append(macro(d, "rate+steven_v2") - macro(d, "rate+v0_attributes"))
diffs = np.array(diffs)
out = [f"== steven v2 minus v0 attributes as tie-breaker: {diffs.mean():+.4f} [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], "
       f"share of resamples > 0: {(diffs > 0).mean():.2f}"]
for st in ["meets", "below", "unsettled", "not_mentioned"]:
    d = cells[cells.status == st].sort_values("steven_v2")
    out.append(f"\n== {st}: {len(d)} cells, {int(d.y.sum())} officially positive; sorted by steven v2 (low to high)")
    for _, r in d.iterrows():
        out.append(f"   v2={r.steven_v2:.2f} gold={r.y} {r.finding:17s} attr={r.v0_attributes:.1f} | {r.quote[:110]}")
(HERE / "out/tiebreak_inspect.txt").write_text("\n".join(out) + "\n")
print(out[0])
