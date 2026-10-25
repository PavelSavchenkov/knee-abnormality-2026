"""Proof for the tier-B (bone) PL lines (Master 17:0x item 3). OAI: no. CPU; runs the oct16 launcher with --dry and imports bone/labels.py
read-only; edits nothing.
1. DRY: launcher --dry of the plain all-data line vs the PL line -> the only differing keys must be seed + tgt_csv.
2. TARGET DUMP through bone/labels.make_targets with the resolved PL config on the fold -1 training set: Fracture / Contusion == the CSV;
   Opus-read cells == the plain targets; every other column and all weights == plain.
  conda run -n rsna python work/pl_alldata_2026-10-21/check_tier_b.py --teacher v0"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent
REPO = D.parents[1]
EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(REPO), str(D)]
import targets as T  # noqa: E402
from common.folds import get_folds  # noqa: E402
from common.paths import FINDINGS  # noqa: E402

SPE = {"b3c_rep": 434, "b3l_rep": 434, "b3n_rep": 435, "bfs_rep": 434}     # = their all-data lines (queue.txt / queue_remote*.txt)
COLS = ["Fracture", "Contusion"]


def dry(r, sets):
    out = subprocess.run([sys.executable, str(EXP / "scripts/launch.py"), "--kind", "bone", "--src", f"runs/{r}/fold0", "--name", "dry_x",
                          "--fold", "-1", "--set", *sets, "--dry"], capture_output=True, text=True, cwd=REPO, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
    line = [x for x in out.stdout.splitlines() if x.startswith("launch resolved config ")]
    assert out.returncode == 0 and len(line) == 1, out.stderr[-2000:]
    return json.loads(line[0][len("launch resolved config "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    a = ap.parse_args()
    os.chdir(REPO)
    lab = T.load(EXP / "bone/labels.py", "o16_bone_labels_chk")
    tr = get_folds(5, 0, include_gold=False).index.tolist()
    read = pd.read_csv(EXP / "data/v3c/tgt_p2_bone.csv").set_index("StudyInstanceUID")[COLS].reindex(tr).notna().to_numpy()
    j = [FINDINGS.index(c) for c in COLS]
    oth = [i for i in range(len(FINDINGS)) if i not in j]
    ok = True
    for r, spe in SPE.items():
        rel = f"data/pl/pl_ct_{r}_{a.teacher}.csv"
        c0 = dry(r, [f"spe_ref={spe}", "seed=101"])
        c1 = dry(r, [f"spe_ref={spe}", "seed=201", f"tgt_csv={rel}"])
        diff = {k: (c0.get(k), c1.get(k)) for k in set(c0) | set(c1) if c0.get(k) != c1.get(k)}
        print(f"\n== {r}: DRY diff plain vs PL: {diff}")
        ok &= set(diff) == {"seed", "tgt_csv"}
        y0, w0 = (np.asarray(x, float) for x in lab.make_targets(tr, c0))
        y1, w1 = (np.asarray(x, float) for x in lab.make_targets(tr, c1))
        csv = pd.read_csv(EXP / rel).set_index("StudyInstanceUID").loc[tr, COLS].to_numpy(float)
        d_csv = np.abs(y1[:, j] - csv).max()
        d_read = np.abs(y1[:, j][read] - y0[:, j][read]).max()
        d_oth, d_w = np.abs(y1[:, oth] - y0[:, oth]).max(), np.abs(w1 - w0).max()
        chg = (np.abs(y1[:, j] - y0[:, j]) > 1e-6).mean(0)
        print(f"  max |Y_pl - CSV| {d_csv:.1e} | read cells vs plain {d_read:.1e} | other columns vs plain {d_oth:.1e} | W vs plain {d_w:.1e}"
              f" | changed share Fracture {chg[0]:.3f} Contusion {chg[1]:.3f}")
        ok &= d_csv < 1e-6 and d_read == 0 and d_oth == 0 and d_w == 0
    print(f"\nALL CHECKS {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
