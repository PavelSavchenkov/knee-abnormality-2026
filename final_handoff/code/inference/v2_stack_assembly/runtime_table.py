"""Runtime table of oct19_stack_fold per variant (Master 10-18, USER U72: "make sure that our submissions fits within 9h"): per-stage local wall
(run_local's local_stats{sfx}.json, 88 studies) -> Kaggle estimate with run_local's per-stage local->Kaggle ratios (from 0.961's Kaggle commit
check). Reference row: the 0.961-equivalent stages (oct19_probe_eff_pw run_local) = the calibration; 0.961 scored 262 min. Writes RUNTIME.md.
conda run -n rsna python submissions/oct19_stack_fold/runtime_table.py"""
import json
from pathlib import Path

import pandas as pd

from run_local import HERE, N_TEST, OVERHEAD_S, RATIO, SFX, SUB

LIMIT = 540
N_CHK = 88


def row(stats):
    walls = {k: v["wall_s"] for k, v in stats.items() if isinstance(v, dict) and "wall_s" in v}
    kag = {k: w / N_CHK * RATIO.get(k, 7.0) * N_TEST / 60 for k, w in walls.items()}
    return walls, kag, sum(kag.values()) + OVERHEAD_S / 60


def main():
    runs = {"0.961 (reference)": SUB / "oct19_probe_eff_pw/local_stats.json"} | {v: HERE / f"local_stats{s}.json" for v, s in SFX.items()}
    R, missing = {}, []
    for name, fp in runs.items():
        kj = HERE / f"kaggle_output{SFX.get(name, '')}/check_timing.json"
        if name in SFX and kj.exists():                     # MEASURED on Kaggle (the commit run's check): s / study of the main group x N_TEST
            t = json.load(open(kj))["main"]
            st = {k: v["wall_s"] / t["n"] * N_TEST / 60 for k, v in t.items() if isinstance(v, dict) and "wall_s" in v}
            tot = t["wall_s"] / t["n"] * N_TEST / 60 + OVERHEAD_S / 60
            R[name + " (Kaggle measured)"] = {**{f"{k} (min)": round(v, 1) for k, v in st.items()}, "Kaggle est (min)": round(tot),
                                              "margin to 540 (min)": round(LIMIT - tot), "local wall (s)": None}
            continue
        if not fp.exists():
            missing.append(name)
            continue
        walls, kag, tot = row(json.load(open(fp)))
        R[name] = {**{f"{k} (min)": round(v, 1) for k, v in kag.items()}, "Kaggle est (min)": round(tot), "margin to 540 (min)": round(LIMIT - tot),
                   "local wall (s)": round(sum(walls.values()))}
    T = pd.DataFrame(R).T
    txt = (f"# Runtime per variant (Kaggle estimate for {N_TEST} test studies + {OVERHEAD_S} s overhead; limit {LIMIT} min; 0.961 scored 262)\n\n"
           f"Rows '(Kaggle measured)' = the variant's own Kaggle commit check (main group s / study x {N_TEST}); other rows = local stages x the ratios {RATIO} (unknown 7.0). Past probes SCORED below their estimates (MCL stack 284 vs 375, ACL 232 vs 338, Syn cnx 244 vs 288; 0.961 262 vs 259). Kaggle machine speed varies ~15 % between commit runs (P2's check ran EVERY stage ~15 % faster than FULL's): read each margin as a range, not a point. Not run yet: {missing or 'none'}.\n\n"
           + T.fillna("").to_markdown() + "\n")
    (HERE / "RUNTIME.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
