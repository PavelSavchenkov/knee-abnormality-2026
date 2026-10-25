"""PF OA image-head targets for the syn-like PF port (USER 10-21 03:5x "synovitis-like recipes for classes which have new labels"; OAI: no;
no gold): data/pf_yard_v3f.csv = the PF yardstick (0.5 Opus + 0.5 text where Opus read = data/v3f/ext_p2_main_r.csv PF OA) with the old PF
target (v1prob) elsewhere = the USER's standard recipe; data/pf_v1prob.csv = the old target everywhere (the seed-matched twin). Non-gold
studies only (gold rows NaN). Column 'PF OA' (syn_train img_pf reads ibone_csv[PF OA]).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_pf_yard.py [SNAP]   (10-21 11:3x: SNAP = v3f default / v3g -> data/pf_yard_<SNAP>.csv;
the v1prob twin is snapshot-free: written once, verified identical afterwards)
  [SNAP dp2u] (Master 10-21 11:4x PF PROTOCOL SHIFT, variant (i)): the p2 PF reads (3-finding panel, data/SNAP/pf_src.csv) of text-v1
  'unsettled' studies are DROPPED (text only there = the old target in the dense file, NaN in the read-only yardstick) -> data/SNAP_dp2u/
  ext_p2_main_r.csv (read by model_read --snap SNAP_dp2u for PF) + data/pf_yard_SNAP_dp2u.csv; variant (ii) = SNAP itself (keep).
  The two rows are always read side by side (never one silently mixed yardstick)."""
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP.parents[2]))   # the repo root (common/); waiters run it without PYTHONPATH
from common.labels import gold_labels, probability_table, status_table  # noqa: E402


def main():
    g = set(gold_labels().index)
    old = probability_table("v1")["pf_oa"].astype(float)
    old[old.index.isin(g)] = float("nan")
    snap = sys.argv[1] if len(sys.argv) > 1 else "v3f"
    variant = sys.argv[2] if len(sys.argv) > 2 else ""
    assert variant in ("", "dp2u"), variant
    yfull = pd.read_csv(EXP / f"data/{snap}/ext_p2_main_r.csv").set_index("StudyInstanceUID")
    if variant:
        src = pd.read_csv(EXP / f"data/{snap}/pf_src.csv").set_index("StudyInstanceUID")["pf_src"]
        st = status_table("v1")["pf_oa"]
        drop = src.index[(src == "p2") & st.reindex(src.index).eq("unsettled").to_numpy()]
        yfull.loc[drop, "PF OA"] = float("nan")
        vd = EXP / "data" / f"{snap}_{variant}"
        assert not vd.exists(), vd
        vd.mkdir()
        yfull.to_csv(vd / "ext_p2_main_r.csv")
        (vd / "ext_p2_main_r.csv").chmod(0o444)
        print(f"variant {variant}: dropped {len(drop)} p2 PF reads of text-'unsettled' studies (of {int((src == 'p2').sum())} p2 PF cells) "
              f"-> data/{snap}_{variant}/ext_p2_main_r.csv")
        snap = f"{snap}_{variant}"
    yd = yfull["PF OA"].dropna()
    assert not yd.index.isin(g).any()
    dense = old.copy()
    dense.loc[yd.index.intersection(dense.index)] = yd
    for f, s in ((f"pf_yard_{snap}.csv", dense), ("pf_v1prob.csv", old)):
        out = EXP / "data" / f
        if f == "pf_v1prob.csv" and out.exists():
            prev = pd.read_csv(out).set_index("StudyInstanceUID")["PF OA"]
            assert prev.reindex(s.index).fillna(-1).equals(s.fillna(-1)), "pf_v1prob.csv differs"
            continue
        assert not out.exists(), out
        s.rename("PF OA").rename_axis("StudyInstanceUID").to_frame().to_csv(out)
        out.chmod(0o444)
    ch = (dense - old).abs()
    print(f"OAI: no. pf_yard_{snap}.csv: {int(dense.notna().sum())} targets, yardstick on {len(yd)}; changed vs v1prob {int((ch > 1e-9).sum())}, "
          f"flips at .5 {int(((dense >= .5) != (old >= .5)).sum())}; pf_v1prob.csv: {int(old.notna().sum())} (the twin)")


if __name__ == "__main__":
    main()
