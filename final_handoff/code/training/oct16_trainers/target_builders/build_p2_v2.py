"""'MORE LABELS' target set v2 (Master 10-20 14:4x: the frozen 559 set stays for every wave 1-4 arm; Phase 2 reads (labelling resumed 15:50,
~860 reads by ~18:00) go into a SEPARATE v2 set used only by a later 'more labels' arm; OAI: no; CPU). Same recipes as the frozen files,
written to data/v2/ (never over the read-only 559 files):
 tgt_p2_bone.csv       Fracture / Contusion Opus p on every read study (build_tgt_p2.py's reader: mapping.probability, one read per study,
                       key-checked, no gold)                                                  [sparse; for reads / sampling only]
 tgt_p2_bone_dv3.csv   dense bone image target: Opus where read, tgt_v3 elsewhere (build_p2_dense.py's dv3 rule)
 ext_p2_main_r.csv     main-recipe target rows: Contusion / Fracture = Opus, PF OA = 0.5 Opus + 0.5 text v1 (text missing -> Opus) (mainp tgt_csv)
 eq_ids_lab.csv        the read studies (samp_ids)
 img_w_opus4.csv / img_w_opus8.csv  dense image-loss weights: read cells x K, old cells x1
CONSISTENCY (asserted): every frozen-559 study is in v2 with the SAME values (a re-read would change them).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_v2.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "work/image_labeling"), str(EXP / "scripts")]
from common.folds import get_folds  # noqa: E402
from common.labels import probability_table  # noqa: E402
from lib import GOLD  # noqa: E402
from mapping import probability  # noqa: E402

LAB, OUT = REPO / "work/claude_labels_oct20", EXP / "data/v2"
COLS = {"fracture": "Fracture", "contusion": "Contusion", "pf_oa": "PF OA"}


def reads():
    key = pd.read_csv(LAB / "_key/p2s.csv")
    rows = {}
    for f in sorted(LAB.glob("p2_s*/*.json")):
        d = json.load(open(f))
        sid = d["study_id"]
        assert f.stem == sid and sid not in rows and d.get("panel") == "p2", f
        k = key[key.study_id == sid]
        assert len(k) == 1 and int(k["index"].iloc[0]) == int(d["index"]), (sid, d["index"])
        rows[sid] = {c: (float(p) if (x := d["findings"].get(fk) or {}).get("status") and (p := probability(x["status"], x["certainty"])) is not None
                         else np.nan) for fk, c in COLS.items()}
    T = pd.DataFrame.from_dict(rows, orient="index").rename_axis("StudyInstanceUID")
    assert not T.index.isin(GOLD.index).any(), "a gold study was read"
    return T


def main():
    OUT.mkdir(exist_ok=True)
    T = reads()
    fz = pd.read_csv(EXP / "data/ext_p2_main.csv").set_index("StudyInstanceUID")      # the frozen 559 (read-only)
    assert fz.index.isin(T.index).all(), f"{int((~fz.index.isin(T.index)).sum())} frozen studies missing from v2"
    d = (T.loc[fz.index, list(COLS.values())] - fz[list(COLS.values())]).abs().max()
    assert (d.fillna(0) < 1e-9).all(), f"frozen values changed: {d.to_dict()}"
    bone = T[["Fracture", "Contusion"]]
    bone.to_csv(OUT / "tgt_p2_bone.csv")
    t3 = pd.read_csv(REPO / "experiments/v1/oct11_fracture_contusion/data/tgt_v3.csv").set_index("StudyInstanceUID")[["Fracture", "Contusion"]]
    t3 = t3[~t3.index.isin(GOLD.index)]
    dv3 = t3.copy()
    rd = bone.dropna(how="all")
    assert rd.index.isin(dv3.index).all()
    for c in dv3.columns:
        m = rd[c].notna()
        dv3.loc[rd.index[m], c] = rd.loc[m, c].to_numpy()
    dv3.rename_axis("StudyInstanceUID").to_csv(OUT / "tgt_p2_bone_dv3.csv")
    t = probability_table("v1")["pf_oa"].reindex(T.index)
    r = T.copy()
    r["PF OA"] = 0.5 * T["PF OA"] + 0.5 * t.fillna(T["PF OA"])
    r[["Contusion", "Fracture", "PF OA"]].to_csv(OUT / "ext_p2_main_r.csv")
    pd.DataFrame({"StudyInstanceUID": T.index}).to_csv(OUT / "eq_ids_lab.csv", index=False)
    for k in (4, 8):
        w = pd.DataFrame(1.0, index=dv3.index, columns=["Fracture", "Contusion"])
        for c in w.columns:
            w.loc[bone.index[bone[c].notna()], c] = float(k)
        w.rename_axis("StudyInstanceUID").to_csv(OUT / f"img_w_opus{k}.csv")
    fo = get_folds().iloc[:, 0]
    print(f"OAI: no. v2 set: {len(T)} read studies (frozen 559 included, values identical); new {len(T) - len(fz)}; written to data/v2/")
    for c in COLS.values():
        v = T[c].dropna()
        print(f"  {c}: read {len(v)}, p >= .5 {int((v >= .5).sum())} (frozen {int((fz[c] >= .5).sum())}); per fold read / pos "
              + ", ".join(f"f{F} {int(v.index.isin(fo.index[fo == F]).sum())} / {int((v[v.index.isin(fo.index[fo == F])] >= .5).sum())}" for F in range(5)))


if __name__ == "__main__":
    main()
