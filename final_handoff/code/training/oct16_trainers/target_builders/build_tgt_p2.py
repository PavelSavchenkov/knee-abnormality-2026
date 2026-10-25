"""P2 image-label targets (Experts V3 10-20, Master 10:2x relaying the USER's 6 Opus labelling sessions; panel p2 = contusion + fracture +
pf_oa on up to 720 non-gold studies; OAI: no; CPU). Reads whatever work/claude_labels_oct20/p2_s*/<study>.json exist (labeller outputs; this
script never writes there) and writes the image-HEAD target CSVs (unlabelled / not_assessable = NaN = masked in the loss, never filled):
 data/tgt_p2_bone.csv  StudyInstanceUID, Fracture, Contusion   (bone_train img_bone_csv -> img_frac / img_cont heads)
 data/tgt_p2_pf.csv    StudyInstanceUID, img_pf_patella        (oa_train aux_csv; the 'patella' name routes the head to the PF bag)
p = work/image_labeling/mapping.probability(status, certainty) (DEFAULT table). Asserts: non-gold, one read per study, the key's study id ==
the file's. Prints the coverage per class: labelled studies, positive-ish (p >= 0.5) count / share, and both counts inside each fold's TRAINING set
(Master 10-20: p2_s7 = 173 fracture-vague studies, picked up by the glob; the fracture head's real positive count must be visible).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_tgt_p2.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "work/image_labeling"), str(EXP / "scripts")]
from common.folds import get_folds  # noqa: E402
from lib import GOLD  # noqa: E402
from mapping import probability  # noqa: E402

LAB = REPO / "work/claude_labels_oct20"
COLS = {"fracture": "Fracture", "contusion": "Contusion", "pf_oa": "img_pf_patella"}


def main():
    key = pd.read_csv(LAB / "_key/p2s.csv")
    rows = {}
    for f in sorted(LAB.glob("p2_s*/*.json")):
        d = json.load(open(f))
        sid = d["study_id"]
        assert f.stem == sid, (f, sid)
        assert sid not in rows, f"two reads of {sid}"
        assert d.get("panel") == "p2", (f, d.get("panel"))
        k = key[key.study_id == sid]
        assert len(k) == 1 and int(k["index"].iloc[0]) == int(d["index"]), (sid, d["index"])
        r = {}
        for fk, col in COLS.items():
            x = d["findings"].get(fk) or {}
            p = probability(x["status"], x["certainty"]) if x.get("status") else None
            r[col] = np.nan if p is None else float(p)
        rows[sid] = r
    T = pd.DataFrame.from_dict(rows, orient="index").rename_axis("StudyInstanceUID")
    assert not T.index.isin(GOLD.index).any(), "a gold study was labelled"
    T[["Fracture", "Contusion"]].to_csv(EXP / "data/tgt_p2_bone.csv")
    T[["img_pf_patella"]].to_csv(EXP / "data/tgt_p2_pf.csv")
    fo = get_folds().iloc[:, 0]
    print(f"P2 targets from {len(T)} labelled studies (key {len(key)}): data/tgt_p2_bone.csv, data/tgt_p2_pf.csv")
    for c in T.columns:
        v = T[c].dropna()
        per = {F: f"{int(v.index.isin(fo.index[fo != F]).sum())} ({int((v[v.index.isin(fo.index[fo != F])] >= 0.5).sum())} pos)" for F in range(5)}
        print(f"  {c}: labelled {len(v)}, p >= 0.5: {int((v >= 0.5).sum())} ({float((v >= 0.5).mean()):.2f}), in each fold's TRAINING set "
              f"labelled (p >= 0.5) {per}"
              + ("  [WEAK: < ~360 labels; the Synovitis precedent needed ~700]" if len(v) < 360 else ""))


if __name__ == "__main__":
    main()
