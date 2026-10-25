"""W2-2 p2b_loc target (Experts V3 10-20 11:4x, Master's wave 2: image-truth localisation supervision for the bone expert's contusion location
heads; OAI: no; CPU). Reads the Opus p2 reads (work/claude_labels_oct20/p2_s*/<study>.json, same checks as build_tgt_p2.py; this script never
writes there) and writes data/tgt_p2_loc.csv: StudyInstanceUID + one column per LOC region the reads grade (medial / lateral femoral condyle,
medial / lateral tibial plateau, patella) = 1 iff findings.contusion.edema[region] >= 1 AND the study is traumatic (Opus contusion p >= .5
OR fracture p >= .5, data/tgt_p2_bone.csv; fracture 'around_fracture' oedema counts), else 0 (degenerative-only oedema = a useful negative;
Master 11:4x, C57: location heads learn traumatic, not degenerative oedema); NaN when the contusion finding is unread / not assessable or the
grade is missing. A region with < MIN_POS positives in fold-1 training is dropped (stays on text locations). bone_train loc_img_csv replaces the text location targets of those heads where set.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_tgt_p2_loc.py"""
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from common.folds import get_folds  # noqa: E402
from lib import GOLD  # noqa: E402

LAB = REPO / "work/claude_labels_oct20"
MIN_POS = 25
REGIONS = ["medial_femoral_condyle", "lateral_femoral_condyle", "medial_tibial_plateau", "lateral_tibial_plateau", "patella"]


def main():
    key = pd.read_csv(LAB / "_key/p2s.csv")
    rows = {}
    for f in sorted(LAB.glob("p2_s*/*.json")):
        d = json.load(open(f))
        sid = d["study_id"]
        assert f.stem == sid and sid not in rows and d.get("panel") == "p2", f
        k = key[key.study_id == sid]
        assert len(k) == 1 and int(k["index"].iloc[0]) == int(d["index"]), (sid, d["index"])
        c = d["findings"].get("contusion") or {}
        ed = c.get("edema") or {}
        ok = c.get("status") not in (None, "not_assessable")
        rows[sid] = {r: float(ed[r] >= 1) if ok and isinstance(ed.get(r), (int, float)) else float("nan") for r in REGIONS}
    T = pd.DataFrame.from_dict(rows, orient="index").rename_axis("StudyInstanceUID")
    assert not T.index.isin(GOLD.index).any(), "a gold study was labelled"
    b = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID").reindex(T.index)
    trauma = ((b.Contusion >= 0.5) | (b.Fracture >= 0.5)).to_numpy()
    raw = T.copy()
    T = T.where(T.isna() | trauma[:, None], 0.0)
    fo = get_folds().iloc[:, 0]
    tr1 = T.index.isin(fo.index[fo != 1])
    pos1 = (T[tr1] == 1).sum()
    print(f"traumatic studies {int(trauma.sum())} / {len(T)}; oedema >= 1 before the trauma rule {(raw == 1).sum().to_dict()}")
    T = T[[r for r in REGIONS if pos1[r] >= MIN_POS]]
    T.to_csv(EXP / "data/tgt_p2_loc.csv")
    print(f"dropped (< {MIN_POS} fold-1 training positives, stay on text): {[r for r in REGIONS if pos1[r] < MIN_POS]}")
    print(f"data/tgt_p2_loc.csv: {len(T)} studies; set {T.notna().sum().to_dict()}; oedema >= 1: {(T == 1).sum().to_dict()}; "
          f"fold-1 training positives {(T[tr1] == 1).sum().to_dict()}")


if __name__ == "__main__":
    main()
