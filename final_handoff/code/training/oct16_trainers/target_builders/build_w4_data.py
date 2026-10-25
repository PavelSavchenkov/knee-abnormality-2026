"""Wave 4 data (Master 10-20 14:3x, USER: "keep experimenting coming up with new ways to use new labels -- they surely should improve quality
when used properly (including oversampling, sampling rules in general, loss coefs, ..."; OAI: no; CPU). From the Opus p2 reads
(work/claude_labels_oct20/p2_s*/<study>.json, the same files build_tgt_p2.py reads) and data/tgt_p2_bone.csv:
 data/samp_w4_disagree.csv  StudyInstanceUID, mult   (W4-2, bone samp_ids): every Opus-labelled study; mult 4 where Opus and the text v1
                            probability (labels.make_targets, v1prob) disagree at .5 on Fracture or Contusion, else 2
 data/img_w_cert.csv        StudyInstanceUID, Fracture, Contusion   (W4-3, bone img_w_csv): per-cell loss weight from the Opus certainty
                            (high 1.0 / moderate 0.6 / low 0.3); NaN = unlabelled; asserted to cover every labelled cell of tgt_p2_bone.csv
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_w4_data.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(EXP / "bone"), str(EXP / "scripts")]
from bone_train import DEFAULTS  # noqa: E402
from common.labels import FINDINGS  # noqa: E402
from labels import make_targets  # noqa: E402

CERT = {"high": 1.0, "moderate": 0.6, "low": 0.3}
COLS = {"fracture": "Fracture", "contusion": "Contusion"}


def main():
    op = pd.read_csv(EXP / "data/tgt_p2_bone.csv").set_index("StudyInstanceUID")
    rows = {}
    for f in sorted((REPO / "work/claude_labels_oct20").glob("p2_s*/*.json")):
        d = json.load(open(f))
        assert f.stem == d["study_id"] and d["study_id"] not in rows, f
        rows[d["study_id"]] = {c: CERT[x["certainty"]] if (x := d["findings"].get(k) or {}).get("status") else np.nan for k, c in COLS.items()}
    cw = pd.DataFrame.from_dict(rows, orient="index").rename_axis("StudyInstanceUID").reindex(op.index)
    for c in COLS.values():
        assert (cw[c].notna() == op[c].notna()).all(), f"certainty coverage != target coverage ({c})"
    cw.to_csv(EXP / "data/img_w_cert.csv")
    yt = pd.DataFrame(make_targets(list(op.index), DEFAULTS | {"tgt_csv": ""})[0], index=op.index, columns=FINDINGS)
    dis = pd.Series(False, index=op.index)
    for c in COLS.values():
        m = op[c].notna()
        dis |= m & ((op[c] >= 0.5) != (yt[c] >= 0.5))
    pd.DataFrame({"mult": np.where(dis, 4.0, 2.0)}, index=op.index).to_csv(EXP / "data/samp_w4_disagree.csv")
    print(f"OAI: no. img_w_cert.csv: {len(cw)} studies; weights per column " + "; ".join(f"{c} {cw[c].value_counts().sort_index().to_dict()}" for c in COLS.values())
          + f"\nsamp_w4_disagree.csv: {len(op)} labelled studies, Opus vs text v1 disagree at .5 (Fracture or Contusion) on {int(dis.sum())} -> mult 4, "
          f"rest mult 2; per column " + ", ".join(f"{c} {int((op[c].notna() & ((op[c] >= .5) != (yt[c] >= .5))).sum())}" for c in COLS.values()))


if __name__ == "__main__":
    main()
