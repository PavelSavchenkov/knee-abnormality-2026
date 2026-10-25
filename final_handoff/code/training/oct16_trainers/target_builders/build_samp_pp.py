"""U100 sampling pools for a bone main-target arm (USER 10-21 via Master: "oversampling claude-labelled positives ... a separate pool for
positives, sample with some coef C sampling C times more frequent from claude-based labelled studies positives for that classes"; OAI: no;
CPU). From data/<SNAP>/opus_raw.csv, column COL: Opus positives (p >= .5) get mult POS, Opus-read negatives NEG, unread 1 (not listed);
bone_train samp_ids draws each study ~mult times per epoch (repeats get fresh crops). Writes data/<SNAP>/samp_pp_<col>_p<POS>_n<NEG>.csv.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_samp_pp.py SNAP [--col Contusion] [--pos 3] [--neg 1.5]"""
import argparse
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap")
    ap.add_argument("--col", default="Contusion")
    ap.add_argument("--pos", type=float, default=3.0)
    ap.add_argument("--neg", type=float, default=1.5)
    a = ap.parse_args()
    op = pd.read_csv(EXP / "data" / a.snap / "opus_raw.csv").set_index("StudyInstanceUID")[a.col].dropna()
    s = pd.DataFrame({"mult": (op >= 0.5).map({True: a.pos, False: a.neg})}).rename_axis("StudyInstanceUID")
    assert (s.mult > 1).all(), "every listed study must be boosted"
    out = EXP / "data" / a.snap / f"samp_pp_{a.col.lower()}_p{a.pos:g}_n{a.neg:g}.csv"
    s.to_csv(out)
    print(f"OAI: no. {out.relative_to(EXP)}: {len(s)} {a.col}-read studies: {int((op >= .5).sum())} positives x{a.pos:g}, "
          f"{int((op < .5).sum())} negatives x{a.neg:g}; unread x1")


if __name__ == "__main__":
    main()
