"""Per-study verdicts from out/series_side.csv: which studies have flipped series, and of what kind.

PYTHONPATH=. conda run -n rsna python data/build/side_check/summarize.py   -> prints a summary, writes out/study_side.csv, data/derived/series_side_final.csv"""
from pathlib import Path

import numpy as np
import pandas as pd

from common.meta import study_table

OUT = Path(__file__).parent / "out"
MENISCI = Path(__file__).parent / "nnunet_sides.csv"  # copy of work/knee_seg/nnunet_knee/out_all/sides.csv (Ideas session)
DERIVED = Path(__file__).resolve().parents[2] / "derived"
s = pd.read_csv(OUT / "series_side.csv", index_col=0)
s = s[s.dup_of.isna()]
st = study_table()
g = s[s.p_flip.notna()].groupby("StudyInstanceUID")
t = pd.DataFrame({"series": g.size(), "flipped": g.flip.sum(), "cor_ax": g.apply(lambda d: d.plane.isin(["Coronal", "Axial"]).sum(), include_groups=False),
                  "cor_ax_flipped": g.apply(lambda d: (d.flip & d.plane.isin(["Coronal", "Axial"])).sum(), include_groups=False),
                  "other_knee": g.other_knee.sum()}).join(st[["side", "side_source", "gold"]])
t["kind"] = "ok"
t.loc[t.flipped > 0, "kind"] = "some series flipped"
t.loc[(t.cor_ax > 0) & (t.cor_ax_flipped == t.cor_ax) & (t.flipped >= t.series - 1), "kind"] = "study side wrong"
t.to_csv(OUT / "study_side.csv")
print(t.kind.value_counts().to_dict())
print(t[t.kind != "ok"].groupby(["kind", "side_source"]).size().to_dict())
print("gold:", t[(t.kind != "ok") & t.gold][["series", "flipped", "cor_ax", "cor_ax_flipped", "other_knee", "side", "side_source", "kind"]].to_string())

# final per-series decision: whole study flipped when the study side is wrong; otherwise flip a coronal/axial series at p_flip >= 0.8
# (held-out: 0 of 2,871 false flags) and a sagittal series only at p_flip >= 0.9 (its detector errs on ~1.4% of held-out series at 0.5)
s = s.join(t["kind"], on="StudyInstanceUID")
some = s[(s.kind == "some series flipped") & s.flip]
print("partial flips by plane:", some.plane.value_counts().to_dict(), "| trusted among them:", int(some.trusted.sum()))
sag = s.plane == "Sagittal"
s["flip_final"] = (s.kind == "study side wrong") | (~sag & (s.p_flip >= 0.8)) | (sag & (s.p_flip >= 0.9))
s["flip_source"] = np.where(s.kind == "study side wrong", "whole study", np.where(s.flip_final, "detector", ""))
# second signal (2026-10-08): anatomical meniscus naming of nnU-Net Knee masks on sagittal fluid-sensitive series (Ideas session,
# work/knee_seg/nnunet_knee/out_all/sides.csv; meniscus_order 0 = medial meniscus found on the lateral side of the cached orientation).
# A sagittal series below the 0.9 threshold also flips when its detector p_flip >= 0.5 and the masks say it is reversed.
men = pd.read_csv(MENISCI, index_col="SeriesInstanceUID").meniscus_order.reindex(s.index)
extra = sag & ~s.flip_final & (s.p_flip >= 0.5) & (men == 0)
s.loc[extra, "flip_final"], s.loc[extra, "flip_source"] = True, "detector + menisci"
print(f"meniscus check: {int((men.notna() & sag).sum())} sagittal series with both menisci; agree with flips "
      f"{int((s.flip_final & (men == 0)).sum())}/{int((s.flip_final & men.notna()).sum())}, with non-flips "
      f"{int((~s.flip_final & (men == 1)).sum())}/{int((~s.flip_final & men.notna()).sum())}; added {int(extra.sum())} sagittal flips")
print("flipped but masks say not reversed:", s[s.flip_final & (men == 1)][["StudyInstanceUID", "p_flip", "flip_source"]].assign(
      StudyInstanceUID=lambda d: d.StudyInstanceUID.str[-12:]).to_dict("index"))
s["side_final"] = s.side.where(~s.flip_final, s.side.map({"L": "R", "R": "L"}))
s[["StudyInstanceUID", "plane", "side", "side_final", "flip_final", "flip_source", "p_flip", "tag", "geo", "x", "other_knee"]].to_csv(DERIVED / "series_side_final.csv")
f = s[s.flip_final]
print(f"final: flip {len(f)} series in {f.StudyInstanceUID.nunique()} studies "
      f"(whole study {int((t.kind == 'study side wrong').sum())}); gold studies flipped: {int(st.loc[f.StudyInstanceUID.unique(), 'gold'].sum())}")
