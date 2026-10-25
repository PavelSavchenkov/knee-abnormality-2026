"""Write text_labeling_bundle/reports.csv: index, study_id, knee_side, language, report (all 4,407 train studies).
Indexes 0-207 are my text-labelled studies (gold-58 + 150 non-gold, shuffled) so the first batches can be scored at once;
208+ are the rest, shuffled. knee_side: DICOM Laterality tag, else sign of the image-centre x (as in text_gold)."""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
train = pd.read_csv(REPO / "data/original_data/train.csv").set_index("StudyInstanceUID")
study = pd.read_csv(REPO / "work/eda/out/study_table.csv").set_index("StudyInstanceUID")
lang = pd.read_csv(REPO / "work/eda/out/reports_table.csv").set_index("StudyInstanceUID").lang
names = {"en": "English", "es": "Spanish", "tr": "Turkish", "hr": "Croatian", "el": "Greek", "de": "German",
         "bg/ru": "Bulgarian or Russian", "nl": "Dutch", "fr": "French", "unk": "unknown"}
labelled = list(pd.read_csv(REPO / "work/text_gold/blind_ids.csv").StudyInstanceUID) + list(pd.read_csv(REPO / "work/text_gold/nongold/sample_ids.csv").StudyInstanceUID)
assert len(set(labelled)) == 208
rng = np.random.default_rng(0)
rest = sorted(set(train.index) - set(labelled))
order = list(rng.permutation(labelled)) + list(rng.permutation(rest))


def side(sid):
    tag = study.at[sid, "lat_tag"]
    return {"L": "LEFT", "LEFT": "LEFT", "R": "RIGHT", "RIGHT": "RIGHT"}.get(tag, "LEFT" if study.at[sid, "cx"] > 0 else "RIGHT")


out = pd.DataFrame({"index": range(len(order)), "study_id": order, "knee_side": [side(s) for s in order],
                    "language": [names[lang[s]] for s in order], "report": [train.at[s, "Report"] for s in order]})
assert len(out) == 4407 and out.study_id.is_unique and out.report.notna().all()
(REPO / "text_labeling_bundle").mkdir(exist_ok=True)
out.to_csv(REPO / "text_labeling_bundle/reports.csv", index=False)
print(out.shape, out.knee_side.value_counts().to_dict(), out.language.value_counts().to_dict())
