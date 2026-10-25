"""Studies with a Codex Baker's read (516 non-gold, text v1 Baker's below / unsettled) for the Baker's reader check: indexes 3000+.
Writes out/bakers_studies.csv (for render_views.py / score.py --studies) and out/codex_bakers.csv (StudyInstanceUID, Baker's probability)."""
import json
from pathlib import Path

import pandas as pd

from common.labels import gold_labels

HERE = Path(__file__).parent
d = json.load(open("data/image_labels/codex_bakers.json"))
cb = pd.Series({k: v["findings"]["bakers_cyst"]["probability"] for k, v in d.items()}, name="Baker's").dropna()
cb = cb[~cb.index.isin(gold_labels().index)]
cb.rename_axis("StudyInstanceUID").to_csv(HERE / "out/codex_bakers.csv")
st = pd.DataFrame({"StudyInstanceUID": cb.index, "stratum": "codex_bakers"}, index=pd.RangeIndex(3000, 3000 + len(cb), name="index"))
st.to_csv(HERE / "out/bakers_studies.csv")
print(len(st), "studies; Codex Baker's >= 0.5:", int((cb >= 0.5).sum()))
