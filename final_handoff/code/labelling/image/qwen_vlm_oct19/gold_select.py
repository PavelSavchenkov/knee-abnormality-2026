"""Gold-58 study list for the one-time reader check (indexes 1000+): out/gold_studies.csv. Never used for tuning."""
from pathlib import Path

import pandas as pd

from common.labels import gold_labels

HERE = Path(__file__).parent
g = pd.DataFrame({"StudyInstanceUID": gold_labels().index, "stratum": "gold"})
g.index = pd.RangeIndex(1000, 1000 + len(g), name="index")
g.to_csv(HERE / "out/gold_studies.csv")
print(len(g))
