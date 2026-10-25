"""Pilot study set for open-VLM image labelling: 300 non-gold train studies with a Codex synovitis read and a 0.961 OOF row.
200 random + 100 enriched for the rare columns (text v1 probability >= 0.5 for ACL / MCL / Lateral OA / Baker's / Contusion / Fracture).
Writes out/pilot_studies.csv (index, StudyInstanceUID, stratum)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels, probability_table
from common.paths import FINDINGS, KEY_OF, REPO

HERE = Path(__file__).parent
OOF = REPO / "experiments/v2/oct13_deep_ensembling_test/out/oof_f5_win.parquet"
RARE = ["ACL", "MCL", "Lateral OA", "Baker's", "Contusion", "Fracture"]

oof = pd.read_parquet(OOF).loc["f5_win"]   # the file holds f5_eff and f5_win; f5_win = shipped 0.961
codex = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")
pool = codex.index.intersection(oof.index).difference(gold_labels().index)
p = probability_table("v1").reindex(pool)
rng = np.random.default_rng(0)
rand = rng.choice(pool, 200, replace=False)
rest = pool.difference(rand)
rich = [s for f in RARE for s in rng.choice(rest[(p.loc[rest, KEY_OF[f]] >= 0.5).to_numpy()], 17, replace=False)]
rich = list(dict.fromkeys(rich))[:100]
out = pd.DataFrame({"StudyInstanceUID": list(rand) + rich, "stratum": ["random"] * 200 + ["rare"] * len(rich)})
out.index.name = "index"
HERE.joinpath("out").mkdir(exist_ok=True)
out.to_csv(HERE / "out/pilot_studies.csv")
print(f"pool {len(pool)}, picked {len(out)}; text v1 >= 0.5 counts:")
print((p.reindex(out.StudyInstanceUID) >= 0.5).sum().rename(dict(zip([KEY_OF[f] for f in FINDINGS], FINDINGS))).to_string())
