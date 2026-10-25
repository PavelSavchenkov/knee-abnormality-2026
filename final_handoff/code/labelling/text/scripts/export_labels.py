"""Usage: export_labels.py VERSION (e.g. v0). Writes data/text_labels/VERSION.json (LLM labels from work/text_labeling/returned_VERSION/, keyed by study_id) and
data/text_labels/gold_text_labels.json (my hand labels for 208 studies, keyed by study_id, statuses in the bundle vocabulary)."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from collect_v0 import load                                        # noqa: E402
from score_returned import CLS, MINE                                # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
TG = REPO / "work/text_gold"
OUT = REPO / "data/text_labels"
OUT.mkdir(parents=True, exist_ok=True)

version = sys.argv[1]
docs = load(f"returned_{version}")
llm = {s: x for s, (_, x) in sorted(docs.items(), key=lambda kv: kv[1][1]["index"])}
(OUT / f"{version}.json").write_text(json.dumps(llm, ensure_ascii=False, indent=1), encoding="utf-8")

gold = {}
for name, labels, ids in [("gold58", TG / "labels_all.txt", TG / "blind_ids.csv"), ("random150", TG / "nongold/labels/labels_all.txt", TG / "nongold/sample_ids.csv")]:
    sid = pd.read_csv(ids).set_index("tid").StudyInstanceUID
    for line in open(labels, encoding="utf-8"):
        if line.strip():
            tid, c, _, reason, quote, note = line.rstrip("\n").split("|")
            g = gold.setdefault(sid[tid], {"set": name, "tid": tid, "findings": {}})
            g["findings"][CLS[c]] = {"status": MINE[reason], "reason": reason, "quote": "" if quote == "-" else quote, "note": note}
assert len(gold) == 208 and all(len(g["findings"]) == 12 for g in gold.values())
(OUT / "gold_text_labels.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{version}.json: {len(llm)} studies | gold_text_labels.json: {len(gold)} studies "
      f"({sum(g['set'] == 'gold58' for g in gold.values())} gold58, {sum(g['set'] == 'random150' for g in gold.values())} random150)")
