"""Write session_prompts.txt: one copy-paste prompt per range of 50 indexes over all of reports.csv."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
n = len(pd.read_csv("/home/pavel/Programming/knee-abnormality-2026/text_labeling_bundle/reports.csv", usecols=["index"]))
blocks = []
for lo in range(0, n, 50):
    r = f"{lo}-{min(lo + 49, n - 1)}"
    blocks.append(f"### {r}\nUnzip text_labeling_bundle.zip from the project files, follow its prompt.txt exactly, and deliver v1_{r}.zip as it specifies.\n\nRange: {r}\n")
(HERE / "session_prompts.txt").write_text("\n".join(blocks), encoding="utf-8")
print(len(blocks), "prompts, last range", r)
