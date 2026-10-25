"""Sanity sheet: the 3 highest and 3 lowest VLM scores of one finding in a pass, with their Codex / text labels, as one downscaled
image of each study's first view sheet. Usage: python sanity_pairs.py SCORES.csv FINDING SHEET OUT.jpg"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.labels import probability_table
from common.paths import KEY_OF

HERE = Path(__file__).parent
scores, f, sheet, out = sys.argv[1:5]
r = pd.read_csv(scores)
r = r[r.finding == f].sort_values("p_yes")
codex = pd.read_csv("data/derived/synovitis_labels_codex.csv", index_col="StudyInstanceUID")
text = probability_table("v1")[KEY_OF[f]]
pick = pd.concat([r.tail(3)[::-1], r.head(3)])
tiles = []
for row in pick.itertuples():
    img = cv2.imread(str(HERE / "views" / str(row.index) / f"{sheet}.png"))
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cx = codex[f].get(row.StudyInstanceUID, np.nan) if f in codex else np.nan
    lab = f"#{row.index} vlm {row.p_yes:.2f} codex {cx:.2f} text {text.get(row.StudyInstanceUID, np.nan):.2f}"
    cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, lab, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    tiles.append(img)
h = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
cv2.imwrite(out, cv2.resize(grid, (grid.shape[1] * 2 // 3, grid.shape[0] * 2 // 3)))
print(pick[["index", "p_yes"]].to_string(index=False))
