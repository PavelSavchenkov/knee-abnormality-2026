"""Render the image-labelling bundle's 9 contact sheets (work/image_labeling/render.py) for every pilot study into
views/<index>/<sheet>.png. Usage: python render_views.py [studies.csv] [sheet,sheet,...]   (optional: write only these sheets)"""
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "image_labeling"))
from render import Renderer  # noqa: E402

HERE = Path(__file__).parent
studies = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else HERE / "out/pilot_studies.csv", index_col="index")
keep = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
r = Renderer()
for i, uid in studies.StudyInstanceUID.items():
    d = HERE / "views" / str(i)
    if d.exists():
        continue
    d.mkdir(parents=True)
    for name, img, _ in r.sheets(uid, i):
        if keep is None or name in keep:
            cv2.imwrite(str(d / f"{name}.png"), img)
    print(i, flush=True) if i % 50 == 0 else None
print("done", len(studies))
