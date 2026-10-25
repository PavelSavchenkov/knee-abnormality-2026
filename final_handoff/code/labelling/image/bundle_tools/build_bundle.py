"""Build image_labeling_bundle/ (repo root) for VLM image labelling in a ChatGPT project.

PYTHONPATH=. conda run -n rsna python work/image_labeling/build_bundle.py                 # full build
PYTHONPATH=. conda run -n rsna python work/image_labeling/build_bundle.py --limit 40 --out /tmp/x
PYTHONPATH=. conda run -n rsna python work/image_labeling/build_bundle.py --prompts-only     # rewrite prompts.zip only (after
                                                                                             # editing bundle_src/), images untouched

Output:
  prompts.zip            prompt.txt + schema.json (12 findings), prompt_fluid.txt + schema_fluid.json, prompt_synovitis.txt +
                         schema_synovitis.json, prompt_bakers.txt + schema_bakers.json, prompt_lateral_oa.txt +
                         schema_lateral_oa.json (panels), definitions.md, host_figures/
                         (the host's 13 annotated examples), validate.py, studies.csv, tiles.csv
  images_FROM-TO.zip     one folder per index with the study's sheets (work/image_labeling/render.py), each zip <= --zip-mb
  README.md, session_prompts.txt
Internal: work/image_labeling/out/order.csv (index -> real study, section, stratum, catch marker).

Order of indexes (see order()): 0-59 the frozen gold rows (58 gold studies + 2 catch items; existing labels refer to them), then the
synovitis queue: priority (effusion stated or likely) and control strata, each shuffled, interleaved 3:1. About 1 in 25 entries is a
catch item: a copy of an earlier study of the same section under a made-up study id, with one small drawn square on one tile (tests
that the model looks at pixels and gives consistent labels).
"""
import argparse
import csv
import io
import json
import re
import shutil
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from render import Renderer  # noqa: E402

from common import meta  # noqa: E402
from common.labels import status_table  # noqa: E402
from common.paths import FINDINGS, KEYS, REPO  # noqa: E402

HERE = Path(__file__).parent
SRC = HERE / "bundle_src"
DEFINITIONS = REPO / "text_labeling_bundle/definitions.md"
OOF_RUN = REPO / "experiments/v0/oct7_baseline/runs/effb0_lr1e-3_R"
CATCH_EVERY, SESSION_SIZE = 25, 10


def oof():
    """Pooled out-of-fold predictions of the R2 model at each fold's last evaluated epoch, columns = KEYS."""
    parts = []
    for f in sorted(OOF_RUN.glob("fold*/oof.parquet")):
        d = pd.read_parquet(f)
        parts.append(d[d.epoch == d.epoch.max()].drop(columns="epoch"))
    return pd.concat(parts).rename(columns=dict(zip(FINDINGS, KEYS)))


def strata(ids):
    """Synovitis queue strata: "priority" = text v1 effusion stated or uncertain (meets / unsettled / conflict / hedged / below), or not
    mentioned with R2 OOF effusion >= 0.4 (gold-58 synovitis rate 63% with effusion, 22% without); "control" = the rest."""
    eff = status_table("v1").loc[ids, "effusion"]
    r2 = oof().reindex(ids).effusion
    pri = eff.isin(["meets", "unsettled", "conflict", "hedged", "below"]) | (eff.isin(["not_mentioned", "general_normal"]) & (r2 >= 0.4))
    return pd.Series(np.where(pri, "priority", "control"), index=ids)


def order(seed=0):
    """Indexes 0-59: the frozen gold rows (gold_rows.csv; labels exist for them). Then the synovitis queue over all non-gold studies:
    each stratum shuffled, interleaved 3 priority : 1 control until priority runs out, then the rest of control. Any queue range is a
    stratified random sample (known selection rate per stratum, for loss weights). A catch item follows every 25th queue entry."""
    gold = pd.read_csv(HERE / "gold_rows.csv", index_col="index")
    rng = np.random.default_rng(seed + 1)
    t = meta.study_table()
    s = strata(list(t.index[~t.gold]))
    pri, ctl = list(rng.permutation(s.index[s == "priority"])), list(rng.permutation(s.index[s == "control"]))
    queue = []
    while pri:
        queue += [(u, "priority") for u in pri[:3]] + [(u, "control") for u in ctl[:1]]
        pri, ctl = pri[3:], ctl[1:]
    queue += [(u, "control") for u in ctl]
    rows, placed = [], []
    for uid, st in queue:
        rows.append(dict(source=uid, section="synovitis_queue", stratum=st, catch=False))
        placed.append(uid)
        if len(placed) % CATCH_EVERY == CATCH_EVERY // 2:
            src = str(rng.choice(placed[:-1]))
            rows.append(dict(source=src, section="synovitis_queue", stratum=s[src], catch=True))
    q = pd.DataFrame(rows)
    taken = set(gold.study_id)
    fake = []
    while len(fake) < int(q.catch.sum()):
        f = "1.2.826.0.1.3680043.8.498." + "".join(map(str, rng.integers(0, 10, 38)))
        if f not in taken:
            fake.append(f)
            taken.add(f)
    q["study_id"] = q.source
    q.loc[q.catch, "study_id"] = fake
    o = pd.concat([gold.drop(columns=["priority", "marker"], errors="ignore").assign(stratum="gold"), q], ignore_index=True)
    o.index.name = "index"
    assert o.study_id.is_unique
    return o


_R = None


def _init():
    global _R
    _R = Renderer()


def render_entry(args):
    """(index, [(sheet, jpeg bytes)], [tile rows], marker) for one entry."""
    index, source, catch, seed = args
    sheets = _R.sheets(source, index)
    marker = None
    if catch:
        rng = np.random.default_rng(seed)
        k = int(rng.integers(len(sheets)))
        name, img, rows = sheets[k]
        tile = rows[int(rng.integers(len(rows)))]["tile"]
        j = [r["tile"] for r in rows].index(tile)
        cols = 3
        r0, c0 = divmod(j, cols)
        x, y = int(rng.integers(80, 320)), int(rng.integers(80, 320))
        X, Y = c0 * (400 + 6) + x, 30 + r0 * (400 + 6) + y                        # grid layout of render.grid
        img = img.copy()
        cv2.rectangle(img, (X - 6, Y - 6), (X + 6, Y + 6), 0, -1)
        cv2.rectangle(img, (X - 4, Y - 4), (X + 4, Y + 4), 255, -1)
        sheets[k] = (name, img, rows)
        marker = dict(tile=tile, x=round(x / 400, 3), y=round(y / 400, 3))
    out = []
    for name, img, rows in sheets:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        out.append((name, buf.tobytes()))
    tiles = [dict(index=index, **{k: v for k, v in r.items() if k != "series"}) for _, _, rows in sheets for r in rows]
    return index, out, tiles, marker


PANELS = {"fluid": (["effusion", "synovitis", "bakers_cyst"], "### Joint effusion", "### Bone contusion"),  # findings, start and end
          "synovitis": (["effusion", "synovitis"], "### Joint effusion", "### Baker's cyst"),             # of their prompt.txt sections
          "synovitis_nf": (["effusion", "synovitis"], "### Joint effusion", "### Baker's cyst"),          # 10-18: synovitis + 4_nonfs sheet
          "bakers": (["bakers_cyst"], "### Baker's cyst", "### Bone contusion"),
          "lateral_oa": (["lateral_oa"], "### Osteoarthritis by compartment", "### Joint effusion")}
# panel -> finding -> attributes added to that finding's schema in the panel (finer measurements for ranking)
PANEL_EXTRA = {"lateral_oa": {"lateral_oa": {"extent_mm": {"$ref": "#/$defs/mm"},
                                             "surfaces": {"type": "array", "uniqueItems": True, "items": {"enum": ["femur", "tibia"]}}}}}


def section(text, start, end):
    """Text from the line starting with `start` up to (not including) the line starting with `end`."""
    a = text.index("\n" + start) + 1
    return text[a:text.index("\n" + end, a) + 1]


def panel_prompt(panel):
    """prompt_<panel>.txt: prompt_<panel>_template.txt filled with the shared sections of prompt.txt (one source for definitions and rules)."""
    p = (SRC / "prompt.txt").read_text()
    parts = {"CONVENTIONS": section(p, "Conventions, true for every tile:", "How to look:"),
             "STATUS": section(p, "## status:", "## certainty:"),
             "CERTAINTY": section(p, "## certainty:", "## Findings"),
             "FINDINGS": section(p, PANELS[panel][1], PANELS[panel][2]),
             "QA": section(p, "## Quality and context (qa)", "## Consistency rules")}
    t = (SRC / f"prompt_{panel}_template.txt").read_text()
    for k, v in parts.items():
        t = t.replace("{" + k + "}", v)
    assert "{" + "FINDINGS}" not in t
    return t


def panel_schema(panel):
    """schema_<panel>.json: schema.json restricted to the panel's findings, no structures, plus panel = <panel>."""
    x = json.loads((SRC / "schema.json").read_text())
    x["title"] = f"Knee MRI image labels, {panel} panel: one object per study"
    x["required"] = ["index", "study_id", "panel", "sheets_viewed", "qa", "findings"]
    x["properties"]["panel"] = {"enum": [panel]}
    del x["properties"]["structures"]
    f = x["properties"]["findings"]
    keys = PANELS[panel][0]
    f["required"] = keys
    f["properties"] = {k: f["properties"][k] for k in keys}
    for k, extra in PANEL_EXTRA.get(panel, {}).items():
        ref = f["properties"][k]["$ref"].split("/")[-1]                     # copy the finding's def and add the panel attributes
        d = json.loads(json.dumps(x["$defs"][ref]))
        d["properties"].update(extra)
        d["required"] = d["required"] + sorted(extra)
        x["$defs"][f"{ref}_{panel}"] = d
        f["properties"][k] = {"$ref": f"#/$defs/{ref}_{panel}"}
    return json.dumps(x, indent=2) + "\n"


def write_prompts(out, studies_csv, tiles_csv):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["prompt.txt", "schema.json", "validate.py"]:
            z.write(SRC / f, f)
        for panel in PANELS:
            z.writestr(f"prompt_{panel}.txt", panel_prompt(panel))
            z.writestr(f"schema_{panel}.json", panel_schema(panel))
        z.write(DEFINITIONS, "definitions.md")
        for f in sorted((SRC / "host_figures").iterdir()):
            z.write(f, f"host_figures/{f.name}")
        z.writestr("studies.csv", studies_csv)
        z.writestr("tiles.csv", tiles_csv)
    (out / "prompts.zip").write_bytes(buf.getvalue())


def prompts_only(out):
    """Rewrite prompts.zip from bundle_src/, keeping its studies.csv and tiles.csv (they describe the image zips, which stay)."""
    with zipfile.ZipFile(out / "prompts.zip") as z:
        studies_csv, tiles_csv = z.read("studies.csv"), z.read("tiles.csv")
    write_prompts(out, studies_csv, tiles_csv)
    mb = round((out / "prompts.zip").stat().st_size / 1e6, 2)
    readme = out / "README.md"
    readme.write_text(re.sub(r"\| prompts\.zip \| [\d.]+ \|", f"| prompts.zip | {mb} |", readme.read_text()))
    print(f"prompts.zip rewritten ({mb} MB)")


def write_sessions(out, n, size):
    blocks = []
    for lo in range(0, n, size):
        hi = min(lo + size, n) - 1
        blocks.append(f"### {lo}-{hi}\nUnzip prompts.zip from the project files, follow its prompt.txt exactly, and deliver img_{lo}-{hi}.zip as it specifies.\n\n"
                      f"Range: {lo}-{hi}\n")
    (out / "session_prompts.txt").write_text("\n".join(blocks))


def write_readme(out, o, zips, sizes):
    ranges = o.reset_index().groupby("section", sort=False)["index"].agg(["min", "max", "size"])
    lines = [f"| {name} | {r['min']}-{r['max']} | {r['size']} |" for name, r in ranges.iterrows()]
    files = [f"| {z} | {sizes[z]} |" for z in zips]
    text = f"""# Image labelling bundle

Built by `work/image_labeling/build_bundle.py` (sheet rendering: `work/image_labeling/render.py`). Upload everything in this folder
except this README and session_prompts.txt to ONE ChatGPT project (Pro limits as of 2026-10-07: 40 files per project, 10 files per
upload, 512 MB per file, 25 GB per account).

| File | MB |
|---|---|
| prompts.zip | {round((out / 'prompts.zip').stat().st_size / 1e6, 2)} |
{chr(10).join(files)}

## Sessions

One chat per range. Paste one block of session_prompts.txt (ranges of {SESSION_SIZE}); nothing else to attach. The model unzips prompts.zip,
extracts only the studies of the range from their image zip, writes one JSON per study, validates and returns `img_FROM-TO.zip`.
Save returned archives to `work/image_labeling/returned/`. If a chat stops early it returns the finished part and a "Next range" line.

Indexes by section (catch items included; they are copies of earlier studies under made-up ids, used for quality checks):

| Section | Indexes | Entries |
|---|---|---|
{chr(10).join(lines)}

Suggested order: gold 0-{ranges.loc['gold', 'max']} first (accuracy against the official gold labels), then the synovitis queue in index
order: every range of it is a stratified random sample (3 priority : 1 control; order.csv `stratum`), so any prefix can be used with
known per-stratum selection rates. Local labellers (Codex) work on ranges of ~300 and write folders of JSON instead of archives.
"""
    (out / "README.md").write_text(text)


def main(limit, out, zip_mb, workers):
    o = order()
    if limit:
        o = o.iloc[:limit]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    jobs = [(i, r.source, bool(r.catch), 1000 + i) for i, r in o.iterrows()]
    tiles, sheets_of, zip_of, markers = [], {}, {}, {}
    limit_bytes = zip_mb * 1e6
    current, current_bytes, lo, zips = None, 0, None, []

    def close(hi):
        current.close()
        name = f"images_{lo:04d}-{hi:04d}.zip"
        (out / "_current.zip").rename(out / name)
        zips.append(name)
        for i in range(lo, hi + 1):
            zip_of[i] = name

    with Pool(workers, initializer=_init) as pool:
        for index, sheets, t, marker in pool.imap(render_entry, jobs, chunksize=4):
            size = sum(len(b) for _, b in sheets)
            if current is not None and current_bytes + size > limit_bytes:
                close(index - 1)
                current = None
            if current is None:
                current, current_bytes, lo = zipfile.ZipFile(out / "_current.zip", "w", zipfile.ZIP_STORED), 0, index
            for name, b in sheets:
                current.writestr(f"{index:04d}/{name}.jpg", b)
            current_bytes += size
            tiles += t
            sheets_of[index] = ";".join(name for name, _ in sheets)
            markers[index] = marker
            if index % 500 == 0:
                print(f"{index}/{len(o)} rendered", flush=True)
    close(index)

    studies = pd.DataFrame({"index": o.index, "study_id": o.study_id, "images_zip": [zip_of[i] for i in o.index],
                            "sheets": [sheets_of[i] for i in o.index]})
    tiles = pd.DataFrame(tiles)[["index", "sheet", "tile", "plane", "sequence", "pos_mm", "zoom"]]
    write_prompts(out, studies.to_csv(index=False), tiles.to_csv(index=False))
    write_sessions(out, len(o), SESSION_SIZE)
    (HERE / "out").mkdir(exist_ok=True)
    o.assign(marker=[json.dumps(markers[i]) if markers[i] else "" for i in o.index]).to_csv(HERE / "out/order.csv" if not limit else out / "order.csv")
    sizes = {z: round((out / z).stat().st_size / 1e6) for z in zips}
    if not limit:
        write_readme(out, o, zips, sizes)
    print(f"{len(o)} entries ({int(o.catch.sum())} catch), {len(zips)} image zips, MB: {sizes}, total {sum(sizes.values())} MB")
    print("sections:", o.groupby("section", sort=False).size().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=REPO / "image_labeling_bundle")
    ap.add_argument("--zip-mb", type=float, default=480)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prompts-only", action="store_true")
    a = ap.parse_args()
    if a.prompts_only:
        prompts_only(a.out)
    else:
        main(a.limit, a.out, a.zip_mb, a.workers)
