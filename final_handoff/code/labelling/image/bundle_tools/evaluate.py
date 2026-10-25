"""Score image-label runs: archives in work/image_labeling/returned/ and folders of local labellers (work/codex_labels_oct7/<run>/,
work/claude_labels_oct7/<run>/, work/claude_labels_fable51_oct7/<run>/; run name = <labeller>_<folder without its range>, e.g. codex_fluid = fluid_0-29 + fluid_30-59).

PYTHONPATH=. conda run -n rsna python work/image_labeling/evaluate.py            # every run found
PYTHONPATH=. conda run -n rsna python work/image_labeling/evaluate.py goldA goldB

Archive names: <run>_img_FROM-TO.zip (gold runs) or img_FROM-TO.zip (run ""). Per run: files, validate.py problems, minutes per study
(from the zip's file times), catch items (marker found; status agreement with the source study), and on gold studies: status counts
per finding for gold positives and negatives, and macro AUC (mapping.py probabilities) next to text v1, steven v4 and R2 on the same studies.
"""
import json
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common.labels import external, gold_labels, probability_table
from common.paths import FINDING_OF, KEYS, REPO
from mapping import probability

HERE = Path(__file__).parent
RETURNED = HERE / "returned"
LABEL_DIRS = sorted(REPO.glob("work/*_labels*_oct7"))                  # local labellers: codex, claude, claude_fable51, ...
BUNDLE = REPO / "image_labeling_bundle"
R2_RUN = REPO / "experiments/v0/oct7_baseline/runs/effb0_lr1e-3_R"
SHORT = {"meets": "mt", "borderline": "bd", "possible": "ps", "below": "bl", "excluded": "ex", "normal": "nm", "not_assessable": "na"}


def validator():
    """validate.check_file from the uploaded prompts.zip (needs its studies.csv and tiles.csv next to it)."""
    tmp = Path(tempfile.mkdtemp())
    zipfile.ZipFile(BUNDLE / "prompts.zip").extractall(tmp)
    sys.path.insert(0, str(tmp))
    import validate
    return validate.check_file


def sources():
    """(run, source name, expected file count, [(file name, bytes, time written)]) for every archive and labeller run folder."""
    for z in sorted(RETURNED.glob("*.zip")):
        m = re.fullmatch(r"(?:(\w+?)_)?img_(\d+)-(\d+)\.zip", z.name)
        assert m, z.name
        with zipfile.ZipFile(z) as zf:
            members = [(Path(i.filename).name, zf.read(i), datetime(*i.date_time)) for i in zf.infolist() if i.filename.endswith(".json")]
        yield m.group(1) or "", z.name, int(m.group(3)) - int(m.group(2)) + 1, members
    for d in LABEL_DIRS:
        for folder in sorted(f for f in d.glob("*") if f.is_dir() and not f.name.startswith("_")):
            m = re.search(r"(\d+)-(\d+)$", folder.name)
            members = [(f.name, f.read_bytes(), datetime.fromtimestamp(f.stat().st_mtime)) for f in sorted(folder.glob("*.json"))]
            if not members:                                                     # folder created but not started
                continue
            run = f"{d.name.replace('_labels', '').replace('_oct7', '')}_{re.sub(r'_\d+-\d+$', '', folder.name)}"  # range folders merge
            yield (run, str(folder.relative_to(REPO)),
                   int(m.group(2)) - int(m.group(1)) + 1 if m else len(members), members)


def load(runs):
    check = validator()
    rows, info = [], []
    for run, name, expected, members in sources():
        if runs and run not in runs:
            continue
        problems = 0
        with tempfile.TemporaryDirectory() as tmp:
            for fname, data, _ in members:
                path = Path(tmp) / fname
                path.write_bytes(data)
                problems += bool(check(path))
                x = json.loads(data)
                for k, f in x["findings"].items():                         # a panel file holds only some findings
                    rows.append(dict(run=run, index=x["index"], finding=k, status=f["status"], certainty=f["certainty"],
                                     p=probability(f["status"], f["certainty"]), marks=json.dumps(x["qa"]["marks"])))
        times = sorted(t for _, _, t in members)
        minutes = (times[-1] - times[0]).total_seconds() / 60 / max(len(times) - 1, 1)
        info.append(dict(run=run, archive=name, files=len(members), expected=expected, with_problems=problems,
                         min_per_study=round(minutes, 1)))
    return pd.DataFrame(rows), pd.DataFrame(info)


def r2_gold():
    parts = [pd.read_parquet(f) for f in sorted(R2_RUN.glob("fold*/gold.parquet"))]
    return pd.concat([d[d.epoch == d.epoch.max()].drop(columns="epoch") for d in parts]).groupby(level=0).mean()


def macro_auc(y, s):
    """Mean AUC over findings that have both classes; also the number of such findings."""
    aucs = [roc_auc_score(y[c], s[c]) for c in y if y[c].nunique() == 2 and s[c].notna().all()]
    return round(float(np.mean(aucs)), 3), len(aucs)


def catch_report(d, order):
    out = []
    for i, r in order[order.catch].iterrows():
        if i not in d.index.get_level_values("index"):
            continue
        src = order.index[(order.study_id == r.source) & ~order.catch][0]
        marker = json.loads(r.marker)
        marks = json.loads(d.loc[i, "marks"].iloc[0])                      # same qa.marks on every finding row of a study
        found = any(m["tile"] == marker["tile"] and abs(m["x"] - marker["x"]) < 0.1 and abs(m["y"] - marker["y"]) < 0.1 for m in marks)
        line = f"  catch {i} (copy of {src}): marker on {marker['tile']} {'found' if found else 'MISSED'}"
        if src in d.index.get_level_values("index"):
            same = (d.loc[i].status == d.loc[src].status).sum()
            line += f", status agrees on {same}/{len(d.loc[i])}"
        out.append(line)
    return out


def gold_report(d, order):
    gold = gold_labels().rename(columns={v: k for k, v in FINDING_OF.items()})
    idx = [i for i in d.index.get_level_values("index").unique() if order.section[i] == "gold" and not order.catch[i]]
    if not idx:
        return []
    uid = order.source[idx].values
    keys = [k for k in KEYS if k in d.index.get_level_values("finding")]
    y = gold.loc[uid, keys]
    status = d.status.unstack().loc[idx, keys].set_axis(uid)
    cert = d.certainty.unstack().loc[idx, keys].set_axis(uid)
    p = d.p.unstack().loc[idx, keys].set_axis(uid).astype(float)
    code = status.replace(SHORT) + "-" + cert.fillna("").map(lambda c: c[:1])
    out = [f"  gold studies: {len(idx)}", f"  {'finding':17} {'n+':>3}  positives -> status                negatives -> status"]
    for k in keys:
        pos = code[k][y[k] == 1].value_counts().to_dict()
        neg = code[k][y[k] == 0].value_counts().to_dict()
        fmt = lambda c: " ".join(f"{s}:{n}" for s, n in sorted(c.items()))
        out.append(f"  {k:17} {int(y[k].sum()):>3}  {fmt(pos):34} {fmt(neg)}")
    text = probability_table("v1").loc[uid, keys].fillna(0.15)
    steven = external("steven_v4_blend").loc[uid].rename(columns={v: k for k, v in FINDING_OF.items()})[keys]
    r2 = r2_gold().loc[uid].rename(columns={v: k for k, v in FINDING_OF.items()})[keys]
    out.append(f"  macro AUC (findings with both classes, n): VLM {macro_auc(y, p.fillna(0.15))}  text v1 {macro_auc(y, text)}  "
               f"steven v4 {macro_auc(y, steven)}  R2 {macro_auc(y, r2)}")
    return out


def main(runs):
    d, info = load(set(runs))
    order = pd.read_csv(HERE / "out/order.csv", index_col="index")
    print(info.to_string(index=False))
    for run, g in d.groupby("run"):
        g = g.drop_duplicates(["index", "finding"], keep="last").set_index(["index", "finding"])
        print(f"\n== run {run or '(production)'}")
        print("\n".join(catch_report(g, order) + gold_report(g, order)))


if __name__ == "__main__":
    main(sys.argv[1:])
