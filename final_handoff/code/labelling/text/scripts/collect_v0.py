"""Collect returned label files (work/text_labeling/returned/<FROM-TO>/) into one table; report coverage, duplicates and per-batch statistics."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
reports = pd.read_csv(REPO / "text_labeling_bundle/reports.csv")


def load(folder="returned_v0"):
    docs = {}
    for d in sorted((HERE / folder).iterdir()):
        if d.is_dir():
            for f in d.glob("*.json"):
                x = json.loads(f.read_text(encoding="utf-8"))
                assert x["study_id"] not in docs, f"{x['study_id']} in two archives"
                docs[x["study_id"]] = (d.name, x)
    return docs


if __name__ == "__main__":
    docs = load()
    have = {x["index"] for _, x in docs.values()}
    missing = sorted(set(reports["index"]) - have)
    print(f"studies labelled: {len(docs)} of {len(reports)}; missing: {len(missing)}", f"({missing[0]}-{missing[-1]})" if missing else "")
    rows = [{"batch": b, "index": x["index"], "finding": k, "status": v["status"], "qlen": len(v["quote"])}
            for b, x in docs.values() for k, v in x["findings"].items()]
    df = pd.DataFrame(rows)
    st = df.groupby("batch").apply(lambda d: pd.Series({
        "studies": d["index"].nunique(), "meets": (d.status == "meets").mean(), "not_mentioned": (d.status == "not_mentioned").mean(),
        "unsettled": (d.status == "unsettled").mean(), "general_normal": (d.status == "general_normal").mean(),
        "quote_len": d.qlen[d.status != "not_mentioned"].mean()}), include_groups=False)
    print("\nper-batch rates: mean, sd, min, max")
    print(st.drop(columns="studies").agg(["mean", "std", "min", "max"]).round(3).to_string())
    z = (st - st.mean()) / st.std()
    out = z.drop(columns="studies").abs().max(axis=1)
    print("\nbatches with any rate more than 2.5 sd from the mean:")
    print(st[out > 2.5].round(3).to_string() if (out > 2.5).any() else "  none")
    print("\noverall status counts:", df.status.value_counts().to_dict())
