"""Score returned LLM label files (work/text_labeling/returned/<range>/; set RETURNED to score another folder) against my hand labels (work/text_gold).
My reason codes are mapped to bundle statuses; statuses are grouped P (meets), N (below/excluded/denied/structure_normal/hedged),
U (unsettled/conflict), S (not_mentioned/general_normal). Writes out/score_returned.txt and out/disagreements.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
TG = REPO / "work/text_gold"
CLS = {"ACL": "acl", "MCL": "mcl", "MM": "medial_meniscus", "LM": "lateral_meniscus", "MOA": "medial_oa", "LOA": "lateral_oa",
       "PFOA": "pf_oa", "EFF": "effusion", "SYN": "synovitis", "BAK": "bakers_cyst", "CON": "contusion", "FX": "fracture"}
MINE = {"meets": "meets", "present_no_threshold": "meets", "indeterminate": "unsettled", "hedged": "hedged", "below": "below",
        "excluded": "excluded", "absent": "denied", "normal_structure": "structure_normal", "silent": "not_mentioned", "conflict": "conflict"}
GROUP = {"meets": "P", "below": "N", "excluded": "N", "denied": "N", "structure_normal": "N", "hedged": "N",
         "unsettled": "U", "conflict": "U", "not_mentioned": "S", "general_normal": "S"}
EXAMPLES = {6, 67, 105, 120, 189, 198}                              # shown in prompt.txt, left out of scoring
RETURNED = sys.argv[1] if len(sys.argv) > 1 else "returned_v0"


def my_labels():
    rows = []
    for labels, ids in [(TG / "labels_all.txt", TG / "blind_ids.csv"), (TG / "nongold/labels/labels_all.txt", TG / "nongold/sample_ids.csv")]:
        m = pd.read_csv(ids).set_index("tid").StudyInstanceUID
        for line in open(labels, encoding="utf-8"):
            if line.strip():
                tid, c, _, reason, quote, note = line.rstrip("\n").split("|")
                rows.append({"study_id": m[tid], "tid": tid, "finding": CLS[c], "mine": MINE[reason], "my_reason": reason, "my_quote": quote, "my_note": note})
    return pd.DataFrame(rows)


def main():
    reports = pd.read_csv(REPO / "text_labeling_bundle/reports.csv")
    idx = reports.set_index("study_id")["index"]
    mine = my_labels()
    mine["index"] = mine.study_id.map(idx)
    rows, batches = [], {}
    for d in sorted((HERE / RETURNED).iterdir()):
        if not d.is_dir():
            continue
        lo = int(d.name.split("-")[0])
        for f in d.glob("*.json"):
            x = json.loads(f.read_text(encoding="utf-8"))
            batches[x["index"]] = d.name
            for k, v in x["findings"].items():
                rows.append({"index": x["index"], "batch": d.name, "pos": x["index"] - lo + 1, "finding": k, "llm": v["status"], "llm_quote": v["quote"],
                             "missing": ",".join(v["missing"]), **{f"a_{a}": json.dumps(b, ensure_ascii=False) for a, b in v.items() if a not in ("status", "missing", "quote")},
                             **{f"s_{a}": b for a, b in x["study"].items()}})
    llm = pd.DataFrame(rows)
    df = llm.merge(mine, on=["index", "finding"], how="inner")                 # only my 208 hand-labelled studies
    df = df[~df["index"].isin(EXAMPLES)].copy()
    df["g_llm"], df["g_mine"] = df.llm.map(GROUP), df.mine.map(GROUP)
    out = [f"== {df['index'].nunique()} studies scored from {df.batch.nunique()} archives; prompt examples excluded; {len(df)} cells"]

    out.append("\n== exact status agreement: " + f"{(df.llm == df.mine).mean():.3f}")
    out.append("== group agreement (P / N / U / S): " + f"{(df.g_llm == df.g_mine).mean():.3f}")
    pn = df[df.g_llm.isin(["P", "N"]) & df.g_mine.isin(["P", "N"])]
    out.append(f"== polarity on cells both call P or N: {len(pn)} cells, agree {(pn.g_llm == pn.g_mine).mean():.3f}, "
               f"LLM P / mine N: {((pn.g_llm == 'P') & (pn.g_mine == 'N')).sum()}, LLM N / mine P: {((pn.g_llm == 'N') & (pn.g_mine == 'P')).sum()}")
    out.append("\n== group confusion (rows mine, columns LLM)")
    out.append(pd.crosstab(df.g_mine, df.g_llm, margins=True).to_string())
    out.append("\n== status confusion (rows mine, columns LLM)")
    out.append(pd.crosstab(df.mine, df.llm).to_string())
    out.append("\n== group agreement per finding")
    out.append(df.groupby("finding").apply(lambda d: pd.Series({"cells": len(d), "group_agree": round((d.g_llm == d.g_mine).mean(), 3),
                                                                "exact": round((d.llm == d.mine).mean(), 3)}), include_groups=False).to_string())

    out.append("\n== by batch and position (does quality drop late in a batch?)")
    df["qlen"] = df.llm_quote.str.len()
    df["pos_bin"] = pd.cut(df.pos, [0, 10, 25, 50], labels=["1-10", "11-25", "26-50"])
    out.append(df.groupby(["batch", "pos_bin"], observed=True).apply(lambda d: pd.Series({
        "studies": d["index"].nunique(), "group_agree": round((d.g_llm == d.g_mine).mean(), 3), "exact": round((d.llm == d.mine).mean(), 3),
        "llm_not_mentioned": round((d.llm == "not_mentioned").mean(), 3), "my_not_mentioned": round((d.mine == "not_mentioned").mean(), 3),
        "mean_quote_len": round(d.qlen[d.llm != "not_mentioned"].mean())}), include_groups=False).to_string())
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/score_returned.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    dis = df[df.g_llm != df.g_mine].sort_values(["finding", "index"])
    lines = [f"{len(dis)} cells where the group differs\n"]
    for _, r in dis.iterrows():
        lines.append(f"[{r['index']} {r.tid} {r.finding}] mine={r.mine} ({r.my_reason}) | LLM={r.llm} missing={r.missing}\n"
                     f"   mine: {r.my_quote} | {r.my_note}\n   LLM:  {r.llm_quote}")
    (HERE / "out/disagreements.txt").write_text("\n".join(lines) + "\n")
    df.to_csv(HERE / "out/scored_cells.csv", index=False)


if __name__ == "__main__":
    main()
