"""'MORE LABELS' target set v3 = v2 (811 reads, frozen) + Phase 3 NEW reads (Master 10-20 17:2x; USER: "at least one class 200+ positives as soon
as possible"; Phase 3 = panel p3, only the listed finding(s) per study, Contusion block first; OAI: no; CPU). Writes data/v3/ (never over v2):
 opus_raw.csv          raw Opus p (Fracture / Contusion / PF OA; NaN = not read for that finding) + wave (1 / 2 = v2, 3 = Phase 3 new)
 tgt_p2_bone.csv       Fracture / Contusion (sparse; bone image heads / reads)
 ext_p2_main_r.csv     main-recipe target rows: Contusion / Fracture = Opus, PF OA = 0.5 Opus + 0.5 text v1 (text missing -> Opus) where read
 eq_ids_lab.csv        every study with at least one read (samp_ids / train_ids / opus_csv)
Phase 3 reader: work/claude_labels_oct20/p3_s*/<study>.json, key-checked against _key/p3.csv (index + study); kind 'reread' (24 blind protocol
re-reads of p2 studies) is SKIPPED - the p2 read stays the target, never a second one; kind 'new' must not overlap v2; no gold anywhere.
CONSISTENCY: v2's 811 studies are present with IDENTICAL values (asserted against data/v2/opus_raw.csv).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_v3.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "work/image_labeling"), str(EXP / "scripts")]
from common.folds import get_folds  # noqa: E402
from common.labels import probability_table  # noqa: E402
from lib import GOLD  # noqa: E402
from mapping import probability  # noqa: E402

LAB, OUT = REPO / "work/claude_labels_oct20", EXP / "data/v3"
COLS = {"fracture": "Fracture", "contusion": "Contusion", "pf_oa": "PF OA"}


def p3_reads(v2_ids):
    key = pd.read_csv(LAB / "_key/p3.csv")
    rows, skipped = {}, 0
    for f in sorted(LAB.glob("p3_s*/*.json")):
        d = json.load(open(f))
        sid = d["study_id"]
        assert f.stem == sid, f
        k = key[(key.study_id == sid) & (key["index"] == int(d["index"]))]
        assert len(k) == 1, (sid, d.get("index"))
        if k.kind.iloc[0] == "reread":                   # blind protocol re-read: the p2 read stays the only target
            assert sid in v2_ids, f"re-read of a non-v2 study {sid}"
            skipped += 1
            continue
        assert k.kind.iloc[0] == "new" and sid not in v2_ids and sid not in rows, (sid, k.kind.iloc[0])
        listed = set(k.findings.iloc[0].split(","))
        r = {}
        for fk, col in COLS.items():
            x = d["findings"].get(fk) or {}
            r[col] = float(probability(x["status"], x["certainty"])) if fk in listed and x.get("status") and probability(x["status"], x["certainty"]) is not None else np.nan
        rows[sid] = r
    T = pd.DataFrame.from_dict(rows, orient="index", columns=list(COLS.values())).rename_axis("StudyInstanceUID")
    return T, skipped


def main():
    OUT.mkdir(exist_ok=True)
    v2 = pd.read_csv(EXP / "data/v2/opus_raw.csv").set_index("StudyInstanceUID")
    T3, skipped = p3_reads(set(v2.index))
    T = pd.concat([v2, T3.assign(wave=3)])
    assert not T.index.duplicated().any() and not T.index.isin(GOLD.index).any(), "duplicate / gold study"
    assert (T.loc[v2.index, list(COLS.values())].fillna(-1) == v2[list(COLS.values())].fillna(-1)).all().all(), "v2 values changed"
    T.to_csv(OUT / "opus_raw.csv")
    T[["Fracture", "Contusion"]].to_csv(OUT / "tgt_p2_bone.csv")
    t = probability_table("v1")["pf_oa"].reindex(T.index)
    r = T[["Contusion", "Fracture", "PF OA"]].copy()
    r["PF OA"] = 0.5 * T["PF OA"] + 0.5 * t.fillna(T["PF OA"])
    r.to_csv(OUT / "ext_p2_main_r.csv")
    pd.DataFrame({"StudyInstanceUID": T.index[T[list(COLS.values())].notna().any(axis=1)]}).to_csv(OUT / "eq_ids_lab.csv", index=False)
    fo = get_folds().iloc[:, 0]
    print(f"OAI: no. v3 set: {len(T)} studies = v2 {len(v2)} (identical) + Phase 3 new {len(T3)} (re-reads skipped {skipped}); written to data/v3/")
    for c in COLS.values():
        v = T[c].dropna()
        print(f"  {c}: read {len(v)} (new {int(T3[c].notna().sum())}), p >= .5 {int((v >= .5).sum())} (v2 {int((v2[c] >= .5).sum())}); per fold read / pos "
              + ", ".join(f"f{F} {int(v.index.isin(fo.index[fo == F]).sum())} / {int((v[v.index.isin(fo.index[fo == F])] >= .5).sum())}" for F in range(5)))


if __name__ == "__main__":
    main()
