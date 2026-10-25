"""Label snapshot v3g = the frozen v3f + the PF batch (USER 10-21 11:2x via Master: 16 chats, p3 sessions 44-59, key block 'pf_depth' in
work/claude_labels_oct20/_key/p3.csv, pf_oa only: 320 unread non-gold old-label PF positives, 160 with text p in [.6, .8) + 160 in [.8, .9),
random seed 0; notes start with the g6 depth record). Copy of build_v3f.py (the closing-batch freeze) with FIRST = 44, block 'pf_depth', the
run file data/image_labels/opus_p3pf.json; Contusion / Fracture cells must stay identical to v3f (asserted). OAI: no; CPU; no gold anywhere.
--- build_v3f.py docstring (the merge rules hold as written, with v3e -> v3f):
Merge per (study, finding): many closing-batch studies already hold a p3 read of OTHER findings, so a study may gain cells; every new cell
must be NaN in v3e (each (study, finding) read once across p2 / p3 / close) and every v3e cell stays IDENTICAL (asserted). Studies new in
the batch get wave 4. Reads: work/claude_labels_oct20/p3_s{36..}/<study>.json keyed by (study, index, session); only the listed findings;
cross-checked against data/image_labels/opus_p3c.json (same status / certainty). Completeness: every closing-batch line needs its JSON
(--partial: a dry run on the reads so far; writes nothing).
Writes data/v3f/ (as v3e) [v3g: + pf_src.csv = the source run of every PF cell]: opus_raw.csv, tgt_p2_bone.csv, ext_p2_main_r.csv, eq_ids_lab.csv, p2_cells_mask.csv (v3e's, the 811 wave 1-2
cells), tgt_bone_v3m.csv (scripts/build_v3b_merged.py v3f), v3f.md5; read-only.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_v3f.py [--partial]
(this file: scripts/build_v3g.py [--partial])"""
import argparse
import hashlib
import json
import shutil
import subprocess
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

LAB, V3E, OUT = REPO / "work/claude_labels_oct20", EXP / "data/v3f", EXP / "data/v3g"   # V3E = the base snapshot (v3f here)
COLS = {"fracture": "Fracture", "contusion": "Contusion", "pf_oa": "PF OA"}
FIRST, LAST, BLOCK, RUNFILE = 44, 59, "pf_depth", "opus_p3pf.json"


def close_reads(partial):
    key = pd.read_csv(LAB / "_key/p3.csv")
    kc = key[(key.session >= FIRST) & (key.session <= LAST)]
    assert (kc.block == BLOCK).all() and (kc.kind == "new").all() and (kc.findings == "pf_oa").all() and len(kc) == 320, "unexpected PF-batch key rows"
    p3c = json.load(open(REPO / "data/image_labels" / RUNFILE))
    cells, missing = [], []
    for _, k in kc.iterrows():
        f = LAB / f"p3_s{k.session}" / f"{k.study_id}.json"
        if not f.exists():
            missing.append((k.session, k.study_id))
            continue
        d = json.load(open(f))
        assert d["study_id"] == k.study_id and int(d["index"]) == int(k["index"]), f
        for fk in k.findings.split(","):
            x = d["findings"].get(fk) or {}
            p = probability(x.get("status"), x.get("certainty")) if x.get("status") else None
            assert p is not None, f"{f}: no usable {fk} read"
            assert partial or k.study_id in p3c, f"{f}: not in {RUNFILE} (rebuild the run file first)"
            if k.study_id in p3c:                                    # the same read in the run file
                y = p3c[k.study_id]["findings"][fk]
                assert (y["status"], y["certainty"]) == (x["status"], x["certainty"]), f"{f}: {fk} differs from {RUNFILE}"
            cells.append((k.study_id, COLS[fk], float(p)))
    assert partial or not missing, f"PF batch incomplete: {len(missing)} of {len(kc)} lines without a read, e.g. {missing[:3]}"
    return pd.DataFrame(cells, columns=["StudyInstanceUID", "col", "p"]), len(kc), len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    v3e = pd.read_csv(V3E / "opus_raw.csv").set_index("StudyInstanceUID")
    C, n_lines, n_miss = close_reads(a.partial)
    assert not C.duplicated(["StudyInstanceUID", "col"]).any(), "a (study, finding) read twice in the batch"
    assert not C.StudyInstanceUID.isin(GOLD.index).any(), "gold study in the batch"
    assert (C.col == "PF OA").all(), "the PF batch carries PF OA cells only"
    T = v3e.copy()
    new = sorted(set(C.StudyInstanceUID) - set(T.index))
    T = pd.concat([T, pd.DataFrame(np.nan, index=pd.Index(new, name="StudyInstanceUID"), columns=T.columns).assign(wave=5)])   # wave 5 = the PF batch
    for s, c, p in C.itertuples(index=False):
        assert pd.isna(T.at[s, c]), f"{s} {c}: already read in v3f"
        T.at[s, c] = p
    T["wave"] = T["wave"].astype(int)
    vc = list(COLS.values())
    read = v3e[vc].notna()                                          # every v3e READ cell stays identical; only NaN cells may be filled
    assert (T.loc[v3e.index, vc][read] == v3e[vc][read]).sum().sum() == read.sum().sum() and (T.loc[v3e.index, "wave"] == v3e.wave).all(), \
        "v3f cells changed"
    assert int(T[vc].notna().sum().sum()) == int(read.sum().sum()) + len(C), "cell count"
    fo = get_folds().iloc[:, 0]
    print(f"OAI: no. v3g{' DRY RUN (--partial)' if a.partial else ''}: closing-batch lines {n_lines}, read {n_lines - n_miss}; new cells "
          + ", ".join(f"{c} {int((C.col == c).sum())} (p >= .5: {int(((C.col == c) & (C.p >= .5)).sum())})" for c in vc)
          + f"; studies {len(v3e)} -> {len(T)} ({len(new)} new, wave 5)")
    for c in vc:
        v = T[c].dropna()
        print(f"  {c}: read {len(v)} (v3f {int(v3e[c].notna().sum())}), p >= .5 {int((v >= .5).sum())} (v3f {int((v3e[c] >= .5).sum())}); per fold read / pos "
              + ", ".join(f"f{F} {int(v.index.isin(fo.index[fo == F]).sum())} / {int((v[v.index.isin(fo.index[fo == F])] >= .5).sum())}" for F in range(5)))
    if a.partial:
        return
    assert not OUT.exists(), f"{OUT} exists (frozen snapshots are never overwritten)"
    OUT.mkdir()
    T.to_csv(OUT / "opus_raw.csv")
    T[["Fracture", "Contusion"]].to_csv(OUT / "tgt_p2_bone.csv")
    t = probability_table("v1")["pf_oa"].reindex(T.index)
    r = T[["Contusion", "Fracture", "PF OA"]].copy()
    r["PF OA"] = 0.5 * T["PF OA"] + 0.5 * t.fillna(T["PF OA"])
    r.to_csv(OUT / "ext_p2_main_r.csv")
    old = pd.read_csv(V3E / "ext_p2_main_r.csv").set_index("StudyInstanceUID")
    keep = old.index[old["PF OA"].notna()]                         # every v3f PF yardstick value stays identical; only new PF cells are added
    assert np.allclose(r.loc[keep, "PF OA"], old.loc[keep, "PF OA"]), "PF blend changed on v3f-read studies"
    assert np.allclose(r.loc[old.index, ["Contusion", "Fracture"]].fillna(-1), old[["Contusion", "Fracture"]].fillna(-1)), "C / F changed"
    pd.DataFrame({"StudyInstanceUID": T.index[T[vc].notna().any(axis=1)]}).to_csv(OUT / "eq_ids_lab.csv", index=False)
    # Master 10-21 11:4x PF PROTOCOL SHIFT: the run of every PF cell (p2 = the 3-finding panel, stricter on vague reports; p3 = wave 3;
    # p3pf = this PF-only batch with the depth record), so a yardstick variant can treat p2-read cells differently (with a row showing it)
    src = {}
    for run in ("p2", "p3", "p3pf"):
        d = json.load(open(REPO / "data/image_labels" / f"opus_{run}.json"))
        for sid in d:
            if (d[sid].get("findings", {}).get("pf_oa") or {}).get("status"):
                assert sid not in src, f"{sid}: PF read in two runs ({src[sid]}, {run})"
                src[sid] = run
    pfc = T.index[T["PF OA"].notna()]
    assert set(src) >= set(pfc) and all(src[s_] == "p3pf" for s_ in C.StudyInstanceUID), "every PF cell needs exactly one source run"
    pd.Series({s_: src[s_] for s_ in pfc}, name="pf_src").rename_axis("StudyInstanceUID").to_csv(OUT / "pf_src.csv")
    print("  PF cell sources: " + ", ".join(f"{k} {v}" for k, v in pd.Series({s_: src[s_] for s_ in pfc}).value_counts().items()))
    shutil.copy(V3E / "p2_cells_mask.csv", OUT / "p2_cells_mask.csv")
    subprocess.run([sys.executable, str(EXP / "scripts/build_v3b_merged.py"), "v3g"], check=True)
    a3, b3 = (pd.read_csv(EXP / f"data/{s}/tgt_bone_v3m.csv").set_index("StudyInstanceUID") for s in ("v3f", "v3g"))
    assert np.allclose(a3.reindex(b3.index).fillna(-1), b3.fillna(-1)), "tgt_bone_v3m changed (C / F must not)"
    md5 = "".join(f"{hashlib.md5(f.read_bytes()).hexdigest()}  data/v3g/{f.name}\n" for f in sorted(OUT.glob("*.csv")))
    (OUT / "v3g.md5").write_text(md5)
    for f in OUT.iterdir():
        f.chmod(0o444)
    print(f"FROZEN data/v3g ({len(list(OUT.iterdir()))} files, read-only)\n{md5}")


if __name__ == "__main__":
    main()
