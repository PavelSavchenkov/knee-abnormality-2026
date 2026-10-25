"""Label snapshot v3h = the frozen v3g with the PF reads of the 249 p2-read text-'unsettled' studies REPLACED by PF-only re-reads (Master
10-21 13:3x; the user's spare limits): p3 sessions 60-72, key block 'pf_reread' (kind new, pf_oa only, exactly the v3g_dp2u-dropped cells),
the p3pf protocol (native zoom + the g6 depth record); run file data/image_labels/opus_p3pfr.json (POOL_ONLY). Every other cell = v3g
(asserted); C / F targets identical; pf_src.csv: the 249 -> 'p3pfr'. Prints the PAIRED protocol shift on the same 249 studies (p2 vs
p3pfr: Opus+ rate, mean p, same side of .5). OAI: no; CPU; no gold anywhere. --partial = dry run on the reads so far (writes nothing).
Writes data/v3h/ (as v3g): opus_raw.csv, tgt_p2_bone.csv, ext_p2_main_r.csv (PF blend 0.5 Opus + 0.5 text), eq_ids_lab.csv,
p2_cells_mask.csv, tgt_bone_v3m.csv, pf_src.csv, v3h.md5; read-only.
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_v3h.py [--partial]"""
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
from common.labels import probability_table  # noqa: E402
from lib import GOLD  # noqa: E402
from mapping import probability  # noqa: E402

LAB, BASE, OUT = REPO / "work/claude_labels_oct20", EXP / "data/v3g", EXP / "data/v3h"
FIRST, LAST, BLOCK, RUNFILE, N = 60, 72, "pf_reread", "opus_p3pfr.json", 249


def reread(partial):
    key = pd.read_csv(LAB / "_key/p3.csv")
    kc = key[(key.session >= FIRST) & (key.session <= LAST)]
    assert len(kc) == N and (kc.block == BLOCK).all() and (kc.kind == "new").all() and (kc.findings == "pf_oa").all(), "unexpected re-read key rows"
    run = json.load(open(REPO / "data/image_labels" / RUNFILE))
    cells, missing = {}, []
    for _, k in kc.iterrows():
        f = LAB / f"p3_s{k.session}" / f"{k.study_id}.json"
        if not f.exists():
            missing.append(k.study_id)
            continue
        d = json.load(open(f))
        assert d["study_id"] == k.study_id and int(d["index"]) == int(k["index"]), f
        x = d["findings"].get("pf_oa") or {}
        p = probability(x.get("status"), x.get("certainty")) if x.get("status") else None
        assert p is not None, f"{f}: no usable pf_oa read"
        assert partial or k.study_id in run, f"{f}: not in {RUNFILE} (rebuild the run file first)"
        if k.study_id in run:
            y = run[k.study_id]["findings"]["pf_oa"]
            assert (y["status"], y["certainty"]) == (x["status"], x["certainty"]), f"{f}: pf_oa differs from {RUNFILE}"
        cells[k.study_id] = float(p)
    assert partial or not missing, f"re-read batch incomplete: {len(missing)} of {N}"
    return pd.Series(cells, dtype=float), set(kc.study_id), len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    base = pd.read_csv(BASE / "opus_raw.csv").set_index("StudyInstanceUID")
    src = pd.read_csv(BASE / "pf_src.csv").set_index("StudyInstanceUID")["pf_src"]
    new, keyset, n_miss = reread(a.partial)
    dp = pd.read_csv(EXP / "data/v3g_dp2u/ext_p2_main_r.csv").set_index("StudyInstanceUID")["PF OA"]
    dropped = set(dp.index[dp.isna() & base["PF OA"].reindex(dp.index).notna()])
    assert keyset == dropped and not keyset & set(GOLD.index), "the re-read set must be exactly the v3g_dp2u-dropped p2-unsettled cells"
    assert src.reindex(list(keyset)).eq("p2").all(), "re-read cells must be p2 reads"
    old = base.loc[new.index, "PF OA"]
    same = int(((old >= .5) == (new >= .5)).sum())
    print(f"OAI: no. v3h{' DRY RUN (--partial)' if a.partial else ''}: re-read lines {N}, read {N - n_miss}. PAIRED PROTOCOL SHIFT on the same {len(new)} "
          f"studies (p2 3-finding panel -> p3pfr PF-only): Opus+ {int((old >= .5).sum())} -> {int((new >= .5).sum())}, mean p {old.mean():.3f} -> "
          f"{new.mean():.3f}, same side of .5 {same} / {len(new)}")
    T = base.copy()
    T.loc[new.index, "PF OA"] = new
    keep = T.index.difference(new.index)
    assert T.loc[keep].fillna(-1).equals(base.loc[keep].fillna(-1)) and T[["Contusion", "Fracture", "wave"]].fillna(-1).equals(
        base[["Contusion", "Fracture", "wave"]].fillna(-1)), "only the re-read PF cells may change"
    v = T["PF OA"].dropna()
    print(f"  PF OA: read {len(v)} (v3g {int(base['PF OA'].notna().sum())}), p >= .5 {int((v >= .5).sum())} (v3g {int((base['PF OA'] >= .5).sum())})")
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
    g = pd.read_csv(BASE / "ext_p2_main_r.csv").set_index("StudyInstanceUID")
    assert np.allclose(r.loc[keep, "PF OA"].fillna(-1), g.loc[keep, "PF OA"].fillna(-1)), "PF blend changed outside the re-read cells"
    shutil.copy(BASE / "eq_ids_lab.csv", OUT / "eq_ids_lab.csv")
    shutil.copy(BASE / "p2_cells_mask.csv", OUT / "p2_cells_mask.csv")
    s2 = src.copy()
    s2.loc[list(keyset)] = "p3pfr"
    s2.to_frame().to_csv(OUT / "pf_src.csv")
    subprocess.run([sys.executable, str(EXP / "scripts/build_v3b_merged.py"), "v3h"], check=True)
    a3, b3 = (pd.read_csv(EXP / f"data/{s}/tgt_bone_v3m.csv").set_index("StudyInstanceUID") for s in ("v3g", "v3h"))
    assert np.allclose(a3.reindex(b3.index).fillna(-1), b3.fillna(-1)), "tgt_bone_v3m changed (C / F must not)"
    md5 = "".join(f"{hashlib.md5(f.read_bytes()).hexdigest()}  data/v3h/{f.name}\n" for f in sorted(OUT.glob("*.csv")))
    (OUT / "v3h.md5").write_text(md5)
    for f in OUT.iterdir():
        f.chmod(0o444)
    print(f"FROZEN data/v3h ({len(list(OUT.iterdir()))} files, read-only)\n{md5}")


if __name__ == "__main__":
    main()
