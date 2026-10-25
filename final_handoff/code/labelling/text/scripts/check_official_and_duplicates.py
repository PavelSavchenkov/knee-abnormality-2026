"""v0 labels vs official gold (58 studies) and consistency between studies whose report text is identical.
Status order is fixed a priori (same order as text_gold section E), nothing is fitted on gold. Writes out/official_and_duplicates.txt."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from collect_v0 import load                                        # noqa: E402

REPO = Path("/home/pavel/Programming/knee-abnormality-2026")
L = {"acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus", "medial_oa": "Medial OA", "lateral_oa": "Lateral OA",
     "pf_oa": "PF OA", "effusion": "Effusion", "synovitis": "Synovitis", "bakers_cyst": "Baker's", "contusion": "Contusion", "fracture": "Fracture"}
ORDER = {"denied": 0, "structure_normal": 0, "excluded": 0, "general_normal": 0.5, "below": 1, "not_mentioned": 2, "hedged": 3, "conflict": 3, "unsettled": 4, "meets": 6}
def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    train = pd.read_csv(REPO / "data/original_data/train.csv").set_index("StudyInstanceUID")
    gold = train[train.ACL.notna()]
    v4 = pd.read_csv(REPO / "work/eda/ext_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    docs = load()


    g = [s for s in gold.index if s in docs]
    status = pd.DataFrame({L[k]: [ORDER[docs[s][1]["findings"][k]["status"]] for s in g] for k in L}, index=g)
    scores = {"v0 status order": status, "steven v4": v4.loc[g, list(L.values())], "v0 status, steven v4 inside a status": status + 0.99 * v4.loc[g, list(L.values())]}
    macro = lambda sc, idx: np.mean([auc(gold.loc[idx, f].astype(int), sc.loc[idx, f]) for f in L.values()])
    out = [f"== official gold: {len(g)} of {len(gold)} gold studies labelled; macro AUC (paired bootstrap over studies, 1000 draws)"]
    rng = np.random.default_rng(0)
    draws = [rng.choice(g, len(g)) for _ in range(1000)]
    draws = [d for d in draws if all(gold.loc[d, f].nunique() == 2 for f in L.values())]
    base = np.array([macro(scores["steven v4"], d) for d in draws])
    for k, sc in scores.items():
        b = np.array([macro(sc, d) for d in draws])
        diff = b - base
        out.append(f"   {k:42s} {macro(sc, g):.4f}   minus steven v4: {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]")
    out.append("   per finding (v0 status order / steven v4): " + ", ".join(
        f"{f} {auc(gold.loc[g, f].astype(int), status[f]):.3f}/{auc(gold.loc[g, f].astype(int), v4.loc[g, f]):.3f}" for f in L.values()))

    rep = pd.read_csv(REPO / "text_labeling_bundle/reports.csv").set_index("study_id")
    rep = rep[rep.index.isin(docs)]
    rep["key"] = rep.report.str.split().str.join(" ") + "|" + rep.knee_side
    pairs = [grp.index.tolist() for _, grp in rep.groupby("key") if len(grp) > 1]
    same = [docs[a][1]["findings"][k]["status"] == docs[b][1]["findings"][k]["status"] for p in pairs for a, b in zip(p, p[1:]) for k in L]
    out.append(f"\n== identical report text and knee side: {len(pairs)} groups; status identical in {np.mean(same):.3f} of {len(same)} finding pairs")
    (HERE / "out/official_and_duplicates.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
