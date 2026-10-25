"""Phase 3 PF RE-READ batch (USER 10-21 13:2x: "so what about PF? there is a chance to get gains there? i still have spare limits"). The 249
p2-read PF cells of text v1 'unsettled' studies are the one PF subset with evidence of wrong labels: the 3-finding p2 protocol called 16 %
Opus+ there vs 34-47 % for the PF-only depth protocol (pf_protocol_shift.txt), and every PF member scores negative on that subset on every
fold although it trains on it (Experts pf_within_groups.py). This batch re-reads exactly those 249 (= the cells the v3g_dp2u variant drops)
with the p3pf protocol (pf_oa only, depth record), so the PF target uses ONE protocol on the label-changing strata; the paired p2 / p3pf
reads also measure the protocol shift on the same studies. Lines kind 'new' (not p3 re-reads: p2 reads are outside the p3 key), block
pf_reread, random order (seed 1), dealt round-robin to sessions 60-72 (chats 96-108), <= 20 lines each. No gold, no anchors. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/build_phase3_pfr.py"""
import numpy as np
import pandas as pd

from common.labels import gold_labels
from common.paths import REPO

D = REPO / "work/claude_labels_oct20"
E = REPO / "experiments/v3/oct16_experts/data"
FIRST, CHATS = 60, 13
key = pd.read_csv(D / "_key/p3.csv")
assert key.session.max() < FIRST, "sessions >= 60 already exist"
a = pd.read_csv(E / "v3g/ext_p2_main_r.csv").set_index("StudyInstanceUID")["PF OA"]
b = pd.read_csv(E / "v3g_dp2u/ext_p2_main_r.csv").set_index("StudyInstanceUID")["PF OA"]
pool = sorted(a[a.notna() & b.reindex(a.index).isna()].index)
src = pd.read_csv(E / "v3g/pf_src.csv").set_index("StudyInstanceUID").pf_src
assert len(pool) == 249 and (src[pool] == "p2").all() and not set(pool) & set(gold_labels().index)
studies = pd.read_csv(D / "_prompts/studies.csv").drop_duplicates("study_id").set_index("study_id")
meta = pd.read_csv(REPO / "data/meta/studies.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")
anchors = {2507, 3297, 3831}
listed_pf = set(key[key.findings.str.contains("pf_oa")].study_id)
ok = [s for s in pool if s in studies.index and studies.at[s, "index"] not in anchors and s not in listed_pf
      and str(meta.Report.get(s, "") or "").strip()]
order = list(np.random.default_rng(1).permutation(ok))
lines = pd.DataFrame({"study_id": order, "findings": "pf_oa", "kind": "new", "block": "pf_reread", "expected_pos": a[order].to_numpy(),
                      "index": [int(studies.at[s, "index"]) for s in order]})
lines["session"] = FIRST + lines.index % CHATS
lines["pos"] = lines.groupby("session").cumcount()
assert lines.groupby("session").size().max() <= 20
new_key = pd.concat([key, lines[key.columns]]).sort_values(["session", "pos"])
cells = new_key.assign(f=new_key.findings.str.split(",")).explode("f")
assert not cells[cells.kind == "new"].duplicated(["study_id", "f"]).any(), "a (study, finding) listed twice"
kf = D / "_key/p3.csv"
kf.chmod(0o644)
new_key.to_csv(kf, index=False)
kf.chmod(0o444)
for s, g in lines.groupby("session"):
    f = D / f"_batches/p3_s{s}.txt"
    assert not f.exists(), f
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
for i, sid in zip(lines["index"], lines.study_id):
    rp = D / f"_reports/{i:04d}.txt"
    if not rp.exists():
        rp.write_text(meta.Report[sid].strip() + "\n")
print(f"pool {len(pool)}, usable {len(ok)}; {len(lines)} lines, sessions {FIRST}..{FIRST + CHATS - 1}: "
      + " ".join(f"s{s}:{k}" for s, k in lines.groupby('session').size().items()))
