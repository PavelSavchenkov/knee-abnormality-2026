"""Queue lines for the PL v2 seeds (check_v2.py's exact line construction: plain all-data line + seed + the v2 keys). OAI: no.
Box lines (oct16 runner): needs=.../data/pl/READY_v2_<stage>.flag, gb = the recipe's plain all-data line (box queues; b3g 24 per Master).
Local lines (oct14 runner, no needs gate): insert after READY_v2_<stage>. Wave order = seed 201..207; r50 after every other local line (USER).
  conda run -n rsna python work/pl_alldata_2026-10-21/make_lines_v2.py > work/pl_alldata_2026-10-21/queue_lines_v2.txt"""
import re
import sys
from pathlib import Path

D = Path(__file__).parent
sys.path.insert(0, str(D))
from check_v2 import LINE16, SRC16, line14, v2_sets  # noqa: E402
from targets_v2 import E16, R  # noqa: E402

S1 = "b3c_rep,b3l_rep,b3n_rep,bfs_rep,z1_acl,w5b_mcl,oasep,oasep_nf,silsep,sxm,oa_v2cor,oa_v2cor_nf".split(",")
SEEDS = range(201, 208)
GB = {"b3g": 24}


def gb(r):
    if r in GB:
        return GB[r]
    for q in ("queue.txt", "queue_remote.txt", "queue_remote2.txt"):
        for ln in open(E16 / q):
            m = re.match(rf"#?\s*{re.escape(r)}_all_s1\d\d .* gb=(\d+)", ln)
            if m:
                return int(m[1])
    raise SystemExit(f"no gb for {r}")


def box(r, s):
    st = "s1" if r in S1 else "s2"
    plain = [x for x in LINE16.get(r, ["spe_ref=434"]) if not x.startswith("tgt_csv=")]      # silsep: its v4b tgt_csv is replaced
    return (f"{r}_pl2_s{s} {R[r][0][4:]} {SRC16.get(r, f'runs/{r}/fold0')} -1 {' '.join(plain)} seed={s} {' '.join(v2_sets(r, 'v1'))} "
            f"needs=experiments/v3/oct16_experts/data/pl/READY_v2_{st}.flag gb={gb(r)}")


def local(r, s):
    src, sets = line14(r)
    hit = [m[1] for ln in open(E16.parents[1] / "v2/oct14_all_data/queue.txt") if (m := re.match(rf"#?\s*A1\d\d_{re.escape(r)} .* gb=(\d+)", ln))]
    g = hit[0] if hit else {"basin": 13}.get(R[r][0][4:], 9)                 # the a14 lines' gb (basin 13, others 9)
    return f"PL{s}_{r} {R[r][0][4:]} {src} -1 {' '.join(sets)} seed={s} {' '.join(v2_sets(r, 'v1'))} gb={g}"


def main():
    o16 = [r for r in R if R[r][0].startswith("o16")]
    o14 = [r for r in R if R[r][0].startswith("o14") and r != "r50"]
    print("# PL v2 lines (teacher v1, USER 21:3x: every target of every recipe). OAI: no. Seeds 201-207 = waves.")
    print("# ===== BOXES (oct16 runner), above the low-value tail; s1 lines gated by READY_v2_s1.flag, s2 by READY_v2_s2.flag")
    for s in SEEDS:
        for r in [x for x in o16 if x in S1] + [x for x in o16 if x not in S1]:
            print(box(r, s))
    print("# ===== LOCAL (oct14 runner), above the low-value tail; insert the s1 block after READY_v2_s1, the s2 block after READY_v2_s2")
    print("# --- s1 (oa_v2cor, oa_v2cor_nf)")
    for s in SEEDS:
        for r in [x for x in o14 if x in S1]:
            print(local(r, s))
    print("# --- s2 (r50 LAST, after every other local line)")
    for s in SEEDS:
        for r in [x for x in o14 if x not in S1]:
            print(local(r, s))
    for s in SEEDS:
        print(local("r50", s))


if __name__ == "__main__":
    main()
