"""Phase 3 re-split (USER 10-20 18:2x: "easier and faster to create more chats -- each chat up to 20 studies ... limit existing chats and
give me the remaining prompts for sessions 9+"). Run only after chats 1-8 have stopped. Every unsaved line of p3_s1..s8 moves to new
sessions of at most MAX lines, in block order (contusion -> fracture -> pf_oa; best-first by P inside a block), dealt round-robin over
the sessions of each block so every chat's prefix is best-first. Old lists are cut to their saved lines, so a resumed old chat has
nothing left (no double reads). Updates _key/p3.csv (session / pos of the moved lines), writes _batches/p3_s<k>.txt for the new
sessions. No gold. OAI: no.
  python work/claude_labels_oct20/build_phase3_resplit.py [--max 20]"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=20)
a = ap.parse_args()
key = pd.read_csv(D / "_key/p3.csv")
pool = pd.read_csv(D / "positive_yield_pool.csv", index_col=0)
saved = {json.load(open(p))["index"] for p in D.glob("p3_s*/*.json")}
assert key["index"].is_unique
old = key.session.max()
rest = key[~key["index"].isin(saved)].copy()
BLOCKS = ("contusion", "fracture", "pf_oa")
rest["bp"] = [pool.at[s, f"p_{b}"] if s in pool.index else 0.0 for s, b in zip(rest.study_id, rest.block)]
out, nxt = [], old + 1
for b in BLOCKS:
    g = rest[rest.block == b].sort_values(["kind", "bp"], ascending=[True, False])   # 'new' before 'reread', best-first
    if g.empty:
        continue
    new_rows, rr = g[g.kind == "new"], g[g.kind == "reread"]
    n = math.ceil(len(g) / a.max)
    for c in range(n):
        part = new_rows.iloc[c::n].to_dict("records")
        for j, r in enumerate(rr.iloc[c::n].to_dict("records")):
            part.insert(min(len(part), 3 + 7 * j), r)
        for pos, r in enumerate(part):
            out.append(r | {"session": nxt + c, "pos": pos})
    nxt += n
moved = pd.DataFrame(out).drop(columns="bp")
assert set(moved["index"]) == set(rest["index"]) and moved.groupby("session").size().max() <= a.max + 1
key = pd.concat([key[key["index"].isin(saved)], moved])[key.columns].sort_values(["session", "pos"])
for s in range(1, old + 1):                                                   # old lists = their saved lines only
    f = D / f"_batches/p3_s{s}.txt"
    f.chmod(0o644)
    g = key[key.session == s]
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
kf = D / "_key/p3.csv"
kf.chmod(0o644)
key.to_csv(kf, index=False)
kf.chmod(0o444)
for s, g in moved.groupby("session"):
    f = D / f"_batches/p3_s{s}.txt"
    assert not f.exists(), f
    f.write_text("".join(f"{i} {x}\n" for i, x in zip(g["index"], g.findings)))
    f.chmod(0o444)
    print(f"p3_s{s}: {len(g)} lines, block {g.block.iloc[0]}{' + ' + g.block.iloc[-1] if g.block.iloc[-1] != g.block.iloc[0] else ''}, "
          f"expected positives {g.expected_pos.sum():.0f}")
print(f"saved {len(saved)}; moved {len(moved)} lines into sessions {old + 1}..{moved.session.max()}")
