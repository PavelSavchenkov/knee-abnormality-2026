"""Split session 7's remaining fracture-vague list across the finished chats (USER 10-20 16:2x: "you can even split among all other sessions
finished with their own data"). Remaining = p2_s7_rest.txt minus what p2_s7/ already holds, in list order (random, so every chunk is a random
sample). Chat 7 keeps the head (it is mid-study there); helpers get the next contiguous chunks, in the order given. Each helper writes to its
own folder p2_s7_h<N>/ (the p2_s*/ glob of build_tgt_p2.py picks it up; it asserts one read per study, so the chunks must not overlap). OAI: no.
  python work/claude_labels_oct20/build_phase2_s7split.py --helpers 1 2 4 6 5 [--head-extra 3]"""
import argparse
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--helpers", type=int, nargs="+", required=True)
ap.add_argument("--head-extra", type=int, default=3, help="extra head entries for chat 7 (it keeps saving while the message arrives)")
a = ap.parse_args()
rest = [int(i) for i in (D / "_batches/p2_s7_rest.txt").read_text().split()]
done = {json.load(open(p))["index"] for p in (D / "p2_s7").glob("*.json")}
left = [i for i in rest if i not in done]
parts = np.array_split(np.array(left[a.head_extra:]), len(a.helpers) + 1)
out = {"p2_s7_part.txt": left[:a.head_extra] + parts[0].tolist()} | {f"p2_s7_h{h}.txt": p.tolist() for h, p in zip(a.helpers, parts[1:])}
assert sum(len(v) for v in out.values()) == len(left) and len({i for v in out.values() for i in v}) == len(left)
for name, v in out.items():
    f = D / "_batches" / name
    assert not f.exists(), f
    f.write_text("\n".join(map(str, v)) + "\n")
    f.chmod(0o444)
    print(f"{name}: {len(v)} (first {v[0]}, last {v[-1]})")
print(f"chat 7 saved {len(done)}; remaining {len(left)}")
