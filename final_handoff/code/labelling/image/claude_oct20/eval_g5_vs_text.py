"""g5 Opus reads vs the report text v1 on gold-58, head to head: pairs (gold positive, gold negative) that text v1 misorders (or ties) and
Opus orders right, and the reverse. Output appended to eval_g5.txt (work/ only, never memory or labellers). OAI: no."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_g5 import FINDINGS, KEY_OF, D, gold, me, text  # noqa: E402

out = []
for f in [c for c in FINDINGS if me[c].notna().any()]:
    s, t = me[f].reindex(gold.index), text[KEY_OF[f]].reindex(gold.index).fillna(0.15)
    m = s.notna() & gold[f].notna()
    y, s, t = gold[f][m], s[m], t[m]
    pairs = [(p, q) for p in y.index[y == 1] for q in y.index[y == 0]]
    opus_only = sum(s[p] > s[q] and not t[p] > t[q] for p, q in pairs)
    text_only = sum(t[p] > t[q] and not s[p] > s[q] for p, q in pairs)
    flips = int(((s >= .5) != (t >= .5)).sum())
    agree = np.corrcoef(s, t)[0, 1]
    out.append(f"{f:16s} pairs {len(pairs)} | Opus right where text not {opus_only} | text right where Opus not {text_only} | "
               f"side-of-.5 differences {flips} / {int(m.sum())} | corr {agree:.2f}")
(D / "eval_g5.txt").open("a").write("\nOpus vs text v1, head to head:\n" + "\n".join(out) + "\n")
print("\n".join(out))
