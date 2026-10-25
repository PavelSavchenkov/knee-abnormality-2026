"""Baker's TWO-READER target (Master 10-21 03:2x ask 2; USER "are you doing experiments with all classes we have new image-based labels
for? you should"; OAI: no; no gold): Codex Baker's read (data/image_labels/codex_bakers.json, 516 non-gold studies) everywhere it read; on
the studies that ALSO have an Opus Baker's second read (opus_bakers_p4.json, 160) the sigmoid of the mean of logit(Codex) and logit(Opus)
z-aligned to the Codex logit scale over those studies (label-free; the tgt_eff_2r recipe of work/image_targets_2026-10-19/build.py).
NaN elsewhere = the recipe's own target (run with tgt_partial=Baker's, as bk1). Writes data/tgt_bak_2r.csv (read-only).
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_bak_2r.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.labels import gold_labels
from common.paths import REPO

OUT = REPO / "experiments/v3/oct16_experts/data/tgt_bak_2r.csv"
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))  # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731


def reads(name):
    return pd.Series({s: r["findings"]["bakers_cyst"]["probability"] for s, r in json.load(open(REPO / f"data/image_labels/{name}.json")).items()},
                     dtype=float).dropna()


def main():
    cx, op = reads("codex_bakers"), reads("opus_bakers_p4")
    gold = set(gold_labels().index)
    assert not (set(cx.index) | set(op.index)) & gold, "gold study in a read set"
    both = op.index.intersection(cx.index)
    assert len(both) == len(op), f"{len(op) - len(both)} Opus Baker's reads without a Codex read"
    lc, lo = lg(cx[both]), lg(op[both])
    lo_al = (lo - lo.mean()) / lo.std() * lc.std() + lc.mean()
    t = cx.copy()
    t[both] = sg((lc + lo_al) / 2)
    assert t.index.is_unique and t.between(0, 1).all() and len(t) == len(cx)
    assert not OUT.exists(), f"{OUT} exists"
    t.rename("Baker's").rename_axis("StudyInstanceUID").to_csv(OUT)
    OUT.chmod(0o444)
    print(f"OAI: no. {OUT.name}: {len(t)} studies (Codex 516), two-reader on {len(both)}; Spearman Codex-Opus {lc.corr(lo, method='spearman'):.3f}; "
          f">= .5 Codex {int((cx[both] >= .5).sum())} / Opus {int((op >= .5).sum())} / two-reader {int((t[both] >= .5).sum())} of {len(both)}; "
          f"flips vs Codex {int(((t[both] >= .5) != (cx[both] >= .5)).sum())}; mean |change| {np.abs(t[both] - cx[both]).mean():.3f}")


if __name__ == "__main__":
    main()
