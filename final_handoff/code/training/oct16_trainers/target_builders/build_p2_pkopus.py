"""Package + Opus image-head targets (Experts V3 10-20 13:3x; Master's CPU evidence work/claude_labels_oct20/eval_pkg_opus_blends.txt: package + w
Opus improves PF at every weight (best PF label form yet) and a single model's Fracture at w 0.1-0.25; OAI: no; CPU). For the labelled studies:
 c = z(package OOF logit) + w x z(Opus logit) (z over the labelled set), mapped back onto the package OOF logit scale (mean + sd x c / sd(c)),
 target = sigmoid(that) - a soft image-head target in [0, 1].
Package OOF = pkg961 oof_f5_win (folds 0-2) + data/clean_f34/<col>.csv (folds 3-4, the C213 clean rebuild). CAVEAT: the package OOF of
fold-1 training studies comes from fold models that trained on fold 1's labels (the standard PL leak path).
 data/tgt_p2_pf_pl2.csv    img_pf_patella = PF with w 1.0 (W3-2's new image-head target) + pl_pf_patella = the shipped PF OOF prob (unchanged PL)
 data/tgt_p2_bone_pko.csv  Fracture = Fracture with w 0.25 (bone img_frac; Contusion NaN = its image head unsupervised)
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_p2_pkopus.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP.parents[2]), str(EXP / "scripts")]
from lib import GOLD  # noqa: E402

lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731


def pkg_oof(c):
    pk = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)[c]
    cl = pd.read_csv(EXP / f"data/clean_f34/{c.replace(' ', '_')}.csv").set_index("StudyInstanceUID").logit
    assert not pk.index.intersection(cl.index).size
    return pd.concat([pk, cl])


def blend(c, w, opus):
    o = opus.dropna()
    k = pkg_oof(c).reindex(o.index)
    assert k.notna().all(), f"{c}: package OOF missing on {int(k.isna().sum())} labelled studies"
    zk, zo = (k - k.mean()) / k.std(), (lgt(o) - lgt(o).mean()) / lgt(o).std()
    cc = zk + w * zo
    t = 1 / (1 + np.exp(-(k.mean() + k.std() * cc / cc.std())))
    return t, float(np.corrcoef(t, o)[0, 1]), float(np.corrcoef(lgt(t), k)[0, 1])


def main():
    op = pd.read_csv(EXP / "data/ext_p2_main.csv").set_index("StudyInstanceUID")
    pf, r_pf, r_pfk = blend("PF OA", 1.0, op["PF OA"])
    pl = pd.read_csv(EXP / "data/tgt_p2_pf_pl.csv").set_index("StudyInstanceUID").pl_pf_patella
    out = pd.concat([pf.rename("img_pf_patella"), pl], axis=1).rename_axis("StudyInstanceUID")
    assert not out.index.isin(GOLD.index).any()
    out.to_csv(EXP / "data/tgt_p2_pf_pl2.csv")
    fr, r_fr, r_frk = blend("Fracture", 0.25, op["Fracture"])
    fb = pd.DataFrame({"Fracture": fr, "Contusion": np.nan}).rename_axis("StudyInstanceUID")
    assert not fb.index.isin(GOLD.index).any()
    fb.to_csv(EXP / "data/tgt_p2_bone_pko.csv")
    print(f"tgt_p2_pf_pl2: PF image target (package + 1.0 Opus) on {len(pf)} labelled, mean {pf.mean():.3f}, >= .5 {int((pf >= .5).sum())}, corr with Opus "
          f"{r_pf:.2f} / with the package logit {r_pfk:.2f}; PL head unchanged ({int(pl.notna().sum())} studies)")
    print(f"tgt_p2_bone_pko: Fracture image target (package + 0.25 Opus) on {len(fr)} labelled, mean {fr.mean():.3f}, >= .5 {int((fr >= .5).sum())}, corr with "
          f"Opus {r_fr:.2f} / with the package logit {r_frk:.2f}")


if __name__ == "__main__":
    main()
