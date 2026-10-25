"""DENSE versions of the queued sparse image / aux targets (USER 10-20: "wtf, you absolutely should not sample data like that. either don't
sample studies with zero gradient from them, or use old labels"; OAI: no; CPU). Every non-gold study gets a target: the frozen Opus-p2 value
where read (559), the recipe's OWN old label elsewhere (each recipe's labels.make_targets with its reference config):
 data/ext_p2_main_dense.csv      Fracture / Contusion / PF OA (mainp ext heads; S1 p2m_ft): ext_p2_main_r where read, main text v1prob elsewhere
 data/tgt_p2_pf_blend_dense.csv  img_pf_patella (oa aux; p2oa_blend): tgt_p2_pf_blend where read, the OA recipe's PF OA target elsewhere
 data/tgt_p2_pf_pl2_dense.csv    img_pf_patella dense as above (Opus where read = tgt_p2_pf_pl2's own column) + pl_pf_patella unchanged (p2oa_pl)
 data/tgt_p2_bone_v3o_dense.csv  Fracture / Contusion (bone image heads; p2b_v3o): 0.5 tgt_v3 + 0.5 Opus where read (tgt_p2_bone_v3o), tgt_v3
                                 elsewhere (= tgt_p2_bone_dv3's fill)
conda run -n rsna python experiments/v3/oct16_experts/scripts/build_dense_fill.py"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[2]
sys.path[:0] = [str(REPO)]
from common.folds import get_folds  # noqa: E402
from common.labels import FINDINGS, gold_labels  # noqa: E402


def recipe_targets(sub, ref, studies):
    """the recipe's own targets (n, 12 FINDINGS) from <sub>/labels.make_targets with <sub>'s trainer DEFAULTS | the reference config."""
    sys.path.insert(0, str(EXP / sub))
    sys.modules.pop("labels", None)                   # each trainer imports its OWN labels.py by that name
    lab = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(f"labels_{sub}", EXP / sub / "labels.py"))
    spec.loader.exec_module(lab)
    tr = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location(f"train_{sub}", EXP / sub / ("train.py" if sub == "mainp" else "oa_train.py")))
    spec.loader.exec_module(tr)
    cfg = tr.DEFAULTS | json.load(open(EXP / ref))
    y = lab.make_targets(studies, cfg)[0]
    assert y.shape == (len(studies), len(FINDINGS)), y.shape
    return pd.DataFrame(y, index=studies, columns=FINDINGS)


def dense(sparse, fill):
    out = fill.astype("float64").rename_axis("StudyInstanceUID")
    assert sparse.index.isin(out.index).all()
    out.loc[sparse.index] = sparse.to_numpy()
    assert out.notna().all().all()
    return out


def main():
    gi = set(gold_labels().index)
    studies = [s for s in get_folds().index if s not in gi]
    d = lambda f: pd.read_csv(EXP / "data" / f).set_index("StudyInstanceUID")  # noqa: E731
    ym = recipe_targets("mainp", "refs/m_prob05/config.json", studies)
    yo = recipe_targets("oa", "refs/exp_oa_v2cor/config.json", studies)
    cols = ["Fracture", "Contusion", "PF OA"]
    dense(d("ext_p2_main_r.csv")[cols], ym[cols]).to_csv(EXP / "data/ext_p2_main_dense.csv")
    pf_fill = yo[["PF OA"]].rename(columns={"PF OA": "img_pf_patella"})
    dense(d("tgt_p2_pf_blend.csv")[["img_pf_patella"]], pf_fill).to_csv(EXP / "data/tgt_p2_pf_blend_dense.csv")
    pl2 = d("tgt_p2_pf_pl2.csv")
    out = dense(pl2[["img_pf_patella"]].dropna(), pf_fill).join(pl2[["pl_pf_patella"]], how="left")
    assert out.notna().all().all(), "pl2 lacks PL on some studies"
    out.to_csv(EXP / "data/tgt_p2_pf_pl2_dense.csv")
    dv3 = d("tgt_p2_bone_dv3.csv")
    sp, op = d("tgt_p2_bone_v3o.csv"), d("tgt_p2_bone.csv")
    assert (sp.index == op.index).all()
    dense(sp[["Fracture", "Contusion"]], dv3[["Fracture", "Contusion"]].loc[dv3.index]).to_csv(EXP / "data/tgt_p2_bone_v3o_dense.csv")
    print(f"OAI: no. dense fills over {len(studies)} non-gold studies (Opus-read 559 kept as built, the recipe's own old label elsewhere):")
    for f in ("ext_p2_main_dense", "tgt_p2_pf_blend_dense", "tgt_p2_pf_pl2_dense", "tgt_p2_bone_v3o_dense"):
        t = d(f + ".csv")
        print(f"  {f}: {t.shape}, NaN {int(t.isna().sum().sum())}, mean {t.mean().round(3).to_dict()}, >= .5 {(t >= .5).sum().to_dict()}")


if __name__ == "__main__":
    main()
