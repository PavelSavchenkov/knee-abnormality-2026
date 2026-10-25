"""Re-score every new-label arm since 10-20 under prompts/validation.md "Yardstick per class" (USER 10-21 03:2x: "go through all older
experiments with new labels ... rescore them against new validation scheme and update learnings"). STEP 1 only: the arm vs its old-label
counterpart (same model / recipe / seed / ids, old labels), held-out soft AUC vs the class yardstick, per fold + pooled, paired bootstrap SE.
READ-ONLY decides; DENSE printed as the error catcher. Yardsticks (snapshot v3f): Contusion / Fracture = Opus (opus_raw), PF = 0.5 Opus + 0.5
text where read (Experts' model_read.yardstick); Effusion = EF3 (data/ieff_codex_vlm_opus.csv); Baker's = Codex (data/tgt_bak_codex.csv).
Uses Experts' scripts/model_read.py functions (identical numbers). Output rescore_all.txt + rescore_all.csv. No gold. OAI: no.
  PYTHONPATH=. conda run -n rsna python work/claude_labels_oct20/rescore_all.py"""
import sys

import numpy as np
import pandas as pd

from common.folds import get_folds
from common.paths import REPO

EXP = REPO / "experiments/v3/oct16_experts"
sys.path[:0] = [str(EXP / "scripts"), str(EXP / "bone")]
from model_read import diff, run_out, yardstick  # noqa: E402
from image_agreement import has  # noqa: E402
from lib import GOLD  # noqa: E402

D = REPO / "work/claude_labels_oct20"
SNAP, B = "v3f", 300
CF, CFP = ("Contusion", "Fracture"), ("Contusion", "Fracture", "PF OA")
IMG = {"Contusion": "img_Contusion", "Fracture": "img_Fracture", "PF OA": "img_PF OA", "Effusion": "img_Effusion"}
MAIN, B3, BL, BN, BFS, OA = "refs/main/sub_b0_s1", "b_v3_s1", "b3l_ctl", "b3n_ctl", "bfs_s1", "refs/v2cor_s1"
# (family, arm, ctrl, columns, arm output map, note)
ARMS = [
    ("b_v3 replace", "b3c_rep", B3, CF, None, "v3c (231 C pos) replace w1"),
    ("b_v3 replace", "b3c_rep_s2", B3, CF, None, "v3c seed 2 (ctrl seed 1)"),
    ("b_v3 replace", "b3d_rep", B3, CF, None, "v3d (298) replace w1"),
    ("b_v3 replace", "b3d_rep_s2", B3, CF, None, "v3d seed 2 (ctrl seed 1)"),
    ("b_v3 replace", "b3e_rep", B3, CF, None, "v3e (484) replace w1"),
    ("b_v3 dose", "b3m_gw8", B3, CF, None, "v3b group-preserving x8"),
    ("b_v3 dose", "b3c_gw8", B3, CF, None, "v3c group-preserving x8"),
    ("b_v3 dose", "b_p2part_w8", B3, CF, None, "v1 p2 x8 (target-base bug: v1prob vs tgt_v3)"),
    ("b_v3 sampling", "b3e_pp", B3, CF, None, "v3e pools pos x3 / neg x1.5"),
    ("b_v3 blend", "p2bl", B3, CF, None, "tgt_v3_p2blend (Opus / old blend)"),
    ("b_v3 SIF", "b_sif", B3, CF, None, "SIF report rule (no Opus)"),
    ("b_v3 SIF", "b_sif_s2", B3, CF, None, "SIF seed 2 (ctrl seed 1)"),
    ("b_lr3 replace", "b3l_rep", BL, CF, None, "v3c replace w1"),
    ("b_lr3 SIF", "b3l_sif", BL, CF, None, "SIF"),
    ("b_lr3 SIF", "b3l_sif_s2", BL, CF, None, "SIF seed 2 (ctrl seed 1)"),
    ("b_nf replace", "b3n_rep", BN, CF, None, "v3c replace w1"),
    ("b_nf SIF", "b3n_sif", BN, CF, None, "SIF"),
    ("bfs replace", "bfs_rep", BFS, CF, None, "v3c replace w1"),
    ("image head", "p2b", B3, CF, IMG, "sparse img head, v1 p2 (main heads tgt_v3)"),
    ("image head", "p2b_dv3", B3, CF, IMG, "dense img head (Opus + tgt_v3 fill)"),
    ("image head", "p2b_dpl", B3, CF, IMG, "img head + pseudo labels"),
    ("image head", "p2b_fs", B3, CF, IMG, "img head, FS series"),
    ("image head", "p2b_plonly", B3, CF, IMG, "img head, PL only"),
    ("image head", "p2b_shuf", B3, CF, IMG, "img head, SHUFFLED labels (null)"),
    ("image head ft", "p2b_ft1e-4", "p2b_ftc1e-4", CF, IMG, "fine-tune lr 1e-4 vs old-label ft"),
    ("image head ft", "p2b_ft3e-5", "p2b_ftc3e-5", CF, IMG, "fine-tune lr 3e-5 vs old-label ft"),
    ("image head w4", "w4_ctl", B3, CF, IMG, "dense dv3 img head x1"),
    ("image head w4", "w4_opw8", "w4_ctl", CF, IMG, "Opus cells x8 vs w4_ctl"),
    ("image head w4", "w4_opw16", "w4_ctl", CF, IMG, "Opus cells x16 vs w4_ctl"),
    ("image head w4", "w4_pn", "w4_ctl", CF, IMG, "pos x8 / neg x2 vs w4_ctl"),
    ("image head w4", "w4s_bal3", "w4_ctl", CF, IMG, "sparse + 3 read / batch vs w4_ctl"),
    ("image head w4", "w5_dcal", "w4_ctl", CF, IMG, "calibrated-text fill vs w4_ctl"),
    ("image head w4", "w5v2_bal3", "w4_ctl", CF, IMG, "v2 sparse + 3 read / batch vs w4_ctl"),
    ("twin (same ids)", "eqb_new", "eqb_old", CF, None, "b_v3 twin v1 p2 ids"),
    ("twin (same ids)", "eqb_v2", "eqb_ov2", CF, None, "b_v3 twin v2 ids"),
    ("twin (same ids)", "eqb_v3", "eqb_ov3", CF, None, "b_v3 twin v3b ids"),
    ("twin (same ids)", "eqbf_v3", "eqbf_ov3", CF, None, "bfs twin v3b"),
    ("twin (same ids)", "eqbl_v3", "eqbl_ov3", CF, None, "b_lr3 twin v3b"),
    ("twin (same ids)", "eqbn_v3", "eqbn_ov3", CF, None, "b_nf twin v3b"),
    ("twin (same ids)", "eqm_new", "eqm_old", CFP, None, "main twin v1 p2"),
    ("twin (same ids)", "eqm_v2", "eqm_ov2", CFP, None, "main twin v2"),
    ("main targets", "m_p2part_w8", MAIN, CFP, None, "v1 p2 x8 in main targets"),
    ("main targets", "m_p2part_w8fc", MAIN, CFP, None, "x8 F / C only"),
    ("main targets", "m_p2part_w8v2", MAIN, CFP, None, "v2 x8"),
    ("main targets", "m_p2part_pn", MAIN, CFP, None, "pos x8 / neg x2 sampling"),
    ("main targets", "p2cal", MAIN, CFP, None, "calibrated p2 target"),
    ("main targets", "m_cal_c", MAIN, CFP, None, "calibrated target c"),
    ("main targets", "m_sif", MAIN, CFP, None, "SIF calibrated"),
    ("main targets", "m_sif_s2", MAIN, CFP, None, "SIF seed 2 (ctrl seed 1)"),
    ("main targets", "m_sifonly", MAIN, CFP, None, "SIF only"),
    ("main ext heads", "p2img_w1", MAIN, CFP, IMG, "ext img heads w 1"),
    ("main ext heads", "p2img_w03", MAIN, CFP, IMG, "ext img heads w 0.3"),
    ("main ext heads", "p2m_ft1e-4", MAIN, CFP, IMG, "ext heads ft 1e-4"),
    ("main ext heads", "p2m_ft3e-5", MAIN, CFP, IMG, "ext heads ft 3e-5"),
    ("OA PF", "p2oa", OA, ("PF OA",), None, "OA expert p2 PF"),
    ("OA PF", "p2oa_w01", OA, ("PF OA",), None, "OA expert p2 PF w 0.1"),
    ("OA PF", "p2oa_pl", OA, ("PF OA",), None, "OA expert PF + PL"),
    ("OA PF", "p2oa_plonly", OA, ("PF OA",), None, "OA expert PL only"),
    ("OA PF", "p2oa_blend", OA, ("PF OA",), None, "OA expert PF blend (v2)"),
    ("OA PF", "p3e_pf", OA, ("PF OA",), None, "OA expert PF blend (v3e)"),
    ("Effusion main", "ef1", MAIN, ("Effusion",), None, "2-reader Codex + Opus target"),
    ("Baker's main", "bk1", MAIN, ("Baker's",), None, "Codex Baker's target"),
    ("Baker's main", "bk2", MAIN, ("Baker's",), None, "Codex / text blend target"),
]


def other_yard(c):
    f = {"Effusion": EXP / "data/ieff_codex_vlm_opus.csv", "Baker's": EXP / "data/tgt_bak_codex.csv"}[c]
    ro = pd.read_csv(f).set_index("StudyInstanceUID")[c].dropna()
    return None, ro


fo = get_folds().iloc[:, 0]
rng, rows = np.random.default_rng(0), []
for fam, arm, ctrl, cols, amap, note in ARMS:
    folds = [F for F in range(5) if has(arm, F) and has(ctrl, F)]
    if not folds:
        rows.append({"family": fam, "arm": arm, "ctrl": ctrl, "note": note, "column": "-", "status": "no shared folds"})
        continue
    for c in cols:
        dense, ro = yardstick(c, SNAP) if c in CFP else other_yard(c)
        out = {"family": fam, "arm": arm, "ctrl": ctrl, "note": note, "column": c, "folds": ",".join(map(str, folds))}
        try:
            for vname, y in (("RO", ro), ("DENSE", dense)):
                if y is None:
                    continue
                A, C = {}, {}
                for F in folds:
                    ids = y.index[y.index.isin(fo.index[fo == F]) & ~y.index.isin(GOLD.index)]
                    xa = run_out(arm, (amap or {}).get(c, c), "oof", F).reindex(ids)
                    xc = run_out(ctrl, c, "oof", F).reindex(ids)
                    ok = xa.notna() & xc.notna()
                    A[F], C[F] = xa[ok], xc[ok]
                per = {F: diff(y.loc[A[F].index], A[F], C[F], B, rng) for F in folds}
                pa, pc = pd.concat(A.values()), pd.concat(C.values())
                d, se, _, n, npos = diff(y.loc[pa.index], pa, pc, B, rng)
                signs = {np.sign(v[0]) for v in per.values()}
                if vname == "RO":
                    out |= {"n / pos": f"{n} / {npos}", "per fold": " ".join(f"{v[0]:+.1f}" for v in per.values()), "RO d": round(d, 1), "RO SE": round(se, 1),
                            "consistent": "1 fold" if len(folds) == 1 else ("YES" if len(signs) == 1 and d * next(iter(signs)) > se else "no")}
                else:
                    out |= {"DENSE d": round(d, 1), "DENSE SE": round(se, 1)}
        except (KeyError, FileNotFoundError, AssertionError) as e:
            out["status"] = f"skip: {type(e).__name__} {str(e)[:60]}"
        rows.append(out)
t = pd.DataFrame(rows)
t.to_csv(D / "rescore_all.csv", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 46)
txt = (f"OAI: no. RESCORE (step 1, model vs old-label counterpart) on yardstick snapshot {SNAP}; READ-ONLY decides, DENSE = error catcher; "
       f"x1e-3; boot {B}\n" + t.to_string(index=False))
(D / "rescore_all.txt").write_text(txt + "\n")
print(txt)
