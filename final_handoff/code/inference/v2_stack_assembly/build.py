"""oct20_stack_all (Experts V3 10-19; USER GO via Master: the merged fold package for a 10-20 slot, "and include our inference optimizations";
OAI: no): CPU build step = the declared assembly re-fit that per-column additivity does not cover (A5):
  Q6 (MCL <- Contusion de-coupling, b 0.2) re-fitted label-free on the NEW Contusion (the 0.961 crop-fused Contusion + the cont_fs bfs splice,
  LB-won 57388555): c, c0 = polyfit(Contusion_new, MCL_pre-Q6) over OOF f0-2 non-gold, exactly Q21's own fitting (oct18_probe_mcl_stack/build.py).
Identity checks first (checklist F6): the pre-Q6 Q21 MCL chain reproduces Q21's stack.json (b3g align) and, with the OLD Contusion context,
Q21's dec.json constants; the cont_fs constants reproduce the probe's member.json (mu_m / sd_m / mu_d / sd_d) and its expected gold.
Control re-fitted the same way = the OLD context (must give Q21's constants). Writes refit.json (DEC for the kernel: MCL re-fitted + LM Q22
unchanged) and expected_gold.csv (MCL after the re-fitted Q6 and Contusion with cont_fs, on gold-58). Info: MCL readers (OOF f0-2) under
the old vs new constants.
conda run -n rsna python submissions/oct20_stack_all/build.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
REPO = HERE.parents[1]
EXP = REPO / "experiments/v3/oct16_experts"
STACK, CONT, Q22 = SUB / "oct18_probe_mcl_stack", SUB / "oct17_probe_cont_fs", SUB / "oct19_probe_acl_b3g_q8"
sys.path[:0] = [str(REPO), str(EXP / "scripts")]
from common.labels import gold_labels, status_table  # noqa: E402

lgt = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
W2, B = 0.25, 0.2


def main_oof(f, set_):                              # = Q21's build
    d = pd.read_parquet(EXP / f"runs/sd1b/fold{f}/{set_}.parquet")
    d = d[d.epoch == d.epoch.max()] if "epoch" in d else d
    d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d else d
    return lgt(d["MCL"].astype(float)).groupby(level=0).mean()


def nf_oof(f, set_):                                # = Q21's build
    d = pd.read_parquet(EXP / f"runs/bsd_nf/fold{f}/preds.parquet")
    d = d[(d.epoch == 14) & (d["set"] == set_)]
    return lgt(d["MCL"].astype(float)).groupby(level=0).mean()


def run_preds(run, kind, col, epoch=None):
    out = []
    for f in range(5):
        d = pd.read_parquet(EXP / "runs" / run / f"fold{f}" / f"{kind}.parquet")
        d = d[d.epoch == (epoch if epoch is not None else d.epoch.max())] if "epoch" in d else d
        d = d.set_index("StudyInstanceUID") if "StudyInstanceUID" in d else d
        out.append(lgt(d[col].astype(float)))
    return out


def mcl_reads(o, ids):
    st = status_table("v1")["mcl"].reindex(ids)
    auc = lambda p, n: roc_auc_score(np.r_[np.ones(p.sum()), np.zeros(n.sum())], np.r_[o[ids][p], o[ids][n]]) * 1e3  # noqa: E731
    return {"text": auc(st.eq("meets"), st.isin(["denied", "structure_normal"])), "thr": auc(st.eq("meets"), st.eq("below")),
            "host-like": auc(st.isin(["meets", "unsettled"]), st.isin(["denied", "structure_normal", "below"]))}


def main():
    G = gold_labels()
    pk = pd.read_parquet(EXP / "refs/pkg961/oof_f5_win.parquet").xs("f5_win", level=0)
    m = pd.read_parquet(EXP / "refs/pkg961/members.parquet")
    lg = lambda s: s if s.min() < 0 else pd.Series(lgt(s.to_numpy(float)), index=s.index)  # noqa: E731
    r50 = {s: lg(m[(m.set == s) & (m.member == "r50")].groupby(["fold", "StudyInstanceUID"])["MCL"].mean()) for s in ("oof", "gold")}
    sd2_oof = pd.concat([(main_oof(f, "oof") + nf_oof(f, "oof").reindex(main_oof(f, "oof").index)
                          + r50["oof"].xs(f, level=0).reindex(main_oof(f, "oof").index)) / 3 for f in (0, 1, 2)]).dropna()
    sd2_gold = sum((main_oof(f, "gold") + nf_oof(f, "gold").reindex(main_oof(f, "gold").index)
                    + r50["gold"].xs(f, level=0).reindex(main_oof(f, "gold").index)) / 3 for f in range(5)) / 5
    zm = json.load(open(SUB / "oct17_probe_mcl_zoom/dataset/member.json"))["mcl_member"]
    ref = json.load(open(SUB / "oct18_probe_mcl_sd2/dataset/member_sd.json"))["mcl_member_refit"]
    w5, w5g = pd.concat(run_preds("w5b_mcl", "oof", "MCL", 19)[:3]), sum(run_preds("w5b_mcl", "gold", "MCL", 19)) / 5
    bo, bg = pd.concat(run_preds("b3g", "oof", "MCL")), sum(run_preds("b3g", "gold", "MCL")) / 5
    ids = sd2_oof.index.intersection(pk.index).intersection(w5.index)
    ids = ids[~ids.isin(G.index)]
    mu_d, sd_d = float(sd2_oof.loc[ids].mean()), float(sd2_oof.loc[ids].std())
    assert abs(mu_d - ref["mu_d"]) < 1e-9 and abs(sd_d - ref["sd_d"]) < 1e-9, (mu_d, sd_d, ref)
    zw = lambda x: (x - zm["mu_m"]) / zm["sd_m"] * sd_d + mu_d  # noqa: E731
    m1o, m1g = 0.75 * sd2_oof.loc[ids] + 0.25 * zw(w5.loc[ids]), 0.75 * sd2_gold + 0.25 * zw(w5g.reindex(sd2_gold.index))
    a2 = json.load(open(STACK / "stack.json"))
    chk = {"mu_m": float(bo.loc[ids].mean()), "sd_m": float(bo.loc[ids].std()), "mu_d": float(m1o.mean()), "sd_d": float(m1o.std())}
    d_a2 = max(abs(chk[k] - a2[k]) for k in chk)
    assert d_a2 < 1e-9 and a2["n_align"] == len(ids), (d_a2, chk, a2)                     # F6: Q21's b3g align reproduced
    z2 = lambda x: (x - a2["mu_m"]) / a2["sd_m"] * a2["sd_d"] + a2["mu_d"]  # noqa: E731
    m2o, m2g = (1 - W2) * m1o + W2 * z2(bo.loc[ids]), (1 - W2) * m1g + W2 * z2(bg.reindex(m1g.index))
    gi = m2g.index
    # OLD context (the control, re-fitted the same way) must reproduce Q21's dec.json
    ctx_old = pk.loc[ids, "Contusion"]
    gctx_old = lgt(pd.read_csv(SUB / "oct13_f5_win/expected_gold_f5_win.csv", index_col=0)["Contusion"])
    c_o, c0_o = np.polyfit(ctx_old, m2o, 1)
    q6 = json.load(open(STACK / "dec.json"))["MCL"]
    d_q6 = max(abs(c_o - q6["c"]), abs(c0_o - q6["c0"]))
    assert d_q6 < 1e-9 and q6["b"] == B, (c_o, c0_o, q6)                                      # F6: Q21's Q6 reproduced
    e21 = pd.read_csv(STACK / "expected_gold_mcl.csv", index_col=0)["logit"]
    d_e21 = float((m2g - B * (c0_o + c_o * gctx_old.reindex(gi)) - e21.reindex(gi)).abs().max())
    assert d_e21 < 1e-6, d_e21                                                                  # F6: Q21's expected gold reproduced
    # NEW context: the shipped crop-fused Contusion + the cont_fs bfs splice (the kernel's Contusion), its constants reproduced
    cf = json.load(open(CONT / "dataset/member.json"))["cont_fs"]
    bfs = pd.concat(run_preds("bfs", "oof", "Contusion", 19))
    bfs = bfs.groupby(level=0).mean()
    ic = pk.index.intersection(bfs.index)
    ic = ic[~ic.isin(G.index)]
    chk_c = {"mu_m": float(bfs.loc[ic].mean()), "sd_m": float(bfs.loc[ic].std()), "mu_d": float(pk.loc[ic, "Contusion"].mean()),
             "sd_d": float(pk.loc[ic, "Contusion"].std())}
    d_cf = max(abs(chk_c[k] - cf[k]) for k in chk_c)
    assert d_cf < 1e-6 and cf["n_align"] == len(ic), (d_cf, chk_c, cf)                         # F6: the cont_fs constants reproduced
    zb = (bfs.reindex(ids) - cf["mu_m"]) / cf["sd_m"] * cf["sd_d"] + cf["mu_d"]
    ctx_new = ((1 - cf["w"]) * ctx_old + cf["w"] * zb).fillna(ctx_old)
    gctx_new = pd.read_csv(CONT / "expected_gold_cont.csv", index_col=0)["logit"].reindex(gi)
    c_n, c0_n = np.polyfit(ctx_new, m2o, 1)
    dec = {"MCL": {"context": "Contusion", "b": B, "c0": float(c0_n), "c": float(c_n)}}
    q22 = json.load(open(Q22 / "dec.json"))
    assert set(q22) == {"Lateral Meniscus"} and q22["Lateral Meniscus"]["context"] == "ACL", q22
    dec |= q22                                                                                 # Q22's c0 / c stay as fitted on the Q16 ACL
    json.dump({"dec": dec, "q6_refit": {"control_old_context": {"c0": float(c0_o), "c": float(c_o)}, "new_context": dec["MCL"],
                                        "n": len(ids), "folds": [0, 1, 2], "method": "label-free polyfit(Contusion, MCL pre-Q6), Q21's own fit"}},
              open(HERE / "refit.json", "w"), indent=1)
    mo_new, mo_old = m2o - B * (c0_n + c_n * ctx_new), m2o - B * (c0_o + c_o * ctx_old)
    mg_new = m2g - B * (c0_n + c_n * gctx_new)
    ex = pd.DataFrame({"MCL_logit": mg_new, "Contusion_logit": gctx_new})
    ex.index.name = "StudyInstanceUID"
    ex.to_csv(HERE / "expected_gold.csv")
    y = G.loc[gi, "MCL"]
    r_o, r_n = mcl_reads(mo_old, ids), mcl_reads(mo_new, ids)
    print(f"OAI: no. F6 identities: SD-2 refit {abs(mu_d - ref['mu_d']):.1e}; b3g align {d_a2:.1e}; Q21 Q6 constants {d_q6:.1e}; Q21 expected gold "
          f"{d_e21:.1e}; cont_fs constants {d_cf:.1e} (n {len(ic)})")
    print(f"Q6 re-fit (label-free, OOF f0-2, n {len(ids)}): control (old Contusion) c0 {c0_o:.4f} c {c_o:.4f} = Q21's; NEW Contusion (+ cont_fs) "
          f"c0 {c0_n:.4f} c {c_n:.4f}")
    print("INFO MCL readers (OOF f0-2, x1e-3): new - old constants: " + ", ".join(f"{k} {r_n[k] - r_o[k]:+.2f}" for k in r_n)
          + f"; gold-58 MCL AUC old {roc_auc_score(y, m2g - B * (c0_o + c_o * gctx_old.reindex(gi))):.4f} -> new {roc_auc_score(y, mg_new):.4f}")
    print(f"refit.json DEC: {dec}")


if __name__ == "__main__":
    main()
