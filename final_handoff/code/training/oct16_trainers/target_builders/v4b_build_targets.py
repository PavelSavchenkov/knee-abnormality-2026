"""Stage target files for this experiment (CPU; per fold K the same rules as oct14_text_labels: anything fitted comes from fold-K-free
files). Writes data/targets/{name}_f{0-4}.csv (StudyInstanceUID + 12 FINDINGS, NaN = keep the trainer's v1 target unless tgt_nan_w0).

  nan_sfc   Synovitis / Fracture / Contusion all NaN, other findings absent from the file -> with tgt_findings=Synovitis+Fracture+
            Contusion tgt_nan_w0=True these 3 columns get loss weight 0 (stage on the 9 generalist-owned columns only)
  v1x_<L>   v1 probability on explicit v1 cells (meets / denied / structure_normal), label set L elsewhere (non-explicit cells)
  xv_<L>    v1 on explicit cells, L on vague cells (below / unsettled / hedged / conflict / excluded), NaN on silent / general_normal
            cells (use with tgt_nan_w0=True: silent cells untrained, as in the x mask); Synovitis / Fracture / Contusion = L on all cells
            (they keep all cells in xF)
  xvcx_<L>  xv_<L> + Codex Effusion / Synovitis / Baker's reads wherever read (those cells trained, also when silent): xF mask idea +
            better vague-cell labels + image reads in one target
  h_v3      hybrid H (work/pl_research/c67): v1 probability where the report states anything (v1 status != not_mentioned), v3's p / 100
            where silent (v3 = Claude forecast of the image label from the report context; fold-free, no fitting; gold read NOT clean:
            v3's prompt saw gold-58 summaries)
  grp1      group-wise recombination (pre-registered 20:10): v4b on ligaments / menisci / OA (ACL, MCL, MM, LM, MOA, LOA, PF) and
            consnp on fluid + expert-owned classes (Effusion, Baker's, Synovitis, Contusion, Fracture). Motivation: group reads (v4b leads
            ligaments / OA over 5 folds, consnp leads fluid on folds 3-4 (lookat +5.3 vs v4b -2.0), both neutral on menisci). Clean test:
            folds 0-2 vs tl_v4b (consnp's fluid was never read there)
  vsf       vague-cell calibration everywhere + v4b's calibrated silent cells only where silence is not absence (pre-registered
            20:31 after vag15 f3): v4b with not_mentioned cells back at v1's 0.15 on the 7 structural classes (ACL, MCL, MM, LM, MOA,
            LOA, PF = vag15 there) and v4b's silent values kept on Effusion, Baker's, Synovitis, Contusion, Fracture. Motivation (f3):
            vag15 lifts report reads / menisci (tr9 +2.7 vs v4b -0.4; menisci trusted +2.6 vs -8.3) but loses fluid / Synovitis
            gold (Baker's -48 vs +12, Synovitis Codex -9 vs +63). Build alone: build_targets.py vsf
  vsfe      vsf with Synovitis / Contusion / Fracture back at vag15 (v1-level silence): v4b's calibrated silent cells only on Effusion /
            Baker's (pre-registered 21:15 after vsf f3: vsf's 7 structural columns equal vag15's, yet its structural report reads
            fell from vag15's, so the extra silent targets act through the shared backbone; Syn / C / F are expert-owned in 0.961,
            so the generalist needs them only for their effect on the owned classes). Build alone: build_targets.py vsf
  v1bak036  v1 everywhere except silent (not_mentioned) Baker's cells = 0.036 (v4b's silent Baker's value; v1 0.15): the pure single-
            rule supplier for the Baker's column splice probe (user ~23:40 via Master). Only Baker's column, NaN elsewhere: train with
            tgt_findings=Baker's tgt_partial=True. Build alone: build_targets.py vsf
  v1pfu85   v1 everywhere except PF 'unsettled' cells = 0.85 (v1 ~0.60; gold-per-status: PF 'unsettled' 7 / 8 positive; GOLD-DERIVED
            rule -> gold reads of this arm are fitted-to in direction): PF-only pilot (user ~00:30 via Master). PF column only, NaN elsewhere:
            tgt_findings=PF_OA tgt_partial=True. Build alone: build_targets.py vsf
  v1syn     v1 everywhere except silent (not_mentioned) Synovitis = v4b's per-study context values (Synovitis column only, NaN
            elsewhere): the crop-expert text targets for the Synovitis 0.961-like test (user ~00:30 via Master). tgt_findings=Synovitis
            tgt_partial=True. Build alone: build_targets.py vsf
  vsy, vcf  decomposition of the shared-backbone interference (Master / user ~21:25 relayed, pre-registered 21:22): base vag15 (all silence at
            v1's 0.15) + v4b's calibrated silent cells on ONE group: vsy = Synovitis only, vcf = Contusion + Fracture only (vsfe =
            Effusion + Baker's only; vsf = all three groups). Build alone: build_targets.py vsf
  v1fxn     v1 everywhere except Fracture 'stated normal' cells at the rules values (user ~01:40 via Master: split the Fracture rules):
            general_normal 0.05 and any cell with a negative bone fact (no fracture / normal bone, no positive fracture fact) 0.03;
            v1 has them at ~0.14, level with silence (0.15). Assumption: a report that states normal bone means no fracture, silence
            does not. Fracture column only: tgt_findings=Fracture tgt_partial=True. Build alone: build_targets.py fx
  v1fxnp    v1fxn + fracture facts set the value (rules 'acute fracture 0.9 x assertion': hedged 0.40 -> 0.50, unsettled 0.60 -> 0.87)
  cx_<L>    label set L with Codex Effusion reads replacing Effusion where read (cross-fitted: the file is only used on fold K's
            training studies, which never include fold K) and Codex Baker's reads replacing Baker's on the 517 report-vague studies
            where read; Synovitis Codex reads likewise (irrelevant for 0.961's columns, kept for features). cx_v1 holds only the
            3 read columns (NaN where unread = the trainer's v1 target): use with tgt_findings=Effusion+Synovitis+Baker's tgt_partial=True

conda run -n rsna python experiments/v3/oct15_labels/scripts/build_targets.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(ROOT))
from common.labels import probability_table, status_table  # noqa: E402
from common.paths import FINDINGS, KEYS  # noqa: E402

T = EXP / "data/targets"
EXPLICIT = ("meets", "denied", "structure_normal")
ST = status_table("v1")
P1 = probability_table("v1")
codex = pd.read_csv(ROOT / "data/derived/synovitis_labels_codex.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
bak = pd.read_csv(ROOT / "data/image_labels/reads.csv").query("run == 'codex_bakers' and finding == 'bakers_cyst'").groupby("study_id").probability.mean()


def write(name, k, df):
    df.index.name = "StudyInstanceUID"
    assert df.columns.isin(FINDINGS).all()
    v = df.to_numpy(float)
    assert np.isnan(v).all() or (np.nanmin(v) >= 0 and np.nanmax(v) <= 1), name
    df.to_csv(T / f"{name}_f{k}.csv")


def v3_table(ids):
    import json
    v3 = json.load(open(ROOT / "data/text_labels/v3.json"))
    return pd.DataFrame([[v3[s]["findings"][k]["p"] / 100 for k in KEYS] for s in ids], index=ids, columns=FINDINGS)


def main():
    ids = ST.index
    expl = pd.DataFrame(ST[KEYS].isin(EXPLICIT).to_numpy(), index=ids, columns=FINDINGS)
    p1 = pd.DataFrame(P1.loc[ids, KEYS].to_numpy(float), index=ids, columns=FINDINGS)
    p3 = v3_table(ids)
    h = p1.where(~ST[KEYS].isin(("not_mentioned",)).to_numpy(), p3)
    for k in range(5):
        v4b = pd.read_csv(T / f"v4b_f{k}.csv").set_index("StudyInstanceUID").reindex(ids)[FINDINGS]
        cns = pd.read_csv(T / f"consnp_f{k}.csv").set_index("StudyInstanceUID").reindex(ids)[FINDINGS]
        g1 = v4b.copy()
        for f in ("Effusion", "Baker's", "Synovitis", "Contusion", "Fracture"):
            g1[f] = cns[f]
        write("grp1", k, g1)
        write("h_v3", k, h)
        write("nan_sfc", k, pd.DataFrame(np.nan, index=ids, columns=["Synovitis", "Fracture", "Contusion"]))
        for L in ("consnp", "consv3", "v5"):
            lab = pd.read_csv(T / f"{L}_f{k}.csv").set_index("StudyInstanceUID").reindex(ids)[FINDINGS]
            write(f"v1x_{L}", k, p1.where(expl, lab))
            xv = p1.where(expl, lab).where(~ST[KEYS].isin(("not_mentioned", "general_normal")).to_numpy())
            xv[["Synovitis", "Fracture", "Contusion"]] = lab[["Synovitis", "Fracture", "Contusion"]]
            write(f"xv_{L}", k, xv)
            xvcx = xv.copy()
            for f in ("Effusion", "Synovitis"):
                r = codex[f].dropna()
                r = r[r.index.isin(ids)]
                xvcx.loc[r.index, f] = r.to_numpy()
            b = bak[bak.index.isin(ids)]
            xvcx.loc[b.index, "Baker's"] = b.to_numpy()
            write(f"xvcx_{L}", k, xvcx)
        for L in ("v1", "consnp", "consv3"):
            lab = pd.DataFrame(np.nan, index=ids, columns=["Effusion", "Synovitis", "Baker's"]) if L == "v1" else \
                pd.read_csv(T / f"{L}_f{k}.csv").set_index("StudyInstanceUID").reindex(ids)[FINDINGS]
            for f in ("Effusion", "Synovitis"):
                r = codex[f].dropna()
                lab.loc[r.index.intersection(ids), f] = r.reindex(r.index.intersection(ids)).to_numpy()
            b = bak[bak.index.isin(ids)]
            lab.loc[b.index, "Baker's"] = b.to_numpy()
            write(f"cx_{L}", k, lab)
    print("written:", sorted({p.name.rsplit("_f", 1)[0] for p in T.glob("*_f0.csv")}))


def vsf():
    for k in range(5):
        v4b = pd.read_csv(T / f"v4b_f{k}.csv").set_index("StudyInstanceUID")
        out = v4b.copy()
        for f in ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA"):
            out.loc[ST.reindex(out.index)[KEYS[FINDINGS.index(f)]] == "not_mentioned", f] = 0.15
        write("vsf", k, out[FINDINGS])
        vag = pd.read_csv(T / f"vag15_f{k}.csv").set_index("StudyInstanceUID").reindex(out.index)
        for f in ("Synovitis", "Contusion", "Fracture"):
            out[f] = vag[f]
        write("vsfe", k, out[FINDINGS])
        bak = pd.DataFrame(np.nan, index=out.index, columns=["Baker's"])   # v1bak036 (user ~23:40 via Master: Baker's-only splice probe)
        bak.loc[ST.reindex(out.index)[KEYS[FINDINGS.index("Baker's")]] == "not_mentioned", "Baker's"] = 0.036
        write("v1bak036", k, bak)                          # NaN = the trainer's v1 target; use tgt_findings=Baker's tgt_partial=True
        pf = pd.DataFrame(np.nan, index=out.index, columns=["PF OA"])     # v1pfu85 (user ~00:30 via Master: PF pilot)
        pf.loc[ST.reindex(out.index)[KEYS[FINDINGS.index("PF OA")]] == "unsettled", "PF OA"] = 0.85
        write("v1pfu85", k, pf)                            # use tgt_findings=PF_OA tgt_partial=True
        sy = pd.DataFrame(np.nan, index=out.index, columns=["Synovitis"])  # v1syn (user ~00:30 via Master: Synovitis 0.961-like test)
        m = (ST.reindex(out.index)[KEYS[FINDINGS.index("Synovitis")]] == "not_mentioned").to_numpy()
        sy.loc[m, "Synovitis"] = v4b.reindex(out.index).loc[m, "Synovitis"].to_numpy()
        write("v1syn", k, sy)                              # v4b's per-study silent Synovitis values; tgt_findings=Synovitis tgt_partial=True
        for name, fs in (("vsy", ("Synovitis",)), ("vcf", ("Contusion", "Fracture"))):
            d = vag.copy()
            for f in fs:
                d[f] = v4b.reindex(d.index)[f]
            write(name, k, d[FINDINGS])


def fx():
    sys.path.insert(0, str(EXP / "scripts"))
    from fx_split import build
    val, src = build()                                     # rules Fracture value + the rule that sets it (fold-free, no fitting)
    groups = {"v1fxn": ["e status general_normal", "g negative fact 0.03"]}
    groups["v1fxnp"] = groups["v1fxn"] + ["h acute fracture 0.9 x assert"]
    for name, rules in groups.items():
        out = pd.DataFrame(np.nan, index=val.index, columns=["Fracture"])
        m = src.isin(rules)
        out.loc[m, "Fracture"] = val[m]
        for k in range(5):
            write(name, k, out)                            # use tgt_findings=Fracture tgt_partial=True


if __name__ == "__main__":
    {"vsf": vsf, "fx": fx}.get(sys.argv[1] if sys.argv[1:] else "", main)()
