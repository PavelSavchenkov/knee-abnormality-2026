"""Local check of oct20_stack_all_v2 (OAI: no; v2 = v1 with P2 OFF by default = eff961, the 0.961 Effusion; --switch p2 = v1's P2) on the 88-study check list (gold-58 + 20 random + 10 edge; md5-equal to every probe's list).
--switch mfs / q20 / q25 / lpw picks the kernel{tag}/ variant (default: all switches OFF). The working package is built by the notebook's own setup
block (dataset slugs mapped to the local dataset folders). Gates:
 G1  untouched columns (MOA, MM, and Fracture / PF / Baker's while their switch is OFF) = 0.961's local check: < 1e-6; Baker's and (v2,
     eff961) Effusion (window fusion fp noise) <= 1e-5;
 G2  each spliced column = its single-column probe's check on the same studies (max |dp| <= 1e-4, Spearman >= 0.99999; Kaggle mode 1e-3 /
     0.9999): ACL vs oct18_probe_acl_b3g, LOA vs oct18_probe_loa_allup, Synovitis vs oct16_syn_cnx_probe, Contusion vs oct17_probe_cont_fs,
     + ON switches: Fracture vs oct19_probe_frac_mfs, PF vs oct18_probe_pf_allup, Baker's vs oct19_probe_bakers_allpw;
     MCL (declared re-fit, refit.json): (a) the pre-DEC MCL = Q21's pre-Q6 MCL (oct18_probe_mcl_stack's final MCL with its own Q6 inverted on
     the 0.961 Contusion) <= 1e-4; (b) the kernel's MCL constants = refit.json and final MCL = decouple(pre-DEC MCL | final Contusion) < 1e-6;
 G3  LM: the kernel's constants = Q22's dec.json and LM = decouple(pre-DEC LM | pre-DEC ACL) < 1e-6;
 G4  P1: the shared b3g parts mcl_member2 / acl_member2 = the probes' parts (oct18_probe_mcl_stack / oct18_probe_acl_b3g) <= 1e-4;
     eff961 (v2 default): the kernel's eff recipe == 0.961's config 'eff' and its eff_pipelines == 0.961's syn pipelines (the effb0 crops);
     P2 (--switch p2 only): the window-fusion Effusion part = 0.961's win_fused part (<= 1e-5) and final Effusion = the eff recipe (P2 stats)
     on this run's win_fused Effusion and eff_expert (< 1e-6, logit); info: Spearman(Effusion, 0.961);
 G5  0 failed studies, all finite, every spliced part present on every study; timing -> Kaggle estimate (limit 540).
 G6  lpw (lm_allpw, C184 G1; no probe of its own, so no G2 line): (a) every non-LM column = the same run without lpw (pre-DEC; < 1e-6,
     Baker's <= 1e-5; that run must exist); (b) pre-DEC LM == the meniscus recipe on this run's own parts (base = mean of the lm_pw_* parts,
     expert = oa_fused LM else the men experts' mean) < 1e-6 logit; (c) on the gold-58, the base shift b_pw - b_dep (this run's lm_pw_* vs
     main / nfnet / rad parts) vs the offline 5-fold shift (expected_lm_allpw.csv, build_lpw.py): Spearman >= 0.9 and median |diff| <= 0.05
     logit (named residual: live DICOM vs cached training preds; main as logit(mean p) vs mean logit).
KAGGLE MODE (--kaggle DIR): the same gates on a commit run's outputs (check_pred.csv, check_pred_base.csv, check_parts.csv, check_timing.json)
vs the 0.961 / probes' Kaggle checks.
conda run -n rsna python submissions/oct20_stack_all_v2/run_local.py [--switch p2 mfs q20 q25 lpw] [--workers 8] [--gpu-gb 11] [--gates-only] [--build-only]"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
REPO = HERE.parents[1]
BASE = SUB / "oct13_f5_win"
Q21, Q22 = SUB / "oct18_probe_mcl_stack", SUB / "oct19_probe_acl_b3g_q8"
LOCAL = {"knee-oct17-mcl-zoom": SUB / "oct17_probe_mcl_zoom/dataset", "knee-oct18-mcl-sd2": SUB / "oct18_probe_mcl_sd2/dataset",
         "knee-oct18-acl-b3g": SUB / "oct18_probe_acl_b3g/dataset", "knee-oct18-loa-allup": SUB / "oct18_probe_loa_allup/dataset",
         "knee-oct16-syn-cnx-probe": SUB / "oct16_syn_cnx_probe/package", "knee-oct17-cont-fs": SUB / "oct17_probe_cont_fs/dataset",
         "knee-oct19-frac-mfs": SUB / "oct19_probe_frac_mfs/dataset", "knee-oct19-bakers-allpw": SUB / "oct19_probe_bakers_allpw/dataset"}
SPLICED = {"ACL": SUB / "oct18_probe_acl_b3g", "Lateral OA": SUB / "oct18_probe_loa_allup", "Synovitis": SUB / "oct16_syn_cnx_probe",
           "Contusion": SUB / "oct17_probe_cont_fs"}
SWITCH = {"mfs": ("Fracture", SUB / "oct19_probe_frac_mfs", "frac_mfs"), "q20": ("PF OA", SUB / "oct18_probe_pf_allup", "oa_fused_pf"),
          "q25": ("Baker's", SUB / "oct19_probe_bakers_allpw", "win_fused_bk")}
RATIO = {"basin": 3.81 / (45 / 88), "main": 0.40 / (11 / 88), "main_sd": 0.40 / (11 / 88), "win": 0.57 / (11 / 88), "oa": 1.94 / (16 / 88),
         "fc": 3.25 / (36 / 88), "syn": 1.76 / (25 / 88)}
N_TEST, OVERHEAD_S = 1300, 249
KAGGLE = None                      # --kaggle DIR: gates on a commit run's outputs
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))  # noqa: E731


def ref_pred(d):
    return pd.read_csv(d / ("kaggle_output/check_pred.csv" if KAGGLE else "local_check_pred.csv"), index_col=0)


def ref_parts(d):
    if KAGGLE:
        pp = pd.read_csv(d / "kaggle_output/check_parts.csv", index_col=[0, 1])
    else:
        pp = pd.read_parquet(d / "local_check_parts.parquet")
    return pp


def kernel(tag):
    return nbformat.read(next((HERE / ("kernel" + tag)).glob("*.ipynb")), 4).cells[0].source


def build_work(code, work):
    import shutil
    if work.exists():                                    # the setup's links assert 'not exists': start from a clean working copy
        shutil.rmtree(work)
    a = code.index("ZMEM = _find(")
    b = code.index("PKG = WORK\n") + len("PKG = WORK\n")
    block = re.sub(r'^WORK = "/kaggle/working/pkg_stack".*$', "", code[a:b], flags=re.M)
    g = {"os": os, "json": json, "PKG": str(BASE / "package"), "WORK": str(work),
         "_find": lambda slug, fname: str(LOCAL[slug]) if (LOCAL[slug] / fname).exists() else (_ for _ in ()).throw(FileNotFoundError(slug))}
    exec("\n".join(code[code.index(k):code.index("\n", code.index(k))] for k in ("STACK = ", "B3G_SHARED = ", "P2_EFF = ", "SWITCHES = ", "PF_Q20 = ")), g)
    exec(block, g)
    return (g["PKG"],) + dec_of(code)


def dec_of(code):
    """The kernel's own post-hoc de-coupling (decouple, DEC), executed from its source."""
    ns = {"np": np, "pd": pd}
    exec(code[code.index("\nDEC = "):code.index('test = pd.read_csv(COMP + "/test.csv")')], ns)
    return ns["decouple"], ns["DEC"]


def gates(final, base, parts, sw, stats, dec, cfg):
    ids = final.index
    ref = ref_pred(BASE).loc[ids]
    tol, rho_min = (1e-3, 0.9999) if KAGGLE else (1e-4, 0.99999)
    src = "Kaggle" if KAGGLE else "local"
    swp = [s for s in sw if s in SWITCH]                  # switches with a single-column probe (lpw has none: G6)
    on = {SWITCH[s][0] for s in swp} | ({"Lateral Meniscus"} if "lpw" in sw else set())
    fp = ["Baker's"] + ([] if "p2" in sw else ["Effusion"])     # window-fusion fp noise columns (v2 eff961: Effusion = 0.961's form)
    changed = set(SPLICED) | {"MCL", "Lateral Meniscus"} | ({"Effusion"} if "p2" in sw else set()) | on
    exact = [c for c in final.columns if c not in changed and c not in fp]
    d1 = float((final[exact] - ref[exact]).abs().to_numpy().max())
    ok = d1 < (1e-3 if KAGGLE else 1e-6)
    msg = f"G1 untouched {exact} vs 0.961 {src}: max |dp| {d1:.2e}"
    for c in [c for c in fp if c not in changed]:
        db = float((final[c] - ref[c]).abs().max())
        ok &= db <= (1e-3 if KAGGLE else 1e-5)
        msg += f"; {c} {db:.2e} (<= 1e-5)"
    print(msg + f": {ok}")
    g2 = dict(SPLICED) | {SWITCH[s][0]: SWITCH[s][1] for s in swp}
    for c, d in g2.items():
        r = ref_pred(d)
        common = ids.intersection(r.index)
        dp = float((final.loc[common, c] - r.loc[common, c]).abs().max())
        rho = final.loc[common, c].corr(r.loc[common, c], method="spearman")
        g = dp <= tol and rho >= rho_min and len(common) == len(ids)
        ok &= g
        print(f"G2 {c} vs {d.name} {src} check ({len(common)} studies): max |dp| {dp:.2e}, Spearman {rho:.6f}: {g}")
    # MCL (declared re-fit)
    q6o, refit = json.load(open(Q21 / "dec.json"))["MCL"], json.load(open(HERE / "refit.json"))["dec"]
    r21 = ref_pred(Q21).loc[ids]
    z21 = lg(r21["MCL"]) + q6o["b"] * (q6o["c0"] + q6o["c"] * lg(ref["Contusion"]))           # Q21's pre-Q6 MCL (its context = 0.961 Contusion)
    da = float((1 / (1 + np.exp(-z21)) - base["MCL"]).abs().max())
    q = dec["MCL"]
    zm = lg(base["MCL"]) - q["b"] * (q["c0"] + q["c"] * lg(base["Contusion"]))
    dbm = float((1 / (1 + np.exp(-zm)) - final["MCL"]).abs().max())
    g = da <= tol and q == refit["MCL"] and dbm < 1e-6
    ok &= g
    print(f"G2 MCL: (a) pre-DEC MCL vs Q21's pre-Q6 MCL {da:.2e} (<= {tol:g}); (b) constants = refit.json {q == refit['MCL']}, final MCL == "
          f"decouple(pre-DEC MCL | final Contusion) {dbm:.2e} (< 1e-6): {g}")
    lm, q22 = "Lateral Meniscus", json.load(open(Q22 / "dec.json"))["Lateral Meniscus"]
    q = dec[lm]
    z3 = lg(base[lm]) - q["b"] * (q["c0"] + q["c"] * lg(base[q["context"]]))
    d3 = float((1 / (1 + np.exp(-z3)) - final[lm]).abs().max())
    g = q == q22 and d3 < 1e-6
    ok &= g
    print(f"G3 LM: constants = Q22's dec.json {q == q22}; LM == decouple(pre-DEC LM | pre-DEC ACL) {d3:.2e} (< 1e-6): {g}")
    for k, d in (("mcl_member2", Q21), ("acl_member2", SUB / "oct18_probe_acl_b3g")):
        pr = ref_parts(d)
        a_, b_ = parts[k].iloc[:, 0], pr.loc[k].dropna(axis=1, how="all").iloc[:, 0].reindex(parts[k].index)
        d4 = float((a_ - b_).abs().max())
        ok &= d4 <= tol
        print(f"G4 P1 shared b3g {k} vs {d.name}'s part: max |dp| {d4:.2e} (<= {tol:g})")
    if "p2" not in sw:                                      # v2 default eff961: the column itself is in G1; here the recipe / members
        c0 = json.load(open(BASE / "package/config.json"))
        ge, gp = cfg["eff"] == c0["eff"], cfg["syn"]["eff_pipelines"] == c0["syn"]["pipelines"]
        ok &= ge and gp
        print(f"G4 eff961: eff recipe == 0.961's config {ge}; eff_pipelines == 0.961's syn pipelines (effb0 crops) {gp}: {ge and gp}")
    else:
        ok &= gate_p2(final, parts, cfg, ids, ref)
    need = ["cont_fs", "mcl_member2", "acl_member2", "oa_fused_loa_extra"] + [SWITCH[s][2] for s in swp] + \
        (["lm_pw_main", "lm_pw_nfnet", "lm_pw_rad"] if "lpw" in sw else [])
    cover = {k: int(parts[k].notna().all(axis=1).sum()) if k in parts else -1 for k in need}
    fin = bool(np.isfinite(final.to_numpy()).all())
    failed = (stats or {}).get("failed_main", 0)
    g = fin and all(v == len(ids) for v in cover.values()) and not failed
    ok &= g
    print(f"G5 finite {fin}, failed main {failed}, part coverage {cover} of {len(ids)}: {g}")
    if "lpw" in sw:
        ok &= gate_lpw(base, parts, sw, cfg)
    if stats and not KAGGLE:
        tot = 0.0
        print("TIMING (local, 88 studies) -> Kaggle estimate (per-stage local->Kaggle ratios from 0.961; the fast runner halves the four GPU stages):")
        for k, v in stats.items():
            if isinstance(v, dict) and "wall_s" in v:
                ks = v["wall_s"] / len(ids) * RATIO.get(k, 7.0)
                tot += ks
                print(f"  {k:8s} local {v['wall_s']:6.1f} s -> Kaggle ~{ks:.2f} s / study")
        print(f"  Kaggle estimate (original runner): {(tot * N_TEST + OVERHEAD_S) / 60:.0f} min (limit 540)")
    print(f"GATES (switches {sw or 'none'}) {'PASS' if ok else 'FAIL'}")
    return ok


def gate_p2(final, parts, cfg, ids, ref):
    """v1's G4 P2 (--switch p2 only)."""
    pb = ref_parts(BASE)
    wf0 = pb.loc["win_fused"].dropna(axis=1, how="all")["Effusion"].reindex(ids)
    d5 = float((parts["win_fused"]["Effusion"] - wf0).abs().max())
    e = cfg["eff"]
    s = e["stats"]["Effusion"]
    zb, ze = lg(parts["win_fused"]["Effusion"]), lg(parts["eff_expert"]["Effusion"])
    exp_eff = (1 - e["w"]) * zb + e["w"] * ((ze - s["mu_e"]) / s["sd_e"] * s["sd_b"] + s["mu_b"])
    d6 = float((lg(final["Effusion"]) - exp_eff).abs().max())
    p2 = json.load(open(SUB.parent / "experiments/v3/oct16_experts/out/p2_eff_stats.json"))
    g = d5 <= (1e-3 if KAGGLE else 1e-5) and d6 < 1e-6 and all(abs(s[k] - p2[k]) < 1e-12 for k in p2)
    print(f"G4 P2: win_fused Effusion vs 0.961's part {d5:.2e} (<= 1e-5); final Effusion == eff recipe (P2 stats) {d6:.2e} (< 1e-6, logit): {g}; "
          f"info Spearman(Effusion, 0.961) {final['Effusion'].corr(ref['Effusion'], method='spearman'):.5f}")
    return g


def gate_lpw(base, parts, sw, cfg):
    lm, ids = "Lateral Meniscus", base.index
    tag0 = "".join(f"_{s}" for s in sw if s != "lpw")
    f0 = (KAGGLE / "check_pred_base.csv") if KAGGLE else HERE / f"local_check_pred_base{tag0}.csv"
    if KAGGLE:
        print("G6 (a) Kaggle mode: compare with the no-lpw commit's check_pred_base.csv by hand")
        ga = True
    elif not f0.exists():
        print(f"G6 (a) {f0.name} missing: run the variant without lpw first: False")
        ga = False
    else:
        r0 = pd.read_csv(f0, index_col=0).loc[ids]
        cols = [c for c in base.columns if c not in (lm, "Baker's")]
        da, db = float((base[cols] - r0[cols]).abs().to_numpy().max()), float((base["Baker's"] - r0["Baker's"]).abs().max())
        ga = da < 1e-6 and db <= 1e-5
        print(f"G6 (a) non-LM pre-DEC vs {f0.name}: max |dp| {da:.2e} (< 1e-6), Baker's {db:.2e} (<= 1e-5): {ga}")
    men = cfg["men"]
    b = sum(lg(parts[k][lm]) for k in ("lm_pw_main", "lm_pw_nfnet", "lm_pw_rad")) / 3
    e = lg(parts["oa_fused"][lm]).fillna(sum(lg(parts[f"oa_{k}"][lm]) for k in men["experts"]) / len(men["experts"]))
    st = men["stats"][lm]
    z = (e - st["mu_e"]) * st["sd_b"] / st["sd_e"] + st["mu_b"]
    o = (1 - men["w"]) * b + men["w"] * z.fillna(b)
    d_b = float((o - lg(base[lm])).abs().max())
    gb = d_b < 1e-6
    print(f"G6 (b) pre-DEC LM == men recipe on the lm_pw_* base: max |dz| {d_b:.2e} (< 1e-6): {gb}")
    x = pd.read_csv(HERE / "expected_lm_allpw.csv", index_col=0)["d_base"]
    g = ids.intersection(x.index)
    dep = sum(lg(parts[k].loc[g, lm]) for k in ("main", "nfnet", "rad")) / 3
    d_pkg = b.loc[g] - dep
    rho, med = d_pkg.corr(x.loc[g], method="spearman"), float((d_pkg - x.loc[g]).abs().median())
    gc = len(g) == 58 and rho >= 0.9 and med <= 0.05
    print(f"G6 (c) gold-{len(g)} base shift vs offline 5-fold: Spearman {rho:.4f} (>= 0.9), median |diff| {med:.4f} (<= 0.05), max "
          f"{float((d_pkg - x.loc[g]).abs().max()):.3f}; shift sd package {d_pkg.std():.3f} vs offline {x.loc[g].std():.3f}: {gc}")
    return ga and gb and gc


def main():
    global KAGGLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", nargs="*", default=[], choices=["p2"] + list(SWITCH) + ["lpw"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu-gb", type=float, default=11)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--kaggle", default="", help="dir of a commit run's outputs (gates vs the Kaggle checks)")
    a = ap.parse_args()
    sw = [s for s in ("p2", "mfs", "q20", "q25", "lpw") if s in a.switch]
    tag = "".join(f"_{s}" for s in sw)
    code = kernel(tag)
    if a.kaggle:
        KAGGLE = HERE / a.kaggle
        final = pd.read_csv(KAGGLE / "check_pred.csv", index_col=0)
        base = pd.read_csv(KAGGLE / "check_pred_base.csv", index_col=0)
        pp = pd.read_csv(KAGGLE / "check_parts.csv", index_col=[0, 1])
        t = json.load(open(KAGGLE / "check_timing.json"))
        cfg = json.load(open(HERE / f"work_pkg{tag}/config.json"))
        print(f"KAGGLE MODE ({sw or 'no switches'}): {KAGGLE} ({len(final)} studies); OAI: no")
        gates(final, base.loc[final.index], {k: pp.loc[k].dropna(axis=1, how="all") for k in pp.index.get_level_values(0).unique()}, sw,
              {"failed_main": sum(len(t[g]["failed"]) for g in ("main", "edge"))}, dec_of(code)[1], cfg)
        return
    if a.gates_only:
        final = pd.read_csv(HERE / f"local_check_pred{tag}.csv", index_col=0)
        base = pd.read_csv(HERE / f"local_check_pred_base{tag}.csv", index_col=0)
        pp = pd.read_parquet(HERE / f"local_check_parts{tag}.parquet")
        gates(final, base, {k: pp.loc[k].dropna(axis=1, how="all") for k in pp.index.get_level_values(0).unique()}, sw,
              json.load(open(HERE / f"local_stats{tag}.json")), json.load(open(HERE / f"local_dec{tag}.json")),
              json.load(open(HERE / f"work_pkg{tag}/config.json")))
        return
    chk = pd.read_csv(HERE / "check_studies.csv")
    (HERE / "run_local.pid").write_text(str(os.getpid()))
    pkg, decouple, dec = build_work(code, HERE / f"work_pkg{tag}")
    sys.path[:0] = [pkg, str(REPO)]
    import torch
    import ensemble
    for mod in (ensemble, ensemble.syn_stage, ensemble.oa_stage, ensemble.fc_stage, ensemble.ki):
        assert Path(mod.__file__).parent == Path(pkg), mod.__file__
    cfg = json.load(open(Path(pkg) / "config.json"))
    if a.build_only:
        sm = json.load(open(Path(pkg) / cfg["syn"]["members"]))
        missing = [f for m in sm.values() for f in m["files"] if not (Path(pkg) / f).exists()]
        missing += [f for m in cfg["oa"]["members"].values() for f in m["files"] if not (Path(pkg) / f).exists()]
        missing += [f for m in cfg["fc"]["members"].values() for f in m["files"] if not (Path(pkg) / f).exists()]
        missing += [f for f in cfg["mcl_sd"]["nfnet"] if not (Path(pkg) / f).exists()]
        if cfg.get("bakers_allpw"):
            bk = cfg["bakers_allpw"]
            missing += [f for f in bk["main"] + [c for cs in bk["basin"].values() for c in cs] if not (Path(pkg) / f).exists()]
        if cfg.get("lm_allpw"):
            lp = cfg["lm_allpw"]
            missing += [f for f in [c for cs in lp["basin"].values() for c in cs] if not (Path(pkg) / f).exists()]
            missing += [f"{lp['main_dir']}/{f}" for f in json.load(open(Path(pkg) / lp["main_dir"] / "config.json"))["weights"]
                        if not (Path(pkg) / lp["main_dir"] / f).exists()]
        print(f"BUILD-ONLY switches {sw}: syn members {len(sm)}, oa members {len(cfg['oa']['members'])}, fc members {list(cfg['fc']['members'])}, "
              f"missing files {missing}; cont_fs {cfg['cont_fs']['name']}; frac_mfs {bool(cfg.get('frac_mfs'))}; oa extra_sets "
              f"{cfg['oa'].get('extra_sets')}; bakers_allpw {bool(cfg.get('bakers_allpw'))}; lm_allpw {cfg.get('lm_allpw')}; eff stats {cfg['eff']['stats']['Effusion']}; DEC {dec}")
        ck, det, wkw = ensemble.basin_args(cfg, pkg)
        print(f"basin_args: {len(ck)} ckpts, win ckpts {len(wkw['ckpts'])}")
        return
    torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    series = pd.read_csv(REPO / "data/original_data/train_series.csv")
    t = time.time()
    base, parts, failed, secs = ensemble.predict(str(REPO / "data/original_data/train_series"), chk.StudyInstanceUID.tolist(), series, pkg, a.workers)
    print(f"local: {len(base)} studies in {time.time() - t:.0f}s, failed main {len(failed)}; " +
          ", ".join(f"{k} {v['wall_s']:.0f}s" for k, v in ensemble.STATS.items() if isinstance(v, dict) and "wall_s" in v), flush=True)
    final = decouple(base)
    base.to_csv(HERE / f"local_check_pred_base{tag}.csv")
    final.to_csv(HERE / f"local_check_pred{tag}.csv")
    parts = {k: v if isinstance(v, pd.DataFrame) else v.to_frame("Synovitis") for k, v in parts.items()}
    pd.concat(parts, names=["part"]).to_parquet(HERE / f"local_check_parts{tag}.parquet")
    stats = dict(ensemble.STATS) | {"failed_main": len(failed)}
    json.dump(stats, open(HERE / f"local_stats{tag}.json", "w"), indent=1, default=str)
    json.dump(dec, open(HERE / f"local_dec{tag}.json", "w"), indent=1)
    gates(final, base, parts, sw, stats, dec, cfg)


if __name__ == "__main__":
    main()
