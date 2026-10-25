"""oct20_stack_all (Experts V3 10-19; USER GO via Master: the merged fold package for a 10-20 slot; OAI: no) = oct19_stack_fold's file + :
- Contusion FS (cont_fs, LB-won 57388555): the bfs_fs bone member (fc 'nofuse') spliced into Contusion after the bone block (its probe's code);
- SWITCHES (config keys present only when ON; each = its probe's own code): frac_mfs (m_fs fsmain member -> Fracture splice, 57399358),
  oa extra_sets {"pf": ...} + oa extra_align (PF Q20: PF from the fusion over ONLY oasep / oasep_nf, 57399364), bakers_allpw (Q25: the pw
  Basin models join the Basin pass for window export only, the pw main models the main window pass; Baker's = their window fusion, 57399558),
  lm_allpw (C184 G1, no probe of its own: LM's 3x3 base = the same pw generalists' study logits - a pw main pass + the pw Basin models in the
  Basin pass - before the meniscus recipe; the meniscus recipe is per column, so only LM changes);
- stage-argument helpers basin_args / oa_cfg / fc_cfg / syn_members, shared with fast.py (the two-GPU runner) so both paths call each stage
  with identical arguments. The post-hoc DEC (MCL Q6 re-fitted on the new Contusion, LM Q22) stays in the notebook.
oct19_stack_fold (Experts V3 10-18, Master: the LB-won column forms stacked on the 0.961 fold package; each block below is its
probe's own code, keyed by its config entry, on disjoint columns): MCL = Q21 (mcl_sd + mcl_member + mcl_member2; Q6 in the notebook),
LOA = Q15 (oa loa_extra / loa_set / loa_extra_align), ACL = Q16 (acl_member + acl_member2), Synovitis = the ConvNeXt crop slots of
57283240 (syn pipelines / fuse designs; eff_pipelines keeps Effusion on the effb0 crop experts). Every other column = 0.961.
oct18_probe_mcl_stack (Q21, USER U55 stack): = oct18_probe_mcl_sd2's file + config "mcl_member2" (b3g's area MCL spliced after the zoom member).
oct18_probe_mcl_sd2: + config "mcl_sd" (MCL = mean of the main SD / NFNet SD / deployed R50 logits, replacing the 3x3 MCL before the
MCL zoom splice). oct17_probe_mcl_zoom: + config "mcl_member" (one extra crop member in the syn stage, spliced into the MCL column; see below).
3-pipeline ensemble from raw DICOM (same file runs locally and on Kaggle):
- main: knee_infer.py (effb0, experiments/v0/oct7_baseline) on its config.json models; probability = mean over its models, synovitis
  logit mixed with effusion per model (config synovitis_effusion_w), exactly as the 0.940 package;
- Basin-code pipelines: basin_predict.py on config.json "basin" = {pipeline: [checkpoint files in the package]};
  pipeline logit = mean of its models' logits (all Basin models share one DICOM pass); config "basin_detector" (optional): Basin's
  mirror detector, which mirrors back series that the test-time side rule mis-sided (Basin models trained on corrected sides; the
  main pipeline trained on its own DICOM sides, so it gets no detector).
Final probability = sigmoid(mean over pipelines of the pipeline logits): equal weight per pipeline (validated by the 3rd baseline:
logit mean ~ z-scored mean ~ prob mean within +-0.0003 OOF).
oct10_syn3x3: Synovitis comes only from image-head models (trained on the Codex image reads, experiments/v1/oct10_synovitis):
- main: the 13-output models in config "models" (knee_infer's mixed mode: Synovitis from the 13-output models' image head with
  synovitis_ext_w = 1, the other 11 findings from the 12-output models; synovitis_effusion_w must be 0);
- Basin pipelines: config "basin_syn" = {pipeline: [image-head checkpoints]} (basin_predict returns their image head as Synovitis);
Synovitis = sigmoid(mean over the 3 pipelines of the pipeline image-head logits); the 11 other findings exactly as oct10_3pipe_3f.
oct11_oa probes (config "oa", experiments/v1/oct10_oa/probes.md): only the 3 OA columns change, to ONE model type (single model vs
single model, no blending): mode "main" = the main pipeline's own OA logits (baseline B); mode "expert" = one compartment-crop expert
(oa_stage.py, mean logit of its fold models); a study the expert cannot read falls back to the main pipeline's OA logits (logged).
oct12_syn3pipe (config "syn"): Synovitis from the synovitis crop experts of experiments/v1/oct11_synovitis_crops (syn_stage.py:
localiser v2 crops, Codex read target) on the 3 pipelines' encoders: per pipeline the mean logit of its experts (2 designs x
folds 0-2), Synovitis = sigmoid(mean over the 3 pipelines); a study the stage cannot read keeps the image-head Synovitis (logged).
oct12_syn_g (config "syn" image_heads_w = w): Synovitis = sigmoid(w x image-head logit (mean over the pipelines in "basin_syn" + main)
+ (1 - w) x crop-expert logit); w = 0 is oct12_syn3pipe.
oct12_men_3pipe: config "men" (make_men_recipe.py, folds 0-2 OOF): Medial / Lateral Meniscus = the pipelines' meniscus logits blended
with the OA experts' meniscus heads (same expert pass): o = (1 - w) b + w z(e), z = e rescaled to b's OOF mean / sd.
oct12_fc_3pipe (config "fc", experiments/v1/oct11_fracture_contusion): Fracture / Contusion = fc_stage.apply_recipe(pipelines' F / C
logits, F / C expert members' logits, fc recipe) = (1 - w) b + w z(e), chosen on folds 0-2 OOF; members: bone experts (localiser v3
crops); a study the stage cannot read keeps the pipelines' F / C (logged).
oct12_global_best: oct12_syn_g + the "men" block of oct12_men_3pipe + the "fc" block of oct12_fc_3pipe (disjoint columns).
oct13_f5_win: oct13_f5_eff + config syn "fuse": the synovitis crop-expert part from crop-level fusion per design (syn_stage.fuse_design,
mean|pm|g20|maxmix) + config oa "fuse": the OA experts' term (OA and meniscus recipes) from their crop-level fusion
(oa_stage, fc_stage.fuse_crops, geo|pm|g10|wmean) + config "win": Effusion / Baker's from the cross-backbone window fusion of the 3x3 (win_stage.py: main
models on every window + Basin per-image export, geo|pm|g20|wmean), before the expert recipes; + config fc "fuse": Fracture / Contusion from the crop-level fusion of the bone experts (fc_stage.fuse_crops:
every member's fold models over the shared crops, top-20 relevance gate, prob mean) in place of the recipe's member z-mean.
oct13_f5_eff (experiments/v2/oct13_deep_ensembling_test): the existing fold 3-4 checkpoints of main / NFNet / R50 / bone b_v3, b_lr3
(config lists only) + config "eff": Effusion blended with the synovitis crop experts' image-read Effusion head (oa_stage.apply_recipe
form, no de-coupling)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import knee_infer as ki  # noqa: E402
import basin_predict as bp  # noqa: E402
import oa_stage  # noqa: E402
import syn_stage  # noqa: E402
import fc_stage  # noqa: E402
import win_stage  # noqa: E402

FINDINGS = ki.FINDINGS
LOA = "Lateral OA"
STATS = {}


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def basin_args(cfg, pkg):
    """-> (ckpts, detector, win kwargs or None): the ONE Basin pass (deployed + basin_syn + mcl_sd NFNet SD + bakers_allpw pw models for window
    export only), as ensemble.predict and fast.py call bp.predict."""
    det = str(Path(pkg) / cfg["basin_detector"]) if cfg.get("basin_detector") else None
    msd, bk, win, lp = cfg.get("mcl_sd"), cfg.get("bakers_allpw"), cfg.get("win"), cfg.get("lm_allpw")
    ckpts = [str(Path(pkg) / c) for cs in list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) + ([msd["nfnet"]] if msd else []) for c in cs]
    bkck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in bk["basin"].items()} if bk else {}
    ckpts += sum(bkck.values(), [])
    if lp:                 # SWITCH lm_allpw: the pw Basin models' study logits (one run per checkpoint when bakers_allpw is also ON)
        ckpts += [c for c in (str(Path(pkg) / x) for cs in lp["basin"].values() for x in cs) if c not in ckpts]
    wck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in win["basin"].items()} if win else {}
    emb = (cfg.get("eff_main") or {}).get("basin")   # oct21_probe_eff_base: the EF3 Basin models (window export only, like the Q25 pw models)
    emck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in emb.items()} if emb else {}
    ckpts += [c for c in sum(emck.values(), []) if c not in ckpts]   # Master 10-21 11:4x: bp.predict exports windows only for checkpoints it RUNS
    wkw = {"ckpts": sum(wck.values(), []) + sum(bkck.values(), []) + sum(emck.values(), []), "cols": [FINDINGS.index(c) for c in win["cols"]]} if win else None
    return ckpts, det, wkw


def oa_cfg(cfg):
    return cfg["oa"]


def fc_cfg(cfg):
    return cfg["fc"]


def syn_members(cfg, pkg):
    members = json.load(open(Path(pkg) / cfg["syn"]["members"]))
    cs = cfg.get("cont_sxm")      # oct21_probe_cont_sxm: + the sxm crop expert (f0-2) in the syn stage (fast.py workers call this function too)
    if cs:
        assert cs["name"] not in members, f"cont_sxm member {cs['name']} shadows a syn member"
        members[cs["name"]] = {"cfg": cs["cfg"], "files": cs["files"]}
    for m in members.values():
        m["files"] = [str(Path(pkg) / f) for f in m["files"]]
    return members


def predict(root, studies, series, pkg, threads=None, workers=4):
    """series: the host series csv (StudyInstanceUID, SeriesInstanceUID, Fat_Suppression). Returns (final probabilities, parts =
    probabilities {pipeline or Basin checkpoint file: DataFrame}, failed main studies, per-study main load seconds)."""
    cfg = json.load(open(Path(pkg) / "config.json"))
    series = series[series.StudyInstanceUID.isin(studies)].assign(Fat_Suppression=lambda d: d.Fat_Suppression.fillna(0).astype(int))
    fs_of = dict(zip(series.SeriesInstanceUID, series.Fat_Suppression))
    t = time.time()   # Basin first: its DataLoader forks workers before knee_infer starts its thread pools
    msd = cfg.get("mcl_sd")    # oct18_probe_mcl_sd2: the NFNet SD fold models run in the same Basin pass (MCL only)
    win = cfg.get("win")   # oct13_f5_win: cross-backbone window fusion for win["cols"] (win_stage.py)
    wcols = [FINDINGS.index(c) for c in win["cols"]] if win else None
    wck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in win["basin"].items()} if win else {}
    bk = cfg.get("bakers_allpw")   # SWITCH Q25: pw basin models for window export only (appended; never in the pipeline means)
    bkck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in bk["basin"].items()} if bk else {}
    emb = (cfg.get("eff_main") or {}).get("basin")   # oct21_probe_eff_base / _c: EF3 Basin windows for Effusion's fusion; absent = form (a)
    emck = {p: [str(Path(pkg) / c) for c in cs] for p, cs in emb.items()} if emb else {}
    assert set(emck) <= set(wck), ("eff_main basin pipelines must be shipped ones", list(emck), list(wck))   # form (c): nfnet only
    ckpts, det, wkw = basin_args(cfg, pkg)
    z = bp.predict(series, root, ckpts, workers=workers, detector=det, win=wkw)
    STATS["basin"] = {"wall_s": time.time() - t, **bp.STATS}
    torch.cuda.empty_cache()   # knee_infer captures its CUDA graphs with nothing else on the GPU
    t = time.time()
    p_main, failed, secs = ki.predict(root, studies, fs_of, pkg, threads)
    STATS["main"] = {"wall_s": time.time() - t, **ki.STATS}
    torch._C._host_emptyCache()   # oct13_f5_win: knee_infer's cached pinned study buffers (Kaggle: 12.7 GB of the 30 GB after 117 studies)
    if msd:                       # oct18_probe_mcl_sd2: a second main pass with the 5 main SD fold models (own weights dir; MCL only)
        torch.cuda.empty_cache()
        t = time.time()
        p_sd, failed_sd, _ = ki.predict(root, studies, fs_of, str(Path(pkg) / msd["main_dir"]), threads)
        STATS["main_sd"] = {"wall_s": time.time() - t, "failed": len(failed_sd)}
        torch._C._host_emptyCache()
    lp = cfg.get("lm_allpw")   # SWITCH lm_allpw (C184 G1): a main pass with the 5 pw main fold models (own weights dir; LM only)
    if lp:
        torch.cuda.empty_cache()
        t = time.time()
        p_pw, failed_pw, _ = ki.predict(root, studies, fs_of, str(Path(pkg) / lp["main_dir"]), threads)
        STATS["main_pw"] = {"wall_s": time.time() - t, "failed": len(failed_pw)}
        torch._C._host_emptyCache()
    assert cfg["synovitis_ext_w"] == 1.0 and cfg["synovitis_effusion_w"] == 0.0, "Synovitis = the image heads alone"
    zs = {"main": pd.DataFrame(logit(p_main.loc[studies, FINDINGS]), index=pd.Index(studies, name="StudyInstanceUID"), columns=FINDINGS)}

    def pipe_logit(p, cs):
        zp = sum(z[str(Path(pkg) / c)] for c in cs) / len(cs)
        miss = [s for s in studies if s not in zp.index]   # study without rows in the series csv: cohort mean logit (logged)
        if miss:
            print(f"basin {p}: {len(miss)} studies not in the series csv -> cohort mean: {miss[:10]}", flush=True)
        return zp.reindex(studies).fillna(zp.mean())
    for p, cs in cfg["basin"].items():
        zs[p] = pipe_logit(p, cs)
    zsyn = {"main": zs["main"]["Synovitis"]} | {p: pipe_logit(p, cs)["Synovitis"] for p, cs in cfg["basin_syn"].items()}
    final_z = sum(zs.values()) / len(zs)
    zs_sd = {}
    if msd:                       # MCL = (main SD + NFNet SD + deployed R50) / 3; a study the SD main pass failed keeps the deployed main's MCL
        z_ms = pd.Series(logit(p_sd.loc[studies, ["MCL"]]).ravel(), index=pd.Index(studies, name="StudyInstanceUID"))
        z_ms[[s for s in studies if s in failed_sd]] = zs["main"].loc[[s for s in studies if s in failed_sd], "MCL"]
        z_nfs = pipe_logit("nfnet_sd", msd["nfnet"])["MCL"]
        zs_sd = {"mcl_sd_main": z_ms.to_frame("MCL"), "mcl_sd_nfnet": z_nfs.to_frame("MCL")}   # parts (checks only), added after final_z
        final_z["MCL"] = (z_ms + z_nfs + zs[msd["r50"]]["MCL"]) / 3
        print(f"mcl_sd: MCL = (main SD + NFNet SD + {msd['r50']}) / 3; SD main failed {len(failed_sd)}", flush=True)
    if lp:                        # LM base = (pw main + pw NFNet + pw R50) / 3; a study the pw main pass failed keeps the deployed main's LM
        c = lp["col"]
        z_mp = pd.Series(logit(p_pw.loc[studies, [c]]).ravel(), index=pd.Index(studies, name="StudyInstanceUID"))
        z_mp[[s for s in studies if s in failed_pw]] = zs["main"].loc[[s for s in studies if s in failed_pw], c]
        lpz = {"lm_pw_main": z_mp.to_frame(c)} | {f"lm_pw_{p}": pipe_logit(p, cs)[[c]] for p, cs in lp["basin"].items()}
        zs_sd |= lpz                                  # parts (checks only)
        final_z[c] = sum(v[c] for v in lpz.values()) / len(lpz)
        print(f"lm_allpw: {c} base = mean of {list(lpz)}; pw main failed {len(failed_pw)}", flush=True)
    if win:
        torch.cuda.empty_cache()
        t = time.time()
        n0 = len(win["main"])
        em = cfg.get("eff_main")   # oct21_probe_eff_main: the main slot of Effusion's 3x3 window fusion = the EF3 main models (own fold models,
        n1 = n0 + (len(bk["main"]) if bk else 0)   # appended to the ONE main window pass); the Basin windows = the shipped ones; no fallback
        mw = win_stage.main_windows(root, studies, fs_of, str(pkg), win["main"] + (bk["main"] if bk else []) + (em["main"] if em else []), cfg, wcols)
        fz, fz2, fz3, miss = {}, {}, {}, []
        for sid in studies:
            if sid in mw and all(c in bp.WIN.get(sid, {}) for cs in list(wck.values()) + list(emck.values()) for c in cs):
                m_ = mw[sid]
                fz[sid] = win_stage.fuse(m_ | {"s": m_["s"][:n0], "u": m_["u"][:n0]}, {p: [bp.WIN[sid][c] for c in cs] for p, cs in wck.items()}, win["gate"])
                if em:                 # the EF3 main models' windows (index n1:) + the SHIPPED Basin windows (wck), the shipped gate
                    fz3[sid] = win_stage.fuse(m_ | {"s": m_["s"][n1:], "u": m_["u"][n1:]}, {p: [bp.WIN[sid][c] for c in cs] for p, cs in (wck | emck).items()},
                                              win["gate"])
                if bk and all(c in bp.WIN.get(sid, {}) for cs in bkck.values() for c in cs):     # SWITCH Q25 (its probe's code)
                    fz2[sid] = win_stage.fuse(m_ | {"s": m_["s"][n0:], "u": m_["u"][n0:]}, {p: [bp.WIN[sid][c] for c in cs] for p, cs in bkck.items()},
                                              win["gate"])
            else:
                miss.append(sid)
        if miss:
            print(f"win: {len(miss)} studies without window outputs -> study-level 3x3 for {win['cols']}: {miss[:10]}", flush=True)
        fz = pd.DataFrame.from_dict(fz, orient="index", columns=win["cols"]).reindex(studies)
        zs["win_fused"] = fz.copy()
        final_z[win["cols"]] = fz.fillna(final_z[win["cols"]])
        if bk:                     # Q25: Baker's from the pw fusion; a study without pw windows keeps the deployed fusion (logged)
            assert set(bk["cols"]) <= set(win["cols"]), bk["cols"]
            assert all(v["s"].shape[0] == n0 + len(bk["main"]) for v in mw.values()), "bakers_allpw main models missing from the window pass"
            fz2 = pd.DataFrame.from_dict(fz2, orient="index", columns=win["cols"]).reindex(studies)[bk["cols"]]
            zs["win_fused_bk"] = fz2.copy()
            final_z[bk["cols"]] = fz2.fillna(final_z[bk["cols"]])
        if em:                     # oct21_probe_eff_main: Effusion's fused base from the EF3 main slot; every study must have it (raise, no fallback)
            assert set(em["cols"]) <= set(win["cols"]) and all(v["s"].shape[0] == n1 + len(em["main"]) for v in mw.values()), "eff_main windows"
            fz3 = pd.DataFrame.from_dict(fz3, orient="index", columns=win["cols"]).reindex(studies)[em["cols"]]
            bad = fz3.index[~np.isfinite(fz3.to_numpy()).all(axis=1)]
            if len(bad):
                raise FloatingPointError(f"eff_main: {len(bad)} studies without a finite EF3-main fused Effusion: {list(bad)[:10]}")
            zs["win_fused_shipped_em"] = final_z[em["cols"]].copy()      # checks only (the shipped fused Effusion)
            final_z[em["cols"]] = fz3
            zs["win_fused_em"] = fz3.copy()
            print(f"eff_main: {em['cols']} fused with the EF3 main slot ({len(em['main'])} models) + {'the EF3 Basin windows ' + str({p: len(cs) for p, cs in emck.items()}) if emck else 'the shipped Basin windows'} "
                  f"for {len(studies)} / {len(studies)} studies", flush=True)
        STATS["win"] = {"wall_s": time.time() - t, "missing": len(miss), "bakers_allpw": int(fz2.notna().all(axis=1).sum()) if bk else 0}
    oa = cfg.get("oa")
    if oa:
        main_oa = zs["main"][oa_stage.OA]
        if oa["mode"] == "main":
            final_z[oa_stage.OA] = main_oa
        else:
            t = time.time()
            m = oa_stage.predict(root, studies, fs_of, pkg, oa_cfg(cfg), workers=workers, threads=threads)
            fused = m.pop("fused", None)
            fx = m.pop("fused_loa_extra", None)
            fxs = {k: m.pop(f"fused_{k}") for k in oa.get("extra_sets", {})}   # SWITCH PF Q20: extra fusions (never in the deployed mean)
            dep = {k: v for k, v in m.items() if k not in set(oa.get("loa_extra", []))}   # oct16_loa_bundle: extras only enter LOA's fusion
            e = (sum(dep.values()) / len(dep)).reindex(studies).astype(np.float64)
            ef = None
            if fused is not None:   # oct13_f5_win: crop-level fusion of the OA experts, aligned to e's OOF distribution, replaces e
                al = oa["fuse"]["align"]
                ef = pd.DataFrame({f: (fused[f] - al[f]["mu_f"]) / al[f]["sd_f"] * al[f]["sd_e"] + al[f]["mu_e"] for f in al}).reindex(studies)
                zs["oa_fused"] = ef.copy()
                e = ef[e.columns].fillna(e)
            miss = e.index[e.isna().any(axis=1)]
            if "recipe" in oa:      # oct11_oa_3pipe: pipelines' OA (final_z) blended with the experts' mean + de-coupling
                if len(miss):
                    print(f"oa: {len(miss)} studies without expert logits -> pipelines' OA in the recipe: {list(miss)[:10]}", flush=True)
                b_oa = final_z[oa_stage.OA].copy()
                final_z[oa_stage.OA] = oa_stage.apply_recipe(b_oa, e, oa["recipe"])
                if fx is not None:    # oct16_loa_bundle (10-16 12:03): Lateral OA from the deployed + extra members' fusion (own align,
                    # oa "loa_extra_align"); its de-coupling reads the deployed MOA / PF; the MOA / PF outputs (their de-couplings read
                    # LOA) and meniscus stay as deployed (output-level splice, as submissions/oct16_probe_loa_splice)
                    a = oa["loa_extra_align"]
                    ex = e.copy()
                    ex[LOA] = ((fx[LOA] - a["mu_f"]) / a["sd_f"] * a["sd_e"] + a["mu_e"]).reindex(studies).fillna(e[LOA])
                    zs["oa_fused_loa_extra"] = ex[[LOA]].copy()
                    final_z[LOA] = oa_stage.apply_recipe(b_oa, ex, oa["recipe"])[LOA]
                for k, fk in fxs.items():   # SWITCH PF Q20 (oct18_probe_pf_allup's splice): column extra_align[k]["col"] from its own fusion
                    a, C = oa["extra_align"][k], oa["extra_align"][k]["col"]
                    ex2 = e.copy()
                    ex2[C] = ((fk[C] - a["mu_f"]) / a["sd_f"] * a["sd_e"] + a["mu_e"]).reindex(studies).fillna(e[C])
                    zs[f"oa_fused_{k}"] = ex2[[C]].copy()
                    final_z[C] = oa_stage.apply_recipe(b_oa, ex2, oa["recipe"])[C]
            else:
                if len(miss):
                    print(f"oa: {len(miss)} studies without expert logits -> main pipeline OA: {list(miss)[:10]}", flush=True)
                final_z[oa_stage.OA] = e[oa_stage.OA].fillna(main_oa)
            men = cfg.get("men")
            if men:                 # oct12_men_3pipe: the same experts' meniscus heads (no extra model pass) -> meniscus recipe
                em = (sum(m[k] for k in men["experts"]) / len(men["experts"])).reindex(studies).astype(np.float64)[oa_stage.MEN]
                if ef is not None:
                    em = ef[oa_stage.MEN].fillna(em)
                miss = em.index[em.isna().any(axis=1)]
                if len(miss):
                    print(f"men: {len(miss)} studies without expert logits -> pipelines' meniscus: {list(miss)[:10]}", flush=True)
                final_z[oa_stage.MEN] = oa_stage.apply_recipe(final_z[oa_stage.MEN], em, men)
            zs["oa_expert"] = final_z.copy()
            zs |= {f"oa_{k}": v.reindex(studies).astype(np.float64) for k, v in m.items()}   # each expert's own OA logits (checks only)
            STATS["oa"] = {"wall_s": time.time() - t, **{k: v for k, v in oa_stage.STATS.items() if k != "failed"},
                           "failed": len(oa_stage.STATS.get("failed", []))}
    fc = cfg.get("fc")
    if fc:
        torch.cuda.empty_cache()
        t = time.time()
        m = {k: v.reindex(studies).astype(np.float64) for k, v in fc_stage.predict_pkg(root, studies, fs_of, pkg, fc_cfg(cfg), workers=workers).items()}
        miss = [s for s in studies if any(np.isnan(v.loc[s]).any() for v in m.values())]
        if miss:
            print(f"fc: {len(miss)} studies without expert logits -> pipelines' F / C: {miss[:10]}", flush=True)
        fc_base = final_z[fc_stage.FC].copy()
        if cfg.get("fc_shipped_off"):   # oct21_final_alldata (Master: dead shipped bone members dropped): Contusion = cont_allmem and Fracture =
            assert cfg.get("cont_allmem") and cfg.get("frac_allmem") and "fused" not in m and \
                not any(k.startswith("runs/b_") for k in fc["members"]), "fc_shipped_off needs cont_allmem + frac_allmem and no shipped members"
        else:                           # frac_allmem below set F / C from fc_base + their own groups; the shipped recipe / default fusion is skipped
            final_z[fc_stage.FC] = fc_stage.apply_recipe(fc_base, m, fc["recipe"])
            if fc.get("fuse"):      # oct13_f5_win: crop-level fusion of all bone members replaces the recipe's member z-mean
                zf = fc_stage.apply_recipe_fused(fc_base, m["fused"], fc["recipe"], fc["fuse"])
                zs["fc_member_recipe"] = final_z[fc_stage.FC].copy()           # checks only
                final_z[fc_stage.FC] = zf.fillna(final_z[fc_stage.FC])
        cf = cfg.get("cont_fs")    # oct17_probe_cont_fs (LB-won): Contusion = (1 - w) x package logit + w x z(FS-only bone member); label-free align
        if cf:
            zc = m[cf["name"]]["Contusion"]
            zs["cont_fs"] = zc.to_frame("Contusion")
            zl = (zc - cf["mu_m"]) / cf["sd_m"] * cf["sd_d"] + cf["mu_d"]
            final_z["Contusion"] = ((1 - cf["w"]) * final_z["Contusion"] + cf["w"] * zl).fillna(final_z["Contusion"])   # no member logit -> deployed
            print(f"cont_fs: {cf['name']} spliced into Contusion at w {cf['w']} ({int(zc.notna().sum())} / {len(zc)} studies)", flush=True)
        ca = cfg.get("cont_allmem")   # oct20_probe_cont_allmem: CONTUSION-ONLY splice of the new-label bone members, with the SHIPPED constants:
        if ca:                        # their crop fusion ('fuse_set' cont) in the fc recipe + their FS member in the cont_fs splice; no fallback
            cf = cfg["cont_fs"]
            zf2 = fc_stage.apply_recipe_fused(fc_base, m[fc_stage.fused_key("cont")], fc["recipe"], fc["fuse"])["Contusion"]
            zc2 = m[ca["fs_member"]]["Contusion"]
            bad = [s for s in studies if not (np.isfinite(zf2.get(s, np.nan)) and np.isfinite(zc2.get(s, np.nan)))]
            if bad:
                raise FloatingPointError(f"cont_allmem: {len(bad)} studies without a finite new-member Contusion: {bad[:10]}")
            zs["cont_allmem_shipped"] = final_z[["Contusion"]].copy()   # checks only (the v2 Contusion)
            final_z["Contusion"] = (1 - cf["w"]) * zf2 + cf["w"] * ((zc2 - cf["mu_m"]) / cf["sd_m"] * cf["sd_d"] + cf["mu_d"])
            zs["cont_allmem"] = final_z[["Contusion"]].copy()
            print(f"cont_allmem: Contusion from the new-label members ({len(studies)} / {len(studies)} studies; fs member {ca['fs_member']})", flush=True)
        fa = cfg.get("frac_allmem")   # oct21_probe_frac_bsif: FRACTURE-ONLY swap of the bone crop fusion ('fuse_set' frac = the SIF-target members)
        if fa:                        # in the fc recipe with the SHIPPED constants (fuse align, recipe stats, w); no fallback
            zf3 = fc_stage.apply_recipe_fused(fc_base, m[fc_stage.fused_key("frac")], fc["recipe"], fc["fuse"])["Fracture"]
            bad = [s for s in studies if not np.isfinite(zf3.get(s, np.nan))]
            if bad:
                raise FloatingPointError(f"frac_allmem: {len(bad)} studies without a finite new-member Fracture: {bad[:10]}")
            zs["frac_allmem_shipped"] = final_z[["Fracture"]].copy()   # checks only (the v2 Fracture)
            final_z["Fracture"] = zf3
            zs["frac_allmem"] = final_z[["Fracture"]].copy()
            print(f"frac_allmem: Fracture from the new bone members ({len(studies)} / {len(studies)} studies; form {fa['form']})", flush=True)
        ff = cfg.get("frac_fs")       # oct21_probe_frac_bsif form v3c_fs: the FS bone member's Fracture logit SPLICED into Fracture (the cont_fs
        if ff:                        # form; label-free constants + w from fit_frac_fs.py); after frac_allmem; no fallback
            zb = m[ff["name"]]["Fracture"]
            bad = [s for s in studies if not np.isfinite(zb.get(s, np.nan))]
            if bad:
                raise FloatingPointError(f"frac_fs: {len(bad)} studies without a finite FS-member Fracture logit: {bad[:10]}")
            zs["frac_fs"] = zb.to_frame("Fracture")
            final_z["Fracture"] = (1 - ff["w"]) * final_z["Fracture"] + ff["w"] * ((zb - ff["mu_m"]) / ff["sd_m"] * ff["sd_d"] + ff["mu_d"])
            print(f"frac_fs: {ff['name']} spliced into Fracture at w {ff['w']} ({len(studies)} / {len(studies)} studies)", flush=True)
        fm = cfg.get("frac_mfs")   # SWITCH oct19_probe_frac_mfs: Fracture = (1 - w) x package logit + w x z(m_fs Fracture logit); label-free align
        if fm:
            zf_ = m[fm["name"]]["Fracture"]
            zs["frac_mfs"] = zf_.to_frame("Fracture")
            zl = (zf_ - fm["mu_m"]) / fm["sd_m"] * fm["sd_d"] + fm["mu_d"]
            final_z["Fracture"] = ((1 - fm["w"]) * final_z["Fracture"] + fm["w"] * zl).fillna(final_z["Fracture"])   # no member logit -> deployed
            print(f"frac_mfs: {fm['name']} spliced into Fracture at w {fm['w']} ({int(zf_.notna().sum())} / {len(zf_)} studies)", flush=True)
        zs |= {f"fc_{k}": v for k, v in m.items()}   # each member's own F / C logits (checks only)
        STATS["fc"] = {"wall_s": time.time() - t, **{k: v for k, v in fc_stage.STATS.items() if k != "failed"},
                       "failed": len(fc_stage.STATS.get("failed", []))}
    syn, synm, zih_parts = cfg.get("syn"), {}, dict(zsyn)
    zsyn_final = sum(zsyn.values()) / len(zsyn)
    if syn:
        torch.cuda.empty_cache()   # the crop stage loads 18 models next to knee_infer's CUDA-graph pools (16 GB T4)
        t = time.time()
        members = syn_members(cfg, pkg)
        zm = pd.DataFrame(syn_stage.predict(root, studies, fs_of, members, str(Path(pkg) / syn["loc"]), str(Path(pkg) / syn["detector"]),
                                            workers=workers, fuse=syn.get("fuse"))).reindex(studies).astype(np.float64)
        ze = sum(zm[names].mean(1) for names in syn["pipelines"].values()) / len(syn["pipelines"])
        if syn.get("fuse"):     # oct13_f5_win: crop-level fusion per design (both backbones), mean over designs, replaces the crop part
            zf = zm[[f"fused@{d}" for d in syn["fuse"]["designs"]]].mean(1)
            zs["syn_fused"] = zf.to_frame("Synovitis")
            ze = zf.fillna(ze)
        miss = ze.index[ze.isna()]
        if len(miss):
            print(f"syn: {len(miss)} studies without crop-expert logits -> image-head Synovitis: {list(miss)[:10]}", flush=True)
        w = syn.get("image_heads_w", 0.0)
        zsyn = {"image_heads": zsyn_final, "crop_experts": ze.fillna(zsyn_final)}
        zsyn_final = w * zsyn["image_heads"] + (1 - w) * zsyn["crop_experts"]
        eff = cfg.get("eff")
        if eff:                 # oct13_f5_eff: Effusion = (1 - w) b + w z(e), e = the crop experts' image-read Effusion (same pass)
            effp = syn.get("eff_pipelines", syn["pipelines"])   # oct16_syn_cnx_probe: Effusion keeps the effb0 crop experts
            ee = sum(zm[[f"{n}@img_eff" for n in names]].mean(1) for names in effp.values()) / len(effp)
            miss = ee.index[ee.isna()]
            if len(miss):
                print(f"eff: {len(miss)} studies without crop-expert logits -> pipelines' Effusion: {list(miss)[:10]}", flush=True)
            zs["eff_expert"] = ee.to_frame("Effusion")
            final_z[["Effusion"]] = oa_stage.apply_recipe(final_z[["Effusion"]], ee.to_frame("Effusion"), eff)
        mm = cfg.get("mcl_member")   # oct17_probe_mcl_zoom: MCL = (1 - w) x package logit + w x z(member MCL logit); label-free align
        if mm:
            zmc = zm[f"{mm['name']}@mcl"]
            zs["mcl_member"] = zmc.to_frame("MCL")
            zl = (zmc - mm["mu_m"]) / mm["sd_m"] * mm["sd_d"] + mm["mu_d"]
            final_z["MCL"] = ((1 - mm["w"]) * final_z["MCL"] + mm["w"] * zl).fillna(final_z["MCL"])   # no member logit -> deployed MCL
            print(f"mcl_member: {mm['name']} spliced into MCL at w {mm['w']} ({int(zmc.notna().sum())} / {len(zmc)} studies)", flush=True)
        m2 = cfg.get("mcl_member2")  # oct18_probe_mcl_stack: MCL'' = (1 - w) x MCL' (the SD-2 + zoom form above) + w x z(b3g area MCL); z aligned to MCL' OOF
        if m2:
            zm2 = zm[f"{m2['name']}@mcl"]
            zs["mcl_member2"] = zm2.to_frame("MCL")
            zl2 = (zm2 - m2["mu_m"]) / m2["sd_m"] * m2["sd_d"] + m2["mu_d"]
            final_z["MCL"] = ((1 - m2["w"]) * final_z["MCL"] + m2["w"] * zl2).fillna(final_z["MCL"])   # no member logit -> the SD-2 + zoom form
            print(f"mcl_member2: {m2['name']} spliced into MCL at w {m2['w']} ({int(zm2.notna().sum())} / {len(zm2)} studies)", flush=True)
        am = cfg.get("acl_member")   # oct18_probe_acl_b3g (Q16): ACL = (1 - w) x package logit + w x z(member ACL logit); label-free align
        if am:
            zac = zm[f"{am['name']}@acl"]
            zs["acl_member"] = zac.to_frame("ACL")
            zla = (zac - am["mu_m"]) / am["sd_m"] * am["sd_d"] + am["mu_d"]
            final_z["ACL"] = ((1 - am["w"]) * final_z["ACL"] + am["w"] * zla).fillna(final_z["ACL"])   # no member logit -> deployed ACL
            print(f"acl_member: {am['name']} spliced into ACL at w {am['w']} ({int(zac.notna().sum())} / {len(zac)} studies)", flush=True)
        a2 = cfg.get("acl_member2")  # oct18_probe_acl_b3g (Q16): ACL'' = (1 - w) x ACL' (the Z1 form above) + w x z(b3g area ACL); z aligned to ACL' OOF
        if a2:
            za2 = zm[f"{a2['name']}@acl"]
            zs["acl_member2"] = za2.to_frame("ACL")
            zla2 = (za2 - a2["mu_m"]) / a2["sd_m"] * a2["sd_d"] + a2["mu_d"]
            final_z["ACL"] = ((1 - a2["w"]) * final_z["ACL"] + a2["w"] * zla2).fillna(final_z["ACL"])   # no member logit -> the Z1 form
            print(f"acl_member2: {a2['name']} spliced into ACL at w {a2['w']} ({int(za2.notna().sum())} / {len(za2)} studies)", flush=True)
        cs = cfg.get("cont_sxm")     # oct21_probe_cont_sxm: Contusion' = (1 - w) x the all-4 Contusion (cont_allmem above) + w x z(sxm img_cont);
        if cs:                       # label-free align (pooled f0-2 non-gold OOF); fail fast: every study needs a finite sxm logit (no fallback)
            assert cfg.get("cont_allmem", {}).get("form") == "all", "cont_sxm splices onto the all-4 Contusion"
            zx = zm[f"{cs['name']}@img_cont"]
            bad = [s for s in studies if not np.isfinite(zx.get(s, np.nan))]
            if bad:
                raise FloatingPointError(f"cont_sxm: {len(bad)} studies without a finite {cs['name']} img_cont logit: {bad[:10]}")
            zs["cont_sxm_all4"] = final_z[["Contusion"]].copy()   # checks only (the all-4 Contusion)
            final_z["Contusion"] = (1 - cs["w"]) * final_z["Contusion"] + cs["w"] * ((zx - cs["mu_m"]) / cs["sd_m"] * cs["sd_d"] + cs["mu_d"])
            zs["cont_sxm"] = final_z[["Contusion"]].copy()
            print(f"cont_sxm: {cs['name']} img_cont spliced into Contusion at w {cs['w']} ({len(studies)} / {len(studies)} studies)", flush=True)
        synm = {f"synm_{k}": 1 / (1 + np.exp(-v)) for k, v in zm.items()}            # each expert fold model (checks only)
        STATS["syn"] = {"wall_s": time.time() - t, "flipped": syn_stage.STATS.get("flipped"), "failed": len(syn_stage.STATS.get("failed", []))}
    final = 1 / (1 + np.exp(-final_z))
    final["Synovitis"] = 1 / (1 + np.exp(-zsyn_final))
    parts = {k: 1 / (1 + np.exp(-v)) for k, v in (zs | zs_sd).items()} | {f"syn_{k}": 1 / (1 + np.exp(-v)) for k, v in zsyn.items()} | \
            {f"syn_ih_{k}": 1 / (1 + np.exp(-v)) for k, v in zih_parts.items()} | \
            {c: 1 / (1 + np.exp(-z[str(Path(pkg) / c)])) for cs in list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) for c in cs} | synm
    return final, parts, failed, secs
