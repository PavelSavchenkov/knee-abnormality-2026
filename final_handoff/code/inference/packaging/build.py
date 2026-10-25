"""FINAL all-data package (USER 10-21 13:5x via Master: train the best fold-avg pipeline per class on 100 % of the data, several seeds;
OAI: no; offline build, nothing uploaded). package/ = ONE flat, self-contained dataset = the runtime work config of the final form (what the
stack kernels build in WORK) with every member's fold models replaced by its recipe's all-data runs (fold -1, the fold runs' step count via
spe_ref; several seeds), materialised as real files. Recipe constants unchanged (OA / men / fc / eff recipes, fuse / align stats, the splice
w / mu / sd, DEC) - as oct15_all_data_s5 did; the member-side constants (mu_m / sd_m) were fitted on fold-model OOF logits and all-data
members are scored out of sample at test time too (accepted as in s5; README).
FORM: v2 (57416173) + the LB winners Contusion all-4 (57455847) and the EF3 img_eff crop slots (57455943); switches for the pending probes
(default OFF): --frac (Fracture v3c_fs 57475191), --effc (Effusion form c 57479153), --sxm (Contusion sxm 57480087), --fc-off (with --frac:
drop the then-dead b_v3 / b_lr3 / b_nf and skip the shipped fc fusion + recipe; ensemble.py fc_shipped_off).
MERGE (kit work configs via oct20_stack_all_v2/run_local.build_work; each kit's diff vs v2 is asserted to touch only its own keys):
all-4 = v2 + cont_allmem; + EF3 (syn: -s_fs / -s_fc125 effb0, +ef3 / ef3fc / ef3nf / ef3nffc; eff_pipelines + Effusion stats);
--frac: the same new bone runs in both groups (fuse_sets ['cont', 'frac']) + frac_allmem + frac_fs; --effc: eff_main; --sxm: cont_sxm.
DEAD (code proofs in README): s_fs / s_fc125 effb0 (EF3 deletes them); bfs_fs (cont_allmem overwrites the cont_fs splice -> cont_fs.name
is pointed at bfs_rep, whose result cont_allmem overwrites with the same constants; bfs_fs dropped).
ALL-DATA FILES: SEEDS = {recipe: [seed labels]} (None = every finished run); reused recipes = experiments/v2/oct14_all_data runs
A<k>_<name>/fold-1 (labels A1..A5), new recipes = experiments/v3/oct16_experts runs <name>_all_s<seed>/fold-1; checkpoint = the LAST saved
file (main / mainp model.pt, Basin the highest EMA model_e*.pt, oa / bone / syn the highest model_ep*.pt); the SD-2 MCL members strip their
training-only sd_* keys exactly as oct18_probe_mcl_sd2/build.py. Every new file is strict-loaded by the package's own classes.
conda run -n rsna python submissions/oct21_final_alldata/build.py --variant NAME [--dry]   (VARIANTS: form + seeds; package_<NAME>/ FLAT)"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
REPO = SUB.parent
EXP = REPO / "experiments/v3/oct16_experts"
A14 = REPO / "experiments/v2/oct14_all_data/runs"
sys.path.insert(0, str(SUB / "oct20_stack_all_v2"))
import run_local as R  # noqa: E402

R.LOCAL |= {"knee-oct20-cont-allmem": SUB / "oct20_probe_cont_allmem/dataset_all", "knee-oct20-eff-ef3": SUB / "oct20_probe_eff_ef3_v2/dataset",
            "knee-oct21-frac-bsif-v3c-fs": SUB / "oct21_probe_frac_bsif/dataset_v3c_fs", "knee-oct21-eff-c": SUB / "oct21_probe_eff_c/dataset",
            "knee-oct21-cont-sxm": SUB / "oct21_probe_cont_sxm/dataset"}
KITS = {"v2": SUB / "oct20_stack_all_v2/kernel", "all4": SUB / "oct20_probe_cont_allmem/kernel_all_short", "ef3": SUB / "oct20_probe_eff_ef3_v2/kernel_short",
        "frac": SUB / "oct21_probe_frac_bsif/kernel_final_v3c_fs", "effc": SUB / "oct21_probe_eff_c/kernel", "sxm": SUB / "oct21_probe_cont_sxm/kernel"}
TOUCH = {"all4": ("cont_allmem", "fc.members"), "ef3": ("eff.pipelines", "eff.stats.Effusion", "syn.eff_pipelines"),
         "frac": ("fc.members", "frac_allmem", "frac_fs"), "effc": ("eff.pipelines", "eff.stats.Effusion", "syn.eff_pipelines", "eff_main"),
         "sxm": ("cont_allmem", "fc.members", "cont_sxm")}
# recipe -> (kind, source, name): source 'a14' = oct14 A<k>_<name>, 'exp' = oct16_experts <name>_all_s<seed>
RECIPES = {"sub_b0": ("main", "a14", "sub_b0"), "m_prob05": ("main", "a14", "m_prob05"), "nfnet": ("basin", "a14", "nfnet"),
           "r50": ("basin", "a14", "r50"), "n_prob05": ("basin", "a14", "n_prob05"), "oa_v2cor": ("oa", "a14", "oa_v2cor"),
           "oa_v2cor_nf": ("oa", "a14", "oa_v2cor_nf"), "b_v3": ("bone", "a14", "b_v3"), "b_lr3": ("bone", "a14", "b_lr3"),
           "b_nf": ("bone", "a14", "b_nf"), "s_fs_nf": ("syn", "a14", "s_fs_nf"), "s_fc125_nf": ("syn", "a14", "s_fc125_nf"),
           "sd1b": ("mainp", "exp", "sd1b"), "bsd_nf": ("basin", "exp", "bsd_nf"), "oasep": ("oa", "exp", "oasep"),
           "oasep_nf": ("oa", "exp", "oasep_nf"), "silsep": ("oa", "exp", "silsep"), "b3c_rep": ("bone", "exp", "b3c_rep"),
           "b3l_rep": ("bone", "exp", "b3l_rep"), "b3n_rep": ("bone", "exp", "b3n_rep"), "bfs_rep": ("bone", "exp", "bfs_rep"),
           "w5b_mcl": ("syn", "exp", "w5b_mcl"), "z1_acl": ("syn", "exp", "z1_acl"), "b3g": ("syn", "exp", "b3g"),
           "cs_fs": ("syn", "exp", "cs_fs"), "cs_fc125": ("syn", "exp", "cs_fc125"), "ef3": ("syn", "exp", "ef3"),
           "ef3fc": ("syn", "exp", "ef3fc"), "ef3nf": ("syn", "exp", "ef3nf"), "ef3nffc": ("syn", "exp", "ef3nffc"),
           "sxm": ("syn", "exp", "sxm"), "ef3m": ("main", "exp", "ef3m"), "e3bn": ("basin", "exp", "e3bn")}
SRC = {"cs_fs": "refs/syn/x_cnx5e-5/fold0", "cs_fc125": "refs/syn/x_cnx_fc125/fold0"}   # the queue's --src per new recipe (default runs/<name>/fold0)
SRC |= {r: f"runs/{r}/fold0" for r, v in RECIPES.items() if v[1] == "exp" and r not in SRC}
# training-only heads the SHIPPED fold files dropped at export (checked key-for-key against SHIP_REF at build time):
# 'sd' = the series-decoupling heads (oct18_probe_mcl_sd2 main / Basin, oct18_probe_loa_allup OA groups: first key component sd_*);
# 'knee' = b3g's knee-context branch (oct18_probe_acl_b3g KNEE_KEYS; the shipped b3g cfg runs the area path, knee_ctx off)
STRIP = {"sd1b": "sd", "bsd_nf": "sd", "oasep": "sd", "oasep_nf": "sd", "silsep": "sd", "b3g": "knee"}
KNEE_KEYS = ("kn_v.", "kn_w.", "kn_cls_w", "kn_cls_b", "fin_w", "fin_b")
SHIP_REF = {"sub_b0": "all4/sub_b0_fold0_model.pt", "m_prob05": "all4/m_prob05_fold0_model.pt", "nfnet": "all4/basin_nfnet_0.pt",
            "r50": "all4/basin_rad_0.pt", "n_prob05": "all4/basinsyn_nfnet_0.pt", "oa_v2cor": "all4/oa_oa_crop_v2cor_f0.pt",
            "oa_v2cor_nf": "all4/oa_oa_crop_v2cor_nfnet_f0.pt", "b_v3": "all4/fc_b_v3_f0.pt", "b_lr3": "all4/fc_b_lr3_f0.pt", "b_nf": "all4/fc_b_nf_f0.pt",
            "s_fs_nf": "all4/syn_s_fs_nf_f0.pt", "s_fc125_nf": "all4/syn_s_fc125_nf_f0.pt", "sd1b": "all4/main_sd/main_sd_f0.pt",
            "bsd_nf": "all4/basin_nfnet_sd_0.pt", "oasep": "all4/oa_oasep_f0.pt", "oasep_nf": "all4/oa_oasep_nf_f0.pt", "silsep": "all4/oa_silsep_f0.pt",
            "b3c_rep": "all4/fc_b3c_rep_f0.pt", "b3l_rep": "all4/fc_b3l_rep_f0.pt", "b3n_rep": "all4/fc_b3n_rep_f0.pt", "bfs_rep": "all4/fc_bfs_rep_f0.pt",
            "w5b_mcl": "all4/syn_w5b_mcl_f0.pt", "z1_acl": "all4/syn_z1_acl_f0.pt", "b3g": "all4/syn_b3g_f0.pt", "cs_fs": "all4/syn_cs_fs_f0.pt",
            "cs_fc125": "all4/syn_cs_fc125_f0.pt", "ef3": "ef3/syn_ef3_f0.pt", "ef3fc": "ef3/syn_ef3fc_f0.pt", "ef3nf": "ef3/syn_ef3nf_f0.pt",
            "ef3nffc": "ef3/syn_ef3nffc_f0.pt", "sxm": "sxm/sxm_f0.pt", "ef3m": "effc/ef3m_fold0_model.pt", "e3bn": "effc/basin_e3bn_0.pt"}
assert set(SHIP_REF) == set(RECIPES) and set(STRIP) <= set(RECIPES)


def strip(sd, how):
    if how == "sd":
        return {k: v for k, v in sd.items() if not k.split(".")[0].startswith("sd_")}
    assert how == "knee", how
    return {k: v for k, v in sd.items() if not k.startswith(KNEE_KEYS)}


def ship_keys(recipe, _c={}):
    """the key set of the SHIPPED fold file of this recipe (the kits' work dirs: what the LB-scored packages loaded)."""
    if recipe not in _c:
        import torch
        _c[recipe] = set(torch.load(HERE / "_work" / SHIP_REF[recipe], map_location="cpu"))
    return _c[recipe]


# file name = f"{STEM[recipe]}_all_{seed label}.pt", the same in every variant (the store / datasets hold each file once)
STEM = {"sub_b0": "sub_b0", "m_prob05": "m_prob05", "nfnet": "basin_nfnet", "r50": "basin_rad", "n_prob05": "basinsyn_nfnet", "sd1b": "main_sd",
        "bsd_nf": "basin_nfnet_sd", "oa_v2cor": "oa_oa_crop_v2cor", "oa_v2cor_nf": "oa_oa_crop_v2cor_nfnet", "oasep": "oa_oa_oasep",
        "oasep_nf": "oa_oa_oasep_nf", "silsep": "oa_oa_silsep", "b_v3": "fc_b_v3", "b_lr3": "fc_b_lr3", "b_nf": "fc_b_nf", "b3c_rep": "fc_b3c_rep",
        "b3l_rep": "fc_b3l_rep", "b3n_rep": "fc_b3n_rep", "bfs_rep": "fc_bfs_rep", "sxm": "syn_sxm", "ef3m": "ef3m", "e3bn": "basin_e3bn"}
STEM |= {r: f"syn_{r}" for r, v in RECIPES.items() if v[0] == "syn" and r not in STEM}
assert set(STEM) == set(RECIPES)
SEEDS = {r: None for r in RECIPES}       # set from the chosen VARIANT in main(); None = every finished run

# oct22 COLUMN ROUTING (Master 07:3x UTC, USER's 5-probe plan): per (recipe, column) the plain seeds, their PL twins or both. Variant JSON keys
# "pl" {recipe: [P labels]}, "routes" {recipe: {column: plain | pl | both}} (unlisted = plain), "window" plain | both (the Effusion / Baker's
# window passes take the routed generalists' plain + PL files), "code": "route" (the routing code overlay without routes: switch-off proof).
# Routed variants ship pkg_code_route/{ensemble,knee_infer}.py; every other variant keeps the proven code. Moot / unsupported pairs are refused.
ROUTE = {"pl": {}, "routes": {}, "window": "plain", "code": ""}
OA_RCOLS = {"Medial OA": "moa", "PF OA": "pf"}              # OA expert columns routed through an extra_sets fusion (key)
MEN_COLS = ("Medial Meniscus", "Lateral Meniscus")           # OA expert meniscus heads: ONE men["experts"] list (MM == LM per recipe)
GEN_COLS = {"ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Fracture"}   # not Effusion / Baker's (window),
ROUTABLE = {"oa_v2cor": set(OA_RCOLS) | set(MEN_COLS), "oa_v2cor_nf": set(OA_RCOLS) | set(MEN_COLS),     # Contusion (cont_allmem), MCL (mcl_sd:
            "sub_b0": GEN_COLS, "m_prob05": {"Synovitis"}, "nfnet": GEN_COLS, "r50": GEN_COLS | {"MCL"}}   # main SD + NFNet SD + R50), Synovitis
VARIANT_KEYS = {"form", "seeds", "cap", "pl", "routes", "window", "code"}                                     # (image-head models)


def check_routes(d, runs=False):
    """The routed-variant rules -> list of problems (empty = OK). runs: also every plain / PL label must be a finished run."""
    p, rt, pl, win = [], d.get("routes", {}), d.get("pl", {}), d.get("window", "plain")
    if set(d) - VARIANT_KEYS:
        p.append(f"unknown keys {sorted(set(d) - VARIANT_KEYS)}")
    if win not in ("plain", "both") or d.get("code", "") not in ("", "route"):
        p.append("window must be plain / both; code '' / route")
    for r, cols in rt.items():
        if r not in ROUTABLE:
            p.append(f"{r}: not routable (routable: {sorted(ROUTABLE)})")
            continue
        if set(cols) - ROUTABLE[r]:
            p.append(f"{r}: columns not routable {sorted(set(cols) - ROUTABLE[r])} (routable: {sorted(ROUTABLE[r])})")
        if set(cols.values()) - {"plain", "pl", "both"}:
            p.append(f"{r}: routes must be plain / pl / both")
        if not pl.get(r) or not all(is_pl(x) for x in pl[r]):
            p.append(f"{r}: routed without 'pl' P<seed> labels")
        if not d.get("seeds", {}).get(r):
            p.append(f"{r}: routed without plain 'seeds'")
        elif "both" in cols.values() and len(pl.get(r, [])) != len(d["seeds"][r]):
            p.append(f"{r}: 'both' needs equal plain / PL counts ({len(d['seeds'][r])} vs {len(pl.get(r, []))})")
        if r in ("oa_v2cor", "oa_v2cor_nf") and cols.get(MEN_COLS[0], "plain") != cols.get(MEN_COLS[1], "plain"):
            p.append(f"{r}: Medial and Lateral Meniscus must route alike (one men experts list)")
    if set(pl) - set(rt):
        p.append(f"'pl' for unrouted recipes {sorted(set(pl) - set(rt))}")
    if ("sub_b0" in rt) != ("m_prob05" in rt):
        p.append("sub_b0 and m_prob05: route both or neither (the PL main set needs both model kinds)")
    if win == "both" and not {"sub_b0", "nfnet", "r50"} <= set(rt):
        p.append("window both needs sub_b0, nfnet and r50 routed (their PL files join the windows)")
    if d.get("form", {}).get("sxm") and rt:
        p.append("routes with sxm on: untested combination")
    if runs:                                             # routed recipes' plain + PL labels (the build resolves every used recipe's seeds itself)
        for r, labs in [(r, d.get("seeds", {}).get(r, [])) for r in rt] + list(pl.items()):
            for x in labs:
                if not (run_dir(r, x) / "done.txt").exists():
                    p.append(f"{r}:{x} not finished")
    return p

# VARIANT = one dataset + one kernel: the form (switches) + the seeds per recipe (Master fills after the allocation / the LB reads), e.g.
# "n3_frac": {"form": {"frac": True, "effc": True, "sxm": True, "fc_off": True}, "seeds": {"b3g": [101, 102, 103], "sub_b0": ["A1", "A2", "A3"]}}
VARIANTS = {"draft": {"form": {"frac": False, "effc": False, "sxm": False, "fc_off": False}, "seeds": {}},
            # end-to-end TEST of build -> audit -> make_kernel on the runs finished so far (Master 14:3x; never uploaded): the frac + fc_off
            # code paths (their runs = the Contusion all-4 runs); effc / sxm wait for their runs (ef3m / e3bn / sxm not queued yet)
            "partial": {"form": {"frac": True, "effc": False, "sxm": False, "fc_off": True}, "seeds": {}}}
# USER 10-21 19:2x (via Master): the FINAL seed plan ("ok, then train only these models"; dropped seeds <= 5e-5 LB worst case)
PLAN = {"bfs_rep": 7, "oa_v2cor": 7, "oa_v2cor_nf": 7, "sub_b0": 7, "w5b_mcl": 7, "z1_acl": 7, "sd1b": 6, "m_prob05": 5, "nfnet": 5, "r50": 5, "b3g": 4}


def load_json_variants():
    """variants/<name>.json = {"form": {...}, "seeds": {recipe: [labels]}, ["cap": bool]} (Master / 8b define PL / mixed sets without code)."""
    for f in sorted((HERE / "variants").glob("*.json")):
        d = json.load(open(f))
        assert f.stem not in VARIANTS, f"variant {f.stem} defined twice"
        assert set(d["form"]) == {"frac", "effc", "sxm", "fc_off"} and set(d["seeds"]) <= set(RECIPES), f.name
        assert set(d) <= VARIANT_KEYS, (f.name, sorted(set(d) - VARIANT_KEYS))
        VARIANTS[f.stem] = {k: d[k] for k in VARIANT_KEYS if k in d}


def plan_seeds(r, n):
    """the plan's seed labels: oct14 recipes A1..A5 then A104, A105, ...; new recipes 101, 102, ..."""
    if RECIPES[r][1] == "a14":                       # the oct14 base seeds A1..A<m> (m = 5 or 3, as trained 10-14), then the extra seeds A104, ...
        m = sum((A14 / f"A{k}_{RECIPES[r][2]}" / "fold-1/done.txt").exists() for k in range(1, 6))
        assert all((A14 / f"A{k}_{RECIPES[r][2]}" / "fold-1/done.txt").exists() for k in range(1, m + 1)), r
        return [f"A{k}" for k in range(1, min(n, m) + 1)] + [f"A{104 + k}" for k in range(max(0, n - m))]
    return list(range(101, 101 + n))


FINAL_FORM = {"frac": True, "effc": False, "sxm": True, "fc_off": True}       # FRAC_V3C won, eff_c lost, fc_off identity 0.0, sxm ON (Master 21:0x:
SAFE3_FORM = {"frac": True, "effc": False, "sxm": False, "fc_off": True}       # 57480087 above all-4); safe3 was built + pushed sxm-OFF (frozen form)
VARIANTS |= {
    # tonight's slots: every FROZEN seed, capped at the plan (a seed frozen beyond the plan is dropped); b3c_rep pinned (s104 ran before the cut)
    "final_a": {"form": FINAL_FORM, "seeds": {"b3c_rep": [101, 102, 103]}, "cap": True},
    # the final: exactly the plan (every other recipe 3 seeds, incl. sxm if it wins); refuses until every planned seed is frozen
    # the GATED 118 (tonight's slot A: cfg-final + kernel_final v1, uploaded / pushed / both gates PASS) - kept for the record, tag 'final'
    "final118": {"form": FINAL_FORM, "tag": "final", "seeds": {r: plan_seeds(r, PLAN.get(r, 3)) for r in RECIPES} | {"b3c_rep": [101, 102, 103]}},
    # Master 22:4x: 'final' = TOMORROW's 123 plan (USER tolerance 5e-5 incl. sxm in the worst-case table: sxm 6, b3g 5, sd1b 7) - 8b's PL
    # twin pairing reads the k-th seed of VARIANTS['final']; artifacts tagged 'final123' so the gated 118 build (tag 'final') is never touched
    "final123": {"form": FINAL_FORM, "tag": "final123", "seeds": {r: plan_seeds(r, PLAN.get(r, 3)) for r in RECIPES} | {"b3c_rep": [101, 102, 103]}
                 | {"sxm": plan_seeds("sxm", 6), "b3g": plan_seeds("b3g", 5), "sd1b": plan_seeds("sd1b", 7)}},
    # USER 10-22 07:5x via Master (Investigate #1: "urgently train more seeds for this (up to 9)"): w5b_mcl is high-variance (inter-seed
    # gold corr 0.84 vs inter-fold 0.92) -> final123 with w5b_mcl 9 seeds (101-109) = 125 models; final123 stays built + gated
    "final125": {"form": FINAL_FORM, "tag": "final125", "seeds": {r: plan_seeds(r, PLAN.get(r, 3)) for r in RECIPES} | {"b3c_rep": [101, 102, 103]}
                 | {"sxm": plan_seeds("sxm", 6), "b3g": plan_seeds("b3g", 5), "sd1b": plan_seeds("sd1b", 7), "w5b_mcl": plan_seeds("w5b_mcl", 9)}},
    # slot A FALLBACK (Master 21:0x): the 115-model plan without sxm, if the sxm seeds miss the d2 freeze
    "final115": {"form": SAFE3_FORM, "seeds": {r: plan_seeds(r, PLAN.get(r, 3)) for r in RECIPES} | {"b3c_rep": [101, 102, 103]}},
    # slot B (the user's idea via Submission Timing, Master 21:0x): exactly 3 seeds of every recipe (oct14 A1-A3, new 101-103) = 84 models
    "safe3": {"form": SAFE3_FORM, "seeds": {r: plan_seeds(r, 3) for r in RECIPES}},
    # Master 10-22 08:0x: final118 timed out (> 540 min; both anomalous runs had cont_sxm) -> NO sxm; safe3 (scored 0.965 in 190 min)
    # + w5b_mcl at 9 seeds (USER: high-variance recipe, up to 9) = 87 models; PL off
    "safe3w9": {"form": SAFE3_FORM, "seeds": {r: plan_seeds(r, 3) for r in RECIPES} | {"w5b_mcl": plan_seeds("w5b_mcl", 9)}},
}
VARIANTS["final"] = VARIANTS["final125"]                 # 10-22 07:5x: 'final' IS the 125 plan (artifacts tag 'final125'); was final123
load_json_variants()


def tag(v):
    """artifact name of variant v (pkg_<tag>/, audit_<tag>.json, kernel_<tag>/, cfg dataset knee-oct21-fa-cfg-<tag>); default = v."""
    return VARIANTS[v].get("tag", v)
SLUG = "knee-oct21-final-{}"             # dataset slug per variant (kernel slug: + "-sub")
FC_MEMBER = {"runs/b_v3@k64": "b_v3", "runs/b_lr3@k64": "b_lr3", "runs/b_nf@k64": "b_nf", "runs/b3c_rep@k64": "b3c_rep",
             "runs/b3l_rep@k64": "b3l_rep", "runs/b3n_rep@k64": "b3n_rep", "bfs_rep": "bfs_rep"}
OA_MEMBER = {"oa_crop_v2cor": "oa_v2cor", "oa_crop_v2cor_nfnet": "oa_v2cor_nf", "oa_oasep": "oasep", "oa_oasep_nf": "oasep_nf", "oa_silsep": "silsep"}
SYN_GROUP = {"cs_fs": "cs_fs", "cs_fc125": "cs_fc125", "s_fs_nf": "s_fs_nf", "s_fc125_nf": "s_fc125_nf", "ef3": "ef3", "ef3fc": "ef3fc",
             "ef3nf": "ef3nf", "ef3nffc": "ef3nffc"}            # per-fold syn members <group>_f<k> -> ONE member <group>_all
SYN_WHOLE = {"w5b_mcl": "w5b_mcl", "z1_acl": "z1_acl", "b3g": "b3g", "sxm": "sxm"}      # members already holding all fold files
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()   # noqa: E731


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out |= flat(v, f"{p}{k}.") if isinstance(v, dict) and k != "members" else {p + k: v}
    return out


def load_kit(name):
    code = nbformat.read(next(KITS[name].glob("*.ipynb")), 4).cells[0].source
    pkg = Path(R.build_work(code, HERE / "_work" / name)[0])
    cfg = json.load(open(pkg / "config.json"))
    return pkg, cfg, json.load(open(pkg / cfg["syn"]["members"]))


def check_touch(kit, cfg, v2):
    f, f0 = flat(cfg), flat(v2)
    bad = [k for k in set(f) | set(f0) if json.dumps(f.get(k), sort_keys=True) != json.dumps(f0.get(k), sort_keys=True)
           and not k.startswith(TOUCH[kit])]
    assert not bad, f"kit {kit} changes keys outside its own blocks: {bad}"


PL_KEYS = {"tgt_csv", "img_csv", "ieff_csv", "ibone_csv", "tgt_cols", "target", "syn_eff", "labels", "ext_labels"}   # PL target inputs
#   (Master 22:1x; oct14 PL lines set labels=ext:pl2_*; 10-22: m_prob05's image-label input is ext_labels -> its PL file pl2_m_prob05_ext_v1)


def is_pl(label):
    """PL seeds are labelled 'P<seed>' (oct16 runs <name>_pl2_s<seed>, oct14 runs PL<seed>_<name>); plain = 'A<k>' / <int seed>."""
    return isinstance(label, str) and label.startswith("P")


def run_dir(recipe, label):
    kind, src, name = RECIPES[recipe]
    if is_pl(label):
        n = int(label[1:])
        return (A14 / f"PL{n}_{name}" / "fold-1") if src == "a14" else (EXP / "runs" / f"{name}_pl2_s{n}" / "fold-1")
    return (A14 / f"{label}_{name}" / "fold-1") if src == "a14" else (EXP / "runs" / f"{name}_all_s{label}" / "fold-1")


def runs_of(recipe, pl=False):
    """-> (kind, {label: run dir}): SEEDS[recipe] None = every finished run of that family (plain, or PL with pl=True; names regex-checked:
    'oasep' never matches 'oasep_nf', a renamed / failed run dir never matches); else exactly the listed labels (plain or 'P<seed>')."""
    kind, src, name = RECIPES[recipe]
    want = SEEDS[recipe]
    if want is not None:
        out = {s: run_dir(recipe, s) for s in want}
        miss = [s for s, d in out.items() if not (d / "done.txt").exists()]
        assert not miss, f"{recipe}: seeds {miss} not finished"
        return kind, out
    if src == "a14":
        base, glob_, rx = A14, (f"PL*_{name}" if pl else f"A*_{name}"), (rf"PL(\d+)_{re.escape(name)}" if pl else rf"(A\d+)_{re.escape(name)}")
    else:
        fam = "pl2" if pl else "all"
        base, glob_, rx = EXP / "runs", f"{name}_{fam}_s*", rf"{re.escape(name)}_{fam}_s(\d+)"
    cands = {}
    for p in sorted(base.glob(f"{glob_}/fold-1")):
        m = re.fullmatch(rx, p.parent.name)
        if m:
            g = m.group(1)
            cands[f"P{g}" if pl else (g if src == "a14" else int(g))] = p
    done = {s_: d for s_, d in cands.items() if (d / "done.txt").exists()}
    assert done, f"{recipe}: no finished all-data {'PL ' if pl else ''}run"
    return kind, done


N_TRAIN = 4349                           # fold -1 = every non-gold study (asserted in each run's log / smoke)


def last_ckpt(d, kind):
    """the checkpoint saved at the run's FINAL step; its epoch must be the one the run's own schedule implies (guards a partial sync /
    an unfinished run: crop experts end at epoch (spe_ref x epochs - 1) // steps, Basin at ceil(stop / steps))."""
    if kind in ("main", "mainp"):
        assert (d / "model.pt").exists(), d
        return d / "model.pt"
    pat = re.compile(r"model_e(\d+)\.pt$" if kind == "basin" else r"model_ep(\d+)\.pt$")
    eps = {int(m.group(1)): f for f in d.iterdir() if (m := pat.match(f.name))}
    assert eps, f"{d}: no checkpoint"
    c = json.load(open(d / "config.json"))
    spe = N_TRAIN // c["bs"] // (c.get("accum") or 1)
    if kind == "basin":
        stop = c.get("stop_step") or (c.get("stop_epoch") or 0) * c["spe_ref"] or c["epochs"] * c["spe_ref"]
        want = -(-stop // spe)
    else:
        want = (c["spe_ref"] * c["epochs"] - 1) // spe
    assert max(eps) == want, f"{d}: last checkpoint epoch {max(eps)} != the schedule's final epoch {want}"
    return eps[max(eps)]


def assemble(a):
    kits = {k: load_kit(k) for k in ["v2", "all4", "ef3"] + (["frac"] if a.frac else []) + (["effc"] if a.effc else []) + (["sxm"] if a.sxm else [])}
    v2 = kits["v2"][1]
    for k in kits:
        if k != "v2":
            check_touch(k, kits[k][1], v2)
    pkg4, cfg, _ = kits["all4"]
    cfg, syn = copy.deepcopy(cfg), copy.deepcopy(kits["ef3"][2])      # syn members: the EF3 set (the all-4 kit keeps v2's)
    e3 = kits["ef3"][1]
    cfg["syn"]["eff_pipelines"], cfg["eff"]["pipelines"], cfg["eff"]["stats"]["Effusion"] = (e3["syn"]["eff_pipelines"], e3["eff"]["pipelines"],
                                                                                            e3["eff"]["stats"]["Effusion"])
    assert not any(n.startswith(("s_fs_f", "s_fc125_f")) for n in syn), "EF3: the effb0 syn members must be gone"
    if a.frac:
        fr = kits["frac"][1]
        for n, e in fr["fc"]["members"].items():
            if n in v2["fc"]["members"]:
                continue
            c4, cf = cfg["fc"]["members"][n], e
            assert c4["files"] == cf["files"], f"{n}: all-4 and frac files differ"
            if n == "bfs_rep":
                assert c4["cfg"].get("nofuse") and cf["cfg"].get("nofuse") and c4 == cf, "bfs_rep"
                continue
            assert c4["cfg"].pop("fuse_set") == "cont" and cf["cfg"].get("fuse_set") == "frac"
            assert {k: v for k, v in c4["cfg"].items()} == {k: v for k, v in cf["cfg"].items() if k != "fuse_set"}, n
            c4["cfg"]["fuse_sets"] = ["cont", "frac"]
        cfg["frac_allmem"], cfg["frac_fs"] = fr["frac_allmem"], fr["frac_fs"]
    if a.effc:
        cfg["eff_main"] = kits["effc"][1]["eff_main"]
    if a.sxm:
        cfg["cont_sxm"] = kits["sxm"][1]["cont_sxm"]
    # DEAD: bfs_fs (cont_fs result overwritten by cont_allmem with the same constants) -> the cont_fs block reads bfs_rep instead
    assert cfg["cont_fs"]["name"] == "bfs_fs" and cfg["cont_allmem"]["fs_member"] == "bfs_rep"
    cfg["cont_fs"]["name"] = "bfs_rep"
    del cfg["fc"]["members"]["bfs_fs"]
    if a.fc_off:
        assert a.frac, "--fc-off needs --frac (F and C then come only from the new groups)"
        for n in ("runs/b_v3@k64", "runs/b_lr3@k64", "runs/b_nf@k64"):
            del cfg["fc"]["members"][n]
        cfg["fc_shipped_off"] = True
    return pkg4, cfg, syn, kits


def swap(cfg, syn, out, plan):
    """Replace every fold file list by all-data files; plan[name] = (source checkpoint, transform, json source or None)."""
    def put(recipe, stem):
        assert stem == STEM[recipe], (recipe, stem)
        kind, runs = runs_of(recipe)
        files = []
        for s, d in runs.items():
            name = f"{stem}_all_{s if isinstance(s, str) else f's{s}'}" + (".pt")
            ck = last_ckpt(d, kind)
            tf = f"strip_{STRIP[recipe]}" if recipe in STRIP else None
            js = d / "config.json" if kind == "basin" else None
            plan[name] = (ck, tf, js, recipe, s, d)
            files.append(name)
        return files
    sub = put("sub_b0", "sub_b0")
    mp = put("m_prob05", "m_prob05")
    cfg["weights"] = sub + mp
    cfg["models"] = [f"{f[:-3]}:all" for f in sub + mp]
    cfg["basin"] = {"nfnet": put("nfnet", "basin_nfnet"), "rad": put("r50", "basin_rad")}
    cfg["basin_syn"] = {"nfnet": put("n_prob05", "basinsyn_nfnet")}
    cfg["win"]["main"], cfg["win"]["basin"] = list(sub), copy.deepcopy(cfg["basin"])
    sd_main = put("sd1b", "main_sd")
    cfg["mcl_sd"]["nfnet"] = put("bsd_nf", "basin_nfnet_sd")
    for n, m in cfg["oa"]["members"].items():
        m["files"] = put(OA_MEMBER[n], f"oa_{n}")
        m["run"], m["epoch"] = f"all-data {OA_MEMBER[n]}", "last"
    for n, m in cfg["fc"]["members"].items():
        m["files"] = put(FC_MEMBER[n], f"fc_{FC_MEMBER[n]}")
        m["run"] = f"all-data {FC_MEMBER[n]}"
        if "epoch" in m:
            m["epoch"] = "last"
    new = {}
    for n, m in syn.items():
        g = n.rsplit("_f", 1)[0]
        if n in SYN_WHOLE:
            new[n] = m | {"files": put(SYN_WHOLE[n], f"syn_{n}")}
        elif g in SYN_GROUP:
            if f"{g}_all" not in new:
                new[f"{g}_all"] = m | {"files": put(SYN_GROUP[g], f"syn_{g}")}
        else:
            raise AssertionError(f"syn member {n}: no all-data recipe")
    ren = lambda ns: list(dict.fromkeys(f"{n.rsplit('_f', 1)[0]}_all" if n.rsplit("_f", 1)[0] in SYN_GROUP else n for n in ns))  # noqa: E731
    cfg["syn"]["pipelines"] = {p: ren(ns) for p, ns in cfg["syn"]["pipelines"].items()}
    cfg["syn"]["eff_pipelines"] = {p: ren(ns) for p, ns in cfg["syn"]["eff_pipelines"].items()}
    for d in cfg["syn"]["fuse"]["designs"].values():
        for p in d:
            d[p] = ren(d[p])
    if cfg.get("cont_sxm"):                              # sxm joins the syn stage via ensemble.syn_members (not the members json)
        cfg["cont_sxm"]["files"] = put("sxm", "syn_sxm")
    if cfg.get("eff_main"):
        em = cfg["eff_main"]
        em["main"] = put("ef3m", "ef3m")
        if em.get("basin"):
            em["basin"] = {"nfnet": put("e3bn", "basin_e3bn")}
    main_pl = route_config(cfg, plan, sub) if ROUTE["routes"] else []
    return new, sd_main, main_pl


def route_config(cfg, plan, sub):
    """oct22 routes -> the PL files in plan + config: main_pl (a 2nd main model set, same pass), basin_pl / basin_route, OA '@pl' copy members
    (in loa_extra: never in the deployed fusion; not in loa_set: never in LOA's) + extra_sets per routed column + the men experts list;
    window 'both' = the routed generalists' PL files join the window passes. -> the main_pl weight files."""
    RT, PLL = ROUTE["routes"], ROUTE["pl"]

    def put_pl(recipe, stem):
        kind, files = RECIPES[recipe][0], []
        for s in PLL[recipe]:
            d = run_dir(recipe, s)
            name = f"{stem}_all_{s}.pt"
            plan[name] = (last_ckpt(d, kind), f"strip_{STRIP[recipe]}" if recipe in STRIP else None, d / "config.json" if kind == "basin" else None,
                          recipe, s, d)
            files.append(name)
        return files
    main_pl = []
    if "sub_b0" in RT:
        main_pl = put_pl("sub_b0", "sub_b0") + put_pl("m_prob05", "m_prob05")
        rt = {c: r for x in ("sub_b0", "m_prob05") for c, r in RT[x].items() if r != "plain"}
        cfg["main_pl"] = {"main_dir": "main_pl", "route": rt}
    bpl = {p: put_pl(r_, STEM[r_]) for p, r_ in (("nfnet", "nfnet"), ("rad", "r50")) if r_ in RT}
    if bpl:
        cfg["basin_pl"] = bpl
        cfg["basin_route"] = {p: {c: r for c, r in RT[r_].items() if r != "plain"} for p, r_ in (("nfnet", "nfnet"), ("rad", "r50")) if r_ in RT}
    oa = cfg["oa"]
    oar = {n: r_ for n, r_ in OA_MEMBER.items() if r_ in RT and n in oa["members"]}
    if oar:
        assert oa.get("loa_set"), "routes: OA PL members need a loa_set (else they would join LOA's fusion)"
        dep = [n for n in oa["members"] if n not in oa.get("loa_extra", [])]          # the deployed (plain) members, in order
        for n, r_ in oar.items():
            assert n in dep, n
            oa["members"][f"{n}@pl"] = copy.deepcopy(oa["members"][n]) | {"files": put_pl(r_, f"oa_{n}"), "run": f"all-data {r_} PL"}
        oa["loa_extra"] = list(oa.get("loa_extra", [])) + [f"{n}@pl" for n in oar]

        def members(col):
            out = []
            for n in dep:
                rr = RT[oar[n]].get(col, "plain") if n in oar else "plain"
                out += ([n] if rr in ("plain", "both") else []) + ([f"{n}@pl"] if rr in ("pl", "both") else [])
            return out
        for col, key in OA_RCOLS.items():
            if members(col) != dep:
                oa.setdefault("extra_sets", {})[key] = members(col)
                oa.setdefault("extra_align", {})[key] = {"col": col, **oa["fuse"]["align"][col]}
        men = cfg["men"]["experts"]
        assert set(men) <= set(dep), men
        mr = {n: RT[oar[n]].get(MEN_COLS[0], "plain") if n in oar else "plain" for n in men}     # MM == LM (check_routes)
        cfg["men"]["experts"] = [x for n in men for x in ([n] if mr[n] in ("plain", "both") else []) + ([f"{n}@pl"] if mr[n] in ("pl", "both") else [])]
    if ROUTE["window"] == "both":
        cfg["win"]["main"] = list(sub) + [f for f in main_pl if f.startswith("sub_b0_")]
        cfg["win"]["basin"] = {p: list(cs) + bpl.get(p, []) for p, cs in cfg["win"]["basin"].items()}
    return main_pl


def stage_counts(cfg, pkg):
    """RUNTIME table: models run per stage (the Basin pass runs every distinct checkpoint once; the main window pass = win main + eff_main
    main; main_sd = the MCL SD-2 main pass)."""
    syn = json.load(open(Path(pkg) / cfg["syn"]["members"]))
    em = cfg.get("eff_main") or {}
    basin = set(sum(list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) + [cfg["mcl_sd"]["nfnet"]] + list(em.get("basin", {}).values()), []))
    sdc = Path(pkg) / cfg["mcl_sd"]["main_dir"] / "config.json"
    sdc = sdc if sdc.exists() else Path(pkg) / "main_sd_config.json"     # the flat package (Kaggle datasets skip subdirectories)
    return {"main": len(cfg["weights"]), "main_sd": len(json.load(open(sdc))["weights"]),
            "basin": len(basin), "win_main": len(cfg["win"]["main"]) + len(em.get("main", [])),
            "oa": sum(len(m["files"]) for m in cfg["oa"]["members"].values()), "fc": sum(len(m["files"]) for m in cfg["fc"]["members"].values()),
            "syn": sum(len(m["files"]) for m in syn.values()) + len((cfg.get("cont_sxm") or {}).get("files", []))}


def materialize(name, entry, store, index):
    """one all-data file (+ its Basin json) into the STORE, once: the store is immutable (a name never changes content: the source md5
    is asserted on every later build); every weight is key-for-key == the shipped fold file of its recipe."""
    import torch
    ck, tf, js, recipe, s, d = entry
    src_md5 = md5(ck)
    if name in index:
        assert index[name]["src_md5"] == src_md5 and index[name]["src"] == str(ck.relative_to(REPO)), f"{name}: source changed since stored"
        return
    dst = store / name
    if tf:
        torch.save(strip(torch.load(ck, map_location="cpu"), STRIP[recipe]), dst)
    else:
        os.link(ck, dst)                                 # hardlink (same filesystem; the store never rewrites a file)
    ks = set(torch.load(dst, map_location="cpu"))
    assert ks == ship_keys(recipe), (name, sorted(ks ^ ship_keys(recipe))[:10])
    rec = {"md5": md5(dst), "src": str(ck.relative_to(REPO)), "src_md5": src_md5, "transform": tf, "recipe": recipe, "seed": s,
           "run": str(d.relative_to(REPO)), "src_cfg": SRC.get(recipe, "oct14_all_data")}
    if js is not None:      # basin_predict reads the run config next to each checkpoint; bsd_nf: the deployed NFNet architecture json (as SD-2)
        src_js = A14 / "A1_nfnet/fold-1/config.json" if recipe == "bsd_nf" else js
        shutil.copyfile(src_js, dst.with_suffix(".json"))
        index[dst.with_suffix(".json").name] = {"md5": md5(dst.with_suffix(".json")), "src": str(src_js.relative_to(REPO)), "src_md5": md5(src_js),
                                                "transform": None, "recipe": recipe, "seed": s, "json_of": name}
    index[name] = rec


ALLOW = {"fold", "seed", "spe_ref", "name", "out", "stop_epoch", "stop_step", "save", "workers"}   # the all-data overrides (queue lines); workers: box-2 lines (Master 17:55: data = per-item rng (seed, epoch, i), main-process sampler -> worker-count independent)


def run_config_problems(recipe, d):
    """audit check 1 BEFORE anything is stored (Master 16:2x): a new run's config == its queue line's --src config except the all-data
    overrides (silsep: + tgt_csv = v4b_fall) and the queue line names that src with fold -1; an oct14 run's config == the recipe's A1 run
    except seed / name / out. A PL run (Master 22:1x) may also differ in PL_KEYS and MUST train on a PL target (a value naming pl2_* / pl_*).
    -> list of problems (empty = OK)."""
    c = json.load(open(d / "config.json"))
    pl = d.parent.name.startswith("PL") or "_pl2_s" in d.parent.name
    extra = PL_KEYS if pl else set()
    if RECIPES[recipe][1] == "a14":
        c0 = json.load(open(A14 / f"A1_{RECIPES[recipe][2]}/fold-1/config.json"))
        diff = sorted({k for k in set(c) | set(c0) if c.get(k) != c0.get(k)} - {"seed", "name", "out"} - extra)
        out = [f"vs A1: {diff}"] if diff else []
    else:
        sys.path.insert(0, str(EXP / "scripts"))
        import launch as L
        q = {ln.split()[0]: ln.split() for qf in ("queue.txt", "queue_remote.txt", "queue_remote2.txt") if (EXP / qf).exists() for ln in open(EXP / qf)
             if ("_all_s" in ln or "_pl2_s" in ln) and not ln.startswith("#")}
        t = q.get(d.parent.name)
        out = [] if t is not None and t[2] == SRC[recipe] and t[3] == "-1" else [f"queue line {t} != src {SRC[recipe]}"]
        src = {k: L.REMAP.get(v, v) if isinstance(v, str) else v for k, v in json.load(open(EXP / SRC[recipe] / "config.json")).items()}
        diff = {k for k in set(c) & set(src) if c[k] != src[k]} - ALLOW - extra
        if recipe == "silsep" and str(c.get("tgt_csv", "")).endswith("data/targets/v4b_fall.csv"):
            diff -= {"tgt_csv"}
        if diff or c.get("fold", -1) not in (-1, None):
            out.append(f"config vs src: {sorted(diff)}")
    if pl and not [k for k in PL_KEYS if re.search(r"(^|[/:])pl2?_", str(c.get(k, "")))]:
        out.append("PL run without a PL target (no PL_KEYS value names pl2_* / pl_*)")
    return out


def all_finished_plan(skip=(), only=None, pl=False):
    """every finished all-data run of every recipe (SEEDS None) -> ({name: plan entry}, skipped): what --stock materialises. A run is
    SKIPPED (reported, retried by the next --stock) unless done.txt + the last checkpoint at the schedule's final epoch (a half-pulled box
    run fails this) + its config passes audit check 1."""
    plan, skipped = {}, []
    for r in RECIPES:
        if r in skip:
            continue
        try:
            kind, runs = runs_of(r, pl)
        except AssertionError:
            continue                                     # no finished run yet (e.g. sxm / ef3m / e3bn before their queue lines)
        for s_, d in runs.items():
            name = f"{STEM[r]}_all_{s_ if isinstance(s_, str) else f's{s_}'}.pt"
            if only is not None and s_ not in only.get(r, []):
                continue                                 # --only-variant: a seed outside that variant's plan stays out of the store
            try:
                ck = last_ckpt(d, kind)
            except AssertionError as e:
                skipped.append((name, str(e)[:160]))
                continue
            prob = run_config_problems(r, d)
            if prob:
                skipped.append((name, "; ".join(prob)[:200]))
                continue
            plan[name] = (ck, f"strip_{STRIP[r]}" if r in STRIP else None, d / "config.json" if kind == "basin" else None, r, s_, d)
    return plan, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), help="write the variant's small CONFIG dataset + the local union mirror pkg_<variant>/")
    ap.add_argument("--stock", action="store_true", help="materialise every finished all-data run of every recipe into the store")
    ap.add_argument("--skip", default="", help="--stock: recipes left out (dead in every planned variant), e.g. b_v3,b_lr3,b_nf")
    ap.add_argument("--only-variant", default="", help="--stock: only the seeds that VARIANT lists (e.g. final: the USER's plan)")
    ap.add_argument("--pl", action="store_true", help="--stock: the PL runs (<name>_pl2_s2xx / PL2xx_<name>, labels P2xx) instead of the plain ones")
    ap.add_argument("--freeze", metavar="SLUG_SUFFIX", help="move every not-yet-frozen store file into a new weights dataset knee-oct21-fa-<suffix>")
    ap.add_argument("--reg", default="final", help="registry name (store / datasets / ds dirs); 'test' for dry end-to-end runs")
    ap.add_argument("--check-variant", default="", help="validate a variant JSON (routes, labels finished) without building")
    ap.add_argument("--dry", action="store_true")
    v = ap.parse_args()
    if v.check_variant:                                  # oct22 (8b): validate a variant JSON without building
        d = json.load(open(v.check_variant))
        prob = check_routes(d, runs=True) + ([] if set(d.get("form", {})) == {"frac", "effc", "sxm", "fc_off"} else ["form keys"]) \
            + [f"unknown recipe {r}" for r in set(d.get("seeds", {})) - set(RECIPES)]
        print(f"CHECK {v.check_variant}: " + ("OK" if not prob else f"{len(prob)} problems: {prob}"))
        sys.exit(1 if prob else 0)
    assert sum(map(bool, (v.variant, v.stock, v.freeze))) == 1, "one of --variant / --stock / --freeze"
    reg = HERE / f"reg_{v.reg}"
    store, ds_root = reg / "store", reg / "ds"
    store.mkdir(parents=True, exist_ok=True)
    ds_root.mkdir(exist_ok=True)
    ipath, dpath = reg / "store_index.json", reg / "datasets.json"
    index = json.load(open(ipath)) if ipath.exists() else {}
    datasets = json.load(open(dpath)) if dpath.exists() else {}
    frozen_of = {f: slug for slug, d in datasets.items() for f in d["files"]}
    if v.stock:
        skip = [r for r in v.skip.split(",") if r]
        assert set(skip) <= set(RECIPES), skip
        only = VARIANTS[v.only_variant]["seeds"] if v.only_variant else None
        plan, skipped = all_finished_plan(skip, only, v.pl)
        new = [n for n in plan if n not in index]
        if v.dry:                                        # report only (10-22 04:26: --stock --dry used to stock for real)
            for n, why in skipped:
                print(f"  SKIPPED {n}: {why}")
            print(f"STOCK DRY ({v.reg}): {len(plan)} would pass, {len(new)} new, {len(skipped)} skipped; nothing written")
            return
        for n, e in plan.items():
            materialize(n, e, store, index)              # + the key-for-key assert vs the shipped fold file
        json.dump(index, open(ipath, "w"), indent=1)
        for n, why in skipped:
            print(f"  SKIPPED {n}: {why}")
        print(f"STOCK ({v.reg}): {len(plan)} finished all-data files passed (done + final-epoch checkpoint + audit check 1 config + key set), "
              f"{len(new)} new in the store, {len(skipped)} skipped; store {len(index)} entries; not yet frozen {len([f for f in index if f not in frozen_of])}")
        return
    if v.freeze:
        slug = f"knee-oct21-fa-{v.freeze}"
        assert slug not in datasets and re.fullmatch(r"[a-z0-9-]{1,30}", v.freeze), slug
        files = sorted(f for f in index if f not in frozen_of)
        if not datasets:                                  # the first (base) dataset also carries the static localiser / detector assets
            kits = {"all4": load_kit("all4")}
            pkg4, c4 = kits["all4"][0], kits["all4"][1]
            for n in sorted({c4["oa"]["loc"], c4["oa"]["detector"], c4["fc"]["loc"], c4["fc"]["detector"], c4["syn"]["loc"], c4["syn"]["detector"],
                             c4["basin_detector"]}):
                shutil.copyfile((pkg4 / n).resolve(), store / n)
                index[n] = {"md5": md5(store / n), "src": str((pkg4 / n).resolve().relative_to(REPO)), "transform": None, "recipe": "asset"}
                files.append(n)
        assert files, "nothing new to freeze"
        out = ds_root / slug
        out.mkdir()
        for f in files:
            assert md5(store / f) == index[f]["md5"], f"{f}: store file changed"
            os.link(store / f, out / f)                  # hardlink: no copy; the store is never rewritten
        json.dump({"title": slug, "id": f"pavelsavchenkov/{slug}", "licenses": [{"name": "CC0-1.0"}]}, open(out / "dataset-metadata.json", "w"), indent=1)
        datasets[slug] = {"files": {f: index[f]["md5"] for f in files}, "frozen": time.strftime("%Y-%m-%d %H:%M"),
                          "gb": round(sum((out / f).stat().st_size for f in files) / 1e9, 2)}
        json.dump(index, open(ipath, "w"), indent=1)
        json.dump(datasets, open(dpath, "w"), indent=1)
        print(f"FROZE {slug} ({v.reg}): {len(files)} files, {datasets[slug]['gb']} GB -> {out} (upload: kaggle datasets create -p {out})")
        return
    var = VARIANTS[v.variant]
    assert set(var["seeds"]) <= set(RECIPES), set(var["seeds"]) - set(RECIPES)
    for r in RECIPES:                                    # seeds not listed = EVERY FROZEN seed of the recipe (a run finished after the last
        if r not in var["seeds"]:                        # freeze waits for the next delta instead of breaking the variant)
            fr = [index[f]["seed"] for f in frozen_of if index.get(f, {}).get("recipe") == r and "json_of" not in index[f]
                  and not is_pl(index[f]["seed"])]        # PL seeds only when a variant lists them
            if var.get("cap"):                           # capped at the USER's plan: only the plan's labels, in plan order
                want = plan_seeds(r, PLAN.get(r, 3))
                fr = [x for x in want if x in fr]
            SEEDS[r] = sorted(fr, key=lambda x: (isinstance(x, str), int(str(x).lstrip("A")))) or None
    SEEDS.update(var["seeds"])
    ROUTE.update(pl=var.get("pl", {}), routes=var.get("routes", {}), window=var.get("window", "plain"), code=var.get("code", ""))
    prob = check_routes(var, runs=True)
    assert not prob, f"variant {v.variant}: {prob}"
    a = argparse.Namespace(**var["form"], dry=v.dry)
    pkg4, cfg, syn, kits = assemble(a)
    print("RUNTIME (models per stage): v2 " + str(stage_counts(kits["v2"][1], kits["v2"][0])), flush=True)
    if a.dry:                                            # merges validated; per-recipe status of the all-data runs
        for r, (kind, src, name) in RECIPES.items():
            base = A14 if src == "a14" else EXP / "runs"
            pat = f"A*_{name}" if src == "a14" else f"{name}_all_s*"
            st = {p.name: (p / "fold-1" / "done.txt").exists() for p in sorted(base.glob(pat))}
            print(f"  {r:12s} {kind:6s} done {[k for k, v in st.items() if v]} running / pending {[k for k, v in st.items() if not v]}")
        return
    plan = {}
    new_syn, sd_main, main_pl = swap(cfg, syn, None, plan)
    print(f"FORM: v2 + all-4 + EF3{' + frac v3c_fs' if a.frac else ''}{' + eff form c' if a.effc else ''}{' + cont sxm' if a.sxm else ''}"
          f"{' + fc_shipped_off' if a.fc_off else ''}; {len(plan)} all-data checkpoints")
    for r in RECIPES:
        used = sorted({str(e[4]) for e in plan.values() if e[3] == r})
        if used:
            print(f"  {r:12s} {RECIPES[r][0]:6s} seeds {used}")
    assets = {cfg["oa"]["loc"], cfg["oa"]["detector"], cfg["fc"]["loc"], cfg["fc"]["detector"], cfg["syn"]["loc"], cfg["syn"]["detector"], cfg["basin_detector"]}
    need = set(plan) | {n[:-3] + ".json" for n, e in plan.items() if e[2] is not None} | assets
    for n, e in plan.items():                            # every planned file must already be in the store with the same source
        assert n in index, f"{n}: not in the store - run --stock (and --freeze) first"
        assert index[n]["src"] == str(e[0].relative_to(REPO)) and index[n]["src_md5"] == md5(e[0]), f"{n}: source differs from the stored one"
    unfrozen = sorted(f for f in need if f not in frozen_of)
    assert not unfrozen, f"{len(unfrozen)} needed files are not in a frozen dataset - run --freeze: {unfrozen[:5]}"
    wsets = sorted({frozen_of[f] for f in need}, key=list(datasets).index)
    cslug = f"knee-oct21-fa-cfg-{tag(v.variant).replace('_', '-')}"
    cout = ds_root / cslug
    shutil.rmtree(cout, ignore_errors=True)
    cout.mkdir()
    for p_ in pkg4.iterdir():                            # code + csv (check_studies.csv) from the all-4 work dir; ensemble.py = the merged one
        if p_.is_file() and p_.suffix in (".py", ".csv") and p_.name != "ensemble.py":
            shutil.copyfile(p_.resolve(), cout / p_.name)
    shutil.copyfile(HERE / "pkg_code/ensemble.py", cout / "ensemble.py")
    if ROUTE["routes"] or ROUTE["code"] == "route":          # oct22: the routing code overlay ONLY for routed variants (+ the switch-off proof)
        for f in ("ensemble.py", "knee_infer.py"):
            shutil.copyfile(HERE / "pkg_code_route" / f, cout / f)
    if main_pl:                                              # the PL main set's weights dir (config main_pl main_dir), mirrored like main_sd
        mpc = {k: cfg[k] for k in ("backbone", "img", "crop_mm", "k_eval", "span", "hidden", "drop", "head", "use_type", "use_pos", "n_aux",
                                   "synovitis_ext_w", "synovitis_effusion_w", "prior") if k in cfg} | {"weights": main_pl, "models": [f"{f[:-3]}:all" for f in main_pl]}
        json.dump(mpc, open(cout / "main_pl_config.json", "w"))
    sdc = json.load(open(pkg4 / "main_sd" / "config.json"))      # the stack kernel's main_sd/config.json (base main config) with the SD files;
    sdc["weights"] = sd_main                                     # the kernel mirrors it to WORK/main_sd/config.json (mcl_sd main_dir)
    json.dump(sdc, open(cout / "main_sd_config.json", "w"))
    json.dump(new_syn, open(cout / cfg["syn"]["members"], "w"), indent=1)
    cfg["all_data"] = {"variant": v.variant, "form": var["form"], "registry": v.reg, "datasets": wsets + [cslug],
                       "sources": {n: index[n]["src"] + (f" [{index[n]['transform']}]" if index[n]["transform"] else "") for n in plan},
                       "md5_src": {n: index[n]["src_md5"] for n in plan},
                       "runs": {n: {k: index[n][k] for k in ("recipe", "seed", "run")} | {"src": index[n]["src_cfg"], "dataset": frozen_of[n]} for n in plan},
                       "note": "every member's fold models -> its recipe's all-data runs (fold -1, spe_ref = the fold runs' steps / epoch); recipe "
                               "constants unchanged (fold OOF), as oct15_all_data_s5; weights in the datasets listed (union), code + config here"}
    bfiles = sorted({c for cs in list(cfg["basin"].values()) + list(cfg["basin_syn"].values()) + list((cfg.get("basin_pl") or {}).values()) for c in cs})
    cfg["basin_sources"] = {n: cfg["all_data"]["sources"][n] for n in bfiles}     # info only (was the fold sources)
    json.dump(cfg, open(cout / "config.json", "w"), indent=1)
    json.dump({"title": cslug, "id": f"pavelsavchenkov/{cslug}", "licenses": [{"name": "CC0-1.0"}]}, open(cout / "dataset-metadata.json", "w"), indent=1)
    for d_ in [ds_root / w for w in wsets] + [cout]:      # no junk: flat, no __pycache__
        assert not [p_ for p_ in d_.iterdir() if p_.is_dir()], f"{d_}: subdirectory"
    pk = HERE / f"pkg_{tag(v.variant)}"                  # the local UNION mirror (what the kernel builds in WORK, minus main_sd/): audit / tests
    shutil.rmtree(pk, ignore_errors=True)
    pk.mkdir()
    for d_ in [ds_root / w for w in wsets] + [cout]:
        for f in d_.iterdir():
            if f.name != "dataset-metadata.json":
                assert not (pk / f.name).exists(), f"{f.name}: in two datasets"
                os.symlink(f.resolve(), pk / f.name)
    print("RUNTIME (models per stage): final " + str(stage_counts(cfg, pk)), flush=True)
    print(f"{cslug}: {len(list(cout.iterdir()))} files; datasets {wsets + [cslug]} "
          f"({sum(datasets[w]['gb'] for w in wsets):.2f} GB weights; unused files in them {len([f for w in wsets for f in datasets[w]['files'] if f not in need])}); "
          f"union mirror {pk.name}/ -> run audit.py --variant {v.variant}")


if __name__ == "__main__":
    main()
