"""oct20_stack_all_v2 (Experts V3 10-19 20:4x; Master's LB audit for the USER 'every LB-confirmed improvement, nothing LB-untested'):
= oct20_stack_all (frozen 16:56) with P2 OFF by default (eff961: Effusion = the 0.961 form - img_eff from the six effb0 crop members s_fs_f* /
s_fc125_f* with 0.961's stats; P2 was a runtime prune with no detected gain, never LB-tested); --p2 restores v1's P2 (tag _p2). Kernel names
carry '-v2'. Everything else is v1's code path unchanged.
--- v1 docstring:
oct20_stack_all (Experts V3 10-19; USER GO via Master 10-19 ~15:40: the merged fold package for a 10-20 slot, "and include our inference
optimizations"; OAI: no) = oct19_stack_fold's notebook with P1 (one shared b3g pass) + P2 (Effusion img_eff from the ConvNeXt crops) + LM Q22
always ON, + Contusion FS (cont_fs bfs splice, LB-won 57388555) + the MCL Q6 re-fitted on the new Contusion (refit.json, build.py), + SWITCHES
(default OFF; --frac-mfs / --pf-q20 / --bakers-allpw; each = its probe's own config): m_fs Fracture (57399358), PF Q20 (57399364; extra fusion
over the already-loaded oasep / oasep_nf members, Q20's align), Q25 Baker's all-pw (57399558); --lm-allpw (C184 G1, never probed alone: LM's
3x3 base from Q25's pw generalists = the bakers-allpw dataset's pw main / NFNet / R50 fold models). A switch flip = regenerate the notebook
(all member datasets are always attached). Kernel dir: kernel{tag}/ (tag = the ON switches: _mfs / _q20 / _q25 / _lpw).
conda run -n rsna python submissions/oct20_stack_all_v2/make_kernel.py [--p2] [--frac-mfs] [--pf-q20] [--bakers-allpw] [--lm-allpw]
--- the base (oct19_stack_fold) docstring:
Write the Kaggle notebook kernel/ of oct19_stack_fold (Experts V3 10-18, Master; USER: "i don't want merged fold-avg probe to be submitted.
i want it to be measured though and pruned, when possible"): the 0.961 fold package (knee-oct13-f5-win) with the four LB-won column forms
stacked, each by its single-column probe's own code path: MCL = Q21 (oct18_probe_mcl_stack: SD-2 + W5b zoom + b3g area MCL + Q6),
LOA = Q15 (oct18_probe_loa_allup), ACL = Q16 (oct18_probe_acl_b3g: Z1 notch40 zoom + b3g area ACL), Synovitis = the ConvNeXt crop slots
(oct16_syn_cnx_probe, 57283240). Datasets: 0.961 + knee-oct17-mcl-zoom + knee-oct18-mcl-sd2 + knee-oct18-acl-b3g + knee-oct18-loa-allup +
knee-oct16-syn-cnx-probe (only its 6 cs_* weights / member cfgs). Built-in prune: ONE b3g pass (keep_bags 'mcl,notch') for MCL and ACL.
DEC (post-hoc, final probabilities): variant a = {MCL: Q21's Q6}; variant b (--lm-q8) = + {LM: Q22's Q8 re-fitted on the Q16 ACL}.
NEVER SUBMITTED (user); measured (run_local, a Kaggle commit run by Master) and pruned.
conda run -n rsna python submissions/oct19_stack_fold/make_kernel.py [--lm-q8]"""
import argparse
import json
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
SUB = HERE.parent
CODE = r'''
import faulthandler, glob, json, os, platform, subprocess, sys, threading, time
T0 = time.time()
faulthandler.enable(all_threads=True)   # a segfault / abort prints every thread's traceback (commit v1 died without one: an OOM kill prints nothing)


def monitor(every=120):                 # process RSS, MemAvailable, GPU memory (nvidia-smi: no CUDA call from this thread)
    while True:
        time.sleep(every)
        try:
            rss = int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 2**20
            av = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=20).stdout.split()
            print(f"[mon {{time.time() - T0:.0f}}s] rss {{rss:.1f}} GB, MemAvailable {{av:.1f}} GB, gpu used {{gpu}} MiB", flush=True)
        except Exception as e:
            print(f"[mon] {{type(e).__name__}}: {{e}}", flush=True)


threading.Thread(target=monitor, daemon=True).start()
import numpy as np, pandas as pd, torch
SLUG = {slug!r}
INPUT = os.environ.get("KNEE_INPUT", "/kaggle/input")  # local dry run: a folder laid out like /kaggle/input
COMP = next(p for p in [INPUT + "/competitions/rsna-knee-abnormality-detection", INPUT + "/rsna-knee-abnormality-detection"]
            if os.path.exists(p + "/test.csv"))
PKG = next(os.path.dirname(p) for pat in ["/*/config.json", "/*/*/config.json", "/*/*/*/config.json"]
           for p in sorted(glob.glob(INPUT + pat)) if SLUG in p)
def _find(slug, fname):
    return next(os.path.dirname(p) for pat in ["/*/" + fname, "/*/*/" + fname, "/*/*/*/" + fname] for p in sorted(glob.glob(INPUT + pat)) if slug in p)


ZMEM = _find({zslug!r}, "member.json")                  # Q21: the MCL zoom member w5b_mcl (+ mcl_member)
SDM = _find({sslug!r}, "member_sd.json")               # Q21: SD-2 (5 main SD + 5 NFNet SD fold weights + the zoom re-fit)
AMEM = _find({aslug!r}, "member.json")                 # Q16: z1_acl + b3g (b3g's files = the MCL-stack dataset's, md5-equal)
LMEM = _find({lslug!r}, "member.json")                 # Q15: oasep / oasep_nf / silsep OA groups + loa_set / align
CMEM = _find({cslug!r}, "syn_members.json")            # Syn crops: the 6 ConvNeXt crop members (cs_*) of 57283240's package
CFM = _find({cfslug!r}, "member.json")                  # Contusion FS: bfs_fs (5 fold bone models, nofuse) + cont_fs constants
MFM = _find({mfslug!r}, "member.json")                  # SWITCH frac_mfs: m_fs (5 fsmain fold models) + frac_mfs constants
BKM = _find({bkslug!r}, "member.json")                  # SWITCH bakers_allpw: pw main f0-4 + pw NFNet / R50 basin f0-4 (T_eval 16 json)
SWITCHES = {switches!r}
PF_Q20 = {pf_q20!r}                                     # SWITCH pf_q20: oct18_probe_pf_allup's loa_set / align (its members = Q15's, md5-equal)
STACK = {stack!r}                                       # Q21: mcl_member2 = b3g aligned to the SD-2 + zoom MCL
# oct19_stack_fold (Experts V3 10-18, Master; USER: measured + pruned, NEVER submitted): the 0.961 fold package for 7 columns (+ LM in variant a)
# and the four LB-won column forms - MCL = Q21, LOA = Q15, ACL = Q16, Synovitis = the ConvNeXt crop slots - each by its probe's own code
# path (config keys on disjoint columns). FULL form: b3g runs twice as in the probes (b3g_mcl keep mcl, b3g_acl keep notch; same weights);
# prune variant P1 (B3G_SHARED): ONE b3g pass (keep_bags 'mcl,notch') serves both.
B3G_SHARED = {p1!r}
P2_EFF = {p2!r}   # prune variant P2 (C125): Effusion's img_eff 'main' pipeline from the ConvNeXt crop members (img_eff stats re-fitted label-free)
WORK = "/kaggle/working/pkg_stack" if os.path.isdir("/kaggle/working") else os.path.join(os.environ.get("TMPDIR", "/tmp"), "pkg_stack")
os.makedirs(WORK, exist_ok=True)
import shutil
PATCHED = {{"syn_stage.py": {syn_src!r}, "ensemble.py": {ens_src!r}, "oa_stage.py": {oa_src!r}, "fc_stage.py": {fc_src!r}, "check_studies.csv": {chk_src!r}}}
for fn in os.listdir(PKG):
    if fn in ("config.json", "__pycache__", *PATCHED) or os.path.exists(os.path.join(WORK, fn)):
        continue
    if fn.endswith(".py"):       # copies: knee_infer puts Path(__file__).resolve().parent first on sys.path
        shutil.copy(os.path.join(PKG, fn), os.path.join(WORK, fn))
    else:
        os.symlink(os.path.join(PKG, fn), os.path.join(WORK, fn))


def _link(src_dir, files, sub=""):
    for fn in files:
        dst = os.path.join(WORK, sub, fn)
        assert not os.path.exists(dst), dst
        os.symlink(os.path.join(src_dir, fn), dst)


for fn, src in PATCHED.items():
    open(os.path.join(WORK, fn), "w").write(src)
_cfg = json.load(open(PKG + "/config.json"))
_sm = json.load(open(PKG + "/" + _cfg["syn"]["members"]))
for k in ("mcl_member", "mcl_member2", "mcl_sd", "acl_member", "acl_member2"):
    assert k not in _cfg, k
# MCL (Q21): zoom member + SD-2 + b3g area MCL
_mz = json.load(open(ZMEM + "/member.json"))
_link(ZMEM, _mz["entry"]["files"])
_sm[_mz["name"]] = {{"cfg": _mz["entry"]["cfg"], "files": _mz["entry"]["files"]}}
_sd = json.load(open(SDM + "/member_sd.json"))
_link(SDM, _sd["main_files"] + _sd["nfnet_files"] + [f.replace(".pt", ".json") for f in _sd["nfnet_files"]])
os.makedirs(WORK + "/main_sd", exist_ok=True)
_link(SDM, _sd["main_files"], "main_sd")
json.dump(json.load(open(PKG + "/config.json")) | {{"weights": _sd["main_files"]}}, open(WORK + "/main_sd/config.json", "w"))
_cfg["mcl_sd"] = {{"main_dir": "main_sd", "nfnet": _sd["nfnet_files"], "r50": "rad"}}
_cfg["mcl_member"] = _mz["mcl_member"] | _sd["mcl_member_refit"]
assert STACK["name"] == "b3g"
_cfg["mcl_member2"] = STACK | {{"name": "b3g" if B3G_SHARED else "b3g_mcl"}}
# ACL (Q16): z1_acl + b3g. FULL: b3g_acl (keep notch) + b3g_mcl (keep mcl) = two passes of the same weights (as the probes ran);
# P1: one entry 'b3g' keep 'mcl,notch'
_am = json.load(open(AMEM + "/member.json"))
for _n, _e in _am["members"].items():
    _link(AMEM, _e["files"])
    _c = dict(_e["cfg"])
    if _n == "b3g":
        assert _c["keep_bags"] == "notch", _c["keep_bags"]
        if B3G_SHARED:
            _sm["b3g"] = {{"cfg": _c | {{"keep_bags": "mcl,notch"}}, "files": _e["files"]}}
        else:
            _sm["b3g_acl"] = {{"cfg": _c, "files": _e["files"]}}
            _sm["b3g_mcl"] = {{"cfg": _c | {{"keep_bags": "mcl"}}, "files": _e["files"]}}
        continue
    assert _n not in _sm, _n
    _sm[_n] = {{"cfg": _c, "files": _e["files"]}}
_cfg["acl_member"] = _am["acl_member"]
_cfg["acl_member2"] = _am["acl_member2"] | {{"name": "b3g" if B3G_SHARED else "b3g_acl"}}
# Synovitis: the ConvNeXt crop slots (the effb0 crop members stay for Effusion's img_eff = eff_pipelines)
_cm = json.load(open(CMEM + "/syn_members.json"))
_cnx = sorted(k for k in _cm if k.startswith("cs_"))
assert len(_cnx) == 6, _cnx
for _n in _cnx:
    _link(CMEM, _cm[_n]["files"])
    assert _n not in _sm, _n
    _sm[_n] = _cm[_n]
_syn = _cfg["syn"]
_syn["eff_pipelines"] = json.loads(json.dumps(_syn["pipelines"]))
_syn["pipelines"]["main"] = [n.replace("s_fs_", "cs_fs_").replace("s_fc125_", "cs_fc125_") for n in _syn["pipelines"]["main"]]
for _d in _syn["fuse"]["designs"].values():
    _d["main"] = [n.replace("s_fs_", "cs_fs_").replace("s_fc125_", "cs_fc125_") for n in _d["main"]]
assert set(_syn["pipelines"]["main"]) == set(_cnx), _syn["pipelines"]["main"]
if P2_EFF:   # P2: the six effb0 crop members served only Effusion's img_eff -> taken from the ConvNeXt crops; the effb0 members leave the stack
    _syn["eff_pipelines"]["main"] = list(_syn["pipelines"]["main"])
    _drop = [n for n in _sm if n.startswith(("s_fs_f", "s_fc125_f"))]
    assert len(_drop) == 6, _drop
    for _n in _drop:
        del _sm[_n]
    assert all(n in _sm for ns in _syn["eff_pipelines"].values() for n in ns) and all(n in _sm for ns in _syn["pipelines"].values() for n in ns)
    _cfg["eff"]["stats"]["Effusion"].update(P2_EFF)
json.dump(_sm, open(WORK + "/syn_members_stack.json", "w"))
_syn["members"] = "syn_members_stack.json"
# LOA (Q15): the upgraded OA groups for LOA only
_lm = json.load(open(LMEM + "/member.json"))
_le = _lm["members"]
for _e in _le.values():
    _link(LMEM, _e["files"])
assert not set(_le) & set(_cfg["oa"]["members"]) and "loa_extra" not in _cfg["oa"]
_cfg["oa"]["members"].update(_le)
_cfg["oa"]["loa_extra"], _cfg["oa"]["loa_set"], _cfg["oa"]["loa_extra_align"] = list(_le), _lm["loa_set"], _lm["align"]
# Contusion FS (LB-won): the bfs_fs bone member (nofuse) + the cont_fs splice constants (oct17_probe_cont_fs)
_cf = json.load(open(CFM + "/member.json"))
_link(CFM, _cf["entry"]["files"])
assert _cf["name"] not in _cfg["fc"]["members"] and "cont_fs" not in _cfg and _cf["entry"]["cfg"].get("nofuse"), "cont_fs config"
_cfg["fc"]["members"][_cf["name"]] = {{"kind": "bone", "cfg": _cf["entry"]["cfg"], "files": _cf["entry"]["files"]}}
_cfg["cont_fs"] = _cf["cont_fs"]
if SWITCHES["frac_mfs"]:          # oct19_probe_frac_mfs
    _mf = json.load(open(MFM + "/member.json"))
    _link(MFM, _mf["entry"]["files"])
    assert _mf["name"] not in _cfg["fc"]["members"] and "frac_mfs" not in _cfg and _mf["entry"]["kind"] == "fsmain", "frac_mfs config"
    _cfg["fc"]["members"][_mf["name"]] = {{"kind": "fsmain", "cfg": _mf["entry"]["cfg"], "files": _mf["entry"]["files"]}}
    _cfg["frac_mfs"] = _mf["frac_mfs"]
if SWITCHES["pf_q20"]:            # oct18_probe_pf_allup: PF from the fusion over ONLY its loa_set (members already in oa.members for Q15)
    assert set(PF_Q20["set"]) <= set(_cfg["oa"]["members"]) and PF_Q20["col"] == "PF OA", PF_Q20
    _cfg["oa"]["extra_sets"] = {{"pf": PF_Q20["set"]}}
    _cfg["oa"]["extra_align"] = {{"pf": PF_Q20["align"] | {{"col": PF_Q20["col"]}}}}
if SWITCHES["bakers_allpw"]:      # oct19_probe_bakers_allpw (Q25)
    _bk = json.load(open(BKM + "/member.json"))["bakers_allpw"]
    _link(BKM, _bk["main"] + [c for cs in _bk["basin"].values() for c in cs] + [c[:-3] + ".json" for cs in _bk["basin"].values() for c in cs])
    assert "bakers_allpw" not in _cfg and _bk["cols"] == ["Baker's"] and len(_bk["main"]) == 5 and all(len(v) == 5 for v in _bk["basin"].values())
    _cfg["bakers_allpw"] = {{"main": _bk["main"], "basin": _bk["basin"], "cols": _bk["cols"]}}
if SWITCHES["lm_allpw"]:          # C184 G1: LM's 3x3 base from Q25's pw generalists (pw main pass in its own weights dir + the pw Basin models)
    _lb = json.load(open(BKM + "/member.json"))["bakers_allpw"]
    _bf = [c for cs in _lb["basin"].values() for c in cs]
    _link(BKM, [f for f in _bf + [c[:-3] + ".json" for c in _bf] if not os.path.exists(os.path.join(WORK, f))])   # shared with bakers_allpw
    os.makedirs(WORK + "/main_pw", exist_ok=True)
    _link(BKM, _lb["main"], "main_pw")
    json.dump(json.load(open(PKG + "/config.json")) | {{"weights": _lb["main"]}}, open(WORK + "/main_pw/config.json", "w"))
    assert "lm_allpw" not in _cfg and len(_lb["main"]) == 5 and set(_lb["basin"]) == set(_cfg["basin"]) and all(len(v) == 5 for v in _lb["basin"].values())
    _cfg["lm_allpw"] = {{"main_dir": "main_pw", "basin": _lb["basin"], "col": "Lateral Meniscus"}}
json.dump(_cfg, open(WORK + "/config.json", "w"))
PKG = WORK
print("stack: syn members", len(_sm), "| mcl_sd", _cfg["mcl_sd"], "| mcl_member", _cfg["mcl_member"]["name"], "| mcl_member2", _cfg["mcl_member2"]["name"],
      "| acl_member", _cfg["acl_member"]["name"], "| acl_member2", _cfg["acl_member2"]["name"], "| syn main", _syn["pipelines"]["main"],
      "| loa_set", _cfg["oa"]["loa_set"], "| fc members", list(_cfg["fc"]["members"]), "| cont_fs", _cfg["cont_fs"]["name"], "| SWITCHES", SWITCHES,
      "| oa extra_sets", _cfg["oa"].get("extra_sets"), "| bakers_allpw", bool(_cfg.get("bakers_allpw")), "| lm_allpw", bool(_cfg.get("lm_allpw")), flush=True)
sys.path.insert(0, PKG)
import cv2, pydicom, timm
import ensemble
ki = ensemble.ki
for _m in (ensemble, ensemble.syn_stage, ensemble.oa_stage, ensemble.fc_stage, ki):     # the stack code is what runs (fail fast otherwise)
    assert os.path.dirname(os.path.abspath(_m.__file__)) == WORK, _m.__file__
_es = open(ensemble.__file__).read()
assert all(k in _es for k in ("mcl_member2", "mcl_sd", "acl_member2", "loa_extra_align", "eff_pipelines", "cont_fs", "frac_mfs", "extra_sets", "def basin_args", "lm_allpw")) \
    and "keep_bags" in open(ensemble.syn_stage.__file__).read() and "nofuse" in open(ensemble.fc_stage.__file__).read() \
    and "extra_sets" in open(ensemble.oa_stage.__file__).read()
print("stack modules loaded from", WORK, flush=True)
NCPU = len(os.sched_getaffinity(0))
print(f"python {{platform.python_version()}} torch {{torch.__version__}} timm {{timm.__version__}} pydicom {{pydicom.__version__}} cv2 {{cv2.__version__}} "
      f"numpy {{np.__version__}} pandas {{pd.__version__}} | cpus {{NCPU}} gpus {{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}} "
      f"| package {{PKG}} | weights {{json.load(open(PKG + '/config.json'))['weights']}} basin {{json.load(open(PKG + '/config.json'))['basin']}}", flush=True)

DEC = {dec!r}                                          # post-hoc de-coupling z' = z - b (c0 + c z_ctx): MCL Q6 RE-FITTED on the new Contusion (refit.json), LM Q22


def decouple(df):
    out = df.copy()
    for f, d in DEC.items():
        p, q = np.clip(df[f].to_numpy(np.float64), 1e-7, 1 - 1e-7), np.clip(df[d["context"]].to_numpy(np.float64), 1e-7, 1 - 1e-7)
        z, zc = np.log(p / (1 - p)), np.log(q / (1 - q))
        out[f] = 1 / (1 + np.exp(-(z - d["b"] * (d["c0"] + d["c"] * zc))))
    return out


test = pd.read_csv(COMP + "/test.csv")
t = time.time()
pred, parts, failed, secs = ensemble.predict(COMP + "/test_series", test.StudyInstanceUID.tolist(), pd.read_csv(COMP + "/test_series.csv"), PKG)
print(f"test: {{len(pred)}} studies in {{time.time() - t:.0f}}s, failed main {{len(failed)}}, {{ensemble.STATS}}", flush=True)
sub = pd.read_csv(COMP + "/sample_submission.csv")
assert set(sub.StudyInstanceUID) == set(test.StudyInstanceUID) and len(sub) == len(test)
sub[ki.FINDINGS] = pred.loc[sub.StudyInstanceUID, ki.FINDINGS].to_numpy()
sub[ki.FINDINGS] = decouple(sub[ki.FINDINGS])
print("de-coupled:", list(DEC), flush=True)
assert np.isfinite(sub[ki.FINDINGS].to_numpy()).all() and sub[ki.FINDINGS].to_numpy().min() >= 0 and sub[ki.FINDINGS].to_numpy().max() <= 1
KEEP = {keep!r}                                              # probe: every other finding is a constant 0.5 (AUC exactly 0.5)
if KEEP:
    sub[[c for c in ki.FINDINGS if c not in KEEP]] = 0.5
    assert (sub[[c for c in ki.FINDINGS if c not in KEEP]] == 0.5).all().all() and sub[KEEP].nunique().min() > 1
sub.to_csv("submission.csv", index=False)
print(f"submission.csv written, {{time.time() - T0:.0f}}s since start", flush=True)

if len(test) <= 3:  # commit run only: Kaggle-side predictions and timing on train studies
    chk = pd.read_csv(PKG + "/check_studies.csv")
    series = pd.read_csv(COMP + "/train_series.csv")
    timing, preds, all_parts = {{"ncpu": NCPU, "gpus": torch.cuda.device_count()}}, [], []
    for name, ids in [("main", chk.StudyInstanceUID[chk.group != "edge"].tolist()), ("edge", chk.StudyInstanceUID[chk.group == "edge"].tolist())]:
        t = time.time()
        p, ps, f, s = ensemble.predict(COMP + "/train_series", ids, series, PKG)
        timing[name] = {{"n": len(ids), "wall_s": time.time() - t, "cpu_s_per_study": float(s.mean()), "failed": f,
                        **{{k: ensemble.STATS.get(k) for k in ensemble.STATS}}}}      # every stage incl. main_sd
        preds.append(p)
        all_parts.append(pd.concat(ps, names=["part"]))
        print(name, timing[name], flush=True)
    pd.concat(preds).to_csv("check_pred_base.csv")
    decouple(pd.concat(preds)).to_csv("check_pred.csv")
    pd.concat(all_parts).to_csv("check_parts.csv")
    timing["files_per_study_main"] = float(np.mean([sum(1 for e in os.scandir(f"{{COMP}}/train_series/{{st}}") for _ in os.scandir(e.path))
                                                    for st in chk.StudyInstanceUID[chk.group != "edge"]]))
    timing["total_s"] = time.time() - T0
    json.dump(timing, open("check_timing.json", "w"), indent=1)
    print(timing, flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="knee-oct13-f5-win")
    ap.add_argument("--keep", nargs="*", default=[])
    ap.add_argument("--frac-mfs", action="store_true", help="SWITCH: Fracture + m_fs (oct19_probe_frac_mfs, 57399358)")
    ap.add_argument("--pf-q20", action="store_true", help="SWITCH: PF Q20 (oct18_probe_pf_allup, 57399364)")
    ap.add_argument("--bakers-allpw", action="store_true", help="SWITCH: Baker's all-pw Q25 (oct19_probe_bakers_allpw, 57399558)")
    ap.add_argument("--lm-allpw", action="store_true", help="SWITCH: LM's 3x3 base from the pw generalists (C184 G1; no probe of its own)")
    ap.add_argument("--p2", action="store_true", help="v1's P2 prune (Effusion img_eff from the ConvNeXt crops, C125 stats; NOT LB-tested); "
                    "default OFF = eff961 (the 0.961 Effusion)")
    a = ap.parse_args()
    refit = json.load(open(HERE / "refit.json"))
    dec = refit["dec"]
    assert set(dec) == {"MCL", "Lateral Meniscus"} and dec["MCL"]["context"] == "Contusion" and dec["Lateral Meniscus"]["context"] == "ACL", dec
    assert dec["Lateral Meniscus"] == json.load(open(SUB / "oct19_probe_acl_b3g_q8/dec.json"))["Lateral Meniscus"], "LM must keep Q22's constants"
    p2 = json.load(open(SUB.parent / "experiments/v3/oct16_experts/out/p2_eff_stats.json")) if a.p2 else None   # C125's re-fit (v2: OFF by default)
    assert p2 is None or set(p2) == {"mu_e", "sd_e"}, p2
    q20 = json.load(open(SUB / "oct18_probe_pf_allup/dataset/member.json"))
    pf_q20 = {"set": q20["loa_set"], "align": q20["align"], "col": q20["col"]}
    sw = {"frac_mfs": a.frac_mfs, "pf_q20": a.pf_q20, "bakers_allpw": a.bakers_allpw, "lm_allpw": a.lm_allpw}
    tag = ("_p2" if a.p2 else "") + "".join(f"_{s}" for k, s in (("frac_mfs", "mfs"), ("pf_q20", "q20"), ("bakers_allpw", "q25"), ("lm_allpw", "lpw")) if sw[k])   # Kaggle titles <= 50 chars
    out = HERE / ("kernel" + tag)
    out.mkdir(exist_ok=True)
    slugs = dict(zslug="knee-oct17-mcl-zoom", sslug="knee-oct18-mcl-sd2", aslug="knee-oct18-acl-b3g", lslug="knee-oct18-loa-allup",
                 cslug="knee-oct16-syn-cnx-probe", cfslug="knee-oct17-cont-fs", mfslug="knee-oct19-frac-mfs", bkslug="knee-oct19-bakers-allpw")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(CODE.format(
        slug=a.slug, keep=a.keep, dec=dec, p1=True, p2=p2, stack=json.load(open(SUB / "oct18_probe_mcl_stack/stack.json")), switches=sw,
        pf_q20=pf_q20, **slugs, syn_src=(HERE / "syn_stage.py").read_text(), ens_src=(HERE / "ensemble.py").read_text(),
        oa_src=(HERE / "oa_stage.py").read_text(), fc_src=(HERE / "fc_stage.py").read_text(),
        chk_src=(HERE / "check_studies.csv").read_text()).strip())],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    name = "knee-oct20-stack-all-v2" + tag.replace("_", "-") + "-sub"
    nbformat.write(nb, out / f"{name}.ipynb")
    meta = {"id": f"pavelsavchenkov/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [f"pavelsavchenkov/{a.slug}"] + [f"pavelsavchenkov/{s}" for s in slugs.values()],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
    print(out / f"{name}.ipynb")


if __name__ == "__main__":
    main()
