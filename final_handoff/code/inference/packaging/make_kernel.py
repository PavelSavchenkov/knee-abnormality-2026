"""FINAL all-data kernel per variant (Master 10-21 14:3x; OAI: no; nothing pushed, nothing uploaded). kernel_<variant>/ = the LB-scored
sxm kit notebook (oct21_probe_cont_sxm/kernel = the all-4 notebook 57455847 + cont_sxm: FAST two-GPU runner, STRICT, the 8-study commit check
ON) with ONE change: its setup region (PKG = knee-oct13-f5-win + the WORK assembly from 10 member datasets + the embedded PATCHED code) is
replaced by the FINAL setup:
  - PKG = the variant's flat dataset knee-oct21-final-<variant> (build.py package_<variant>/), nothing else mounted;
  - every package file md5-asserted against the locally AUDITED build (EXPECT table embedded here; audit_<variant>.json PASS for this
    config.json required) + all_data.runs / sources consistency;
  - Kaggle datasets are flat, so WORK = a mirror (code copied, weights symlinked) + main_sd/ rebuilt from main_sd_config.json (the
    config's mcl_sd main_dir); config / code used unchanged;
  - RUNTIME print (models per stage), the module-origin + code-feature asserts.
Everything after the setup (versions print, DEC = v2's constants (asserted), fast.py, strict_check, fast_predict_all, the commit check, the
submission) is the sxm kit's text, unchanged.
conda run -n rsna python submissions/oct21_final_alldata/make_kernel.py --variant NAME"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402

SRC_KIT = B.SUB / "oct21_probe_cont_sxm/kernel"
A = "SLUG = 'knee-oct13-f5-win'\n"
Z = 'print("stack modules loaded from", WORK, flush=True)\n'
SETUP = r'''SLUGS = {slugs!r}   # the variant's datasets: frozen WEIGHTS datasets (union) + its small CONFIG dataset (last: code + config.json)
INPUT = os.environ.get("KNEE_INPUT", "/kaggle/input")  # local dry run: a folder laid out like /kaggle/input
COMP = next(p for p in [INPUT + "/competitions/rsna-knee-abnormality-detection", INPUT + "/rsna-knee-abnormality-detection"]
            if os.path.exists(p + "/test.csv"))
# oct21_final_alldata (FINAL all-data package, variant {variant}; OAI: no): the weights live in frozen datasets (uploaded once, shared by
# every variant), code + config in the variant's config dataset. Every dataset's file set and every file's md5 are checked against the
# locally audited build (make_kernel.py: audit_{variant}.json PASS for this config.json) before anything runs; a file name may live in
# ONE dataset only.
import hashlib, shutil
MD5_MODE = {md5_modes!r}   # per dataset: 'full' = md5 of every file; 'ends' = big files: size + md5 of the first / last 4 MB (--md5 / --md5-light)
EXPECT = {expect!r}
{fp_src}


def _mount(slug):
    hits = sorted({{p for pat in ("/*", "/*/*", "/*/*/*") for p in glob.glob(INPUT + pat) if os.path.isdir(p) and os.path.basename(p) == slug}})
    assert len(hits) == 1, f"dataset {{slug}}: mount dirs {{hits}}"
    return hits[0]


MOUNT = {{s: _mount(s) for s in SLUGS}}
_names = [f for s in SLUGS for f in EXPECT[s]]
assert list(EXPECT) == SLUGS and len(_names) == len(set(_names)), "a file name in two datasets"
_t5 = time.time()
for _s in SLUGS:
    _have = set(os.listdir(MOUNT[_s])) - {{"__pycache__", "dataset-metadata.json"}}
    assert _have == set(EXPECT[_s]), f"{{_s}}: files != the audited build: extra {{sorted(_have - set(EXPECT[_s]))[:5]}}, missing {{sorted(set(EXPECT[_s]) - _have)[:5]}}"
    _bad = [f for f in sorted(EXPECT[_s]) if _fp(os.path.join(MOUNT[_s], f), MD5_MODE[_s]) != EXPECT[_s][f]]
    assert not _bad, f"package md5 != the audited build ({{_s}}): {{_bad[:10]}}"
MD5_S = time.time() - _t5
CFGD = MOUNT[SLUGS[-1]]
_cfg = json.load(open(CFGD + "/config.json"))
_ad = _cfg["all_data"]
assert _ad["variant"] == {variant!r} and _ad["datasets"] == SLUGS and set(_ad["runs"]) == set(_ad["sources"]) and set(_ad["runs"]) <= set(_names), \
    "all_data variant / datasets / runs"
_seeds = {{r: sorted({{str(x["seed"]) for x in _ad["runs"].values() if x["recipe"] == r}}) for r in sorted({{x["recipe"] for x in _ad["runs"].values()}})}}
print(f"FINAL package {variant}: {{len(SLUGS)}} datasets {{SLUGS}}, {{len(_names)}} files md5 == the audited build (MD5 {{MD5_MODE}} wall {{MD5_S:.0f}} s, "
      f"{{sum(os.path.getsize(os.path.join(MOUNT[s], f)) for s in SLUGS for f in EXPECT[s]) / 1e9:.1f}} GB); form {{_ad['form']}}; seeds per recipe "
      f"{{_seeds}}", flush=True)
# WORK = the union of the datasets (code copied so knee_infer's sys.path stays in WORK, the rest symlinked) + the MCL SD-2 main pass dir
# (config mcl_sd main_dir) rebuilt from main_sd_config.json; config and code are used unchanged.
WORK = "/kaggle/working/pkg_final" if os.path.isdir("/kaggle/working") else os.path.join(os.environ.get("TMPDIR", "/tmp"), "pkg_final")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(os.path.join(WORK, _cfg["mcl_sd"]["main_dir"]))
for _s in SLUGS:
    for fn in EXPECT[_s]:
        (shutil.copy if fn.endswith(".py") else os.symlink)(os.path.join(MOUNT[_s], fn), os.path.join(WORK, fn))
_sdc = json.load(open(CFGD + "/main_sd_config.json"))
json.dump(_sdc, open(os.path.join(WORK, _cfg["mcl_sd"]["main_dir"], "config.json"), "w"))
for fn in _sdc["weights"]:
    os.symlink(os.path.realpath(os.path.join(WORK, fn)), os.path.join(WORK, _cfg["mcl_sd"]["main_dir"], fn))
PKG = WORK
_sm = json.load(open(PKG + "/" + _cfg["syn"]["members"]))
_em = _cfg.get("eff_main") or {{}}
print("RUNTIME models per stage:", {{"main": len(_cfg["weights"]), "main_sd": len(_sdc["weights"]),
      "basin": len(set(sum(list(_cfg["basin"].values()) + list(_cfg["basin_syn"].values()) + [_cfg["mcl_sd"]["nfnet"]] + list(_em.get("basin", {{}}).values()), []))),
      "win_main": len(_cfg["win"]["main"]) + len(_em.get("main", [])), "oa": sum(len(m["files"]) for m in _cfg["oa"]["members"].values()),
      "fc": sum(len(m["files"]) for m in _cfg["fc"]["members"].values()),
      "syn": sum(len(m["files"]) for m in _sm.values()) + len((_cfg.get("cont_sxm") or {{}}).get("files", []))}}, "| fc members", list(_cfg["fc"]["members"]),
      "| fc_shipped_off", bool(_cfg.get("fc_shipped_off")), "| frac_allmem", bool(_cfg.get("frac_allmem")), "| eff_main", bool(_em),
      "| cont_sxm", bool(_cfg.get("cont_sxm")), flush=True)
sys.path.insert(0, PKG)
import cv2, pydicom, timm
import ensemble
ki = ensemble.ki
for _m in (ensemble, ensemble.syn_stage, ensemble.oa_stage, ensemble.fc_stage, ki):     # the final package's code is what runs (fail fast otherwise)
    assert os.path.dirname(os.path.abspath(_m.__file__)) == WORK, _m.__file__
_es = open(ensemble.__file__).read()
assert all(k in _es for k in ("mcl_member2", "mcl_sd", "acl_member2", "loa_extra_align", "eff_pipelines", "cont_fs", "cont_allmem", "frac_allmem",
                              "frac_fs", "eff_main", "cont_sxm", "fc_shipped_off", "def basin_args")) \
    and "keep_bags" in open(ensemble.syn_stage.__file__).read() and "def member_sets" in open(ensemble.fc_stage.__file__).read()
print("final modules loaded from", WORK, flush=True)
'''


FP_SRC = r'''

def _fp(p, mode):
    """'full' = md5 of the file; 'ends' = a file > 64 MB: its size + md5 of its first and last 4 MB (smaller files: the full md5)."""
    n = os.path.getsize(p)
    h = hashlib.md5()
    with open(p, "rb") as f:
        if mode == "ends" and n > 64 << 20:
            h.update(f.read(4 << 20))
            f.seek(n - (4 << 20))
            h.update(f.read(4 << 20))
            return f"{n}:{h.hexdigest()}"
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
'''
_ns = {"os": __import__("os"), "hashlib": hashlib}
exec(FP_SRC, _ns)
fp = _ns["_fp"]                            # the kernel's own fingerprint function (one source for EXPECT and the Kaggle check)


def dec_of(src):
    m = re.findall(r"^DEC = (\{.*?\})\s+#", src, re.M)
    assert len(m) == 1, len(m)
    return ast.literal_eval(m[0])


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


LOCAL_COPY = r'''LOCAL = os.environ.get("KNEE_LOCAL", "/tmp/pkgw")   # --local-copy (Submission Timing 21:1x): every package file copied to local disk with its
shutil.rmtree(LOCAL, ignore_errors=True)                # md5 computed on the fly (ONE read of gcsfuse, full md5 of everything); WORK links there,
os.makedirs(LOCAL)                                      # so every (re)load reads local disk (~300+ MB/s) instead of gcsfuse (~5 MB/s cold)
from concurrent.futures import ThreadPoolExecutor


def _cp(job):
    src, dst = job
    h = hashlib.md5()
    with open(src, "rb", buffering=0) as fi, open(dst, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 24), b""):
            h.update(b)
            fo.write(b)
    return os.path.basename(dst), h.hexdigest()


with ThreadPoolExecutor(16) as _ex:
    _got = dict(_ex.map(_cp, [(os.path.join(MOUNT[s], f), os.path.join(LOCAL, f)) for s in SLUGS for f in EXPECT[s]]))
_bad = [f for s in SLUGS for f in EXPECT[s] if _got[f] != EXPECT[s][f]]
assert not _bad, f"package md5 != the audited build (local copy): {_bad[:10]}"
MOUNT_SRC = {s: LOCAL for s in SLUGS}
'''


BASIN_OLD = '    def basin():\n        ckpts, det, win_kw = H.basin_args(cfg, pkg)\n        z = bp.predict(series, root, ckpts, workers=workers, detector=det, win=win_kw) if len(series) else {}\n        return z, dict(bp.WIN), dict(bp.STATS)\n'
BASIN_NEW = '    def basin():\n        ckpts, det, win_kw = H.basin_args(cfg, pkg)\n        if not len(series):\n            return {}, dict(bp.WIN), dict(bp.STATS)\n        import dicom_prefetch as DP   # --readahead (Master 10-22, safe3w9ra): Basin\'s upcoming studies read into the page cache by a\n        pf = DP.start(root, list(dict.fromkeys(series.StudyInstanceUID)), threads=64, ahead=24, tag=f"basin_{J[\'name\']}")   # separate\n        unhook = DP.hook_getitem(bp.DicomStudies, pf)   # process (exact: reads only); hooked BEFORE Basin\'s DataLoader forks\n        try:\n            z = bp.predict(series, root, ckpts, workers=workers, detector=det, win=win_kw)\n        finally:\n            unhook()\n            st = DP.stop(pf)   # performance-only: a dead prefetcher = Basin reads as before (same outputs), logged in T\n            T.update({f"pf_{k}": float(v) for k, v in st.items() if isinstance(v, (int, float))})\n            print(f"READAHEAD {J[\'name\']}: {st}", flush=True)\n        return z, dict(bp.WIN), dict(bp.STATS)\n'


W_OLD = '    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)\n    import torch\n    T = {"t_enter": t_enter,'
W_NEW = '    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)\n    import infer_prep as _ip   # --ra2 (A) (Master / Submission Timing 10-22): Basin\'s per-file page-cache eviction OFF, so the crop stage and\n    _ip._read = lambda f, **kw: _ip.pydicom.dcmread(f, **kw)   # main reuse its reads (exact: the same dcmread, only POSIX_FADV_DONTNEED dropped)\n    import torch\n    T = {"t_enter": t_enter,'

def render(v, pkg, md5_mode="full", md5_light=(), quick=0, prewarm=0, local_copy=False, stage=False, readahead=False, ra2=0):
    """-> (notebook, kernel metadata, slugs, EXPECT) of variant v's kernel; pkg = its union mirror (build.py pkg_<v>/), whose config.json
    lists the datasets (registry reg_<name>/ds/<slug>/). md5_light: datasets fingerprinted 'ends' (e.g. an already uploaded + verified base);
    quick = N > 3: the COMMIT check runs N edge studies in ONE FAST chunk (default: the kit's 8 studies in 2 chunks); the scoring path
    (FAST_CHUNKS 4) is untouched."""
    ad = json.load(open(pkg / "config.json"))["all_data"]
    slugs, ds = ad["datasets"], HERE / f"reg_{ad['registry']}" / "ds"
    assert set(md5_light) <= set(slugs), (md5_light, slugs)
    modes = {s_: "ends" if s_ in md5_light else md5_mode for s_ in slugs}
    expect = {}
    for s_ in slugs:
        d_ = ds / s_
        assert not [p for p in d_.iterdir() if p.is_dir()], f"{d_}: subdirectory"
        expect[s_] = {p.name: fp(str(p), modes[s_]) for p in sorted(d_.iterdir()) if p.name != "dataset-metadata.json"}
    nb = nbformat.read(open(next(SRC_KIT.glob("*.ipynb"))), 4)
    assert len(nb.cells) == 1
    src = nb.cells[0].source
    assert src.count(A) == 1 and src.count(Z) == 1 and src.index(A) < src.index(Z)
    v2 = nbformat.read(open(next((B.SUB / "oct20_stack_all_v2/kernel").glob("*.ipynb"))), 4).cells[0].source
    assert dec_of(src) == dec_of(v2), "DEC must be v2's"
    assert "if len(test) <= 3:   # commit run only" in src, "the 8-study commit check must be ON in the source kernel"
    slug = f"knee-oct21-fa-{B.tag(v).replace('_', '-')}"
    src = src[:src.index(A)] + SETUP.format(slugs=slugs, variant=v, expect=expect, md5_modes=modes, fp_src=FP_SRC) + src[src.index(Z) + len(Z):]
    if quick:                                            # Master 19:1x (USER: quicker commit): one FAST chunk of N > 3 edge studies (FAST path kept)
        assert quick > 3, "quick N must be > 3: <= 3 studies take the sequential path, the FAST path would go untested"
        for a_, b_ in (('CHK_IDS = chk.StudyInstanceUID[chk.group == "edge"].tolist()[:8]',
                        f'CHK_IDS = chk.StudyInstanceUID[chk.group == "edge"].tolist()[:{quick}]   # QUICK commit (make_kernel --quick {quick})'),
                       ('pd.read_csv(COMP + "/train_series.csv"), n_chunks=2)', 'pd.read_csv(COMP + "/train_series.csv"), n_chunks=1)')):
            assert src.count(a_) == 1, a_
            src = src.replace(a_, b_)
    assert "FAST_CHUNKS, FAST_PAD_LOC = 4, 32" in src, "the scoring path's FAST_CHUNKS must stay 4"
    if local_copy:                                       # md5 during the local copy (one pass): EXPECT must be the FULL md5 of every file
        assert md5_mode == "full" and not md5_light, "--local-copy verifies full md5 while copying: no --md5 ends / --md5-light"
        for a_, b_ in (('    _bad = [f for f in sorted(EXPECT[_s]) if _fp(os.path.join(MOUNT[_s], f), MD5_MODE[_s]) != EXPECT[_s][f]]\n'
                        '    assert not _bad, f"package md5 != the audited build ({_s}): {_bad[:10]}"\n', ''),
                       ('MD5_S = time.time() - _t5\n', LOCAL_COPY + 'MD5_S = time.time() - _t5\n'),
                       ('(shutil.copy if fn.endswith(".py") else os.symlink)(os.path.join(MOUNT[_s], fn), os.path.join(WORK, fn))',
                        '(shutil.copy if fn.endswith(".py") else os.symlink)(os.path.join(MOUNT_SRC[_s], fn), os.path.join(WORK, fn))')):
            assert src.count(a_) == 1, a_[:80]
            src = src.replace(a_, b_)
    if prewarm:                                          # Master 21:0x (Submission Timing): page-cache PREWARM = read every package file once in parallel
        pw = (HERE / "pkg_code/prewarm.py").read_text()  # 16 MB blocks (exact: changes nothing) at start + before every FAST chunk (gcsfuse cold
        a1 = 'sys.path.insert(0, "/kaggle/working")\nimport fast\n'   # torch.load ~3 MB/s vs ~500 MB/s from page cache; sxm 57480087 = 499 min)
        b1 = (a1 + f'open("/kaggle/working/prewarm.py", "w").write({pw!r})\nimport prewarm as _pw\nPREWARM_THREADS = {prewarm}\n'
              'print("PREWARM start:", _pw.prewarm(_pw.package_files(PKG), PREWARM_THREADS), flush=True)\n')
        a2 = "        ensemble.STATS.clear()\n        r = fast.predict("
        b2 = '        print(f"PREWARM chunk {i}:", _pw.prewarm(_pw.package_files(PKG), PREWARM_THREADS), flush=True)\n' + a2
        for a_, b_ in ((a1, b1), (a2, b2)):
            assert src.count(a_) == 1, a_
            src = src.replace(a_, b_)
    if stage:                                            # Master 10-22 08:2x (Submission Timing): per-chunk DICOM STAGING to local disk
        assert not prewarm, "--stage and --prewarm both patch the FAST driver: one at a time"   # (pkg_code/dicom_stage.py): each FAST chunk's
        ds = (HERE / "pkg_code/dicom_stage.py").read_text()                                   # studies copied to /tmp/stage/c<i> (64 threads,
        a1 = 'sys.path.insert(0, "/kaggle/working")\nimport fast\n'                           # strict tree check), chunk i+1 in a background
        b1 = a1 + f'open("/kaggle/working/dicom_stage.py", "w").write({ds!r})\nimport dicom_stage as _ds\nSTAGE = "/tmp/stage"\n'   # process
        a2 = ('    R, LOG = [], []\n    for i, ch in enumerate(chunks):\n        t = time.time()\n        ensemble.STATS.clear()\n'
              '        r = fast.predict(root, ch, series, PKG, ')                               # during chunk i; every stage reads the copy
        b2 = ('    R, LOG = [], []\n    job = _ds.start(root, chunks[0], f"{STAGE}/c0")   # --stage: chunk 0 (waited for below)\n'
              '    for i, ch in enumerate(chunks):\n        t = time.time()\n'
              '        SST = _ds.finish(job)                                   # raises if the copy failed: no network fallback\n'
              '        job = _ds.start(root, chunks[i + 1], f"{STAGE}/c{i + 1}") if i + 1 < len(chunks) else None   # next chunk, background\n'
              '        print(f"STAGE chunk {i}: {SST}, /tmp free {_ds.free_gb(\'/tmp\')} GB", flush=True)\n'
              '        ensemble.STATS.clear()\n'
              '        r = fast.predict(f"{STAGE}/c{i}", ch, series, PKG, ')
        a3 = '        LOG.append({"chunk": i, "n": len(ch), "mode": "fast", "wall_s": time.time() - t, '
        b3 = '        LOG.append({"chunk": i, "n": len(ch), "mode": "fast", "stage": SST, "wall_s": time.time() - t, '
        a4 = '        gc.collect()\n        fast.release()\n    pred = pd.concat([r[0] for r in R])'
        b4 = '        gc.collect()\n        fast.release()\n        _ds.remove(f"{STAGE}/c{i}")                             # before chunk i+2 is staged\n    pred = pd.concat([r[0] for r in R])'
        for a_, b_ in ((a1, b1), (a2, b2), (a3, b3), (a4, b4)):
            assert src.count(a_) == 1, a_[:80]
            src = src.replace(a_, b_)
        assert src.count("fast.predict(root,") == 0 and src.count('fast.predict(f"{STAGE}/c{i}",') == 1, "every FAST chunk must read the stage root"
    if (pkg / "main_pl_config.json").exists():          # oct22 routes: the PL main set's weights dir rebuilt like main_sd (routed variants only)
        a_ = "PKG = WORK\n"
        b_ = ('if True:   # oct22 routes: WORK/<main_pl main_dir> from main_pl_config.json (the PL main set, same pass as main)\n'
              '    _mpc = json.load(open(CFGD + "/main_pl_config.json"))\n'
              '    os.makedirs(os.path.join(WORK, _cfg["main_pl"]["main_dir"]))\n'
              '    json.dump(_mpc, open(os.path.join(WORK, _cfg["main_pl"]["main_dir"], "config.json"), "w"))\n'
              '    for fn in _mpc["weights"]:\n'
              '        os.symlink(os.path.realpath(os.path.join(WORK, fn)), os.path.join(WORK, _cfg["main_pl"]["main_dir"], fn))\n'
              '    print("main_pl:", len(_mpc["weights"]), "models, route", _cfg["main_pl"]["route"], flush=True)\n' + a_)
        assert src.count(a_) == 1, "SETUP anchor"
        src = src.replace(a_, b_)
    if readahead:                                        # Master 10-22 08:3x (USER vetoed disk staging): in-RAM Basin READ-AHEAD
        assert not stage, "--readahead and --stage: one at a time"   # (pkg_code/dicom_prefetch.py, Submission Timing's reference + stats)
        a0 = 'open("/kaggle/working/fast.py", "w").write('
        i0 = src.index(a0)
        assert src.count(a0) == 1
        line = src[i0:src.index("\n", i0)]
        code = ast.literal_eval(line[len(a0):-1])
        assert code.count(BASIN_OLD) == 1, "fast.py worker basin() closure not found"
        code = code.replace(BASIN_OLD, BASIN_NEW)
        pf_src = (HERE / "pkg_code/dicom_prefetch.py").read_text()
        src = src.replace(line, a0 + repr(code) + ")\n" + 'open("/kaggle/working/dicom_prefetch.py", "w").write(' + repr(pf_src) + ")")
    if ra2:                                              # Master 10-22 08:3x UTC (slow VMs are throughput-capped): read each study ONCE
        assert readahead, "--ra2 builds on --readahead"
        assert ra2 in (8, 16, 32), "FAST_CHUNKS: the driver halves chunks until >= N, so only powers of 2 are exact"
        a0 = 'open("/kaggle/working/fast.py", "w").write('
        i0 = src.index(a0)
        line = src[i0:src.index("\n", i0)]
        code = ast.literal_eval(line[len(a0):-1])
        assert code.count(W_OLD) == 1, "fast.py worker anchor"
        src = src.replace(line, a0 + repr(code.replace(W_OLD, W_NEW)) + ")")
        a_ = "FAST_CHUNKS, FAST_PAD_LOC = 4, 32"
        assert src.count(a_) == 1
        src = src.replace(a_, f"FAST_CHUNKS, FAST_PAD_LOC = {ra2}, 32   # --ra2 (B): one chunk's DICOM fits the free page cache (~115 MB / study)")
    for gone in ("PATCHED", "_find(", "ZMEM", "SWITCHES", "B3G_SHARED"):
        assert gone not in src, f"{gone} left in the final kernel"
    chk = HERE / f"_code_check_{v}.py"
    chk.write_text(src)
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(chk)], capture_output=True, text=True)
    chk.unlink()
    assert r.returncode in (0, 1) and "No module named" not in r.stderr, f"pyflakes did not run: {r.stderr[:300]}"
    undefined = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
    assert not undefined, undefined
    nb.cells[0].source = src
    meta = json.load(open(SRC_KIT / "kernel-metadata.json"))
    assert meta["is_private"] and not meta["enable_internet"] and meta["machine_shape"] == "NvidiaTeslaT4"
    meta |= {"id": f"pavelsavchenkov/{slug}-sub", "title": f"{slug}-sub", "code_file": f"{slug}-sub.ipynb",
             "dataset_sources": [f"pavelsavchenkov/{s_}" for s_ in slugs]}       # = exactly the variant's datasets
    return nb, meta, slug, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(B.VARIANTS))
    ap.add_argument("--md5", default="full", choices=["full", "ends"], help="Kaggle-side file check (Master 14:42: 'ends' if full > 5 min)")
    ap.add_argument("--md5-light", default="", help="comma list of datasets fingerprinted 'ends' (already uploaded + verified, e.g. knee-oct21-fa-base)")
    ap.add_argument("--quick", type=int, default=0, help="commit check = N (> 3) edge studies in ONE FAST chunk (default: 8 in 2 chunks)")
    ap.add_argument("--prewarm", type=int, default=0, help="page-cache prewarm threads (0 = off): start + before every FAST chunk")
    ap.add_argument("--local-copy", action="store_true", help="copy every package file to /tmp/pkgw (md5 on the fly) and run from there")
    ap.add_argument("--stage", action="store_true", help="per-chunk DICOM staging to /tmp/stage (pkg_code/dicom_stage.py; Master 10-22)")
    ap.add_argument("--readahead", action="store_true", help="in-RAM Basin read-ahead in the FAST workers (pkg_code/dicom_prefetch.py; Master 10-22)")
    ap.add_argument("--ra2", type=int, default=0, help="read-once render on top of --readahead: Basin eviction OFF in the FAST workers + "
                    "FAST_CHUNKS = N (power of 2)")
    ap.add_argument("--slug-suffix", default="", help="a second render of the SAME variant + cfg dataset under its own kernel slug (e.g. io)")
    a = ap.parse_args()
    v = a.variant
    pkg = HERE / f"pkg_{B.tag(v)}"
    au = json.load(open(HERE / f"audit_{B.tag(v)}.json"))
    assert au["pass"] and au["config_md5"] == md5(pkg / "config.json") and au["datasets"] == json.load(open(pkg / "config.json"))["all_data"]["datasets"], \
        "audit.py must PASS on this very build first"
    nb, meta, slug, expect = render(v, pkg, a.md5, [x for x in a.md5_light.split(",") if x], a.quick, a.prewarm, a.local_copy, a.stage, a.readahead, a.ra2)
    if a.slug_suffix:                                    # same datasets (incl. the variant's cfg dataset), own kernel id / notebook name
        assert a.slug_suffix.isalnum() and a.slug_suffix.islower(), a.slug_suffix
        old = f"{slug}-sub"
        slug = f"{slug}{a.slug_suffix}"
        meta |= {"id": f"pavelsavchenkov/{slug}-sub", "title": f"{slug}-sub", "code_file": f"{slug}-sub.ipynb"}
        assert old not in meta["id"] or a.slug_suffix in meta["id"]
    out = HERE / f"kernel_{B.tag(v)}{a.slug_suffix}"
    out.mkdir(exist_ok=True)
    nbformat.write(nb, open(out / f"{slug}-sub.ipynb", "w"))
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
    print(f"{out.name}/: {slug}-sub on datasets {meta['dataset_sources']} ({sum(map(len, expect.values()))} files md5-pinned; audit {au['date']}); "
          "pyflakes: no undefined names; NOT pushed, nothing uploaded")


if __name__ == "__main__":
    main()
