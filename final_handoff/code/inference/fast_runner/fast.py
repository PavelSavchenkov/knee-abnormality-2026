"""Two-GPU runner, port to the merged stacked fold package submissions/oct20_stack_all (from submissions/oct19_f5_win_fast, which was
bit-identical to the 0.961 package on Kaggle). The workers take their stage arguments from the package's ensemble.py hooks
(basin_args / oa_cfg / fc_cfg / syn_members: the same functions ensemble.predict calls), so every switch (frac_mfs / pf_q20 / bakers_allpw)
gives identical stage calls. Dev runs against oct19_stack_fold use a hook shim (env FAST_HOOKS). Serial test mode (env FAST_SERIAL=1):
the two workers run one after another on GPU 0 (local parity on a 1-GPU machine; same outputs as two GPUs of one type).

Original header (0.961 runner): Predictions = the package's own.

The package runs its stages one after another, and four of them use one T4 only (Basin, OA experts, bone experts, synovitis crops: ~90 % of
its 262 min). Here two worker processes (CUDA_VISIBLE_DEVICES 0 / 1) each run those four stages on half of the studies, at the same time.
The halves follow Basin's study order (the series csv order) and split at an even index, so Basin's 2-study batches are the package's.
knee_infer (main) and the window stage already use both GPUs: they stay in the parent. The parent then calls the package's unchanged
ensemble.predict with the four stage functions returning the workers' merged outputs, so the combination code is the package's own.
One difference by construction: Basin's fallback for a study without any readable series (cohort mean logit) is the mean over that
worker's half instead of over all studies (never triggered on the 275 check studies of 10-13).

Profiling (prof=True): every stage call's start / end (parent and workers), and the time inside the crop stages' building blocks
(knee_infer.study_series in the DataLoader workers, localise, study_bags, study_tensors, correct_sides, expert / fsmain logits);
read with read_prof(). Timing wrappers only: no change to any call's arguments or results.

RAM (10-19, after the ON bench's host OOM): every worker stage starts and ends with release() (gc, CUDA cache, pinned-host cache, glibc
arenas; RSS logged in the worker timings), the crop loop releases the pinned-host cache every RELEASE_EVERY studies, pad_loc 32 is the
default, and every predict() call spawns fresh worker processes that exit at its end (all their memory returns to the OS).

  worker: python fast.py worker PKG ROOT JOB.pkl OUT.pkl"""
import contextlib
import json
import os
import pickle
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

STAGE = ["parent"]          # current stage label; DataLoader workers inherit it at fork
PROF_DIR = os.environ.get("KNEE_PROF_DIR", "/kaggle/working/prof")
STAGE_LOG = []              # (process, stage, t_start, t_end)
RELEASE_EVERY = 16          # crop loop: return the pinned-host cache / glibc arenas every N studies (10-19 RAM fix; no effect on outputs)
PARENT_RELEASE_EVERY = 16   # parent: the same for knee_infer's main passes (10-19 Kaggle parent_mem: 25 GB of cached pinned study buffers on 325 studies)


def rss_gb():
    return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 2**20


@contextlib.contextmanager
def parent_release(ki, every=PARENT_RELEASE_EVERY):
    """Parent-side pinned-cache release (10-19; Kaggle parent_mem: knee_infer's cached pinned study buffers reached 25 GB in one main pass of
    325 heavy studies, all returned by the package's own _host_emptyCache AFTER the pass). Every `every` knee_infer.load_study calls (they run
    in knee_infer's loader threads) the cache's FREE pinned blocks are returned; blocks in use or with pending copy events stay (the host
    allocator is mutex-protected and frees only unused, event-complete blocks). Memory only: outputs unchanged."""
    import torch
    orig, n, lock = ki.load_study, [0], threading.Lock()

    def wrapped(*a, **k):
        r = orig(*a, **k)
        with lock:
            n[0] += 1
            go = every and n[0] % every == 0
        if go and torch.cuda.is_available():
            torch._C._host_emptyCache()
        return r
    ki.load_study = wrapped
    try:
        yield n
    finally:
        ki.load_study = orig


def release(full=True):
    """Return freed memory (10-19 RAM fix after the ON bench's host OOM; no effect on any output): Python garbage (full), the CUDA caching
    allocator (full), the pinned-host cache (grows with every new pinned buffer size: knee_infer pins each study's DICOM bytes) and glibc's
    malloc arenas."""
    import ctypes
    import gc
    import torch
    if full:
        gc.collect()
    if torch.cuda.is_available():
        if full:
            torch.cuda.empty_cache()
        torch._C._host_emptyCache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
CROP_FUNCS = ["localise", "study_bags", "study_tensors", "correct_sides", "expert_logits", "fsmain_logits"]


def _mods(pkg):
    sys.path.insert(0, pkg)
    import basin_predict as bp
    import fc_stage
    import knee_infer as ki
    import oa_stage
    import syn_stage
    import win_stage
    return bp, ki, oa_stage, fc_stage, syn_stage, win_stage


def _hooks(pkg):
    """The package's stage-argument hooks (ensemble.py). FAST_HOOKS=path/to/shim.py only for dev against a package without them."""
    sys.path.insert(0, pkg)
    import importlib.util
    shim = os.environ.get("FAST_HOOKS")
    if shim:
        spec = importlib.util.spec_from_file_location("fast_hooks", shim)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    import ensemble
    assert all(hasattr(ensemble, h) for h in ("basin_args", "oa_cfg", "fc_cfg", "syn_members")), "ensemble.py lacks the stage hooks"
    return ensemble


def _append(name, sec):
    os.makedirs(PROF_DIR, exist_ok=True)
    with open(f"{PROF_DIR}/prof_{os.getpid()}.tsv", "a") as fh:
        fh.write(f"{STAGE[0]}\t{name}\t{sec:.6f}\n")


def _timed(fn, name):
    def w(*a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            _append(name, time.time() - t)
    w.__wrapped__ = fn
    return w


def _stage(fn, label, proc):
    def w(*a, **k):
        prev, STAGE[0] = STAGE[0], label
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            STAGE_LOG.append((proc, label, t, time.time()))
            STAGE[0] = prev
    w.__wrapped__ = fn
    return w


def _timed_sync(fn, name):
    """Timing wrapper that synchronises the GPU before and after (profiling runs only: adds two syncs per call)."""
    import torch

    def w(*a, **k):
        torch.cuda.synchronize()
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            _append(name, time.time() - t)
    w.__wrapped__ = fn
    return w


def install_prof(pkg, proc, sync=False):
    """Timing wrappers on the package modules (idempotent). sync: also Basin's gpu_views / export_windows / model forwards with GPU
    syncs (their GPU time; changes scheduling, not results)."""
    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)
    if sync and getattr(bp.export_windows, "__wrapped__", None) is None:
        bp.export_windows = _timed_sync(bp.export_windows, "basin.export_windows")
        bp.gpu_views = _timed_sync(bp.gpu_views, "basin.gpu_views")
        orig_load = bp.load_model

        def load_model(ckpt):
            m, cfg = orig_load(ckpt)
            m.forward = _timed_sync(m.forward, "basin.model_forward")
            return m, cfg
        bp.load_model = load_model
    if getattr(ki.study_series, "__wrapped__", None) is None:
        ki.study_series = _timed(ki.study_series, "study_series")
        for mod in (oa_stage, fc_stage, syn_stage):
            for f in CROP_FUNCS:
                if hasattr(mod, f):
                    setattr(mod, f, _timed(getattr(mod, f), f))
        for mod, f, label in ((bp, "predict", "basin"), (ki, "predict", "main"), (win_stage, "main_windows", "win"),
                              (oa_stage, "predict", "oa"), (fc_stage, "predict_pkg", "fc"), (syn_stage, "predict", "syn")):
            setattr(mod, f, _stage(getattr(mod, f), label, proc))


def install_pad_loc(pkg, bucket):
    """The crop stages' localiser (a per-slice MobileNetV3 + MLP head, BatchNorm in eval mode) on series batches padded to a multiple of
    `bucket` slices (zero rows, outputs dropped): a few fixed batch shapes instead of one per series length (Kaggle's cuDNN builds kernels
    for every new shape; bench v1: the OA stage's localise took 78 s for 101 calls, the same calls 2.6 s in later stages). Real rows'
    outputs can differ only by cuDNN's per-shape fp32 algorithm choice. NOT exact: a separate bench variant."""
    import cv2
    import numpy as np
    import torch
    bp, ki, oa_stage, fc_stage, syn_stage, _ = _mods(pkg)
    for mod in (oa_stage, fc_stage, syn_stage):
        @torch.no_grad()
        def localise(loc, vol, plane, pos, dev, mod=mod):
            n = len(vol)
            small = np.stack([cv2.resize(a, (mod.LOC_PX, mod.LOC_PX), interpolation=cv2.INTER_AREA) for a in vol])
            w = np.stack([small[np.clip(np.arange(n) + d, 0, n - 1)] for d in (-1, 0, 1)], 1)
            aux = mod.aux_features(plane, pos, n)
            m = -(-n // bucket) * bucket
            if m > n:
                w = np.concatenate([w, np.zeros((m - n,) + w.shape[1:], w.dtype)])
                aux = np.concatenate([aux, np.zeros((m - n,) + aux.shape[1:], aux.dtype)])
            p, b = loc(torch.from_numpy(w).to(dev), torch.from_numpy(aux).to(dev))
            return torch.sigmoid(p[:n]).float().cpu().numpy(), b[:n].float().cpu().numpy()
        mod.localise = localise


def read_prof():
    """-> DataFrame stage x function -> total seconds, calls (all processes' files)."""
    rows = []
    for p in Path(PROF_DIR).glob("prof_*.tsv"):
        d = pd.read_csv(p, sep="\t", names=["stage", "func", "sec"])
        d["pid"] = p.stem[5:]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows)
    return d.groupby(["stage", "func"]).sec.agg(["sum", "count"]).round(2)


def worker(pkg, root, job, out):
    """The package's four single-GPU stage calls (arguments as ensemble.predict builds them) on this worker's studies."""
    t_enter = time.time()
    J = pickle.load(open(job, "rb"))
    if J.get("pad_loc"):
        install_pad_loc(pkg, J["pad_loc"])
    if J["prof"]:
        install_prof(pkg, J["name"], sync=J.get("prof_sync", False))
    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)
    import torch
    T = {"t_enter": t_enter, "import_s": time.time() - t_enter}       # spawn -> enter = t_enter - FAST["t0"] (per-call overhead)
    FINDINGS = ki.FINDINGS
    cfg = json.load(open(Path(pkg) / "config.json"))
    studies, workers = J["studies"], J["workers"]
    series = J["series"]
    series = series[series.StudyInstanceUID.isin(studies)].assign(Fat_Suppression=lambda d: d.Fat_Suppression.fillna(0).astype(int))
    fs_of = dict(zip(series.SeriesInstanceUID, series.Fat_Suppression))
    R = {}

    H = _hooks(pkg)

    def basin():
        ckpts, det, win_kw = H.basin_args(cfg, pkg)
        z = bp.predict(series, root, ckpts, workers=workers, detector=det, win=win_kw) if len(series) else {}
        return z, dict(bp.WIN), dict(bp.STATS)

    def oa():
        return oa_stage.predict(root, studies, fs_of, pkg, H.oa_cfg(cfg), workers=workers, threads=None), dict(oa_stage.STATS)

    def fc():
        return fc_stage.predict_pkg(root, studies, fs_of, pkg, H.fc_cfg(cfg), workers=workers), dict(fc_stage.STATS)

    def syn():
        s = cfg["syn"]
        members = H.syn_members(cfg, pkg)
        zm = pd.DataFrame(syn_stage.predict(root, studies, fs_of, members, str(Path(pkg) / s["loc"]), str(Path(pkg) / s["detector"]),
                                            workers=workers, fuse=s.get("fuse")))
        return zm, dict(syn_stage.STATS)

    run = {"basin": basin, "oa": oa, "fc": fc, "syn": syn}
    have = {"basin": True, "oa": bool(cfg.get("oa")) and H.oa_cfg(cfg)["mode"] != "main", "fc": bool(cfg.get("fc")), "syn": bool(cfg.get("syn"))}
    order = J.get("order", ["basin", "oa", "fc", "syn"])
    if J.get("merge_crops"):              # one study loop for the three crop stages (crops), at the position of the first crop stage
        assert all(have[s] for s in ("oa", "fc", "syn")), "merge_crops needs the oa, fc and syn stages"
        first = min(order.index(s) for s in ("oa", "fc", "syn"))
        order = [s for s in order if s not in ("oa", "fc", "syn")]
        order.insert(first, "crops")
        have["crops"] = True
        run["crops"] = lambda: crop_all(pkg, root, studies, fs_of, cfg, workers)
    for st in order:
        if have[st]:
            release()                         # the previous stage's models / buffers are unreferenced locals by now: give their memory back
            T[f"{st}_rss_in_gb"] = rss_gb()
            t = time.time()
            r = run[st]()
            if st == "crops":
                R.update(r)
                T["crops_load_s"] = r["oa"][1]["load_s"]
            else:
                R[st] = r
            T[st] = time.time() - t
            T[f"{st}_rss_out_gb"] = rss_gb()
            release()
            T[f"{st}_rss_released_gb"] = rss_gb()
    T["rss_peak_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    T["rss_peak_loader_gb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    T["gpu_peak_gb"] = torch.cuda.max_memory_reserved() / 2**30
    pickle.dump({"R": R, "T": T, "stage_log": STAGE_LOG}, open(out, "wb"))


def crop_all(pkg, root, studies, fs_of, cfg, workers):
    """oa_stage.predict + fc_stage.predict_pkg + syn_stage.predict in ONE study loop: each study's DICOM is decoded once (one
    DataLoader), the shared mirror detector (one weight file, identical correct_sides in the three modules) runs once, then each stage's
    own localiser and experts run as in its predict(), with that stage's failure handling. The bone stage's main-type members still see
    the uncorrected series. -> {"oa": (out, STATS), "fc": (out, STATS), "syn": (DataFrame, STATS)} in the stages' return formats."""
    import collections

    import numpy as np
    import torch
    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)
    from basin_flipdet import FlipNet
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    H = _hooks(pkg)
    ocfg, fcfg, scfg = H.oa_cfg(cfg), H.fc_cfg(cfg), cfg["syn"]
    dets = {str(Path(pkg) / ocfg["detector"]), str(Path(pkg) / fcfg["detector"]), str(Path(pkg) / scfg["detector"])}
    blobs = {open(p, "rb").read() for p in dets}
    assert len(blobs) == 1, "merge_crops needs one detector for the three crop stages"
    assert oa_stage.correct_sides.__code__.co_code == fc_stage.correct_sides.__code__.co_code == syn_stage.correct_sides.__code__.co_code \
        or all(hasattr(f, "__wrapped__") for f in (oa_stage.correct_sides, fc_stage.correct_sides, syn_stage.correct_sides))
    det = FlipNet(pretrained=False)
    det.load_state_dict(torch.load(sorted(dets)[0], map_location="cpu"))
    det = det.to(dev).eval()
    # OA (as oa_stage.predict)
    assert all(m["kind"] == "expert" for m in ocfg["members"].values()), "merge_crops: OA main-type members not supported"
    cor_v2 = ocfg.get("loc_version", "v1") == "v2cor"
    structs = oa_stage.STRUCTS_V2 if cor_v2 else oa_stage.STRUCTS
    oloc = oa_stage.Loc(structs=structs)
    oloc.load_state_dict(torch.load(f"{pkg}/{ocfg['loc']}", map_location="cpu"))
    oloc = oloc.to(dev).to(memory_format=torch.channels_last).eval()
    oexp = oa_stage.load_experts(pkg, ocfg, dev)
    xsets = {k: frozenset(v) for k, v in (ocfg.get("extra_sets") or {}).items()}     # oct20: oa_stage.predict's extra fusions
    orows = {k: {} for k in list(oexp) + (["fused"] if ocfg.get("fuse") else []) + (["fused_loa_extra"] if ocfg.get("loa_extra") else [])
             + [f"fused_{k}" for k in xsets]}
    okw = {"extra_sets": xsets} if xsets else {}
    # bone (as fc_stage.predict_pkg / predict)
    fmem = {n: {"kind": e["kind"], "cfg": e["cfg"], "files": [str(Path(pkg) / f) for f in e["files"]]} for n, e in fcfg["members"].items()}
    floc = fc_stage.Loc()
    floc.load_state_dict(torch.load(str(Path(pkg) / fcfg["loc"]), map_location="cpu"))
    floc = floc.to(dev).to(memory_format=torch.channels_last).eval()
    fexp, fsm = fc_stage.load_experts(fmem, dev), fc_stage.load_fsmain(fmem, dev)
    frows = {k: {} for k in list(fexp) + list(fsm) + (["fused"] if fcfg.get("fuse") else [])}
    # synovitis (as syn_stage.predict)
    smem = H.syn_members(cfg, pkg)
    sloc = syn_stage.Loc()
    sloc.load_state_dict(torch.load(str(Path(pkg) / scfg["loc"]), map_location="cpu"))
    sloc = sloc.to(dev).to(memory_format=torch.channels_last).eval()
    sexp = syn_stage.load_experts(smem, dev)
    srows = collections.defaultdict(dict)
    S = {k: {"failed": [], "flipped": 0} for k in ("oa", "fc", "syn")}
    load_s = time.time() - t0                  # the three stages' models + localisers + detector loaded (per-call overhead)
    dl = torch.utils.data.DataLoader(oa_stage.Studies(root, studies, fs_of), batch_size=None, num_workers=workers, collate_fn=lambda x: x,
                                     prefetch_factor=2 if workers else None)
    for i_study, (sid, series, err) in enumerate(dl):
        if RELEASE_EVERY and i_study and i_study % RELEASE_EVERY == 0:
            release(full=False)
        if err or not series:
            for k in S:
                S[k]["failed"].append(sid)
            print(f"crops: study {sid} failed ({err or 'no readable series'})", flush=True)
            continue
        try:
            for k, v in fc_stage.fsmain_logits(fsm, series, dev).items():   # main-type bone members: before the side correction (as trained)
                frows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            S["fc"]["failed"].append(sid)
            print(f"bone_stage: study {sid} failed ({type(e).__name__}: {e})", flush=True)
            fc_failed = True
        else:
            fc_failed = False
        try:
            n_flip = oa_stage.correct_sides(series, det, dev)
        except Exception as e:  # noqa: BLE001
            for k in S:
                if not (k == "fc" and fc_failed):
                    S[k]["failed"].append(sid)
            print(f"crops: study {sid} failed at the side correction ({type(e).__name__}: {e})", flush=True)
            continue
        for k in S:
            S[k]["flipped"] += n_flip
        try:
            bags = oa_stage.study_bags(series, oloc, dev, structs, cor_v2)
            for k, v in oa_stage.expert_logits(oexp, series, bags, dev, ocfg.get("fuse"), set(ocfg.get("loa_extra", [])),
                                               set(ocfg.get("loa_set", [])), **okw).items():   # = the stack's oa_stage.predict call
                orows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            S["oa"]["failed"].append(sid)
            print(f"oa_stage: study {sid} failed ({type(e).__name__}: {e}); OA keeps the pipeline blend", flush=True)
        if not fc_failed:
            try:
                locs = [fc_stage.localise(floc, s["vol"], s["plane"], s["pos"], dev) if len(s["vol"]) >= 3 else None for s in series]
                for k, v in fc_stage.expert_logits(fexp, series, locs, dev, fcfg.get("fuse")).items():
                    frows[k][sid] = v
            except Exception as e:  # noqa: BLE001
                S["fc"]["failed"].append(sid)
                print(f"bone_stage: study {sid} failed ({type(e).__name__}: {e})", flush=True)
        try:
            locs = [syn_stage.localise(sloc, s["vol"], s["plane"], s["pos"], dev) if len(s["vol"]) >= 3 else None for s in series]
            for k, v in syn_stage.expert_logits(sexp, series, locs, dev, scfg.get("fuse")).items():
                srows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            S["syn"]["failed"].append(sid)
            print(f"syn_stage: study {sid} failed ({type(e).__name__}: {e})", flush=True)
    wall = time.time() - t0
    for k in S:
        S[k]["wall_s"], S[k]["load_s"] = wall, load_s
    oout = {k: pd.DataFrame.from_dict(v, orient="index", columns=oa_stage.OUTS).astype(np.float64) for k, v in orows.items()}
    fout = {k: pd.DataFrame.from_dict(v, orient="index", columns=["Fracture", "Contusion"]) for k, v in frows.items()}
    sout = pd.DataFrame({k: pd.Series(v, dtype=np.float64) for k, v in srows.items()})
    return {"oa": (oout, S["oa"]), "fc": (fout, S["fc"]), "syn": (sout, S["syn"])}


def split(studies, series):
    """Two study lists: Basin's order (series csv order of the studies), cut at an even index; studies without series rows go to the
    first list (Basin never sees them; the crop stages log them as failed, as in the package)."""
    order = list(dict.fromkeys(series.StudyInstanceUID[series.StudyInstanceUID.isin(studies)]))
    k = (len(order) + 1) // 2
    k += k % 2
    missing = [s for s in studies if s not in set(order)]
    return [order[:k] + missing, order[k:]]


def _merge(parts):
    """Worker outputs -> one stage output per stage, in the package's return formats."""
    out = {}
    zs = [p["R"]["basin"] for p in parts]
    keys = list(dict.fromkeys(k for z, _, _ in zs for k in z))
    out["basin"] = ({k: pd.concat([z[k] for z, _, _ in zs if k in z]) for k in keys},
                    {s: v for _, w, _ in zs for s, v in w.items()},
                    {"workers": [st for _, _, st in zs]})
    for st in ("oa", "fc"):
        if all(st in p["R"] for p in parts):
            ms = [p["R"][st][0] for p in parts]
            out[st] = ({k: pd.concat([m[k] for m in ms if k in m]) for k in dict.fromkeys(k for m in ms for k in m)},
                       {"failed": sum((p["R"][st][1].get("failed", []) for p in parts), []),
                        "flipped": sum(p["R"][st][1].get("flipped", 0) for p in parts),
                        "wall_s": max(p["R"][st][1].get("wall_s", 0) for p in parts)})
    if all("syn" in p["R"] for p in parts):
        out["syn"] = (pd.concat([p["R"]["syn"][0] for p in parts]),
                      {"failed": sum((p["R"]["syn"][1].get("failed", []) for p in parts), []),
                       "flipped": sum(p["R"]["syn"][1].get("flipped", 0) or 0 for p in parts)})
    return out


def predict(root, studies, series, pkg, threads=None, workers=2, prof=False, work="/kaggle/working/fast_jobs", log=print,
            orders=(("basin", "oa", "fc", "syn"), ("basin", "oa", "fc", "syn")), merge_crops=False, pad_loc=32, prof_sync=False):
    """= ensemble.predict(root, studies, series, pkg, threads) with the four single-GPU stages on two GPUs. Returns its 4-tuple; FAST
    holds the worker / stage timings. orders: the stage order per worker (independent stages: outputs do not depend on it)."""
    bp, ki, oa_stage, fc_stage, syn_stage, win_stage = _mods(pkg)
    import ensemble
    import torch
    if prof:
        install_prof(pkg, "parent")
    serial = os.environ.get("FAST_SERIAL") == "1"           # local parity: both workers on GPU 0, one after another
    ngpu = torch.cuda.device_count()
    assert serial or ngpu >= 2, f"fast.predict needs 2 GPUs, found {ngpu}"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    procs = []
    for g, part in enumerate(split(studies, series)):
        job, out = f"{work}/job{g}.pkl", f"{work}/out{g}.pkl"
        pickle.dump({"studies": part, "series": series, "workers": workers, "prof": prof, "name": f"gpu{g}", "order": list(orders[g]), "merge_crops": merge_crops, "pad_loc": pad_loc, "prof_sync": prof_sync},
                    open(job, "wb"))
        if os.path.exists(out):
            os.remove(out)
        fh = open(f"{work}/worker{g}.log", "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0" if serial else str(g)}
        procs.append((subprocess.Popen([sys.executable, __file__, "worker", pkg, root, job, out], env=env, stdout=fh, stderr=subprocess.STDOUT), out, fh, len(part)))
        if serial:
            procs[-1][0].wait()
    for p, _, fh, _ in procs:
        p.wait()
        fh.close()
    t_par = time.time() - t0
    for g, (p, out, _, n) in enumerate(procs):
        if p.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"fast worker gpu{g} failed (code {p.returncode}); log tail:\n" + open(f"{work}/worker{g}.log").read()[-4000:])
    parts = [pickle.load(open(out, "rb")) for _, out, _, _ in procs]
    M = _merge(parts)
    log(f"fast: workers done in {t_par:.0f}s ({', '.join(f'gpu{g} {n} studies ' + str({k: round(v) for k, v in pp['T'].items()}) for g, ((_, _, _, n), pp) in enumerate(zip(procs, parts)))})")
    STAGE_LOG.extend(e for pp in parts for e in pp["stage_log"])

    orig = {"bp": bp.predict, "oa": oa_stage.predict, "fc": fc_stage.predict_pkg, "syn": syn_stage.predict}

    def bp_fake(*a, **k):
        z, w, st = M["basin"]
        bp.WIN.clear()
        bp.WIN.update(w)
        bp.STATS.clear()
        bp.STATS.update(st)
        return z

    def oa_fake(*a, **k):
        m, st = M["oa"]
        oa_stage.STATS.clear()
        oa_stage.STATS.update(st)
        return dict(m)

    def fc_fake(*a, **k):
        m, st = M["fc"]
        fc_stage.STATS.clear()
        fc_stage.STATS.update(st)
        return dict(m)

    def syn_fake(*a, **k):
        zm, st = M["syn"]
        syn_stage.STATS.clear()
        syn_stage.STATS.update(st)
        return zm.copy()

    bp.predict, oa_stage.predict, fc_stage.predict_pkg, syn_stage.predict = bp_fake, oa_fake, fc_fake, syn_fake
    try:
        with parent_release(ki):
            res = ensemble.predict(root, studies, series, pkg, threads)
    finally:
        bp.predict, oa_stage.predict, fc_stage.predict_pkg, syn_stage.predict = orig["bp"], orig["oa"], orig["fc"], orig["syn"]
    release()
    FAST.clear()
    FAST.update(t0=t0, parallel_s=t_par, parent_rss_gb=rss_gb(), workers=[{"n": n, **pp["T"]} for (_, _, _, n), pp in zip(procs, parts)], total_s=time.time() - t0,
                main_s=ensemble.STATS.get("main", {}).get("wall_s"), win_s=ensemble.STATS.get("win", {}).get("wall_s"))
    return res


FAST = {}

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "worker":
    worker(*sys.argv[2:6])
