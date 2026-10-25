"""oct13_f5_win: cross-backbone window-level fusion of the 3x3 for chosen columns (experiments/v2/oct13_deep_ensembling_test: dump_main.py,
dump_basin.py, wfuse.py, w_search.py; 5-fold cross-fitted on Codex image reads for Effusion / Baker's).
- main_windows(): the main pipeline's attention-MIL models (knee_infer DICOM path and window rules, but EVERY (series, centre) window of
  the 5-95% span instead of the 96-window grid); each model's head split into per-window attention logits s and logits u
  (study logit = sum_k softmax_k(s) u).
- fuse(): per study, the main windows are the common grid; each Basin model's per-image export (basin_predict.WIN, deployed T_eval 24)
  is mapped onto them by (series uid, slice) when the series has the same slice count in both pipelines (thin series are slab-averaged
  in main only: unmapped), unmapped windows get s = -30, u = that model's study logit (as the experiment's dumps). Spec
  geo|pm|g20|wmean: relevance = softmax over windows of the mean attention logit over all models; value = mean over pipelines of the
  pipeline's mean window probability; top-20 windows by relevance per output; pooled prob = relevance-weighted mean -> logit."""
import collections
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

import knee_infer as ki
from model import KneeNet, sincos

CHUNK = 128


def load_models(pkg, files, cfg, dev):
    ms = []
    for f in files:
        sd = torch.load(os.path.join(pkg, f), map_location="cpu")
        m = KneeNet(cfg["backbone"], n_out=sd["cls_b"].shape[0], pretrained=False, hidden=cfg["hidden"], drop=cfg["drop"],
                    use_type=cfg["use_type"], use_pos=cfg["use_pos"], head=cfg["head"])
        m.load_state_dict(sd)
        m.eval().to(dev)
        m.backbone.half().to(memory_format=torch.channels_last)
        assert m.head == "attn" and m.att_u is None and not m.n_aux
        ms.append(m)
    return ms


def n_pairs(ns, span=(0.05, 0.95)):
    """Number of (series, centre) windows of knee_infer.window_rows (same span rule)."""
    t = 0
    for n in ns:
        if n < 3:
            continue
        lo, hi = max(1, int(np.floor(span[0] * n))), min(n - 2, int(np.ceil(span[1] * n)) - 1)
        t += (hi - lo + 1) if hi >= lo else n - 2
    return t


def rss_gb():
    return int(next(line for line in open("/proc/self/status") if line.startswith("VmRSS")).split()[1]) / 2**20


def prep(root, sid, fs_of):
    """CPU part of one study (knee_infer DICOM path: IO threads, parse, series plans) and its windows: every (series, centre) window."""
    raw, side, skipped, buf = ki.read_study(os.path.join(root, sid), fs_of, buffered=True)
    plans, poss, types, uids = [], [], [], []
    for (info, hs, imgs, offs), s_side, order, pos in ki.kept_series(raw, side, skipped):
        try:
            plan, p = ki.series_plan(info, imgs, offs, s_side, order, pos)
        except Exception:  # noqa: BLE001  (as study_item: the series is skipped)
            continue
        plans.append(plan), poss.append(p), types.append(info["type"]), uids.append(info["uid"])
    ns = [p["n"] for p in plans]
    K = n_pairs(ns)
    uniq, idx, mask, wins = ki.window_rows(ns, K)
    assert len(wins) == K and bool(mask.all())
    return {"buf": buf, "plans": plans, "uniq": torch.from_numpy(uniq), "idx": torch.from_numpy(idx),
            "typ": torch.tensor([types[a] for a, _ in wins]), "pos": torch.tensor([poss[a][cc] for a, cc in wins], dtype=torch.float32),
            "keys": [(uids[a], int(cc)) for a, cc in wins], "n": dict(zip(uids, ns))}


@torch.no_grad()
def run_study(it, ms, dev, cfg, cols, crop_px):
    """GPU part of one study: window slices, each model's per-window attention logits s and logits u."""
    c = ms[0].backbone.pretrained_cfg
    mean, std = ki.norm_stats(tuple(c["mean"]), tuple(c["std"]), dev)
    slices = ki.gpu_slices(it, dev, crop_px)
    s = F.interpolate(slices[:, None].float().div_(255), size=(cfg["img"], cfg["img"]), mode="bilinear", antialias=True,
                      align_corners=False)[:, 0]
    K = len(it["keys"])
    pad = torch.zeros((-K) % CHUNK, dtype=torch.int64)       # every backbone call has batch CHUNK (window 0 repeated, outputs dropped):
    x = ((s[torch.cat([it["idx"], it["idx"][pad]]).to(dev)] - mean) / std).half().contiguous(memory_format=torch.channels_last)
    typ, pos = it["typ"].to(dev), it["pos"].to(dev)          # Kaggle's cuDNN builds kernels per new batch size: ~9.5 s and host RAM each
    S, U = [], []
    for m in ms:
        f = torch.cat([m.backbone(x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)])[:K].float()
        h = f
        if m.use_type:
            h = h + m.type_emb(typ)
        if m.use_pos:
            h = h + m.pos_proj(sincos(pos))
        h = m.norm(h)
        S.append(m.att_w(torch.tanh(m.att_v(h)))[:, cols])
        U.append((h @ m.cls_w.T + m.cls_b)[:, cols])
    return {"keys": it["keys"], "n": it["n"], "s": torch.stack(S).cpu().numpy(), "u": torch.stack(U).cpu().numpy()}


def main_windows(root, studies, fs_of, pkg, files, cfg, cols, devices=None, threads=None, ahead=None, log_every=50):
    """-> {study: {"keys": [(uid, centre)], "n": {uid: slices}, "s", "u": (models, windows, len(cols)) float32 numpy}}; a study that
    fails is missing (logged). Pipelined as knee_infer.predict: `threads` CPU threads (default CPUs) prepare up to `ahead` studies
    (default 2 x threads) ahead; one GPU thread per device (default all GPUs, each with its own copy of the models) takes the next study.
    The per-study computation does not depend on the device or the order."""
    devices = devices or [f"cuda:{g}" for g in range(torch.cuda.device_count())]
    threads = threads or len(os.sched_getaffinity(0))
    ahead = ahead or 2 * threads
    crop_px = min(ki.PX, int(round(cfg["crop_mm"] / ki.MM_PER_PX)))
    models = [load_models(pkg, files, cfg, torch.device(d)) for d in devices]
    out, t0, done = {}, time.time(), [0]
    q = queue.Queue(maxsize=len(devices))

    def gpu_loop(ms, dev):
        while (job := q.get()) is not None:
            sid, it = job
            try:
                out[sid] = run_study(it, ms, torch.device(dev), cfg, cols, crop_px)
            except Exception as e:  # noqa: BLE001
                print(f"win_stage: study {sid} failed on {dev} ({type(e).__name__}: {e})", flush=True)
            done[0] += 1
            if log_every and done[0] % log_every == 0:
                print(f"win_stage: {done[0]}/{len(studies)} studies, {time.time() - t0:.0f}s, rss {rss_gb():.1f} GB, gpu alloc / reserved " +
                      ", ".join(f"{d} {torch.cuda.memory_allocated(d) / 2**30:.1f} / {torch.cuda.memory_reserved(d) / 2**30:.1f}"
                                for d in devices) + " GB", flush=True)

    gpu_threads = [threading.Thread(target=gpu_loop, args=(ms, d), daemon=True) for ms, d in zip(models, devices)]
    for th in gpu_threads:
        th.start()
    with ThreadPoolExecutor(threads) as loader:
        futs = collections.deque(loader.submit(prep, root, sid, fs_of) for sid in studies[:ahead])
        for n, sid in enumerate(studies):
            try:
                it = futs.popleft().result()
            except Exception as e:  # noqa: BLE001
                print(f"win_stage: study {sid} failed in prep ({type(e).__name__}: {e})", flush=True)
                it = None
            if n + ahead < len(studies):
                futs.append(loader.submit(prep, root, studies[n + ahead], fs_of))
            if it is not None:
                q.put((sid, it))
            else:
                done[0] += 1
    for _ in gpu_threads:
        q.put(None)
    for th in gpu_threads:
        th.join()
    return out


def basin_rows(mw, w):
    """One Basin model's export mapped onto the main windows: (s, u) (windows, cols); unmapped: s = -30, u = study logit."""
    K, C = len(mw["keys"]), len(w["z"])
    s, u = np.full((K, C), -30.0), np.tile(np.asarray(w["z"], np.float64), (K, 1))
    for k, (uid, cc) in enumerate(mw["keys"]):
        if (uid, cc) in w["acc"] and w["n"].get(uid) == mw["n"][uid]:
            ws, wu = w["acc"][(uid, cc)]
            s[k] = np.log(np.maximum(ws, 1e-12))
            u[k] = wu / np.maximum(ws, 1e-12)
    return s, u


def fuse(mw, basin, gate=20):
    """mw: main_windows() entry; basin: {pipeline: [WIN entries of its models]} -> fused logits (len(cols),)."""
    S, U, G = [m for m in mw["s"].astype(np.float64)], [m for m in mw["u"].astype(np.float64)], [0] * len(mw["s"])
    for gi, ws in enumerate(basin.values(), start=1):
        for w in ws:
            s, u = basin_rows(mw, w)
            S.append(s), U.append(u), G.append(gi)
    S, U, G = np.stack(S), np.stack(U), np.asarray(G)
    a = S.mean(0)
    r = np.exp(a - a.max(0))
    r /= r.sum(0)                                                        # geo relevance (K, C)
    P = np.stack([(1 / (1 + np.exp(-U[G == g]))).mean(0) for g in sorted(set(G))]).mean(0)
    if r.shape[0] > gate:
        keep = np.zeros_like(r)
        np.put_along_axis(keep, np.argsort(-r, 0)[:gate], 1.0, 0)
        r = r * keep
        r /= r.sum(0)
    p = np.clip((r * P).sum(0), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
