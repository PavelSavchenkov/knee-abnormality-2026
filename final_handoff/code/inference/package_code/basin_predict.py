"""Inference from DICOM for the Basin-code pipelines (2nd baseline NFNet-L0, 3rd baseline RadImageNet-R50): a copy of
experiments/v0/oct10_baseline_3rd/src/predict.py (= oct9_baseline_2nd/src/predict.py) with these changes for the 3-pipeline package:
- returns per-model logits (12 FINDINGS) instead of the averaged probability, so pipelines can be weighted equally;
- --mono models (RadImageNet) see the centre slice of each triplet in all 3 channels (train.py mono());
- the mirror detector (basin_flipdet.py) also works for position-aware dense models (the source asserted it off for them): a
  sagittal series called mirrored gets its token order reversed with pos -> 1 - pos and tmask reversed, exactly as the training-time
  mirror of pos models (oct9_baseline_2nd/src/train.py); coronal / axial columns flip as in the source;
- non-finite model outputs are counted (STATS["nonfinite"]) before nan_to_num, so the check run can assert there are none;
- image-head models (experiments/v1/oct10_synovitis/basin/train.py --img_w > 0: one extra output after the asym outputs, trained on
  the Codex image reads of synovitis): their Synovitis logit is the image-head output (submissions/oct10_syn3x3);
- flat Kaggle dataset: model.py -> basin_model.py (the main pipeline has its own model.py in the same process), common/X.py ->
  common_X.py, checkpoint config next to it as basin_<pipeline>_<i>.json (was config.json in the run dir).
Preprocessing: infer_prep.py (DICOM -> the s320 training cache, max 1 grey level; slot selection as in training)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from slots import series_spacing, slab_indices, slab_k, slab_mean, WEIGHTINGS, select_slots, triplet_indices  # noqa: E402
from infer_prep import scan_study, series_volume  # noqa: E402
from basin_model import StudyModel, gpu_views, view_fov  # noqa: E402
from basin_flipdet import FlipNet, det_input  # noqa: E402
from common_paths import FINDINGS  # noqa: E402

SYN = FINDINGS.index("Synovitis")


CACHE_GEOM = {"s320": (0.5, 320), "s352": (0.4, 352), "s448": (0.3, 448)}  # build_cache.py --px / --size per cache
STATS = {}  # last predict(): per group wall / data wait / GPU seconds, decode failures, non-finite outputs


class DicomStudies(torch.utils.data.Dataset):
    """One item = one study: uint8 (V, S, T, 3, 320, 320) for V slice-offset views (decoded once) + slot mask."""
    def __init__(self, series_csv, root, slots, T, lo, hi, offsets=(0.0,), dense=0, pos_window=(0.0, 1.0), cache="s320", det=False,
                 slab=False):
        self.slab = slab  # thin series as ~3.3 mm slabs (slots.slab_k), as in training
        self.det = det  # also return the mirror-detector input per series (det_input of the canonical volume)
        self.info = series_csv
        self.px_mm, self.size = CACHE_GEOM[cache]  # same preprocessing as the training cache of the model
        self.pos_window = pos_window
        self.dense = dense  # > 0: one position-aware view with min(n, dense) evenly spaced centres over the whole stack
        self.studies = list(dict.fromkeys(series_csv.StudyInstanceUID))
        self.root, self.slots, self.T = Path(root), slots, T
        s = (hi - lo) / (T - 1)
        self.views = [(lo + o * s, hi + o * s) for o in offsets]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        """Failures are contained and logged: a series that cannot be read leaves its slot empty; a study whose header scan
        fails gets an all-empty slot mask (its predictions are replaced by the cohort mean in predict())."""
        s = self.studies[i]
        Tm, V = (self.dense, 1) if self.dense else (self.T, len(self.views))
        x = np.zeros((V, len(self.slots), Tm, 3, self.size, self.size), np.uint8)
        dx = np.zeros((len(self.slots), 3, self.size, self.size), np.uint8) if self.det else None
        dm = np.zeros(len(self.slots), bool) if self.det else None
        nfail = np.zeros(4, np.int64)  # study header scan failed, series skipped at header scan, series failed, frames replaced
        pos, tmask = np.zeros((len(self.slots), Tm), np.float32), np.zeros((len(self.slots), Tm), bool)
        widx = np.zeros(len(self.slots), np.int64)
        mask = np.zeros(len(self.slots), bool)
        uids, ns = [""] * len(self.slots), np.zeros(len(self.slots), np.int64)   # window export (oct13_f5_win): series and length per slot
        try:
            info_s = self.info[self.info.StudyInstanceUID == s]
            meta = scan_study(self.root / s, info_s)
            nfail[1] = len(info_s) - len(meta)
            meta["StudyInstanceUID"] = s
            table = select_slots(meta.rename_axis("SeriesInstanceUID"), self.slots)
        except Exception as e:  # noqa: BLE001
            print(f"study {s}: header scan failed: {type(e).__name__}: {e}", flush=True)
            table = None
            nfail[0] = 1
        if table is not None and s in table.index:
            for k, uid in enumerate(table.loc[s].to_numpy()):
                if not isinstance(uid, str):
                    continue
                try:
                    def log(m):
                        nfail[3] += 1
                        print(f"study {s}: {m}", flush=True)
                    vol = series_volume(meta.loc[uid], log=log, px_mm=self.px_mm, size=self.size)
                    if self.det:
                        dx[k], dm[k] = det_input(vol, meta.loc[uid, "plane"]), True
                    widx[k] = WEIGHTINGS.index(meta.loc[uid, "weighting"])
                    n = len(vol)
                    uids[k], ns[k] = uid, n
                    if self.dense:
                        plo, phi = self.pos_window
                        c = np.unique(np.round(np.linspace(plo * (n - 1), phi * (n - 1), min(n, self.dense))).astype(int))
                        x[0, k, :len(c)] = vol[np.clip(c[:, None] + np.array([-1, 0, 1])[None], 0, n - 1)]
                        pos[k, :len(c)], tmask[k, :len(c)] = c / max(n - 1, 1), True
                    else:
                        k_sl = slab_k(series_spacing(meta.loc[uid, "pos"], n)) if self.slab else 1
                        for j, (lo, hi) in enumerate(self.views):
                            x[j, k] = slab_mean(vol[slab_indices(triplet_indices(n, self.T, lo, hi, False, None), k_sl, n)])
                    mask[k] = True
                except Exception as e:  # noqa: BLE001
                    print(f"study {s} series {uid}: failed, slot left empty: {type(e).__name__}: {e}", flush=True)
                    nfail[2] += 1
        return {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "pos": torch.from_numpy(pos), "tmask": torch.from_numpy(tmask),
                "widx": torch.from_numpy(widx), "i": i, "nfail": torch.from_numpy(nfail), "uids": "|".join(uids), "n": torch.from_numpy(ns),
                **({"det": torch.from_numpy(dx), "detm": torch.from_numpy(dm)} if self.det else {})}


def img_col(cfg):
    """Output column of the image-read synovitis head (train.py --img_w), or None."""
    return 12 + (2 if cfg.get("asym") else 0) + (2 if cfg.get("asym") and cfg.get("asym_pf") else 0) if cfg.get("img_w") else None


def load_model(ckpt):
    cfg = json.load(open(Path(ckpt).with_suffix(".json")))  # build_package.py: basin_<pipeline>_<i>.pt + .json
    n_out = 12 + (2 if cfg.get("asym") else 0) + (2 if cfg.get("asym") and cfg.get("asym_pf") else 0) + (1 if cfg.get("img_w") else 0)
    m = StudyModel(cfg["backbone"], len(cfg["slots"].split(",")), cfg["T"], n_out=n_out, d=cfg["d"], layers=cfg["layers"], drop_path=0.0,
                   head=cfg.get("head", "cls"), pretrained=False, n_aux=5 if cfg.get("aux") else 0,
                   seq=cfg.get("seq", "tf"), pool=cfg.get("pool", "avg"), slot_aux=bool(cfg.get("slot_aux")), grid=cfg.get("grid", 1),
                   comp=[not sl.startswith("sag") for sl in cfg["slots"].split(",")] if cfg.get("comp") else None, crop_frac=cfg.get("crop_frac", 0.6),
                   sag_size=cfg.get("sag_size", 0), img_size=cfg["size"],
                   sag=[sl.startswith("sag") for sl in cfg["slots"].split(",")], bin_all=bool(cfg.get("bin_all")),
                   rows=[sl.startswith("ax") for sl in cfg["slots"].split(",")] if cfg.get("comp") and cfg.get("ax_rows") else None,
                   bins_sym=bool(cfg.get("bins_sym")), bin_max=bool(cfg.get("bin_max")),
                   posenc="fourier" if cfg.get("sample") == "pos" else "index", wemb=bool(cfg.get("wemb")),
                   skip_missing=bool(cfg.get("skip_missing")), men_reg=int(cfg.get("men_reg", 0)), bn_single=bool(cfg.get("bn_single")))
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    return m.eval(), cfg


def mono(v, on):
    """(B, S, T, 3, H, W) views -> the centre slice of each triplet copied to all 3 channels when on (train.py mono())."""
    return v[:, :, :, 1:2].expand_as(v).contiguous() if on else v


WIN = {}  # last predict() with win: {study: {ckpt: {"acc": {(series uid, slice): (sum w, sum w u)}, "n": {uid: n}, "z": z}}}


def export_windows(m, calls, z, b, uids, ns, pos, flipped, cols, out):
    """Per-image decomposition of one battn / layers-0 Basin model for batch element b (experiments/v2/oct13_deep_ensembling_test/
    dump_basin.py, = work/pl_research/basin_window_export.py): image k of region r gets weight a_kf / R and substitution logit
    u'_kf = z_f - reg_w[f, r] . z_rf + reg_w[f, r] . h_k, so sum_k w u' = z_f exactly. Images are keyed by (series uid, DICOM slice
    index): the token's centre from the dataset positions, reversed for a mirrored sagittal slot (its tokens were flipped)."""
    reg_w = m.reg_w.float()[cols]
    S = len(uids)
    sag = np.array([not c for c in m.comp])
    sag_slots, crop_slots = np.where(sag)[0], np.where(~sag)[0]
    regions = []
    for k in range(3):
        h, att, w = calls[k]
        for j in range(h.shape[1]):
            regions.append((k * len(sag_slots) + j, [(sag_slots[j], t, h[b, j, t], att[b, j, t][cols]) for t in range(h.shape[2]) if bool(w[b, j, t])]))
    h, att, w = calls[3]
    for j in range(h.shape[1]):
        for c in range(2):
            regions.append((3 * len(sag_slots) + j * 2 + c, [(crop_slots[j], t, h[b, j, t, c], att[b, j, t, c][cols]) for t in range(h.shape[2])
                                                          if bool(w[b, j, t, c])]))
    nonempty = [(r, items) for r, items in regions if items]
    acc = {}
    T = pos.shape[1]
    for r, items in nonempty:
        H = torch.stack([it[2] for it in items])
        A = torch.stack([it[3] for it in items])
        cr = (reg_w[:, r] * (A.T @ H)).sum(-1)
        uk = z[None] - cr[None] + H @ reg_w[:, r].T
        wk = A / len(nonempty)
        for (slot, t, _, _), wi, ui in zip(items, wk.cpu().numpy(), uk.cpu().numpy()):
            tt = T - 1 - t if flipped[slot] else t
            key = (uids[slot], int(round(float(pos[slot, tt]) * max(int(ns[slot]) - 1, 1))))
            ws, wu = acc.get(key, (0.0, 0.0))
            acc[key] = (ws + wi, wu + wi * ui)
    out.update(acc=acc, n={uids[k]: int(ns[k]) for k in range(S) if uids[k]}, z=z.cpu().numpy())


@torch.no_grad()
def predict(series_csv, root, ckpts, dtype=torch.float16, workers=4, bs=2, offsets=(0.0, -1 / 3, 1 / 3, -2 / 3, 2 / 3),
            detector=None, det_thr=0.9, win=None):
    """win (oct13_f5_win): {"ckpts": [paths], "cols": [output columns]}: also fills WIN with the per-image decomposition of those
    models (dense position-aware battn / layers 0 / comp models only)."""
    """Per-model logits {ckpt: DataFrame studies x FINDINGS}: mean over views (slice-offset TTA for grid models; dense
    position-aware models have one view); models sharing (slots, T, lo, hi, dense, cache) share one DICOM pass. Weights stay fp32,
    the forward runs under autocast(dtype). Studies without any readable series get the cohort mean logit of each model (logged).
    detector: path of a FlipNet state dict; series it calls mirrored (p > det_thr) are mirrored back before the models see them
    (test-time laterality; logged, listed in STATS["mirrored"])."""
    STATS.clear()
    WIN.clear()
    STATS["nonfinite"], STATS["det_nonfinite"], STATS["mirrored"] = 0, 0, []
    models = [load_model(c) for c in ckpts]
    calls = []
    for c, (m, cfg) in zip(ckpts, models):
        if win and c in win["ckpts"]:
            assert cfg.get("sample") == "pos" and cfg.get("head") == "battn" and not cfg.get("layers") and cfg.get("comp") \
                and not cfg.get("bin_all") and not cfg.get("ax_rows"), f"window export not supported for {c}"
            def rec(h, w, dim, m=m, orig=m.att_pool):
                sc = m.reg_score(h).float().masked_fill(~w[..., None], float("-inf"))
                calls.append((h.float(), torch.nan_to_num(sc.softmax(dim), nan=0.0), w))
                return orig(h, w, dim)
            m.att_pool = rec
    det_net = None
    if detector:
        det_net = FlipNet(pretrained=False)
        det_net.load_state_dict(torch.load(detector, map_location="cpu"))
        det_net.cuda().eval()
    groups = {}
    for c, (m, cfg) in zip(ckpts, models):
        dense = cfg["T_eval"] if cfg.get("sample") == "pos" else 0
        pw = (cfg["lo"], cfg["hi"]) if cfg.get("pos_window") else (0.0, 1.0)
        groups.setdefault((cfg["slots"], cfg["T"], cfg["lo"], cfg["hi"], dense, pw, cfg.get("cache", "s320"), bool(cfg.get("slab"))), []).append((c, m, cfg))
    res, empty_studies = {}, set()
    for (slots, T, lo, hi, dense, pw, cache, slab), ms in groups.items():
        ds = DicomStudies(series_csv, root, slots.split(","), T, lo, hi, offsets, dense=dense, pos_window=pw, cache=cache,
                          det=det_net is not None, slab=slab)
        sag_slots = torch.tensor([sl.startswith("sag") for sl in slots.split(",")], device="cuda")
        nfail, t_wait = np.zeros(4, np.int64), 0.0
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers)
        outs = [[] for _ in ms]
        for _, m, _ in ms:
            m.cuda()
        t0 = tq = time.time()
        for b in dl:
            t_wait += time.time() - tq
            nfail += b["nfail"].sum(0).numpy()
            x, mask = b["x"].cuda(non_blocking=True), b["mask"].cuda(non_blocking=True)
            pt = (b["pos"].cuda(), b["tmask"].cuda()) if dense else (None, None)
            if det_net is not None:  # mirror back the series the detector calls mirrored
                dm = b["detm"].cuda()
                with torch.autocast("cuda", dtype=dtype):
                    zf = det_net(b["det"].cuda().flatten(0, 1)).float().view(dm.shape)
                STATS["det_nonfinite"] += int((~torch.isfinite(zf) & dm).sum())
                pf = torch.sigmoid(zf)
                fl = dm & (pf > det_thr)
                if bool(fl.any()):
                    for (bi, si) in fl.nonzero().tolist():
                        STATS["mirrored"].append((ds.studies[int(b["i"][bi])], slots.split(",")[si], round(float(pf[bi, si]), 4)))
                        print(f"study {STATS['mirrored'][-1][0]} slot {STATS['mirrored'][-1][1]}: detector p_mirrored "
                              f"{STATS['mirrored'][-1][2]:.3f} -> mirrored back", flush=True)
                    x = x.clone()
                    s_sag = (fl & sag_slots[None])[:, None].expand(-1, x.shape[1], -1)  # (B, V, S): sagittal -> reverse slices
                    s_oth = (fl & ~sag_slots[None])[:, None].expand(-1, x.shape[1], -1)  # coronal / axial -> flip columns
                    x[s_sag] = x[s_sag].flip(1).flip(2)
                    x[s_oth] = x[s_oth].flip(-1)
                    if dense:  # token order of the flagged sagittal series reversed: reverse positions / masks too, p -> 1 - p
                        fs = fl & sag_slots[None]
                        pos_t, tm_t = pt[0].clone(), pt[1].clone()
                        pos_t[fs] = 1 - pt[0][fs].flip(-1)
                        tm_t[fs] = pt[1][fs].flip(-1)
                        pt = (pos_t, tm_t)
            empty_studies |= {ds.studies[int(i)] for i, mk in zip(b["i"], b["mask"]) if not bool(mk.any())}
            mask = mask | ~mask.any(1, keepdim=True)  # all-empty study: run on zeros (overwritten later) instead of NaN
            for j, (c, m, cfg) in enumerate(ms):
                z = 0
                for v in range(x.shape[1]):
                    calls.clear()
                    with torch.autocast("cuda", dtype=dtype):
                        out = m(mono(gpu_views(x[:, v], cfg["size"], False, fov=view_fov(cfg)[0], yshift=view_fov(cfg)[1], norm=bool(cfg.get("std_norm"))), cfg.get("mono")),
                                mask, *pt, widx=b["widx"].cuda() if cfg.get("wemb") else None).float()
                    if win and c in win["ckpts"]:
                        assert len(calls) == 4 and x.shape[1] == 1, len(calls)
                        flip = fl.cpu().numpy() & sag_slots.cpu().numpy()[None] if det_net is not None else np.zeros(mask.shape, bool)
                        for bi in range(x.shape[0]):
                            if bool(b["mask"][bi].any()):
                                export_windows(m, calls, out[bi, win["cols"]], bi, b["uids"][bi].split("|"), b["n"][bi].numpy(),
                                               b["pos"][bi].numpy(), flip[bi], win["cols"],
                                               WIN.setdefault(ds.studies[int(b["i"][bi])], {}).setdefault(c, {}))
                    ic = img_col(cfg)
                    out = torch.cat([out[:, :SYN], out[:, ic:ic + 1], out[:, SYN + 1:12]], 1) if ic is not None else out[:, :12]
                    STATS["nonfinite"] += int((~torch.isfinite(out)).sum())
                    z = z + torch.nan_to_num(out, nan=0.0)
                outs[j].append((z / x.shape[1]).cpu())
            tq = time.time()
        t_all = time.time() - t0
        STATS[f"group{len(STATS) - 1}"] = {"studies": len(ds), "models": len(ms), "wall_s": t_all, "wait_data_s": t_wait, "nfail": nfail.tolist()}
        print(f"group {slots} T={T} {cache}: {len(ds)} studies x {len(ms)} models x {1 if dense else len(offsets)} views in "
              f"{t_all:.0f}s (data wait {t_wait:.0f}s, GPU {t_all - t_wait:.0f}s)", flush=True)
        print(f"decode summary: study scans failed {nfail[0]}, series skipped at scan {nfail[1]}, series failed {nfail[2]}, "
              f"frames replaced {nfail[3]}; non-finite outputs so far {STATS['nonfinite']}", flush=True)
        if det_net is not None:
            print(f"detector: {len(STATS['mirrored'])} series mirrored back, non-finite detector outputs {STATS['det_nonfinite']}", flush=True)
        for (c, m, _), o in zip(ms, outs):
            m.cpu()
            res[c] = pd.DataFrame(torch.cat(o).numpy().astype(np.float64), index=pd.Index(ds.studies, name="StudyInstanceUID"), columns=FINDINGS)
    empty = sorted(empty_studies)
    if empty:
        print(f"{len(empty)} studies without readable series -> cohort mean logit: {empty[:10]}", flush=True)
        for c, z in res.items():
            z.loc[empty] = z.drop(index=empty).mean().to_numpy()
    return res
