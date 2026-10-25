"""Test-time path of the fracture / contusion bone experts (main/bone_data.py + main/bone_train.py BoneNet), adapted from
experiments/v1/oct11_synovitis_crops/deploy/syn_stage.py. Flat: no experiment imports, so it can be copied into a Kaggle package.

Per study:
  1. knee_infer.study_series: the CPU path, byte-identical to the training cache fov160_px400 (uint8 (n, 400, 400), 0.4 mm/px);
  2. Basin's mirror detector (p > 0.9) per series, flipped back like the training flip table (sagittal: stack reversed, pos -> -pos;
     coronal / axial: columns mirrored): the experts were trained in the side-corrected view;
  3. localiser v3 (18 structures: v2 + femur / tibia) on every slice -> series boxes (presence-weighted mean over slices with
     p > 0.5) -> bone bags with the training rules of each expert's config (bags, planes, series, trim, crop_scale);
  4. each expert = mean Fracture / Contusion logits over its fold models (k_eval windows per bag, evenly spread).
Members of kind "fsmain" (pipeline-1 main models trained with series_keep=fs, e.g. runs/m_fs) are scored before step 2, as the main
models are (no side correction): fat-suppressed series only, knee_infer.windows (k_eval evenly spread (series, centre) pairs in the
5-95% span = data.StudyDataset eval windows), KneeNet, mean Fracture / Contusion logits over the fold models.
predict() -> {member name: DataFrame (study x Fracture, Contusion logits)}; studies that fail are missing (logged).
Package use (submissions/oct12_fc_3pipe, reusable in a combined package): config.json "fc" = {"members": {name: {kind, files, cfg}},
"loc", "detector", "recipe"}; ensemble.py calls predict_pkg() and apply_recipe() (blend with the pipelines' F / C logits). Source:
experiments/v1/oct11_fracture_contusion/deploy/bone_stage.py (verified vs the experiment on gold-58) + the two package functions.
Needs knee_infer.py, basin_flipdet.py, model.py of this toolchain (the same files the OA stage uses)."""
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import knee_infer as ki
from basin_flipdet import FlipNet, det_input
from model import KneeNet

STRUCTS_V3 = ["comp_medial", "comp_lateral", "pf_joint", "patella", "joint", "med_tib_cart", "lat_tib_cart", "pat_cart", "fem_cart", "med_men",
              "lat_men", "notch", "mcl", "suprapatellar", "hoffa", "popliteal", "femur", "tibia"]
STRUCT = {"femur": "femur", "tibia": "tibia", "patella": "patella", "knee": "joint"}
LOC_BAG = {"c_lfc": "femur", "c_mfc": "femur", "c_fem": "femur", "c_ltp": "tibia", "c_mtp": "tibia", "c_tib": "tibia", "c_fib": "tibia",
           "c_pat": "patella", "f_fem": "femur", "f_tib": "tibia", "f_fib": "tibia", "f_pat": "patella"}
FEM_H, TIB_H, MARGIN, TIB_XM, MIN_MM, MAX_MM = 72.0, 64.0, 8.0, 16.0, 48.0, 120.0
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
P_MIN, TOP_FALLBACK, LOC_PX, MM_PER_PX, PX = 0.5, 3, 128, 0.4, 400
AMP = getattr(torch, os.environ.get("BONE_AMP_DTYPE", "float16"))
FINDINGS_FC = {"Fracture": 11, "Contusion": 10}                      # column of each finding in the 12 main outputs (common.paths order)   # T4: fp16; local checks may set bfloat16 = the experiment
FC = ["Fracture", "Contusion"]
STATS = {}


def sincos(pos, n_freq=8, scale=60.0):
    f = torch.exp2(torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    a = pos[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Loc(nn.Module):
    """scripts/loc_train.py Loc (MobileNetV3, 3-slice 128 px windows -> presence logit + box per structure)."""
    def __init__(self, arch="mobilenetv3_large_100", structs=STRUCTS_V3):
        super().__init__()
        self.n_s = len(structs)
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.head = nn.Sequential(nn.Linear(d + 5, 256), nn.ReLU(), nn.Linear(256, self.n_s * 5))
        c = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(c["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(c["std"]).view(1, 3, 1, 1) * 255)

    def forward(self, x, aux):
        x = ((x.float() - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        o = self.head(torch.cat([self.backbone(x), aux], 1)).view(-1, self.n_s, 5)
        return o[..., 0], torch.sigmoid(o[..., 1:])


def aux_features(plane, pos, n):
    p = np.zeros((n, 3), np.float32)
    p[:, PLANE_ID[plane]] = 1
    return np.concatenate([p, (np.asarray(pos, np.float32) / 60)[:, None], (np.arange(n) / max(n - 1, 1)).astype(np.float32)[:, None]], 1)


class BoneNet(nn.Module):
    """main/bone_train.py BoneNet (per-output attention pooling over the allowed bags; out_fs outputs read FS windows only)."""
    def __init__(self, backbone, out_bags, n_bags, n_types, out_fs, fs_type, hidden=256, bagsum=False, bb_pool="avg"):
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        self.backbone = timm.create_model("resnet50" if backbone == "radimagenet_resnet50" else backbone, pretrained=False, num_classes=0,
                                          global_pool=bb_pool)
        if backbone == "radimagenet_resnet50":   # bone_train.encoder: timm resnet50 with RadImageNet weights, ImageNet mean / std
            self.backbone.pretrained_cfg = dict(self.backbone.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.type_emb, self.plane_emb, self.bag_emb, self.pos_proj = nn.Embedding(n_types, d), nn.Embedding(3, d), nn.Embedding(n_bags, d), nn.Linear(16, d)
        self.norm = nn.LayerNorm(d)
        n_out = out_bags.shape[0]
        self.att_v, self.att_w = nn.Linear(d, hidden), nn.Linear(hidden, n_out)
        self.cls_w = nn.Parameter(torch.zeros(n_out, n_bags, d) if bagsum else torch.zeros(n_out, d))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("out_bags", torch.as_tensor(out_bags).T.contiguous(), persistent=False)
        self.register_buffer("out_fs", torch.as_tensor(out_fs), persistent=False)
        self.register_buffer("fs_type", torch.as_tensor(fs_type), persistent=False)

    def forward(self, x, mask, typ, plane, pos, bag, crops=False):
        """crops=True (oct13_f5_win, no bagsum): per-crop attention logits s and logits u (b, k, n_out); the study logit is exactly
        sum_k softmax_k(s) u."""
        b, k = mask.shape
        f = self.backbone(x).float().view(b, k, -1)
        h = self.norm(f + self.type_emb(typ) + self.plane_emb(plane) + self.bag_emb(bag.clamp(min=0)) + self.pos_proj(sincos(pos, scale=240.0)))
        allowed = self.out_bags[bag.clamp(min=0)] & mask[..., None]
        if self.out_fs.any():
            fs = allowed & self.fs_type[typ][..., None]
            keep = ~self.out_fs | ~fs.any(1)
            allowed = torch.where(keep[:, None, :], allowed, fs)
        s = self.att_w(torch.tanh(self.att_v(h))).masked_fill(~allowed, float("-inf"))
        if self.bagsum:
            out = self.cls_b.expand(b, -1)
            for j in range(self.n_bags):
                a = torch.softmax(s.masked_fill((bag != j)[..., None], float("-inf")), 1).nan_to_num(0.0)
                out = out + (torch.einsum("bkn,bkd->bnd", a, h) * self.cls_w[:, j]).sum(-1)
            return out
        if crops:
            return s, h @ self.cls_w.T + self.cls_b
        a = torch.softmax(s, 1).nan_to_num(0.0)
        return (torch.einsum("bkn,bkd->bnd", a, h) * self.cls_w).sum(-1) + self.cls_b


def out_masks(cfg, names):
    outs = cfg["outs"].split(",")
    ob = np.ones((len(outs), len(names)), bool)
    for j, o in enumerate(outs):
        if cfg.get("loc_restrict", True) and o in LOC_BAG and LOC_BAG[o] in names:
            ob[j] = [b == LOC_BAG[o] for b in names]
    fs_outs = cfg.get("fs_outs", "").split(",")
    return ob, np.array([o in fs_outs for o in outs])


def crop_box(bag, plane, b, crop_scale, pat_margin=MARGIN):
    """bone_data.crop_box: series box (y0, y1, x0, x1) in cache px -> (cy, cx, side_px)."""
    y0, y1, x0, x1 = b
    m = (pat_margin if bag == "patella" else MARGIN) / MM_PER_PX
    xm = (TIB_XM if bag == "tibia" else pat_margin if bag == "patella" else MARGIN) / MM_PER_PX
    if plane != "Axial" and bag == "femur":
        y0 = max(y0, y1 - FEM_H / MM_PER_PX)
    elif plane != "Axial" and bag == "tibia":
        y1 = min(y1, y0 + TIB_H / MM_PER_PX)
    side = float(np.clip(max(y1 - y0 + 2 * m, x1 - x0 + 2 * xm), MIN_MM / MM_PER_PX, MAX_MM / MM_PER_PX)) * crop_scale
    return (y0 + y1) / 2, (x0 + x1) / 2, side


def series_box(p, b, j, px=PX):
    pres = p[:, j] > P_MIN
    if not pres.any():
        return None
    return (b[pres, j] * p[pres, j, None]).sum(0) / p[pres, j].sum() * px


@torch.no_grad()
def localise(loc, vol, plane, pos, dev):
    n = len(vol)
    small = np.stack([cv2.resize(a, (LOC_PX, LOC_PX), interpolation=cv2.INTER_AREA) for a in vol])
    w = np.stack([small[np.clip(np.arange(n) + d, 0, n - 1)] for d in (-1, 0, 1)], 1)
    p, b = loc(torch.from_numpy(w).to(dev), torch.from_numpy(aux_features(plane, pos, n)).to(dev))
    return torch.sigmoid(p).float().cpu().numpy(), b.float().cpu().numpy()


def study_bags(series, locs, cfg):
    """series: list of dicts vol, plane, type, pos; locs: per series (p, b) or None -> bag name -> windows (si, i, cy, cx, side_px,
    type, plane, pos, p), with main/bone_data.py build_bags + finalize rules (crops in the side-corrected view: no mirroring)."""
    names, planes = cfg["bags"].split(","), cfg.get("planes", "")
    out = {n: [] for n in names}
    for si, (s, lb) in enumerate(zip(series, locs)):
        n = len(s["vol"])
        if n < 3 or lb is None or (planes and s["plane"] not in planes.split(",")):
            continue
        p, b = lb
        for bag in names:
            j = STRUCTS_V3.index(STRUCT[bag])
            box = series_box(p, b, j)
            if box is None:
                continue
            cy, cx, side_px = (PX / 2, PX / 2, 160 / MM_PER_PX) if bag == "knee" else crop_box(bag, s["plane"], box, cfg.get("crop_scale", 1.0), cfg.get("pat_margin", MARGIN))
            keep = [i for i in range(1, n - 1) if p[i, j] > P_MIN]
            if len(keep) > 3:
                cut = int(round(cfg.get("trim", 0.05) * len(keep)))
                keep = keep[cut:len(keep) - cut]
            out[bag] += [(si, i, cy, cx, side_px, s["type"], PLANE_ID[s["plane"]], float(s["pos"][i]), float(p[i, j])) for i in keep]
    if cfg.get("series", "all") != "all":
        want = cfg["series"] == "fs"
        for key, c in out.items():
            out[key] = [w for w in c if ki.SERIES_TYPES[w[5]].endswith("_FS") == want] or c
    for key in names:
        if not out[key]:
            donor = [x for k in names if k != key for x in out[k]]
            out[key] = sorted(donor, key=lambda w: -w[-1])[:TOP_FALLBACK]
    return out


def crop(img3, cy, cx, side, px):
    h = int(round(side))
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    out = np.zeros((3, h, h), np.uint8)
    sy, sx = max(y0, 0), max(x0, 0)
    ey, ex = min(y0 + h, img3.shape[1]), min(x0 + h, img3.shape[2])
    if ey > sy and ex > sx:
        out[:, sy - y0:ey - y0, sx - x0:ex - x0] = img3[:, sy:ey, sx:ex]
    if h == px:
        return out
    return np.stack([cv2.resize(np.ascontiguousarray(c), (px, px), interpolation=cv2.INTER_AREA if h > px else cv2.INTER_LINEAR) for c in out])


def study_tensors(series, bags, names, m, px):
    kt = m * len(names)
    x = np.zeros((kt, 3, px, px), np.uint8)
    mask, bag = np.zeros(kt, bool), np.full(kt, -1, np.int64)
    typ, plane, pos = np.zeros(kt, np.int64), np.zeros(kt, np.int64), np.zeros(kt, np.float32)
    k = 0
    for j, name in enumerate(names):
        c = bags[name]
        for i in np.unique(np.linspace(0, len(c) - 1, min(m, len(c))).round().astype(int)) if c else []:
            si, sl, cy, cx, side, t, pl, ps, _ = c[i]
            x[k] = crop(series[si]["vol"][[sl - 1, sl, sl + 1]], cy, cx, side, px)
            mask[k], bag[k], typ[k], plane[k], pos[k] = True, j, t, pl, ps
            k += 1
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a[None])).to(dt)
    return torch.from_numpy(x[None]), t(mask, torch.bool), t(typ, torch.long), t(plane, torch.long), t(pos, torch.float32), t(bag, torch.long)


def correct_sides(series, det, dev, thr=0.9):
    """Basin's mirror detector on each series; flipped back like the training flip table. Returns the number flipped."""
    n = 0
    with torch.no_grad():
        for s in series:
            if len(s["vol"]) < 3:
                continue
            p = torch.sigmoid(det(torch.from_numpy(det_input(s["vol"], s["plane"]))[None].to(dev))).item()
            if p > thr:
                n += 1
                if s["plane"] == "Sagittal":
                    s["vol"], s["pos"] = s["vol"][::-1].copy(), -np.asarray(s["pos"])[::-1].copy()
                else:
                    s["vol"] = s["vol"][..., ::-1].copy()
    return n


def load_fsmain(members, dev):
    """members of kind "fsmain": {name: {"kind": "fsmain", "cfg": main train config, "files": [...]}} -> {name: (models, cfg)}."""
    out = {}
    for name, e in members.items():
        if e.get("kind") != "fsmain":
            continue
        c = e["cfg"]
        ms = []
        for f in e["files"]:
            m = KneeNet(c["backbone"], n_out=12, pretrained=False, hidden=c["hidden"], drop=c["drop"], use_type=c["use_type"],
                        use_pos=c["use_pos"], head=c["head"])
            m.load_state_dict(torch.load(f, map_location="cpu"))
            ms.append(m.to(dev).to(memory_format=torch.channels_last).eval())
        out[name] = (ms, c)
    return out


@torch.no_grad()
def fsmain_logits(fsm, series, dev):
    """(Fracture, Contusion) mean logits of each fsmain member on the fat-suppressed series of one study (all series if none)."""
    out = {}
    fs = [s for s in series if ki.SERIES_TYPES[s["type"]].endswith("_FS") and len(s["vol"]) >= 3] or series
    for name, (ms, c) in fsm.items():
        crop_px = min(PX, int(round(c["crop_mm"] / MM_PER_PX)))
        slices, idx, mask, typ, pos = ki.windows(fs, c["k_eval"], crop_px, tuple(c["span"]))
        x = torch.from_numpy(slices).to(dev)[:, None].float().div_(255)
        x = F.interpolate(x, size=(c["img"], c["img"]), mode="bilinear", antialias=True, align_corners=False)[:, 0]
        x = x[torch.from_numpy(idx).to(dev).flatten()].view(-1, 3, c["img"], c["img"])
        cb = ms[0].backbone.pretrained_cfg
        mean = torch.tensor(cb["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cb["std"], device=dev).view(1, 3, 1, 1)
        xv = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)[None]).to(dev, dt)
        lg = []
        for m in ms:
            with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
                lg.append(m(xv, t(mask, torch.bool), t(typ, torch.long), t(pos, torch.float32)).float()[0, [FINDINGS_FC["Fracture"], FINDINGS_FC["Contusion"]]])
        v = torch.stack(lg).mean(0).cpu().numpy()
        if not np.isfinite(v).all():
            raise FloatingPointError(f"non-finite fsmain {name} logit")
        out[name] = v
    return out


def load_experts(members, dev):
    """members: {name: {"cfg": bone_train config, "files": [weight paths]}} -> {name: (models, cfg, names, mean, std)}."""
    experts = {}
    fs_type = np.array([t.endswith("_FS") for t in ki.SERIES_TYPES])
    for name, e in members.items():
        if e.get("kind") == "fsmain":
            continue
        c = e["cfg"]
        names = c["bags"].split(",")
        ob, ofs = out_masks(c, names)
        ms = []
        for f in e["files"]:
            m = BoneNet(c["backbone"], ob, len(names), len(ki.SERIES_TYPES), ofs, fs_type, c.get("hidden", 256), c.get("bagsum", False), c.get("bb_pool", "avg"))
            m.load_state_dict(torch.load(f, map_location="cpu"))
            ms.append(m.to(dev).to(memory_format=torch.channels_last).eval())
        cfgb = ms[0].backbone.pretrained_cfg
        mean = torch.tensor(cfgb["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfgb["std"], device=dev).view(1, 3, 1, 1)
        experts[name] = (ms, c, names, mean, std)
    return experts


CROP_KEYS = {"bags": None, "planes": "", "series": "all", "trim": None, "crop_scale": 1.0, "loc": None, "pat_margin": MARGIN, "k_eval": None,
             "px": None, "mono": False}      # crop configuration keys with the defaults study_bags / study_tensors use


def fuse_crops(S, U, groups, gate=20):
    """oct13_f5_win crop-level fusion (experiments/v2/oct13_deep_ensembling_test/wfuse.py spec 'geo|pm|g20|wmean'): S, U (models, k,
    outputs) over the same crops; fold models of one member (group) are averaged first (window probs); relevance = softmax over crops
    of the mean attention logit over all models; value = mean over members of the window prob; the gate keeps the `gate` crops with
    the highest relevance per output; pooled prob = relevance-weighted mean of the values -> logit."""
    r = torch.softmax(S.mean(0), 0)                                            # (k, outputs)
    P = torch.stack([torch.sigmoid(U[[i for i, g in enumerate(groups) if g == gg]]).mean(0) for gg in sorted(set(groups))]).mean(0)
    if gate and r.shape[0] > gate:
        keep = torch.zeros_like(r).scatter_(0, torch.topk(r, gate, 0).indices, 1.0)
        r = r * keep
        r = r / r.sum(0, keepdim=True)
    p = (r * P).sum(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


@torch.no_grad()
def expert_logits(experts, series, locs, dev, fuse=None):
    """fuse (config fc "fuse", oct13_f5_win): also returns out["fused"] = crop-level fusion of every member's fold models over the
    shared crops (members must share the crop configuration)."""
    out, cache, crops = {}, {}, []
    for name, (ms, c, names, mean, std) in experts.items():
        key = tuple(str(c.get(k, d) if c.get(k) is not None else d) for k, d in CROP_KEYS.items()) + (tuple(names),)
        if key not in cache:                       # members with one crop configuration share the crops (built once)
            bags = study_bags(series, locs, c)
            cache[key] = tuple(t.to(dev) for t in study_tensors(series, bags, names, c["k_eval"], c["px"]))
        x, mask, typ, plane, pos, bag = cache[key]
        xf = x.flatten(0, 1).float() / 255
        if c.get("mono", False):                  # data.gpu_prep mono: the centre slice in all three channels (RadImageNet R50)
            xf = xf[:, 1:2].expand(-1, 3, -1, -1)
        xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
        outs = c["outs"].split(",")
        jj = [outs.index("frac"), outs.index("cont")]
        lg = []
        for m in ms:
            with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
                if fuse and not c.get("nofuse", False):     # oct17_probe_cont_fs: a 'nofuse' member (own crop config) stays out of the fusion
                    s, u = m(xf, mask, typ, plane, pos, bag, crops=True)
                    s, u = s.float()[0][:, jj], u.float()[0][:, jj]
                    n = int(mask[0].sum())
                    assert bool(mask[0, :n].all()), "crop fusion expects the valid crops first"
                    for fs in member_sets(c):     # a member may join several fusions (its crops, no second model pass)
                        crops.append((key, fs, name, s[:n], u[:n]))
                    lg.append((torch.softmax(s[:n], 0) * u[:n]).sum(0))
                else:
                    lg.append(m(xf, mask, typ, plane, pos, bag).float()[0, jj])
        v = torch.stack(lg).mean(0).cpu().numpy()
        if not np.isfinite(v).all():
            raise FloatingPointError(f"non-finite expert {name} logit")
        out[name] = v
    if fuse:                   # oct20_probe_cont_allmem: one fusion per member 'fuse_set' (absent = the shipped "fused"; "cont" = "fused_cont")
        for fs in sorted({f for _, f, *_ in crops}):
            cs = [c for c in crops if c[1] == fs]
            assert len({k for k, *_ in cs}) == 1, f"crop fusion {fs or 'shipped'} needs one shared crop configuration"   # nofuse members add no crops
            names = [c[2] for c in cs]
            z = fuse_crops(torch.stack([c[3] for c in cs]), torch.stack([c[4] for c in cs]), [sorted(set(names)).index(n) for n in names],
                           fuse["gate"]).cpu().numpy()
            if not np.isfinite(z).all():
                raise FloatingPointError(f"non-finite fused crop logit ({fs or 'shipped'})")
            out[fused_key(fs)] = z
    return out


def fused_key(fs):
    return "fused" + (f"_{fs}" if fs else "")


def member_sets(c):
    """oct20_probe_cont_allmem: the fusion groups of a member cfg: "fuse_sets" (list) or "fuse_set" (one name); absent = the shipped [""]."""
    assert not ("fuse_sets" in c and "fuse_set" in c), "fuse_sets and fuse_set are exclusive"
    return list(c["fuse_sets"]) if "fuse_sets" in c else [c.get("fuse_set", "")]


class Studies(torch.utils.data.Dataset):
    def __init__(self, root, studies, fs_of):
        self.root, self.studies, self.fs_of = root, list(studies), fs_of

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        try:
            series, _ = ki.study_series(f"{self.root}/{s}", self.fs_of)
            return s, [{"vol": d["vol"], "plane": d["plane"], "type": int(d["type"]), "pos": np.asarray(d["pos"], np.float32)} for d in series], ""
        except Exception as e:  # noqa: BLE001
            return s, [], f"{type(e).__name__}: {e}"


def predict(root, studies, fs_of, members, loc_path, det_path, workers=4, fuse=None):
    """-> {expert name: DataFrame (study x Fracture, Contusion logits)} (+ "fused" when fuse is set: the crop-level fusion of all
    members' fold models); failed studies are missing (STATS['failed'])."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    loc = Loc()
    loc.load_state_dict(torch.load(loc_path, map_location="cpu"))
    loc = loc.to(dev).to(memory_format=torch.channels_last).eval()
    det = FlipNet(pretrained=False)
    det.load_state_dict(torch.load(det_path, map_location="cpu"))
    det = det.to(dev).eval()
    experts = load_experts(members, dev)
    fsm = load_fsmain(members, dev)
    dl = torch.utils.data.DataLoader(Studies(root, studies, fs_of), batch_size=None, num_workers=workers, collate_fn=lambda x: x,
                                     prefetch_factor=2 if workers else None)
    fsets = sorted({f for e in members.values() if e["kind"] == "bone" and not e["cfg"].get("nofuse", False) for f in member_sets(e["cfg"])})
    rows = {k: {} for k in list(experts) + list(fsm) + ([fused_key(f) for f in fsets] if fuse else [])}
    STATS.update(failed=[], flipped=0)
    for sid, series, err in dl:
        try:
            if err or not series:
                raise RuntimeError(err or "no readable series")
            for k, v in fsmain_logits(fsm, series, dev).items():      # main-type members: before the side correction (as trained)
                rows[k][sid] = v
            STATS["flipped"] += correct_sides(series, det, dev)
            locs = [localise(loc, s["vol"], s["plane"], s["pos"], dev) if len(s["vol"]) >= 3 else None for s in series]
            for k, v in expert_logits(experts, series, locs, dev, fuse).items():
                rows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            STATS["failed"].append(sid)
            print(f"bone_stage: study {sid} failed ({type(e).__name__}: {e})", flush=True)
    STATS["wall_s"] = time.time() - t0
    return {k: pd.DataFrame.from_dict(v, orient="index", columns=["Fracture", "Contusion"]) for k, v in rows.items()}


def predict_pkg(root, studies, fs_of, pkg, fc, workers=4):
    """config.json "fc" block -> {member: DataFrame (study x Fracture, Contusion logits)}; member files, localiser and detector are
    paths inside the package dir."""
    members = {n: {"kind": e["kind"], "cfg": e["cfg"], "files": [str(Path(pkg) / f) for f in e["files"]]} for n, e in fc["members"].items()}
    return predict(root, studies, fs_of, members, str(Path(pkg) / fc["loc"]), str(Path(pkg) / fc["detector"]), workers, fc.get("fuse"))


def apply_recipe(b, member_logits, recipe):
    """b: DataFrame (study x Fracture, Contusion) = the package's F / C logits (mean of its pipelines). Per finding f (recipe
    "per_finding"): e = mean over its members of (logit - mean) / sd (recipe member_z); o = (1 - w) b + w ((e - mu_e) / sd_e * sd_b + mu_b)
    (stats from the folds 0-2 OOF). A study without expert logits (stage failure) keeps b."""
    out = b.copy()
    for f, r in recipe["per_finding"].items():
        zs = [(member_logits[m][f] - recipe["member_z"][m][f]["mean"]) / recipe["member_z"][m][f]["sd"] for m in r["members"]]
        e = (sum(z.reindex(b.index) for z in zs) / len(zs)).astype(np.float64)
        s = r["stats"]
        o = (1 - r["w"]) * b[f] + r["w"] * ((e - s["mu_e"]) / s["sd_e"] * s["sd_b"] + s["mu_b"])
        out[f] = o.where(e.notna(), b[f])
    return out


def apply_recipe_fused(b, fused, recipe, fuse):
    """oct13_f5_win: the F / C recipe with its expert term e replaced by the crop-level fusion of all members (config fc "fuse"):
    e_f = (fused - mu_f) / sd_f * sd_e + mu_e (aligned to the deployed e's OOF distribution, folds 0-2), then o = (1 - w) b + w z(e_f)
    as apply_recipe. A study without a fused logit is NaN here (ensemble.py keeps apply_recipe's member-based value)."""
    out = b.copy()
    for f, r in recipe["per_finding"].items():
        al, st = fuse["align"][f], r["stats"]
        ef = (fused[f].reindex(b.index) - al["mu_f"]) / al["sd_f"] * al["sd_e"] + al["mu_e"]
        out[f] = (1 - r["w"]) * b[f] + r["w"] * ((ef - st["mu_e"]) / st["sd_e"] * st["sd_b"] + st["mu_b"])
    return out
