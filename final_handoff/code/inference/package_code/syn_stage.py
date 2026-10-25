"""oct18_probe_acl_b3g (Experts V3, 2026-10-18; = oct17_probe_acl_zoom's file + member cfg 'keep_bags' in study_tensors / CROP_KEYS): the 0.961 package's syn_stage.py + additive changes for one extra crop member that is
not a synovitis expert (the MCL zoom w5b_mcl / the ACL notch40 zoom z1_acl of experiments/v3/oct16_experts): bag tables = that experiment's syn_data.py (adds mclz and
the other region / zoom views; the deployed bags keep their entries), per-output view routing (config out_bag), img_syn optional, every
text output returned as '<expert>@<out>'. The deployed members' path is unchanged (same code for them).
Test-time path of the synovitis crop experts (main/syn_data.py + main/syn_train.py SynNet), adapted from the OA stage
(submissions/oct11_oa_e3/oa_stage.py: localiser, series boxes, mirror detector). Flat: no experiment imports, so it can be copied
into a Kaggle package as is.

Per study:
  1. knee_infer.study_series: the CPU path, byte-identical to the training cache fov160_px400 (uint8 (n, 400, 400), 0.4 mm/px);
  2. Basin's mirror detector (p > 0.9) per series, flipped back like the training flip table (sagittal: stack reversed, pos -> -pos;
     coronal / axial: columns mirrored): the experts were trained in the side-corrected view;
  3. localiser v2 (16 structures) on every slice -> series boxes (presence-weighted mean over slices with p > 0.5) -> bags with the
     training rules of each expert's config (bags, planes, fs_only, trim, crop_scale, plane_bags, clamp);
  4. each expert = mean img_Synovitis logit over its fold models (k_eval windows per bag, evenly spread).
predict() -> {expert name: Series of synovitis logits} + {expert name@img_eff: Series of image-read Effusion logits} (oct13_f5_eff, experts
with an img_eff output); studies that fail are missing (logged)."""
import collections
import math
import os
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

import knee_infer as ki
from basin_flipdet import FlipNet, det_input

STRUCTS_V2 = ["comp_medial", "comp_lateral", "pf_joint", "patella", "joint", "med_tib_cart", "lat_tib_cart", "pat_cart", "fem_cart", "med_men",
              "lat_men", "notch", "mcl", "suprapatellar", "hoffa", "popliteal"]
STRUCT = {"pouch": "suprapatellar", "hoffa": "hoffa", "post": "popliteal", "notch": "notch", "knee": "joint", "gutter": "patella",
          "mcl": "mcl", "notchsc": "notch",           # oct16_experts: ligament expert bags (mcl: coronal + axial MCL box; notchsc: notch sag + cor)
          "patcart": "pat_cart", "pfj": "pf_joint", "latmen": "lat_men", "medmen": "med_men", "postz": "popliteal",
          "lattib": "lat_tib_cart", "mclz": "mcl",    # oct16_experts W batch (user U8): region zooms for PF / menisci / Baker's / LOA / MCL
          "medc": "comp_medial", "latc": "comp_lateral", "pfl": "pf_joint",   # ME1 merged multi-view (user U11): 100 mm compartment / PF locals
          "medc60": "comp_medial", "latc60": "comp_lateral", "pfl60": "pf_joint", "medc40": "comp_medial", "latc40": "comp_lateral",
          "pfl40": "pf_joint", "notch40": "notch", "knee140": "joint"}
PLANES_OF = {"pouch": ("Sagittal", "Coronal", "Axial"), "hoffa": ("Sagittal", "Coronal", "Axial"), "post": ("Sagittal", "Axial"),
             "notch": ("Sagittal", "Coronal", "Axial"), "knee": ("Sagittal", "Coronal", "Axial"), "gutter": ("Axial",),
             "mcl": ("Coronal", "Axial"), "notchsc": ("Sagittal", "Coronal"),
             "patcart": ("Axial", "Sagittal"), "pfj": ("Axial", "Sagittal"), "latmen": ("Sagittal", "Coronal"), "medmen": ("Sagittal", "Coronal"),
             "postz": ("Axial", "Sagittal"), "lattib": ("Coronal", "Sagittal"), "mclz": ("Coronal",),
             "medc": ("Coronal", "Sagittal"), "latc": ("Coronal", "Sagittal"), "pfl": ("Sagittal", "Axial"),
             "medc60": ("Coronal", "Sagittal"), "latc60": ("Coronal", "Sagittal"), "pfl60": ("Sagittal", "Axial"),
             "medc40": ("Coronal", "Sagittal"), "latc40": ("Coronal", "Sagittal"), "pfl40": ("Sagittal", "Axial"), "notch40": ("Sagittal", "Coronal"),
             "knee140": ("Sagittal", "Coronal", "Axial")}
CROP_MM = {"pouch": 80, "hoffa": 72, "post": 100, "notch": 64, "knee": 160, "gutter": 110, "mcl": 70, "notchsc": 64,
           "patcart": 48, "pfj": 72, "latmen": 48, "medmen": 48, "postz": 64, "lattib": 48, "mclz": 48, "medc": 100, "latc": 100, "pfl": 100,
           "medc60": 60, "latc60": 60, "pfl60": 60, "medc40": 40, "latc40": 40, "pfl40": 40, "notch40": 40, "knee140": 140}
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
P_MIN, TOP_FALLBACK, LOC_PX, MM_PER_PX, PX = 0.5, 3, 128, 0.4, 400
AMP = getattr(torch, os.environ.get("SYN_AMP_DTYPE", "float16"))   # T4: fp16; local checks may set bfloat16 = the experiment
STATS = {}


def sincos(pos, n_freq=8, scale=60.0):
    f = torch.exp2(torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    a = pos[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Loc(nn.Module):
    """experiments/v1/oct10_oa/scripts/loc_train.py Loc (MobileNetV3, 3-slice 128 px windows -> presence logit + box per structure)."""
    def __init__(self, arch="mobilenetv3_large_100", structs=STRUCTS_V2):
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


class SynNet(nn.Module):
    """main/syn_train.py SynNet (per-output attention pooling over the allowed bags; bagsum: per-bag readout)."""
    def __init__(self, backbone, out_bags, n_bags, n_types, hidden=256, bagsum=False):
        super().__init__()
        self.bagsum, self.n_bags = bagsum, n_bags
        if backbone == "radimagenet_resnet50":         # training: timm resnet50 + RadImageNet weights, ImageNet mean / std, mono input
            self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
            self.backbone.pretrained_cfg = dict(self.backbone.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.type_emb, self.plane_emb, self.bag_emb, self.pos_proj = nn.Embedding(n_types, d), nn.Embedding(3, d), nn.Embedding(n_bags, d), nn.Linear(16, d)
        self.norm = nn.LayerNorm(d)
        n_out = out_bags.shape[0]
        self.att_v, self.att_w = nn.Linear(d, hidden), nn.Linear(hidden, n_out)
        self.cls_w = nn.Parameter(torch.zeros(n_out, n_bags, d) if bagsum else torch.zeros(n_out, d))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("out_bags", torch.as_tensor(out_bags).T.contiguous(), persistent=False)

    def forward(self, x, mask, typ, plane, pos, bag, crops=False):
        """crops=True (oct13_f5_win, no bagsum): per-crop attention logits s and logits u (b, k, n_out); study logit = sum_k
        softmax_k(s) u."""
        b, k = mask.shape
        f = self.backbone(x).float().view(b, k, -1)
        h = self.norm(f + self.type_emb(typ) + self.plane_emb(plane) + self.bag_emb(bag.clamp(min=0)) + self.pos_proj(sincos(pos, scale=240.0)))
        allowed = self.out_bags[bag.clamp(min=0)] & mask[..., None]
        s = self.att_w(torch.tanh(self.att_v(h))).masked_fill(~allowed, float("-inf"))
        if crops:
            assert not self.bagsum
            return s, h @ self.cls_w.T + self.cls_b
        if self.bagsum:
            out = self.cls_b.expand(b, -1)
            for j in range(self.n_bags):
                a = torch.softmax(s.masked_fill((bag != j)[..., None], float("-inf")), 1).nan_to_num(0.0)
                out = out + (torch.einsum("bkn,bkd->bnd", a, h) * self.cls_w[:, j]).sum(-1)
            return out
        a = torch.softmax(s, 1).nan_to_num(0.0)
        return (torch.einsum("bkn,bkd->bnd", a, h) * self.cls_w).sum(-1) + self.cls_b


def bag_names(cfg):
    bags, planes = cfg["bags"].split(","), cfg.get("planes", "")
    keep = lambda p: not planes or p in planes.split(",")
    return [f"{b}:{p}" for b in bags for p in PLANES_OF[b] if keep(p)] if cfg.get("plane_bags") else bags


def out_bags(cfg, names):
    outs = cfg["outs"].split(",")
    ob = np.ones((len(outs), len(names)), bool)
    for kv in [x for x in cfg.get("out_bag", "").split("+") if x]:      # oct17: per-output view routing (syn_train order)
        o, bs = kv.split(":")
        ob[outs.index(o)] = [b.split(":")[0] in bs.split("|") for b in names]
    if cfg.get("bakers_post", True) and "bakers" in outs and "post" in cfg["bags"].split(","):
        ob[outs.index("bakers")] = [b.split(":")[0] == "post" for b in names]
    return ob


def series_box(p, b, j, px=PX):
    pres = p[:, j] > P_MIN
    if not pres.any():
        return None
    bb = (b[pres, j] * p[pres, j, None]).sum(0) / p[pres, j].sum() * px
    return (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2


@torch.no_grad()
def localise(loc, vol, plane, pos, dev):
    n = len(vol)
    small = np.stack([cv2.resize(a, (LOC_PX, LOC_PX), interpolation=cv2.INTER_AREA) for a in vol])
    w = np.stack([small[np.clip(np.arange(n) + d, 0, n - 1)] for d in (-1, 0, 1)], 1)
    p, b = loc(torch.from_numpy(w).to(dev), torch.from_numpy(aux_features(plane, pos, n)).to(dev))
    return torch.sigmoid(p).float().cpu().numpy(), b.float().cpu().numpy()


def study_bags(series, locs, cfg):
    """series: list of dicts vol, plane, type, pos; locs: per series (p, b) or None -> bag name -> windows (si, i, cy, cx, side_px,
    type, plane, pos, p), with main/syn_data.py build_bags + finalize rules (crops in the side-corrected view: no mirroring)."""
    bags, planes = cfg["bags"].split(","), cfg.get("planes", "")
    names = bag_names(cfg)
    out = {n: [] for n in names}
    for si, (s, lb) in enumerate(zip(series, locs)):
        n = len(s["vol"])
        if n < 3 or lb is None or (planes and s["plane"] not in planes.split(",")):
            continue
        p, b = lb
        for bag in bags:
            if s["plane"] not in PLANES_OF[bag]:
                continue
            j = STRUCTS_V2.index(STRUCT[bag])
            c = series_box(p, b, j)
            if c is None:
                continue
            cy, cx = (PX / 2, PX / 2) if bag in ("knee", "knee140") else c
            side_px = CROP_MM[bag] * cfg.get("crop_scale", 1.0) / MM_PER_PX
            if cfg.get("clamp"):
                lo, hi = min(side_px / 2, PX / 2), max(PX - side_px / 2, PX / 2)
                cy, cx = float(np.clip(cy, lo, hi)), float(np.clip(cx, lo, hi))
            keep = [i for i in range(1, n - 1) if p[i, j] > P_MIN]
            if len(keep) > 3:
                cut = int(round(cfg.get("trim", 0.1) * len(keep)))
                keep = keep[cut:len(keep) - cut]
            key = f"{bag}:{s['plane']}" if cfg.get("plane_bags") else bag
            out[key] += [(si, i, cy, cx, side_px, s["type"], PLANE_ID[s["plane"]], float(s["pos"][i]), float(p[i, j])) for i in keep]
    if cfg.get("fs_only"):
        for key, c in out.items():
            out[key] = [w for w in c if ki.SERIES_TYPES[w[5]].endswith("_FS")] or c
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


def study_tensors(series, bags, names, m, px, keep=None):
    """keep (oct18_probe_acl_b3g, member cfg 'keep_bags'): crops only for these bags; bag ids keep their index in the FULL bag list (bag
    embedding / routing as trained). Bag coordinates (study_bags) are still built for every bag, so the empty-bag donor rule is unchanged."""
    keep = set(keep.split(",")) if keep else None
    kt = m * (len(keep) if keep else len(names))
    x = np.zeros((kt, 3, px, px), np.uint8)
    mask, bag = np.zeros(kt, bool), np.full(kt, -1, np.int64)
    typ, plane, pos = np.zeros(kt, np.int64), np.zeros(kt, np.int64), np.zeros(kt, np.float32)
    k = 0
    for j, name in enumerate(names):
        if keep is not None and name not in keep:
            continue
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


def load_experts(members, dev):
    """members: {name: {"cfg": syn_train config, "files": [weight paths]}} -> {name: (models, cfg, names, mean, std)}."""
    experts = {}
    for name, e in members.items():
        c = e["cfg"]
        names = bag_names(c)
        ms = []
        for f in e["files"]:
            m = SynNet(c["backbone"], out_bags(c, names), len(names), len(ki.SERIES_TYPES), c.get("hidden", 256), c.get("bagsum", False))
            m.load_state_dict(torch.load(f, map_location="cpu"))
            ms.append(m.to(dev).to(memory_format=torch.channels_last).eval())
        cfgb = ms[0].backbone.pretrained_cfg
        mean = torch.tensor(cfgb["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfgb["std"], device=dev).view(1, 3, 1, 1)
        experts[name] = (ms, c, names, mean, std)
    return experts


CROP_KEYS = ("bags", "planes", "fs_only", "plane_bags", "trim", "crop_scale", "clamp", "k_eval", "px", "keep_bags")


def fuse_design(S, U, groups, gate=20):
    """oct13_f5_win synovitis crop fusion (experiments/v2/oct13_deep_ensembling_test wfuse spec 'mean|pm|g20|maxmix', w_synx.py
    'design'): S, U (models, k) of one design's members (both backbones) on its shared crops; relevance = mean over backbones of the
    backbone's mean attention prob; value = mean over backbones of the mean window prob; top-`gate` crops; pooled prob =
    0.7 x relevance-weighted mean + 0.3 x max gated value -> logit."""
    A = np.exp(S - S.max(1, keepdims=True))
    A = A / A.sum(1, keepdims=True)
    g = np.asarray(groups)
    r = np.stack([A[g == k].mean(0) for k in sorted(set(groups))]).mean(0)
    P = np.stack([(1 / (1 + np.exp(-U[g == k]))).mean(0) for k in sorted(set(groups))]).mean(0)
    keep = np.zeros_like(r)
    keep[np.argsort(-r)[:gate]] = 1.0
    r = r * keep
    r = r / r.sum()
    p = np.clip(0.7 * (r * P).sum() + 0.3 * np.where(r > 0, P, 0).max(), 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))


@torch.no_grad()
def expert_logits(experts, series, locs, dev, fuse=None):
    """Crops are built once per crop configuration (the fold models of one design share them) and reused. fuse (config syn "fuse",
    oct13_f5_win): the members' outputs come from the per-crop path (same math) and out["fused@<design>"] = fuse_design of each
    design's members (both backbones) on img_syn."""
    out, cache = {}, {}
    member_of = {n: (d, gi) for d, e in (fuse or {}).get("designs", {}).items() for gi, ns in enumerate(e.values()) for n in ns}
    crops = {}
    for name, (ms, c, names, mean, std) in experts.items():
        key = tuple(str(c.get(k)) for k in CROP_KEYS)
        if key not in cache:
            bags = study_bags(series, locs, c)
            cache[key] = tuple(t.to(dev) for t in study_tensors(series, bags, names, c["k_eval"], c["px"], c.get("keep_bags")))
        x, mask, typ, plane, pos, bag = cache[key]
        xf = x.flatten(0, 1).float() / 255
        if c.get("mono"):                                 # centre slice in all three channels (main/data.py gpu_prep mono)
            xf = xf[:, 1:2].expand(-1, 3, -1, -1)
        xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
        outs = c["outs"].split(",")
        lg = []
        for m in ms:
            with torch.autocast(dev.type, dtype=AMP, enabled=dev.type == "cuda"):
                if fuse:          # oct16_syn_cnx_probe: every member takes the per-crop path (0.961: all members are in a design)
                    sc, uc = m(xf, mask, typ, plane, pos, bag, crops=True)
                    sc, uc = sc.float()[0], uc.float()[0]
                    lg.append((torch.softmax(sc, 0).nan_to_num(0.0) * uc).sum(0))
                    if name in member_of:
                        j = outs.index("img_syn")
                        d, gi = member_of[name]
                        crops.setdefault(d, []).append((gi, sc[:, j].cpu().numpy().astype(np.float64), uc[:, j].cpu().numpy().astype(np.float64)))
                else:
                    lg.append(m(xf, mask, typ, plane, pos, bag).float()[0])
        o = torch.stack(lg).mean(0)
        if not torch.isfinite(o).all():
            raise FloatingPointError(f"non-finite expert {name} logit")
        if "img_syn" in outs:
            out[name] = float(o[outs.index("img_syn")])
        if "img_eff" in outs:                             # oct13_f5_eff: the image-read Effusion head of the same forward pass
            out[f"{name}@img_eff"] = float(o[outs.index("img_eff")])
        for j, on in enumerate(outs):                     # oct17: every text output of a non-synovitis member (e.g. w5b_mcl@mcl)
            if on not in ("img_syn", "img_eff"):
                out[f"{name}@{on}"] = float(o[j])
    for d, cs in crops.items():
        z = fuse_design(np.stack([c[1] for c in cs]), np.stack([c[2] for c in cs]), [c[0] for c in cs], fuse["gate"])
        out[f"fused@{d}"] = z if np.isfinite(z) else float("nan")
    return out


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
    """-> {expert name: Series (study -> synovitis logit)}; failed studies are missing (STATS['failed'])."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    loc = Loc()
    loc.load_state_dict(torch.load(loc_path, map_location="cpu"))
    loc = loc.to(dev).to(memory_format=torch.channels_last).eval()
    det = FlipNet(pretrained=False)
    det.load_state_dict(torch.load(det_path, map_location="cpu"))
    det = det.to(dev).eval()
    experts = load_experts(members, dev)
    dl = torch.utils.data.DataLoader(Studies(root, studies, fs_of), batch_size=None, num_workers=workers, collate_fn=lambda x: x,
                                     prefetch_factor=2 if workers else None)
    rows = collections.defaultdict(dict)
    STATS.update(failed=[], flipped=0)
    for sid, series, err in dl:
        try:
            if err or not series:
                raise RuntimeError(err or "no readable series")
            STATS["flipped"] += correct_sides(series, det, dev)
            locs = [localise(loc, s["vol"], s["plane"], s["pos"], dev) if len(s["vol"]) >= 3 else None for s in series]
            for k, v in expert_logits(experts, series, locs, dev, fuse).items():
                rows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            STATS["failed"].append(sid)
            print(f"syn_stage: study {sid} failed ({type(e).__name__}: {e})", flush=True)
    STATS["wall_s"] = time.time() - t0
    return {k: pd.Series(v, dtype=np.float64) for k, v in rows.items()}
