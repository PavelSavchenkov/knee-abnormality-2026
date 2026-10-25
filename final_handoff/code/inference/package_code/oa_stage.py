"""OA stage of the oct11_oa packages: compartment-crop OA experts (experiments/v1/oct10_oa) blended into the 3-pipeline OA logits.
Copied (flat, no experiment imports) from experiments/v1/oct10_oa: scripts/oa_expert_infer.py (verified test-time path, corr 0.9999 vs
OOF), main/oa_train.py OANet + encoder, main/model.py sincos, main/oa_data.py constants, scripts/loc_train.py Loc + aux_features.

Per study:
  1. knee_infer.study_series: the CPU path, byte-identical to the training cache fov160_px400 (uint8 (n, 400, 400), 0.4 mm/px);
  2. Basin's mirror detector (basin_flipdet, p > 0.9) per series: the experts were trained in the side-corrected view (flip table
     data/derived/series_side_final.csv); a series called mirrored is flipped back like the table did (sagittal: stack reversed,
     pos -> -pos reversed; coronal / axial: columns mirrored);
  3. localiser on every slice (128 px 3-slice windows; v2 with 16 structures for loc_version v2cor) -> series boxes -> compartment bags (medial / lateral: all planes, lateral
     coronal / axial crops mirrored; pf: sagittal + axial) -> k_eval windows per bag, crops of CROP_MM resized to the expert's px;
  4. each expert = mean logit over its fold models (OUTS = Medial OA, Lateral OA, PF OA, Medial / Lateral Meniscus: the experts'
     auxiliary meniscus heads come from the same forward pass, oct12_men_3pipe); mono experts (RadImageNet R50) read
     the centre slice in all three channels, as in training.
Members (config "oa" -> "members"): kind "expert" (the steps above) or kind "main" (main-pipeline effb0 models run through
knee_infer.predict on a temporary weights dir with their own config: the same-setting control, trained the normal way on all 12 labels).
Blend (ensemble.py): per OA finding F, with OOF statistics of the experiment (pooled folds 0-2, scripts in make_oa_config.py):
  z_m = (m - mu_m) * sd_b / sd_m + mu_b per member;  o = (1 - w) * b + w * mean_m(z_m);  out = o_F - b_F * (c0 + c . other two o).
A study a member cannot read keeps the pipeline blend for OA (logged)."""
import json
import math
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

import fc_stage
import knee_infer as ki
from basin_flipdet import FlipNet, det_input

OA = ["Medial OA", "Lateral OA", "PF OA"]
MEN = ["Medial Meniscus", "Lateral Meniscus"]
OUTS = OA + MEN          # the experts' 5 outputs (oa_train.py OUTS): oct12_men_3pipe also uses their meniscus heads
OUT_BAG = [0, 1, 2, 0, 1]
BAGS = ["medial", "lateral", "pf"]
STRUCTS = ["comp_medial", "comp_lateral", "pf_joint", "patella", "joint"]
STRUCT = {"medial": "comp_medial", "lateral": "comp_lateral", "pf": "pf_joint"}
STRUCTS_V2 = STRUCTS + ["med_tib_cart", "lat_tib_cart", "pat_cart", "fem_cart", "med_men", "lat_men", "notch", "mcl", "suprapatellar",
                        "hoffa", "popliteal"]
STRUCT_V2 = {"medial": "med_tib_cart", "lateral": "lat_tib_cart", "pf": "pat_cart"}
PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_ID = {p: i for i, p in enumerate(PLANES)}
PLANES_OF = {"medial": ("Sagittal", "Coronal", "Axial"), "lateral": ("Sagittal", "Coronal", "Axial"), "pf": ("Sagittal", "Axial")}
CROP_MM = {("medial", "Sagittal"): 72, ("medial", "Coronal"): 56, ("medial", "Axial"): 56,
           ("lateral", "Sagittal"): 72, ("lateral", "Coronal"): 56, ("lateral", "Axial"): 56,
           ("pf", "Sagittal"): 64, ("pf", "Axial"): 64}
P_MIN, TRIM, TOP_FALLBACK, LOC_PX, MM_PER_PX = 0.5, 0.15, 3, 128, 0.4
STATS = {}
AMP = getattr(torch, os.environ.get("OA_AMP_DTYPE", "float16"))   # T4: fp16; local checks may set bfloat16 = the experiment


def sincos(pos, n_freq=8, scale=60.0):
    f = torch.exp2(torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    a = pos[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Loc(nn.Module):
    def __init__(self, arch="mobilenetv3_large_100", structs=STRUCTS):
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


class OANet(nn.Module):
    """main/oa_train.py OANet (attention / lse / topk / hybrid pooling over each output's own bag)."""
    def __init__(self, backbone, n_types, hidden=256, split=False, bml=False, lse_r=0.0, topk=0, hybrid=False, pl_head=False):
        super().__init__()
        assert not (split and pl_head)
        self.lse_r, self.topk, self.hybrid = lse_r, topk, hybrid
        self.backbone = timm.create_model("resnet50" if backbone == "radimagenet_resnet50" else backbone, pretrained=False, num_classes=0)
        if backbone == "radimagenet_resnet50":   # oa_train.encoder: timm resnet50 with RadImageNet weights, ImageNet mean / std
            self.backbone.pretrained_cfg = dict(self.backbone.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        with torch.no_grad():
            d = self.backbone(torch.zeros(1, 3, 64, 64)).shape[1]
        self.type_emb, self.plane_emb, self.pos_proj = nn.Embedding(n_types, d), nn.Embedding(3, d), nn.Linear(16, d)
        self.norm = nn.LayerNorm(d)
        out_bag = OUT_BAG + ([0, 1, 2] if bml else []) + ([0, 1, 2] if pl_head else [])   # pl_head (oct16_loa_bundle): 3 'T:' OA heads last
        self.head_of = ([0, 0, 1, 2, 2] + ([3, 3, 4] if bml else [])) if split else list(range(len(out_bag)))
        n_head = max(self.head_of) + 1
        self.att_v, self.att_w = nn.Linear(d, hidden), nn.Linear(hidden, n_head)
        self.cls_w = nn.Parameter(torch.zeros(n_head, d))
        self.cls_w2 = nn.Parameter(torch.zeros(n_head, d)) if hybrid else None
        self.cls_b = nn.Parameter(torch.zeros(len(out_bag)))
        self.register_buffer("out_bag", torch.tensor(out_bag), persistent=False)

    def inst(self, h, allowed, w):
        return torch.einsum("bkd,nd->bkn", h, w[self.head_of]).masked_fill(~allowed, float("-inf"))

    def lme(self, s, allowed):
        return (torch.logsumexp(self.lse_r * s, 1) - torch.log(allowed.sum(1).clamp(min=1).float())) / self.lse_r

    def forward(self, x, mask, typ, plane, pos, bag, crops=False):
        """crops=True (oct13_f5_win; plain attention head): per-crop attention logits s (-inf outside an output's bag) and logits u
        (b, k, n_out); the study logit is exactly sum_k softmax_k(s) u."""
        b, k = mask.shape
        f = self.backbone(x).float().view(b, k, -1)
        h = self.norm(f + self.type_emb(typ) + self.plane_emb(plane) + self.pos_proj(sincos(pos, scale=240.0)))
        allowed = (bag[..., None] == self.out_bag) & mask[..., None]
        if crops:
            assert not self.topk and not self.lse_r and not self.hybrid
            s = self.att_w(torch.tanh(self.att_v(h)))[..., self.head_of].masked_fill(~allowed, float("-inf"))
            return s, h @ self.cls_w[self.head_of].T + self.cls_b
        if self.topk:
            v = self.inst(h, allowed, self.cls_w).topk(min(self.topk, k), dim=1).values
            fin = torch.isfinite(v)
            return v.where(fin, torch.zeros_like(v)).sum(1) / fin.sum(1).clamp(min=1) + self.cls_b
        if self.lse_r and not self.hybrid:
            return self.lme(self.inst(h, allowed, self.cls_w), allowed) + self.cls_b
        s = self.att_w(torch.tanh(self.att_v(h)))[..., self.head_of].masked_fill(~allowed, float("-inf"))
        att = (torch.einsum("bkn,bkd->bnd", torch.softmax(s, 1), h) * self.cls_w[self.head_of]).sum(-1)
        if self.hybrid:
            return 0.5 * (att + self.lme(self.inst(h, allowed, self.cls_w2), allowed)) + self.cls_b
        return att + self.cls_b


def series_box(p, b, j, px=400):
    pres = p[:, j] > 0.5
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


def study_bags(series, loc, dev, structs=STRUCTS, cor_v2=False):
    """series: list of dicts vol, plane, type (int), pos -> bag -> windows (series index, slice, mirror, cy, cx, side_px, type, plane, pos, p).
    cor_v2 (loc_version v2cor, localiser v2): coronal TF bags select slices and centre crops by tibial-cartilage presence."""
    out = {b: [] for b in BAGS}
    for si, s in enumerate(series):
        n = len(s["vol"])
        if n < 3:
            continue
        p, b = localise(loc, s["vol"], s["plane"], s["pos"], dev)
        for bag in BAGS:
            if s["plane"] not in PLANES_OF[bag]:
                continue
            j = structs.index(STRUCT_V2[bag] if cor_v2 and s["plane"] == "Coronal" and bag != "pf" else STRUCT[bag])
            c = series_box(p, b, j)
            if c is None:
                continue
            keep = [i for i in range(1, n - 1) if p[i, j] > P_MIN]
            if len(keep) > 3:
                cut = int(round(TRIM * len(keep)))
                keep = keep[cut:len(keep) - cut]
            mirror = bag == "lateral" and s["plane"] != "Sagittal"
            side_px = CROP_MM[bag, s["plane"]] / MM_PER_PX
            out[bag] += [(si, i, mirror, c[0], c[1], side_px, s["type"], PLANE_ID[s["plane"]], float(s["pos"][i]), float(p[i, j]))
                         for i in keep]
    for bag in BAGS:
        if not out[bag]:
            donor = [x for k in BAGS if k != bag for x in out[k]]
            out[bag] = sorted(donor, key=lambda w: -w[-1])[:TOP_FALLBACK]
    return out


def crop(img3, cy, cx, side, mirror, px):
    h = int(round(side))
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    out = np.zeros((3, h, h), np.uint8)
    sy, sx = max(y0, 0), max(x0, 0)
    ey, ex = min(y0 + h, img3.shape[1]), min(x0 + h, img3.shape[2])
    if ey > sy and ex > sx:
        out[:, sy - y0:ey - y0, sx - x0:ex - x0] = img3[:, sy:ey, sx:ex]
    if mirror:
        out = out[:, :, ::-1]
    return np.stack([cv2.resize(np.ascontiguousarray(c), (px, px), interpolation=cv2.INTER_LINEAR) for c in out])


def study_tensors(series, bags, m, px):
    x = np.zeros((len(BAGS), m, 3, px, px), np.uint8)
    mask = np.zeros((len(BAGS), m), bool)
    meta = np.zeros((len(BAGS), m, 3), np.float32)
    for j, bag in enumerate(BAGS):
        c = bags[bag]
        sel = np.unique(np.linspace(0, len(c) - 1, min(m, len(c))).round().astype(int))
        for k, w in enumerate(c[i] for i in sel):
            si, i, mirror, cy, cx, side, typ, plane, pos, _ = w
            x[j, k] = crop(series[si]["vol"][[i - 1, i, i + 1]], cy, cx, side, mirror, px)
            mask[j, k] = True
            meta[j, k] = (typ, plane, pos)
    bag = np.broadcast_to(np.arange(len(BAGS))[:, None], (len(BAGS), m))
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a.reshape(1, -1))).to(dt)
    return (torch.from_numpy(x.reshape(1, -1, 3, px, px)), t(mask, torch.bool), t(meta[..., 0], torch.long), t(meta[..., 1], torch.long),
            t(meta[..., 2], torch.float32), t(np.where(mask, bag, -1), torch.long))


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


def load_experts(pkg, ocfg, dev):
    experts = {}
    for name, e in ocfg["members"].items():
        if e["kind"] != "expert":
            continue
        c = e["cfg"]
        ms = []
        for f in e["files"]:
            m = OANet(c["backbone"], ocfg["n_types"], c.get("hidden", 256), c.get("split", False), c.get("bml_weight", 0) > 0,
                      c.get("lse_r", 0.0), c.get("topk", 0), c.get("hybrid", False), c.get("pl_head", False))
            sd = torch.load(f"{pkg}/{f}", map_location="cpu")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not missing and not [u for u in unexpected if not u.startswith(("rel_proj", "cls_w2"))], (f, missing, unexpected)
            ms.append(m.to(dev).eval())
        mean = torch.tensor(ms[0].backbone.pretrained_cfg["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(ms[0].backbone.pretrained_cfg["std"], device=dev).view(1, 3, 1, 1)
        # output columns read as OUTS: the 5 main heads, or (head 'T', oct16_loa_bundle) the 3 teacher-target OA heads + the meniscus heads
        n_out = len(ms[0].out_bag)
        cols = [n_out - 3 + j for j in range(3)] + [3, 4] if e.get("head") == "T" else list(range(len(OUTS)))
        assert e.get("head") != "T" or c.get("pl_head"), f"{name}: head 'T' needs a pl_head model"
        experts[name] = (ms, c["k_eval"], c["px"], mean, std, c.get("mono", False), cols)
    return experts


@torch.no_grad()
def expert_logits(experts, series, bags, dev, fuse=None, loa_extra=frozenset(), loa_set=frozenset(), extra_sets=None):
    """fuse (config oa "fuse", oct13_f5_win): also out["fused"] = crop-level fusion (fc_stage.fuse_crops) of every expert's fold
    models over the shared crops (experts must share k_eval / px / mono); outputs with an empty bag stay NaN (-> deployed e).
    loa_extra (config oa "loa_extra", oct16_loa_bundle): 'fused' stays the fusion over the deployed members (extras excluded: an empty
    set = oct13_f5_win exactly); out["fused_loa_extra"] = the fusion over deployed + extra members, read for Lateral OA only
    (ensemble.py splices it at the output level, 10-16 12:03).
    extra_sets (config oa "extra_sets", oct20_stack_all: PF Q20): {key: member names} -> out[f"fused_{key}"] = the fusion over ONLY those
    members (as loa_set), e.g. {"pf": [oa_oasep, oa_oasep_nf]}; None / {} = the stack exactly."""
    out, cache, crops = {}, {}, []
    for gi, (name, (ms, m_win, px, mean, std, mono, cols)) in enumerate(experts.items()):
        if (m_win, px) not in cache:              # experts with one crop configuration share the crops (built once)
            cache[(m_win, px)] = tuple(t.to(dev) for t in study_tensors(series, bags, m_win, px))
        x, mask, typ, plane, pos, bag = cache[(m_win, px)]
        xf = x.flatten(0, 1).float() / 255
        if mono:                                  # oa_train mono (data.gpu_prep): the centre slice in all three channels (R50)
            xf = xf[:, 1:2].expand(-1, 3, -1, -1)
        xf = ((xf - mean) / std).contiguous(memory_format=torch.channels_last)
        lg = []
        for m in ms:
            with torch.autocast("cuda", dtype=AMP, enabled=dev.type == "cuda"):
                if fuse:
                    assert not mono and len(cache) == 1, "OA crop fusion needs one shared crop configuration"
                    sc, uc = m(xf, mask, typ, plane, pos, bag, crops=True)
                    sc, uc = sc.float()[0][:, cols], uc.float()[0][:, cols]
                    crops.append((gi, sc, uc, name))
                    lg.append((torch.softmax(sc, 0) * uc).sum(0))
                else:
                    lg.append(m(xf, mask, typ, plane, pos, bag).float()[0, cols])
        v = torch.stack(lg).mean(0).cpu().numpy()
        if not np.isfinite(v).all():
            raise FloatingPointError(f"non-finite expert {name} logits")
        out[name] = v
    if fuse:
        dep = [c for c in crops if c[3] not in loa_extra]
        z = fc_stage.fuse_crops(torch.stack([c[1] for c in dep]), torch.stack([c[2] for c in dep]), [c[0] for c in dep],
                                fuse["gate"]).cpu().numpy()
        out["fused"] = np.where(np.isfinite(z), z, np.nan)
        if loa_extra:                             # fusion over deployed + extra members (Lateral OA read only)
            # oct18_probe_loa_allup (U50): loa_set = the ONLY members of the LOA fusion (the upgraded effb0 / NFNet / silL groups replace the
            # deployed ones for LOA); empty = deployed + extras (the silL probe form)
            la = [c for c in crops if c[3] in loa_set] if loa_set else crops
            assert la, "loa_set matches no expert"
            za = fc_stage.fuse_crops(torch.stack([c[1] for c in la]), torch.stack([c[2] for c in la]), [c[0] for c in la],
                                     fuse["gate"]).cpu().numpy()
            out["fused_loa_extra"] = np.where(np.isfinite(za), za, np.nan)
        for key, names in (extra_sets or {}).items():  # oct20_stack_all: one more fusion per extra set (no extra model pass)
            xs = [c for c in crops if c[3] in set(names)]
            assert xs, f"extra set {key} matches no expert"
            zx = fc_stage.fuse_crops(torch.stack([c[1] for c in xs]), torch.stack([c[2] for c in xs]), [c[0] for c in xs],
                                     fuse["gate"]).cpu().numpy()
            out[f"fused_{key}"] = np.where(np.isfinite(zx), zx, np.nan)
    return out


class Studies(torch.utils.data.Dataset):
    def __init__(self, root, studies, fs_of):
        self.root, self.studies, self.fs_of = root, list(studies), fs_of

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        s = self.studies[i]
        try:
            series, skipped = ki.study_series(f"{self.root}/{s}", self.fs_of)
            return s, [{"vol": d["vol"], "plane": d["plane"], "type": int(d["type"]), "pos": np.asarray(d["pos"], np.float32)}
                       for d in series], ""
        except Exception as e:  # noqa: BLE001
            return s, [], f"{type(e).__name__}: {e}"


def main_member(root, studies, fs_of, pkg, member, threads=None):
    """Main-pipeline models (knee_infer) as an OA member: probabilities averaged over its fold models (knee_infer's rule) -> logits."""
    with tempfile.TemporaryDirectory() as tmp:
        for f in member["files"]:
            os.symlink(os.path.abspath(f"{pkg}/{f}"), f"{tmp}/{f}")
        json.dump(member["cfg"] | {"weights": member["files"]}, open(f"{tmp}/config.json", "w"))
        p, failed, _ = ki.predict(root, studies, fs_of, tmp, threads)
    p = p.drop(index=list(failed))
    lg = np.log(np.clip(p[OA].to_numpy(np.float64), 1e-7, 1 - 1e-7) / np.clip(1 - p[OA].to_numpy(np.float64), 1e-7, 1))
    return pd.DataFrame(lg, index=p.index, columns=OA)


def apply_recipe(b, e, rec):
    """OA logits of the oct11_oa_3pipe recipe (config "oa" -> "recipe", chosen by make_oa_recipe.py on folds 0-2 OOF).
    b: the pipelines' mean OA logits, e: the experts' mean OA logits (DataFrames with the OA columns, same index; a NaN row in e =
    no expert read the study: it keeps b). Per finding F: z = (e - mu_e) * sd_b / sd_e + mu_b (OOF statistics), o = (1 - w) * b + w * z,
    then de-coupling out_F = o_F - k_F * (c0 + c . o of the other two OA findings) for the F in rec["dec"] ([k_F, c0, c1, c2])."""
    o = b.copy()
    for f in rec["stats"]:       # OA recipe: the 3 OA columns; oct12_men_3pipe meniscus recipe: the 2 meniscus columns (no "dec")
        s = rec["stats"][f]
        z = (e[f] - s["mu_e"]) * s["sd_b"] / s["sd_e"] + s["mu_b"]
        o[f] = (1 - rec["w"]) * b[f] + rec["w"] * z.fillna(b[f])
    out = o.copy()
    for f, (k, c0, *c) in rec["dec"].items():
        out[f] = o[f] - k * (c0 + o[[g for g in OA if g != f]].to_numpy() @ np.asarray(c))
    return out


def predict(root, studies, fs_of, pkg, ocfg, workers=4, threads=None):
    """-> {member name: DataFrame of member logits (experts: studies x OUTS = 3 OA + 2 meniscus heads; main members: OA)};
    failed studies are missing (logged)."""
    out = {name: main_member(root, studies, fs_of, pkg, m, threads) for name, m in ocfg["members"].items() if m["kind"] == "main"}
    if not any(m["kind"] == "expert" for m in ocfg["members"].values()):
        return out
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    cor_v2 = ocfg.get("loc_version", "v1") == "v2cor"
    structs = STRUCTS_V2 if cor_v2 else STRUCTS
    loc = Loc(structs=structs)
    loc.load_state_dict(torch.load(f"{pkg}/{ocfg['loc']}", map_location="cpu"))
    loc = loc.to(dev).to(memory_format=torch.channels_last).eval()
    det = FlipNet(pretrained=False)
    det.load_state_dict(torch.load(f"{pkg}/{ocfg['detector']}", map_location="cpu"))
    det = det.to(dev).eval()
    experts = load_experts(pkg, ocfg, dev)
    dl = torch.utils.data.DataLoader(Studies(root, studies, fs_of), batch_size=None, num_workers=workers, collate_fn=lambda x: x,
                                     prefetch_factor=2 if workers else None)   # identity collate: keep numpy volumes
    rows = {k: {} for k in list(experts) + (["fused"] if ocfg.get("fuse") else []) + (["fused_loa_extra"] if ocfg.get("loa_extra") else [])
            + [f"fused_{k}" for k in ocfg.get("extra_sets", {})]}
    STATS.update(failed=[], flipped=0)
    for sid, series, err in dl:
        try:
            if err or not series:
                raise RuntimeError(err or "no readable series")
            STATS["flipped"] += correct_sides(series, det, dev)
            bags = study_bags(series, loc, dev, structs, cor_v2)
            for k, v in expert_logits(experts, series, bags, dev, ocfg.get("fuse"), set(ocfg.get("loa_extra", [])),
                                      set(ocfg.get("loa_set", [])), ocfg.get("extra_sets")).items():
                rows[k][sid] = v
        except Exception as e:  # noqa: BLE001
            STATS["failed"].append(sid)
            print(f"oa_stage: study {sid} failed ({type(e).__name__}: {e}); OA keeps the pipeline blend", flush=True)
    STATS["wall_s"] = time.time() - t0
    return out | {k: pd.DataFrame.from_dict(v, orient="index", columns=OUTS).astype(np.float64) for k, v in rows.items()}


def blend(z, oa, ocfg):
    """z: pipeline-mean logits (studies x FINDINGS); oa: predict() output -> z with the OA columns replaced (C3 / C4 recipe)."""
    out = z.copy()
    b = z[OA]
    have = b.index.intersection(*[d.index for d in oa.values()]) if oa else b.index[:0]
    zs = []
    for name, d in oa.items():
        st = ocfg["members"][name]
        e = d.loc[have, OA]
        zs.append(pd.DataFrame({f: (e[f] - st["mu"][f]) * ocfg["blend_sd"][f] / st["sd"][f] + ocfg["blend_mu"][f] for f in OA}))
    o = b.copy()
    o.loc[have] = (1 - ocfg["w"]) * b.loc[have] + ocfg["w"] * sum(zs) / len(zs)
    for f in OA:
        dc = ocfg["decouple"][f]
        others = [g for g in OA if g != f]
        out[f] = o[f] - dc["b"] * (dc["intercept"] + sum(c * o[g] for c, g in zip(dc["coef"], others)))
    return out
