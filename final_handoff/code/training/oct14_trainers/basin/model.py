"""GPU augmentation / resampling and the study model (shared 2D encoder on slice triplets + transformer over all tokens)."""
import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def std_norm(v, n, T):
    """Pathology-invariant intensity standardisation per series (the cache window [p0.5, p99.8] is anchored on the brightest
    voxels, so effusion darkens all other tissue): median / IQR of the foreground (pixels above the series' 30th percentile)
    -> v' = clip(0.5 + 0.15 (v - median) / IQR). v: (n * T, C, H, W) in [0, 1]."""
    vals = v.view(n, -1)[:, ::16].float()
    q = torch.quantile(vals, torch.tensor([0.475, 0.65, 0.825], device=v.device), dim=1)  # foreground 25 / 50 / 75 %
    med, iqr = q[1], (q[2] - q[0]).clamp(min=1e-3)
    return (0.5 + 0.15 * (v - med.repeat_interleave(T)[:, None, None, None]) / iqr.repeat_interleave(T)[:, None, None, None]).clamp(0, 1)


def slot_view(slots, fov, fov_sag=0.0, fov_cor=0.0, fov_ax=0.0, cor_yshift=0.0, device="cuda"):
    """Per-slot fov / vertical shift tensors (S,) from slot names; 0 = use the global fov."""
    f = [(fov_sag if sl.startswith("sag") else fov_cor if sl.startswith("cor") else fov_ax) or fov for sl in slots]
    y = [cor_yshift if sl.startswith("cor") else 0.0 for sl in slots]
    return torch.tensor(f, device=device), torch.tensor(y, device=device)


# RandAugment pool: cap per op at ra_m = 1 (classical, simple ops; strength ~ U(0, ra_m) x cap per draw unit; caps from the
# Ideas-session review: low-res within the acquisition range, Gibbs mild (ringing mimics meniscal tears), small erasing)
RA_OPS = {"rotate": math.radians(15), "zoom": 0.15, "shear": 0.1, "aniso": 0.1, "elastic": 0.02,  # geometric
          "gamma": 0.4, "contrast": 0.3, "brightness": 0.12, "slice_gamma": 0.1, "bias": 0.2,  # intensity
          "blur": 0.5, "sharpen": 0.5, "ghost": 0.15, "noise": 0.03,  # texture / artefacts (realistic amplitudes)
          "lowres": 0.35, "gibbs": 0.3, "erase": 0.1}  # subset ops: f >= 0.65 (acquisition range), cut-off >= 0.7 Nyquist, side <= 20 %
RA_FLOOR = {"lowres": 0.6, "gibbs": 0.6, "erase_max": 0.25}  # hard limits at any ra_m: evidence-erasing ops never go beyond


def view_fov(cfg, device="cuda"):
    """(fov, yshift) for gpu_views from a run config (dict): the global scalar unless per-plane fov / coronal shift are set."""
    if not any(cfg.get(k) for k in ("fov_sag", "fov_cor", "fov_ax", "cor_yshift")):
        return cfg["fov"], None
    return slot_view(cfg["slots"].split(","), cfg["fov"], cfg.get("fov_sag", 0), cfg.get("fov_cor", 0), cfg.get("fov_ax", 0),
                     cfg.get("cor_yshift", 0), device)


def gpu_views(x, out_size, train, fov=1.0, rot=10.0, scale=0.1, shift=0.06, intensity=True, noise=0.0, res_aug=0.0,
              int_s=1.0, cutout=0.0, bias=0.0, blur=0.0, norm=False, per_image=False, yshift=None, shear=0.0, aniso=0.0,
              elastic=0.0, chan_jit=0.0, ghost=0.0, gibbs=0.0, ra_n=0, ra_m=1.0, cutout_max=0.3, cutout_fill="zero"):
    """x: (B, S, T, 3, H, W) uint8 on GPU -> float (B, S, T, 3, out, out) in [0, 1].
    Random affine + intensity changes per series (shared by its slices), or per image with per_image; fov = share of the
    cached FOV kept; norm = tissue-anchored standardisation (std_norm). Training-only extras: shear / aniso (max shear factor /
    log aspect ratio of the affine), elastic (amplitude of a smooth 4x4 displacement field, in normalised coordinates),
    chan_jit (independent gamma per slice of a triplet), ghost (prob. of an MRI ghost: a shifted copy at 5-25 % amplitude along
    a random axis), gibbs (prob. of k-space truncation ringing at 40-80 % of Nyquist). ra_n > 0: RandAugment-style pool on top of the base affine /
    intensity: each draw unit (series, or triplet with per_image) gets ra_n ops chosen at random from RA_OPS, each at strength
    U(0, ra_m) x its cap."""
    B, S, T, C, H, W = x.shape
    n = B * S
    dev = x.device
    m = n * T if per_image else n  # number of independent random draws
    rep = (lambda t: t) if per_image else (lambda t: t.repeat_interleave(T, 0))
    # fov / yshift: scalar or per-slot tensor (S,); yshift = view centre moved down by that share of the image height
    slot = torch.arange(m, device=dev) // (T if per_image else 1) % S
    fv = fov[slot] if torch.is_tensor(fov) else torch.full((m,), float(fov), device=dev)
    ys = 2 * yshift[slot] if torch.is_tensor(yshift) else torch.zeros(m, device=dev)
    ra = None
    if train and ra_n:  # per draw unit: which ops, at which strength (0 = op not drawn = identity for most ops)
        sel = torch.zeros(m, len(RA_OPS), device=dev)
        sel.scatter_(1, torch.rand(m, len(RA_OPS), device=dev).argsort(1)[:, :ra_n], 1.0)
        mag = sel * torch.rand(m, len(RA_OPS), device=dev) * ra_m
        ra = {op: mag[:, i] * cap for i, (op, cap) in enumerate(RA_OPS.items())}
        sgn = lambda: torch.randint(0, 2, (m,), device=dev).float() * 2 - 1  # noqa: E731
    if train:
        a = (torch.rand(m, device=dev) * 2 - 1) * math.radians(rot)
        s = fv * (1 + (torch.rand(m, device=dev) * 2 - 1) * scale)
        t = (torch.rand(m, 2, device=dev) * 2 - 1) * shift
    else:
        a, s, t = torch.zeros(m, device=dev), fv, torch.zeros(m, 2, device=dev)
    t = torch.stack([t[:, 0], t[:, 1] + ys], 1)
    if ra is not None:
        a = a + sgn() * ra["rotate"]
        s = s * torch.exp(sgn() * ra["zoom"])
    if train and (shear or aniso or ra is not None):  # A = s R(a) diag(e^an, e^-an) [[1, sh], [0, 1]]
        an = (torch.rand(m, device=dev) * 2 - 1) * aniso
        sh = (torch.rand(m, device=dev) * 2 - 1) * shear
        if ra is not None:
            an, sh = an + sgn() * ra["aniso"], sh + sgn() * ra["shear"]
        ca, sa, ex, ey = torch.cos(a), torch.sin(a), torch.exp(an), torch.exp(-an)
        theta = torch.stack([torch.stack([s * ca * ex, s * (ca * ex * sh - sa * ey), t[:, 0]], 1),
                             torch.stack([s * sa * ex, s * (sa * ex * sh + ca * ey), t[:, 1]], 1)], 1)
    else:
        theta = torch.stack([torch.stack([s * torch.cos(a), -s * torch.sin(a), t[:, 0]], 1),
                             torch.stack([s * torch.sin(a), s * torch.cos(a), t[:, 1]], 1)], 1)  # (m, 2, 3)
    theta = rep(theta)
    grid = F.affine_grid(theta, (n * T, C, out_size, out_size), align_corners=False)
    if train and (elastic or ra is not None):  # smooth random displacement field
        amp = torch.full((m,), float(elastic), device=dev) + (ra["elastic"] if ra is not None else 0)
        disp = F.interpolate(torch.randn(m, 2, 4, 4, device=dev) * amp[:, None, None, None], size=(out_size, out_size), mode="bicubic",
                             align_corners=False)
        grid = grid + rep(disp).permute(0, 2, 3, 1)
    v = x.reshape(n * T, C, H, W).float() / 255
    v = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if norm:
        v = std_norm(v, n, T)
    if train and bias:  # smooth multiplicative bias field (coil inhomogeneity / glare): 1 + bias * N(0,1) on a 4x4 grid
        fld = 1 + bias * torch.randn(m, 1, 4, 4, device=dev)
        fld = F.interpolate(fld, size=(out_size, out_size), mode="bicubic", align_corners=False).clamp(0.3, 2.0)
        v = v * rep(fld)
    if train and intensity:
        g = rep(torch.exp((torch.rand(m, device=dev) * 2 - 1) * 0.2 * int_s))[:, None, None, None]
        c = rep(1 + (torch.rand(m, device=dev) * 2 - 1) * 0.15 * int_s)[:, None, None, None]
        b = rep((torch.rand(m, device=dev) * 2 - 1) * 0.08 * int_s)[:, None, None, None]
        v = (v.clamp(0, 1) ** g * c + b).clamp(0, 1)
    if ra is not None:  # pixel ops (strength 0 = identity), then the subset ops
        r = {k: rep(vv)[:, None, None, None] for k, vv in ra.items()}
        sg = {k: rep(sgn())[:, None, None, None] for k in ("gamma", "contrast", "brightness")}
        v = v.clamp(0, 1) ** torch.exp(sg["gamma"] * r["gamma"])
        mu = v.mean((1, 2, 3), keepdim=True)
        v = ((v - mu) * (1 + sg["contrast"] * r["contrast"]) + mu + sg["brightness"] * r["brightness"]).clamp(0, 1)
        v = v.clamp(0, 1) ** torch.exp((torch.rand(v.shape[0], C, 1, 1, device=dev) * 2 - 1) * r["slice_gamma"])
        fld = F.interpolate(torch.randn(m, 1, 4, 4, device=dev), size=(out_size, out_size), mode="bicubic", align_corners=False)
        v = (v * (1 + rep(fld) * r["bias"]).clamp(0.3, 2.0)).clamp(0, 1)
        sm = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        v = (v + (r["blur"] - r["sharpen"]) * (sm - v)).clamp(0, 1)
        k_sh = int(out_size * (0.25 + 0.25 * float(torch.rand(1))))
        v = (v + r["ghost"] * torch.roll(v, k_sh, dims=2 + int(float(torch.rand(1)) < 0.5))).clamp(0, 1)
        v = (v + torch.randn_like(v) * r["noise"]).clamp(0, 1)
        for op in ("lowres", "gibbs", "erase"):
            idx = (r[op][:, 0, 0, 0] > 0).nonzero()[:, 0]
            if len(idx) == 0:
                continue
            g = r[op][idx]
            if op == "lowres":  # per image f = 1 - strength (floored, rounded to 0.05), along rows, columns or both
                fi = ((1 - g[:, 0, 0, 0]).clamp(min=RA_FLOOR["lowres"]) / 0.05).round() * 0.05
                ax = torch.randint(0, 3, (len(idx),), device=dev)  # 0: rows only (phase-like), 1: columns only, 2: both
                for bf in fi.unique().tolist():
                    for a_ in range(3):
                        sub = idx[(fi == bf) & (ax == a_)]
                        if len(sub) == 0 or bf >= 1:
                            continue
                        hs = max(int(round(out_size * (bf if a_ in (0, 2) else 1))), 8)
                        ws = max(int(round(out_size * (bf if a_ in (1, 2) else 1))), 8)
                        lo_res = F.interpolate(v[sub], size=(hs, ws), mode="bilinear", antialias=True, align_corners=False)
                        v[sub] = F.interpolate(lo_res, size=(out_size, out_size), mode="bilinear", align_corners=False).clamp(0, 1)
            elif op == "gibbs":  # k-space truncation at (1 - strength) of Nyquist
                fq = torch.fft.fftfreq(out_size, device=dev).abs() * 2
                cut = (1 - g).clamp(min=RA_FLOOR["gibbs"])
                mask = ((fq[None, None, :, None] <= cut) & (fq[None, None, None, :] <= cut)).float()
                v[idx] = torch.fft.ifft2(torch.fft.fft2(v[idx].float()) * mask).real.clamp(0, 1).to(v.dtype)
            else:  # random erasing: one rectangle, side 10 % .. 10 % + strength, filled with noise at the image mean level
                Ni = len(idx)
                hh = (0.1 + torch.rand(Ni, device=dev) * g[:, 0, 0, 0]).clamp(max=RA_FLOOR["erase_max"]) * out_size
                ww = (0.1 + torch.rand(Ni, device=dev) * g[:, 0, 0, 0]).clamp(max=RA_FLOOR["erase_max"]) * out_size
                cy, cx = torch.rand(Ni, device=dev) * out_size, torch.rand(Ni, device=dev) * out_size
                yy = torch.arange(out_size, device=dev)[None, :, None]
                xx = torch.arange(out_size, device=dev)[None, None, :]
                box = (((yy - cy[:, None, None]).abs() < hh[:, None, None] / 2) & ((xx - cx[:, None, None]).abs() < ww[:, None, None] / 2))[:, None]
                vi = v[idx]
                v[idx] = torch.where(box, (torch.rand_like(vi) * 2 * vi.mean((1, 2, 3), keepdim=True)).clamp(0, 1), vi)
    if train and chan_jit:  # slice-to-slice intensity variation inside a triplet
        v = v.clamp(0, 1) ** torch.exp((torch.rand(v.shape[0], C, 1, 1, device=dev) * 2 - 1) * chan_jit)
    if train and ghost:  # MRI ghosting: faint copy shifted by 1/4..1/2 of the image along one axis
        sel = rep((torch.rand(m, device=dev) < ghost).float() * (0.05 + 0.2 * torch.rand(m, device=dev)))[:, None, None, None]
        k = int(out_size * (0.25 + 0.25 * float(torch.rand(1))))
        v = (v + sel * torch.roll(v, k, dims=2 if float(torch.rand(1)) < 0.5 else 3)).clamp(0, 1)
    if train and gibbs:  # k-space truncation (ringing), cut-off 40-80 % of Nyquist per image
        sel = rep(torch.rand(m, device=dev) < gibbs)
        if bool(sel.any()):
            f = torch.fft.fftfreq(out_size, device=dev).abs() * 2  # 0..1 of Nyquist
            cut = (0.4 + 0.4 * torch.rand(int(sel.sum()), device=dev))[:, None, None, None]
            mask = ((f[None, None, :, None] <= cut) & (f[None, None, None, :] <= cut)).float()
            v[sel] = torch.fft.ifft2(torch.fft.fft2(v[sel].float()) * mask).real.clamp(0, 1).to(v.dtype)
    if train and blur:  # blur (k > 0) or unsharp-mask sharpening (k < 0), |k| ~ U(0, blur)
        k = rep((torch.rand(m, device=dev) * 2 - 1) * blur)
        sm = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        v = (v + k[:, None, None, None] * (sm - v)).clamp(0, 1)
    if train and res_aug:  # with prob res_aug: down-sample by f in [0.5, 1] and back (scanner resolution)
        sel = rep(torch.rand(m, device=dev) < res_aug)
        if bool(sel.any()):
            f = float(torch.empty(1).uniform_(0.5, 1.0))
            lo_res = F.interpolate(v[sel], scale_factor=f, mode="bilinear", antialias=True, align_corners=False)
            v[sel] = F.interpolate(lo_res, size=(out_size, out_size), mode="bilinear", align_corners=False)
    if train and noise:  # Gaussian noise, sigma ~ U(0, noise)
        sig = rep(torch.rand(m, device=dev) * noise)[:, None, None, None]
        v = (v + torch.randn_like(v) * sig).clamp(0, 1)
    if train and cutout:  # spatial dropout: per image with prob cutout, one rectangle of 10-30% side length set to 0
        N = v.shape[0]
        sel = torch.rand(N, device=dev) < cutout
        h = (torch.rand(N, device=dev) * (cutout_max - 0.1) + 0.1) * out_size
        w = (torch.rand(N, device=dev) * (cutout_max - 0.1) + 0.1) * out_size
        cy, cx = torch.rand(N, device=dev) * out_size, torch.rand(N, device=dev) * out_size
        yy = torch.arange(out_size, device=dev)[None, :, None]
        xx = torch.arange(out_size, device=dev)[None, None, :]
        box = ((yy - cy[:, None, None]).abs() < h[:, None, None] / 2) & ((xx - cx[:, None, None]).abs() < w[:, None, None] / 2)
        hit = (box & sel[:, None, None])[:, None]
        if cutout_fill == "noise":  # noise at the image's mean level instead of zeros
            v = torch.where(hit, (torch.rand_like(v) * 2 * v.mean((1, 2, 3), keepdim=True)).clamp(0, 1), v)
        else:
            v = v * ~hit
    return v.view(B, S, T, C, out_size, out_size)


WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


def create_encoder(backbone, pretrained, drop_path, pool="avg", img_size=None):
    """timm encoder (num_classes=0); 'radimagenet_resnet50' = timm resnet50 with RadImageNet weights (ImageNet mean/std per its README)."""
    if backbone == "radimagenet_resnet50":
        enc = timm.create_model("resnet50", pretrained=False, num_classes=0, drop_path_rate=drop_path)
        if pretrained:
            sd = torch.load(WEIGHTS / "radimagenet/ResNet50.pt", map_location="cpu")
            names = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
            sd = {names[k.split(".")[1]] + k[len("backbone.") + len(k.split(".")[1]):]: v for k, v in sd.items()}
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            assert not unexpected and not [m for m in missing if not m.startswith("fc")], (missing, unexpected)
        enc.pretrained_cfg = dict(enc.pretrained_cfg, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return enc
    kw = {"dynamic_img_size": True} if "vit" in backbone and "maxvit" not in backbone else {"global_pool": pool}
    if ("coatnet" in backbone or "maxvit" in backbone or "maxxvit" in backbone) and img_size:
        kw["img_size"] = img_size  # window / grid partitions are tied to the input size
    return timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path, **kw)


SWAP = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]  # FINDINGS order with medial <-> lateral swapped (MM/LM, MOA/LOA); MCL = 1


def mirror(x, sag, flip):
    """Left-right mirror of the canonical knee for studies where flip (B,) is True: sagittal series reverse their slice
    order (also inside each triplet), coronal / axial images flip horizontally. x: (B, S, T, 3, H, W); sag: (S,) bool."""
    xs = x.clone()
    f = flip[:, None] & sag[None]  # (B, S) sagittal series to reverse
    xs[f] = x[f].flip(1).flip(2)
    f = flip[:, None] & ~sag[None]
    xs[f] = x[f].flip(-1)
    return xs


def init_output_bias(model, prev):
    """Output bias = logit(prevalence) per finding (prior carried by the bias, not by the encoder features)."""
    b = torch.logit(torch.as_tensor(prev, dtype=torch.float32).clamp(1e-3, 1 - 1e-3))
    with torch.no_grad():
        if getattr(model, "bin_out", None) is not None:  # binned / battn readout: its final Linear carries the output
            target = model.bin_out[-1].bias
        elif hasattr(model, "b") and model.b.shape == b.shape:
            target = model.b
        else:
            target = model.head.bias
        target.copy_(b.to(target.device))


class FeatureMap(nn.Module):
    """Encoder -> spatial feature map (B, C, h, w) for CNNs and ViTs (patch tokens reshaped); compile-friendly wrapper."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.npre = getattr(enc, "num_prefix_tokens", 0)

    def forward(self, x):
        f = self.enc.forward_features(x)
        if f.ndim == 3:  # ViT tokens
            p = f[:, self.npre:]
            g = int(p.shape[1] ** 0.5)
            f = p.transpose(1, 2).reshape(len(p), p.shape[2], g, g)
        return f


class StudyModel(nn.Module):
    def __init__(self, backbone, n_slots, T, n_out=12, d=384, layers=2, heads=8, drop=0.1, pretrained=True, drop_path=0.1,
                 head="cls", grad_ckpt=False, n_aux=0, seq="tf", pool="avg", slot_aux=False, grid=1, comp=None, crop_frac=0.6,
                 sag_size=0, img_size=None, sag=None, bin_all=False, rows=None, bins_sym=False, bin_drop=0.3, bin_max=False, men_reg=False,
                 bn_single=False,
                 posenc="index", wemb=False, skip_missing=False):
        super().__init__()
        self.hybrid = head == "hattn"  # triage: battn readout + global per-finding attention path over all tokens (summed)
        head = "battn" if self.hybrid else head
        self.head_type = head
        self.enc = create_encoder(backbone, pretrained, drop_path, pool, img_size)
        self.bn_single = bn_single  # comp mode: one encoder call for whole images + crops (BatchNorm encoders)
        self.grid = grid  # > 1: G x G spatial tokens per image (keeps where a lesion is, e.g. medial vs lateral in coronal)
        self.fmap = FeatureMap(self.enc) if grid > 1 else None
        self.cell_emb = nn.Parameter(torch.randn(grid * grid, d) * 0.02) if grid > 1 else None
        self.skip_missing = skip_missing  # comp mode: encode only existing series (padded to a multiple of 4 series)
        self.w_emb = nn.Embedding(5, d) if wemb else None  # series weighting (T1 / PD / T2 / GRE / UNK) added to its tokens
        if wemb:
            nn.init.normal_(self.w_emb.weight, std=0.02)
        # posenc "fourier": each triplet's relative stack position p in [0, 1] -> 16 Fourier features -> Linear (any T)
        self.posenc = posenc
        if posenc == "fourier":
            self.register_buffer("freq", (2.0 ** torch.arange(8)) * math.pi, persistent=False)
            self.pos_proj = nn.Linear(16, d)
        # compartment crops: slots flagged in comp (coronal / axial) are split into a medial and a lateral column crop
        self.register_buffer("comp", torch.as_tensor(comp, dtype=torch.bool) if comp is not None else None, persistent=False)
        self.crop_frac = crop_frac
        self.sag_size = sag_size  # comp mode: resize whole (sagittal) images to this size; crops keep the input resolution
        self.side_emb = nn.Parameter(torch.randn(4, d) * 0.02) if comp is not None else None  # 0-1 medial / lateral, 2-3 anterior / posterior
        # comp slots flagged in rows (axial) are cropped into anterior [0, 0.6H] / posterior [0.4H, H] rows instead of columns
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.bool) if rows is not None else
                             (torch.zeros(len(comp), dtype=torch.bool) if comp is not None else None), persistent=False)
        assert comp is None or (grid == 1 and seq == "tf" and not slot_aux), "compartment crops: grid 1, transformer, no slot aux"
        if head in ("binned", "battn"):  # readout: sagittal slices in 3 position bins; cor / ax as medial / lateral crop means (comp) or means
            self.bin_all = bin_all  # cor / ax slots also in 3 slice-position bins (anterior-posterior / inferior-superior)
            self.bins_sym = bins_sym  # symmetric bins (s, T - 2s, s), s = (T + 1) // 3, e.g. (3, 2, 3) for T = 8
            self.bin_max = bin_max  # per bin: concat(mean, max) over its triplets (focal lesions seen on one triplet)
            dpool = d * (2 if bin_max else 1)
            nb = 3 if bin_all else 1
            if comp is not None:
                n_sag, n_comp = int((~torch.as_tensor(comp)).sum()), int(torch.as_tensor(comp).sum())
                nin = (n_sag * 3 + n_comp * 2 * nb) * dpool + n_slots
            else:
                assert sag is not None and grid == 1
                self.register_buffer("sag", torch.as_tensor(sag, dtype=torch.bool), persistent=False)
                n_sag = int(self.sag.sum())
                nin = (n_sag * 3 + (n_slots - n_sag) * nb) * dpool + n_slots
            self.bin_out = nn.Sequential(nn.LayerNorm(nin), nn.Dropout(bin_drop), nn.Linear(nin, n_out))
            self.register_buffer("bin_mask", None, persistent=False)
            if men_reg:  # MM / LM logits do not read the opposite side (sagittal third 0 / crop 0 = medial, third 2 / crop 1 = lateral);
                # men_reg 1 also blocks the central sagittal third, 2 keeps it (meniscal roots lie next to the tibial spines)
                assert comp is not None and not bin_all
                side = torch.zeros(nin, dtype=torch.long)  # 0 neutral (row crops, missing flags), 1 medial, 2 lateral, 3 central sag
                for k in range(n_sag):
                    for b, sd in enumerate((1, 3, 2)):
                        side[(k * 3 + b) * dpool:(k * 3 + b + 1) * dpool] = sd
                off = n_sag * 3 * dpool
                for k in range(int((torch.as_tensor(comp) & ~self.rows.cpu()).sum())):
                    side[off + 2 * k * dpool:off + (2 * k + 1) * dpool] = 1
                    side[off + (2 * k + 1) * dpool:off + (2 * k + 2) * dpool] = 2
                m = torch.ones(n_out, nin)
                m[2, (side == 2) | ((side == 3) & (men_reg == 1))] = 0
                m[3, (side == 1) | ((side == 3) & (men_reg == 1))] = 0
                self.bin_mask = m
            if head == "battn":  # + per-finding, content-dependent attention pooling inside every region (comp mode)
                assert comp is not None
                n_reg = n_sag * 3 + n_comp * 2 * nb
                self.reg_score = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, n_out))
                self.reg_w = nn.Parameter(torch.zeros(n_out, n_reg, d))
                self.register_buffer("reg_mask", None, persistent=False)
                if men_reg:  # same side rule for the attention readout: regions = sag bins (bin-major: 0 medial, 1 central,
                    # 2 lateral) x sag slots, then column crops slot-major (0 medial, 1 lateral), then row crops (neutral)
                    n_cc = int((torch.as_tensor(comp) & ~self.rows.cpu()).sum())
                    side = torch.zeros(n_reg, dtype=torch.long)
                    for b, sd in enumerate((1, 3, 2)):
                        side[b * n_sag:(b + 1) * n_sag] = sd
                    for k in range(n_cc):
                        side[3 * n_sag + 2 * k], side[3 * n_sag + 2 * k + 1] = 1, 2
                    rm = torch.ones(n_out, n_reg)
                    rm[2, (side == 2) | ((side == 3) & (men_reg == 1))] = 0
                    rm[3, (side == 1) | ((side == 3) & (men_reg == 1))] = 0
                    self.reg_mask = rm[..., None]
        # pretrained final norm that timm applies after pooling (ConvNeXt head.norm, LayerNorm2d); ViT features are already normed
        hn = getattr(getattr(self.enc, "head", None), "norm", None)
        self.cell_norm = hn if grid > 1 and isinstance(hn, nn.Module) and not isinstance(hn, nn.Identity) else None
        cfg = self.enc.pretrained_cfg
        if grad_ckpt:
            self.enc.set_grad_checkpointing(True)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Sequential(nn.Linear(getattr(self.enc, "head_hidden_size", None) or self.enc.num_features, d), nn.LayerNorm(d))  # pooled-output width (MobileNetV4 / HGNetV2 add a head conv)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        self.pos_emb = nn.Parameter(torch.zeros(T, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.slot_emb, std=0.02), nn.init.trunc_normal_(self.pos_emb, std=0.02), nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if layers and seq == "tf" else None
        self.gru = nn.GRU(d, d // 2, num_layers=layers, batch_first=True, bidirectional=True, dropout=drop) if seq == "gru" else None
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        self.drop = nn.Dropout(drop)
        self.aux = nn.Linear(d, n_out * n_aux) if n_aux else None
        self.n_aux = n_aux
        # deep supervision: per-slot linear heads on the mean of that slot's encoder tokens (before the transformer)
        self.slot_w = nn.Parameter(torch.zeros(n_slots, d, n_out)) if slot_aux else None
        self.slot_b = nn.Parameter(torch.zeros(n_slots, n_out)) if slot_aux else None
        if head == "attn":  # one attention query per finding over all tokens
            self.q = nn.Parameter(torch.randn(n_out, d) * 0.02)
            self.k = nn.Linear(d, d)
            self.w = nn.Parameter(torch.zeros(n_out, d))
            self.b = nn.Parameter(torch.zeros(n_out))
        if head == "gattn" or self.hybrid:  # triage: the main pipeline's aggregation (Ilse attention MIL per finding: LN -> Linear -> tanh -> dropout ->
            # Linear(n_out) scores per token, softmax over all valid tokens, linear readout per finding), as in BagModel
            self.att = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n_out))
            self.w = nn.Parameter(torch.zeros(n_out, d))
            self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, v, mask, pos=None, tmask=None, widx=None, tidx=None):
        """v: (B, S, T, 3, H, W) float [0, 1]; mask: (B, S) bool; pos / tmask (B, S, T): triplet positions and validity
        (position-aware sampling, comp mode only); tidx (B, S, K): grid indices of a training subset of K of the T grid
        triplets (comp mode, grid sampling): position embedding and readout bins follow each kept triplet's grid index."""
        B, S, T = v.shape[:3]
        if self.comp is not None:
            return self.forward_comp(v, mask, pos, tmask, widx, tidx)
        assert pos is None, "position-aware sampling needs comp mode"
        xv = v.flatten(0, 2)  # all slots, fixed shape (B*S*T, 3, H, W): avoids cuDNN re-tuning per batch
        xv = ((xv - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        if self.grid > 1:
            fm = F.adaptive_avg_pool2d(self.fmap(xv).float(), self.grid)  # (N, C, G, G)
            if self.cell_norm is not None:
                fm = self.cell_norm(fm)
            G2 = self.grid * self.grid
            cells = self.proj(fm.flatten(2).transpose(1, 2)) + self.cell_emb  # (N, G2, d)
            f = cells.mean(1).view(B, S, T, -1)  # per-image summary (slot deep supervision)
            tok = cells.view(B, S, T, G2, -1) + self.slot_emb[None, :, None, None] + self.pos_emb[None, None, :, None]
            tok = tok.flatten(2, 3)  # (B, S, T*G2, d)
            T = T * G2
        else:
            f = self.proj(self.enc(xv)).view(B, S, T, -1)
            tok = f + self.slot_emb[None, :, None] + self.pos_emb[None, None]
        if self.gru is not None:  # bidirectional GRU along the slices of each series
            tok = tok + self.gru(tok.flatten(0, 1))[0].view(B, S, T, -1)
        seq = torch.cat([self.cls.expand(B, -1, -1), tok.flatten(1, 2)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=v.device), (~mask)[:, :, None].expand(B, S, T).flatten(1)], 1)
        h = self.drop(seq)
        h = self.norm(self.tf(h, src_key_padding_mask=pad) if self.tf is not None else h)
        if self.head_type == "binned":
            ht = h[:, 1:].view(B, S, T, -1)
            ms = mask.float()[..., None]
            bins = self.bins(T, v.device)
            sagb = torch.stack([self.tpool(ht[:, self.sag][:, :, b], 2) for b in bins], 2) * ms[:, self.sag, None]  # (B, Ss, 3, d')
            if self.bin_all:
                oth = torch.stack([self.tpool(ht[:, ~self.sag][:, :, b], 2) for b in bins], 2) * ms[:, ~self.sag, None]
            else:
                oth = self.tpool(ht[:, ~self.sag], 2) * ms[:, ~self.sag]  # (B, So, d')
            out = self.bin_readout(torch.cat([sagb.flatten(1), oth.flatten(1), (~mask).float()], 1))
        else:
            out = self.pool(h, pad)
        if not self.training or (self.aux is None and self.slot_w is None):
            return out
        extra = {}
        if self.aux is not None:
            keep = (~pad[:, 1:]).float()[..., None]
            m = (h[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1)
            extra["aux"] = self.aux(m).view(B, -1, self.n_aux)
        if self.slot_w is not None:
            extra["slot"] = torch.einsum("bsd,sdf->bsf", f.mean(2), self.slot_w) + self.slot_b  # (B, S, n_out)
        return out, extra

    def bin_readout(self, z):
        if self.bin_mask is None:
            return self.bin_out(z)
        ln, dr, lin = self.bin_out
        return F.linear(dr(ln(z)), lin.weight * self.bin_mask, lin.bias)

    def bins(self, T, device):
        if self.bins_sym:
            s = (T + 1) // 3
            return [torch.arange(0, s, device=device), torch.arange(s, T - s, device=device), torch.arange(T - s, T, device=device)]
        return torch.tensor_split(torch.arange(T, device=device), 3)

    def tpool(self, x, dim):
        """Pool over triplets: mean, or concat(mean, max) with bin_max."""
        return torch.cat([x.mean(dim), x.amax(dim)], -1) if self.bin_max else x.mean(dim)

    def encode_valid(self, x, valid, d):
        """x: (B, S', *img) per series, valid (B, S') -> features (B, S', n_img, d) with zeros for missing series; only valid
        series go through the encoder (bucketed to a multiple of 4 series to keep the number of compiled shapes small)."""
        B, Sp = valid.shape
        per = x.shape[2:-3].numel()  # images per series (T, or T * 2 crops)
        out = x.new_zeros(B * Sp, per, d, dtype=torch.float32)
        idx = valid.flatten().nonzero().squeeze(1)
        if len(idx):
            nb = -(-len(idx) // 4) * 4
            xs = x.flatten(0, 1)[idx]
            if nb > len(idx):
                xs = torch.cat([xs, xs.new_zeros(nb - len(idx), *xs.shape[1:])])
            f = self.encode(xs.flatten(0, -4)).view(nb, per, d)[: len(idx)]
            out[idx] = f.float()
        return out.view(B, Sp, per, d)

    def encode(self, x):
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        return self.proj(self.enc(x))

    def att_pool(self, h, w, dim):
        """Per-finding attention pooling over dim (the triplets of a region): softmax of reg_score(h) where w; empty -> 0.
        h (B, S', T, [2,] d), w = h.shape[:-1] -> (B, S', [2,] F, d)."""
        sc = self.reg_score(h).float().masked_fill(~w[..., None], float("-inf"))  # (..., T, ..., F)
        a = torch.nan_to_num(sc.softmax(dim), nan=0.0)
        return torch.einsum("bstf,bstd->bsfd", a, h) if h.ndim == 4 else torch.einsum("bstcf,bstcd->bscfd", a, h)

    def masked_pool(self, h, w, dim):
        """Mean (and max with bin_max) of h over dim where w; empty -> 0. h (..., d), w = h.shape[:-1] bool."""
        wf = w.float()[..., None]
        mean = (h * wf).sum(dim) / wf.sum(dim).clamp(min=1)
        if not self.bin_max:
            return mean
        mx = h.masked_fill(~w[..., None], float("-inf")).amax(dim)
        return torch.cat([mean, torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)], -1)

    def forward_comp(self, v, mask, pos=None, tmask=None, widx=None, tidx=None):
        """Sagittal slots whole; coronal slots as medial / lateral column crops; slots flagged in rows (axial) as anterior /
        posterior row crops. Token order: sagittal, column-crop slots, row-crop slots. With pos / tmask (position-aware
        sampling): Fourier position embedding, token masks, and binned readout bins by position instead of index."""
        B, S, T, C, H, W = v.shape
        c, r = self.comp, self.rows
        cc, cr = c & ~r, c & r
        cw = int(round(W * self.crop_frac / 32)) * 32
        d = self.slot_emb.shape[1]
        if pos is None and tidx is not None:  # grid subset: embeddings / bins of the kept grid positions
            assert self.head_type in ("binned", "battn"), "triplet subsets need the binned / battn readout"
            pe = self.pos_emb[tidx]
            tm = mask[:, :, None].expand(B, S, T)
        elif pos is None:
            pe = self.pos_emb[None, None].expand(B, S, T, d)
            tm = mask[:, :, None].expand(B, S, T)
            if tmask is not None:  # grid mode with masked (dropped) triplets: excluded from bin means and region attention
                tm = tm & tmask
        else:
            ang = pos[..., None] * self.freq
            pe = self.pos_proj(torch.cat([ang.sin(), ang.cos()], -1))  # (B, S, T, d)
            tm = mask[:, :, None] & tmask
        if self.w_emb is not None:
            pe = pe + self.w_emb(widx)[:, :, None]
        xs = v[:, ~c].flatten(0, 2)
        if self.sag_size and self.sag_size != H:
            xs = F.interpolate(xs, size=(self.sag_size, self.sag_size), mode="bilinear", antialias=True, align_corners=False)
        f_cc = None
        if self.bn_single:  # BatchNorm encoders: ONE encoder call per step for whole images and column crops, the crops
            # zero-padded back to full width at their own position (medial = image with the lateral part blanked, and v.v.)
            assert not self.skip_missing and not bool(cr.any()) and not (self.sag_size and self.sag_size != H)
            assert getattr(self.enc, "head_hidden_size", None) in (None, self.enc.num_features), "bn_single pools forward_features: no head-conv encoders"
            vc = v[:, cc]
            col = torch.arange(W, device=v.device)
            med, lat = vc * (col < cw).to(vc.dtype), vc * (col >= W - cw).to(vc.dtype)
            allx = torch.cat([xs, torch.stack([med, lat], 3).flatten(0, 3)])
            fm = self.enc.forward_features(((allx - self.mean) / self.std).contiguous(memory_format=torch.channels_last))  # (N, C, h, w)
            n1, w_ = xs.shape[0], fm.shape[-1]
            fcol = torch.arange(w_, device=v.device)
            cm = torch.stack([fcol < math.ceil(cw / W * w_), fcol >= math.floor((W - cw) / W * w_)]).to(fm.dtype)  # (2, w): valid columns
            sag_f = fm[:n1].float().mean((2, 3))
            crop_m = cm.repeat(fm.shape[0] - n1 >> 1, 1)[:, None, None, :]  # med / lat alternate (stack dim 3 is innermost)
            crop_f = (fm[n1:].float() * crop_m).sum((2, 3)) / (crop_m.sum((2, 3)) * fm.shape[2])  # masked mean over the crop only
            allf = self.proj(torch.cat([sag_f, crop_f]).to(self.proj[0].weight.dtype))
            ts, f_cc = allf[:n1].view(B, -1, T, d), allf[n1:].view(B, -1, T, 2, d)
        elif self.skip_missing:
            ts = self.encode_valid(xs.view(B, -1, T, *xs.shape[1:]), mask[:, ~c], d).to(pe.dtype)
        else:
            ts = self.encode(xs).view(B, -1, T, d)
        ts = ts + self.slot_emb[~c][None, :, None] + pe[:, ~c]
        parts, valid = [ts.flatten(1, 2)], [tm[:, ~c].flatten(1)]
        crop_tok = []
        for li, (sel, side, cut) in enumerate(((cc, 0, lambda x: (x[..., :cw], x[..., W - cw:])), (cr, 2, lambda x: (x[..., :cw, :], x[..., H - cw:, :])))):
            if not bool(sel.any()):
                crop_tok.append(None)
                continue
            if li == 0 and f_cc is not None:
                f = f_cc
            else:
                a, b = cut(v[:, sel])
                if self.skip_missing:
                    f = self.encode_valid(torch.stack([a, b], 3), mask[:, sel], d).view(B, -1, T, 2, d).to(pe.dtype)
                else:
                    f = self.encode(torch.stack([a, b], 3).flatten(0, 3)).view(B, -1, T, 2, d)
            t = f + self.slot_emb[sel][None, :, None, None] + pe[:, sel][:, :, :, None] + self.side_emb[side:side + 2][None, None, None]
            crop_tok.append(t)
            parts.append(t.flatten(1, 3))
            valid.append(tm[:, sel][..., None].expand(B, -1, T, 2).flatten(1))
        tok, valid = torch.cat(parts, 1), torch.cat(valid, 1)
        seq = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=v.device), ~valid], 1)
        h = self.drop(seq)
        h = self.norm(self.tf(h, src_key_padding_mask=pad) if self.tf is not None else h)
        if self.head_type in ("binned", "battn"):
            if pos is None:
                Tf = self.pos_emb.shape[0]  # full grid size (T used in training)
                bid = torch.cat([torch.full((len(b),), k, device=v.device) for k, b in enumerate(self.bins(Tf, v.device))])
                bid = bid[tidx] if tidx is not None else bid[None, None].expand(B, S, T)
            else:
                cuts = (0.375, 0.625) if self.bins_sym else (1 / 3, 2 / 3)
                bid = (pos >= cuts[0]).long() + (pos >= cuts[1]).long()
            ns = int((~c).sum()) * T
            hs = h[:, 1:1 + ns].view(B, -1, T, d)
            ws, bs = tm[:, ~c], bid[:, ~c]
            feats = [(torch.stack([self.masked_pool(hs, ws & (bs == k), 2) for k in range(3)], 2) * mask[:, ~c].float()[..., None, None]).flatten(1)]
            regions = [self.att_pool(hs, ws & (bs == k), 2) for k in range(3)] if self.head_type == "battn" else []  # (B, Ss, F, d) each
            off = 1 + ns
            for sel, t in zip((cc, cr), crop_tok):
                if t is None:
                    continue
                n = t.shape[1] * T * 2
                hc = h[:, off:off + n].view(B, -1, T, 2, d)
                off += n
                wc, bc, mc = tm[:, sel][..., None].expand(B, -1, T, 2), bid[:, sel][..., None].expand(B, -1, T, 2), mask[:, sel].float()
                if self.bin_all:
                    feats.append((torch.stack([self.masked_pool(hc, wc & (bc == k), 2) for k in range(3)], 2) * mc[..., None, None, None]).flatten(1))
                    if self.head_type == "battn":
                        regions += [self.att_pool(hc, wc & (bc == k), 2).flatten(1, 2) for k in range(3)]
                else:
                    feats.append((self.masked_pool(hc, wc, 2) * mc[..., None, None]).flatten(1))
                    if self.head_type == "battn":
                        regions.append(self.att_pool(hc, wc, 2).flatten(1, 2))  # (B, Sc * 2, F, d)
            out = self.bin_readout(torch.cat(feats + [(~mask).float()], 1))
            if self.head_type == "battn":
                pooled = torch.cat(regions, 1)  # (B, R, F, d)
                out = out + torch.einsum("brfd,frd->bf", pooled, self.reg_w if self.reg_mask is None else self.reg_w * self.reg_mask)
            if self.hybrid:
                tok, valid = h[:, 1:], ~pad[:, 1:]
                a = self.att(tok).float().masked_fill(~valid[..., None], float("-inf")).softmax(1)  # (B, N, F)
                out = out + (torch.einsum("bnf,bnd->bfd", a, tok.float()) * self.w).sum(-1) + self.b
        else:
            out = self.pool(h, pad)
        if self.training and self.aux is not None:
            keep = valid.float()[..., None]
            m = (h[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1)
            return out, {"aux": self.aux(m).view(B, -1, self.n_aux)}
        return out

    def pool(self, h, pad):
        if self.head_type == "cls":
            return self.head(h[:, 0])
        if self.head_type == "max":  # per-token logits, max over valid tokens
            return self.head(h[:, 1:]).masked_fill(pad[:, 1:, None], float("-inf")).amax(1)
        if self.head_type == "mean":  # mean over valid tokens
            keep = (~pad[:, 1:]).float()[..., None]
            return self.head((h[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1))
        if self.head_type == "gattn":  # over the image tokens only (no CLS token)
            tok, valid = h[:, 1:], ~pad[:, 1:]
            a = self.att(tok).float().masked_fill(~valid[..., None], float("-inf")).softmax(1)  # (B, N, F)
            pooled = torch.einsum("bnf,bnd->bfd", a, tok.float())
            return (pooled * self.w).sum(-1) + self.b
        att = torch.einsum("fd,bnd->bfn", self.q, self.k(h)) / h.shape[-1] ** 0.5
        att = att.masked_fill(pad[:, None], float("-inf")).softmax(-1)
        pooled = torch.einsum("bfn,bnd->bfd", att, h)
        return (pooled * self.w).sum(-1) + self.b


def mirror_bag(x, pos, slot, sag, flip):
    """Bag version of mirror(): sagittal windows reverse their 3 slices and position; others flip horizontally.
    x: (B, K, 3, H, W); pos, slot: (B, K); sag: (S,) bool; flip: (B,) bool."""
    is_sag = sag[slot] & flip[:, None]
    other = ~sag[slot] & flip[:, None]
    x = x.clone()
    x[is_sag] = x[is_sag].flip(1)
    x[other] = x[other].flip(-1)
    pos = torch.where(is_sag, 1 - pos, pos)
    return x, pos


class BagModel(nn.Module):
    """Shared 2D encoder on 3-slice windows + slot / position embeddings + optional transformer + per-finding attention MIL."""
    def __init__(self, backbone, n_slots, n_out=12, d=384, layers=0, heads=8, drop=0.1, pretrained=True, drop_path=0.1,
                 head="gattn", n_aux=0, pool="avg", grad_ckpt=False):
        super().__init__()
        self.enc = create_encoder(backbone, pretrained, drop_path, pool)
        if grad_ckpt:
            self.enc.set_grad_checkpointing(True)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("freq", (2.0 ** torch.arange(8)) * math.pi, persistent=False)
        self.proj = nn.Sequential(nn.Linear(getattr(self.enc, "head_hidden_size", None) or self.enc.num_features, d), nn.LayerNorm(d))  # pooled-output width (MobileNetV4 / HGNetV2 add a head conv)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        self.pos_proj = nn.Linear(16, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, drop, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if layers else None
        self.norm = nn.LayerNorm(d)
        self.head_type = head
        if head == "gattn":  # Raptor-style: LN -> Linear -> tanh -> dropout -> Linear(n_out) scores per window
            self.att = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n_out))
        else:
            self.q = nn.Parameter(torch.randn(n_out, d) * 0.02)
            self.k = nn.Linear(d, d)
        self.w = nn.Parameter(torch.zeros(n_out, d))
        self.b = nn.Parameter(torch.zeros(n_out))
        self.aux = nn.Linear(d, n_out * n_aux) if n_aux else None
        self.n_aux = n_aux

    def forward(self, v, slot, pos, mask):
        """v: (B, K, 3, H, W) float [0, 1]; slot (B, K) long; pos (B, K) float in [0, 1]; mask (B, K) bool."""
        B, K = v.shape[:2]
        xv = ((v.flatten(0, 1) - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        h = self.proj(self.enc(xv)).view(B, K, -1)
        ang = pos[..., None] * self.freq
        h = h + self.slot_emb[slot] + self.pos_proj(torch.cat([ang.sin(), ang.cos()], -1))
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        if self.head_type == "gattn":
            score = self.att(h)  # (B, K, n_out)
        else:
            score = torch.einsum("fd,bkd->bkf", self.q, self.k(h)) / h.shape[-1] ** 0.5
        a = score.masked_fill(~mask[..., None], float("-inf")).softmax(1)
        out = (torch.einsum("bkf,bkd->bfd", a, h) * self.w).sum(-1) + self.b
        if self.aux is not None and self.training:
            keep = mask.float()[..., None]
            m = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
            return out, {"aux": self.aux(m).view(B, -1, self.n_aux)}
        return out
