"""Window encoder (timm backbone) + pooling head over the study's windows.

h_k = LayerNorm(backbone(window_k) + type_emb(series type) + pos_enc(slice mm))   (meta terms optional, zero-initialised)
heads:
  attn   per-finding attention (Ilse et al. 2018, as in the public Raptor model): a_{k,f} = softmax_k(w_f . tanh(W h_k)),
         logit_f = v_f . sum_k a_{k,f} h_k + b_f
  gated  same with tanh(W h) * sigmoid(U h)
  attn1  one attention over windows shared by all findings (a_k = softmax_k(w . tanh(W h_k))), then a linear layer per finding
  mean   masked mean of h_k, then a linear layer per finding (no attention)
  max    masked max of h_k, then a linear layer per finding
  meanmax  (mean + max) / 2 of h_k, then a linear layer per finding
  wlse   per-window logits w_f . h_k, pooled over windows with log-sum-exp (r = 5): (1/r) log mean_k exp(r s_kf) + b_f
  query  12 learned finding queries, multi-head cross-attention over projected window tokens (q_layers decoder layers), linear per query
  sattn  per-finding attention over the spatial tokens of all windows (CNN feature map before pooling + learned spatial embedding):
         "where in the slice" and "which window" in one softmax
context before the attn / gated / attn1 / sattn heads (optional): ctx_layers transformer encoder layers over the window tokens;
gru: a bidirectional GRU along the windows sorted by (series, position), i.e. along each series's slice order
ctx and gru start as identity (zero-initialised residual branch / gain) and can train at their own lr (train.py lr_seq)
lora_r > 0: backbone frozen except LoRA adapters (rank r) on its Linear layers named in lora_targets (ViT attention: qkv, proj)
bb_weights: checkpoint loaded into the timm backbone instead of its timm weights (OrthoFoundation-L, RadImageNet R50)
grad_ckpt: timm gradient checkpointing in the backbone (memory for compute, e.g. ViT-L)
prior "true": attention logits get + beta_f * coverage of finding f's anatomical structure in the window (data.structure_coverage,
from nnU-Net masks, so test studies would need masks); "pred": the same with the coverage head's own sigmoid output (detached).
prior_findings: comma-separated findings that get the prior ("" = all)
cov_head: per-window linear head predicting that coverage (train.py cov_weight: auxiliary BCE, an anatomy-aware feature loss)
type_prior: attention logits get + t[series type, finding] (zero-init): attention followed each sequence's window share, so
T1 windows diluted meniscus pooling in full-protocol studies (Ideas session, 03:50)
syn_tie: synovitis attention logits = effusion's + synovitis' own (synovitis attention was at chance on its structure)
bb_pool: the backbone's own spatial pooling of each window's feature map (avg = timm default, avgmax, max: favours small focal signal)
n_aux > 0 (attn / gated / mean heads): also returns (b, n_out, n_aux) logits from the same pooled vector per finding (auxiliary
targets such as the 10-class v1 status)."""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.eva import checkpoint_filter_fn

from common.meta import SERIES_TYPES
from common.paths import FINDINGS


def sincos(pos, n_freq=8, scale=60.0):
    """pos (mm) -> (.., 2*n_freq) sinusoidal features; wavelengths scale/2^i mm. The code repeats every `scale` mm: with the old
    default 60, positions 60 mm apart (e.g. +35 and -25) get the same code, and ~45% of slices lie beyond +-30 mm. Use >= 240."""
    f = torch.exp2(torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    a = pos[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


RESNET_SEQ = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def load_backbone_weights(backbone, path):
    """Checkpoint with a "backbone." prefix into the timm backbone; every model key must match. Two layouts:
    - Meta DINOv3 torch.hub (e.g. OrthoFoundation-L, arxiv 2601.18250: DINOv3 ViT-L/16 with continued DINO training on ~1.2M knee
      X-ray + MRI images) into timm vit_*_dinov3, via timm's own converter (mapping from work/ortho_foundation/load.py);
    - torchvision ResNet as nn.Sequential (e.g. RadImageNet R50, HF Lab-Rasool/RadImageNet) into timm resnet*: 0 conv1, 1 bn1,
      4-7 layer1-4 (mapping from work/knee_seg/radimagenet/load.py). Both mappings from the Ideas session."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    assert all(k.startswith("backbone.") for k in sd), "unexpected checkpoint layout"
    sd = {k.removeprefix("backbone."): v for k, v in sd.items()}
    if next(iter(sd)).split(".")[0].isdigit():
        sd = {f"{RESNET_SEQ[k.split('.', 1)[0]]}.{k.split('.', 1)[1]}": v for k, v in sd.items()}
    else:
        sd = checkpoint_filter_fn(sd, backbone)
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(("head.", "fc."))]
    assert not missing and not unexpected, f"missing {missing[:5]}, unexpected {unexpected[:5]}"


class LoRA(nn.Module):
    """base(x) + (alpha / r) * B(A(x)) with base frozen and B zero-initialised, so training starts from the pretrained layer
    (Hu et al. 2021). Written here because installing peft would pull transformers and downgrade huggingface_hub in the shared env."""
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.a = nn.Linear(base.in_features, r, bias=False)
        self.b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


def add_lora(backbone, r, targets):
    """Freeze the backbone and wrap every nn.Linear whose last name part is in targets (e.g. qkv, proj) with LoRA (alpha = 2r)."""
    backbone.requires_grad_(False)
    names = [n for n, m in backbone.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    assert names, f"no Linear named {targets}"
    for n in names:
        parent, _, leaf = n.rpartition(".")
        owner = backbone.get_submodule(parent) if parent else backbone
        setattr(owner, leaf, LoRA(getattr(owner, leaf), r, 2 * r))
    return len(names)


class QueryHead(nn.Module):
    def __init__(self, d, n_out, dim=256, heads=8, layers=1, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(d, dim)
        self.q = nn.Parameter(torch.randn(n_out, dim) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(dim, heads, dim * 2, drop, batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out_w = nn.Parameter(torch.zeros(n_out, dim))
        self.out_b = nn.Parameter(torch.zeros(n_out))

    def forward(self, h, mask):
        mem = self.proj(h)
        q = self.q.expand(h.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, mem, memory_key_padding_mask=~mask)
        return (self.norm(q) * self.out_w).sum(-1) + self.out_b


class KneeNet(nn.Module):
    def __init__(self, backbone, n_out=12, pretrained=True, hidden=256, drop=0.2, use_type=True, use_pos=True, head="attn", drop_path=0.0,
                 q_layers=1, q_heads=8, n_aux=0, pos_scale=60.0, ctx_layers=0, gru=False, img_size=None, bb_pool="avg", lora_r=0,
                 lora_targets="qkv,proj", grad_ckpt=False, bb_weights="", prior="", cov_head=False, syn_tie=False, type_prior=False, prior_findings=""):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path > 0 else {}
        if ("vit" in backbone or backbone.startswith(("coatnet", "maxxvit"))) and img_size:
            kw["img_size"] = img_size                              # ViTs resize their position embedding; CoAtNet / MaxViT their windows
        if head == "sattn":
            kw["global_pool"] = ""                                 # keep the feature map
        elif bb_pool != "avg":
            assert bb_pool in ("avgmax", "max"), bb_pool           # same width as avg (catavgmax would double it)
            kw["global_pool"] = bb_pool
        self.backbone = timm.create_model(backbone, pretrained=pretrained and not bb_weights, num_classes=0, **kw)
        if bb_weights and pretrained:                              # predict.py (pretrained=False) loads everything from model.pt
            load_backbone_weights(self.backbone, bb_weights)
        if lora_r:
            add_lora(self.backbone, lora_r, lora_targets.split(",") if isinstance(lora_targets, str) else lora_targets)  # --set gives a tuple
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        d = self.backbone.num_features
        self.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 2 * d, drop, batch_first=True, norm_first=True), ctx_layers,
                                         enable_nested_tensor=False) if ctx_layers else None
        for layer in self.ctx.layers if ctx_layers else []:       # residual branches start at 0: each layer starts as identity
            for lin in (layer.self_attn.out_proj, layer.linear2):  # (ReZero / LayerScale-0); random init diverged (ctx1_b0)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        assert not (gru and head == "sattn"), "gru runs over window tokens, sattn uses spatial tokens"
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True) if gru else None
        self.gru_gain = nn.Parameter(torch.zeros(d)) if gru else None   # h + gain * gru(h): starts as identity
        self.spatial = nn.Parameter(torch.zeros(1, 1, 256, d)) if head == "sattn" else None
        self.use_type, self.use_pos, self.head, self.pos_scale = use_type, use_pos, head, pos_scale
        self.type_emb = nn.Embedding(len(SERIES_TYPES), d)
        nn.init.zeros_(self.type_emb.weight)
        self.pos_proj = nn.Linear(16, d)
        nn.init.zeros_(self.pos_proj.weight)
        nn.init.zeros_(self.pos_proj.bias)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        if head in ("attn", "gated", "attn1", "sattn"):
            self.att_v = nn.Linear(d, hidden)
            self.att_u = nn.Linear(d, hidden) if head == "gated" else None
            self.att_w = nn.Linear(hidden, 1 if head == "attn1" else n_out)
        if head == "query":
            self.qhead = QueryHead(d, n_out, hidden, q_heads, q_layers, drop)
        self.cls_w = nn.Parameter(torch.zeros(n_out, d))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        assert n_aux == 0 or head not in ("query", "wlse")
        assert not (prior or syn_tie or type_prior) or head in ("attn", "gated", "sattn"), "per-finding attention heads only"
        self.type_prior = nn.Parameter(torch.zeros(len(SERIES_TYPES), n_out)) if type_prior else None   # s_kf += t[type_k, f]
        assert not cov_head or head in ("attn", "gated"), "coverage head reads window tokens"
        assert prior in ("", "true", "pred") and (prior != "pred" or cov_head), prior
        self.prior_beta = nn.Parameter(torch.ones(n_out)) if prior else None    # s_kf += beta_f * coverage_kf
        self.prior = prior
        keep = FINDINGS if not prior_findings else prior_findings.split(",") if isinstance(prior_findings, str) else prior_findings
        keep = [f.replace("_", " ") for f in keep]                # queue lines split on spaces: Medial_OA = Medial OA
        assert set(keep) <= set(FINDINGS), keep
        self.register_buffer("prior_mask", torch.tensor([f in keep for f in FINDINGS] + [False] * (n_out - len(FINDINGS)), dtype=torch.float32),
                             persistent=False)                  # findings that get the structure prior
        self.cov_head = nn.Linear(d, len(FINDINGS)) if cov_head else None     # per-window structure coverage logits
        self.cov_logit = None
        self.syn_tie = syn_tie
        self.n_aux = n_aux
        if n_aux:
            self.aux_w = nn.Parameter(torch.zeros(n_out, n_aux, d))
            self.aux_b = nn.Parameter(torch.zeros(n_out, n_aux))

    def norm_stats(self):
        """Input mean / std (1, 3, 1, 1) the pretrained backbone expects."""
        c = self.backbone.pretrained_cfg
        return torch.tensor(c["mean"]).view(1, 3, 1, 1), torch.tensor(c["std"]).view(1, 3, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def seq_params(self):
        """Parameters of the sequence layers over windows (ctx transformer, GRU), a subset of head_params."""
        return [p for n, p in self.named_parameters() if n.startswith(("ctx.", "gru.", "gru_gain"))]

    def bias(self):
        return self.qhead.out_b if self.head == "query" else self.cls_b

    def forward(self, x, mask, typ, pos, sid=None, return_att=False, cov=None):
        b, k = mask.shape
        f = self.backbone(x).float()
        if self.head == "sattn":                                   # (b*k, d, gh, gw) -> spatial tokens per window
            gh, gw = f.shape[-2:]
            f = f.flatten(2).transpose(1, 2).view(b, k, gh * gw, -1) + self.spatial[:, :, : gh * gw]
        else:
            f = f.view(b, k, -1)
        meta = 0
        if self.use_type:
            meta = meta + self.type_emb(typ)
        if self.use_pos:
            meta = meta + self.pos_proj(sincos(pos, scale=self.pos_scale))
        if self.head == "sattn":
            h = self.norm(f + (meta[:, :, None] if torch.is_tensor(meta) else 0))
            n_sp = h.shape[2]
            h, mask = h.flatten(1, 2), mask.repeat_interleave(n_sp, 1)
        else:
            h = self.norm(f + meta)
        if self.gru is not None:                                   # along each series: sort windows by (series, position)
            key = torch.where(mask, sid.float() * 1e4 + pos, torch.full_like(pos, 1e9))
            order = key.argsort(1)
            hs = torch.gather(h, 1, order[..., None].expand_as(h))
            lengths = mask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(hs, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=k)
            h = torch.zeros_like(h).scatter(1, order[..., None].expand_as(h), out.to(h.dtype)) * self.gru_gain + h
        if self.ctx is not None:
            h = self.ctx(h, src_key_padding_mask=~mask)
        if self.head == "query":
            return self.qhead(self.drop(h), mask)
        if self.head in ("mean", "max", "meanmax"):
            m, hd = mask[..., None], self.drop(h)
            mean = (hd * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            mx = hd.masked_fill(~m, float("-inf")).max(1, keepdim=True).values
            z = {"mean": mean, "max": mx, "meanmax": (mean + mx) / 2}[self.head]
            return self.out(z.expand(-1, self.cls_w.shape[0], -1))
        if self.head == "wlse":
            r = 5.0
            sc = torch.einsum("bkd,nd->bkn", self.drop(h), self.cls_w).masked_fill(~mask[..., None], float("-inf"))
            return (torch.logsumexp(r * sc, 1) - torch.log(mask.sum(1, keepdim=True).float())) / r + self.cls_b
        if self.cov_head is not None:
            self.cov_logit = self.cov_head(h)                      # (b, k, 12); read by train.py for its auxiliary loss
        v = torch.tanh(self.att_v(self.drop(h)))
        if self.att_u is not None:
            v = v * torch.sigmoid(self.att_u(self.drop(h)))
        s = self.att_w(v)
        if self.syn_tie:                                           # synovitis attention = effusion attention + its own residual
            s = s + F.one_hot(torch.tensor(FINDINGS.index("Synovitis")), s.shape[-1]).to(s) * s[..., FINDINGS.index("Effusion"), None]
        if self.type_prior is not None:                            # which sequences each finding should read (zero-init)
            tp = self.type_prior[typ]
            s = s + (tp.repeat_interleave(s.shape[1] // tp.shape[1], 1) if self.head == "sattn" else tp)
        if self.prior_beta is not None:
            cov = torch.sigmoid(self.cov_logit).detach() if self.prior == "pred" else cov   # "pred": no masks needed at test time
            c = F.pad(cov.to(s.dtype), (0, s.shape[-1] - cov.shape[-1]))   # extra outputs get no prior
            s = s + self.prior_beta * self.prior_mask * (c.repeat_interleave(s.shape[1] // c.shape[1], 1) if self.head == "sattn" else c)
        s = s.masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(s, dim=1)
        z = torch.einsum("bkn,bkd->bnd", a, self.drop(h))
        if self.head == "attn1":
            z = z.expand(-1, self.cls_w.shape[0], -1)
        if self.head == "sattn":
            a = a.view(b, k, -1, a.shape[-1]).sum(2)              # attention per window (summed over space) for inspection
        out = self.out(z)
        return (out, a) if return_att else out

    def out(self, z):
        """z (b, n_out, d) pooled per finding -> logits, or (logits, aux logits) when n_aux > 0."""
        logits = (z * self.cls_w).sum(-1) + self.cls_b
        if self.n_aux:
            return logits, torch.einsum("bnd,nsd->bns", z, self.aux_w) + self.aux_b
        return logits
