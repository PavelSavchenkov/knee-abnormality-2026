"""Copy of experiments/v0/oct7_baseline/model.py (SERIES_TYPES inlined from common/meta.py).

Window encoder (timm backbone) + pooling head over the study's windows.

h_k = LayerNorm(backbone(window_k) + type_emb(series type) + pos_enc(slice mm))   (meta terms optional, zero-initialised)
heads:
  attn   per-finding attention (Ilse et al. 2018, as in the public Raptor model): a_{k,f} = softmax_k(w_f . tanh(W h_k)),
         logit_f = v_f . sum_k a_{k,f} h_k + b_f
  gated  same with tanh(W h) * sigmoid(U h)
  mean   masked mean of h_k, then a linear layer (reference)
  query  12 learned finding queries, multi-head cross-attention over projected window tokens (q_layers decoder layers), linear per query
n_aux > 0 (attn / gated / mean heads): also returns (b, n_out, n_aux) logits from the same pooled vector per finding (auxiliary
targets such as the 10-class v1 status)."""
import math

import timm
import torch
import torch.nn as nn

PLANES = ["Sagittal", "Coronal", "Axial"]
WEIGHTINGS = ["T1", "PD", "T2", "GRE", "UNK"]
SERIES_TYPES = [f"{p[:3].upper()}_{w}_{'FS' if fs else 'NOFS'}" for p in PLANES for w in WEIGHTINGS for fs in (0, 1)]  # common/meta.py


def sincos(pos, n_freq=8, scale=60.0):
    """pos (mm) -> (.., 2*n_freq) sinusoidal features; wavelengths from scale/2^i mm."""
    f = torch.exp2(torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * (2 * math.pi / scale)
    a = pos[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


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
                 q_layers=1, q_heads=8, n_aux=0):
        super().__init__()
        kw = {"drop_path_rate": drop_path} if drop_path > 0 else {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kw)
        d = self.backbone.num_features
        self.use_type, self.use_pos, self.head = use_type, use_pos, head
        self.type_emb = nn.Embedding(len(SERIES_TYPES), d)
        nn.init.zeros_(self.type_emb.weight)
        self.pos_proj = nn.Linear(16, d)
        nn.init.zeros_(self.pos_proj.weight)
        nn.init.zeros_(self.pos_proj.bias)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        if head in ("attn", "gated"):
            self.att_v = nn.Linear(d, hidden)
            self.att_u = nn.Linear(d, hidden) if head == "gated" else None
            self.att_w = nn.Linear(hidden, n_out)
        if head == "query":
            self.qhead = QueryHead(d, n_out, hidden, q_heads, q_layers, drop)
        self.cls_w = nn.Parameter(torch.zeros(n_out, d))
        self.cls_b = nn.Parameter(torch.zeros(n_out))
        assert n_aux == 0 or head != "query"
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

    def bias(self):
        return self.qhead.out_b if self.head == "query" else self.cls_b

    def forward(self, x, mask, typ, pos, return_att=False):
        b, k = mask.shape
        h = self.backbone(x).view(b, k, -1).float()
        if self.use_type:
            h = h + self.type_emb(typ)
        if self.use_pos:
            h = h + self.pos_proj(sincos(pos))
        h = self.norm(h)
        if self.head == "query":
            return self.qhead(self.drop(h), mask)
        if self.head == "mean":
            m = mask[..., None].float()
            z = (self.drop(h) * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
            return self.out(z.expand(-1, self.cls_w.shape[0], -1))
        v = torch.tanh(self.att_v(self.drop(h)))
        if self.att_u is not None:
            v = v * torch.sigmoid(self.att_u(self.drop(h)))
        s = self.att_w(v).masked_fill(~mask[..., None], float("-inf"))
        a = torch.softmax(s, dim=1)
        z = torch.einsum("bkn,bkd->bnd", a, self.drop(h))
        out = self.out(z)
        return (out, a) if return_att else out

    def out(self, z):
        """z (b, n_out, d) pooled per finding -> logits, or (logits, aux logits) when n_aux > 0."""
        logits = (z * self.cls_w).sum(-1) + self.cls_b
        if self.n_aux:
            return logits, torch.einsum("bnd,nsd->bns", z, self.aux_w) + self.aux_b
        return logits
