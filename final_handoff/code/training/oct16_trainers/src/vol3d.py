"""F2 (Brainstorm, 10-19; FRESH_IDEAS.md F2; OAI: no): ordered-slice 3D / (2+1)D OA crop expert as a wrapper of oa/oa_train.py.
Patch for Experts to review; nothing in oa/ is edited.

Mechanism: OA / meniscus / fracture gold criteria are through-plane facts (cartilage-loss extent >= 10 mm, the two-slice-touch rule,
fracture-line continuity). The deployed OA expert draws an UNORDERED bag of 3-slice windows per compartment. Here each bag is drawn as
R runs of L CONSECUTIVE windows of one series (k = R * L per bag), and the trunk mixes features along the slice axis:
  t_mode=trunk (default): (2+1)D inflation of the effb0 trunk. After each stage in t_stages, a depthwise slice-axis conv (kernel 3) +
                          GELU + a ZERO-INIT pointwise conv are added as a residual (forward hooks, so the 2D state dict keeps its keys);
  t_mode=feat:            the same adapter on the pooled per-window features (a 1D conv along the run) = F1's cheapest form.
At init, the output equals the 2D expert on the same windows exactly (zero-init residual; unit-tested), so init_full from the deployed
v2cor checkpoint starts at the deployed model.
NULL: shuffle_runs=True permutes the windows inside every run (train and eval). Each window keeps its own slice position embedding, so
only the adapter loses the order.

Keys (on top of oa_train.DEFAULTS):
  run_len    L (0 = off: plain oa_train behaviour); k / k_eval (or every k_bags value) must be multiples of L;
  t_mode     trunk | feat;
  t_stages   '+'-joined effb0 stage indices for t_mode=trunk (default 3+5: 80 ch at stride 16, 192 ch at stride 32);
  shuffle_runs  the order null.
Not combinable with: strat, aug_jit, aug_dil (they break run contiguity), trunk_by (splits windows across trunks), series_heads with
t_mode=trunk is fine (heads read the pooled features).

Example (f1, from the deployed v2cor checkpoint; read as an added OA member in the Q15 / Q20 forms vs the v2cor seed member):
  PYTHONPATH=. python experiments/v3/oct16_experts/src/vol3d.py --name v3d_l8 --folds 1 --set run_len=8 k=16 k_eval=32 \\
      init_full=refs/ckpt/oa_crop_v2cor/fold{fold}/model_ep19.pt epochs=8 lr=1e-4 lr_head=1e-4 warmup=0 eval_from=5 eval_every=1 save_from=5
  null: the same + shuffle_runs=True (name v3d_l8_shuf).
Unit tests: src/test_vol3d.py (CPU)."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "oa"), str(HERE.parents[3])]
import oa_data  # noqa: E402
import oa_train  # noqa: E402

EXTRA = dict(run_len=0, t_mode="trunk", t_stages="3+5", shuffle_runs=False)


def segments(cands):
    """Maximal runs of candidates that are adjacent slices of one series (centre row + 1, same series type / plane)."""
    out, cur = [], [0] if cands else []
    for i in range(1, len(cands)):
        a, b = cands[i - 1], cands[i]
        if b[1] == a[1] + 1 and b[7] == a[7] and b[8] == a[8]:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    return out + ([cur] if cur else [])


def run_at(seg, start, L):
    """L consecutive indices of a segment from start; a segment shorter than L repeats its last window (edge replicate)."""
    idx = seg[start:start + L]
    return idx + [idx[-1]] * (L - len(idx))


class RunDataset(oa_data.CompartmentDataset):
    run_len, shuffle = 0, False

    def choose(self, cands, rng, k):
        L = self.run_len
        if not L or not cands:
            return super().choose(cands, rng, k)
        assert k % L == 0, (k, L)
        segs = segments(cands)
        runs = []
        if self.train:
            p = np.array([len(s) for s in segs], float)
            for j in rng.choice(len(segs), k // L, p=p / p.sum()):
                s = segs[j]
                runs.append(run_at(s, int(rng.integers(0, max(len(s) - L, 0) + 1)), L))
        else:                                            # eval: run centres evenly spaced over the candidates (as the 2D linspace rule)
            pos = {i: (j, n) for j, s in enumerate(segs) for n, i in enumerate(s)}
            for c in np.linspace(0, len(cands) - 1, k // L).round().astype(int):
                j, n = pos[c]
                s = segs[j]
                runs.append(run_at(s, min(max(n - L // 2, 0), max(len(s) - L, 0)), L))
        if self.shuffle:
            r = rng if self.train else np.random.default_rng(len(cands))
            runs = [list(r.permutation(x)) for x in runs]
        return [cands[i] for x in runs for i in x]


class SliceAdapter(nn.Module):
    """Residual along the slice axis: depthwise slice conv (kernel 3, zero padding at run ends) + GELU + zero-init channel mixing.
    Input (N, C, H, W) or (N, C) with N = n_runs * L consecutive windows; output has the input's shape. Plain 1D / 2D parameters
    (no Conv3d): run_fold converts the model to channels_last, which only applies to 4D weights."""

    def __init__(self, c, L):
        super().__init__()
        self.L = L
        self.dw = nn.Parameter(torch.empty(c, 3).uniform_(-3 ** -0.5, 3 ** -0.5))
        self.dw_b = nn.Parameter(torch.zeros(c))
        self.pw = nn.Parameter(torch.zeros(c, c))
        self.pw_b = nn.Parameter(torch.zeros(c))

    def forward(self, y):
        flat = y.dim() == 2
        z = (y[:, :, None, None] if flat else y).float()
        n, c, h, w = z.shape
        assert n % self.L == 0, (n, self.L)
        v = z.reshape(n // self.L, self.L, c, h, w)
        prev = nn.functional.pad(v[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))      # slice l-1 (zero before the run)
        nxt = nn.functional.pad(v[:, 1:], (0, 0, 0, 0, 0, 0, 0, 1))        # slice l+1 (zero after the run)
        k = self.dw[None, None, :, :, None, None]
        u = nn.functional.gelu(k[..., 0, :, :] * prev + k[..., 1, :, :] * v + k[..., 2, :, :] * nxt + self.dw_b[:, None, None])
        d = torch.einsum("brchw,oc->brohw", u, self.pw) + self.pw_b[:, None, None]
        d = d.reshape(n, c, h, w).to(y.dtype)
        return y + (d[:, :, 0, 0] if flat else d)


class OANet3D(oa_train.OANet):
    run_len, t_mode, t_stages = 0, "trunk", (3, 5)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        L = self.run_len
        self.t_adapt = nn.ModuleDict()
        if not L:
            return
        assert not self.trunk_by, "run_len with trunk_by: windows are split across trunks"
        if self.t_mode == "trunk":
            blocks = self.backbone.blocks                # timm EfficientNet stages; a stage's last conv is its output projection
            for i in self.t_stages:
                c = [m for m in blocks[i].modules() if isinstance(m, nn.Conv2d)][-1].out_channels
                self.t_adapt[str(i)] = SliceAdapter(c, L)
            for i in self.t_stages:
                blocks[i].register_forward_hook(lambda m, inp, out, a=self.t_adapt[str(i)]: a(out))
        else:
            assert self.t_mode == "feat", self.t_mode
            self.t_adapt["feat"] = SliceAdapter(self.type_emb.embedding_dim, L)

    def encode(self, x, typ):
        f = super().encode(x, typ)
        return self.t_adapt["feat"](f) if "feat" in self.t_adapt else f

    def head_params(self):                               # the adapters train at the trunk LR (they live inside / next to the trunk)
        return [p for n, p in self.named_parameters() if not n.startswith(("backbone.", "backbone2.", "t_adapt."))]

    def load_state_dict(self, sd, strict=True):
        """A 2D OANet checkpoint (init_full) loads into the 3D net: only the zero-init adapters may be missing."""
        res = super().load_state_dict(sd, strict=False)
        assert not res.unexpected_keys and all(k.startswith("t_adapt.") for k in res.missing_keys), res
        return res


def configure(cfg):
    L = int(cfg["run_len"])
    if L:
        assert not (cfg["strat"] or cfg["aug_jit"] or cfg["aug_dil"] or cfg["trunk_by"]), "run_len: strat / aug_jit / aug_dil / trunk_by off"
        ks = [int(v.split(":")[1]) for v in cfg["k_bags"].split("+")] if cfg["k_bags"] else [cfg["k"]]
        assert all(k % L == 0 for k in ks + [cfg["k_eval"]]), f"k / k_bags / k_eval must be multiples of run_len {L}"
    RunDataset.run_len, RunDataset.shuffle = L, bool(cfg["shuffle_runs"])
    OANet3D.run_len, OANet3D.t_mode = L, cfg["t_mode"]
    OANet3D.t_stages = tuple(int(s) for s in str(cfg["t_stages"]).split("+"))
    oa_train.CompartmentDataset, oa_train.OANet = RunDataset, OANet3D


DEFAULTS = dict(oa_train.DEFAULTS) | EXTRA                # scripts/launch.py --kind vol (box runner): same entry points as oa/oa_train.py
bags_for = oa_train.bags_for


def run_fold(cfg, fold, out, bags):
    configure(cfg)
    return oa_train.run_fold(cfg, fold, out, bags)


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="1")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(oa_train.DEFAULTS) | EXTRA
    for kv in a.set:
        k, v = kv.split("=", 1)
        assert k in cfg, f"unknown key {k}"
        cfg[k] = oa_train.parse_value(v)
    assert cfg["side_fix"], "the localiser boxes are in the side-corrected view"
    configure(cfg)
    bags = oa_train.bags_for(cfg)
    for fold in [int(f) for f in a.folds.split(",")]:
        out = oa_train.RUNS / a.name / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out / "config.json", "w"), indent=1)
        oa_train.run_fold(cfg, fold, out, bags)


if __name__ == "__main__":
    main()
