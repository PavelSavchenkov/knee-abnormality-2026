"""CPU unit tests for src/vol3d.py (F2). Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python experiments/v3/oct16_experts/src/test_vol3d.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import vol3d  # noqa: E402
from vol3d import OANet3D, RunDataset, SliceAdapter, oa_train  # noqa: E402

torch.manual_seed(0)
BB = "efficientnet_b0.ra_in1k"


def cand(row, typ, plane, pos):
    return (row - 1, row, row + 1, False, 100.0, 100.0, 80.0, typ, plane, pos, 0.5, 0.5, 0.9)


CANDS = [cand(100 + i, 1, 0, 3.0 * i) for i in range(10)] + [cand(500 + i, 3, 1, 3.0 * i) for i in range(3)]


def dataset(train, L, shuffle=False):
    d = RunDataset.__new__(RunDataset)
    d.train, d.strat = train, False
    RunDataset.run_len, RunDataset.shuffle = L, shuffle
    return d


def test_runs():
    assert [len(s) for s in vol3d.segments(CANDS)] == [10, 3]
    d = dataset(True, 4)
    for seed in range(20):
        w = d.choose(CANDS, np.random.default_rng(seed), 8)
        assert len(w) == 8
        for r in (w[:4], w[4:]):
            rows = [x[1] for x in r]
            assert len({x[7] for x in r}) == 1                               # one series per run
            steps = np.diff(rows)
            assert set(steps) <= {0, 1} and (0 not in steps or steps[-1] == 0), rows   # consecutive; only edge replicates
    e = dataset(False, 4)
    assert e.choose(CANDS, None, 8) == e.choose(CANDS, None, 8)                # eval is deterministic
    s = dataset(True, 4, shuffle=True)
    w0 = dataset(True, 4).choose(CANDS, np.random.default_rng(3), 8)
    RunDataset.shuffle = True
    w1 = s.choose(CANDS, np.random.default_rng(3), 8)
    assert sorted(x[1] for x in w0[:4]) == sorted(x[1] for x in w1[:4]) and [x[1] for x in w0] != [x[1] for x in w1]
    RunDataset.shuffle = False
    print("runs OK")


def batch(b=2, per_bag=4, n_bags=3, px=96):
    k = per_bag * n_bags
    x = torch.randn(b * k, 3, px, px)
    mask = torch.ones(b, k, dtype=torch.bool)
    typ = torch.full((b, k), 1, dtype=torch.long)
    plane = torch.zeros(b, k, dtype=torch.long)
    pos = torch.arange(k, dtype=torch.float32).repeat(b, 1) * 3.0
    bag = torch.arange(n_bags).repeat_interleave(per_bag).repeat(b, 1)
    return x, mask, typ, plane, pos, bag


CKPT = Path(__file__).resolve().parents[1] / "refs/ckpt/oa_crop_v2cor/fold1/model_ep19.pt"   # the deployed 2D OA expert (init_full path)


def nets(mode, L=4):
    m2 = oa_train.OANet(BB, pretrained=False)
    m2.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)   # random-init effb0 features collapse (~1e-5): use real weights
    m2.eval()
    OANet3D.run_len, OANet3D.t_mode, OANet3D.t_stages = L, mode, (3, 5)
    m3 = OANet3D(BB, pretrained=False).eval()
    res = m3.load_state_dict(m2.state_dict(), strict=True)
    assert res.missing_keys and all(k.startswith("t_adapt.") for k in res.missing_keys), res
    return m2, m3


def permute_runs(bt, L, rng):
    x, mask, typ, plane, pos, bag = bt
    b, k = mask.shape
    idx = torch.cat([torch.as_tensor(rng.permutation(L)) + r * L for r in range(k // L)])
    xs = x.view(b, k, *x.shape[1:])[:, idx].reshape(x.shape)
    return xs, mask[:, idx], typ[:, idx], plane[:, idx], pos[:, idx], bag[:, idx]


def test_identity_and_order(mode):
    m2, m3 = nets(mode)
    bt = batch()
    with torch.no_grad():
        y2, y3 = m2(*bt), m3(*bt)
        d0 = float((y2 - y3).abs().max())
        assert d0 < 1e-5, d0                                                  # zero-init: the 3D net IS the 2D net
        bp = permute_runs(bt, 4, np.random.default_rng(0))
        dp0 = float((m3(*bp) - y3).abs().max())
        assert dp0 < 1e-4, dp0                                                # at init the order does not matter (attention pooling)
        for a in m3.t_adapt.values():
            a.pw.normal_(0, 0.05)
        y3b = m3(*bt)
        assert float((y3b - y2).abs().max()) > 1e-3                           # a trained adapter changes the output
        dp1 = float((m3(*bp) - y3b).abs().max())
        assert dp1 > 1e-4, dp1                                                # ... and is order-sensitive (the null is meaningful)
    print(f"{mode}: identity max |d| {d0:.1e}; permuted at init {dp0:.1e}; permuted with a non-zero adapter {dp1:.1e} OK")


def test_grad_and_channels_last():
    _, m3 = nets("trunk")
    m3 = m3.train().to(memory_format=torch.channels_last)
    y = m3(*batch())
    y.sum().backward()
    g = [p.grad for n, p in m3.named_parameters() if n.startswith("t_adapt.") and n.endswith(".pw")]
    assert g and all(x is not None and x.abs().sum() > 0 for x in g)          # zero-init pw gets gradient at step 0
    assert all(not n.startswith("t_adapt.") for n, _ in m3.named_parameters() if any(p is _ for p in m3.head_params()))
    print("grad + channels_last OK")


def test_adapter_shapes():
    a = SliceAdapter(8, 4)
    for shape in ((12, 8, 5, 5), (12, 8)):
        y = torch.randn(*shape)
        assert a(y).shape == y.shape and torch.equal(a(y), y)
    print("adapter shapes OK")


if __name__ == "__main__":
    test_runs()
    test_adapter_shapes()
    test_identity_and_order("trunk")
    test_identity_and_order("feat")
    test_grad_and_channels_last()
    print("ALL OK")


def integration_real_bags(L=8, k=16, k_eval=32):
    """Real bags (oa_train.bags_for, deployed v2cor settings) -> RunDataset items: shapes, every run = consecutive slices of one series."""
    cfg = dict(oa_train.DEFAULTS) | vol3d.EXTRA | dict(run_len=L, k=k, k_eval=k_eval, loc_version="v2cor")
    vol3d.configure(cfg)
    bags = oa_train.bags_for(cfg)
    studies = list(bags)[:6]
    for train, kk in ((True, k), (False, k_eval)):
        ds = RunDataset(studies, bags, None, None, kk, train, 224, seed=0)
        for i in range(len(studies)):
            it = ds[i]
            assert it["x"].shape == (kk * 3, 3, 224, 224) and it["mask"].all(), it["x"].shape
            wins = [w for b in oa_train.BAGS[:3] for w in ds.choose(bags[studies[i]][b], np.random.default_rng((0, 0, i)) if train else None, kk)]
            rows = np.array([w[1] for w in wins]).reshape(-1, L)
            assert ((np.diff(rows, axis=1) == 1) | (np.diff(rows, axis=1) == 0)).all(), rows
    print(f"real bags: train k {k} / eval k {k_eval}, runs of {L}: shapes + contiguity OK on {len(studies)} studies")


if __name__ == "__main__" and "--real" in sys.argv:
    integration_real_bags()
