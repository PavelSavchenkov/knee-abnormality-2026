"""Minimal nnU-Net v2 inference for aagatti/nnunet_knee (MIT) without installing nnunetv2 (it would downgrade timm in the rsna env).
Needs dynamic_network_architectures + nibabel on PYTHONPATH (installed --no-deps into a private dir).

Model: ResidualEncoderUNet (ResEnc-M), 10 classes: 0 bg, 1 patellar cart, 2 femoral cart, 3 medial tibial cart, 4 lateral tibial cart,
5 medial meniscus, 6 lateral meniscus, 7 femur, 8 tibia, 9 patella. Trained on 342 sagittal 3D volumes (512 x 512 x 160 at
0.3125 x 0.3125 x 0.8 mm; SKM-TEA-like qDESS). nnU-Net array order = (slice L-R, rows superior->inferior, cols anterior->posterior),
which is our canonical sagittal layout (rows S->I, cols A->P).

Preprocessing as nnU-Net: resample to the configuration spacing (linear here), z-score per image; sliding window with Gaussian
weighting, overlap 0.5; softmax average."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

HERE = Path(__file__).resolve().parent
CLASSES = ["bg", "pat_cart", "fem_cart", "med_tib_cart", "lat_tib_cart", "med_men", "lat_men", "femur", "tibia", "patella"]


def load(config="3d_lowres"):
    d = HERE / "models/Dataset500_KneeMRI" / f"nnUNetTrainer__nnUNetResEncUNetMPlans__{config}"
    plans = json.load(open(d / "plans.json"))
    c = plans["configurations"][config]
    kw = dict(c["architecture"]["arch_kwargs"])
    for k in c["architecture"]["_kw_requires_import"]:
        if kw[k]:
            mod, name = kw[k].rsplit(".", 1)
            kw[k] = getattr(__import__(mod, fromlist=[name]), name)
    net = ResidualEncoderUNet(input_channels=1, num_classes=len(CLASSES), **kw)
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType]):
        ck = torch.load(d / "fold_1" / "checkpoint_final.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(ck["network_weights"])
    return net.eval(), c


def gaussian(patch, sigma_scale=1 / 8):
    g = np.ones(patch, np.float32)
    for ax, n in enumerate(patch):
        x = np.arange(n) - (n - 1) / 2
        w = np.exp(-x ** 2 / (2 * (n * sigma_scale) ** 2))
        g *= w.reshape([-1 if i == ax else 1 for i in range(3)])
    return torch.from_numpy(g / g.max())


@torch.no_grad()
def predict(net, cfg, vol, spacing, step=0.5):
    """vol (z, y, x) float array in nnU-Net axis order with spacing (z, y, x) mm -> label map at the input grid (z, y, x)."""
    tgt = cfg["spacing"]
    x = torch.from_numpy(np.ascontiguousarray(vol, np.float32))[None, None]
    size = [max(1, int(round(s * sp / t))) for s, sp, t in zip(vol.shape, spacing, tgt)]
    x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    x = (x - x.mean()) / (x.std() + 1e-8)
    patch = cfg["patch_size"]
    pad = [max(0, p - s) for p, s in zip(patch, size)]
    x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    shp = x.shape[2:]
    starts = [sorted({min(i, s - p) for i in range(0, max(1, s - p + 1), max(1, int(p * step)))} | {s - p}) for s, p in zip(shp, patch)]
    acc = torch.zeros((len(CLASSES), *shp))
    wsum = torch.zeros(shp)
    g = gaussian(patch)
    for z in starts[0]:
        for yy in starts[1]:
            for xx in starts[2]:
                sl = (slice(z, z + patch[0]), slice(yy, yy + patch[1]), slice(xx, xx + patch[2]))
                p = torch.softmax(net(x[(0, 0) + sl][None, None])[0], 0)
                acc[(slice(None),) + sl] += p * g
                wsum[sl] += g
    prob = (acc / wsum)[:, : size[0], : size[1], : size[2]]
    prob = F.interpolate(prob[None], size=vol.shape, mode="trilinear", align_corners=False)[0]
    return prob.argmax(0).numpy().astype(np.uint8)


def dice(a, b, k):
    a, b = a == k, b == k
    return 2 * (a & b).sum() / max(1, a.sum() + b.sum())


if __name__ == "__main__":                               # validate on the repo's test image (nibabel axes P, I, R -> nnU-Net z, y, x)
    import nibabel as nib
    torch.set_num_threads(6)
    net, cfg = load("3d_lowres")
    im = nib.load(HERE / "test_data/test_image.nii.gz")
    gt = np.asarray(nib.load(HERE / "test_data/test_ground_truth.nii.gz").dataobj).astype(np.uint8)
    vol = np.asarray(im.dataobj, np.float32).transpose(2, 1, 0)    # (R, I, P) = (slice, rows S->I, cols A->P)
    sp = np.array(im.header.get_zooms())[[2, 1, 0]]
    pred = predict(net, cfg, vol, sp)
    gt = gt.transpose(2, 1, 0)
    print("test-image Dice (my lowres inference):", {c: round(float(dice(pred, gt, k)), 3) for k, c in enumerate(CLASSES) if k})
