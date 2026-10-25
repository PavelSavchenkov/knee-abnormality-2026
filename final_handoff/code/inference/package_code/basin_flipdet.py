"""Inference half of experiments/v0/oct10_baseline_3rd/src/flipdet.py (= oct8_baseline_v2/src/flipdet.py), verbatim: the
self-supervised mirror detector (canonical cache series = 0, mirrored copy = 1) and its input. Weights: oct8_baseline_v2/runs/
flipdet/flipdet.pt (trained on folds 0 / 2 / 3 / 4, fold 1 held out)."""
import numpy as np
import timm
import torch
import torch.nn.functional as F

SIZE = 224


def det_slices(n, plane):
    """Slice indices read for the detector input of a stack of n slices. Sagittal: [k, lo, hi, n-1-k] with k ~ 25 % and the
    two central slices (lo, hi) averaged into the middle channel, so reversing the stack is exactly a channel reversal."""
    if plane.startswith("Sag"):
        k = int(round(0.25 * (n - 1)))
        return [k, (n - 1) // 2, n // 2, n - 1 - k]
    c = (n - 1) // 2
    return [max(c - 1, 0), c, min(c + 1, n - 1)]


def combine(sl, plane):
    """Slices read at det_slices -> (3, H, W) uint8."""
    if plane.startswith("Sag"):
        mid = ((sl[1].astype(np.uint16) + sl[2] + 1) // 2).astype(np.uint8)
        return np.stack([sl[0], mid, sl[3]])
    return np.stack(sl)


def det_input(vol, plane):
    """(3, H, W) uint8 detector input from a canonical volume (n, H, W)."""
    return combine([vol[k] for k in det_slices(len(vol), plane)], plane)


def det_mirror(img, plane):
    """Mirror a detector input (torch or numpy, (..., 3, H, W)) the way the whole series would be mirrored."""
    return img.flip(-3) if plane.startswith("Sag") else img.flip(-1)


class FlipNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=pretrained, num_classes=1)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):  # x: (N, 3, H, W) uint8 / float [0, 255] -> logit of "mirrored"
        x = F.interpolate(x.float() / 255, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
        return self.enc((x - self.mean) / self.std)[:, 0]
