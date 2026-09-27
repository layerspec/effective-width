"""Width-parameterised VGG-16 (BN) with the ImageNet head (A27, plan 9.16, 2026-09-27): torchvision's vgg16_bn
with the 13 convolution widths given explicitly; the first classifier layer follows the last width.  With
VGG16_WIDTHS it is exactly torchvision.models.vgg16_bn (same module names, same parameter count)."""
from __future__ import annotations

import torch.nn as nn
import torchvision
from torchvision.models.vgg import VGG, make_layers

VGG16_WIDTHS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
_POOL_AFTER = {1, 3, 6, 9, 12}          # a max-pool follows these conv indices (cfg 'D')


def _cfg(widths):
    widths = [int(w) for w in widths]
    if len(widths) != 13 or min(widths) < 1:
        raise ValueError(f"need 13 positive widths, got {widths}")
    cfg = []
    for i, w in enumerate(widths):
        cfg.append(w)
        if i in _POOL_AFTER:
            cfg.append("M")
    return cfg


def build_vgg16_bn_imagenet(widths=VGG16_WIDTHS, num_classes: int = 1000) -> nn.Module:
    m = VGG(make_layers(_cfg(widths), batch_norm=True), num_classes=num_classes, init_weights=True)
    last = int(widths[-1])
    if last != 512:
        m.classifier[0] = nn.Linear(last * 7 * 7, 4096)
        nn.init.normal_(m.classifier[0].weight, 0, 0.01); nn.init.constant_(m.classifier[0].bias, 0)
    return m


def n_params(widths=VGG16_WIDTHS, num_classes: int = 1000) -> int:
    return sum(p.numel() for p in build_vgg16_bn_imagenet(widths, num_classes).parameters())
