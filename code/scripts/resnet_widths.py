"""Width-parameterised ResNet-18 for the width-from-the-ruler experiment on a
basic-block network (analysis-plan 9.11; the CIFAR preflight uses the same
builder with the CIFAR stem).

A basic block is  out = relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x)).
The residual sum forces every block of a stage to share ONE output width
(the stage width); the conv1 output of each block is free.  So ResNet-18
(stem + 4 stages x 2 blocks) has 13 free widths, listed flat in this order:

    [stem,
     stage1_out, stage1_block0_mid, stage1_block1_mid,
     stage2_out, stage2_block0_mid, stage2_block1_mid,
     stage3_out, stage3_block0_mid, stage3_block1_mid,
     stage4_out, stage4_block0_mid, stage4_block1_mid]

With RESNET18_WIDTHS it is exactly torchvision's resnet18 (same module names,
same state_dict keys, same parameter count).  The block class is named
BasicBlock on purpose: layerspec hooks residual blocks by class name.

Width rule from a measured profile (`widths_from_profile`, plan 9.11): every
width is read AFTER the ReLU that the next layer consumes,
    stem            = k*(0.999) at relu#0
    block mid       = k*(0.999) at layerS.B.relu#0   (after conv1's ReLU)
    stage out       = max over the stage's blocks of k*(0.999) at
                      layerS.B.relu#1                (after the residual add + ReLU)
each ceil'd and floored at MIN_WIDTH.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import conv1x1, conv3x3

RESNET18_WIDTHS = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512)
N_WIDTHS = 13
MIN_WIDTH = 8
STAGE_STRIDES = (1, 2, 2, 2)
BLOCKS_PER_STAGE = 2


class BasicBlock(torchvision.models.resnet.BasicBlock):
    """torchvision's BasicBlock with independent input, mid and output widths.
    The shortcut is a 1x1 conv + BN whenever the stride or the width changes,
    exactly as in torchvision (identity otherwise)."""

    def __init__(self, c_in: int, c_mid: int, c_out: int, stride: int = 1):
        downsample = None
        if stride != 1 or c_in != c_out:
            downsample = nn.Sequential(conv1x1(c_in, c_out, stride), nn.BatchNorm2d(c_out))
        super().__init__(c_in, c_mid, stride, downsample)
        self.conv2 = conv3x3(c_mid, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)


class ResNet18W(nn.Module):
    """ResNet-18 whose 13 widths are given explicitly (see module docstring).
    stem="imagenet": 7x7 stride-2 conv + max-pool (torchvision);
    stem="cifar":    3x3 stride-1 conv, no max-pool (as build_resnet50_cifar)."""

    def __init__(self, widths=RESNET18_WIDTHS, num_classes: int = 1000, stem: str = "imagenet"):
        super().__init__()
        widths = [int(w) for w in widths]
        if len(widths) != N_WIDTHS or min(widths) < 1:
            raise ValueError(f"need {N_WIDTHS} positive widths, got {widths}")
        if stem not in ("imagenet", "cifar"):
            raise ValueError(f"stem must be 'imagenet' or 'cifar', got {stem!r}")
        self.widths = tuple(widths)
        self.stem = stem
        w_stem = widths[0]
        if stem == "imagenet":
            self.conv1 = nn.Conv2d(3, w_stem, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, w_stem, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        self.bn1 = nn.BatchNorm2d(w_stem)
        self.relu = nn.ReLU(inplace=True)
        c_in = w_stem
        for s in range(4):
            out, mids = stage_widths(widths, s)
            blocks = []
            for b, mid in enumerate(mids):
                blocks.append(BasicBlock(c_in, mid, out, stride=STAGE_STRIDES[s] if b == 0 else 1))
                c_in = out
            setattr(self, f"layer{s + 1}", nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c_in, num_classes)
        # torchvision's initialisation (zero_init_residual=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def stage_widths(widths, s: int) -> tuple[int, list[int]]:
    """(stage output width, [block mid widths]) of stage s (0-based) from the flat list."""
    i = 1 + s * (1 + BLOCKS_PER_STAGE)
    return int(widths[i]), [int(w) for w in widths[i + 1: i + 1 + BLOCKS_PER_STAGE]]


def build_resnet18_widths(widths, num_classes: int = 1000, stem: str = "imagenet") -> nn.Module:
    return ResNet18W(widths, num_classes, stem)


def n_params(widths, num_classes: int = 1000, stem: str = "imagenet") -> int:
    return sum(p.numel() for p in build_resnet18_widths(widths, num_classes, stem).parameters())


def uniform_widths_matching(target_params: int, tol: float = 0.02, num_classes: int = 1000,
                            stem: str = "imagenet", base=RESNET18_WIDTHS):
    """Scale `base` by one factor (bisection) until the parameter count is
    within `tol` of `target_params`; widths rounded and floored at MIN_WIDTH.
    Returns (widths, n_params, factor)."""
    lo, hi = 0.01, 1.0
    best = None
    for _ in range(40):
        f = (lo + hi) / 2
        w = [max(MIN_WIDTH, int(round(f * c))) for c in base]
        n = n_params(w, num_classes, stem)
        best = (w, n, f)
        if abs(n - target_params) <= tol * target_params:
            break
        if n > target_params:
            hi = f
        else:
            lo = f
    return best


def _kstar(table, layer: str, call_index: int, col: str) -> float:
    rows = table[(table["kind"] == "act") & (table["layer"] == layer) & (table["call_index"] == call_index)]
    if len(rows) != 1:
        raise ValueError(f"expected one act row for {layer}#{call_index}, found {len(rows)}")
    if not bool(rows["ok"].iloc[0]):
        raise ValueError(f"{layer}#{call_index} fails the n/C >= 50 gate; use more images")
    return float(rows[col].iloc[0])


def widths_from_profile(table, tau: float = 0.999, min_width: int = MIN_WIDTH) -> dict:
    """Apply the plan-9.11 rule to a layerspec profile table of a ResNet18W
    (measured with include_activations=True).  Returns the flat 13-width list
    plus the per-site readings it was built from."""
    col = f"k_star_{tau}"
    w = lambda k: max(min_width, math.ceil(k))                         # noqa: E731
    sites = {"stem": _kstar(table, "relu", 0, col)}
    flat = [w(sites["stem"])]
    for s in range(4):
        outs, mids = [], []
        for b in range(BLOCKS_PER_STAGE):
            mids.append(_kstar(table, f"layer{s + 1}.{b}.relu", 0, col))
            outs.append(_kstar(table, f"layer{s + 1}.{b}.relu", 1, col))
        sites[f"stage{s + 1}"] = {"block_out": outs, "block_mid": mids}
        flat.append(w(max(outs)))
        flat.extend(w(m) for m in mids)
    return {"widths": flat, "sites": sites, "tau": tau, "min_width": min_width}


def macs(model: nn.Module, input_hw: tuple[int, int] = (224, 224)) -> int:
    """Multiply-adds of one forward pass at `input_hw` (convs and linears; BN/ReLU/pool ignored)."""
    total = [0]

    def conv_hook(m, i, o):
        total[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        total[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hs += [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, *input_hw, device=dev))
    for h in hs:
        h.remove()
    model.train(was_training)
    return total[0]
