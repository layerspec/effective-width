"""ResNet-20 with every stage widened by --width-mult, trained by trajectory_cifar.main with the same
controller (P2l, 2026-09-25; hostile review 09-25 E1b / (iv)): does idle width appear when a narrow
residual network is made wide relative to a ten-class task, and does the controller then beat a uniform
network on a second architecture?

Implemented as a wrapper so the trainer the Mac queues import is not edited while they run: the
module-level builder `build_resnet20_cifar` of trajectory_cifar is replaced by the wide one, which the
`--arch resnet20` entry of ARCHS and the resume path (`build_with_widths`) both resolve at call time,
so `--realloc-budget`, `--widths` (9 block-internal widths) and resume.pt all work unchanged.

    python scripts/trajectory_r20wide.py --width-mult 4 --device mps --dataset cifar10 --seed 0 \\
        --out ../results/p2/r20w4_full_s0 [trajectory_cifar.py args, e.g. --realloc-budget 0.6 --realloc-cost flops]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.trajectory_cifar as tc                      # noqa: E402


class WideCifarResNet(nn.Module):
    """CifarResNet of trajectory_cifar (He 2016 section 4.2, option-A shortcuts) with stage widths
    16m / 32m / 64m; `inner` = the 3n block-internal widths (the controller's sites), default = stage width."""

    def __init__(self, n: int = 3, num_classes: int = 10, inner=None, mult: int = 1):
        super().__init__()
        planes0 = 16 * mult
        self.conv1 = nn.Conv2d(3, planes0, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes0)
        self.relu = nn.ReLU(inplace=True)
        inner = None if inner is None else [int(w) for w in inner]
        if inner is not None and (len(inner) != 3 * n or min(inner) < 1):
            raise ValueError(f"inner widths need {3 * n} positive integers, got {inner}")
        layers, inp, b = [], planes0, 0
        for planes, stride in ((16 * mult, 1), (32 * mult, 2), (64 * mult, 2)):
            for i in range(n):
                layers.append(tc._CifarBasicBlock(inp, planes, stride if i == 0 else 1,
                                                  None if inner is None else inner[b]))
                inp, b = planes, b + 1
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * mult, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_resnet20_wide(num_classes: int = 10, widths=None, mult: int = 4):
    return WideCifarResNet(3, num_classes, widths, mult)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    p.add_argument("--width-mult", type=int, default=4)
    p.add_argument("--arch", default="resnet20")
    a, rest = p.parse_known_args(argv)
    if a.arch != "resnet20":
        p.error("this wrapper widens --arch resnet20 only")
    base = tc.build_resnet20_cifar
    tc.build_resnet20_cifar = lambda nc=10, widths=None, _m=a.width_mult: build_resnet20_wide(nc, widths, _m)
    try:
        return tc.main(["--arch", "resnet20"] + rest)
    finally:
        tc.build_resnet20_cifar = base


if __name__ == "__main__":
    raise SystemExit(main())
