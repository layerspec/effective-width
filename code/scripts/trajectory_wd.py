"""The weight-decay sweep of B7 (plan 9.27 B, P-b7.4): a CIFAR run exactly as scripts/trajectory_cifar.py trains it, with the
optimiser's weight decay replaced.  A wrapper so the trainer the Mac queues import is not edited while they run: torch.optim.SGD is
swapped for a subclass that forces weight_decay, trajectory_cifar.main runs unchanged, and <out>/wd.json records the value.

    python scripts/trajectory_wd.py --wd 5e-3 --arch vgg16_bn --seed 0 --device mps --save-checkpoints \\
        --schedule 0 40 50 60 70 80 90 100 --out ../results/wd_sweep/wd5e-03_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.trajectory_cifar as tc                      # noqa: E402


def sgd_with_decay(wd: float):
    real = torch.optim.SGD

    class SGD(real):
        def __init__(self, params, *args, **kw):
            kw["weight_decay"] = wd
            super().__init__(params, *args, **kw)
    return SGD


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wd", type=float, required=True)
    a, rest = p.parse_known_args(argv)
    real = torch.optim.SGD
    torch.optim.SGD = sgd_with_decay(a.wd)
    try:
        rc = tc.main(rest)
    finally:
        torch.optim.SGD = real
    out = rest[rest.index("--out") + 1] if "--out" in rest else "../results/trajectory"
    if os.path.isdir(out):
        with open(os.path.join(out, "wd.json"), "w") as fh:
            json.dump({"weight_decay": a.wd}, fh)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
