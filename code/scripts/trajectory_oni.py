"""The ONI arm of the orthogonality intervention (plan 9.2, submission bar D5; 2026-09-24).

Trains a CIFAR run exactly as scripts/trajectory_cifar.py does, with every dense conv / linear
kernel orthogonalised by construction (scripts/oni.py) instead of penalised (--ortho so/srip).
Implemented as a wrapper so the trainer the Mac queues import is not edited while they run:
the architecture builder is replaced by one that applies the parametrisation, then
trajectory_cifar.main runs unchanged (--ortho none).  When the run ends the checkpoint is
re-saved BAKED (orthogonalised kernels as plain weights, plain state_dict keys) as
<arch>_oni_<dataset>_final.pt so decompose_check.py and checkpoint_analysis.py read it like
any other arm.

    python scripts/trajectory_oni.py --arch vgg16_bn --seed 0 --device cuda --gpu-data \\
        --out ../results/round3/a5_vgg16_bn_oni_s0 [--oni-iters 20] [trajectory_cifar.py args]
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.trajectory_cifar as tc                      # noqa: E402
from scripts.oni import apply_oni, bake_oni                # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    p.add_argument("--oni-iters", type=int, default=20, help="Newton iterations T (Huang 2020 use 5: partial; 20 = exact isometry, see oni.py)")
    p.add_argument("--arch", default="vgg16_bn")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--out", required=True)
    a, rest = p.parse_known_args(argv)
    if "--ortho" in rest or "--widths" in rest:
        p.error("the ONI arm takes no --ortho / --widths")
    base = tc.ARCHS[a.arch]
    tc.ARCHS[a.arch] = lambda nc=10, _b=base: apply_oni(_b(nc), a.oni_iters)
    try:
        rc = tc.main(["--arch", a.arch, "--dataset", a.dataset, "--out", a.out] + rest)
    finally:
        tc.ARCHS[a.arch] = base
    if rc != 0:
        return rc
    raw = os.path.join(a.out, f"{a.arch}_{a.dataset}_final.pt")
    num_classes = tc.CIFAR_STATS[a.dataset][2]
    model = apply_oni(base(num_classes), a.oni_iters)
    model.load_state_dict(torch.load(raw, map_location="cpu"))
    bake_oni(model)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
               os.path.join(a.out, f"{a.arch}_oni_{a.dataset}_final.pt"))
    os.remove(raw)
    print(f"baked orthogonalised kernels -> {a.arch}_oni_{a.dataset}_final.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
