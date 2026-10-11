"""The whitening arms of plan 9.56 (T3): (c) nd = Network Deconvolution as published (block 64, replaces BN),
(d) wo = full patch-covariance whitening + batch norm + polar-factor kernel (the maximiser of T3.1).

Trains a CIFAR run exactly as scripts/trajectory_cifar.py does (same wrapper pattern as trajectory_oni.py:
the architecture builder is replaced, the trainer is not edited).  At the end the final checkpoint is
re-saved BAKED into the plain architecture (scripts/whiten_conv.bake_to_reference) as
<arch>_<arm>_<dataset>_final.pt so decompose_check.py and the probes read it like any other arm.

    python scripts/trajectory_whiten.py --arm nd --arch vgg16_bn --seed 0 --device mps --out ../results/round3/a5_vgg16_bn_nd_s0
    python scripts/trajectory_whiten.py --arm wo --arch resnet_basic52 --seed 1 --device cuda --gpu-data --out ...
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.trajectory_cifar as tc                                   # noqa: E402
from scripts.whiten_conv import apply_whitening, bake_to_reference       # noqa: E402

ARMS = {
    "nd": dict(mode="nd", remove_bn=True, block=64, eps=1e-5, n_iter=5, momentum=0.1, sampling_stride=3),
    "wo": dict(mode="full", remove_bn=False, momentum=0.1, sampling_stride=3, update_every=50, gain_cap=8.0,
               warmup=50, orthogonal=True, oni_iters=20),
}


def build_arm(base, arm: str, num_classes: int):
    kw = dict(ARMS[arm])
    mode, remove_bn = kw.pop("mode"), kw.pop("remove_bn")
    return apply_whitening(base(num_classes), mode=mode, remove_bn=remove_bn, **kw)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--arch", default="vgg16_bn")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--out", required=True)
    a, rest = p.parse_known_args(argv)
    if "--ortho" in rest or "--widths" in rest:
        p.error("the whitening arms take no --ortho / --widths")
    base = tc.ARCHS[a.arch]
    tc.ARCHS[a.arch] = lambda nc=10, _b=base: build_arm(_b, a.arm, nc)
    try:
        rc = tc.main(["--arch", a.arch, "--dataset", a.dataset, "--out", a.out] + rest)
    finally:
        tc.ARCHS[a.arch] = base
    if rc != 0:
        return rc
    raw = os.path.join(a.out, f"{a.arch}_{a.dataset}_final.pt")
    num_classes = tc.CIFAR_STATS[a.dataset][2]
    model = build_arm(base, a.arm, num_classes)
    model.load_state_dict(torch.load(raw, map_location="cpu"))
    model.eval()
    baked = bake_to_reference(model, base(num_classes).eval())
    torch.save({k: v.detach().cpu() for k, v in baked.state_dict().items()},
               os.path.join(a.out, f"{a.arch}_{a.arm}_{a.dataset}_final.pt"))
    os.rename(raw, os.path.join(a.out, f"{a.arch}_{a.arm}_{a.dataset}_final_unbaked.pt"))
    print(f"baked whitened kernels -> {a.arch}_{a.arm}_{a.dataset}_final.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
