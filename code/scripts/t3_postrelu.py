"""Post-ReLU width against the width of the tensor the ReLU reads (plan 9.56, P9.130; proofs T3.3).

For a trained CIFAR network (plain architecture, any arm once baked), one traced forward pass pairs every
BatchNorm2d output with the activation module (name, call index) that reads it directly, and every dense conv
with its BN; then one SpectrumProbe pass over the CIFAR-10 test split (16 positions per image) records the
channel covariances of conv outputs, BN outputs and activation outputs, and the script writes per pair
    k*(tau) of the conv output, of the BN output (what the ReLU reads) and of the ReLU output,
    ratio_tau = k*_act / k*_bn,  ceiling = ceil(tau C)/r_max.
Under the Gaussian surrogate with white input the ratio is 1 (T3.3); the measured ratio is the real-data test.

    python scripts/t3_postrelu.py --cifar-arch vgg16_bn --checkpoint ../results/round3/a5_vgg16_bn_wo_s0/vgg16_bn_wo_cifar10_final.pt \
        --data ../data --device mps --out ../results/t3/postrelu_a5_vgg16_bn_wo_s0.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.decomposition import kstar      # noqa: E402
from layerspec.hooks import ACTIVATIONS, run_probe  # noqa: E402

TAUS = (0.95, 0.999)


@torch.no_grad()
def trace_pairs(model: nn.Module, sample: torch.Tensor):
    """[(conv_name, bn_name, act_name, act_call_index, C, r_max)] for every conv -> BN -> activation chain
    in which each arrow is a direct tensor hand-over (in-place ReLUs included: the hook fires before the
    in-place write; a residual `out += identity` between BN and ReLU changes the tensor version and breaks the chain)."""
    keep, conv_out, bn_in, bn_out, act_in, calls, hooks = [], {}, {}, {}, {}, {}, []

    def on_conv(mod, i, o, n):
        keep.append(o); conv_out[id(o)] = n

    def on_bn_pre(mod, i, n):
        keep.append(i[0]); bn_in[n] = id(i[0])

    def on_bn(mod, i, o, n):
        keep.append(o); bn_out[id(o)] = (n, o._version)        # version: an in-place `out += identity` after the BN
                                                               # (torchvision BasicBlock) must not count as a direct hand-over

    def on_act_pre(mod, i, n):
        c = calls.get(n, 0); calls[n] = c + 1
        keep.append(i[0]); act_in.setdefault((id(i[0]), i[0]._version), (n, c))

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1:
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: on_conv(mod, i, o, n)))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_pre_hook(lambda mod, i, n=name: on_bn_pre(mod, i, n)))
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: on_bn(mod, i, o, n)))
        elif isinstance(m, ACTIVATIONS):
            hooks.append(m.register_forward_pre_hook(lambda mod, i, n=name: on_act_pre(mod, i, n)))
    was_training = model.training
    model.eval()
    model(sample)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    mods = dict(model.named_modules())
    pairs = []
    for bn, t_in in bn_in.items():
        conv = conv_out.get(t_in)
        if conv is None:
            continue
        t_out = next(((t, v) for t, (b, v) in bn_out.items() if b == bn), None)
        act = act_in.get(t_out)
        if act is None:
            continue
        m = mods[conv]
        d = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        pairs.append((conv, bn, act[0], act[1], m.out_channels, min(d, m.out_channels)))
    return pairs


def postrelu_table(model: nn.Module, loader, device: str, positions: int = 16, seed: int = 0, max_batches=None) -> pd.DataFrame:
    pairs = trace_pairs(model, torch.zeros(1, 3, 32, 32))
    probe = run_probe(model, loader, device=device, positions_per_image=positions, include_activations=True,
                      include_bn=True, include_blocks=False, pooled=False, progress=False, seed=seed, max_batches=max_batches)
    recs = {f"{r.name}::{r.kind}#{r.call_index}": r for r in probe.records() if r.acc is not None and r.acc.n >= 2}
    rows = []
    for conv, bn, act, ci, C, r_max in pairs:
        rc, rb, ra = recs.get(f"{conv}::conv#0"), recs.get(f"{bn}::bn#0"), recs.get(f"{act}::act#{ci}")
        if rc is None or rb is None or ra is None:
            continue
        row = dict(conv=conv, bn=bn, act=f"{act}#{ci}", C=C, r_max=r_max, n=int(ra.acc.n))
        for tau in TAUS:
            tk = str(tau)[2:]
            kc, kb, ka = (kstar(r.acc.eigenvalues(), tau) for r in (rc, rb, ra))
            row.update({f"conv_{tk}": kc / r_max, f"bn_{tk}": kb / r_max, f"act_{tk}": ka / r_max,
                        f"ratio_{tk}": ka / kb if kb else float("nan"), f"ceiling_{tk}": min(math.ceil(tau * C), r_max) / r_max})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cifar-arch", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="../data")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from scripts.trajectory_cifar import ARCHS
    from scripts.reproduce_garg import cifar_loaders
    torch.manual_seed(a.seed)
    model = ARCHS[a.cifar_arch]()
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    _, loader = cifar_loaders(a.data, a.batch_size, 2, "cifar10")
    df = postrelu_table(model, loader, a.device, a.positions, a.seed, a.max_batches)
    df.insert(0, "checkpoint", os.path.basename(a.checkpoint))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df[["conv", "act", "C", "conv_95", "bn_95", "act_95", "ratio_95", "ceiling_95"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
