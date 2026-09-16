"""P1 on ImageNet ResNets: zero-retraining width reduction of the block-INTERNAL widths
(BasicBlock: conv1 output; Bottleneck: conv1 and conv2 outputs) by column-subset selection on
the post-activation covariance, folded into the next convolution of the block.  Stage widths
(the residual stream) are not touched.

Fold of the constant term: torchvision convolutions have no bias and are followed by BN, so
the reconstruction offset c is folded into the next BN's running mean:
    BN(y + b) = gamma (y + b - mu)/sigma + beta  ->  mu' = mu - b.
Exact in the interior; the one-pixel zero-padding border is the only approximation.

    python scripts/fold_resnet.py --model resnet50 --device mps --out ../results/cssp_resnet
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.data import IMAGENET_MEAN, IMAGENET_STD                    # noqa: E402
from layerspec.models import build                                          # noqa: E402
from scripts.cssp_check import greedy_cssp, kstar                           # noqa: E402
from scripts.fold_mlp import Stats, set_module                              # noqa: E402
from scripts.projection_check import LabelledFolder, top1, wnid_index       # noqa: E402


def internal_sites(model: nn.Module):
    """(bn_name, next_conv_name, next_bn_name) for every block-internal post-activation."""
    sites = []
    for name, m in model.named_modules():
        cls = type(m).__name__
        if cls == "BasicBlock":
            sites.append((f"{name}.bn1", f"{name}.conv2", f"{name}.bn2"))
        elif cls == "Bottleneck":
            sites.append((f"{name}.bn1", f"{name}.conv2", f"{name}.bn2"))
            sites.append((f"{name}.bn2", f"{name}.conv3", f"{name}.bn3"))
    return sites


def fold_site(model: nn.Module, bn_name: str, conv_next: str, bn_next: str, mean, cov, S: list[int]):
    mods = dict(model.named_modules())
    bn, conv2, bn2 = mods[bn_name], mods[conv_next], mods[bn_next]
    conv1_name = bn_name.replace(".bn", ".conv")
    conv1 = mods[conv1_name]
    C = bn.num_features
    keep = torch.tensor(sorted(S))
    drop = torch.tensor([j for j in range(C) if j not in set(S)])
    if len(drop) == 0:
        return
    if conv2.groups != 1:
        raise ValueError(f"{conv_next}: grouped convolution not supported")
    Sss = cov[keep][:, keep]
    B = torch.linalg.pinv(Sss, rtol=1e-8) @ cov[keep][:, drop]              # |S| x |drop|
    c = mean[drop] - B.T @ mean[keep]                                       # |drop|
    W2 = conv2.weight.detach().cpu().double()                               # O x C x kh x kw
    W2k = W2[:, keep] + torch.einsum("kj,ojhw->okhw", B, W2[:, drop])
    b_const = torch.einsum("j,ojhw->o", c, W2[:, drop])                     # per output channel
    # narrow conv1 / bn
    n1 = nn.Conv2d(conv1.in_channels, len(keep), conv1.kernel_size, conv1.stride, conv1.padding,
                   bias=conv1.bias is not None)
    nb = nn.BatchNorm2d(len(keep), eps=bn.eps, momentum=bn.momentum)
    n2 = nn.Conv2d(len(keep), conv2.out_channels, conv2.kernel_size, conv2.stride, conv2.padding,
                   bias=conv2.bias is not None)
    with torch.no_grad():
        n1.weight.copy_(conv1.weight[keep])
        if conv1.bias is not None:
            n1.bias.copy_(conv1.bias[keep])
        for att in ("weight", "bias", "running_mean", "running_var"):
            getattr(nb, att).copy_(getattr(bn, att)[keep])
        nb.num_batches_tracked.copy_(bn.num_batches_tracked)
        n2.weight.copy_(W2k.float())
        if conv2.bias is not None:
            n2.bias.copy_(conv2.bias + b_const.float().to(conv2.bias.device))
        else:
            bn2.running_mean.sub_(b_const.float().to(bn2.running_mean.device))
    dev = conv1.weight.device
    set_module(model, conv1_name, n1.to(dev))
    set_module(model, bn_name, nb.to(dev))
    set_module(model, conv_next, n2.to(dev))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="resnet50")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--taus", type=float, nargs="+", default=[0.999, 0.99, 0.95])
    p.add_argument("--eps", type=float, nargs="*", default=[0.001, 0.01])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/cssp_resnet")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ds = LabelledFolder(a.data, tf, wnid_index())
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(a.seed)).tolist()
    half = len(ds) // 2
    mk = lambda d: DataLoader(d, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    cal, ev = mk(Subset(ds, perm[:half])), mk(Subset(ds, perm[half:]))

    model, tag = build(a.model, pretrained=True)
    model = model.to(a.device).eval()
    sites = internal_sites(model)
    mods = dict(model.named_modules())
    print(f"{a.model} ({tag}): {len(sites)} block-internal sites; calibrate {half}, evaluate {len(ds) - half}", flush=True)

    stats = {s[0]: Stats() for s in sites}
    # the post-activation is relu(bn(.)); hook the BN and apply the ReLU here (the block's nn.ReLU is shared)
    hs = [mods[bn].register_forward_hook(lambda m, i, o, bn=bn: stats[bn].update(F.relu(o).permute(0, 2, 3, 1)))
          for bn, _, _ in sites]
    t0 = time.time()
    with torch.no_grad():
        for xb, _ in cal:
            model(xb.to(a.device))
    for h in hs:
        h.remove()
    moments = {bn: stats[bn].finish() for bn, _, _ in sites}
    print(f"  calibration {time.time() - t0:.0f}s", flush=True)
    spec = {bn: np.clip(np.linalg.eigvalsh(moments[bn][1].numpy())[::-1], 0, None) for bn, _, _ in sites}
    piv = {bn: greedy_cssp(moments[bn][1], moments[bn][1].shape[0]) for bn, _, _ in sites}

    acc_full = top1(model, ev, a.device)
    n_full = sum(q.numel() for q in model.parameters())
    print(f"  full top-1 {acc_full:.2f}%  params {n_full / 1e6:.2f}M", flush=True)

    rows, per_site = [], []
    schemes = [("tau", t) for t in a.taus] + [("eps", e) for e in a.eps]
    for kind, val in schemes:
        narrow = copy.deepcopy(model)
        widths = []
        for bn, conv_next, bn_next in sites:
            C = spec[bn].shape[0]
            order, traces = piv[bn]
            if kind == "tau":
                m = min(kstar(spec[bn], val), C)
            else:
                tr = np.asarray(traces) / traces[0]
                m = int(np.argmax(tr <= val)) if (tr <= val).any() else C
            widths.append(m)
            per_site.append({"model": a.model, "scheme": f"{kind}_{val}", "site": bn, "C": C, "m": m,
                             "subspace_resid": float(spec[bn][m:].sum() / spec[bn].sum()) if m < C else 0.0,
                             "cssp_resid": float(traces[m] / traces[0]) if m < len(traces) else 0.0})
            fold_site(narrow, bn, conv_next, bn_next, *moments[bn], sorted(order[:m]))
        t0 = time.time()
        acc = top1(narrow, ev, a.device)
        n_narrow = sum(q.numel() for q in narrow.parameters())
        rows.append({"model": a.model, "scheme": f"{kind}_{val}", "acc_full": acc_full, "acc_fold": acc,
                     "params_full": n_full, "params_fold": n_narrow, "param_frac": n_narrow / n_full,
                     "internal_width_frac": float(np.sum(widths) / np.sum([spec[s[0]].shape[0] for s in sites])),
                     "widths": " ".join(map(str, widths))})
        print(f"  {kind} {val:g}: top-1 {acc:.2f} (full {acc_full:.2f})  params {n_narrow / n_full:.3f}  "
              f"internal widths kept {rows[-1]['internal_width_frac']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
        del narrow
    stem = a.model.replace(":", "_").replace("/", "_")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{stem}_fold.csv"), index=False)
    pd.DataFrame(per_site).to_csv(os.path.join(a.out, f"{stem}_sites.csv"), index=False)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
