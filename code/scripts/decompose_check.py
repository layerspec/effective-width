"""Where does a layer's effective width come from: the kernel or the data?

At the tensor the paper measures (conv output, before the nonlinearity and
the shortcut) the channel covariance is exactly

    Sigma_out = W Sigma_patch W^T,

W the (C_out x d) kernel matrix, d = C_in*k_h*k_w, Sigma_patch the d x d
covariance of the receptive-field patch (im2col).  This script accumulates
Sigma_patch per dense conv layer on the 6,400-image subset, takes W from the
checkpoint, and reports for each layer, at each tau:

    out      k*(W Sigma W^T) / r_max          the measured quantity (recomputed)
    kernel   k*(W W^T) / r_max                what the layer would show under white patches
    data     k*(Sigma_patch) / r_max          the patch covariance itself, capped by r_max
    ortho    k*(W_o Sigma W_o^T) / r_max      W_o = polar factor of W (nearest isometry):
                                              what orthogonalising THIS kernel would give
                                              with THIS data, no training
plus the kernel's own spectrum (effective rank of W W^T over r_max, and the
condition number of its top-r_max singular values).  Referee-driven addition
of 2026-09-07; see notes/orthogonality-novelty-2026-09-07.md section 3.

    python scripts/decompose_check.py --model resnet50 --data ../data/imagenet_val_6400 \
        --device mps --out ../results/decompose
    # a CIFAR-10 network from trajectory_cifar.py (round 3, A5):
    python scripts/decompose_check.py --cifar-arch vgg16_bn --checkpoint ../results/round3/a5_vgg16_bn_so_s0/vgg16_bn_so_cifar10_final.pt \
        --data ../data --device mps --out ../results/round3_decompose
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.accumulate import CovarianceAccumulator   # noqa: E402
from layerspec.data import build_loader                  # noqa: E402
from layerspec.models import build                       # noqa: E402

TAUS = [0.9, 0.95, 0.99, 0.999]


def kstar(eig: np.ndarray, tau: float) -> int:
    eig = np.clip(eig, 0, None)
    if eig.sum() <= 0:
        return 0
    c = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(c, tau) + 1)


def erank(eig: np.ndarray) -> float:
    eig = np.clip(eig, 0, None); p = eig / eig.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


class PatchProbe:
    def __init__(self, model: nn.Module, positions: int, seed: int, device):
        self.acc, self.meta, self.handles = {}, {}, []
        self.p = positions
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device
        order = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                self.meta[name] = dict(depth_index=order, module=m)
                order += 1
                self.handles.append(m.register_forward_pre_hook(
                    lambda mod, inp, _n=name: self._consume(_n, mod, inp[0])))

    @torch.no_grad()
    def _consume(self, name, m: nn.Conv2d, x: torch.Tensor):
        N = x.shape[0]
        cols = F.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride)  # (N, d, L)
        d, L = cols.shape[1], cols.shape[2]
        if L == 1:
            return                                   # conv on a pooled tensor: not in the profile
        p = min(self.p, L)
        idx = torch.randint(0, L, (N, p), generator=self.gen).to(cols.device)
        cols = torch.gather(cols, 2, idx.unsqueeze(1).expand(N, d, p))   # (N, d, p)
        xs = cols.permute(0, 2, 1).reshape(N * p, d).to(torch.float32)
        acc = self.acc.get(name)
        if acc is None:
            acc = self.acc[name] = CovarianceAccumulator(d)
            self.meta[name].update(d=d, C_in=m.in_channels, C_out=m.out_channels,
                                   k=m.kernel_size[0] * m.kernel_size[1])
        bm = xs.mean(0); xc = xs - bm
        acc.update_precomputed(xs.shape[0], bm.cpu().double().numpy(), (xc.T @ xc).cpu().double().numpy())

    def remove(self):
        for h in self.handles:
            h.remove()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="resnet50")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--positions", type=int, default=48)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/decompose")
    p.add_argument("--cifar-arch", default=None,
                   help="a CIFAR-10 architecture from trajectory_cifar.ARCHS; with --checkpoint, "
                        "decompose a network trained by trajectory_cifar.py on the CIFAR-10 test split")
    p.add_argument("--checkpoint", default=None, help="state_dict .pt for --cifar-arch (omit = random init)")
    p.add_argument("--name", default=None, help="output name (default: --model or the checkpoint stem)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if a.cifar_arch:
        from scripts.trajectory_cifar import ARCHS
        from scripts.reproduce_garg import cifar_loaders
        torch.manual_seed(a.seed)
        model = ARCHS[a.cifar_arch]()
        if a.checkpoint:
            model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
        tag = a.checkpoint or f"random_init(seed={a.seed})"
        _, loader = cifar_loaders(a.data, a.batch_size, a.workers, "cifar10")
        a.model = a.name or (os.path.splitext(os.path.basename(a.checkpoint))[0] if a.checkpoint
                             else f"{a.cifar_arch}_random{a.seed}")
    else:
        loader, src = build_loader(a.data, a.batch_size, a.workers, limit=a.limit, seed=a.seed)
        model, tag = build(a.model, pretrained=True, seed=a.seed)
        a.model = a.name or a.model
    model = model.to(a.device).eval()
    probe = PatchProbe(model, a.positions, a.seed, a.device)
    t0 = time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(a.device))
    probe.remove()
    print(f"{a.model} ({tag}): {len(probe.acc)} dense conv layers, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for name, acc in probe.acc.items():
        meta = probe.meta[name]; m = meta["module"]
        d, C_out = meta["d"], meta["C_out"]
        r_max = min(d, C_out)
        W = m.weight.detach().cpu().double().reshape(C_out, d).numpy()
        Sig = acc.covariance
        lam_data = np.sort(np.linalg.eigvalsh(Sig))[::-1]
        S_out = W @ Sig @ W.T
        lam_out = np.sort(np.linalg.eigvalsh((S_out + S_out.T) / 2))[::-1]
        G = W @ W.T
        lam_kernel = np.sort(np.linalg.eigvalsh((G + G.T) / 2))[::-1]
        U, sv, Vt = np.linalg.svd(W, full_matrices=False)
        Wo = U @ Vt                                     # polar factor, nearest (semi-)isometry
        S_o = Wo @ Sig @ Wo.T
        lam_ortho = np.sort(np.linalg.eigvalsh((S_o + S_o.T) / 2))[::-1]
        row = dict(model=a.model, layer=name, depth_index=meta["depth_index"], C_in=meta["C_in"],
                   C_out=C_out, k=meta["k"], d=d, r_max=r_max, n_samples=acc.n, n_over_d=acc.n / d,
                   kernel_erank_over_rmax=erank(sv ** 2) / r_max,
                   kernel_cond_top_rmax=float(sv[0] / sv[min(r_max, len(sv)) - 1]),
                   data_erank_over_rmax=min(erank(lam_data), r_max) / r_max)
        for tau in TAUS:
            row[f"out_{tau}"] = kstar(lam_out, tau) / r_max
            row[f"kernel_{tau}"] = kstar(lam_kernel, tau) / r_max
            row[f"data_{tau}"] = min(kstar(lam_data, tau), r_max) / r_max
            row[f"ortho_{tau}"] = kstar(lam_ortho, tau) / r_max
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("depth_index")
    path = os.path.join(a.out, f"{a.model.replace(':', '_')}_decompose.csv")
    df.to_csv(path, index=False)
    print(df[["layer", "d", "C_out", "n_over_d", "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95",
              "kernel_erank_over_rmax"]].to_string(index=False), flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
