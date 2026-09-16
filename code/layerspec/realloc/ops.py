"""The two operators that change a width, written against Site alone.

shrink   keep a column subset S of the post-activation (Proposition A), reconstruct the dropped
         channels linearly from S and fold the reconstruction into every consumer; the constant
         goes to the consumer's bias or, when it has none, into the running mean of the
         BatchNorm that follows it.  Exact up to zero-padding borders and any pooling between
         the activation and the consumer.
grow     add k output channels to the producer along the input directions that carry the most
         variance outside the producer's row space (Proposition B); the consumers' new columns
         are zero and the norm's new channels are identity, so the function is unchanged bit
         for bit.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .sites import Site
from .stats import Moments


def set_module(model: nn.Module, name: str, new: nn.Module) -> None:
    """Replace the module `name`, keeping the replaced module's train/eval mode (a fresh
    BatchNorm is in training mode by default, which would silently switch to batch statistics)."""
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    old = getattr(parent, parts[-1])
    new.train(old.training)
    setattr(parent, parts[-1], new)


# ----------------------------------------------------------------------------- column subset

def greedy_cssp(cov: torch.Tensor, m: int | None = None):
    """Pivoted Cholesky on a covariance: at each step keep the channel with the largest residual
    variance.  Returns (pivot order, residual trace after 0..m picks)."""
    R = cov.clone().double()
    C = R.shape[0]
    m = C if m is None else min(m, C)
    order, traces = [], [float(torch.trace(R))]
    taken = torch.zeros(C, dtype=torch.bool)
    for _ in range(m):
        d = torch.diagonal(R).clone()
        d[taken] = -1.0
        j = int(torch.argmax(d))
        if d[j] <= 1e-12 * traces[0]:
            break
        r = R[:, j] / R[j, j]
        R = R - torch.outer(r, R[j, :])
        R = 0.5 * (R + R.T)
        taken[j] = True
        order.append(j)
        traces.append(max(float(torch.trace(R)), 0.0))
    return order, np.asarray(traces)


def removable_width(traces: np.ndarray, eps: float) -> int:
    """m_S(eps): the smallest subset size whose CSSP residual fraction is <= eps."""
    tr = traces / traces[0]
    hit = np.nonzero(tr <= eps)[0]
    return int(hit[0]) if len(hit) else len(traces) - 1


def reconstruction(mom: Moments, keep: list[int]):
    """B (|S| x |drop|) and c (|drop|) with a_drop ~ c + B^T a_keep, plus the index tensors."""
    cov, mean = mom.act_cov, mom.act_mean
    C = cov.shape[0]
    keep_t = torch.tensor(sorted(keep))
    drop_t = torch.tensor([j for j in range(C) if j not in set(keep)])
    if len(drop_t) == 0:
        return keep_t, drop_t, torch.zeros(len(keep_t), 0, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)
    Sss = cov[keep_t][:, keep_t]
    B = torch.linalg.pinv(Sss, rtol=1e-8) @ cov[keep_t][:, drop_t]
    c = mean[drop_t] - B.T @ mean[keep_t]
    return keep_t, drop_t, B, c


# ----------------------------------------------------------------------------- shrink

def _new_conv_like(conv: nn.Conv2d, in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                     conv.groups, conv.bias is not None, conv.padding_mode)


@torch.no_grad()
def shrink(model: nn.Module, site: Site, keep: list[int], mom: Moments) -> None:
    """Keep the channels `keep` of `site`; fold the reconstruction of the rest into the consumers."""
    mods = dict(model.named_modules())
    prod = mods[site.producer]
    keep_t, drop_t, B, c = reconstruction(mom, keep)
    dev = prod.weight.device
    # producer rows
    if isinstance(prod, nn.Conv2d):
        n = _new_conv_like(prod, prod.in_channels, len(keep_t))
    else:
        n = nn.Linear(prod.in_features, len(keep_t), bias=prod.bias is not None)
    n.weight.copy_(prod.weight[keep_t])
    if prod.bias is not None:
        n.bias.copy_(prod.bias[keep_t])
    set_module(model, site.producer, n.to(dev))
    # norm
    if site.norm is not None:
        bn = mods[site.norm]
        if isinstance(bn, nn.BatchNorm2d):
            nb = nn.BatchNorm2d(len(keep_t), bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
            for att in ("weight", "bias", "running_mean", "running_var"):
                if getattr(bn, att) is not None:
                    getattr(nb, att).copy_(getattr(bn, att)[keep_t])
            if bn.num_batches_tracked is not None:
                nb.num_batches_tracked.copy_(bn.num_batches_tracked)
        elif isinstance(bn, nn.LayerNorm):
            nb = nn.LayerNorm(len(keep_t), bn.eps, bn.elementwise_affine)
            if bn.elementwise_affine:
                nb.weight.copy_(bn.weight[keep_t]); nb.bias.copy_(bn.bias[keep_t])
        else:
            raise TypeError(f"norm {site.norm}: {type(bn).__name__} not supported")
        set_module(model, site.norm, nb.to(dev))
    # consumers
    for con in site.consumers:
        m = mods[con.name]
        W = m.weight.detach().cpu().double()
        if len(drop_t):
            if con.kind == "conv":
                Wk = W[:, keep_t] + torch.einsum("kj,ojhw->okhw", B, W[:, drop_t])
                const = torch.einsum("j,ojhw->o", c, W[:, drop_t])
            else:
                Wk = W[:, keep_t] + W[:, drop_t] @ B.T
                const = W[:, drop_t] @ c
        else:
            Wk, const = W[:, keep_t], torch.zeros(W.shape[0], dtype=torch.float64)
        if con.kind == "conv":
            nm = _new_conv_like(m, len(keep_t), m.out_channels)
        else:
            nm = nn.Linear(len(keep_t), m.out_features, bias=m.bias is not None)
        nm.weight.copy_(Wk.to(m.weight.dtype))
        if m.bias is not None:
            nm.bias.copy_(m.bias)
        set_module(model, con.name, nm.to(dev))
        kind, target = con.const
        if kind == "bias":
            mods2 = dict(model.named_modules())
            mods2[target].bias.add_(const.to(dev, mods2[target].bias.dtype))
        elif kind == "bn_mean":
            mods[target].running_mean.sub_(const.to(dev, mods[target].running_mean.dtype))
        else:
            raise ValueError(con.const)


# ----------------------------------------------------------------------------- grow

def unmet_directions(prod: nn.Module, patch_cov: torch.Tensor, k: int):
    """Top-k eigenvectors of the input patch covariance restricted to the orthogonal complement
    of the producer's row space, with their variances."""
    W = prod.weight.detach().cpu().double().reshape(prod.weight.shape[0], -1)       # C_out x d
    d = W.shape[1]
    if patch_cov.shape[0] != d:
        raise ValueError(f"patch covariance is {patch_cov.shape[0]}-dimensional but the producer's input patch "
                         f"has {d} dimensions: an upstream width changed since measure(); re-measure, or grow "
                         f"sites from the deepest to the shallowest")
    Q, _ = torch.linalg.qr(W.T, mode="reduced")                                     # d x r
    P = torch.eye(d, dtype=torch.float64) - Q @ Q.T
    M = P @ patch_cov.double() @ P
    M = 0.5 * (M + M.T)
    evals, evecs = torch.linalg.eigh(M)
    idx = torch.argsort(evals, descending=True)[:k]
    return evecs[:, idx], evals[idx]


@torch.no_grad()
def grow(model: nn.Module, site: Site, k: int, mom: Moments, scale: float | None = None) -> torch.Tensor:
    """Add k channels to `site` along the unmet input directions.  Function-preserving.
    Returns the variances the new directions carry (the gain Proposition B bounds)."""
    if mom.patch_cov is None:
        raise ValueError("grow needs patch statistics (measure(..., patch=True))")
    mods = dict(model.named_modules())
    prod = mods[site.producer]
    dev, dt = prod.weight.device, prod.weight.dtype
    V, lam = unmet_directions(prod, mom.patch_cov, k)                                # d x k
    if scale is None:
        scale = float(prod.weight.detach().reshape(prod.weight.shape[0], -1).norm(dim=1).median())
    new_rows = (V.T * scale).to(dt)                                                  # k x d
    C = prod.weight.shape[0]
    if isinstance(prod, nn.Conv2d):
        n = _new_conv_like(prod, prod.in_channels, C + k)
        n.weight[:C].copy_(prod.weight)
        n.weight[C:].copy_(new_rows.reshape(k, *prod.weight.shape[1:]).to(dev))
    else:
        n = nn.Linear(prod.in_features, C + k, bias=prod.bias is not None)
        n.weight[:C].copy_(prod.weight)
        n.weight[C:].copy_(new_rows.to(dev))
    if prod.bias is not None:
        n.bias[:C].copy_(prod.bias); n.bias[C:].zero_()
    set_module(model, site.producer, n.to(dev))
    if site.norm is not None:
        bn = mods[site.norm]
        if isinstance(bn, nn.BatchNorm2d):
            nb = nn.BatchNorm2d(C + k, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
            if bn.affine:
                nb.weight[:C].copy_(bn.weight); nb.weight[C:].fill_(1.0)
                nb.bias[:C].copy_(bn.bias); nb.bias[C:].zero_()
            if bn.track_running_stats:
                nb.running_mean[:C].copy_(bn.running_mean); nb.running_mean[C:].zero_()
                nb.running_var[:C].copy_(bn.running_var); nb.running_var[C:].fill_(1.0)
                nb.num_batches_tracked.copy_(bn.num_batches_tracked)
        elif isinstance(bn, nn.LayerNorm):
            nb = nn.LayerNorm(C + k, bn.eps, bn.elementwise_affine)
            if bn.elementwise_affine:
                nb.weight[:C].copy_(bn.weight); nb.weight[C:].fill_(1.0)
                nb.bias[:C].copy_(bn.bias); nb.bias[C:].zero_()
        else:
            raise TypeError(f"norm {site.norm}: {type(bn).__name__} not supported")
        set_module(model, site.norm, nb.to(dev))
    for con in site.consumers:
        m = mods[con.name]
        if con.kind == "conv":
            nm = _new_conv_like(m, C + k, m.out_channels)
            nm.weight[:, :C].copy_(m.weight); nm.weight[:, C:].zero_()
        else:
            nm = nn.Linear(C + k, m.out_features, bias=m.bias is not None)
            nm.weight[:, :C].copy_(m.weight); nm.weight[:, C:].zero_()
        if m.bias is not None:
            nm.bias.copy_(m.bias)
        set_module(model, con.name, nm.to(dev))
    return lam
