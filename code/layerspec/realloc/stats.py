"""Forward statistics a site needs: the post-activation mean and covariance (for shrinking) and
the covariance of the producer's input patch (for growing).  Label-free; a few forward passes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sites import Site


class _Acc:
    """Streaming mean / covariance in float64 on the CPU, rows = observations."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def update(self, x: torch.Tensor):
        x = x.detach().to("cpu").double()
        self.n += x.shape[0]
        self.s = x.sum(0) if self.s is None else self.s + x.sum(0)
        self.ss = x.T @ x if self.ss is None else self.ss + x.T @ x

    def finish(self):
        mean = self.s / self.n
        cov = self.ss / self.n - torch.outer(mean, mean)
        return mean, 0.5 * (cov + cov.T)


@dataclass
class Moments:
    n: int
    act_mean: torch.Tensor            # (C,)
    act_cov: torch.Tensor             # (C, C)   post-activation channel covariance
    patch_cov: torch.Tensor | None    # (d, d)   covariance of the producer's input patch (im2col column)
    patch_mean: torch.Tensor | None


def _rows_from_act(out: torch.Tensor, apply_relu: bool, positions: int | None, gen) -> torch.Tensor:
    if apply_relu:
        out = F.relu(out)
    if out.dim() == 4:                                  # N, C, H, W  -> rows of C
        x = out.permute(0, 2, 3, 1).reshape(-1, out.shape[1])
    else:                                               # N, T, H (tokens) or N, H
        x = out.reshape(-1, out.shape[-1])
    if positions is not None and positions < x.shape[0]:
        idx = torch.randperm(x.shape[0], generator=gen)[:positions]
        x = x[idx.to(x.device)]
    return x


def _patch_rows(module: nn.Module, inp: torch.Tensor, positions: int | None, gen) -> torch.Tensor:
    if isinstance(module, nn.Conv2d):
        cols = F.unfold(inp, module.kernel_size, dilation=module.dilation, padding=module.padding,
                        stride=module.stride)                       # N, d, L
        x = cols.permute(0, 2, 1).reshape(-1, cols.shape[1])
    else:
        x = inp.reshape(-1, inp.shape[-1])
    if positions is not None and positions < x.shape[0]:
        idx = torch.randperm(x.shape[0], generator=gen)[:positions]
        x = x[idx.to(x.device)]
    return x


@torch.no_grad()
def measure(model: nn.Module, sites: list[Site], loader, device, *, patch: bool = True,
            max_rows_per_batch: int | None = 65536, seed: int = 0) -> dict[str, Moments]:
    """Accumulate each site's statistics over the loader (inputs only; labels ignored)."""
    mods = dict(model.named_modules())
    gen = torch.Generator().manual_seed(seed)
    acc_a = {s.name: _Acc() for s in sites}
    acc_p = {s.name: _Acc() for s in sites} if patch else {}
    handles = []
    for s in sites:
        hook_mod, apply_relu = s.act
        handles.append(mods[hook_mod].register_forward_hook(
            lambda m, i, o, s=s, r=apply_relu: acc_a[s.name].update(_rows_from_act(o, r, max_rows_per_batch, gen))))
        if patch:
            handles.append(mods[s.producer].register_forward_hook(
                lambda m, i, o, s=s: acc_p[s.name].update(_patch_rows(m, i[0], max_rows_per_batch, gen))))
    was_training = model.training
    model.eval()
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(x.to(device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    out = {}
    for s in sites:
        am, ac = acc_a[s.name].finish()
        pm, pc = acc_p[s.name].finish() if patch else (None, None)
        out[s.name] = Moments(acc_a[s.name].n, am, ac, pc, pm)
    return out
