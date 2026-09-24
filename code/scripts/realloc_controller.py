"""The width controller's action, shared by the trainers (2026-09-24; first user: train_imagenet_r18.py).

One call = one action of the controller of notes/controller-criterion-2026-09-16.md on a model whose
Sites the package found:

  pass 1 (unless shrink_only)  measure with patch statistics, water-fill with the unmet-variance
                               eigenvalues appended (Proposition B) and GROW the sites the fill asks
                               for, deepest first (grow cap per site);
  pass 2, iterated             measure again (post-activation only), water-fill the budget with the
                               costs of the grown network and SHRINK by CSSP (Proposition A).  Iterated
                               because a channel's cost is bilinear in its neighbours' widths: one
                               linearised fill after growth lands over budget.  Stops within 0.5%.

The cost of a channel is either its parameters (Site.cost_per_channel) or its forward MACs at the
given input resolution (site_macs_per_channel; the FLOPs-conserving setting of P2g, the only one
in which accuracy comparisons are meaningful).  The caller rebuilds its optimiser afterwards.
trajectory_cifar.py / trajectory_vit.py still carry their own copy of this logic (they must not be
edited while the Mac queues import them); fold them onto this module when the queues are idle.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from layerspec.realloc import greedy_cssp, grow, measure, shrink, unmet_directions, water_fill


def module_hw(model, sample) -> dict:
    """Output (H, W) of every Conv2d / Linear for one forward pass of `sample` (Linear -> (1, 1))."""
    hw, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, n=name: hw.__setitem__(n, (int(o.shape[-2]), int(o.shape[-1])) if o.dim() == 4 else (1, 1))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return hw


def model_macs(model, hw) -> int:
    """Forward multiply-adds per image (convs and linears; norms, activations and pools ignored)."""
    total = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            H, W = hw[name]
            total += m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1] * H * W
        elif isinstance(m, torch.nn.Linear):
            total += m.in_features * m.out_features
    return int(total)


def site_macs_per_channel(site, model, hw) -> int:
    """Forward MACs per image that one channel of the site costs: its producer row at the producer's
    resolution plus each consumer's column at the consumer's resolution."""
    mods = dict(model.named_modules())
    p = mods[site.producer]
    if isinstance(p, torch.nn.Conv2d):
        H, W = hw[site.producer]
        c = (p.in_channels // p.groups) * p.kernel_size[0] * p.kernel_size[1] * H * W
    else:
        c = p.in_features
    for con in site.consumers:
        m = mods[con.name]
        if isinstance(m, torch.nn.Conv2d):
            H, W = hw[con.name]
            c += (m.out_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups) * H * W
        else:
            c += m.out_features
    return int(c)


def reallocate(model, sites, calib, device, *, budget_fraction: float, cost: str, full_params: int, full_macs: int,
               input_hw, grow_cap: float = 0.25, min_width: int = 8, grow_random: bool = False,
               shrink_only: bool = False, max_shrink_iters: int = 6) -> dict:
    """One action.  `calib`: loader of calibration images (plain transform).  Returns the record the
    trainers append to realloc.json (without 'epoch').  Leaves the model in train mode."""
    if cost not in ("params", "flops"):
        raise ValueError(cost)
    t0 = time.time()
    by_flops = cost == "flops"
    model.eval()
    hw = module_hw(model, torch.zeros(1, 3, *input_hw, device=device))
    cur0 = [s.width(model) for s in sites]
    n0 = sum(q.numel() for q in model.parameters())
    f0 = model_macs(model, hw)

    def current_total():
        return model_macs(model, hw) if by_flops else sum(q.numel() for q in model.parameters())
    budget_total = budget_fraction * (full_macs if by_flops else full_params)

    def fill(mom, allow_grow):
        spectra = [np.clip(np.linalg.eigvalsh(mom[s.name].act_cov.numpy())[::-1], 0, None) for s in sites]
        c = np.array([site_macs_per_channel(s, model, hw) if by_flops else s.cost_per_channel(model)
                      for s in sites], dtype=float)
        cur = np.array([s.width(model) for s in sites])
        fixed = current_total() - float(c @ cur)
        hi = cur.copy()
        if allow_grow:
            hi = np.maximum(cur, np.floor(cur * (1 + grow_cap)).astype(int))
            mods = dict(model.named_modules())
            for l, s in enumerate(sites):
                k = int(hi[l] - cur[l])
                if k > 0:
                    _, lam = unmet_directions(mods[s.producer], mom[s.name].patch_cov, k)
                    ext = (lam.numpy() / float(torch.trace(mom[s.name].patch_cov))).clip(0, None)
                    spectra[l] = np.concatenate([spectra[l] / spectra[l].sum(), ext])
        target, mu, obj = water_fill(spectra, c, budget_total - fixed,
                                     min_width=[min_width] * len(sites), max_width=hi.tolist())
        return cur, target, mu, obj

    grown = []
    if not shrink_only:
        mom = measure(model, sites, calib, device, patch=True, max_rows_per_batch=2048)
        cur, target, _, _ = fill(mom, True)
        for s, t, c in reversed(list(zip(sites, target, cur))):
            if t > c:
                grow(model, s, int(t - c), mom[s.name], directions="random" if grow_random else "unmet")
                grown.append((s.producer, int(c), int(t)))
    mu = obj = float("nan")
    for _it in range(max_shrink_iters):
        mom = measure(model, sites, calib, device, patch=False, max_rows_per_batch=8192)
        cur, target, mu, obj = fill(mom, False)
        if not any(t < c for t, c in zip(target, cur)):
            break
        for s, t, c in zip(sites, target, cur):
            if t < c:
                order, _ = greedy_cssp(mom[s.name].act_cov, int(t))
                shrink(model, s, order[:int(t)], mom[s.name])
        if current_total() <= budget_total * 1.005:
            break
    model.train()
    after = [s.width(model) for s in sites]
    n1 = sum(q.numel() for q in model.parameters())
    f1 = model_macs(model, hw)
    return {"shrink_only": shrink_only, "before": cur0, "grown": grown, "after": after,
            "params_before": int(n0), "params_after": int(n1), "budget_fraction": budget_fraction,
            "actual_fraction": n1 / full_params, "mu": float(mu), "objective": float(obj), "cost": cost,
            "macs_before": int(f0), "macs_after": int(f1), "macs_fraction": f1 / full_macs,
            "input_hw": list(input_hw), "seconds": round(time.time() - t0)}
