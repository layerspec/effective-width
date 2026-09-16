"""One test per guarantee of layerspec.realloc (notes/controller-criterion-2026-09-16.md, section 9)."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.realloc import (discover_sites, greedy_cssp, grow, measure, reconstruction,  # noqa: E402
                               shrink, water_fill)
from scripts.trajectory_cifar import CifarResNet, build_vgg16_bn_cifar_widths           # noqa: E402

torch.manual_seed(0)


class TinyMlp(nn.Module):
    def __init__(self, d=16, h=40):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


def _loader(shape, n=6, bs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(bs, *shape, generator=g), torch.zeros(bs, dtype=torch.long)) for _ in range(n)]


def _models():
    vgg = build_vgg16_bn_cifar_widths([12] * 13, num_classes=10).eval()
    r20 = CifarResNet(1, 10).eval()
    mlp = TinyMlp().eval()
    # non-trivial BN statistics so that folding into running_mean is exercised
    for m in (vgg, r20):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.normal_(0, 0.5); mod.running_var.uniform_(0.5, 2.0)
                mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.3)
    return [(vgg, (3, 32, 32)), (r20, (3, 32, 32)), (mlp, (7, 16))]


def test_discover_sites():
    vgg, r20, mlp = (m for m, _ in _models())
    assert len(discover_sites(vgg)) == 13
    assert len(discover_sites(r20)) == 3 and all(s.act[1] for s in discover_sites(r20))
    assert [s.producer for s in discover_sites(mlp)] == ["fc1"]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_grow_preserves_function(idx):
    model, shape = _models()[idx]
    sites = discover_sites(model)
    loader = _loader(shape)
    mom = measure(model, sites, loader, "cpu", patch=True)
    x = loader[0][0]
    with torch.no_grad():
        before = model(x).clone()
    for s in reversed(sites[: min(3, len(sites))]):     # deepest first: growth changes the next site's input
        w0 = s.width(model)
        lam = grow(model, s, 3, mom[s.name])
        assert s.width(model) == w0 + 3 and lam.shape == (3,)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_shrink_dead_channel_exact(idx):
    model, shape = _models()[idx]
    s = discover_sites(model)[0]
    mods = dict(model.named_modules())
    prod = mods[s.producer]
    with torch.no_grad():                      # make channel 2 dead (post-activation identically 0):
        prod.weight[2].zero_()                 # a NONZERO constant would be exact only away from the
        if prod.bias is not None:              # zero-padding border of a conv consumer (see ops.shrink)
            prod.bias[2] = 0.0
        if s.norm is not None:
            bn = mods[s.norm]
            bn.weight[2] = 0.0; bn.bias[2] = -0.9
    loader = _loader(shape)
    mom = measure(model, s and [s], loader, "cpu", patch=False)
    x = loader[1][0]
    with torch.no_grad():
        before = model(x).clone()
    keep = [i for i in range(s.width(model)) if i != 2]
    shrink(model, s, keep, mom[s.name])
    with torch.no_grad():
        after = model(x)
    assert s.width(model) == len(keep)
    assert torch.allclose(before, after, atol=1e-5), (before - after).abs().max()


def test_shrink_fold_equals_hook_mlp():
    model, shape = _models()[2]
    s = discover_sites(model)[0]
    loader = _loader(shape)
    mom = measure(model, [s], loader, "cpu", patch=False)
    order, traces = greedy_cssp(mom[s.name].act_cov)
    m = 24
    keep = sorted(order[:m])
    keep_t, drop_t, B, c = reconstruction(mom[s.name], keep)
    x = loader[2][0]

    def hook(mod, inp, out):
        o = out.clone()
        o[..., drop_t] = c.float() + out[..., keep_t] @ B.float()
        return o
    h = dict(model.named_modules())[s.act[0]].register_forward_hook(hook)
    with torch.no_grad():
        hooked = model(x).clone()
    h.remove()
    shrink(model, s, keep, mom[s.name])
    with torch.no_grad():
        folded = model(x)
    assert s.width(model) == m
    assert torch.allclose(hooked, folded, atol=1e-5), (hooked - folded).abs().max()


def test_cssp_bounds():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 60))
    cov = torch.tensor(A @ A.T / 60 + 0.01 * np.eye(30))
    order, traces = greedy_cssp(cov)
    eig = np.sort(np.linalg.eigvalsh(cov.numpy()))[::-1]
    for m in (5, 10, 20):
        tail = eig[m:].sum()
        assert tail - 1e-9 <= traces[m] <= (m + 1) * tail + 1e-9
    assert len(set(order)) == 30


def test_water_fill_kkt_and_budget():
    rng = np.random.default_rng(2)
    spectra = [np.sort(rng.exponential(size=n))[::-1] * s for n, s in ((64, 1.0), (128, 3.0), (256, 0.5))]
    cost = np.array([10.0, 20.0, 40.0])
    budget = 0.5 * float(cost @ [64, 128, 256])
    widths, mu, obj = water_fill(spectra, cost, budget, discrete=False)
    assert float(cost @ widths) <= budget + 1e-9
    norm = [s / s.sum() for s in spectra]
    for l in range(3):                              # KKT: kept channels above the threshold, dropped below
        if widths[l] > 0:
            assert norm[l][widths[l] - 1] / cost[l] >= mu - 1e-12
        if widths[l] < len(norm[l]):
            assert norm[l][widths[l]] / cost[l] < mu + 1e-12
    wd, _, objd = water_fill(spectra, cost, budget, discrete=True)
    assert float(cost @ wd) <= budget + 1e-9 and objd <= obj + 1e-12
