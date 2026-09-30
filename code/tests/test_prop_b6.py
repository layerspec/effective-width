"""Proposition B6 (theory-chain note, plan 9.25): for a convolution followed by training-mode batch normalisation,
<grad_W L0, W> equals sum_c eps/(sigma_c^2 + eps) <dL0/dy_c, y_c - beta_c> (zero at eps = 0), and the conv bias has zero gradient."""
import torch
import torch.nn as nn


def _layer(eps, seed=0):
    torch.manual_seed(seed)
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=True).double()
    bn = nn.BatchNorm2d(8, eps=eps).double()
    bn.train()
    with torch.no_grad():
        bn.weight.mul_(torch.rand(8, dtype=torch.float64) + 0.5)
        bn.bias.add_(torch.randn(8, dtype=torch.float64) * 0.3)
        conv.bias.add_(torch.randn(8, dtype=torch.float64))
    head = nn.Linear(8, 5).double()
    x = torch.randn(16, 3, 10, 10, dtype=torch.float64)
    labels = torch.randint(0, 5, (16,))
    u = conv(x)
    y = bn(u)
    loss = nn.functional.cross_entropy(head(torch.relu(y).mean((2, 3))), labels)
    g_w, g_b, g_y = torch.autograd.grad(loss, [conv.weight, conv.bias, y])
    sig2 = u.var((0, 2, 3), unbiased=False)
    formula = sum((eps / (sig2[c] + eps)) * (g_y[:, c] * (y[:, c] - bn.bias[c])).sum() for c in range(8))
    return (g_w * conv.weight).sum().item(), formula.item(), g_b.abs().max().item(), g_w.norm().item() * conv.weight.norm().item()


def test_inner_product_follows_the_epsilon_formula_and_the_bias_gradient_vanishes():
    for eps in (1e-12, 1e-5, 1e-2, 1.0):
        inner, formula, grad_bias, scale = _layer(eps)
        assert abs(inner - formula) <= 1e-9 * scale             # the correction term is exact
        assert grad_bias < 1e-12                                # the batch mean removes the bias exactly
    inner, _, _, scale = _layer(1e-12)
    assert abs(inner) < 1e-10 * scale                           # eps -> 0: exact scale invariance in W alone
    inner5, _, _, scale5 = _layer(1e-5)
    assert abs(inner5) / scale5 < 1e-3                          # eps = 1e-5 (PyTorch default): the cosine is below 1e-3
