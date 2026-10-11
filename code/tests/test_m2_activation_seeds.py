"""Tests for scripts/m2_activation_seeds.py (plan 9.57; Theorem M.2 / Lemma M.2.0 on the activation step, patch route)."""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from scripts.m2_activation_seeds import (TAUS, gather_patches, judge, rk_and_grad, sample_positions, tail_statistics)


def test_gather_patches_matches_unfold():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 7, 9)
    k, pad = 3, 1
    gen = torch.Generator().manual_seed(1)
    pos = sample_positions(3, 7, 9, 5, gen)
    P = gather_patches(x, k, pad, pos)                                     # (3*5, 4*9)
    U = Fn.unfold(x, k, padding=pad)                                       # (3, 36, 63)
    ref = torch.stack([U[i, :, pos[i]] for i in range(3)], 0).transpose(1, 2).reshape(-1, 36)
    assert P.shape == (15, 36)
    assert torch.allclose(P, ref)


def test_patches_reproduce_conv_output():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1, bias=False)
    x = torch.randn(2, 4, 8, 8)
    gen = torch.Generator().manual_seed(2)
    pos = sample_positions(2, 8, 8, 7, gen)
    P = gather_patches(x, 3, 1, pos)                                       # (14, 36)
    Wm = conv.weight.detach().reshape(6, -1)
    z = (Wm @ P.T)                                                         # (6, 14)
    y = conv(x).detach()                                                   # (2, 6, 8, 8)
    ref = torch.cat([y[i].reshape(6, -1)[:, pos[i]] for i in range(2)], 1)
    assert torch.allclose(z, ref, atol=1e-5)


def _finite_difference(G, X, s, k, relu, eps=1e-6):
    g = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            Gp, Gm = G.copy(), G.copy()
            Gp[i, j] += eps
            Gm[i, j] -= eps
            g[i, j] = (rk_and_grad(Gp, X, s, k, relu)[0] - rk_and_grad(Gm, X, s, k, relu)[0]) / (2 * eps)
    return g


def test_lemma_m20_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    d, C, n, s = 6, 4, 40, 0.7
    X = rng.standard_normal((d, n)) + 0.3
    G = rng.standard_normal((C, d))
    for relu in (True, False):
        for k in (1, 2):
            Rk, grad = rk_and_grad(G, X, s, k, relu)
            fd = _finite_difference(G, X, s, k, relu)
            assert Rk > 0
            assert np.allclose(grad, fd, rtol=1e-4, atol=1e-6), (relu, k, np.abs(grad - fd).max())


def test_lemma_m20_poincare_quantity_bounded():
    rng = np.random.default_rng(1)
    d, C, n, s = 8, 6, 60, 1.3
    X = rng.standard_normal((d, n)) * np.linspace(2, 0.2, d)[:, None] + 0.5
    Mp = X @ X.T / n
    lam1 = np.linalg.eigvalsh(Mp).max()
    for _ in range(5):
        G = rng.standard_normal((C, d))
        for k in range(0, min(C, d)):
            Rk, grad = rk_and_grad(G, X, s, k, True)
            assert np.sum(grad ** 2) <= 4 * s ** 2 * lam1 * Rk * (1 + 1e-9) + 1e-12


def test_tail_statistics_consistent_with_lemma():
    torch.manual_seed(0)
    d, C, n, s = 10, 6, 200, 0.5
    X = torch.randn(d, n, dtype=torch.float64) * torch.linspace(2, 0.3, d)[:, None].double() + 0.4
    G = torch.randn(C, d, dtype=torch.float64)
    Z = s * (G @ X)
    for relu in (True, False):
        Y = torch.relu(Z) if relu else Z
        S = (Z > 0).to(Y.dtype) if relu else None
        F = (Y - Y.mean(1, keepdim=True)) / math.sqrt(n)
        st = tail_statistics(F, S, X, s)
        r = min(C, d)
        assert st["lam"].shape == (r,) and st["R"].shape == (r + 1,) and st["G2"].shape == (r + 1,)
        assert np.isclose(st["R"][0], st["T2"]) and np.isclose(st["R"][0], st["lam"].sum())
        assert np.all(np.diff(st["R"]) <= 1e-9) and st["R"][-1] < 1e-9 and st["G2"][-1] < 1e-9
        for tau in TAUS:
            assert 1 <= st["kstar"][tau] <= r
        Gn, Xn = G.numpy(), X.numpy()
        for k in (0, 1, 3):
            Rk, grad = rk_and_grad(Gn, Xn, s, k, relu)
            assert np.isclose(st["R"][k], Rk, rtol=1e-6)
            assert np.isclose(st["G2"][k], np.sum(grad ** 2), rtol=1e-6)      # G2_k = ||grad R_k||^2 (M.2.0 iii)


def _rows(cov, hw, sd, deg=0):
    rows = []
    for L in range(13):
        for step in ("z", "a"):
            for tau in TAUS:
                rows.append(dict(layer=f"features.{L}", step=step, tau=tau, C=64, d=576, r_max=64, n=1000,
                                 cov_holdout=cov, n_holdout=20, halfwidth_over_rmax=hw, degenerate=deg,
                                 sd_over_rmax=sd))
    return pd.DataFrame(rows)


def test_judge_holds_and_fails():
    v, _ = judge(_rows(1.0, 0.05, 0.01))
    assert v["P9.132"] and v["P9.133"] and v["P9.134"]
    v, _ = judge(_rows(0.7, 0.05, 0.01))
    assert not v["P9.132"]
    v, _ = judge(_rows(1.0, 0.15, 0.01))
    assert not v["P9.133"]
    v, text = judge(_rows(1.0, 0.05, 0.08))
    assert not v["P9.134"] and "P9.134" in text
