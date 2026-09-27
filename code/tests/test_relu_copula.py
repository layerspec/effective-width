"""scripts/relu_copula_check.py (2026-09-27, feasibility note section C, route 1): the ReLU second-moment map
with a Gaussian copula and the EMPIRICAL marginals of the pre-activation -- does the level error of the Gaussian
recursion come from the marginals or from the joint structure?"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.relu_copula_check import copula_relu_cov, marginal_tables   # noqa: E402
from scripts.apriori_relu_check import gaussian_relu_cov                 # noqa: E402


def _random_cov(C, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((C, C + 3)); S = A @ A.T / (C + 3)
    return S


def test_copula_with_gaussian_marginals_reduces_to_the_gaussian_formula():
    C, rng = 5, np.random.default_rng(0)
    S = _random_cov(C, 1); m = rng.standard_normal(C) * 0.5
    z = rng.multivariate_normal(m, S, size=200_000)                      # Gaussian samples: empirical marginals ~ Gaussian
    tab = marginal_tables(z)
    pred = copula_relu_cov(np.corrcoef(z.T), tab)
    ref = gaussian_relu_cov(torch.as_tensor(m), torch.as_tensor(S))[0].numpy()
    assert np.abs(pred - ref).max() < 0.02 * np.abs(ref).max()


def test_copula_matches_monte_carlo_for_a_uniform_marginal():
    C, rng = 4, np.random.default_rng(2)
    S = _random_cov(C, 3); s = np.sqrt(np.diag(S)); R = S / np.outer(s, s)
    g = rng.multivariate_normal(np.zeros(C), R, size=300_000)
    from scipy.stats import norm
    u = norm.cdf(g)
    z = np.column_stack([np.sqrt(12) * (u[:, 0] - 0.5) - 0.3, norm.ppf(u[:, 1]), 2 * (u[:, 2] ** 2) - 0.5, norm.ppf(u[:, 3]) * 0.7 + 0.4])
    a = np.clip(z, 0, None)
    mc = np.cov(a.T)
    tab = marginal_tables(z)
    pred = copula_relu_cov(R, tab)
    assert np.abs(pred - mc).max() < 0.03 * np.abs(mc).max()


def test_channel_kurtosis_of_random_projections_equals_three_times_energy_cv2():
    from scripts.relu_copula_check import predicted_kurtosis
    rng = np.random.default_rng(5)
    d, n = 40, 200_000
    x = rng.standard_normal((n, d)) * (1 + 0.8 * rng.random(n))[:, None]   # a Gaussian scale mixture: heavy-tailed patches
    e = (x ** 2).sum(1)
    lam = np.linalg.eigvalsh(np.cov(x.T)); pr = lam.sum() ** 2 / (lam ** 2).sum()
    pred = predicted_kurtosis(e.mean(), (e ** 2).mean(), pr)
    W = rng.standard_normal((300, d))
    z = x @ W.T
    zc = z - z.mean(0); k = (zc ** 4).mean(0) / (zc ** 2).mean(0) ** 2 - 3
    assert abs(pred - k.mean()) < 0.15 * max(1.0, abs(pred))


def test_kurtosis_matched_marginal_tables_reproduce_a_student_t_marginal():
    from scripts.relu_copula_check import matched_tables
    from scipy.stats import t as student
    rng = np.random.default_rng(7)
    z = student.rvs(df=6, size=300_000, random_state=rng) * 1.3 + 0.4          # excess kurtosis 3, mean 0.4, sd 1.3*sqrt(1.5)
    z = z[:, None]
    tab_emp = marginal_tables(z)
    zc = z - z.mean(0); kurt = (zc ** 4).mean(0) / (zc ** 2).mean(0) ** 2 - 3
    tab_fit = matched_tables(z.mean(0), z.std(0), kurt)
    inner = np.abs(GRID_ABS := np.linspace(-5, 5, 1001)) < 3
    assert np.abs(tab_fit[0][inner] - tab_emp[0][inner]).max() < 0.15


def test_box_filtered_patch_energy_equals_unfold():
    import torch.nn as nn
    from scripts.relu_copula_check import _patch_energy
    torch.manual_seed(0)
    x = torch.randn(2, 5, 12, 12); conv = nn.Conv2d(5, 7, 3, padding=1)
    cols = torch.nn.functional.unfold(x, 3, padding=1)                     # (2, 45, 144)
    ref = (cols ** 2).sum(1).reshape(2, 1, 12, 12)
    got = _patch_energy(x, conv, max_pos=10_000).reshape(2, 1, 12, 12)
    assert torch.allclose(got, ref, atol=1e-4)


def test_cornish_fisher_tables_are_monotone_and_reduce_to_gaussian():
    from scripts.relu_copula_check import cornish_fisher_tables
    g = cornish_fisher_tables([0.2], [1.5], [0.0], [0.0])
    ref = marginal_tables((np.random.default_rng(1).standard_normal((400_000, 1)) * 1.5 + 0.2))
    assert np.all(np.diff(g[0]) >= 0) and np.abs(g[0] - ref[0])[np.abs(GRID_ABS := np.linspace(-5, 5, 1001)) < 3].max() < 0.05
    sk = cornish_fisher_tables([0.0], [1.0], [1.0], [1.5])
    assert np.all(np.diff(sk[0]) >= 0)
