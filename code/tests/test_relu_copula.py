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
