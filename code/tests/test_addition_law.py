import numpy as np

from scripts.addition_law import kstar, sampled_widths, sigma_y


def _joint(rng, C=60, Ch=20, dep=0.8):
    # x with a decaying spectrum; h = dep * (linear image of x) + noise, so the cross block is nonzero
    n = 4000
    x = rng.standard_normal((n, C)) * np.exp(-np.arange(C) / 10.0)
    A = rng.standard_normal((C, Ch)) / np.sqrt(C)
    h = dep * x @ A + (1 - dep) * rng.standard_normal((n, Ch))
    X = np.concatenate([x, h], 1)
    return np.cov(X.T), x, h


def test_rebuilt_equals_direct():
    rng = np.random.default_rng(0)
    J, x, h = _joint(rng)
    W = rng.standard_normal((60, 20)) * 0.3
    y = x + h @ W.T
    assert np.allclose(sigma_y(J, 60, W), np.cov(y.T), atol=1e-8)


def test_cross_block_changes_the_width():
    rng = np.random.default_rng(1)
    J, _, _ = _joint(rng, dep=0.95)
    k_c, _ = sampled_widths(J, 60, 0.3, 4, rng, cross=True)
    k_n, _ = sampled_widths(J, 60, 0.3, 4, rng, cross=False)
    assert k_c != k_n   # with a strongly dependent h the two laws differ


def test_kstar_monotone_in_tau():
    e = np.exp(-np.arange(30) / 5.0)
    assert kstar(e, 0.5) <= kstar(e, 0.9) <= kstar(e, 0.99)
