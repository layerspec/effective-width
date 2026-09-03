"""Correctness checks for the parts that must not be wrong quietly.

The accumulator is the load-bearing piece: if it drifts, every k* is wrong in a
way no plot would reveal.  So it is checked against an exact one-shot
computation, in a regime chosen to punish a naive implementation (large mean
relative to the variance, many small blocks).
"""

import numpy as np

from layerspec.accumulate import CovarianceAccumulator, merge
from layerspec import metrics


def test_accumulator_matches_exact():
    rng = np.random.default_rng(0)
    C, n = 32, 20_000
    A = rng.normal(size=(C, C))
    cov_true = A @ A.T / C
    # Large offset: this is what breaks accumulators that sum uncentred outer
    # products, because the mean^2 term swamps the variance in float32.
    X = rng.multivariate_normal(np.full(C, 1e4), cov_true, size=n)

    acc = CovarianceAccumulator(C)
    for block in np.array_split(X, 400):
        acc.update(block)

    exact = np.cov(X, rowvar=False)
    assert acc.n == n
    assert np.allclose(acc.covariance, exact, rtol=1e-9, atol=1e-9), \
        np.abs(acc.covariance - exact).max()


def test_merge_is_associative_with_single_pass():
    rng = np.random.default_rng(1)
    C = 16
    X = rng.normal(size=(5000, C)) * rng.uniform(0.1, 5.0, size=C)

    whole = CovarianceAccumulator(C)
    whole.update(X)

    a, b = CovarianceAccumulator(C), CovarianceAccumulator(C)
    a.update(X[:1234])
    b.update(X[1234:])
    combined = merge(a, b)

    assert combined.n == whole.n
    assert np.allclose(combined.covariance, whole.covariance, rtol=1e-10, atol=1e-12)


def test_k_star_on_known_spectrum():
    # Exactly 10 units of variance in 10 equal components, then nothing.
    eig = np.array([1.0] * 10 + [0.0] * 90)
    assert metrics.k_star(eig, 0.95) == 10
    assert metrics.k_star(eig, 0.50) == 5
    assert metrics.k_star(eig, 0.10) == 1


def test_participation_ratio_bounds():
    # PR of k equal eigenvalues is exactly k; PR of a single spike is 1.
    for k in (1, 3, 17):
        eig = np.array([1.0] * k + [0.0] * 50)
        assert abs(metrics.participation_ratio(eig) - k) < 1e-9
    spike = np.array([1.0] + [0.0] * 99)
    assert abs(metrics.participation_ratio(spike) - 1.0) < 1e-12


def test_effective_rank_equals_k_for_flat_spectrum():
    for k in (1, 4, 64):
        eig = np.array([1.0] * k)
        assert abs(metrics.effective_rank(eig) - k) < 1e-9


def test_powerlaw_recovers_known_exponent():
    idx = np.arange(1, 2001, dtype=float)
    for alpha in (0.5, 1.0, 1.8):
        eig = idx ** (-alpha)
        est, lo, hi, r2 = metrics.powerlaw_alpha(eig, lo=10)
        assert abs(est - alpha) < 1e-6, (alpha, est)
        assert r2 > 0.999


def test_metrics_ratios_use_nominal_C():
    # The whole paper is about k*/C, so C must be the nominal channel count and
    # not the number of nonzero eigenvalues.
    eig = np.array([1.0] * 8 + [0.0] * 120)
    m = metrics.compute(eig, C=128, n_samples=10_000)
    assert m.k_star["0.95"] == 8
    assert abs(m.k_star_ratio["0.95"] - 8 / 128) < 1e-12
    assert m.C == 128


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
