"""Numerical check of Proposition Th2 (notes/proofs-2026-09-29.md): the finite-sample bias of the sum of the k
largest sample eigenvalues has a sign and a bound that need no spectral gap."""
import numpy as np
import pytest


def _spectra():
    C = 24
    return {"identity (no gap anywhere)": np.ones(C),
            "power law with repeated roots": np.repeat(np.arange(1, C // 2 + 1.0) ** -0.5, 2)}


def _draw(lam, n, reps, rng, centre=False):
    C = len(lam)
    S, T = np.empty((reps, C)), np.empty(reps)
    for r in range(reps):
        x = rng.standard_normal((n, C)) * np.sqrt(lam)
        if centre:
            x = x - x.mean(0)
            cov = x.T @ x / (n - 1)
        else:
            cov = x.T @ x / n                                   # known mean: unbiased
        e = np.sort(np.linalg.eigvalsh(cov))[::-1]
        S[r], T[r] = np.cumsum(e), e.sum()
    return S, T


@pytest.mark.parametrize("name", list(_spectra()))
@pytest.mark.parametrize("ratio", [2, 10, 50])
def test_direction_and_bound(name, ratio):
    lam = _spectra()[name]; C = len(lam); n = ratio * C
    rng = np.random.default_rng(ratio)
    S, T = _draw(lam, n, 1500, rng)
    true = np.cumsum(np.sort(lam)[::-1]); tr = lam.sum()
    se = S.std(0) / np.sqrt(len(S))
    bias = S.mean(0) - true
    assert np.all(bias >= -4 * se)                               # (1): E S_k >= S_k, every k
    assert abs(bias[-1]) <= 4 * se[-1] + 1e-12                   # k = C: the trace, unbiased
    pr = tr ** 2 / np.sum(lam ** 2)
    bound = np.sqrt(np.arange(1, C + 1) / n) * np.sqrt(1 + 1 / pr) * tr
    assert np.all(bias <= bound + 4 * se)                        # (2): the gap-free upper bound
    if name.startswith("identity") and ratio == 2:
        assert bias[C // 2 - 1] > 10 * se[C // 2 - 1]            # strict where Lawley's expansion is undefined
    f_hat = (S / T[:, None]).mean(0)
    slack = np.sqrt(2 / (n * pr))
    assert np.all(f_hat >= true / tr - slack - 4 * (S / T[:, None]).std(0) / np.sqrt(len(S)))   # (3)


@pytest.mark.parametrize("name", list(_spectra()))
def test_estimated_width_is_biased_downwards(name):
    lam = _spectra()[name]; C = len(lam)
    rng = np.random.default_rng(1)
    f_true = np.cumsum(np.sort(lam)[::-1]) / lam.sum()
    k_true = int(np.searchsorted(f_true, 0.95) + 1)
    means = []
    for ratio in (2, 10, 50):
        S, T = _draw(lam, ratio * C, 800, rng, centre=True)
        means.append(np.mean([np.searchsorted(s / t, 0.95) + 1 for s, t in zip(S, T)]))
    assert means[0] <= means[1] + 0.05 and means[1] <= means[2] + 0.05     # rises towards the true width with n
    assert means[0] < k_true and means[2] <= k_true + 0.05
