import numpy as np

from layerspec.accumulate import CovarianceAccumulator
from scripts.bootstrap_images import draw_counts, merge_blocks, merge_diag, spearman


def _blocks(X, B):
    parts = np.array_split(X, B)
    n = np.array([p.shape[0] for p in parts])
    mean = np.stack([p.mean(0) for p in parts])
    M2 = np.stack([(p - p.mean(0)).T @ (p - p.mean(0)) for p in parts])
    return n, mean, M2


def test_all_blocks_once_equals_streaming_covariance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 7)) @ rng.standard_normal((7, 7)) + 3.0
    n, mean, M2 = _blocks(X, 8)
    acc = CovarianceAccumulator(7)
    for chunk in np.array_split(X, 13):          # different chunking on purpose
        acc.update(chunk)
    ntot, cov = merge_blocks(np.ones(8, dtype=int), n, mean, M2.astype(np.float32))
    assert ntot == 1000
    assert np.allclose(cov, acc.covariance, rtol=1e-5, atol=1e-6)


def test_multiplicities_equal_duplicated_blocks():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 5)) * np.array([1, 2, 3, 4, 5]) + rng.standard_normal(5)
    n, mean, M2 = _blocks(X, 6)
    counts = np.array([2, 0, 1, 3, 0, 0])
    parts = np.array_split(X, 6)
    Xdup = np.concatenate([p for c, p in zip(counts, parts) for _ in range(c)])
    ntot, cov = merge_blocks(counts, n, mean, M2)
    assert ntot == Xdup.shape[0]
    assert np.allclose(cov, np.cov(Xdup, rowvar=False), rtol=1e-9)
    _, var = merge_diag(counts, n, mean, np.stack([np.diag(m) for m in M2]))
    assert np.allclose(var, np.diag(cov))


def test_draws_and_spearman():
    c = draw_counts(10, 50, seed=3)
    assert c.shape == (51, 10) and (c[0] == 1).all() and (c.sum(1) == 10).all()
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
