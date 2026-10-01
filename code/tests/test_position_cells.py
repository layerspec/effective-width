"""B9 part B (plan 9.29): the (class, position) cell decomposition on synthetic samples."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_position_cells import cell_decomposition  # noqa: E402


def _make(P, K, C, per_cell, within, rng, shared=False):
    mu = rng.standard_normal((1 if shared else P, K, C))
    X, cls, pos = [], [], []
    for p in range(P):
        for c in range(K):
            X.append(mu[0 if shared else p, c] + within * rng.standard_normal((per_cell, C)))
            cls += [c] * per_cell; pos += [p] * per_cell
    perm = rng.permutation(len(cls))
    return np.concatenate(X)[perm].astype(np.float32), np.array(cls)[perm], np.array(pos)[perm]


def test_position_dependent_class_means():
    rng = np.random.default_rng(0)
    X, cls, pos = _make(4, 10, 96, 200, 0.01, rng)
    r = cell_decomposition(X, cls, pos, chunk=1000)
    assert r["PK"] == 40 and r["w_cp"] < 1e-3
    assert r["w_c"] > 0.5                          # class-only cells miss the position-specific means
    assert r["kB_cp_0.999"] >= 36 and r["k0.999"] <= 39 and r["cliff_g"] == 39
    assert r["kB_c_0.999"] <= 9


def test_position_independent_class_means():
    rng = np.random.default_rng(1)
    X, cls, pos = _make(4, 10, 96, 200, 0.01, rng, shared=True)
    r = cell_decomposition(X, cls, pos)
    assert r["w_cp"] < 1e-3 and r["w_c"] < 2e-3    # the class alone explains everything
    assert r["k0.999"] <= 9 and r["w_p"] > 0.99      # (the cliff search starts at K by registration, so it is not asked here)


def test_fractions_agree_with_direct_computation():
    rng = np.random.default_rng(2)
    X, cls, pos = _make(3, 5, 20, 50, 0.7, rng)
    r = cell_decomposition(X, cls, pos, chunk=97)
    Xd = X.astype(np.float64); tot = ((Xd - Xd.mean(0)) ** 2).sum()
    within = sum(((Xd[(cls == c) & (pos == p)] - Xd[(cls == c) & (pos == p)].mean(0)) ** 2).sum() for c in range(5) for p in range(3))
    assert r["w_cp"] == pytest.approx(within / tot, rel=1e-6)
    assert 0 < r["w_cp"] < r["w_c"] <= 1 and r["w_cp"] < r["w_p"]
