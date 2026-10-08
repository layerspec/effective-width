import numpy as np
import pytest

from scripts.free_superposition_scale import exact_bracket, free_width, haar, k_star, tails


def test_haar_is_orthogonal():
    o = haar(40, np.random.default_rng(0))
    assert np.allclose(o @ o.T, np.eye(40), atol=1e-10)


def test_tails_and_kstar_agree():
    e = np.array([5.0, 3.0, 1.0, 1.0])
    t = tails(e)
    assert np.allclose(t, [10, 5, 2, 1, 0])
    assert k_star(e, 0.5) == 1 and k_star(e, 0.8) == 2 and k_star(e, 0.95) == 4


def test_free_width_inside_exact_bracket():
    rng = np.random.default_rng(1)
    c = 120
    lx = np.sort(rng.pareto(1.2, c) + 0.01)[::-1]
    lf = np.sort(np.exp(-np.arange(c) / 15.0))[::-1] * 2.0
    lo, up = exact_bracket(lx, lf, 0.95)
    assert 1 <= lo <= up <= 2 * c
    k, sd = free_width(lx, lf, 0.95, 4, rng)[0.95]
    assert lo - 1 <= k <= up + 1
    # the bounds are attained by construction when one summand is zero
    assert exact_bracket(lx, np.zeros(c), 0.95) == (k_star(lx, 0.95), k_star(lx, 0.95))


def test_bracket_matches_aligned_and_orthogonal_extremes():
    # orthogonal supports: sum of widths is exact and equals the upper bound
    lx = np.array([4.0, 4.0, 0.0, 0.0]); lf = np.array([0.0, 0.0, 4.0, 4.0])
    lo, up = exact_bracket(lx, lf, 0.95)
    assert up == 4 and lo <= 4
    m = np.diag(lx) + np.diag(lf)
    assert k_star(np.linalg.eigvalsh(m), 0.95) == 4
