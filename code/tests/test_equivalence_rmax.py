"""Unit checks for scripts/equivalence_rmax.py (D3 effect sizes)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from equivalence_rmax import hl_shift, layer_bootstrap, superiority  # noqa: E402


def test_hl_shift_recovers_a_pure_shift():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.8, 1.0, 40)
    b = u[:20] - 0.1
    assert abs(hl_shift(b, u) - (-0.1)) < 0.02
    assert abs(hl_shift(u, u)) < 1e-12


def test_superiority_is_half_for_identical_and_zero_for_separated():
    u = np.linspace(0.8, 1.0, 30)
    assert abs(superiority(u, u) - 0.5) < 1e-12
    assert superiority(u - 1.0, u) == 0.0
    assert superiority(u + 1.0, u) == 1.0


def test_bootstrap_ci_covers_the_true_shift():
    rng = np.random.default_rng(1)
    u = rng.normal(0.95, 0.03, 33)
    b = rng.normal(0.93, 0.03, 20)
    sh, au = layer_bootstrap(b, u, 500, 0)
    lo, hi = np.percentile(sh, 2.5), np.percentile(sh, 97.5)
    assert lo < -0.02 < hi
    assert 0 <= au.min() and au.max() <= 1
