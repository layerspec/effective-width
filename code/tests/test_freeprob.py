"""scripts/freeprob_check.py (2026-09-26, hostile review 09-25 item 1): at initialisation the conv output
covariance is W Sigma_patch W^T with i.i.d. Gaussian W, so its spectrum is the free multiplicative convolution
of the patch spectrum with a Marchenko-Pastur law (Silverstein's equation).  The predicted k*(tau) must agree
with a Monte Carlo over W draws to within draw noise, and the layer-by-layer check must run on a model."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.freeprob_check import kstar_from_spectrum, mp_convolve_kstar, mc_kstar   # noqa: E402


def test_silverstein_prediction_matches_monte_carlo_on_a_power_law_spectrum():
    d, C = 600, 200
    lam = 1.0 / np.arange(1, d + 1) ** 1.2                          # a heavy-tailed patch spectrum
    for tau in (0.9, 0.95, 0.99):
        pred = mp_convolve_kstar(lam, C, tau)
        mc = mc_kstar(lam, C, tau, draws=8, seed=0)
        assert abs(pred - np.mean(mc)) <= max(2.0, 2.5 * np.std(mc) + 1), (tau, pred, mc)


def test_kstar_from_spectrum_counts_to_tau():
    assert kstar_from_spectrum(np.array([4.0, 3.0, 2.0, 1.0]), 0.5) == 2     # 4/10 < 0.5 <= 7/10
    assert kstar_from_spectrum(np.array([4.0, 3.0, 2.0, 1.0]), 0.7) == 2


def test_rank_deficient_input_gives_at_most_rank_directions():
    d, C = 300, 100
    lam = np.concatenate([np.ones(40), np.zeros(d - 40)])           # rank-40 input
    assert mp_convolve_kstar(lam, C, 0.999) <= 40
