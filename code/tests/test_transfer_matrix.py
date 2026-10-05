"""Tests for scripts/transfer_matrix.py (plan 9.31): the identity Sigma = U g^{1/2} M~ g^{1/2} U^T and the two limiting cases."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.transfer_matrix import transfer, k_star      # noqa: E402


def _gamma(C, rng, decay=0.05):
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    g = np.exp(-decay * np.arange(C))
    return Q @ np.diag(g) @ Q.T, Q, g


def test_scalar_transfer_gives_equal_widths_and_flat_diagonal():
    rng = np.random.default_rng(0)
    G, _, _ = _gamma(64, rng)
    r = transfer(G, 3.0 * G)
    assert r["k_M"] == r["H"] and r["k_sigma"] == r["k_gamma"]
    assert r["diag_cv"] < 1e-8 and abs(r["diag_slope"]) < 1e-8 and r["off_frac"] < 1e-12


def test_coherent_transfer_narrows_sigma_with_a_flat_diagonal():
    rng = np.random.default_rng(1)
    C = 64
    G, Q, g = _gamma(C, rng, decay=0.02)
    H = k_star(g)
    v = rng.standard_normal((H, 4)); v /= np.linalg.norm(v, axis=0)
    Mt = v @ v.T                                           # rank 4, diagonal not constant yet
    Mt = Mt / np.sqrt(np.outer(np.diag(Mt), np.diag(Mt)))  # flat diagonal, rank 4 (correlation matrix of rank 4)
    S = Q[:, :H] @ (np.sqrt(np.outer(g[:H], g[:H])) * Mt) @ Q[:, :H].T
    r = transfer(G, S)
    assert r["diag_cv"] < 1e-8
    assert r["k_M"] <= 4 and r["k_sigma"] <= 4 and r["k_diag"] >= 0.8 * r["H"]
    assert r["off_frac"] > 0.5


def test_diagonal_transfer_keeps_gamma_width_when_flat():
    rng = np.random.default_rng(2)
    G, Q, g = _gamma(48, rng)
    r = transfer(G, 0.5 * G + 1e-9 * np.eye(48))
    assert abs(r["k_diag"] - r["k_gamma"]) <= 1
