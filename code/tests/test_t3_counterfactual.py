"""Tests for scripts/t3_counterfactual.py (plan 9.56; proofs T3.0-T3.2)."""
import math

import numpy as np

from scripts.t3_counterfactual import block_deconv, counterfactual_row, cross_block_share, isqrt_sym


def _spd(rng, d, cond=50.0):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.logspace(0, -np.log10(cond), d)
    return Q @ np.diag(lam) @ Q.T


def test_isqrt_and_block_deconv_whiten_blocks():
    rng = np.random.default_rng(0)
    C_in, k2 = 8, 9
    S = _spd(rng, C_in * k2)
    D = isqrt_sym(S)
    assert np.allclose(D @ S @ D, np.eye(C_in * k2), atol=1e-8)
    D_B = block_deconv(S, C_in, k2, block=4, eps=0.0)
    f = 4 * k2
    DSD = D_B @ S @ D_B
    for i in range(2):
        assert np.allclose(DSD[i * f:(i + 1) * f, i * f:(i + 1) * f], np.eye(f), atol=1e-8)
    assert 0 < cross_block_share(S, C_in, k2, 4) < 1
    assert cross_block_share(np.eye(C_in * k2), C_in, k2, 4) == 0.0


def test_counterfactual_row_ceiling_and_orderings():
    rng = np.random.default_rng(1)
    C_in, k2, C = 8, 9, 16
    d = C_in * k2
    S = _spd(rng, d, cond=1e3)
    W = rng.standard_normal((C, d)) * np.logspace(0, -1, C)[:, None]     # anisotropic kernel
    row = counterfactual_row("l", W, S, C_in, k2, 0, block=4, eps=1e-8, gain_cap=1e5)
    assert row["ceiling_95"] == math.ceil(0.95 * C) / C
    assert row["both_pred_95"] == row["ceiling_95"]                      # T3.1: W_o D Sigma D W_o^T = I
    for key in ("out", "ortho", "nd_pred", "full_pred"):
        assert row[f"{key}_95"] <= row["ceiling_95"] + 1e-12               # Lemma T3.0
    assert row["full_pred_95"] == row["kernel_95"]                          # T3.2 (ii): whitening leaves W W^T
    assert row["out_95"] <= row["nd_pred_95"] + 1e-12 or row["out_95"] <= row["ortho_95"] + 1e-12


def test_block_whitening_equals_full_when_blocks_are_independent():
    rng = np.random.default_rng(2)
    C_in, k2 = 8, 4
    f = 4 * k2
    A, B = _spd(rng, f), _spd(rng, f)
    S = np.block([[A, np.zeros((f, f))], [np.zeros((f, f)), B]])
    W = rng.standard_normal((6, C_in * k2))
    row = counterfactual_row("l", W, S, C_in, k2, 0, block=4, eps=0.0, gain_cap=1e6)
    assert row["cross_block_share"] == 0.0
    assert row["nd_pred_95"] == row["full_pred_95"]
