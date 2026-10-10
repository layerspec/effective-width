import numpy as np
import pandas as pd

from scripts.t1_kernel_reading import eig, kstar
from scripts.t2a_orientation import analyse_row, rotated, summarise


def test_rotation_keeps_spectrum():
    rng = np.random.default_rng(0)
    spec = np.linspace(0.1, 5, 30)
    M = rotated(rng, spec, 30)
    assert np.allclose(np.sort(eig(M)), np.sort(spec), atol=1e-8)


def test_peel_identity_is_exact():
    """R o R = lam1^2 (u1 o u1)(u1 o u1)^T + 2 lam1 D_u1 T D_u1 + T o T with T = R - lam1 u1 u1^T."""
    rng = np.random.default_rng(1)
    C = 20
    W = rng.standard_normal((C, 3 * C)); G = W @ W.T / (3 * C)
    d = 1 / np.sqrt(np.diag(G)); R = G * d[:, None] * d[None, :]
    w, V = np.linalg.eigh(R); i = np.argmax(w); u1 = V[:, i]; l1 = w[i]
    T = R - l1 * np.outer(u1, u1)
    lhs = R * R
    rhs = l1 ** 2 * np.outer(u1 * u1, u1 * u1) + 2 * l1 * (u1[:, None] * T) * u1[None, :] + T * T
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_analyse_row_and_summary():
    rng = np.random.default_rng(2)
    C = 24
    W = rng.standard_normal((C, 4 * C)); G = W @ W.T / (4 * C)
    m = 0.5 * rng.standard_normal(C)
    out = analyse_row(m, G, n_rot=2)
    assert 1 <= out["k_surr@0.95"] <= C and 1 <= out["k_rotRoR@0.95"] <= C and 1 <= out["k_peel@0.95"] <= C
    assert 0 <= out["cos_m_u1"] <= 1 + 1e-12
    rows = [dict(out, group="vgg", layer="a", C=C), dict(out, group="vgg", layer="b", C=512)]
    text = summarise(pd.DataFrame(rows))
    assert "rotated-true-spectrum" in text and "cos(m,u1)" in text
