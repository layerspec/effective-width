"""Numerical check of Propositions Th5 and Th5' (notes/proofs-2026-09-29.md): the width a layer has at
initialisation concentrates; its tail sum obeys a Poincare variance bound; k* lies in the stated interval."""
import numpy as np
import pytest

C, D, ALPHA, TAU = 64, 128, 1.2, 0.95
LAM = np.arange(1, D + 1.0) ** -ALPHA


def _sample(kind, reps, rng, k):
    rows = []
    for _ in range(reps):
        G = rng.standard_normal((C, D)) if kind == "gauss" else rng.uniform(-np.sqrt(3), np.sqrt(3), (C, D))
        _, s, vt = np.linalg.svd(G * np.sqrt(LAM), full_matrices=False)
        e = s ** 2; c = np.cumsum(e)
        grad = 4 * np.sum(e[k:] * np.einsum("ij,j,ij->i", vt[k:], LAM, vt[k:]))
        rows.append((np.sqrt(c[k - 1]), c[-1] - c[k - 1], c[-1], grad, int(np.searchsorted(c / c[-1], TAU) + 1),
                     (c[-1] - c) / c[-1]))
    N, R, T2, G2, K = (np.array([r[i] for r in rows]) for i in range(5))
    return N, R, T2, G2, K, np.array([r[5] for r in rows])


@pytest.mark.parametrize("kind,c_p", [("gauss", 1.0), ("uniform", 12 / np.pi ** 2)])
def test_concentration(kind, c_p):
    rng = np.random.default_rng(0)
    k = 40
    N, R, T2, G2, K, H = _sample(kind, 600, rng, k)
    L = np.sqrt(LAM[0])
    assert N.std() <= L                                            # Th5: N_k is L-Lipschitz (Poincare: Var <= c_P L^2)
    assert np.mean(np.abs(N - N.mean()) >= 2.5 * L) <= 2 * np.exp(-2.5 ** 2 / 2) + 0.01
    assert R.var() <= 1.15 * c_p * G2.mean()                       # Th5'(1), 15% for Monte Carlo error
    assert G2.mean() <= 4 * LAM[0] * R.mean()
    if kind == "gauss":
        assert abs(T2.mean() / (C * LAM.sum()) - 1) < 0.02         # Th5'(2)
        assert abs(T2.var() / (2 * C * np.sum(LAM ** 2)) - 1) < 0.2
        x = 3.0
        dev = 2 * np.sqrt(C * np.sum(LAM ** 2) * x) + 2 * LAM[0] * x
        assert np.mean(np.abs(T2 - C * LAM.sum()) >= dev) <= 2 * np.exp(-x)


def test_width_interval():
    rng = np.random.default_rng(1)
    N, R, T2, G2, K, H = _sample("gauss", 600, rng, 40)
    u = 4.0
    hbar, r, q = H.mean(0), H.std(0) * 0 + 0, T2.std() / T2.mean()
    # r_k for every k from the same draws (the proposition takes it as given; here it is estimated)
    Rk = H * T2[:, None]
    r = Rk.std(0) / np.maximum(Rk.mean(0), 1e-12)
    hbar = Rk.mean(0) / T2.mean()
    upper, lower = hbar * (1 + u * r) / (1 - u * q), hbar * (1 - u * r) / (1 + u * q)
    k_plus = int(np.argmax(upper <= 1 - TAU) + 1)
    k_minus = int(np.argmax(lower <= 1 - TAU) + 1)
    assert k_minus <= K.min() and K.max() <= k_plus              # 1 - 4/u^2 = 0.75 is the guarantee; all 600 draws fall inside
    assert k_plus - k_minus <= 0.5 * C                           # and the interval is not vacuous
    assert K.std() < 3
