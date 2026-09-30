"""Proposition B7.2' (proofs note): with Sigma_out = D M D^T and Gamma = D D^T (M PSD on the range of D^T), the tau-widths obey
k*_{tau m-/m+}(Gamma) <= k*_tau(Sigma_out) <= k*_{1-(1-tau) m-/m+}(Gamma); equality of widths when M is a multiple of the identity."""
import numpy as np


def k_star(eig, tau):
    e = np.clip(np.sort(eig)[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / e.sum(), tau) + 1)


def test_two_sided_width_bounds_and_the_scalar_case():
    rng = np.random.default_rng(0)
    C, N = 64, 300
    D = rng.standard_normal((C, N)) * np.linspace(2.0, 0.05, C)[:, None]        # output-side gradients, anisotropic
    for cond in (1.0, 3.0, 30.0):
        A = rng.standard_normal((N, N)); Q, _ = np.linalg.qr(A)
        M = Q @ np.diag(np.linspace(1.0, cond, N)) @ Q.T                          # PSD with condition number `cond`
        Gam = D @ D.T; Sig = D @ M @ D.T
        eg, es = np.linalg.eigvalsh(Gam), np.linalg.eigvalsh(Sig)
        m_minus, m_plus = 1.0, cond
        for tau in (0.9, 0.95, 0.999):
            lo = k_star(eg, tau * m_minus / m_plus); hi = k_star(eg, 1 - (1 - tau) * m_minus / m_plus); mid = k_star(es, tau)
            assert lo <= mid <= hi, (cond, tau, lo, mid, hi)
            if cond == 1.0:
                assert lo == mid == hi                                            # M = I: the two widths coincide
