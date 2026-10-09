import numpy as np
import pandas as pd

from scripts.t1_kernel_reading import eig, kstar, surrogate
from scripts.t2_bounds import TAUS, analyse_row, coefficients, judge, kappa, lower_bound, generic_prediction, lower_bound_per_term, product_energies, upper_bound


def _psd(rng, C, spike=0.0):
    X = rng.standard_normal((C, int(rng.integers(2, C + 1))))
    G = X @ X.T / C
    if spike:
        u = rng.standard_normal(C); G += spike * np.outer(u, u) / C
    return G


def _Sk(M):
    return np.cumsum(np.clip(np.sort(np.linalg.eigvalsh(M))[::-1], 0, None))


def test_lemma_t2_1_hadamard_ky_fan():
    rng = np.random.default_rng(0)
    for _ in range(300):
        C = int(rng.integers(3, 30))
        A = _psd(rng, C); d = np.sqrt(np.diag(A)); A = A / np.outer(d, d) * rng.uniform(0.2, 1.0)   # diag <= 1
        B = _psd(rng, C, spike=rng.uniform(0, 20))
        assert (_Sk(A * B) - _Sk(B) <= 1e-9).all()


def test_theorem_t2_2_zero_mean_and_kappa():
    rng = np.random.default_rng(1)
    for _ in range(150):
        C = int(rng.integers(3, 40))
        G = _psd(rng, C, spike=rng.uniform(0, 30)) * rng.uniform(0.1, 5)
        R, s, mu0 = coefficients(np.zeros(C), G)
        assert abs(kappa(mu0, s, float(np.trace(G))) - 1.0) < 1e-6
        m = rng.standard_normal(C) * s * rng.uniform(0, 2)
        for mean in (np.zeros(C), m):
            full, _, _, _ = surrogate(mean, G)
            R, s, mu = coefficients(mean, G)
            for tau in (0.5, 0.9, 0.95, 0.999):
                klo, _ = lower_bound(G, mu, s, tau)
                klo2 = lower_bound_per_term(G, R, mu, s, tau)
                assert kstar(eig(full), tau) >= klo2 >= klo


def test_prop_t2_3_upper_bound_is_valid_and_improves_with_more_vectors():
    rng = np.random.default_rng(2)
    for _ in range(60):
        C = int(rng.integers(4, 30))
        G = _psd(rng, C, spike=rng.uniform(0, 30))
        mean = rng.standard_normal(C) * np.sqrt(np.diag(G)) * rng.choice([0.0, 1.0])
        full, lin, _, _ = surrogate(mean, G)
        R, s, mu = coefficients(mean, G)
        total = float((mu[:, 1:] ** 2).sum())
        assert abs(total - np.trace(full)) < 1e-8 * max(1, total)
        e_small = product_energies(R, mu, topk={2: 2})
        e_big = product_energies(R, mu, topk={2: C, 3: min(C, 8), 4: min(C, 5)})
        for tau in (0.9, 0.95, 0.999):
            ks = kstar(eig(full), tau)
            hi_s, _, _ = upper_bound(lin, e_small, total, tau)
            hi_b, _, _ = upper_bound(lin, e_big, total, tau)
            assert ks <= hi_b <= C
            assert hi_b <= hi_s


def test_product_energy_multiplicity_sums_to_hadamard_power_trace():
    # sum over ALL multisets (with multiplicity) of prod lambda * |mu_j o u...|^2 equals tr((mu_j mu_j^T) o R^{oj}) = |mu_j|^2
    rng = np.random.default_rng(3)
    C = 6
    G = _psd(rng, C, spike=3.0)
    R, s, mu = coefficients(np.zeros(C), G)
    e2 = product_energies(R, mu, topk={2: C})
    assert np.isclose(sum(v for _, v in e2), (mu[:, 2] ** 2).sum(), rtol=1e-8)
    e4 = product_energies(R, mu, topk={4: C})
    assert np.isclose(sum(v for _, v in e4), (mu[:, 4] ** 2).sum(), rtol=1e-8)


def test_analyse_row_and_judge():
    rng = np.random.default_rng(4)
    C = 12
    G = _psd(rng, C, spike=5.0)
    mean = rng.standard_normal(C) * 0.3
    full, _, _, _ = surrogate(mean, G)
    r = analyse_row(mean, G, np.zeros(C), full)
    assert r["sandwich_ok@0.95"] and r["sandwich_ok@0.999"]
    rows = []
    for i in range(31):
        d = dict(group="vgg" if i < 13 else "stream_init", layer=f"l{i}", C=64, kappa=1.2)
        for t in TAUS:
            tt = f"{t:g}"
            d.update({f"k_pred@{tt}": 21, f"pred_err@{tt}": 0.05, f"k_lo@{tt}": 10, f"k_lo2@{tt}": 12, f"lo2_ratio@{tt}": 0.6, f"lo2_ok@{tt}": True, f"k_lin@{tt}": 14, f"k_surr@{tt}": 20, f"k_hi@{tt}": 24 if i < 25 else 40, f"k_a@{tt}": 18, f"k_hi_lin@{tt}": 15, f"k_hi_prod@{tt}": 9,
                      f"sandwich_ok@{tt}": True, f"hi_ratio@{tt}": (24 if i < 25 else 40) / 20, f"lo_ratio@{tt}": 0.5, f"k_G@{tt}": 12})
        rows.append(d)
    v, text = judge(pd.DataFrame(rows))
    assert v["p120"] and v["p120_rows"] == 31
    assert v["p118"] and v["p118_rows"] == 25 and v["p119"] and v["sandwich"] == 62


def test_generic_prediction_total_and_isotropic_limit():
    rng = np.random.default_rng(5)
    C = 8
    G = _psd(rng, C, spike=2.0)
    R, s, mu = coefficients(np.zeros(C), G)
    lam = np.linalg.eigvalsh(R) / C
    # the j = 2 generic energies over all ordered pairs sum to |mu_2|^2 (outer product of the normalised spectrum sums to 1)
    assert np.isclose(np.outer(lam, lam).sum(), 1.0)
    # with every energy included the tau-count is between 1 and the number of elements, monotone in tau
    _, lin, _, _ = surrogate(np.zeros(C), G)
    ks = [generic_prediction(lin, R, mu, t) for t in (0.5, 0.9, 0.95, 0.999)]
    assert ks == sorted(ks) and ks[0] >= 1
