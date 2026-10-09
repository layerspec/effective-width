import numpy as np
import pandas as pd

from scripts.t1_kernel_reading import TAUS, angles, centring, judge, kstar, surrogate


def _psd(rng, C, alpha=1.0):
    A = rng.standard_normal((C, C))
    S = A @ A.T / C
    w, V = np.linalg.eigh(S)
    w = np.sort(w)[::-1] * np.arange(1, C + 1) ** (-alpha)          # power-law-ish spectrum
    return (V * w) @ V.T


def test_kstar_convention():
    assert kstar([0.5, 0.3, 0.2], 0.95) == 3
    assert kstar([0.5, 0.3, 0.2], 0.5) == 1
    assert kstar([0.5, 0.3, 0.2], 0.0) == 1          # tau'' <= 0 convention of Lemma T1.3
    assert kstar([0.0, 0.0], 0.5) == 0


def test_centring_lemma_holds_on_random_pairs():
    rng = np.random.default_rng(0)
    for trial in range(200):
        C = int(rng.integers(5, 60))
        S = _psd(rng, C, alpha=rng.uniform(0.2, 2.0))
        m = rng.standard_normal(C)
        m *= np.sqrt(rng.choice([0.05, 0.5, 5.0, 50.0]) * np.trace(S)) / np.linalg.norm(m)
        for tau in (0.9, 0.95, 0.999):
            c = centring(S, m, tau)
            assert c["lemma_ok"], (trial, tau, c)


def test_surrogate_matches_gaussian_sampling_and_forms_agree_on_diagonal():
    rng = np.random.default_rng(1)
    C, n = 6, 400_000
    A = rng.standard_normal((C, C)); S = A @ A.T / C + 0.1 * np.eye(C)
    m = rng.standard_normal(C) * 0.8
    z = rng.multivariate_normal(m, S, size=n)
    a = np.maximum(z, 0)
    cov_emp = np.cov(a.T)
    full, lin, ge, m_g = surrogate(m, S)
    assert np.linalg.norm(full - cov_emp) / np.linalg.norm(cov_emp) < 0.03
    assert np.allclose(np.diag(ge), np.diag(full), rtol=1e-8)
    assert np.allclose(np.diag(lin) + np.diag(ge) - np.diag(lin), np.diag(full))
    assert np.allclose(m_g, a.mean(axis=0), rtol=0.02)


def test_angles_estimate_effective_rank_of_sigma_p():
    rng = np.random.default_rng(2)
    d, C = 400, 120
    lam = np.arange(1, d + 1) ** -1.0
    reff_true = lam.sum() ** 2 / (lam ** 2).sum()
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sp = (Q * lam) @ Q.T
    est_p, est_rho = [], []
    for _ in range(8):
        W = rng.standard_normal((C, d))
        G = W @ Sp @ W.T
        a = angles(G)
        est_p.append(a["reff_p_hat"]); est_rho.append(a["reff_hat"])
    assert abs(np.mean(est_p) / reff_true - 1) < 0.25
    assert abs(np.mean(est_rho) / reff_true - 1) < 0.35


def _frame(n_vgg_ok=13, n_init_ok=12, rho=0.3, reff_p=10.0):
    rows = []
    for g, n, n_ok in (("vgg", 13, n_vgg_ok), ("stream_init", 12, n_init_ok), ("stream_trained", 6, 3)):
        for i in range(n):
            r = dict(group=g, layer=f"l{i}", C=64, rms_rho=rho, mean_rho2=rho ** 2, reff_hat=1 / rho ** 2, reff_p_hat=reff_p,
                     beta=0.4, beta_over_const=0.4 * (np.pi - 1), snr_z=1.0, surr_xcheck=0.01)
            for t in TAUS:
                tt = f"{t:g}"
                r.update({f"k_lin@{tt}": 10, f"k_a@{tt}": 15, f"k_surr@{tt}": 20, f"k_ge@{tt}": 25, f"k_M@{tt}": 14, f"tau2@{tt}": 0.93,
                          f"k_a_tau2@{tt}": 14, f"lemma_ok@{tt}": True, f"order_ok@{tt}": i < n_ok})
            rows.append(r)
    return pd.DataFrame(rows)


def test_judge_counts_and_verdicts():
    v, text = judge(_frame())
    assert v["p116"] and v["p117"] and v["lemma_ok"] == 62 and v["xcheck_ok"] == 13
    v, _ = judge(_frame(n_vgg_ok=10))
    assert not v["p117"] and v["p117_vgg"] == 10
    v, _ = judge(_frame(rho=0.05))
    assert not v["p116"] and v["p116_rows"] == 0
    v, _ = judge(_frame(reff_p=80.0))
    assert not v["p116"] and v["p116_vgg_reff"] == 0
