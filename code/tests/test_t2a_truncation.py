import numpy as np
import pandas as pd

from scripts.t1_kernel_reading import eig, kstar
from scripts.t2a_truncation import (actual_tuple_norm, analyse_row, haar_value, hadamard_square_variants, judge, models,
                                    predicted_tuple_norm)


def _corr_from_cov(G):
    d = 1 / np.sqrt(np.diag(G))
    return G * d[:, None] * d[None, :]


def test_lemma_t2a1_tuple_energy_bound():
    """lam_i lam_i' ||u_i o u_i'||^2 <= min(lam_i, lam_i') for every correlation matrix (exact)."""
    rng = np.random.default_rng(0)
    for C, spike in ((30, 0.0), (30, 20.0), (50, 45.0)):
        W = rng.standard_normal((C, 3 * C))
        v = rng.standard_normal(C)
        G = W @ W.T / (3 * C) + spike * np.outer(v, v) / C
        R = _corr_from_cov(G)
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 0, None)
        for i in range(C):
            for j in range(C):
                e = w[i] * w[j] * ((V[:, i] * V[:, j]) ** 2).sum()
                assert e <= min(w[i], w[j]) + 1e-9
        assert np.all(C * (V ** 4).sum(axis=0) >= 1 - 1e-9)


def test_predicted_norm_limits():
    C = 64
    flat = np.ones(C)
    assert np.allclose(predicted_tuple_norm(flat), haar_value(C))           # tr R^2 = C -> ratio 0 -> Haar expectation 3C/(C+2)
    rank_one = np.zeros(C); rank_one[0] = C
    p = predicted_tuple_norm(rank_one)
    assert abs(p[0] - 1.0) < 1e-9                                          # rank-one spike: u_1 flat, C ||u o u||^2 = 1
    assert p[0] <= p[1:].max()
    assert np.all(p >= 1 - 1e-12) and np.all(p <= haar_value(C) + 1e-12)


def test_haar_vectors_match_unconstrained_value():
    rng = np.random.default_rng(1)
    C = 200
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    a = C * (Q ** 4).sum(axis=0)
    assert abs(a.mean() - haar_value(C)) < 0.1


def test_strong_spike_truncates_actual_norm():
    """A correlation matrix with one eigenvalue > C/3 has C ||u_1 o u_1||^2 well below the Haar value 3 and above 1."""
    rng = np.random.default_rng(2)
    C = 64
    v = rng.standard_normal(C)
    G = 0.05 * np.eye(C) + np.outer(v, v)
    lam, a = actual_tuple_norm(_corr_from_cov(G))
    assert lam[0] > C / 3
    assert 1.0 <= a[0] < 1.6
    p = predicted_tuple_norm(np.clip(np.sort(eig(_corr_from_cov(G)))[::-1], 0, None))
    assert abs(p[0] - a[0]) < 0.5


def test_bdprime_top_eigenvalue_respects_schur_bound():
    """B'' never puts more than lam_1 at the top of R^{o2} when lam_1 > C/3; B' does (3 lam_1^2 / C > lam_1)."""
    rng = np.random.default_rng(3)
    C = 64
    lam = np.zeros(C); lam[0] = 0.8 * C; lam[1:] = (C - lam[0]) / (C - 1)
    Mp, Mpp, diag = hadamard_square_variants(rng, lam)
    assert diag['tr_ratio_Bpp'] < diag['tr_ratio_Bp']                     # B'' needs less trace renormalisation
    assert abs(np.trace(Mpp) - C) < 1e-8
    assert eig(Mpp).max() <= lam[0] * 1.05
    assert eig(Mp).max() > lam[0]                                           # B' violates the Schur bound S_1(R^{o2}) <= lam_1


def test_models_and_judge_run():
    rng = np.random.default_rng(4)
    C = 32
    W = rng.standard_normal((C, 4 * C)); G = W @ W.T / (4 * C)
    m = 0.3 * rng.standard_normal(C)
    Tp, Tpp, _ = models(rng, m, G)
    assert Tp.shape == (C, C) and np.allclose(Tpp, Tpp.T)
    out, vec = analyse_row(m, G, n_draws=2)
    assert out["C"] == C and 1 <= out["k_surr@0.95"] <= C and len(vec) == min(C, 48)
    rows = []
    for g, lay in (("vgg", "features.2"), ("vgg", "features.5"), ("stream_init", "layer1.0"), ("stream_trained", "layer1.0")):
        r = dict(out); r.update(group=g, layer=lay, tag="t"); rows.append(r)
    df = pd.DataFrame(rows)
    vecs = pd.concat([vec.assign(group=g, layer=lay, C=C) for g, lay, _ in [(r["group"], r["layer"], 0) for r in rows]], ignore_index=True)
    verdict, text = judge(df, vecs)
    assert "P9.124" in text and "P9.125" in text and set(verdict) >= {"p124", "p125"}
