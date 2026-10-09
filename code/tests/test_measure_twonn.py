import numpy as np
import pandas as pd

from scripts.measure_twonn import judge_frame, kstar, ordering_class, two_nearest, twonn, twonn_from_ratios


def _embed(rng, n, d, D):
    Z = rng.standard_normal((n, d))
    Q, _ = np.linalg.qr(rng.standard_normal((D, d)))
    return Z @ Q.T


def test_two_nearest_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 7))
    r1, r2 = two_nearest(X, chunk=16)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    S = np.sort(D, axis=1)
    assert np.allclose(r1, S[:, 0], atol=1e-4) and np.allclose(r2, S[:, 1], atol=1e-4)


def test_twonn_recovers_gaussian_dimension():
    rng = np.random.default_rng(1)
    d, _ = twonn(_embed(rng, 4000, 5, 50))
    assert 4.2 <= d <= 5.8


def test_twonn_recovers_a_two_dimensional_sheet():
    rng = np.random.default_rng(2)
    U = rng.uniform(size=(3000, 2))
    X = np.concatenate([U, np.sin(3 * U), np.cos(2 * U[:, :1])], axis=1)    # curved 2-D sheet in 5-D
    d, _ = twonn(X)
    assert 1.7 <= d <= 2.4


def test_duplicates_are_dropped_not_fitted():
    rng = np.random.default_rng(3)
    X = _embed(rng, 1500, 4, 20)
    Xd = np.concatenate([X, X[:300]])                                           # 300 exact duplicates (r1 = 0)
    d0, m0 = twonn(X)
    d1, m1 = twonn(Xd)
    assert np.isfinite(d1) and m1 == m0 and abs(d1 - d0) < 1e-6


def test_dependent_near_duplicates_lower_the_estimate():
    # 16 jittered copies per point: the neighbourhood is a tiny 2-D sheet, so TwoNN falls below the 6-D value
    rng = np.random.default_rng(4)
    X = _embed(rng, 400, 6, 30)
    sheet = rng.standard_normal((30, 2)); sheet /= np.linalg.norm(sheet, axis=0)
    jit = rng.uniform(-0.05, 0.05, size=(400, 16, 2)) @ sheet.T
    Xdep = (X[:, None, :] + jit).reshape(-1, 30)
    d_ind, _ = twonn(X)
    d_dep, _ = twonn(Xdep)
    assert d_dep < d_ind


def test_twonn_from_ratios_trims_top_fraction():
    r1 = np.ones(100); r2 = np.linspace(1.01, 3.0, 100)
    d_full, m = twonn_from_ratios(r1, r2, trim=0.1)
    assert m == 90 and np.isfinite(d_full)


def test_kstar_and_ordering_class():
    e = np.exp(-np.arange(20) / 3.0)
    assert kstar(e, 0.5) <= kstar(e, 0.95) <= 20
    assert ordering_class(0.9) == "same" and ordering_class(0.6) == "partial" and ordering_class(0.1) == "different"


def _frame(id_s1, id_s2, id_s3, k95, model="m"):
    n = len(id_s1)
    return pd.DataFrame({"model": [model] * n, "layer": [f"l{i}" for i in range(n)], "id_S1": id_s1, "id_S2": id_s2,
                         "id_S3": id_s3, "id_P": id_s1, "k90": np.array(k95) - 2, "k95": k95,
                         "k95_rmax": np.array(k95) / 100.0, "r_max": 100.0, "k95_S1_check": np.nan})


def test_judge_frame_registered_rules():
    k = [40, 60, 80, 100, 120]
    good = _frame([13, 10, 12, 11, 10.5], [5, 6, 5, 7, 6], [13, 10, 12, 11, 10.5], k)  # ladder holds, S2 < S3, ID not ordered like k
    res = judge_frame(good)
    assert res["P9.110"]["holds"] and res["P9.111"]["holds"]
    assert res["P9.112"]["per_model"]["m"]["class"] == "different" and res["P9.112"]["holds"]
    bad = _frame([50, 70, 90, 110, 130], [60, 80, 100, 120, 140], [50, 70, 90, 110, 130], k)   # ID above k90, S2 > S3, same order
    res = judge_frame(bad)
    assert not res["P9.110"]["holds"] and not res["P9.111"]["holds"] and not res["P9.112"]["holds"]
    assert res["P9.112"]["per_model"]["m"]["class"] == "same"


def test_judge_frame_ignores_random_controls_for_the_registered_counts():
    k = [40, 60, 80]
    df = pd.concat([_frame([10, 11, 12], [5, 5, 5], [10, 11, 12], k, "a"),
                    _frame([99, 99, 99], [99, 99, 99], [1, 1, 1], k, "a_random")], ignore_index=True)
    res = judge_frame(df)
    assert res["P9.110"]["total"] == 3 and res["P9.110"]["holds"]
    assert "a_random" in res["exploratory"] and "a_random" not in res["P9.112"]["per_model"]
