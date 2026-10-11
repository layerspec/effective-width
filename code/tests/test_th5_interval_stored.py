"""Tests for scripts/th5_interval_stored.py (plan 9.55; Theorem M.2 / Th5' on stored spectra)."""
import numpy as np
import pandas as pd

from scripts.th5_interval_stored import TAUS, chebyshev_interval, draw_tail_statistics, interval_from_spectrum, judge, npz_key

C, D, ALPHA = 64, 128, 1.2
LAM = np.arange(1, D + 1.0) ** -ALPHA


def test_npz_key():
    assert npz_key("features.0") == "features_0"
    assert npz_key("layer1.0.conv2") == "layer1_0_conv2"
    assert npz_key("conv1") == "conv1"


def test_tail_statistics_consistent():
    rng = np.random.default_rng(0)
    E, R, G2, T2, K = draw_tail_statistics(LAM, C, 5, rng)
    r = min(C, D)
    assert E.shape == (5, r) and R.shape == (5, r + 1) and G2.shape == (5, r + 1)
    assert np.allclose(R[:, 0], E.sum(1)) and np.allclose(R[:, 0], T2) and np.all(R[:, -1] == 0)
    assert np.all(np.diff(R, axis=1) <= 1e-9) and np.all(np.diff(G2, axis=1) <= 1e-9)   # tails decrease in k
    assert np.all(G2 <= 4 * LAM.max() * R + 1e-9)                                       # Sigma <= lambda_1 I
    for j, tau in enumerate(TAUS):
        assert np.all(K[:, j] >= 1) and np.all(K[:, j] <= r)


def test_poincare_bound_dominates_variance():
    rng = np.random.default_rng(1)
    E, R, G2, T2, K = draw_tail_statistics(LAM, C, 600, rng)
    k = int(round(K[:, 0].mean()))
    assert R[:, k].var() <= 1.15 * G2[:, k].mean()          # Var R_k <= c_P E G2_k (15% Monte Carlo slack)
    assert T2.var() <= 1.15 * G2[:, 0].mean()


def test_q_exact_matches_monte_carlo():
    res = interval_from_spectrum(LAM, C, draws=400, seed=2)
    assert abs(res["q_mc"] / res["q"] - 1) < 0.25
    assert abs(res["q"] - np.sqrt(2 * np.sum(LAM ** 2) / (C * LAM.sum() ** 2))) < 1e-12


def test_interval_covers_fresh_draws_and_orders():
    res = interval_from_spectrum(LAM, C, u=4.0, draws=300, seed=3)
    rng = np.random.default_rng(99)
    _, _, _, _, K = draw_tail_statistics(LAM, C, 300, rng)      # independent draws of the same law
    for j, tau in enumerate(TAUS):
        o = res[tau]
        assert o["k_minus"] <= o["k_plus"]
        assert o["k_minus_lam"] <= o["k_minus"] and o["k_plus"] <= o["k_plus_lam"]   # lambda_1 version is looser
        inside = np.mean((o["k_minus"] <= K[:, j]) & (K[:, j] <= o["k_plus"]))
        assert inside >= 1 - 4 / 4.0 ** 2
    assert res[0.95]["k_plus"] - res[0.95]["k_minus"] <= 0.5 * C                        # not vacuous at tau = 0.95


def test_chebyshev_interval_edge_cases():
    r = 10
    hbar = np.linspace(1, 0, r + 1)
    rk = np.zeros(r + 1)
    km, kp = chebyshev_interval(hbar, rk, q=0.0, u=4.5, tau=0.95, r=r)
    assert km == kp == 10                     # h_k = 1 - k/10 <= 0.05 first at k = 10
    km, kp = chebyshev_interval(hbar, rk, q=0.5, u=4.5, tau=0.95, r=r)
    assert kp == r                            # u q >= 1: upper curve undefined -> vacuous
    km, kp = chebyshev_interval(hbar, np.full(r + 1, 10.0), q=0.0, u=4.5, tau=0.95, r=r)
    assert km == 1 and kp == r                # huge r_k: lower bound trivially met at k = 1, upper never


def test_judge_logic():
    def frame(inside95, inside999, hw, degenerate):
        rows = []
        for i in range(10):
            for tau, ins in ((0.95, inside95[i]), (0.999, inside999[i])):
                rows.append(dict(family="f", layer=f"l{i}", C=64, d=576, r_max=64, n_over_d=100.0, tau=tau, measured=10, mc_mean=10.0, mc_sd=1.0,
                                 k_minus=5, k_plus=15, k_minus_mc=5, k_plus_mc=15, k_minus_lam=1, k_plus_lam=64, q=0.1, q_mc=0.1,
                                 r_poinc_at_k=0.1, r_mc_at_k=0.05, inside=ins, inside_mc=ins, inside_lam=1,
                                 halfwidth_over_rmax=hw, halfwidth_lam_over_rmax=0.5, degenerate=degenerate, degenerate_lam=1))
        return pd.DataFrame(rows)
    good = frame([1] * 9 + [0], [1] * 8 + [0, 0], 0.05, 0)
    v, text = judge(good)
    assert v["P9.126"] and v["P9.127"] and "HOLDS" in text
    v, _ = judge(frame([1] * 7 + [0] * 3, [1] * 10, 0.05, 0))
    assert not v["P9.126"]
    v, _ = judge(frame([1] * 10, [1] * 10, 0.2, 0))
    assert not v["P9.127"]
    v, _ = judge(frame([1] * 10, [1] * 10, 0.05, 1))
    assert not v["P9.127"]
