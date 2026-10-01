"""B8 (plan 9.28): the judge of scripts/pah_width_judge.py on synthetic matrices with known exponents."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import pah_width_judge as P  # noqa: E402


def _orth(n, rng):
    return np.linalg.qr(rng.standard_normal((n, n)))[0]


def test_lemma_b8_0_width_is_monotone_in_the_power():
    """k*_tau(lambda^alpha) is non-increasing in alpha, and so is every normalised partial sum's complement."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        lam = np.sort(rng.gamma(0.3, size=rng.integers(5, 300)))[::-1]
        for tau in (0.5, 0.95, 0.999):
            ks = [P.power_width(lam, a, tau) for a in (0.25, 0.5, 1.0, 1.7, 2.0, 3.0)]
            assert all(x >= y for x, y in zip(ks, ks[1:])), (ks, tau)
        for a, b in ((2.0, 1.0), (1.0, 0.4), (3.0, 2.5)):
            fa = np.cumsum(lam ** a) / (lam ** a).sum(); fb = np.cumsum(lam ** b) / (lam ** b).sum()
            assert np.all(fa >= fb - 1e-12)


def test_basis_stats_recovers_a_planted_power():
    rng = np.random.default_rng(1)
    C = 160
    U = _orth(C, rng)
    gam = np.exp(-np.arange(C) / 12.0)
    G = (U * gam) @ U.T
    for alpha in (1.0, 2.0, 2.6):
        S = 3.7 * (U * gam ** alpha) @ U.T
        r = P.basis_stats(S, G, np.random.default_rng(0))
        assert r["rho_b"] == pytest.approx(1.0)
        assert r["codiag"] == pytest.approx(1.0, abs=1e-8)
        assert r["codiag_null"] < 0.3
        assert r["alpha_hat"] == pytest.approx(alpha, abs=1e-6)
        for tau in P.TAUS:
            assert r[f"kS_{tau}"] == r[f"kGhat_{tau}"] == P.power_width(gam, alpha, tau)
        assert r["kG2_0.999"] <= r["kG1_0.999"]


def test_basis_stats_sees_an_unrelated_basis():
    rng = np.random.default_rng(2)
    C = 120
    gam = np.exp(-np.arange(C) / 10.0)
    G = (_orth(C, rng) * gam) @ _orth(C, rng).T
    G = G @ G.T
    V = _orth(C, rng)
    S = (V * gam ** 2) @ V.T
    r = P.basis_stats(S, G, np.random.default_rng(0))
    assert r["codiag"] < 0.5 and abs(r["rho_b"]) < 0.5


def test_three_matrix_phase_8():
    """W^T W prop. H_a and W W^T prop. Gamma give Sigma_out prop. (W W^T)^2: a_HZ = 2, a_ZG = 1, a_HG = 2."""
    rng = np.random.default_rng(3)
    C, d = 64, 200
    U = _orth(C, rng); Vf = _orth(d, rng)[:, :C]
    s = np.exp(-np.arange(C) / 9.0)
    W = (U * s) @ Vf.T
    Ha = W.T @ W
    G = W @ W.T
    S = W @ Ha @ W.T
    r = P.three_matrix(W.reshape(C, d, 1, 1), S, G)
    assert r["a_HZ"] == pytest.approx(2.0, abs=1e-6)
    assert r["a_ZG"] == pytest.approx(1.0, abs=1e-6)
    assert r["a_HG"] == pytest.approx(2.0, abs=1e-6)


def test_exponents_compose_for_general_powers():
    """gain prop. sigma^p and gain prop. gamma^q in a shared basis: a_HZ = 1 + 1/p, a_ZG = q, a_HG = q (1 + 1/p)."""
    gain = np.exp(-np.arange(80) / 7.0)
    for p, q in ((1.0, 1.0), (2.0, 0.5), (0.5, 1.5)):
        sigma = gain ** (1 / p); gamma = gain ** (1 / q)
        r = P.exponents(gain, gain * sigma, gamma)
        assert r["a_HZ"] == pytest.approx(1 + 1 / p, abs=1e-8)
        assert r["a_ZG"] == pytest.approx(q, abs=1e-8)
        assert r["a_HG"] == pytest.approx(q * (1 + 1 / p), abs=1e-8)


def _fake_run(results, run, alpha, rng, widths=(64, 128, 256, 256)):
    os.makedirs(os.path.join(results, "gradwindow"), exist_ok=True)
    arrs = {}
    for i, C in enumerate(widths):
        U = _orth(C, rng)
        gam = np.exp(-np.arange(C) / (C / 20.0))
        arrs[f"features.{3 * i}::grad"] = ((U * gam) @ U.T).astype(np.float32)
        arrs[f"features.{3 * i}::out"] = ((U * gam ** alpha) @ U.T).astype(np.float32)
    np.savez(os.path.join(results, "gradwindow", f"{run}_epoch_100.npz"), **arrs)


def test_judge_withholds_until_six_runs_then_judges(tmp_path, capsys):
    rng = np.random.default_rng(4)
    results = str(tmp_path)
    for run in P.FULL_RUNS[:2]:
        _fake_run(results, run, 2.0, rng)
    assert P.judge(results) == {}
    assert "withheld" in capsys.readouterr().out
    for run in P.FULL_RUNS[2:]:
        _fake_run(results, run, 2.0, rng)
    V = P.judge(results)
    assert V["P-b8.1"] and V["P-b8.2"] and V["P-b8.3"]
    assert V["P-b8.4"] is False                       # no weights, no ImageNet directions in the synthetic tree
    saved = json.load(open(os.path.join(results, "pah", "pah_judgement.json")))
    assert saved["verdicts"]["P-b8.3"] is True
    df = __import__("pandas").read_csv(os.path.join(results, "pah", "pah_layers.csv"))
    assert len(df) == 6 * 4 and (df.alpha_hat.sub(2.0).abs() < 1e-3).all()


def test_judge_fails_the_width_prediction_when_the_power_is_one(tmp_path):
    rng = np.random.default_rng(5)
    results = str(tmp_path)
    for run in P.FULL_RUNS:
        _fake_run(results, run, 1.0, rng)
    V = P.judge(results)
    assert V["P-b8.1"] and not V["P-b8.2"] and not V["P-b8.3"]


def test_window_exponent_verdicts(tmp_path):
    """P-b8.5 / P-b8.6: the window Gram (weights json + per-epoch files) is read next to the final-epoch Gram."""
    rng = np.random.default_rng(6)
    results = str(tmp_path)
    gw = os.path.join(results, "gradwindow"); os.makedirs(gw)
    end_power = {"5e-5": 3.0, "5e-4": 3.0, "5e-3": 1.2}            # Sigma_out = Gamma_end^(1/power): exponent 1/3 or 1/1.2 on the final Gram
    for run in P.FULL_RUNS:
        lam = run[2:].split("_")[0]
        a90, a100 = {}, {}
        for i, C in enumerate((128, 256)):
            U = _orth(C, rng); gam = np.exp(-np.arange(C) / (C / 20.0))
            name = f"features.{3 * i}"
            a90[f"{name}::grad"] = ((U * gam) @ U.T).astype(np.float64); a90[f"{name}::out"] = a90[f"{name}::grad"]
            a100[f"{name}::grad"] = (U * gam ** end_power[lam]) @ U.T    # the window is all at epoch 90; the final Gram is a power of it
            a100[f"{name}::out"] = (U * gam) @ U.T                       # Sigma_out prop. window Gram exactly
        np.savez(os.path.join(gw, f"{run}_epoch_090.npz"), **a90); np.savez(os.path.join(gw, f"{run}_epoch_100.npz"), **a100)
        json.dump({"weights": {"90": 1.0, "100": 0.0}}, open(os.path.join(gw, f"{run}_weights.json"), "w"))
    V = P.judge(results)
    assert V["P-b8.5"] is True and V["P-b8.6"] is True
    df = __import__("pandas").read_csv(os.path.join(results, "pah", "pah_layers.csv"))
    assert (df.alpha_hat_win.sub(1.0).abs() < 1e-6).all() and (df.codiag_win > 0.999).all()
