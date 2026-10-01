"""B9 (plan 9.29): Lemma B9.0 on synthetic cell-structured samples, and the cliff judge on synthetic spectra."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import position_collapse_judge as J  # noqa: E402


def _samples(P, K, C, n_per_cell, within, rng, shared_positions=False):
    mu = rng.standard_normal((1 if shared_positions else P, K, C))
    X = []
    for p in range(P):
        for c in range(K):
            m = mu[0 if shared_positions else p, c]
            X.append(m + within * rng.standard_normal((n_per_cell, C)))
    return np.concatenate(X)


def test_lemma_b9_0_bound_and_attainment():
    rng = np.random.default_rng(0)
    P, K, C = 4, 10, 128
    for tau in (0.95, 0.999):
        X = _samples(P, K, C, 400, 0.002, rng)
        S = np.cov(X.T)
        w = (0.002 ** 2 * C) / np.trace(S)
        assert w <= 1 - tau
        e = np.linalg.eigvalsh(S)
        assert J.k_star(e, tau) <= P * K - 1
    g, ratio = J.cliff(e, K)
    assert g == P * K - 1 and ratio > 100                        # independent position-wise class means: rank PK - 1 exactly
    Xs = _samples(P, K, C, 400, 0.002, rng, shared_positions=True)
    g, ratio = J.cliff(np.linalg.eigvalsh(np.cov(Xs.T)), 2)
    assert g == K - 1 and ratio > 100                            # means shared across positions: the classical K - 1


def test_lemma_b9_0_is_silent_when_cells_are_not_collapsed():
    rng = np.random.default_rng(1)
    X = _samples(4, 10, 128, 400, 1.0, rng)
    assert J.k_star(np.linalg.eigvalsh(np.cov(X.T)), 0.999) > 39


def test_judge_on_synthetic_spectra(tmp_path, capsys):
    rng = np.random.default_rng(2)
    gg = tmp_path / "gradgram"; gg.mkdir()

    def spec(rank, C=512):
        e = np.concatenate([np.sort(rng.uniform(1, 2, rank))[::-1], 1e-4 * np.sort(rng.uniform(0.5, 1, C - rank))[::-1]])
        return e
    for arm, K in J.ARM_K.items():
        for s in range(3):
            rank = 4 * K - 1 if arm != "c10_pc640" else 300
            np.savez(gg / f"{arm}_s{s}_spectra.npz", **{f"{layer}::out": spec(min(rank, 250)) if arm == "c100_k100" else spec(rank) for layer in J.BLOCK5})
    V = J.judge(str(tmp_path))
    assert V["P-b9.1"] is True and V["P-b9.2"] is True and V["P-b9.3"] is True
    out = capsys.readouterr().out
    assert "18/18" in out
    for arm in ("c100_k13", "c100_k20"):                          # move the blind arms' cliff away from 4K: the verdict must fail
        for s in range(3):
            np.savez(gg / f"{arm}_s{s}_spectra.npz", **{f"{layer}::out": spec(30) for layer in J.BLOCK5})
    assert J.judge(str(tmp_path))["P-b9.2"] is False
