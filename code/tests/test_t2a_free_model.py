import numpy as np
import pandas as pd

from scripts.t1_kernel_reading import eig, kstar, surrogate
from scripts.t2a_free_model import TAUS, analyse_row, compress, compressed_powers, haar, hadamard_square_bprime, judge, model_a, model_b, model_bprime


def test_haar_is_orthogonal():
    rng = np.random.default_rng(0)
    O = haar(rng, 20)
    assert np.allclose(O @ O.T, np.eye(20), atol=1e-10)


def test_compression_keeps_trace_and_flattens_identity():
    rng = np.random.default_rng(1)
    C = 40
    big = np.ones(C * C)
    M = compress(rng, big, C)
    e = eig(M)
    assert abs(np.trace(M) / C - 1) < 0.1              # tr M ~ (1/N) sum_t b_t |g_t|^2 ~ C
    assert e.max() - e.min() < 0.6                     # flat spectrum compresses to ~ 1 +- O(1/sqrt C)
    lam = np.arange(1, C + 1) ** -1.5; lam = lam / lam.sum() * C
    comp = compressed_powers(rng, lam, 3)
    assert set(comp) == {2, 3}
    assert abs(np.trace(comp[2]) / C - 1) < 0.15       # diag of R^{o2} is 1, so the modelled trace is ~ C


def test_model_a_contains_truth_for_haar_rotated_gram():
    rng = np.random.default_rng(2)
    C = 24
    lam = np.arange(1, C + 1) ** -1.0
    U = haar(rng, C)
    G = (U * lam) @ U.T
    m = rng.standard_normal(C) * np.sqrt(np.diag(G)) * 0.5
    ks = model_a(rng, m, G, 16)
    full, _, _, _ = surrogate(m, G)
    for t in TAUS:
        k = kstar(eig(full), t)
        assert min(ks[t]) - 1 <= k <= max(ks[t]) + 1


def test_model_b_shapes_and_trace():
    rng = np.random.default_rng(3)
    C = 16
    U = haar(rng, C); lam = np.arange(1, C + 1) ** -0.8
    G = (U * lam) @ U.T
    m = np.zeros(C)
    tot, simple = model_b(rng, m, G)
    full, _, _, _ = surrogate(m, G)
    assert tot.shape == (C, C) and simple.shape == (C, C)
    assert abs(np.trace(tot) / np.trace(full) - 1) < 0.15
    r = analyse_row(m, G, n_rot=3, n_draws=2)
    assert "kB_med@0.95" in r and "kBp_med@0.95" in r and r["k_surr@0.95"] >= 1
    totp = model_bprime(rng, m, G)
    assert abs(np.trace(totp) / np.trace(full) - 1) < 0.15


def test_bprime_square_has_exact_trace_and_spike():
    rng = np.random.default_rng(6)
    C = 64
    lam = np.arange(1, C + 1) ** -1.5; lam = lam / lam.sum() * C
    M = hadamard_square_bprime(rng, lam)
    assert abs(np.trace(M) - C) < 1e-8
    # top eigenvalue carries the diagonal-tuple structure: at least the e-direction value tr R^2 / C after rescaling
    assert eig(M).max() >= 0.5 * (lam ** 2).sum() / C


def test_judge_counts():
    rows = []
    for i in range(31):
        g = "vgg" if i < 13 else ("stream_init" if i < 25 else "stream_trained")
        d = dict(group=g, layer=f"l{i}", C=64, top5_ratio_B_over_true="1.00;1.00;1.00;1.00;1.00")
        for t in TAUS:
            tt = f"{t:g}"
            d.update({f"k_surr@{tt}": 20, f"kA_min@{tt}": 18, f"kA_max@{tt}": 22, f"kA_med@{tt}": 20.0, f"kA_sd@{tt}": 1.0, f"A_in_range@{tt}": True, f"A_err@{tt}": 0.0,
                      f"kB_med@{tt}": 21.0, f"B_err@{tt}": 0.05, f"kB2_med@{tt}": 23.0, f"B2_err@{tt}": 0.15, f"kBp_med@{tt}": 20.5, f"Bp_err@{tt}": 0.025})
        d["top5_ratio_Bp_over_true"] = "1.00;1.00;1.00;1.00;1.00"
        rows.append(d)
    v, text = judge(pd.DataFrame(rows))
    assert v["p121"] and v["p122"] and v["p122_vgg"] == 13 and v["p122_init"] == 12
    assert v["p123"] and v["p123_better"] == 13
