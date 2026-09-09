"""rho_align, the kernel-data alignment index of analysis-plan 9.6."""
import numpy as np
import torch
import torch.nn as nn

from layerspec.decomposition import rho_align


def _svd(W):
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    return sv, Vt


def test_isotropic_kernel_gives_nan_and_aligned_kernels_have_the_right_sign():
    rng = np.random.default_rng(0)
    d = 16
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.linspace(10, 1, d)                      # data variance, decreasing along Q's columns
    Sig = Q @ np.diag(lam) @ Q.T
    # isometric kernel: all singular values equal -> constant gain -> nan
    sv, Vt = _svd(Q.T[:8])
    assert np.isnan(rho_align(sv, Vt, Sig, 8))
    # widening kernel: gain decreasing along the same directions as the data variance -> +1
    W = np.diag(np.linspace(4, 1, 8)) @ Q.T[:8]
    sv, Vt = _svd(W)
    assert rho_align(sv, Vt, Sig, 8) > 0.99
    # compressive kernel: gain increasing where the data variance decreases -> -1
    W = np.diag(np.linspace(1, 4, 8)) @ Q.T[:8]
    sv, Vt = _svd(W)
    assert rho_align(sv, Vt, Sig, 8) < -0.99


def test_decompose_table_carries_rho_align_in_range():
    import layerspec
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 8, 1))
    x = torch.randn(128, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=32)
    dec = layerspec.decompose(net, loader, device="cpu", positions=16)
    assert list(dec.table.columns)[-2:] == ["rho_align", "rho_align_sigma"]
    v = dec.table.rho_align.dropna()
    assert len(v) == 2 and ((v >= -1) & (v <= 1)).all()
    assert "rho_align_median" in dec.check()


def test_rho_align_sigma_matches_proposition_4_for_least_squares():
    """W* = Sigma_tx Sigma^-1: gain on q_j is c_j / lambda_j^2, so the index equals
    Spearman(c_j / lambda_j^2, lambda_j); flat relevance gives -1."""
    from layerspec.decomposition import rho_align_sigma
    rng = np.random.default_rng(1)
    d, m = 24, 6
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.sort(rng.uniform(0.5, 20, d))[::-1]
    Sig = Q @ np.diag(lam) @ Q.T
    # random relevance
    A = rng.standard_normal((m, d))                    # Sigma_tx = A Q^T  ->  c_j = ||A[:, j]||^2
    Sig_tx = A @ Q.T
    W = Sig_tx @ np.linalg.inv(Sig)
    c = np.sum(A ** 2, axis=0)
    from scipy.stats import spearmanr
    expected = spearmanr(c / lam ** 2, lam).statistic
    assert abs(rho_align_sigma(W, Sig) - expected) < 1e-9
    # flat relevance: whitening, index -1
    A_flat = np.ones((1, d))
    W_flat = (A_flat @ Q.T) @ np.linalg.inv(Sig)
    assert rho_align_sigma(W_flat, Sig) < -0.999
    # isotropic random kernel: near 0 in expectation (|index| small for a wide sample)
    vals = [rho_align_sigma(rng.standard_normal((m, d)), Sig) for _ in range(200)]
    assert abs(np.mean(vals)) < 0.1


def test_two_sided_corollary_holds_on_a_small_network():
    """k*_ortho(tau'') <= k*_out(tau) <= k*_ortho(tau'), tau' = k^2 tau/(1-tau+k^2 tau), tau'' = tau/(k^2(1-tau)+tau)."""
    import layerspec
    torch.manual_seed(3)
    net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 1), nn.ReLU(), nn.Conv2d(32, 8, 3, padding=1))
    x = torch.randn(256, 3, 8, 8, generator=torch.Generator().manual_seed(4))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=32)
    dec = layerspec.decompose(net, loader, device="cpu", positions=16)
    for tau in (0.9, 0.95, 0.99, 0.999):
        assert dec.table[f"bound_ok_{tau}"].all() and dec.table[f"bound_lo_ok_{tau}"].all()
        assert (dec.table[f"bound_lo_{tau}"] <= dec.table[f"out_{tau}"] + 1e-9).all()
    c = dec.check()
    assert c["corollary_lo"][0.95] == (3, 3)
