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
    assert "rho_align" in dec.table.columns and list(dec.table.columns)[-1] == "rho_align"
    v = dec.table.rho_align.dropna()
    assert len(v) == 2 and ((v >= -1) & (v <= 1)).all()
    assert "rho_align_median" in dec.check()
