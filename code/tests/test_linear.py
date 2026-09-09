"""The ruler on nn.Linear: (N, D) and token (N, T, D) inputs, r_max = min(in, out),
and the exact identity Sigma_out = W Sigma_in W^T through decompose(include_linear=True)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import layerspec


class TokenMLP(nn.Module):
    """A transformer-style MLP block on token sequences (N, T, D)."""
    def __init__(self, d=16, hidden=40, out=12):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden); self.act = nn.GELU(); self.fc2 = nn.Linear(hidden, out)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def token_loader(n=128, t=8, d=16, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, t, d, generator=g) @ torch.diag(torch.linspace(2.0, 0.2, d))
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=batch)


def test_linear_rows_have_rmax_and_token_positions_are_samples():
    torch.manual_seed(0)
    m = TokenMLP()
    prof = layerspec.profile(m, token_loader(n=640), device="cpu", positions=4, include_activations=True)
    lin = prof.table[prof.table.kind == "linear"].sort_values("depth_index")
    assert list(lin.layer) == ["fc1", "fc2"]
    assert list(lin.r_max) == [16, 12]                      # min(in, out)
    assert list(lin.C) == [40, 12]
    assert list(lin.n_samples) == [640 * 4, 640 * 4]        # 4 tokens per sequence
    assert (lin["k_star_rmax_0.999"] <= 1.0 + 1e-9).all()
    act = prof.table[prof.table.kind == "act"]
    assert len(act) == 1 and int(act.n_samples.iloc[0]) == 640 * 4
    assert lin["ok"].all()                                   # 2560/40 = 64 >= 50
    d = prof.dense(tau=0.95, kinds=("linear",))
    assert list(d.layer) == ["fc1", "fc2"] and prof.dense().empty   # no conv in this model
    assert prof.summary(kinds=("linear",))["L"] == 2


def test_plain_mlp_two_dimensional_input():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(256, 16, generator=torch.Generator().manual_seed(1))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=64)
    prof = layerspec.profile(m, loader, device="cpu")
    lin = prof.table[prof.table.kind == "linear"]
    assert list(lin.r_max) == [8, 4] and list(lin.n_samples) == [256, 256]
    assert lin["ok"].tolist() == [False, True]               # 256/8 = 32 < 50; 256/4 = 64


def test_decompose_linear_identity_is_exact_with_all_tokens():
    torch.manual_seed(0)
    m = TokenMLP()
    loader = token_loader()
    dec = layerspec.decompose(m, loader, device="cpu", positions=8, include_linear=True)   # 8 >= T: every token
    assert list(dec.table.layer) == ["fc1", "fc2"]
    assert list(dec.table.r_max) == [16, 12] and list(dec.table.d) == [16, 40]
    # independent direct measurement of fc1's output covariance over the same tokens
    prof = layerspec.profile(m, loader, device="cpu", positions=8, include_activations=False)
    W = m.fc1.weight.detach().double().numpy()
    lam_direct = np.sort(prof.spectra["fc1::linear#0"])[::-1]
    lam_recomputed = np.sort(np.linalg.eigvalsh(W @ dec.patch_covariance["fc1"] @ W.T))[::-1]
    assert np.allclose(lam_direct, lam_recomputed, rtol=1e-6, atol=1e-9)
    assert dec.table.ostrowski_ok.all()


def test_decompose_ignores_linear_by_default():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Flatten(), nn.Linear(8 * 8 * 8, 4))
    x = torch.randn(64, 3, 8, 8, generator=torch.Generator().manual_seed(2))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=32)
    assert list(layerspec.decompose(m, loader, device="cpu").table.layer) == ["0"]
    assert list(layerspec.decompose(m, loader, device="cpu", include_linear=True).table.layer) == ["0", "2"]
