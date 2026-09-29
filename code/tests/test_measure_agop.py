"""scripts/measure_agop.py (A32, analysis-plan 9.22): the trace formulas, the gradient probe against the exact
Jacobian on a small network, the layer statistics on constructed cases, and the verdicts."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.measure_agop as ma                                      # noqa: E402
from tests.test_train_imagenet_a17 import _fake_flat                   # noqa: E402


def test_trace_formulas_match_the_explicit_matrices():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((6, 20)); G = rng.standard_normal((6, 6)); G = G @ G.T
    A, B = W.T @ W, W.T @ G @ W
    p, c = ma.gram_correlations(W, G)
    assert abs(p - np.corrcoef(A.ravel(), B.ravel())[0, 1]) < 1e-10
    assert abs(c - np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B))) < 1e-10
    p, c = ma.gram_correlations(W, np.eye(6))                            # Gamma = I: the two matrices are the same
    assert abs(p - 1) < 1e-12 and abs(c - 1) < 1e-12


def test_partial_correlation():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(4000); x = z + 0.1 * rng.standard_normal(4000); y = z + 0.1 * rng.standard_normal(4000)
    assert ma.spearman(x, y) > 0.9 and abs(ma.partial_spearman(x, y, z)) < 0.1      # x and y related only through z
    w = rng.standard_normal(4000)
    assert ma.partial_spearman(x, x + 0.1 * w, w) > 0.9


def _layer(align: bool, white_gamma: bool, rng):
    C, d = 12, 30
    U, _ = np.linalg.qr(rng.standard_normal((C, C))); V, _ = np.linalg.qr(rng.standard_normal((d, d)))
    s = np.linspace(3.0, 0.5, C)
    W = U @ np.diag(s) @ V[:, :C].T
    lam = np.concatenate([np.linspace(5.0, 0.2, C) if align else rng.permutation(np.linspace(5.0, 0.2, C)), np.full(d - C, 0.1)])
    Sigma = V @ np.diag(lam) @ V.T
    g = np.ones(C) if white_gamma else np.linspace(0.2, 4.0, C)         # low gradient energy on the high-gain directions
    return W, Sigma, U @ np.diag(g) @ U.T


def test_layer_statistics_separate_the_two_regularities():
    rng = np.random.default_rng(2)
    row, arr = ma.layer_stats(*_layer(True, True, rng), rng)             # aligned, ansatz exact
    assert row["s2_rho_align"] > 0.99 and abs(row["s1_pearson"] - 1) < 1e-9 and row["s3_gamma_cv"] < 1e-9 and row["r"] == 12
    assert set(arr) == {"gain", "sigma", "gamma", "lam", "gain_q"} and row["rho_align_sigma_basis"] > 0.5
    row, _ = ma.layer_stats(*_layer(True, False, rng), rng)              # aligned, gradient energy anti-ordered
    assert row["s2_rho_align"] > 0.99 and row["s3_rho_gain_gamma"] < -0.99 and row["s1_pearson"] < 1
    row, _ = ma.layer_stats(*_layer(False, True, rng), rng)              # ansatz exact, no alignment
    assert abs(row["s2_rho_align"]) < 0.8 and abs(row["s1_pearson"] - 1) < 1e-9


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 5, 3, padding=1); self.c2 = nn.Conv2d(5, 4, 3, padding=1); self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(torch.relu(self.c2(torch.relu(self.c1(x)))).mean((2, 3)))


def test_probe_estimates_the_full_jacobian_gram():
    torch.manual_seed(0)
    net = Tiny().double().eval()
    x = torch.randn(2, 3, 6, 6, dtype=torch.double)
    exact = 0
    for k in range(3):                                                   # one backward pass per output
        probe = ma.GradProbe(net)
        net(x.clone().requires_grad_(True))[:, k].sum().backward()
        probe.collect(2); probe.remove()
        exact = exact + probe.gamma["c2"]
    est, gen = 0, torch.Generator().manual_seed(0)
    reps = 3000
    for _ in range(reps):
        probe = ma.GradProbe(net)
        out = net(x.clone().requires_grad_(True))
        v = torch.randint(0, 2, out.shape, generator=gen).to(out) * 2 - 1
        (out * v).sum().backward()
        probe.collect(2); probe.remove()
        est = est + probe.gamma["c2"] / reps
    assert np.linalg.norm(est - exact) / np.linalg.norm(exact) < 0.05
    assert probe.n_vectors["c2"] == 2 * 36 and set(probe.gamma) == {"c1", "c2"}


def test_measure_end_to_end(tmp_path, monkeypatch):
    import layerspec.models as _models
    torch.manual_seed(0)
    monkeypatch.setattr(_models, "build", lambda name, pretrained=True, seed=0: (Tiny(), "tiny"))
    flat = str(tmp_path / "flat"); _fake_flat(flat, n=8)
    a = SimpleNamespace(data=flat, device="cpu", positions=16, batch_size=4, workers=0, limit=None)
    rows, arrays = ma.measure("tiny", a)
    assert rows.layer.tolist() == ["c1", "c2"] and rows.d.tolist() == [27, 45] and (rows.n_patches == 8 * 16).all()
    assert not rows.ok.any() and rows.s1_pearson.between(-1, 1).all() and "c2::gamma" in arrays


def _csv(path, n, **kw):
    base = dict(s1_pearson=0.9, s1_null=0.7, s2_rho_align=0.9, s3_rho_gain_gamma=-0.5, s3_gamma_cv=0.5,
                s4_align_given_gamma=0.85, s4_gamma_given_align=-0.2, ok=True)
    base.update(kw)
    pd.DataFrame([dict(base, layer=f"l{i}") for i in range(n)]).to_csv(path, index=False)


def test_verdicts(tmp_path, capsys):
    out = str(tmp_path)
    _csv(os.path.join(out, "vgg16_bn_agop.csv"), 13); _csv(os.path.join(out, "resnet18_agop.csv"), 20)
    _csv(os.path.join(out, "resnet18_random_agop.csv"), 20, s2_rho_align=0.0)
    assert ma.judge(out) == {"P9.63": True, "P9.64": True, "P9.65": True, "P9.66": True}
    assert "not accounted for by the ansatz" in capsys.readouterr().out
    _csv(os.path.join(out, "resnet18_agop.csv"), 20, s4_align_given_gamma=0.1, s1_null=0.2)
    v = ma.judge(out)
    assert not v["P9.66"] and not v["P9.64"] and "shadow of the ansatz" in capsys.readouterr().out
    os.remove(os.path.join(out, "vgg16_bn_agop.csv"))
    assert ma.judge(out) == {}
