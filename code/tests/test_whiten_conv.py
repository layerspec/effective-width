"""Tests for scripts/whiten_conv.py and scripts/trajectory_whiten.py (plan 9.56, T3 arms)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from scripts.whiten_conv import (WhitenedConv2d, apply_whitening, bake_to_reference, isqrt_eigh,
                                 isqrt_newton_schulz, pair_conv_bn, whitened_modules)
from scripts.trajectory_whiten import ARMS, build_arm
import scripts.trajectory_cifar as tc

torch.manual_seed(0)


def _spd(n, cond=100.0, batch=None):
    g = torch.Generator().manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    lam = torch.logspace(0, -np.log10(cond), n)
    A = Q @ torch.diag(lam) @ Q.T
    return A.expand(batch, n, n).clone() if batch else A


def test_isqrt_newton_schulz_and_eigh():
    A = _spd(12, cond=20.0, batch=3)
    Z = isqrt_newton_schulz(A, iters=25)
    assert torch.allclose(Z @ A @ Z, torch.eye(12).expand_as(A), atol=1e-3)
    Ze = isqrt_eigh(A[0])
    assert torch.allclose(Ze @ A[0] @ Ze, torch.eye(12), atol=1e-5)


def _correlated_input(N, C, H, rng):
    """Spatially and channel-correlated maps: smoothed noise mixed across channels."""
    x = torch.from_numpy(rng.standard_normal((N, C, H, H)).astype("float32"))
    x = nn.functional.avg_pool2d(x, 3, 1, 1)
    M = torch.from_numpy((np.eye(C) + 0.6 * rng.standard_normal((C, C))).astype("float32"))
    return torch.einsum("nchw,dc->ndhw", x, M)


def _out_cov(y):
    z = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])
    z = z - z.mean(0)
    return z.T @ z / z.shape[0]


def test_full_whitening_with_polar_kernel_gives_isotropic_output():
    rng = np.random.default_rng(0)
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
    m = WhitenedConv2d(conv, mode="full", update_every=1, warmup=0, gain_cap=1e4, orthogonal=True, sampling_stride=1, momentum=1.0)
    x = _correlated_input(64, 8, 12, rng)
    m.train()
    m(x)                                   # one step: running_cov = batch cov, deconv refreshed
    m.eval()
    S = _out_cov(m(x))
    S = S / S.diagonal().mean()
    assert torch.allclose(S, torch.eye(16), atol=0.08)          # Sigma_out ~ I (T3.1): d = 72 >= C = 16
    e = torch.linalg.eigvalsh(S).flip(0)
    assert int(np.searchsorted(np.cumsum(e.numpy()) / e.sum().item(), 0.95) + 1) == int(np.ceil(0.95 * 16))


def test_nd_block_whitening_matches_counterfactual_formula():
    rng = np.random.default_rng(1)
    conv = nn.Conv2d(8, 6, 3, padding=1, bias=True)
    m = WhitenedConv2d(conv, mode="nd", block=4, eps=1e-6, n_iter=30, momentum=1.0, sampling_stride=1)
    x = _correlated_input(32, 8, 10, rng)
    m.train()
    y_train = m(x)
    m.eval()
    y_eval = m(x)
    assert torch.allclose(y_train, y_eval, atol=1e-4)           # momentum 1: running = batch statistics
    # prediction W D_B Sigma_p D_B W^T from the patches (T3.2 (iv))
    X = m._patches(x)
    Xc = X - X.mean(0)
    Sigma = Xc.T @ Xc / X.shape[0]
    D = torch.block_diag(*[m.running_deconv[b] for b in range(m.nb)])
    W = conv.weight.reshape(6, -1)
    pred = W @ D @ Sigma @ D @ W.T
    assert torch.allclose(_out_cov(y_eval), pred, atol=float(1e-3 * pred.diagonal().mean()))
    # within-block correlations removed: block-diagonal of D Sigma D is identity
    DSD = D @ Sigma @ D
    for b in range(m.nb):
        f = m.block * 9
        assert torch.allclose(DSD[b * f:(b + 1) * f, b * f:(b + 1) * f], torch.eye(f), atol=2e-2)


def test_gradients_flow_in_both_modes():
    for mode, kw in (("nd", dict(block=4)), ("full", dict(orthogonal=True, update_every=1, warmup=0))):
        conv = nn.Conv2d(8, 6, 3, padding=1, bias=False)
        m = WhitenedConv2d(conv, mode=mode, sampling_stride=1, **kw).train()
        x = torch.randn(4, 8, 8, 8)
        m(x).square().mean().backward()
        assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()


@pytest.mark.parametrize("arch,n_conv", [("vgg16_bn", 13), ("resnet_basic52", 52)])
def test_pairing_and_apply(arch, n_conv):
    model = tc.ARCHS[arch](10)
    pairs = pair_conv_bn(model, torch.zeros(1, 3, 32, 32))
    assert len(pairs) == n_conv and len(set(pairs.values())) == n_conv
    nd = apply_whitening(tc.ARCHS[arch](10), mode="nd", remove_bn=True, block=64, sampling_stride=3)
    assert len(whitened_modules(nd)) == n_conv
    assert sum(isinstance(m, nn.BatchNorm2d) for m in nd.modules()) == 0
    assert sum(isinstance(m, nn.Conv2d) for m in nd.modules()) == n_conv      # subclass still counts as Conv2d
    wo = build_arm(tc.ARCHS[arch], "wo", 10)
    assert sum(isinstance(m, nn.BatchNorm2d) for m in wo.modules()) == n_conv
    assert all(m.oni is not None and m.mode == "full" for m in whitened_modules(wo).values())


@pytest.mark.parametrize("arch", ["vgg16_bn", "resnet_basic52"])
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_bake_reproduces_eval_forward(arch, arm):
    torch.manual_seed(3)
    model = build_arm(tc.ARCHS[arch], arm, 10)
    model.train()
    x = torch.randn(8, 3, 32, 32)
    for _ in range(2):
        model(x)                                   # populate running statistics
    model.eval()
    with torch.no_grad():
        y = model(x)
        ref = bake_to_reference(model, tc.ARCHS[arch](10).eval())
        yb = ref(x)
    assert sum(isinstance(m, WhitenedConv2d) for m in ref.modules()) == 0
    assert torch.allclose(y, yb, atol=float(2e-3 * y.abs().max()))
    sd = ref.state_dict()
    assert set(sd) == set(tc.ARCHS[arch](10).state_dict())        # plain keys: decompose_check can read it


def test_state_dict_roundtrip_for_resume():
    m = build_arm(tc.ARCHS["vgg16_bn"], "nd", 10).train()
    m(torch.randn(2, 3, 32, 32))
    m2 = build_arm(tc.ARCHS["vgg16_bn"], "nd", 10)
    m2.load_state_dict(m.state_dict())
    for (n1, b1), (n2, b2) in zip(m.named_buffers(), m2.named_buffers()):
        assert n1 == n2 and torch.equal(b1, b2)
