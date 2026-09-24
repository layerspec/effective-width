"""CPU checks for scripts/oni.py (ONI layer of Huang et al. 2020, submission bar D5) and the
trajectory_oni.py wrapper that trains a CIFAR arm with every dense conv / linear kernel
orthogonalised by construction."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.oni import ONI, apply_oni, bake_oni, oni_modules      # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")


def _erank_ratio(W2d: torch.Tensor) -> float:
    s = torch.linalg.svdvals(W2d.double())
    p = s ** 2 / (s ** 2).sum()
    er = torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum()).item()
    return er / min(W2d.shape)


def test_rows_orthonormal_when_out_le_fanin():
    torch.manual_seed(0)
    conv = nn.Conv2d(128, 64, 3, bias=False)              # n = 64 <= d = 1152
    W = ONI(iters=30)(conv.weight).reshape(64, -1)
    assert torch.allclose(W @ W.T, torch.eye(64), atol=1e-3)


def test_columns_orthonormal_when_out_gt_fanin():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 64, 3, bias=False)                # n = 64 > d = 27: columns, not rows
    W = ONI(iters=30)(conv.weight).reshape(64, -1)
    assert torch.allclose(W.T @ W, torch.eye(27), atol=1e-3)


def test_default_iterations_are_isometric_for_a_spread_spectrum_at_vgg_sizes():
    torch.manual_seed(0)
    conv = nn.Conv2d(512, 512, 3, bias=False)
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    with torch.no_grad():                                      # a trained-looking spectrum: rows scaled 1 .. 0.03
        conv.weight.mul_(torch.logspace(0, -1.5, 512).view(512, 1, 1, 1))
    assert _erank_ratio(conv.weight.reshape(512, -1)) < 0.9   # not isometric (A5 trained kernels: 0.53-0.56)
    assert _erank_ratio(ONI()(conv.weight).reshape(512, -1)) > 0.99


def test_gradient_flows_through_the_orthogonalised_kernel():
    conv = nn.Conv2d(16, 8, 3, bias=False)
    apply_oni(conv)
    conv(torch.randn(2, 16, 8, 8)).sum().backward()
    assert conv.parametrizations.weight.original.grad is not None


def test_apply_and_bake_preserve_function_and_give_plain_keys():
    from scripts.trajectory_cifar import ARCHS
    torch.manual_seed(0)
    m = ARCHS["vgg16_bn"](10).eval()
    n_conv_linear = sum(isinstance(q, (nn.Conv2d, nn.Linear)) for q in m.modules())
    apply_oni(m)
    assert len(oni_modules(m)) == n_conv_linear
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y1 = m(x)
    bake_oni(m)
    with torch.no_grad():
        y2 = m(x)
    assert torch.allclose(y1, y2, atol=1e-5)
    sd = m.state_dict()
    assert not any("parametrizations" in k for k in sd)
    assert set(sd) == set(ARCHS["vgg16_bn"](10).state_dict())


def test_wrapper_trains_and_writes_a_baked_isometric_checkpoint(tmp_path):
    from scripts.trajectory_oni import main
    out = str(tmp_path / "run")
    assert main(["--arch", "vgg16_bn", "--device", "cpu", "--workers", "0", "--data-root", DATA, "--out", out,
                 "--seed", "0", "--epochs", "1", "--max-train-batches", "2", "--max-measure-batches", "2",
                 "--schedule", "0", "1"]) == 0
    pt = os.path.join(out, "vgg16_bn_oni_cifar10_final.pt")
    assert os.path.exists(pt) and not os.path.exists(os.path.join(out, "vgg16_bn_cifar10_final.pt"))
    sd = torch.load(pt, map_location="cpu")
    assert not any("parametrizations" in k for k in sd)
    for k, v in sd.items():                                  # every dense kernel is (near) isometric
        if k.endswith("weight") and v.dim() == 4:
            assert _erank_ratio(v.reshape(v.shape[0], -1)) > 0.99, k
