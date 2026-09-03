"""Correctness checks for the parts that must not be wrong quietly.

The accumulator is the load-bearing piece: if it drifts, every k* is wrong in a
way no plot would reveal.  So it is checked against an exact one-shot
computation, in a regime chosen to punish a naive implementation (large mean
relative to the variance, many small blocks).
"""

import numpy as np

from layerspec.accumulate import CovarianceAccumulator, merge
from layerspec import metrics


def test_accumulator_matches_exact():
    rng = np.random.default_rng(0)
    C, n = 32, 20_000
    A = rng.normal(size=(C, C))
    cov_true = A @ A.T / C
    # Large offset: this is what breaks accumulators that sum uncentred outer
    # products, because the mean^2 term swamps the variance in float32.
    X = rng.multivariate_normal(np.full(C, 1e4), cov_true, size=n)

    acc = CovarianceAccumulator(C)
    for block in np.array_split(X, 400):
        acc.update(block)

    exact = np.cov(X, rowvar=False)
    assert acc.n == n
    assert np.allclose(acc.covariance, exact, rtol=1e-9, atol=1e-9), \
        np.abs(acc.covariance - exact).max()


def test_merge_is_associative_with_single_pass():
    rng = np.random.default_rng(1)
    C = 16
    X = rng.normal(size=(5000, C)) * rng.uniform(0.1, 5.0, size=C)

    whole = CovarianceAccumulator(C)
    whole.update(X)

    a, b = CovarianceAccumulator(C), CovarianceAccumulator(C)
    a.update(X[:1234])
    b.update(X[1234:])
    combined = merge(a, b)

    assert combined.n == whole.n
    assert np.allclose(combined.covariance, whole.covariance, rtol=1e-10, atol=1e-12)


def test_k_star_on_known_spectrum():
    # Exactly 10 units of variance in 10 equal components, then nothing.
    eig = np.array([1.0] * 10 + [0.0] * 90)
    assert metrics.k_star(eig, 0.95) == 10
    assert metrics.k_star(eig, 0.50) == 5
    assert metrics.k_star(eig, 0.10) == 1


def test_participation_ratio_bounds():
    # PR of k equal eigenvalues is exactly k; PR of a single spike is 1.
    for k in (1, 3, 17):
        eig = np.array([1.0] * k + [0.0] * 50)
        assert abs(metrics.participation_ratio(eig) - k) < 1e-9
    spike = np.array([1.0] + [0.0] * 99)
    assert abs(metrics.participation_ratio(spike) - 1.0) < 1e-12


def test_effective_rank_equals_k_for_flat_spectrum():
    for k in (1, 4, 64):
        eig = np.array([1.0] * k)
        assert abs(metrics.effective_rank(eig) - k) < 1e-9


def test_powerlaw_recovers_known_exponent():
    idx = np.arange(1, 2001, dtype=float)
    for alpha in (0.5, 1.0, 1.8):
        eig = idx ** (-alpha)
        est, lo, hi, r2 = metrics.powerlaw_alpha(eig, lo=10)
        assert abs(est - alpha) < 1e-6, (alpha, est)
        assert r2 > 0.999


def test_metrics_ratios_use_nominal_C():
    # The whole paper is about k*/C, so C must be the nominal channel count and
    # not the number of nonzero eigenvalues.
    eig = np.array([1.0] * 8 + [0.0] * 120)
    m = metrics.compute(eig, C=128, n_samples=10_000)
    assert m.k_star["0.95"] == 8
    assert abs(m.k_star_ratio["0.95"] - 8 / 128) < 1e-12
    assert m.C == 128


def test_rmax_matches_the_architecture():
    """r_max is min(fan-in, width) per group, summed over groups.

    Three cases that must come out differently: a dense 3x3, a 1x1 expansion
    (the ResNet bottleneck case, where the bound bites), and a depthwise conv
    (where it must NOT bite, because the blocks have disjoint supports and
    their ranks add to C).
    """
    import torch
    import torch.nn as nn
    from layerspec.hooks import SpectrumProbe

    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),      # fan-in 3*9=27 > 16 -> r_max 16
        nn.Conv2d(16, 64, 1),                # fan-in 16 < 64     -> r_max 16
        nn.Conv2d(64, 64, 3, padding=1, groups=64),   # depthwise -> r_max 64
        nn.Conv2d(64, 8, 1),                 # fan-in 64 > 8      -> r_max 8
    )
    probe = SpectrumProbe(net, positions_per_image=4, pooled=False)
    with torch.no_grad():
        net(torch.randn(2, 3, 8, 8))
    got = [(r.C, r.r_max) for r in probe.records() if r.kind == "conv"]
    probe.remove()
    assert got == [(16, 16), (64, 16), (64, 64), (8, 8)], got


def test_measured_rank_respects_the_bound():
    """The empirical check behind Section V-B of the paper.

    A 1x1 expansion cannot produce more nonzero eigenvalues than it has input
    channels, however much data is pushed through it.  If this fails, either
    r_max is wrong or the accumulator is.
    """
    import torch
    import torch.nn as nn
    from layerspec.hooks import SpectrumProbe

    net = nn.Sequential(nn.Conv2d(8, 32, 1))
    probe = SpectrumProbe(net, positions_per_image=16, pooled=False)
    with torch.no_grad():
        for _ in range(20):
            net(torch.randn(16, 8, 8, 8))
    rec = [r for r in probe.records() if r.kind == "conv"][0]
    probe.remove()

    assert rec.C == 32 and rec.r_max == 8
    assert rec.acc.n >= 50 * rec.C          # well past the reporting gate
    eig = rec.acc.eigenvalues()
    # Everything beyond the 8th eigenvalue is numerically zero.
    assert np.sum(eig > eig[0] * 1e-10) <= rec.r_max
    # ... so k* can never see more than r_max components, at any threshold.
    for tau in (0.9, 0.95, 0.99, 0.999):
        assert metrics.k_star(eig, tau) <= rec.r_max


def test_model_name_parsing_and_tiers():
    """timm names, the _random suffix, tier expansion and file stems."""
    from layerspec import models

    assert models.parse("resnet50") == ("torchvision", "resnet50", False)
    assert models.parse("resnet50_random") == ("torchvision", "resnet50", True)
    assert models.parse("timm:resnet50.a1_in1k") == ("timm", "resnet50.a1_in1k", False)
    assert models.parse("timm:resnet50.a1_in1k_random") == ("timm", "resnet50.a1_in1k", True)
    assert models.file_stem("timm:resnet50.a1_in1k") == "timm_resnet50.a1_in1k"

    names = models.expand_names(["vgg16_bn", "tier1"])
    assert names[0] == "vgg16_bn" and len(names) == 1 + len(models.TIER1)
    assert all(n.startswith("timm:") for n in models.TIER1 + models.TIER2)
    assert len(set(models.TIER1 + models.TIER2)) == len(models.TIER1 + models.TIER2)


def test_conv_on_pooled_input_is_not_a_conv_layer():
    """An SE gate / conv_head acts on a 1x1 map: kind must be conv_gap."""
    import torch
    import torch.nn as nn
    from layerspec.hooks import SpectrumProbe

    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(8, 4, 1),
    )
    probe = SpectrumProbe(net, positions_per_image=4, pooled=False)
    with torch.no_grad():
        net(torch.randn(4, 3, 8, 8))
    kinds = [(r.kind, r.C, r.acc.n) for r in probe.records()]
    probe.remove()
    assert kinds == [("conv", 8, 16), ("conv_gap", 4, 4)], kinds


def test_block_outputs_are_hooked_separately():
    """A residual block's output is recorded as kind='block', with no r_max,
    alongside -- not instead of -- its inner conv layers."""
    import torch
    import torch.nn as nn
    from torchvision.models.resnet import BasicBlock
    from layerspec.hooks import SpectrumProbe

    net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), BasicBlock(16, 16))
    probe = SpectrumProbe(net, positions_per_image=4, pooled=False)
    with torch.no_grad():
        net(torch.randn(2, 3, 8, 8))
    recs = probe.records()
    probe.remove()
    blocks = [r for r in recs if r.kind == "block"]
    assert len(blocks) == 1 and blocks[0].C == 16 and blocks[0].r_max is None
    assert sum(r.kind == "conv" for r in recs) == 3
    # Block output comes after the convs it contains.
    assert blocks[0].depth_index > max(r.depth_index for r in recs if r.kind == "conv")

    off = SpectrumProbe(net, positions_per_image=4, pooled=False, include_blocks=False)
    with torch.no_grad():
        net(torch.randn(2, 3, 8, 8))
    assert not any(r.kind == "block" for r in off.records())
    off.remove()


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


def test_random_init_is_seeded():
    """A `_random` control must be the same weights on every run.

    The second measurement run (2026-09-03) drew a different resnet50_random
    from the first because build() did not seed the initialiser: the manifest
    said seed=0 but the weights were not.  k*(0.999) differed by up to 71.
    """
    import torch
    from layerspec import models

    a, _ = models.build("resnet18_random", seed=0)
    b, _ = models.build("resnet18_random", seed=0)
    c, _ = models.build("resnet18_random", seed=1)
    for (na, pa), (_, pb), (_, pc) in zip(a.named_parameters(), b.named_parameters(),
                                          c.named_parameters()):
        assert torch.equal(pa, pb), na
    assert any(not torch.equal(pa, pc)
               for (_, pa), (_, pc) in zip(a.named_parameters(), c.named_parameters()))


def test_manifest_merges_across_invocations(tmp_path):
    """The GPU script calls run.py once per stage; each call must ADD to the
    manifest, not replace it.  The 2026-09-03 tarball kept only the last
    stage's two entries out of 22."""
    from layerspec.run import write_manifest

    write_manifest(str(tmp_path), [{"model": "a", "x": 1}, {"model": "b", "x": 1}])
    write_manifest(str(tmp_path), [{"model": "b", "x": 2}, {"model": "c", "x": 1}])
    import json
    with open(tmp_path / "manifest.json") as f:
        got = {e["model"]: e["x"] for e in json.load(f)}
    assert got == {"a": 1, "b": 2, "c": 1}
