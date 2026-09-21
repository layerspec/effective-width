"""--widths / --save-checkpoints for the width-from-the-ruler experiment (analysis-plan 9.5)."""
import pytest
import torch
import torch.nn as nn

from scripts.reproduce_garg import build_vgg16_bn_cifar
from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths, VGG16_WIDTHS


def conv_widths(m):
    return [c.out_channels for c in m.modules() if isinstance(c, nn.Conv2d)]


def test_default_widths_reproduce_the_standard_network():
    torch.manual_seed(0); a = build_vgg16_bn_cifar(10, "small")
    torch.manual_seed(0); b = build_vgg16_bn_cifar_widths(VGG16_WIDTHS, 10)
    assert conv_widths(b) == list(VGG16_WIDTHS) == [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)          # same seed, same init


def test_custom_widths_change_every_layer_and_the_head():
    w = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    m = build_vgg16_bn_cifar_widths(w, 10)
    assert conv_widths(m) == w
    assert m.classifier.in_features == 20 and m.classifier.out_features == 10
    y = m(torch.randn(2, 3, 32, 32))
    assert tuple(y.shape) == (2, 10)


def test_resnet20_default_inner_widths_reproduce_the_standard_network():
    from layerspec.realloc import discover_sites
    from scripts.trajectory_cifar import RESNET20_INNER, build_resnet20_cifar, CifarResNet
    m = build_resnet20_cifar(10)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in CifarResNet(3, 10).parameters())
    sites = discover_sites(m)
    assert [s.width(m) for s in sites] == RESNET20_INNER            # 9 block-internal sites, stem and stream untouched
    assert all(s.producer.endswith("conv1") and s.consumers[0].name.endswith("conv2") for s in sites)


def test_resnet20_custom_inner_widths_and_resume_roundtrip():
    """Shrinking / growing a site, then rebuilding with the recorded widths and loading the state dict,
    is the resume path of trajectory_cifar.py for a controller run."""
    from layerspec.realloc import discover_sites, greedy_cssp, grow, measure, shrink
    from scripts.trajectory_cifar import build_resnet20_cifar, build_with_widths
    torch.manual_seed(0)
    m = build_resnet20_cifar(10).eval()
    sites = discover_sites(m)
    x = torch.randn(4, 3, 32, 32)
    mom = measure(m, sites, [(x,)], "cpu", patch=True)
    grow(m, sites[8], 6, mom[sites[8].name])                        # deepest first
    order, _ = greedy_cssp(mom[sites[0].name].act_cov, 10)
    shrink(m, sites[0], order[:10], mom[sites[0].name])
    widths = [s.width(m) for s in sites]
    assert widths == [10, 16, 16, 32, 32, 32, 64, 64, 70]
    with torch.no_grad():
        y = m(x)
    m2 = build_with_widths("resnet20", widths, 10).eval()
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert torch.allclose(m2(x), y)
    with pytest.raises(ValueError):
        build_with_widths("resnet20", widths[:8], 10)
