"""--widths / --save-checkpoints for the width-from-the-ruler experiment (analysis-plan 9.5)."""
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
