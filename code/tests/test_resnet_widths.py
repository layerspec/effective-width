"""Width-parameterised ResNet-18 (analysis-plan 9.11) and its width rule."""
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torchvision

from scripts.resnet_widths import (RESNET18_WIDTHS, build_resnet18_widths, n_params,
                                   uniform_widths_matching, widths_from_profile, stage_widths)


def conv_widths(m):
    return [c.out_channels for c in m.modules() if isinstance(c, nn.Conv2d)]


def test_default_widths_are_torchvision_resnet18():
    torch.manual_seed(0); a = torchvision.models.resnet18(weights=None)
    torch.manual_seed(0); b = build_resnet18_widths(RESNET18_WIDTHS, 1000, "imagenet")
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(sa[k].shape == sb[k].shape for k in sa)
    # (bit-identical init is not required: the blocks draw their conv2 after torchvision's order)
    assert n_params(RESNET18_WIDTHS) == sum(p.numel() for p in a.parameters()) == 11_689_512
    b.load_state_dict(sa)                                        # keys and shapes interchangeable
    x = torch.randn(2, 3, 64, 64)
    a.eval(); b.eval()
    assert torch.allclose(a(x), b(x))


def test_custom_widths_land_on_every_conv_and_the_head():
    w = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    m = build_resnet18_widths(w, 10, "cifar")
    # stem, then per stage: conv1(mid) conv2(out) [downsample(out)] for each block
    expect = [8]
    c_in = 8
    for s in range(4):
        out, mids = stage_widths(w, s)
        for b, mid in enumerate(mids):
            expect += [mid, out]
            if (s > 0 and b == 0) or c_in != out:
                expect.append(out)
            c_in = out
    assert conv_widths(m) == expect
    assert m.fc.in_features == 18 and m.fc.out_features == 10
    assert tuple(m(torch.randn(2, 3, 32, 32)).shape) == (2, 10)


def test_cifar_stem_keeps_resolution():
    m = build_resnet18_widths(RESNET18_WIDTHS, 10, "cifar")
    assert isinstance(m.maxpool, nn.Identity) and m.conv1.stride == (1, 1) and m.conv1.kernel_size == (3, 3)
    feats = m.layer4(m.layer3(m.layer2(m.layer1(m.relu(m.bn1(m.conv1(torch.randn(1, 3, 32, 32))))))))
    assert tuple(feats.shape[-2:]) == (4, 4)


def test_bad_widths_rejected():
    with pytest.raises(ValueError):
        build_resnet18_widths([64] * 12)
    with pytest.raises(ValueError):
        build_resnet18_widths([64] * 13, stem="tiny")


def test_uniform_matching_hits_the_parameter_target():
    target = n_params(RESNET18_WIDTHS, 10, "cifar") // 3
    w, n, f = uniform_widths_matching(target, num_classes=10, stem="cifar")
    assert abs(n - target) <= 0.02 * target
    assert 0 < f < 1 and min(w) >= 8


def _fake_table():
    rows = [("relu", "act", 0, 60.2, True)]
    ks = {1: ([30.0, 40.0], [31.5, 35.0]), 2: ([100.0, 90.0], [70.0, 60.0]),
          3: ([200.0, 210.0], [150.0, 3.0]), 4: ([400.0, 380.0], [300.0, 310.0])}
    for s, (outs, mids) in ks.items():
        for b in range(2):
            rows.append((f"layer{s}.{b}.relu", "act", 0, mids[b], True))
            rows.append((f"layer{s}.{b}.relu", "act", 1, outs[b], True))
            rows.append((f"layer{s}.{b}.conv1", "conv", 0, 999.0, True))   # must be ignored
    return pd.DataFrame(rows, columns=["layer", "kind", "call_index", "k_star_0.999", "ok"])


def test_width_rule_stage_max_block_mid_floor():
    r = widths_from_profile(_fake_table())
    assert r["widths"] == [61, 40, 32, 35, 100, 70, 60, 210, 150, 8, 400, 300, 310]
    assert r["sites"]["stage3"]["block_out"] == [200.0, 210.0]


def test_width_rule_refuses_gated_layers():
    t = _fake_table()
    t.loc[(t.layer == "layer4.1.relu") & (t.call_index == 1), "ok"] = False
    with pytest.raises(ValueError):
        widths_from_profile(t)
