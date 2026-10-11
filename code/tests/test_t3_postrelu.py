"""Tests for scripts/t3_postrelu.py (plan 9.56, P9.130)."""
import torch

import scripts.trajectory_cifar as tc
from scripts.t3_postrelu import postrelu_table, trace_pairs


def test_trace_pairs_counts():
    pairs = trace_pairs(tc.ARCHS["vgg16_bn"](10), torch.zeros(1, 3, 32, 32))
    assert len(pairs) == 13 and all(p[3] == 0 for p in pairs)              # Sequential: one ReLU per conv
    pairs = trace_pairs(tc.ARCHS["resnet_basic52"](10), torch.zeros(1, 3, 32, 32))
    assert len(pairs) == 1 + 24                                            # stem + first ReLU of each block
    assert sum(p[3] == 0 for p in pairs) == 25                             # the post-addition ReLU is call 1, never paired


def test_postrelu_table_on_random_vgg():
    torch.manual_seed(0)
    model = tc.ARCHS["vgg16_bn"](10)
    x = torch.randn(96, 3, 32, 32)
    loader = [(x[i:i + 32], torch.zeros(32, dtype=torch.long)) for i in range(0, 96, 32)]
    df = postrelu_table(model, loader, "cpu", positions=8, seed=0)
    assert len(df) == 13
    assert (df["bn_95"] <= df["ceiling_95"] + 1e-12).all()                 # BN output: rank <= r_max (Lemma T3.0)
    assert (df["act_95"] * df["r_max"] <= df["C"] + 1e-9).all()               # ReLU output: rank can exceed r_max (rule 6), never C
    assert df["ratio_95"].between(0.1, 5.0).all()
