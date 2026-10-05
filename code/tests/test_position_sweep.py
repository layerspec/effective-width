"""Tests for scripts/position_sweep.py (plan 9.30 A)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import position_sweep as ps                      # noqa: E402
from scripts.position_collapse_judge import cliff, k_star      # noqa: E402


def test_upsampled_network_has_nine_positions_in_the_last_block():
    m = ps.build(48).eval()
    seen = {}
    mods = dict(m.named_modules())
    hs = [mods[n].register_forward_hook(lambda mod, i, o, n=n: seen.__setitem__(n, o.shape)) for n in ps.BLOCK5]
    with torch.no_grad():
        out = m(torch.randn(2, 3, 32, 32))
    for h in hs:
        h.remove()
    assert out.shape == (2, 10)
    assert all(tuple(seen[n][2:]) == (3, 3) for n in ps.BLOCK5)


def test_same_parameter_names_at_both_resolutions():
    assert list(ps.build(48).state_dict()) == list(ps.build(32).state_dict())


def test_cell_statistics_on_a_collapsed_layer():
    rng = np.random.default_rng(0)
    P, K, C, per = 9, 10, 256, 60
    means = rng.normal(size=(P * K, C))
    cell = np.repeat(np.arange(P * K), per)
    v = means[cell] + 1e-3 * rng.normal(size=(len(cell), C))
    st = ps.cell_statistics(v, cell, P * K)
    assert st["rank_means"] == P * K - 1
    assert st["w_cp"] < 1e-3 and st["w_out"] < 1e-3
    g, ratio = cliff(st["eig"], K)
    assert g == P * K - 1 and ratio > 100
    assert k_star(st["eig"], 0.999) <= P * K - 1


def test_within_cell_variance_inside_the_span_leaves_w_out_small():
    rng = np.random.default_rng(1)
    P, K, C, per = 4, 10, 64, 200
    means = rng.normal(size=(P * K, C))
    centred = means - means.mean(0)
    cell = np.repeat(np.arange(P * K), per)
    noise = rng.normal(size=(len(cell), P * K)) @ centred * 0.2          # within-cell variance, inside the span
    st = ps.cell_statistics(means[cell] + noise, cell, P * K)
    assert st["w_cp"] > 0.05 and st["w_out"] < 1e-6
