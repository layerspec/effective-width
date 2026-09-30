"""measure_gradgram.py (theory-chain B7): sampled activations and gradients share positions, the Grams have the right
shape, the widths are bounded by C, and the judge returns verdicts on synthetic rows."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import measure_gradgram as G   # noqa: E402
from scripts.reproduce_garg import build_vgg16_bn_cifar   # noqa: E402


def _loader(n=64, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g); y = torch.randint(0, k, (n,), generator=g)
    return torch.utils.data.DataLoader(list(zip(x, y)), batch_size=32, shuffle=False)


def test_sampling_uses_the_same_positions_for_activation_and_gradient():
    gen = torch.Generator().manual_seed(3)
    t = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    st = gen.get_state(); a = G._sample(t, gen); gen.set_state(st); b = G._sample(t, gen)
    assert a.shape == (2 * G.POSITIONS, 3) and torch.equal(a, b)


def test_grams_have_one_matrix_per_convolution_and_widths_within_C():
    torch.manual_seed(0)
    model = build_vgg16_bn_cifar(5, "small")
    sig, gam = G.gradient_grams(model, _loader(), "cpu")
    assert len(sig) == 13 and len(gam) == 13
    for (n, m) in G.conv_layers(model):
        C = m.out_channels
        assert sig[n].shape == (C, C) and gam[n].shape == (C, C)
        assert torch.allclose(sig[n], sig[n].T) and torch.allclose(gam[n], gam[n].T)
        es = np.linalg.eigvalsh(sig[n].numpy()); eg = np.linalg.eigvalsh(gam[n].numpy())
        assert es.min() > -1e-6 * max(es.max(), 1e-30) and eg.min() > -1e-6 * max(eg.max(), 1e-30)
        assert 1 <= G.k_star(es, 0.999) <= C and 1 <= G.k_star(eg, 0.999) <= C


def test_judge_verdicts_on_synthetic_rows(tmp_path, capsys):
    rows = []
    for arm, (_, _, _, K) in G.ARMS.items():
        for s in (0, 1, 2):
            for i in range(13):
                g = 60 if arm == "c10_pc640_e780" else 100
                rows.append({"arm": arm, "seed": s, "index": i, "layer": f"f{i}", "C": 512, "K": K, "n": 1,
                             "k0.95_out": g // 2, "k0.95_grad": g, "k0.999_out": g + 50, "k0.999_grad": g + 300})
    os.makedirs(tmp_path / "gradgram"); pd.DataFrame(rows).to_csv(tmp_path / "gradgram" / "gradgram.csv", index=False)
    a = type("A", (), dict(out=str(tmp_path / "gradgram")))()
    v = G.judge(a)
    assert v["P-b7.3"] is True
    assert "P-b7.2" in v and "Spearman" in capsys.readouterr().out
