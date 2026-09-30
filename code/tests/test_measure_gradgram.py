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
    assert "P-b7.5" not in v                                            # no prediction columns / spectra in this table


def test_judge_p_b7_5_and_6_on_synthetic_spectra(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for arm, (_, _, _, K) in G.ARMS.items():
        for s in (0, 1, 2):
            hard = 0.02 * K
            eg = np.sort(rng.exponential(1.0, 512) * np.linspace(3, 0.01, 512))[::-1]
            np.savez(tmp_path / f"{arm}_s{s}_spectra.npz", **{"f12::grad": eg, "last::pred": eg * 1.1, "el2n": np.zeros(3)})
            for i in range(13):
                row = {"arm": arm, "seed": s, "index": i, "layer": f"f{i}", "C": 512, "K": K, "n": 1,
                       "k0.95_out": 40, "k0.95_grad": int(20 * K ** 0.5), "k0.999_out": 60, "k0.999_grad": 300}
                if i == 12:
                    row.update({"k0.95_pred": int(20 * K ** 0.5) + 1, "k0.999_pred": 200, "el2n_frac_gt_0.1": hard, "el2n_mean": hard,
                                "mask_on": 0.3, "mask_distinct": 1000})
                rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "gradgram.csv", index=False)
    v = G.judge(type("A", (), dict(out=str(tmp_path)))())
    assert v["P-b7.5"] is True and v["P-b7.6"] is True


def test_extras_give_el2n_masks_and_a_prediction_gram_for_the_last_layer():
    torch.manual_seed(0)
    model = build_vgg16_bn_cifar(5, "small")
    extras = {}
    sig, gam = G.gradient_grams(model, _loader(n=40), "cpu", extras=extras)
    assert len(extras["el2n"]) == 40 and all(0 <= v <= 2 ** 0.5 + 1e-9 for v in extras["el2n"])
    assert 0 < extras["mask_on"] / extras["n_pos"] < 1 and 1 <= len(extras["masks"]) <= 40 * G.POSITIONS
    P = extras["pred"].moment()
    assert P.shape == (512, 512) and torch.allclose(P, P.T)
    # a network whose last BN has gamma = 0 predicts a zero Gram (no gradient reaches the convolution output)
    with torch.no_grad():
        [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)][-1].weight.zero_()
    extras2 = {}
    G.gradient_grams(model, _loader(n=40), "cpu", extras=extras2)
    assert extras2["pred"].moment().abs().max() == 0
