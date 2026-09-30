"""measure_stationarity.py (plan 9.25): the algebra of Proposition B6 on a real conv-BN stack, and the statistics."""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import measure_stationarity as S   # noqa: E402
from scripts.reproduce_garg import build_vgg16_bn_cifar   # noqa: E402


def _loader(n=64, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g); y = torch.randint(0, k, (n,), generator=g)
    return torch.utils.data.DataLoader(list(zip(x, y)), batch_size=32, shuffle=False)


def test_gradient_is_orthogonal_to_the_weight_in_training_mode_bn():
    torch.manual_seed(0)
    model = build_vgg16_bn_cifar(5, "small")
    with torch.no_grad():                       # a trained-looking network: non-trivial BN affine parameters
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.mul_(torch.rand_like(m.weight) + 0.5); m.bias.add_(torch.randn_like(m.bias) * 0.1)
    loader = _loader()
    rows = S.gradient_rows(model, S.mean_gradient(model, loader, "cpu", train_mode=True))
    cos = np.array([c for c, _ in rows.values()]); r = np.array([r for _, r in rows.values()])
    assert np.abs(cos).max() < 1e-3                  # Proposition B6: scale invariance under batch statistics
    assert (r >= 0.99).all()                          # so the gradient cannot cancel lambda W
    rows_eval = S.gradient_rows(model, S.mean_gradient(model, loader, "cpu", train_mode=False))
    assert np.abs([c for c, _ in rows_eval.values()]).max() > 1e-3   # running statistics break the invariance


def test_tail_energy_and_within_class_fraction():
    w = torch.randn(8, 3) @ torch.randn(3, 27)                        # rank 3 as a 8 x 27 matrix
    assert S.tail_energy(w.reshape(8, 3, 3, 3), 3) < 1e-10 and S.tail_energy(w.reshape(8, 3, 3, 3), 2) > 0.05
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    clustered = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0], [-1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
    assert S.within_class_fraction(clustered, y) < 1e-10
    yy = torch.arange(3).repeat_interleave(200)                                   # 200 per class, noise dominates
    spread = clustered[yy] + torch.randn(600, 2, generator=torch.Generator().manual_seed(1)).double() * 10
    assert S.within_class_fraction(spread, yy) > 0.9


def test_pooled_features_have_one_row_per_convolution_and_one_vector_per_image():
    model = build_vgg16_bn_cifar(5, "small")
    feats, y = S.pooled_features(model, _loader(n=40), "cpu")
    assert len(feats) == 13 and all(f.shape[0] == 40 for f in feats.values()) and len(y) == 40
    assert feats[list(feats)[-1]].shape[1] == 512


def test_judge_verdicts_on_synthetic_rows(tmp_path, capsys):
    rows = []
    for arm, (_, _, _, K) in S.ARMS.items():
        for s in (0, 1, 2):
            for i in range(13):
                rows.append(dict(arm=arm, seed=s, index=i, layer=f"f{i}", C=512, d=4608, K=K, cos_train=0.001, r_train=1.0,
                                 cos_eval=0.2, r_eval=1.1, tail_K=0.6 if i < 10 else 0.4,
                                 wcv=0.9 - 0.05 * i - (0.1 if arm == "c10_pc640_e780" else 0.0)))
    os.makedirs(tmp_path / "stationarity"); pd.DataFrame(rows).to_csv(tmp_path / "stationarity" / "stationarity.csv", index=False)
    a = type("A", (), dict(out=str(tmp_path / "stationarity"), results=str(tmp_path)))()
    v = S.judge(a)
    assert v == {"P9.76": True, "P9.77": True, "P9.78": True}
    assert "P9.79" not in capsys.readouterr().out          # no two_rulers table in this tmp results dir
