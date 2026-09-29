"""scripts/measure_residual.py (Th9, analysis-plan 9.21): the block metrics on synthetic streams, the taps on a
timm ResNet, and the verdicts."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_residual import TRAINED, block_metrics, canonical_correlations, judge, k_star, measure   # noqa: E402
from tests.test_train_imagenet_a17 import _fake_flat                                                            # noqa: E402


def _joint(x, f):
    j = np.concatenate([x, f], 1); j = j - j.mean(0)
    z = np.maximum(x + f, 0); z = z - z.mean(0)
    return j.T @ j / (len(j) - 1), z.T @ z / (len(z) - 1)


def test_floors_hold_and_are_approached():
    rng = np.random.default_rng(0)
    C, n, r = 16, 4000, 4
    x = rng.standard_normal((n, C)) * np.linspace(2.0, 0.3, C)
    # a branch of rank r that cancels the stream's four largest directions exactly: floor (a) is attained
    f = np.zeros_like(x); f[:, :r] = -x[:, :r]
    m, s = block_metrics(*_joint(x, f), r)
    assert m["floor_a_min"] >= -1e-9 and m["floor_b_min"] >= -1e-9
    # coordinates are not the sample eigenvectors, so the floor is approached (interlacing), not met exactly
    assert np.all(s["y"][: C - r] >= s["x"][r:] - 1e-12) and np.allclose(s["y"][: C - r], s["x"][r:], rtol=0.01)
    assert m["rho_max"] > 0.999999
    assert m["k95_y"] < m["k95_x"]
    # an independent branch: no cancellation, the sum dominates the stream eigenvalue by eigenvalue up to sampling
    g = rng.standard_normal((n, C)) * 0.5
    m, s = block_metrics(*_joint(x, g), C)
    assert m["rho_max"] < 0.3 and m["floor_b_min"] >= -1e-9 and m["retention_min"] > 0.9 and m["k95_y"] >= m["k95_x"]
    # a partly anti-correlated full-rank branch: floor (b) holds with its measured rho
    h = -0.6 * x + 0.3 * rng.standard_normal((n, C))
    m, _ = block_metrics(*_joint(x, h), C)
    assert 0.5 < m["rho_max"] < 1 and m["floor_b_min"] >= -1e-9 and m["retention_at_k"] < 1


def test_canonical_correlations_and_width():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5000, 6)); f = np.concatenate([x[:, :2], rng.standard_normal((5000, 4))], 1)
    j = np.concatenate([x, f], 1); j -= j.mean(0); c = j.T @ j / (len(j) - 1)
    cc = np.sort(canonical_correlations(c[:6, :6], c[6:, 6:], c[:6, 6:]))[::-1]
    assert np.allclose(cc[:2], 1, atol=1e-6) and cc[2] < 0.1
    assert k_star(np.array([0.9, 0.06, 0.04]), 0.95) == 2 and k_star(np.ones(10), 0.95) == 10


def test_taps_on_a_random_resnet(tmp_path):
    from types import SimpleNamespace
    flat = str(tmp_path / "flat"); _fake_flat(flat, n=6)
    a = SimpleNamespace(data=flat, device="cpu", positions=8, batch_size=3, workers=0, limit=None)
    rows, spectra = measure("resnet18.tv_in1k", 1, a)
    assert len(rows) == 8 and rows.C.tolist() == [64, 64, 128, 128, 256, 256, 512, 512] and (rows.r == rows.C).all()
    assert rows.downsample.tolist() == [False, False, True, False, True, False, True, False]
    assert (rows.n == 6 * 8).all() and not rows.ok.any()                       # 48 samples: nothing passes the gate
    assert rows.floor_a_min.min() >= -1e-6 and rows.floor_b_min.min() >= -1e-6   # algebraic, whatever the sample size
    assert set(k.split("::")[1] for k in spectra) == {"x", "f", "y", "z", "cc"}


def _blocks(path, n, k_y, rho, ret, floor=0.0):
    pd.DataFrame([{"block": f"layer{i}", "C": 64, "r": 64, "k95_x": 20, "k95_y": k_y, "k95_z": 22, "rho_max": rho, "rho_median": 0.2,
                   "retention_at_k": ret, "floor_a_min": floor, "floor_b_min": floor, "downsample": i % 4 == 0, "ok": True}
                  for i in range(n)]).to_csv(path, index=False)


def test_verdicts(tmp_path, capsys):
    out = str(tmp_path)
    for w in TRAINED:
        _blocks(os.path.join(out, f"timm_{w}_blocks.csv"), 8, 24, 0.97, 0.9)
    _blocks(os.path.join(out, "timm_resnet18.tv_in1k_random_s1_blocks.csv"), 8, 10, 0.1, 0.1)   # random networks are not judged
    assert judge(out) == {"P9.60": True, "P9.61": True, "P9.62": True}
    _blocks(os.path.join(out, f"timm_{TRAINED[0]}_blocks.csv"), 8, 12, 0.3, 0.9, floor=-0.01)
    v = judge(out)
    assert not v["P9.60"] and v["P9.61"] and not v["P9.62"]
    assert "FAILS" in capsys.readouterr().out
    assert judge(str(tmp_path / "none")) == {}
