"""checkpoint_analysis.py section for A29 (plan 9.18): our own ImageNet-1k ResNet-18 weights against rebuilt random
networks, the six public ResNet-18 weights and the basic [6,6,6,6] network of A17 (P9.48, P9.49)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checkpoint_analysis import section_a29   # noqa: E402
from tests.test_a17_analysis import _public, _traj    # noqa: E402

L = [f"layer{i}.conv" for i in range(20)]
INIT = np.linspace(0.6, 0.2, 20)


def _setup(res, final_fn, basic_seeds=(0, 1)):
    rng = np.random.default_rng(0)
    for s in (0, 1):
        _traj(os.path.join(res, "a29", f"resnet18_s{s}", "trajectory_layers.csv"), L,
              {0: INIT + rng.normal(0, 0.005, 20), 90: final_fn(rng)}, acc=69.0)
    for s in basic_seeds:                                     # A17 basic [6,6,6,6] reorders
        _traj(os.path.join(res, "a17", f"basic6666_s{s}", "trajectory_layers.csv"), L,
              {0: INIT + rng.normal(0, 0.005, 20), 90: INIT[::-1] + 0.2 + rng.normal(0, 0.01, 20)})
    os.makedirs(os.path.join(res, "local6400", "archs", "trained"))
    for name in ("timm_resnet18.tv_in1k", "timm_resnet18.a1_in1k"):
        _public(os.path.join(res, "local6400", "archs", "trained", f"{name}_layers.csv"), L, INIT + 0.2 + rng.normal(0, 0.01, 20))


def test_order_kept_holds_both(tmp_path, capsys):
    _setup(str(tmp_path), lambda rng: INIT + 0.2 + rng.normal(0, 0.01, 20))
    v = section_a29(str(tmp_path))
    out = capsys.readouterr().out
    assert v == {"P9.48": True, "P9.49": True}
    assert "keeps the" in out and "public ResNet-18" in out


def test_reordered_network_fails_and_returns_to_the_dataset(tmp_path, capsys):
    _setup(str(tmp_path), lambda rng: INIT[::-1] + 0.2 + rng.normal(0, 0.01, 20))
    v = section_a29(str(tmp_path))
    assert v == {"P9.48": False, "P9.49": False}
    assert "attribution returns to the dataset" in capsys.readouterr().out


def test_p949_is_provisional_until_both_a17_basic_seeds_are_in(tmp_path, capsys):
    _setup(str(tmp_path), lambda rng: INIT + 0.2 + rng.normal(0, 0.01, 20), basic_seeds=(0,))
    v = section_a29(str(tmp_path))
    assert v == {"P9.48": True}
    assert "provisional" in capsys.readouterr().out


def test_nothing_measured_prints_nothing(tmp_path, capsys):
    assert section_a29(str(tmp_path)) is None
    assert capsys.readouterr().out == ""
