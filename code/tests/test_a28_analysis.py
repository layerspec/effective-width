"""checkpoint_analysis.py section for A28 (plan 9.17): zones by same-seed T-R, P9.42-P9.47 and the registered
consequences, from results/a28/<cell>_s<k>/ and the A17 ResNet-50 runs."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checkpoint_analysis import a28_zone, section_a28   # noqa: E402
from tests.test_a17_analysis import _traj                        # noqa: E402

L = [f"layer{i}.conv" for i in range(20)]
INIT = np.linspace(0.6, 0.2, 20)                 # falls with depth
REORDERED = INIT[::-1] + 0.2                     # rises with depth: T-R = -1
HALFWAY = np.abs(np.arange(20) - 9.5) / 40 + 0.3   # a valley: T-R and rho_depth near 0
BETWEEN = 0.2 * INIT + HALFWAY                     # T-R +0.27: neither zone


def _cell(res, name, final, seeds=(0, 1), secondary=True, last=90):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    for s in seeds:
        d = os.path.join(res, "a28", f"{name}_s{s}")
        prof = {0: INIT + rng.normal(0, 0.002, 20), last: final + rng.normal(0, 0.002, 20)}
        _traj(os.path.join(d, "trajectory_layers.csv"), L, prof)
        if secondary:
            _traj(os.path.join(d, "secondary_layers.csv"), L, prof)


def _base(res, seeds=(0, 1)):
    rng = np.random.default_rng(7)
    for s in seeds:
        _traj(os.path.join(res, "a17", f"resnet50_s{s}", "trajectory_layers.csv"), L,
              {0: INIT + rng.normal(0, 0.002, 20), 90: REORDERED + rng.normal(0, 0.002, 20)})


def test_zones():
    assert a28_zone([-0.25, 0.10]) == "reorders" and a28_zone([0.40, 0.9]) == "preserves"
    assert a28_zone([0.05, 0.45]) == "intermediate" and a28_zone([0.2, 0.3]) == "intermediate" and a28_zone([]) == "n/a"


def test_class_count_outcome(tmp_path, capsys):
    res = str(tmp_path); _base(res)
    _cell(res, "k1000_64", REORDERED); _cell(res, "k100_224", HALFWAY); _cell(res, "k100_64", HALFWAY)
    _cell(res, "k100_64_long", HALFWAY, last=900); _cell(res, "k1000_n130_224", REORDERED, secondary=False)
    v = section_a28(res)
    out = capsys.readouterr().out
    assert v == {"P9.42": True, "P9.43": True, "P9.44": True, "P9.45": True, "P9.46": True, "P9.47": True}
    assert "the number of classes" in out and "ZONES DIFFER" not in out and "k1000_224 (A17)" in out


def test_nothing_restores_the_order(tmp_path, capsys):
    res = str(tmp_path); _base(res)
    for c in ("k1000_64", "k100_224", "k100_64"):
        _cell(res, c, REORDERED)
    _cell(res, "k100_64_long", REORDERED, last=900); _cell(res, "k1000_n130_224", REORDERED, secondary=False)
    v = section_a28(res)
    assert v["P9.42"] and not v["P9.43"] and not v["P9.44"] and v["P9.45"] and v["P9.46"] and v["P9.47"]
    assert "neither a hundred classes" in capsys.readouterr().out


def test_resolution_and_image_count_outcomes(tmp_path, capsys):
    res = str(tmp_path); _base(res)
    _cell(res, "k1000_64", INIT + 0.2); _cell(res, "k100_224", HALFWAY); _cell(res, "k100_64", INIT + 0.2)
    _cell(res, "k100_64_long", REORDERED, last=900); _cell(res, "k1000_n130_224", BETWEEN, secondary=False)
    v = section_a28(res)
    out = capsys.readouterr().out
    assert not v["P9.42"] and v["P9.43"] and not v["P9.46"] and not v["P9.47"]
    assert "input resolution is a factor" in out and "a smaller task" in out


def test_one_seed_is_not_judged(tmp_path, capsys):
    res = str(tmp_path); _base(res)
    _cell(res, "k100_224", HALFWAY, seeds=(0,))
    v = section_a28(res)
    out = capsys.readouterr().out
    assert v == {} and "not judged yet" in out and "Registered consequence" not in out
    assert section_a28(str(tmp_path / "empty")) is None
