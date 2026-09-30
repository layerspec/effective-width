"""checkpoint_analysis.py section for A31 (plan 9.26): zones by same-seed T-R on the standard subset, P9.80-P9.84
and the registered consequences, from results/a31/<cell>_s<k>/, the A17 ResNet-50 runs and A28's hundred-class cell."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checkpoint_analysis import section_a31                                      # noqa: E402
from tests.test_a17_analysis import _traj                                                 # noqa: E402
from tests.test_a28_analysis import BETWEEN, INIT, L, REORDERED, _base                    # noqa: E402

PRESERVED = INIT + 0.15                          # same order, higher level


def _cell(res, name, final, seeds=(0, 1), last=90):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    for s in seeds:
        d = os.path.join(res, "a31", f"{name}_s{s}")
        prof = {0: INIT + rng.normal(0, 0.002, 20), last: final + rng.normal(0, 0.002, 20)}
        _traj(os.path.join(d, "trajectory_layers.csv"), L, prof)
        _traj(os.path.join(d, "secondary_layers.csv"), L[:12], {0: prof[0][:12], last: prof[last][:12]})


def _a28_hundred(res):
    rng = np.random.default_rng(11)
    for s in (0, 1):
        d = os.path.join(res, "a28", f"k100_224_s{s}")
        prof = {0: INIT + rng.normal(0, 0.002, 20), 90: REORDERED + rng.normal(0, 0.002, 20)}
        _traj(os.path.join(d, "trajectory_layers.csv"), L, prof)
        _traj(os.path.join(d, "secondary_layers.csv"), L, prof)


def test_registered_branch_class_count(tmp_path, capsys):
    res = str(tmp_path)
    _base(res); _a28_hundred(res)
    _cell(res, "k10", PRESERVED); _cell(res, "k20", BETWEEN); _cell(res, "k50", REORDERED); _cell(res, "k100", REORDERED)
    V = section_a31(res)
    out = capsys.readouterr().out
    assert V == {"P9.80": True, "P9.81": True, "P9.82": True, "P9.83": True, "P9.84": True}
    assert "ten classes keep the order of initialisation and a hundred lose it" in out
    assert "own held-out images (12, 12 gate-passing layers)" in out


def test_branch_ten_classes_reorder_too(tmp_path, capsys):
    res = str(tmp_path)
    _base(res); _a28_hundred(res)
    for c in ("k10", "k20", "k50", "k100"):
        _cell(res, c, REORDERED)
    V = section_a31(res)
    out = capsys.readouterr().out
    assert V["P9.81"] is False and V["P9.82"] is True and V["P9.80"] is False and V["P9.84"] is False
    assert "the class count is excluded" in out and "not monotone" in out


def test_partial_data_is_not_judged(tmp_path, capsys):
    res = str(tmp_path)
    _base(res)
    _cell(res, "k10", PRESERVED, seeds=(0,))
    V = section_a31(res)
    out = capsys.readouterr().out
    assert V == {} and "not judged yet" in out and "Registered consequence" not in out
    assert section_a31(str(tmp_path / "empty")) is None
