"""scripts/bplus_controls_judge.py: P9.56-P9.59 (analysis-plan 9.20) on synthetic profiles."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bplus_controls_judge import gamma1, renorm      # noqa: E402
from scripts.measure_controls import GAMMA1, RENORM          # noqa: E402

L = [f"blocks.{i}.conv_pw" for i in range(16)]
BASE = np.linspace(0.2, 0.7, 16)


def _w(path, prof, names=L):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [{"layer": n, "kind": "conv", "depth_index": i, "C": 64, "r_max": 64, "n_over_C_ok": True, "is_depthwise": False,
             "k_star_0.95": 64 * v} for i, (n, v) in enumerate(zip(names, prof))]
    rows.append({"layer": "dw", "kind": "conv", "depth_index": 99, "C": 64, "r_max": 1, "n_over_C_ok": True, "is_depthwise": True, "k_star_0.95": 1})
    pd.DataFrame(rows).to_csv(path, index=False)


def _public(res, rng, shared):
    fams = {"mobilenetv3_large_100": ["ra_in1k", "ra4_e3600_r224_in1k", "miil_in21k_ft_in1k"], "efficientnet_b0": ["ra_in1k", "ra4_e3600_r224_in1k"]}
    out = {}
    for fam, tags in fams.items():
        for t in tags:
            out[f"{fam}.{t}"] = (BASE if shared else rng.permutation(BASE)) + rng.normal(0, 0.005, 16)
    out["tf_mobilenetv3_large_100.in1k"] = (BASE if shared else rng.permutation(BASE)) + rng.normal(0, 0.005, 16)
    out["tf_efficientnet_b0.ap_in1k"] = (BASE if shared else rng.permutation(BASE)) + rng.normal(0, 0.005, 16)
    for w, p in out.items():
        _w(os.path.join(res, "local6400", "archs", "trained", f"timm_{w}_layers.csv"), p)
    return out


def test_normalisation_does_not_move_the_profile(tmp_path, capsys):
    res = str(tmp_path); rng = np.random.default_rng(0)
    old = _public(res, rng, shared=False)
    for w in RENORM:
        _w(os.path.join(res, "bplus_controls", "renorm", f"timm_{w}_layers.csv"), old[w] + 0.01)
    v = renorm(res)
    assert v == {"P9.56": True, "P9.57": True}
    assert "stands" in capsys.readouterr().out


def test_wrong_normalisation_was_hiding_a_shared_profile(tmp_path, capsys):
    res = str(tmp_path); rng = np.random.default_rng(1)
    _public(res, rng, shared=False)
    for w in ("mobilenetv3_large_100.ra_in1k", "efficientnet_b0.ra_in1k"):          # the correctly measured ones follow BASE
        _w(os.path.join(res, "local6400", "archs", "trained", f"timm_{w}_layers.csv"), BASE + rng.normal(0, 0.005, 16))
    for w in RENORM:
        _w(os.path.join(res, "bplus_controls", "renorm", f"timm_{w}_layers.csv"), BASE + rng.normal(0, 0.005, 16))
    v = renorm(res)
    assert v == {"P9.56": False, "P9.57": False}
    assert "WITHDRAWN" in capsys.readouterr().out


def test_incomplete_is_not_judged(tmp_path):
    res = str(tmp_path); rng = np.random.default_rng(2)
    old = _public(res, rng, shared=False)
    _w(os.path.join(res, "bplus_controls", "renorm", f"timm_{RENORM[0]}_layers.csv"), old[RENORM[0]])
    assert renorm(res) == {} and gamma1(res) == {}


def test_gamma1_controls(tmp_path, capsys):
    res = str(tmp_path); rng = np.random.default_rng(3)
    names = [f"layer{i}.conv" for i in range(16)]
    for fam in GAMMA1:
        base = fam.split(".")[0]
        for t in ("tv_in1k", "a1_in1k"):
            _w(os.path.join(res, "local6400", "archs", "trained", f"timm_{base}.{t}_layers.csv"), BASE + 0.2 + rng.normal(0, 0.02, 16), names)
        for s in (1, 2, 3):
            _w(os.path.join(res, "local6400", "archs", f"seed{s}", f"timm_{base}.tv_in1k_random_layers.csv"), BASE + rng.normal(0, 0.003, 16), names)
            _w(os.path.join(res, "bplus_controls", "gamma1", f"seed{s}", f"timm_{fam}_random_layers.csv"),
               BASE + 0.1 + rng.normal(0, 0.03, 16), names)
    v = gamma1(res)
    assert v == {"P9.58": True, "P9.59": True}
    assert "ordering statements stand" in capsys.readouterr().out
