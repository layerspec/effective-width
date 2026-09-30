"""checkpoint_analysis.py section for A17 reduced (ImageNet basic vs bottleneck): R-R / T-T / T-R per
architecture from results/a17/<arch>_s<k>/trajectory_layers.csv, H9.4a / H9.4b verdicts, and the
correlation of the final ResNet-50 profile with the public ResNet-50 recipes of section 6."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.checkpoint_analysis import section_a17, section_a29b   # noqa: E402


def _traj(path, layers, profiles_by_epoch, acc=70.0):
    rows = []
    for e, prof in profiles_by_epoch.items():
        for i, (lay, k) in enumerate(zip(layers, prof)):
            rows.append({"layer": lay, "kind": "conv", "depth_index": i, "C": 64, "r_max": 64, "n_over_C": 100.0,
                         "k_star_0.95": k * 64, "k_star_ratio_0.95": k, "epoch": e, "test_acc": acc, "is_depthwise": False})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    with open(os.path.join(os.path.dirname(path), "result.json"), "w") as fh:   # a finished run
        fh.write('{"final_top1": %s, "epochs": 90}' % acc)


def _public(path, layers, prof):
    pd.DataFrame([{"layer": lay, "kind": "conv", "depth_index": i, "C": 64, "r_max": 64, "n_over_C": 100.0,
                   "k_star_0.95": k * 64, "is_depthwise": False} for i, (lay, k) in enumerate(zip(layers, prof))]).to_csv(path, index=False)


def test_section_a17_verdicts(tmp_path, capsys):
    res = str(tmp_path); rng = np.random.default_rng(0)
    L = [f"layer{i}.conv" for i in range(20)]
    init = np.linspace(0.2, 0.6, 20)
    # basic keeps its initial order (T-R high); bottleneck reorders (T-R low)
    for s in (0, 1):
        _traj(os.path.join(res, "a17", f"basic6666_s{s}", "trajectory_layers.csv"), L,
              {0: init + rng.normal(0, 0.01, 20), 90: init + 0.2 + rng.normal(0, 0.02, 20)})
        _traj(os.path.join(res, "a17", f"resnet50_s{s}", "trajectory_layers.csv"), L,
              {0: init + rng.normal(0, 0.01, 20), 90: rng.permutation(init) + 0.2})
    os.makedirs(os.path.join(res, "local6400", "trained"))
    for name in ("timm_resnet50.a1_in1k", "timm_resnet50.tv2_in1k"):
        _public(os.path.join(res, "local6400", "trained", f"{name}_layers.csv"), L, rng.permutation(init) + 0.2)
    section_a17(res)
    out = capsys.readouterr().out
    assert "A17" in out and "basic6666" in out and "resnet50" in out
    assert "H9.4a" in out and "SUPPORTED" in out.split("H9.4a")[1].split("\n")[0]        # bottleneck T-R below basic
    assert "H9.4b" in out and "NOT supported" in out.split("H9.4b")[1].split("\n")[0]   # this-recipe ResNet-50 T-R < 0.4
    assert "public ResNet-50" in out                                                    # T-T against the section-6 recipes


def test_section_a29b_verdicts(tmp_path, capsys):
    res = str(tmp_path); rng = np.random.default_rng(1)
    def fam(arch, root, L, tr):
        layers = [f"layer{i}.conv" for i in range(L)]
        init = np.linspace(0.2, 0.6, L)
        for s in (0, 1):
            final = (init[::-1] if tr == 0 else (1 - tr) * rng.permutation(init) + tr * init) + 0.2   # tr = 0: reversed
            _traj(os.path.join(res, root, f"{arch}_s{s}", "trajectory_layers.csv"), layers,
                  {0: init + rng.normal(0, 0.005, L), 90: final + rng.normal(0, 0.005, L)})
        return layers, init
    L18, init18 = fam("resnet18", "a29b", 20, 0.95)          # keeps its order
    fam("resnet34", "a29b", 36, 0.5)
    fam("basic6666", "a17", 52, 0.0)                         # reorders
    os.makedirs(os.path.join(res, "local6400", "archs", "trained"))
    _public(os.path.join(res, "local6400", "archs", "trained", "timm_resnet18.tv_in1k_layers.csv"), L18, init18 + 0.2)
    assert section_a29b(res) is True
    out = capsys.readouterr().out
    assert "P9.50" in out and "holds" in out.split("P9.50")[1] and "median" in out and "20" in out
    # the depth sentence is not written when the middle point breaks the order
    res2 = str(tmp_path / "b"); rng = np.random.default_rng(2)
    def fam2(arch, root, L, tr):
        layers = [f"layer{i}.conv" for i in range(L)]; init = np.linspace(0.2, 0.6, L)
        for s in (0, 1):
            final = (init[::-1] if tr == 0 else (1 - tr) * rng.permutation(init) + tr * init) + 0.2
            _traj(os.path.join(res2, root, f"{arch}_s{s}", "trajectory_layers.csv"), layers, {0: init, 90: final})
    fam2("resnet18", "a29b", 20, 0.95); fam2("resnet34", "a29b", 36, 0.0); fam2("basic6666", "a17", 52, 0.5)
    assert section_a29b(res2) is False
