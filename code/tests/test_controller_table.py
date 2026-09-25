"""scripts/controller_table.py: the main-text table of 'width read from the covariance, used three ways'
(fold without retraining, retrain at the ruler's widths, one training with the controller) from the result
files, with inference MACs, parameters and training compute in units of one full-width training."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.controller_table import build_rows, render                      # noqa: E402
from scripts.p2_flops import FULL, macs                                        # noqa: E402


def _traj(root, name, accs, widths=None, actions=None):
    for s, a in enumerate(accs):
        d = os.path.join(root, name.format(s=s)); os.makedirs(d, exist_ok=True)
        pd.DataFrame([{"epoch": 100, "test_acc": a, "kind": "conv", "layer": "features.0", "C": 64}]).to_csv(
            os.path.join(d, "trajectory_layers.csv"), index=False)
        open(os.path.join(d, "epoch_100_layers.csv"), "w").write("epoch\n100\n")
        if actions:
            json.dump(actions, open(os.path.join(d, "realloc.json"), "w"))


def test_rows_cover_the_three_uses_with_compute_units(tmp_path):
    res = str(tmp_path)
    os.makedirs(os.path.join(res, "cssp")); os.makedirs(os.path.join(res, "a18"))
    pd.DataFrame([{"scheme": "kstar_0.999", "widths": "47 64 127 128 255 255 254 504 486 469 349 285 168", "params": 9928855,
                   "param_frac": 0.674, "acc_full": 93.83, "acc_hook": 93.81, "acc_fold": 93.82},
                  {"scheme": "kstar_0.99", "widths": "30 59 120 123 246 246 245 463 386 318 120 77 42", "params": 5770366,
                   "param_frac": 0.392, "acc_full": 93.83, "acc_hook": 93.44, "acc_fold": 93.34}]).to_csv(
        os.path.join(res, "cssp", "fold_results.csv"), index=False)
    json.dump({"widths": {"ruler": [47, 64, 127, 128, 255, 255, 254, 504, 486, 469, 347, 285, 168],
                          "uniform": [52, 52, 104, 104, 208, 208, 208, 417, 417, 417, 417, 417, 417]}},
              open(os.path.join(res, "a18", "widths_act.json"), "w"))
    _traj(res, "round3/a5_vgg16_bn_none_s{s}", [93.83, 93.74, 93.74])
    _traj(res, "a18/vgg16_bn_ruler_act_s{s}", [93.80, 93.87, 93.85])
    _traj(res, "a18/vgg16_bn_uniform_s{s}", [93.31, 92.91, 93.37])
    f60 = [{"epoch": 15, "before": FULL, "after": [40] * 13, "shrink_only": False, "macs_fraction": 0.62},
           {"epoch": 40, "before": [40] * 13, "after": [40] * 13, "shrink_only": True, "macs_fraction": 0.60}]
    _traj(res, "p2/c10_f60_s{s}", [93.60, 93.53, 93.68], actions=f60)
    _traj(res, "p2/c10_ffixed342_s{s}", [93.46, 93.38, 93.48],
          actions=[{"epoch": 15, "before": [37] * 13, "after": [30] * 13, "shrink_only": False, "macs_fraction": 0.342},
                   {"epoch": 40, "before": [30] * 13, "after": [30] * 13, "shrink_only": True, "macs_fraction": 0.342}])
    rows = build_rows(res)
    by = {r["arm"]: r for r in rows}
    assert "full width" in by and by["full width"]["train"] == 1.0 and by["full width"]["n"] == 3
    assert abs(by["fold, $\\tau = 0.999$"]["macs"] - macs([47, 64, 127, 128, 255, 255, 254, 504, 486, 469, 349, 285, 168]) / macs(FULL)) < 1e-6
    assert by["fold, $\\tau = 0.999$"]["train"] == 1.0 and by["fold, $\\tau = 0.999$"]["acc"] == 93.82
    assert by["retrain at the ruler's widths"]["train"] > 1.5           # measure full + retrain
    assert 0.6 < by["controller, 0.60 MACs"]["train"] < 0.8 and by["controller, 0.60 MACs"]["macs"] == 0.60
    assert by["controller, 0.60 MACs"]["n"] == 3 and abs(by["controller, 0.60 MACs"]["acc"] - 93.603) < 0.01
    tex = render(rows)
    assert "\\begin{tabular}" in tex and "93.82" in tex and "\\pm" in tex and "0.60" in tex
