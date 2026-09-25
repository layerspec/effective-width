"""scripts/p2_analysis.py P2l section: FLOPs-conserving shrink-only arm vs f60, and the x4-wide ResNet-20
(full / controller / uniform) with the late-layer ruler reading (P-l1..l4)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.p2_analysis import p2l_section   # noqa: E402


def _run(root, name, acc, C, k999, actions=None):
    d = os.path.join(root, "p2", name); os.makedirs(d)
    rows = [{"layer": f"layers.{i}.conv1", "kind": "conv", "depth_index": i + 1, "C": c, "k_star_0.999": k * c,
             "k_star_ratio_0.999": k, "k_star_0.95": 0.5 * c, "k_star_ratio_0.95": 0.5, "n_over_C": 100.0,
             "epoch": 100, "test_acc": acc, "is_depthwise": False} for i, (c, k) in enumerate(zip(C, k999))]
    rows.insert(0, {**rows[0], "layer": "conv1", "depth_index": 0})
    pd.DataFrame(rows).to_csv(os.path.join(d, "epoch_100_layers.csv"), index=False)
    if actions is not None:
        json.dump(actions, open(os.path.join(d, "realloc.json"), "w"))


def test_p2l_verdicts(tmp_path, capsys):
    res = str(tmp_path)
    act = [{"epoch": 15, "shrink_only": False, "before": [64] * 9, "after": [40] * 9, "macs_fraction": 0.6,
            "params_before": 4.3e6, "params_after": 2.6e6, "cost": "flops"},
           {"epoch": 40, "shrink_only": True, "before": [40] * 9, "after": [40] * 9, "macs_fraction": 0.6,
            "params_before": 2.6e6, "params_after": 2.6e6, "cost": "flops"}]
    for s, a in enumerate((93.30, 93.20, 93.25)):
        _run(res, f"c10_f60_nogrow_s{s}", a, [64] * 13, [0.9] * 13, act)
    for s, a in enumerate((93.60, 93.53, 93.68)):
        _run(res, f"c10_f60_s{s}", a, [64] * 13, [0.9] * 13, act)
    wide_C = [64] * 3 + [128] * 3 + [256] * 3
    for s, a in enumerate((94.9, 94.8, 95.0)):
        _run(res, f"r20w4_full_s{s}", a, wide_C, [0.95] * 3 + [0.9] * 3 + [0.5] * 3)          # late layers idle
    for s, a in enumerate((94.7, 94.6, 94.8)):
        _run(res, f"r20w4_f60_s{s}", a, [40] * 3 + [80] * 3 + [150] * 3, [0.95] * 9, act)
    for s, a in enumerate((94.2, 94.1, 94.3)):
        _run(res, f"r20w4_ufixed60_s{s}", a, [38] * 3 + [77] * 3 + [154] * 3, [0.95] * 9)
    p2l_section(res)
    out = capsys.readouterr().out
    line = lambda tag: out.split("\n  " + tag + " (")[1].split("\n")[0]   # noqa: E731
    assert "P2l" in out
    assert "-> yes" in line("P-l1")          # nogrow 93.25 <= f60 93.60 - 0.2
    assert "-> yes" in line("P-l2")          # stage-3 ruler 0.5 < 0.7, stage 1 0.95 >= 0.9
    assert "-> yes" in line("P-l3")          # 94.7 >= 94.2 + 0.3, all above all
    assert "-> yes" in line("P-l4")          # 94.7 >= 94.9 - 0.5
