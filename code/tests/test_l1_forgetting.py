"""l1_forgetting_judge.py (plan 9.27 C): P9.85 / P9.86 from synthetic random-network profiles on two image sets."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.l1_forgetting_judge import FAMILIES, FORGET, KEEP, main   # noqa: E402


def _csv(path, k95, k999):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([{"layer": f"c{i}", "kind": "conv", "depth_index": i, "n_over_C": 100.0, "is_depthwise": False, "r_max": 64.0,
                   "k_star_0.95": a * 64, "k_star_0.999": b * 64} for i, (a, b) in enumerate(zip(k95, k999))]).to_csv(path, index=False)


def _build(res, forget_ok=True, keep_ok=True):
    L = 20
    base95 = np.linspace(0.2, 0.5, L); base999 = np.linspace(0.5, 0.9, L)
    for fam, name in FAMILIES.items():
        for s in (1, 2, 3):
            im = os.path.join(res, "local6400", f"seed{s}", f"{name}_layers.csv") if fam == "resnet50" \
                else os.path.join(res, "local6400", "archs", f"seed{s}", f"{name}_layers.csv")
            ci = os.path.join(res, "apriori", "cifar10", f"seed{s}", f"{name}_layers.csv")
            _csv(im, base95, base999)
            if fam in FORGET and forget_ok or fam in KEEP and not keep_ok:
                rel = np.linspace(0.6, 0.0, L)                     # forgets with depth
            else:
                rel = np.full(L, 0.5)                              # keeps a constant relative difference
            _csv(ci, base95 * (1 + rel), base999 * (1 + rel * 0.3))


def test_both_hold(tmp_path, capsys):
    _build(str(tmp_path))
    main(["--results", str(tmp_path)])
    out = capsys.readouterr().out
    assert "P9.85 (" in out and "-> holds" in out.split("P9.85")[1].split("\n")[0]
    assert "-> holds" in out.split("P9.86")[1].split("\n")[0] and "three-seed sentence" in out


def test_keep_fails_when_residual_nets_forget(tmp_path, capsys):
    _build(str(tmp_path), keep_ok=False)
    main(["--results", str(tmp_path)])
    out = capsys.readouterr().out
    assert "-> fails" in out.split("P9.86")[1].split("\n")[0] and "stays exploratory" in out
