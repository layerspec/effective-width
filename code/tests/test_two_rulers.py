"""two_rulers.py and its judge (plan 9.24): the numerical rank, and verdicts that can come out both ways."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import two_rulers as T            # noqa: E402
from scripts import two_rulers_judge as J      # noqa: E402
from layerspec.metrics import k_star           # noqa: E402


def test_numerical_rank_counts_eigenvalues_above_the_threshold():
    e = np.array([1.0, 0.5, 2e-3, 9e-4, 1e-7, 0.0])
    assert T.numerical_rank(e) == 3
    assert T.numerical_rank(e, 1e-6) == 4
    assert T.numerical_rank(np.zeros(5)) == 0 and T.numerical_rank(np.array([])) == 0
    assert T.numerical_rank(np.array([3.0, -1e-9, 1.0])) == 2          # a rounding negative is not counted


def test_the_two_rulers_can_disagree_on_one_spectrum():
    # ten large eigenvalues and a long flat tail: numerical rank 10, but 99.9% of the variance needs most of the tail
    e = np.concatenate([np.full(10, 1.0), np.full(502, 5e-4)])
    assert T.numerical_rank(e) == 10
    assert k_star(e, 0.999) > 400


def test_arms_and_steps_are_the_registered_ones():
    assert list(T.ARMS) == ["c10_full", "c10_pc640", "c10_pc640_e780", "c100_k13", "c100_k20", "c100_k50", "c100_k100"]
    assert T.STEPS == (0, 25, 75, 200, 390, 1950) and T.SEEDS == (0, 1, 2)
    assert T.RECIPES["ours"]["onecycle"] and not T.RECIPES["theirs"]["onecycle"] and T.RECIPES["theirs"]["wd"] == 1e-4


def _end(path, late_nr, late_k, arms=None, k_by_class=None, nr_by_class=None, e780=(0.02, 0.2)):
    rows = []
    for arm in (arms or T.ARMS):
        for s in (0, 1, 2):
            for i in range(13):
                nr, k = (0.5, 0.9) if i < 10 else (late_nr, late_k)
                if arm.startswith("c100_k") and i == 12:
                    K = int(arm.split("k")[1]); nr, k = (nr_by_class or {}).get(K, nr), (k_by_class or {}).get(K, k)
                if arm == "c10_pc640_e780" and i >= 10:
                    nr, k = e780
                rows.append(dict(arm=arm, seed=s, index=i, layer=f"f{i}", C=512, ok=True, nr=round(512 * nr), k999=round(512 * k)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_end_state_verdicts(tmp_path):
    p = str(tmp_path / "a" / "end_state.csv")
    _end(p, 0.02, 0.45, k_by_class={13: 0.5, 20: 0.7, 50: 0.9, 100: 0.98}, nr_by_class={13: 0.03, 20: 0.04, 50: 0.1, 100: 0.2})
    assert J.end_state(p) == {"P9.72": True, "P9.73": True, "P9.75": True}
    q = str(tmp_path / "b" / "end_state.csv")
    _end(q, 0.30, 0.45, k_by_class={13: 0.5, 20: 0.7, 50: 0.9, 100: 0.98}, nr_by_class={13: 0.2, 20: 0.1, 50: 0.3, 100: 0.4},
         e780=(0.10, 0.40))
    assert J.end_state(q) == {"P9.72": False, "P9.73": False, "P9.75": False}


def _early(path, nr_at, k_at_1950):
    rows = []
    for recipe in ("ours", "theirs"):
        for s in (0, 1, 2):
            for step in T.STEPS:
                for i in range(13):
                    rows.append(dict(recipe=recipe, seed=s, step=step, loss=1.0, lr=0.1, index=i, layer=f"f{i}", C=512, ok=True,
                                     nr=round(512 * nr_at(recipe, step)), k999=round(512 * (k_at_1950 if step == 1950 else 0.95))))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_early_verdicts(tmp_path):
    e = str(tmp_path / "end_state.csv"); _end(e, 0.02, 0.30)
    a = str(tmp_path / "early_a.csv")
    _early(a, lambda r, s: 0.6 if s == 0 else (0.1 if (r == "theirs" and s >= 75) or s >= 390 else 0.55), 0.80)
    assert J.early(a, e) == {"P9.74a": True, "P9.74b": True, "P9.74c": True}
    b = str(tmp_path / "early_b.csv")
    _early(b, lambda r, s: 0.6 if s < 1950 else 0.5, 0.35)
    assert J.early(b, e) == {"P9.74a": False, "P9.74b": False, "P9.74c": False}
