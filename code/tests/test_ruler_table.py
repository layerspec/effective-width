import os

import numpy as np
import pandas as pd
import pytest

from scripts.ruler_table import CHECKPOINTS, RULERS, numerical_rank, per_layer, summarise, to_latex

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def test_numerical_rank_threshold():
    eig = np.array([1.0, 0.5, 2e-3, 1e-3, 5e-4, 0.0])
    assert numerical_rank(eig) == 3            # strictly above 1e-3 of the largest
    assert numerical_rank(np.zeros(4)) == 0


def test_numerical_rank_scale_invariant():
    eig = np.geomspace(1, 1e-6, 50)
    assert numerical_rank(eig) == numerical_rank(eig * 123.4)


@pytest.fixture(scope="module")
def table():
    need = [os.path.join(RESULTS, "twonn", f"{s}_twonn.csv") for s, _, _ in CHECKPOINTS]
    if not all(os.path.exists(f) for f in need):
        pytest.skip("stored twonn tables missing")
    return pd.concat([per_layer(RESULTS, s, lb) for s, lb, _ in CHECKPOINTS], ignore_index=True)


def test_join_is_complete_and_consistent(table):
    assert len(table) == 86 + 33                      # plan 10.52: 86 trained + 33 random gate-passing layers
    assert table[RULERS].notna().all().all()
    assert (table["k999"] >= table["k95"]).all()
    assert (table["NR"] >= 1).all()
    assert (table["k95_rmax"] <= 1.0 + 1e-9).all()


def test_summary_and_latex(table):
    s = summarise(table)
    assert len(s) == len(CHECKPOINTS) * len(RULERS)
    assert s["rho_raw"].between(-1, 1).all()
    tex = to_latex(s, [lb for _, lb, _ in CHECKPOINTS])
    assert tex.count("\\\\") == len(RULERS) + 3       # header x2, ruler rows, layers row
    assert "TwoNN ID (pooled)" in tex
