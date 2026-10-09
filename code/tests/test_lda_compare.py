import numpy as np
import pandas as pd

from scripts.lda_compare import FEW, LAST, derive, judge_frame, spearman

LAYERS = ["features.24", "features.27", "features.30", "features.34", "features.37", "features.40"]


def _rows(arm, K, seed, w_c, k95, kB=None):
    out = []
    for i, (lay, w, k) in enumerate(zip(LAYERS, w_c, k95)):
        out.append({"arm": arm, "seed": seed, "layer": lay, "K": K, "w_c": w, "k0.95": k, "k0.999": k + 50,
                    "kB_c_0.999": (K - 1) if kB is None else kB[i]})
    return out


def _frame(saturate=True, hi_exceed=True, rho_pos=True):
    rows = []
    arms = [("c10_full", 10), ("c10_pc640", 10), ("c10_pc640_e780", 10), ("c100_k13", 13), ("c100_k20", 20), ("c100_k50", 50), ("c100_k100", 100)]
    for arm, K in arms:
        for s in range(3):
            w = [0.6, 0.5, 0.4, 0.3, 0.1, 0.02]                      # within-class share falls with depth
            if rho_pos:
                k = [int((K - 1) * f) for f in (8, 6, 5, 3, 1.5, 0.5)]  # scale above K-1 while w_c is large, below at the end
            else:
                k = [int((K - 1) * f) for f in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
            if not hi_exceed:
                k = [min(v, K - 2) for v in k]
            kB = None if saturate else [K - 3] * 6
            rows += _rows(arm, K, s, w, k, kB)
    return pd.DataFrame(rows)


def test_derive_nu_and_ratios():
    d = derive(_frame())
    r = d.iloc[0]
    assert np.isclose(r["nu"], r["w_c"] / (1 - r["w_c"])) and r["Km1"] == r["K"] - 1


def test_spearman_basic():
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_judge_all_hold():
    res = judge_frame(_frame())
    assert res["P9.113"]["holds"] and res["P9.113"]["total"] == 15
    assert res["P9.114"]["holds"] and res["P9.114"]["lemma_ok"]
    assert res["P9.115"]["holds"] and res["rank_bound_violations"] == 0


def test_judge_failures():
    res = judge_frame(_frame(saturate=False))
    assert not res["P9.113"]["holds"]
    res = judge_frame(_frame(hi_exceed=False))
    assert not res["P9.114"]["holds"]
    res = judge_frame(_frame(rho_pos=False))
    assert not res["P9.115"]["holds"]


def test_lemma_half_flags_violation():
    df = _frame()
    i = df.index[(df["layer"] == LAST) & (df["arm"] == FEW[0])][0]
    df.loc[i, "k0.95"] = 100                                     # w_c = 0.02 but the scale above K-1: lemma violated
    res = judge_frame(df)
    assert not res["P9.114"]["lemma_ok"] and len(res["P9.114"]["lo_violations"]) == 1


def test_first_layer_nu_below_threshold():
    res = judge_frame(_frame())
    assert set(res["exploratory"]["first_layer_nu_below_0.1"].values()) == {"features.40"}   # w_c 0.1 -> nu 0.111 (not below); 0.02 -> 0.020
