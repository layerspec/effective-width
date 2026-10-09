"""Plan 9.48 (P9.113-P9.115): the tau-width against the LDA dimension, on the stored B9 cell decomposition.

Reads results/position_collapse/cells.csv (plan 9.29: Sigma = B_c + W_c per late VGG-16 convolution of the A34 arms) and
compares, per row,

    LDA dimension      kB_c_0.999 = k*_0.999(B_c)  <= K - 1          (rank of the between-class part)
    NC1 ratio          nu = tr W_c / tr B_c = w_c / (1 - w_c)
    the scale          k*_0.95(Sigma), k*_0.999(Sigma)                 (r_max = 512 for all six layers)

    cd code && python scripts/lda_compare.py --results ../results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

LAST = "features.40"
FEW = ("c10_full", "c10_pc640", "c10_pc640_e780", "c100_k13", "c100_k20")     # K <= 20


def derive(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Km1"] = d["K"] - 1
    d["nu"] = d["w_c"] / (1.0 - d["w_c"])
    d["lda_over_Km1"] = d["kB_c_0.999"] / d["Km1"]
    d["k95_over_Km1"] = d["k0.95"] / d["Km1"]
    return d


def spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    return float(a.rank().corr(b.rank())) if len(a) >= 3 else float("nan")


def judge_frame(df: pd.DataFrame) -> dict:
    d = derive(df)
    res: dict = {"n_rows": int(len(d))}
    # P9.113: LDA dimension saturates at the last conv for K <= 20
    last = d[(d["layer"] == LAST) & d["arm"].isin(FEW)]
    sat = last["kB_c_0.999"] == last["Km1"]
    res["P9.113"] = {"pass": int(sat.sum()), "total": int(len(last)), "holds": bool(len(last) == 15 and sat.sum() >= 12),
                     "rows": last[["arm", "seed", "K", "kB_c_0.999"]].to_dict("records")}
    big = d[(d["layer"] == LAST) & ~d["arm"].isin(FEW)]
    res["P9.113_descriptive_K50_K100"] = big[["arm", "seed", "K", "kB_c_0.999", "nu"]].to_dict("records")
    # sanity: the rank bound itself
    res["rank_bound_violations"] = int((d["kB_c_0.999"] > d["Km1"]).sum())
    # P9.114: w_c > 0.2  =>  k*_0.95(Sigma) > K - 1 in >= 90 %; lemma half: w_c <= 0.05  =>  <= K - 1 in 100 %
    hi = d[d["w_c"] > 0.2]
    lo = d[d["w_c"] <= 0.05]
    frac_hi = float((hi["k0.95"] > hi["Km1"]).mean()) if len(hi) else float("nan")
    frac_lo = float((lo["k0.95"] <= lo["Km1"]).mean()) if len(lo) else float("nan")
    res["P9.114"] = {"n_hi": int(len(hi)), "frac_hi_exceeds": frac_hi, "holds": bool(len(hi) and frac_hi >= 0.90),
                     "n_lo": int(len(lo)), "frac_lo_within": frac_lo, "lemma_ok": bool(len(lo) == 0 or frac_lo == 1.0),
                     "hi_violations": hi.loc[hi["k0.95"] <= hi["Km1"], ["arm", "seed", "layer", "K", "w_c", "k0.95"]].to_dict("records"),
                     "lo_violations": lo.loc[lo["k0.95"] > lo["Km1"], ["arm", "seed", "layer", "K", "w_c", "k0.95"]].to_dict("records")}
    # P9.115: rho(nu, k95/(K-1)) >= 0.7 over all rows
    rho = spearman(d["nu"], d["k95_over_Km1"])
    per = {f"{a}/s{s}": spearman(g["nu"], g["k95_over_Km1"]) for (a, s), g in d.groupby(["arm", "seed"])}
    res["P9.115"] = {"rho_all": rho, "holds": bool(rho >= 0.7), "rho_within_median": float(np.nanmedian(list(per.values()))),
                     "rho_within": per}
    # exploratory: per-arm last conv medians over seeds; depth at which nu < 0.1
    ex = {}
    for arm, g in d[d["layer"] == LAST].groupby("arm"):
        ex[arm] = {"K": int(g["K"].iloc[0]), "lda_median": float(g["kB_c_0.999"].median()), "nu_median": float(g["nu"].median()),
                   "k95_median": float(g["k0.95"].median()), "k999_median": float(g["k0.999"].median()),
                   "w_c_median": float(g["w_c"].median())}
    first_nc1 = {}
    for (arm, s), g in d.groupby(["arm", "seed"]):
        g = g.sort_values("layer", key=lambda c: c.str.extract(r"(\d+)")[0].astype(int))
        hit = g[g["nu"] < 0.1]
        first_nc1[f"{arm}/s{s}"] = str(hit["layer"].iloc[0]) if len(hit) else "none"
    res["exploratory"] = {"last_conv_by_arm": ex, "first_layer_nu_below_0.1": first_nc1}
    return res


def format_judgement(res: dict) -> str:
    L = [f"rows {res['n_rows']}; rank-bound violations (k*_0.999(B_c) > K-1): {res['rank_bound_violations']}"]
    a = res["P9.113"]
    L.append(f"P9.113 LDA dimension = K-1 at {LAST}, K <= 20: {a['pass']}/{a['total']} -> {'HOLDS' if a['holds'] else 'FAILS'} (>= 12/15)")
    for r in a["rows"]:
        if r["kB_c_0.999"] != r["K"] - 1:
            L.append(f"    not saturated: {r['arm']} s{r['seed']} K={r['K']} kB_c={r['kB_c_0.999']}")
    for r in res["P9.113_descriptive_K50_K100"]:
        L.append(f"    descriptive {r['arm']} s{r['seed']}: K-1={r['K']-1} kB_c={r['kB_c_0.999']} nu={r['nu']:.3f}")
    b = res["P9.114"]
    L.append(f"P9.114 w_c > 0.2 => k95 > K-1: {b['frac_hi_exceeds']:.3f} of {b['n_hi']} rows -> {'HOLDS' if b['holds'] else 'FAILS'} (>= 0.90); "
             f"lemma half w_c <= 0.05 => k95 <= K-1: {b['frac_lo_within']:.3f} of {b['n_lo']} rows -> {'OK' if b['lemma_ok'] else 'VIOLATED'}")
    for r in b["hi_violations"][:10]:
        L.append(f"    hi-violation {r['arm']} s{r['seed']} {r['layer']} K={r['K']} w_c={r['w_c']:.3f} k95={r['k0.95']}")
    for r in b["lo_violations"][:10]:
        L.append(f"    lemma-violation {r['arm']} s{r['seed']} {r['layer']} K={r['K']} w_c={r['w_c']:.3f} k95={r['k0.95']}")
    c = res["P9.115"]
    L.append(f"P9.115 rho(nu, k95/(K-1)) all rows {c['rho_all']:+.3f} -> {'HOLDS' if c['holds'] else 'FAILS'} (>= 0.7); "
             f"within (arm, seed) six-layer rho median {c['rho_within_median']:+.2f}")
    L.append("exploratory, last conv by arm (median over seeds): K-1 | LDA dim | nu | w_c | k95 | k999")
    for arm, v in res["exploratory"]["last_conv_by_arm"].items():
        L.append(f"    {arm:16s} {v['K']-1:4d} | {v['lda_median']:5.0f} | {v['nu_median']:.3f} | {v['w_c_median']:.3f} | {v['k95_median']:5.0f} | {v['k999_median']:5.0f}")
    L.append("first layer with nu < 0.1: " + ", ".join(f"{k}:{v}" for k, v in res["exploratory"]["first_layer_nu_below_0.1"].items()))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    args = ap.parse_args(argv)
    df = pd.read_csv(os.path.join(args.results, "position_collapse", "cells.csv"))
    res = judge_frame(df)
    txt = format_judgement(res)
    print(txt)
    out = os.path.join(args.results, "position_collapse")
    with open(os.path.join(out, "lda_compare.txt"), "w") as fh:
        fh.write(txt + "\n")
    with open(os.path.join(out, "lda_compare.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    derive(df).to_csv(os.path.join(out, "lda_compare.csv"), index=False)


if __name__ == "__main__":
    main()
