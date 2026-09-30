#!/usr/bin/env python
"""Verdicts of the two controls on the measurement (analysis-plan 9.20), written before their data:
P9.56 / P9.57 (input normalisation of five public weights) and P9.58 / P9.59 (initialisation of the random networks
of the five ResNet families).  Profiles are k*(0.95)/r_max over the dense convolutions that pass n/C >= 50.

    python scripts/bplus_controls_judge.py --results ../results
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_controls import GAMMA1, RENORM   # noqa: E402

RENORM_FAMILIES = {"MobileNetV3-L": "timm_*mobilenetv3_large_100.*_layers.csv", "EfficientNet-B0": "timm_*efficientnet_b0.*_layers.csv"}


def profile(path: str) -> pd.Series:
    d = pd.read_csv(path)
    d = d[(d.kind == "conv") & d.n_over_C_ok.astype(bool)]
    if "is_depthwise" in d:
        d = d[~d.is_depthwise.astype(bool)]
    d = d.sort_values("depth_index")
    return pd.Series((d["k_star_0.95"] / d["r_max"]).to_numpy(), index=d.layer.to_numpy())


def rho(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def aligned(profiles: list) -> list:
    """The profiles on their common layers; by position when the layer counts agree and the names do not."""
    common = set.intersection(*[set(p.index) for p in profiles])
    if len(common) >= 5 and all(len(common) == len(p) for p in profiles):
        return [p[[k for k in profiles[0].index if k in common]].to_numpy() for p in profiles]
    if len({len(p) for p in profiles}) == 1:
        return [p.to_numpy() for p in profiles]
    keep = [k for k in profiles[0].index if k in common]
    return [p[keep].to_numpy() for p in profiles]


def old_path(results: str, weight: str):
    # the first measurement (ImageNet statistics) was moved aside when the corrected one replaced it, 2026-09-30
    for d in ("local6400/archs/superseded_wrong_norm", "local6400/archs/trained", "local6400/trained"):
        f = os.path.join(results, d, f"timm_{weight}_layers.csv")
        if os.path.exists(f):
            return f
    return None


def renorm(results: str) -> dict:
    print("==== P9.56 / P9.57: five public weights measured with their own input statistics (plan 9.20 A) ====")
    V, new = {}, {}
    rows = []
    for w in RENORM:
        f, g = old_path(results, w), os.path.join(results, "bplus_controls", "renorm", f"timm_{w}_layers.csv")
        if f is None or not os.path.exists(g):
            continue
        a, b = aligned([profile(f), profile(g)])
        new[w] = g
        rows.append((w, rho(a, b), float(np.median(a)), float(np.median(b)), len(a)))
        print(f"  {w:44s} layers {len(a):3d}  rho {rows[-1][1]:+.3f}  level {rows[-1][2]:.3f} -> {rows[-1][3]:.3f}")
    if len(rows) == len(RENORM):
        V["P9.56"] = all(r[1] >= 0.95 and abs(r[3] - r[2]) <= 0.03 for r in rows)
        print(f"  P9.56 (every weight: rho >= 0.95 and |level difference| <= 0.03) -> {'holds' if V['P9.56'] else 'fails'}")
        ok = []
        for fam, pat in RENORM_FAMILIES.items():
            files = sorted(glob.glob(os.path.join(results, "local6400", "archs", "trained", pat)))
            before = aligned([profile(old_path(results, os.path.basename(f)[5:-11]) if os.path.basename(f)[5:-11] in new else f)
                              for f in files])
            after = aligned([profile(new.get(os.path.basename(f)[5:-11], f)) for f in files])
            tt = lambda P: float(np.median([rho(x, y) for x, y in itertools.combinations(P, 2)]))   # noqa: E731
            n_new = sum(os.path.basename(f)[5:-11] in new for f in files)
            print(f"  {fam}: {len(files)} weights ({n_new} re-measured), T-T median {tt(before):+.2f} -> {tt(after):+.2f}; "
                  f"level range {min(np.median(x) for x in after):.2f}-{max(np.median(x) for x in after):.2f}")
            ok.append(tt(after) < 0.5)
        V["P9.57"] = all(ok)
        print(f"  P9.57 (both families still share no profile: corrected T-T median < 0.5) -> {'holds' if V['P9.57'] else 'fails'}")
        print("  Registered consequence: " + ("the corrected numbers replace the old ones; " if True else "")
              + ("'no profile shared across recipes' stands." if V["P9.57"] else "'no profile shared across recipes' is WITHDRAWN for the failing family."))
    else:
        print(f"  {len(rows)}/{len(RENORM)} weights re-measured -- not judged yet")
    return V


def gamma1(results: str) -> dict:
    from scripts.checkpoint_analysis import ARCH_FAMILIES
    print("==== P9.58 / P9.59: random networks of the ResNet families with gamma = 1 (plan 9.20 B) ====")
    key = {"resnet18.tv_in1k": "resnet18", "resnet34.tv_in1k": "resnet34", "resnet101.tv_in1k": "resnet101",
           "wide_resnet50_2.tv_in1k": "wide_resnet50_2", "resnext50_32x4d.tv_in1k": "resnext50"}
    rows = []
    print(f"  {'family':18s} {'T':>2s} {'R0':>2s} {'R1':>2s}  {'T-R g=0':>8s} {'T-R g=1':>8s}  {'level g=0':>9s} {'g=1':>5s}  {'R-R g=0':>8s} {'g=1':>6s}")
    for fam in GAMMA1:
        t_glob, r_glob = ARCH_FAMILIES[key[fam]]
        T = [profile(f) for f in sorted(glob.glob(os.path.join(results, t_glob)))]
        R0 = [profile(f) for f in sorted(glob.glob(os.path.join(results, r_glob)))]
        R1 = [profile(f) for f in sorted(glob.glob(os.path.join(results, "bplus_controls", "gamma1", "seed*", f"timm_{fam}_random_layers.csv")))]
        if not T or not R0 or len(R1) < 3:
            print(f"  {fam:18s} {len(T):2d} {len(R0):2d} {len(R1):2d}  not complete"); continue
        A = aligned(T + R0 + R1)
        T_, R0_, R1_ = A[:len(T)], A[len(T):len(T) + len(R0)], A[len(T) + len(R0):]
        med = lambda X, Y: float(np.median([rho(x, y) for x in X for y in Y]))                    # noqa: E731
        rr = lambda X: float(np.median([rho(x, y) for x, y in itertools.combinations(X, 2)]))     # noqa: E731
        lev = lambda X: float(np.median([np.median(x) for x in X]))                               # noqa: E731
        rows.append(dict(fam=fam, tr0=med(T_, R0_), tr1=med(T_, R1_), l0=lev(R0_), l1=lev(R1_), rr0=rr(R0_), rr1=rr(R1_)))
        r = rows[-1]
        print(f"  {fam:18s} {len(T):2d} {len(R0):2d} {len(R1):2d}  {r['tr0']:+8.2f} {r['tr1']:+8.2f}  {r['l0']:9.2f} {r['l1']:5.2f}  {r['rr0']:+8.2f} {r['rr1']:+6.2f}")
    V = {}
    if len(rows) == len(GAMMA1):
        V["P9.58"] = all(abs(r["tr1"] - r["tr0"]) <= 0.10 for r in rows)
        V["P9.59"] = all(r["l1"] >= r["l0"] + 0.05 and r["rr1"] < r["rr0"] for r in rows)
        print(f"  P9.58 (T-R changes by at most 0.10 in every family) -> {'holds' if V['P9.58'] else 'fails'}"
              + ("" if V["P9.58"] else "  [" + ", ".join(r["fam"] for r in rows if abs(r["tr1"] - r["tr0"]) > 0.10) + "]"))
        print(f"  P9.59 (gamma = 1: level higher by 0.05 and R-R lower, every family; descriptive) -> {'holds' if V['P9.59'] else 'fails'}")
        print("  Registered consequence: both controls are reported side by side; "
              + ("the ordering statements stand." if V["P9.58"] else "the ordering statement of the failing families is conditioned on the initialisation."))
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    renorm(a.results); print(); gamma1(a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
