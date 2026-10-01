"""B9, part A (plan 9.29): is the tau-width of a collapsed layer the number of spatial positions times the number of classes?

Lemma B9.0: split the (image, position) samples into P*K cells (class, position); Sigma = B + W with B the covariance of the cell
means (rank <= P*K - 1) and W the within-cell covariance; if tr W / tr Sigma <= 1 - tau then k*_tau(Sigma) <= P*K - 1.

Existing spectra only: results/gradgram/<arm>_s<seed>_spectra.npz holds the eigenvalues of the output covariance of every
convolution of 21 VGG-16_BN checkpoints (CIFAR-10 and K-class subsets of CIFAR-100).  The last block of the CIFAR VGG-16
(features.34/37/40) works on a 2 x 2 map, P = 4.

    python scripts/position_collapse_judge.py --results ../results          # P-b9.1 .. P-b9.3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLOCK5 = ("features.34", "features.37", "features.40")
P_BLOCK5 = 4                                    # 2 x 2 positions on 32 x 32 inputs
ARM_K = {"c10_full": 10, "c10_pc640": 10, "c10_pc640_e780": 10, "c100_k13": 13, "c100_k20": 20, "c100_k50": 50, "c100_k100": 100}
TRAINED_K10 = ("c10_full", "c10_pc640_e780")
BLIND = ("c100_k13", "c100_k20")
RATIO_MIN = 3.0


def k_star(eig: np.ndarray, tau: float) -> int:
    e = np.clip(np.sort(np.asarray(eig, dtype=np.float64))[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-300), tau) + 1)


def cliff(eig: np.ndarray, lo: int) -> tuple[int, float]:
    """(g, ratio): the 1-based index g in [lo, C/2] maximising lambda_g / lambda_{g+1}, and that ratio."""
    e = np.clip(np.sort(np.asarray(eig, dtype=np.float64))[::-1], 0, None)
    hi = len(e) // 2
    idx = np.arange(lo, hi + 1)                                  # 1-based g
    r = e[idx - 1] / np.maximum(e[idx], 1e-300)
    j = int(np.argmax(r))
    return int(idx[j]), float(r[j])


def rows(results: str) -> pd.DataFrame:
    out = []
    for path in sorted(glob.glob(os.path.join(results, "gradgram", "*_spectra.npz"))):
        m = re.match(r"(.+)_s(\d+)_spectra\.npz$", os.path.basename(path))
        arm, seed = m.group(1), int(m.group(2))
        if arm not in ARM_K:
            continue
        K = ARM_K[arm]; z = np.load(path)
        for layer in BLOCK5:
            e = z[f"{layer}::out"]
            g, ratio = cliff(e, K)
            out.append({"arm": arm, "seed": seed, "layer": layer, "K": K, "PK": P_BLOCK5 * K, "g": g, "ratio": ratio,
                        "k0.95": k_star(e, 0.95), "k0.999": k_star(e, 0.999)})
    return pd.DataFrame(out)


def judge(results: str, out: str | None = None) -> dict:
    out = out or os.path.join(results, "position_collapse")
    os.makedirs(out, exist_ok=True)
    df = rows(results)
    V: dict = {}
    print("==== B9 part A (plan 9.29): is the cliff of the last block's spectrum at positions x classes = 4K? ====")
    if df.empty:
        print("  no spectra found"); return V
    df.to_csv(os.path.join(out, "cliff.csv"), index=False)
    print(f"    {'arm':16s} {'K':>4s} {'4K':>4s}   cliff index g (ratio), three layers x seeds                         k*(0.999)            k*(0.95)")
    for arm, d in df.groupby("arm", sort=False):
        d = d.sort_values(["seed", "layer"])
        cl = " ".join(f"{g}({r:.1f})" for g, r in zip(d.g, d.ratio))
        print(f"    {arm:16s} {d.K.iloc[0]:4d} {d.PK.iloc[0]:4d}   {cl:60s}  {' '.join(map(str, d['k0.999']))}   {' '.join(map(str, d['k0.95']))}")
    at = lambda d: (d.g == d.PK - 1) | (d.g == d.PK)                                     # noqa: E731
    t = df[df.arm.isin(TRAINED_K10)]
    if len(t) == 18:
        n1 = int((at(t) & (t.ratio >= RATIO_MIN)).sum())
        V["P-b9.1"] = n1 >= 15
        V["P-b9.3"] = bool((t["k0.999"] <= t.PK).all())
        print(f"  P-b9.1 (K = 10, trained to the terminal phase: cliff at 39 or 40 with ratio >= {RATIO_MIN:g}, >= 15/18) -> {'holds' if V['P-b9.1'] else 'fails'} ({n1}/18)")
        print(f"  P-b9.3 (same layers: k*(0.999) <= 40, 18/18) -> {'holds' if V['P-b9.3'] else 'fails'} ({int((t['k0.999'] <= t.PK).sum())}/18)")
    b = df[df.arm.isin(BLIND) & (df.ratio >= RATIO_MIN)]
    if len(df[df.arm.isin(BLIND)]) == 18:
        if len(b) == 0:
            V["P-b9.2"] = None
            print("  P-b9.2 (K = 13, 20: where a cliff of ratio >= 3 exists it sits at 4K - 1 or 4K, >= 80%) -> not judged (no cliff: not collapsed)")
        else:
            n2 = int(at(b).sum())
            V["P-b9.2"] = n2 / len(b) >= 0.8
            print(f"  P-b9.2 (K = 13, 20: where a cliff of ratio >= 3 exists it sits at 4K - 1 or 4K, >= 80%) -> {'holds' if V['P-b9.2'] else 'fails'} ({n2}/{len(b)})")
    json.dump({"verdicts": V}, open(os.path.join(out, "judgement.json"), "w"), indent=1)
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    judge(a.results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
