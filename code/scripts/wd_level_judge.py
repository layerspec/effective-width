"""P-b7.4d' (plan 9.30 B): does the late-layer level fall with the weight decay, among arms of comparable accuracy?

Reads results/gradwindow/wd<lam>_s<seed>_{layers.csv,weights.json} (measure_gradwindow.py).  An arm is eligible when its
test accuracy is within 2.0 points of the same seed's 5e-4 arm; the level is the median k*(0.999)/C of the output
covariance over the last three convolutions (the late_level of P-b7.4d).

    python scripts/wd_level_judge.py --results ../results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

LAMS = ("5e-5", "5e-4", "1e-3", "2e-3", "5e-3")
NEW = ("1e-3", "2e-3")
TOL = 2.0


def load(results: str) -> dict:
    R = {}
    for lam in LAMS:
        for s in (0, 1):
            base = os.path.join(results, "gradwindow", f"wd{lam}_s{s}")
            if os.path.exists(base + "_layers.csv") and os.path.exists(base + "_weights.json"):
                df = pd.read_csv(base + "_layers.csv")
                R[(lam, s)] = dict(level=float(np.median(df.iloc[-3:]["k0.999_out"] / df.iloc[-3:].C)),
                                   acc=json.load(open(base + "_weights.json"))["test_acc"])
    return R


def verdict(R: dict) -> tuple[bool | None, dict]:
    """True / False, or None when not judgeable; and the eligible arms per seed."""
    elig = {}
    for s in (0, 1):
        if ("5e-4", s) not in R:
            return None, elig
        ref = R[("5e-4", s)]["acc"]
        elig[s] = [lam for lam in LAMS if (lam, s) in R and abs(R[(lam, s)]["acc"] - ref) <= TOL]
    if not any(all(lam in elig[s] for s in (0, 1)) for lam in NEW):
        return None, elig
    ok = all(all(R[(a, s)]["level"] > R[(b, s)]["level"] for a, b in zip(elig[s], elig[s][1:])) for s in (0, 1))
    return ok, elig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    R = load(a.results)
    print("==== P-b7.4d' (plan 9.30 B): late level against weight decay ====")
    for (lam, s), r in sorted(R.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        print(f"  lam {lam} seed {s}: late level {r['level']:.3f}  acc {r['acc']}")
    v, elig = verdict(R)
    print(f"  eligible arms (within {TOL} points of the 5e-4 arm): {elig}")
    print("  P-b7.4d' (level strictly decreasing in lambda over the eligible arms, both seeds; at least one new arm eligible in both) -> "
          + ("not judged" if v is None else "holds" if v else "fails"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
