"""Proposition 1 (finite-sample bias of the captured fraction): the Lawley term against the delta-method correction for the
ratio F^ = S_k / tr Sigma^ (review finding 16, 2026-10-01).  Gaussian samples, centred sample covariance, three spectra.

    python scripts/prop1_delta_check.py [--reps 1500] [--n-over-C 50]
"""
import argparse

import numpy as np


def terms(lam, n, k):
    C = len(lam); t = lam.sum(); F = lam[:k].sum() / t
    L = (1 / (n * t)) * sum(lam[i] * lam[j] / (lam[i] - lam[j]) for i in range(k) for j in range(k, C))
    var_T = 2 * (lam ** 2).sum() / n; cov = 2 * (lam[:k] ** 2).sum() / n
    D = (-cov + F * var_T) / t ** 2
    return F, L, D


def simulate(lam, n, reps, ks, seed=0):
    rng = np.random.default_rng(seed); C = len(lam); A = np.sqrt(lam)
    rows = []
    acc = {k: [] for k in ks}
    for _ in range(reps):
        X = rng.standard_normal((n, C)) * A; X = X - X.mean(0)
        l = np.sort(np.linalg.eigvalsh(X.T @ X / (n - 1)))[::-1]; cs = np.cumsum(l)
        for k in ks:
            acc[k].append(cs[k - 1] / cs[-1])
    for k in ks:
        F, L, D = terms(lam, n, k); Fh = np.asarray(acc[k])
        rows.append(dict(k=k, bias=Fh.mean() - F, se=Fh.std() / np.sqrt(reps), lawley=L, delta=D, total=L + D))
    return rows


SPECTRA = {
    "power law alpha=1.5": np.arange(1, 65.) ** -1.5,
    "power law alpha=1.0": np.arange(1, 65.) ** -1.0,
    "factor-2 gap at k=8": np.r_[np.linspace(2, 1, 8), 0.5 * np.arange(1, 57.) ** -1.2],
}


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--reps", type=int, default=1500); p.add_argument("--n-over-C", type=int, default=50)
    a = p.parse_args(argv)
    for name, lam in SPECTRA.items():
        n = a.n_over_C * len(lam)
        print(f"== {name}  C={len(lam)} n={n} reps={a.reps}")
        print(f"{'k':>3} {'E F^ - F':>10} {'se':>8} {'Lawley':>10} {'delta':>10} {'sum':>10} {'delta/Lawley':>13}")
        for r in simulate(lam, n, a.reps, [4, 8, 16, 32]):
            print(f"{r['k']:3d} {r['bias']:10.2e} {r['se']:8.1e} {r['lawley']:10.2e} {r['delta']:10.2e} {r['total']:10.2e} {r['delta'] / r['lawley']:13.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
