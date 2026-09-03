"""Recompute every number in notes/review-2026-09-03.md.

That document is a hostile self-review of the round-one results.  Its claims
are only worth anything if they can be re-derived, so this script derives them.
Nothing here is exploratory analysis: it is the audit of an analysis that had
already been run, and it exists because the pre-registered shape test turned
out to be computed on four data points.

    python scripts/review_checks.py --results ../results
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import (binomtest, mannwhitneyu, spearmanr, wilcoxon)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.analyse import PRIMARY, conv_ok, shape   # noqa: E402

K = PRIMARY
TRAINED = ["vgg16_bn", "resnet18", "resnet50", "convnext_tiny"]
ALL = TRAINED + ["resnet50_random", "convnext_tiny_random"]

# Garg, Panda & Roy (IEEE Access 2019) Table 2, VGG-16_BN / CIFAR-10.
GARG_K = np.array([11, 42, 103, 118, 238, 249, 249, 424, 271, 160, 36, 38, 42])
GARG_C = np.array([64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512])


def load(results, model):
    return conv_ok(pd.read_csv(os.path.join(results, f"{model}_layers.csv")))


def exact_spearman_p(rho: float, n: int, direction: int) -> float:
    """P(rho_perm beyond rho) under the null of no association.

    Exact by enumeration for n <= 8, which covers every rho_late in this study
    except ResNet-50's; sampled otherwise.  One-sided, and note that the
    direction was chosen after seeing the sign -- double these for a two-sided
    reading.
    """
    base = np.arange(n)
    if n <= 8:
        rs = np.array([spearmanr(base, np.array(p)).correlation
                       for p in itertools.permutations(base)])
    else:
        rng = np.random.default_rng(0)
        rs = np.array([spearmanr(base, rng.permutation(base)).correlation
                       for _ in range(200_000)])
    return float((rs <= rho + 1e-12).mean() if direction < 0
                 else (rs >= rho - 1e-12).mean())


def section_1(results):
    print("=" * 70)
    print("1. rho_late is computed on the last third of the layers.")
    print("   For VGG-16 that is FOUR points.  Exact permutation p below.")
    print("=" * 70)
    for m in TRAINED:
        d = load(results, m)
        n, cut = len(d), max(3, len(d) // 3)
        for est, col in (("sampled", K), ("pooled", "pooled_" + K)):
            s = shape(d[col].to_numpy())
            rl = s["rho_late"]
            p = exact_spearman_p(rl, cut, -1 if rl < 0 else 1)
            print(f"  {m:14s} {est:8s} layers={n:2d} cut={cut:2d} "
                  f"rho_late={rl:+.3f}  one-sided p={p:.3f}"
                  f"{'  *' if p < 0.05 else ''}")
    n_tests = len(ALL) * 3 * 2
    print(f"\n  {n_tests} rho_late tests in total, no correction applied.")
    print(f"  Bonferroni alpha at 0.05 over {n_tests}: {0.05 / n_tests:.4f}")
    print("  -> nothing survives.")


def section_2(results):
    print("\n" + "=" * 70)
    print("2. Claimed effects against the profile's own layer-to-layer scatter")
    print("=" * 70)
    for m in TRAINED:
        d = load(results, m)
        v = d[K].to_numpy()
        x = np.arange(len(v))
        res = v - np.polyval(np.polyfit(x, v, 1), x)
        adj = np.abs(np.diff(v))
        s = shape(v)
        print(f"  {m:14s} late_drop={s['late_drop']:+.3f}   "
              f"residual SD={res.std(ddof=2):.3f}   "
              f"median |adjacent step|={np.median(adj):.3f}")
    print("\n  Sampling noise, by contrast, is negligible:")
    for m in TRAINED:
        c = pd.read_csv(os.path.join(results, f"{m}_convergence.csv"))
        c = c[c.kind == "conv"]
        ds = []
        for _, g in c.groupby("layer"):
            g = g.sort_values("n_over_C")
            if {100, 250} <= set(g.n_over_C):
                ds.append(abs(g[g.n_over_C == 250][K].iloc[0]
                              - g[g.n_over_C == 100][K].iloc[0]))
        ds = np.array(ds)
        print(f"    {m:14s} |k*/C at n/C=250 minus at n/C=100|: "
              f"median {np.median(ds):.4f}  max {ds.max():.4f}")


def section_3(results):
    print("\n" + "=" * 70)
    print("3. The Garg comparison, over the subset quoted and over all layers")
    print("=" * 70)
    d = load(results, "vgg16_bn")
    ours = d["k_star_ratio_0.999"].to_numpy()
    if not (d.C.to_numpy() == GARG_C).all():
        print("  !! channel alignment differs; the comparison is meaningless")
        return
    theirs = GARG_K / GARG_C
    for label, sl in (("first 8 (as quoted)", slice(0, 8)),
                      ("ALL 13", slice(None))):
        a, b = ours[sl], theirs[sl]
        print(f"  {label:20s} Pearson r={np.corrcoef(a, b)[0, 1]:+.3f}  "
              f"MAD={np.abs(a - b).mean():.3f}  "
              f"Spearman={spearmanr(a, b).correlation:+.3f}")
    print("\n  Ours is ImageNet; theirs is CIFAR-10.  Different data, class")
    print("  count and head.  This is a cross-dataset comparison, never a")
    print("  reproduction -- reproduce_garg.py has not been run.")


def section_5(results):
    print("\n" + "=" * 70)
    print("5. The same claims as PAIRED per-layer tests, using every layer")
    print("=" * 70)

    print("\n  (a) trained > random-init, per layer")
    for m in ("convnext_tiny", "resnet50"):
        a = load(results, m)[K].to_numpy()
        b = load(results, f"{m}_random")[K].to_numpy()
        n = min(len(a), len(b))
        dd = a[:n] - b[:n]
        k = int((dd > 0).sum())
        print(f"    {m:14s} {k}/{n} higher   "
              f"sign p={binomtest(k, n, 0.5, alternative='greater').pvalue:.2e}   "
              f"Wilcoxon p={wilcoxon(dd, alternative='greater').pvalue:.2e}")

    print("\n  (b) H2 as a paired test: does (pooled - sampled) vary with depth?")
    print("      A shape difference REQUIRES this to be non-zero.")
    for m in TRAINED:
        d = load(results, m)
        delta = d["pooled_" + K].to_numpy() - d[K].to_numpy()
        r = spearmanr(np.arange(len(delta)), delta)
        print(f"    {m:14s} n={len(delta):2d}  rho={r.correlation:+.3f}  "
              f"p={r.pvalue:.2e}")

    print("\n  (c) is pooled below sampled at all? -- sign test")
    print("      (layers are correlated, so these p-values are optimistic;")
    print("       the zero exceptions are what carries the claim)")
    tot_k = tot_n = 0
    for m in TRAINED:
        d = load(results, m)
        delta = d["pooled_" + K].to_numpy() - d[K].to_numpy()
        k, n = int((delta < 0).sum()), len(delta)
        tot_k += k
        tot_n += n
        print(f"    {m:14s} pooled lower on {k}/{n}")
    print(f"    {'TOTAL':14s} pooled lower on {tot_k}/{tot_n}")

    print("\n  (d) r_max: does the right denominator dissolve the difference?")
    d = load(results, "resnet50")
    bd, un = d[d.r_max < d.C], d[d.r_max >= d.C]
    print(f"    bounded   n={len(bd):2d}  k*(.999)/C {bd['k_star_ratio_0.999'].median():.3f}"
          f" -> /r_max {bd['k_star_rmax_0.999'].median():.3f}")
    print(f"    unbounded n={len(un):2d}  k*(.999)/C {un['k_star_ratio_0.999'].median():.3f}"
          f" -> /r_max {un['k_star_rmax_0.999'].median():.3f}")
    print(f"    Mann-Whitney by C     : p="
          f"{mannwhitneyu(bd['k_star_ratio_0.999'], un['k_star_ratio_0.999']).pvalue:.2e}")
    print(f"    Mann-Whitney by r_max : p="
          f"{mannwhitneyu(bd['k_star_rmax_0.999'], un['k_star_rmax_0.999']).pvalue:.2e}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    for fn in (section_1, section_2, section_3, section_5):
        fn(a.results)
    print("\n" + "=" * 70)
    print("Survives: r_max (5d), pooling lowers the level (5c),")
    print("          trained != random (5a).")
    print("Does not: every shape claim (1, 2), the 'reproduction' (3),")
    print("          pooling changing the shape (5b).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
