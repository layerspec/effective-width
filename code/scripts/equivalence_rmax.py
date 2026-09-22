"""D3 (hostile review 2026-09-15, M3.3): effect sizes and an equivalence test for the
bounded-vs-unbounded comparison, replacing "p > 0.01 in all 14".

Section 1(iii) of checkpoint_analysis.py shows that, within each of the 14 trained
checkpoints that have 1x1 expansion layers, the bounded layers (r_max < C) differ
from the unbounded ones in k*(0.999)/C (Mann-Whitney p < 0.01 in 14/14) and do not
under k*(0.999)/r_max (p > 0.01 in 14/14).  The second half uses non-significance
as evidence of equivalence.  This script reports, per model and pooled:

  * the Hodges-Lehmann shift  Delta = median_{i,j}(b_i - u_j)  under C and under
    r_max, with a 95% percentile CI from a layer bootstrap (layers resampled
    within each group; images are not resampled here because the D1 image
    bootstrap put the per-layer k*/C CI at one channel or less);
  * the probability of superiority  A = P(b > u) = U/(n_b n_u), with the same CI
    (A = 1/2 means the two populations are interchangeable);
  * a TOST-style equivalence test under r_max: equivalence at level 0.05 is
    declared when the 90% CI of Delta lies inside (-delta, +delta).  Three margins
    are used: the profile's own adjacent-layer jitter (CLAUDE.md rule 9) and a
    fixed 0.10 (ten per cent of the bound), plus 0.05, the per-layer SD across the
    six ResNet-50 recipes (the between-weights scatter).  delta_min is the smallest
    margin at which equivalence would be declared, i.e. max |90% CI bound|; it can
    never be smaller than the 90% CI half-width, so with 20 bounded layers a margin
    below about 0.03 is out of reach whatever the true shift.

Layers are the observations; within a model they are correlated along depth, so
the CI is honest about between-layer scatter and nothing else, and every count is
reported next to its effect size (rule 9).

    python scripts/equivalence_rmax.py --results ../results [--latex ../paper/table_equivalence.tex]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_analysis import K999, MODEL_LABELS, conv_rows, jitter, load_all, trained_names  # noqa: E402

KR = "k_star_rmax_0.999"
REPS = 4000
FIXED_MARGIN = 0.10          # ten per cent of the bound
RECIPE_MARGIN = 0.05         # the per-layer SD across the six ResNet-50 recipes (CLAUDE.md rule 10)


def hl_shift(b: np.ndarray, u: np.ndarray) -> float:
    return float(np.median(np.subtract.outer(b, u)))


def superiority(b: np.ndarray, u: np.ndarray) -> float:
    U = mannwhitneyu(b, u).statistic
    return float(U / (len(b) * len(u)))


def layer_bootstrap(b: np.ndarray, u: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    sh, au = np.empty(reps), np.empty(reps)
    for r in range(reps):
        bb = b[rng.integers(0, len(b), len(b))]
        uu = u[rng.integers(0, len(u), len(u))]
        sh[r] = hl_shift(bb, uu)
        au[r] = superiority(bb, uu)
    return sh, au


def ci(v: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def per_model(layers: dict, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for m in trained_names(layers):
        d = conv_rows(layers[m], dense_only=False)
        d = d[d.r_max.notna()]
        bd, un = d[d.r_max < d.C], d[d.r_max >= d.C]
        if len(bd) < 3 or len(un) < 3:
            continue
        bC, uC = bd[K999].to_numpy(float), un[K999].to_numpy(float)
        bR, uR = bd[KR].to_numpy(float), un[KR].to_numpy(float)
        shC, _ = layer_bootstrap(bC, uC, reps, seed)
        shR, auR = layer_bootstrap(bR, uR, reps, seed + 1)
        lo90, hi90 = ci(shR, 5, 95)
        jit = jitter(d.sort_values("depth_index")[KR].to_numpy(float))
        rows.append(dict(
            model=m, n_b=len(bd), n_u=len(un),
            med_b_C=float(np.median(bC)), med_u_C=float(np.median(uC)),
            med_b_R=float(np.median(bR)), med_u_R=float(np.median(uR)),
            dC=hl_shift(bC, uC), dC_lo=ci(shC, 2.5, 97.5)[0], dC_hi=ci(shC, 2.5, 97.5)[1],
            dR=hl_shift(bR, uR), dR_lo=ci(shR, 2.5, 97.5)[0], dR_hi=ci(shR, 2.5, 97.5)[1],
            dR_lo90=lo90, dR_hi90=hi90,
            A_C=superiority(bC, uC),
            A_R=superiority(bR, uR), A_R_lo=ci(auR, 2.5, 97.5)[0], A_R_hi=ci(auR, 2.5, 97.5)[1],
            jitter=jit, delta_min=max(abs(lo90), abs(hi90)),
            equiv_jitter=bool(-jit < lo90 and hi90 < jit),
            equiv_fixed=bool(-FIXED_MARGIN < lo90 and hi90 < FIXED_MARGIN),
            equiv_recipe=bool(-RECIPE_MARGIN < lo90 and hi90 < RECIPE_MARGIN),
            p_C=float(mannwhitneyu(bC, uC).pvalue), p_R=float(mannwhitneyu(bR, uR).pvalue),
        ))
    return pd.DataFrame(rows)


def pooled(t: pd.DataFrame, reps: int, seed: int) -> dict:
    """The 14 per-model shifts as observations: median, sign count, bootstrap CI."""
    rng = np.random.default_rng(seed)
    dR = t.dR.to_numpy(float)
    meds = np.array([np.median(dR[rng.integers(0, len(dR), len(dR))]) for _ in range(reps)])
    return dict(
        n=len(dR), median=float(np.median(dR)), lo=ci(meds, 2.5, 97.5)[0], hi=ci(meds, 2.5, 97.5)[1],
        lo90=ci(meds, 5, 95)[0], hi90=ci(meds, 5, 95)[1],
        neg=int((dR < 0).sum()), pos=int((dR > 0).sum()),
        max_abs=float(np.abs(dR).max()), ratio_median=float(np.median(np.abs(dR) / np.abs(t.dC))),
    )


def _lab(m: str) -> str:
    name, wt = MODEL_LABELS.get(m, (m, ""))
    return f"{name} ({wt})" if wt else name


def write_latex(t: pd.DataFrame, P: dict, path: str) -> None:
    L = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Model & $n_b/n_u$ & $\Delta_C$ & $\Delta_{r_{\max}}$ [95\% CI] & $A_{r_{\max}}$ [95\% CI] "
         r"& $\delta_{\min}$ & jitter & eq. \\", r"\midrule"]
    for _, r in t.iterrows():
        eq = "/".join(r"\checkmark" if v else "--" for v in (r.equiv_jitter, r.equiv_recipe, r.equiv_fixed))
        L.append(f"{_lab(r.model)} & {r.n_b}/{r.n_u} & ${r.dC:+.3f}$ & "
                 f"${r.dR:+.3f}$ $[{r.dR_lo:+.3f}, {r.dR_hi:+.3f}]$ & "
                 f"${r.A_R:.2f}$ $[{r.A_R_lo:.2f}, {r.A_R_hi:.2f}]$ & "
                 f"${r.delta_min:.3f}$ & ${r.jitter:.3f}$ & {eq} \\\\")
    L += [r"\midrule",
          f"Median of {P['n']} models & & & ${P['median']:+.3f}$ $[{P['lo']:+.3f}, {P['hi']:+.3f}]$ & & & & \\\\",
          r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    layers = load_all(a.results)
    t = per_model(layers, a.reps, a.seed)
    P = pooled(t, a.reps, a.seed + 7)

    print("D3: bounded (r_max < C) vs unbounded layers within each trained checkpoint,")
    print("k*(0.999) under C and under r_max.  Delta = Hodges-Lehmann shift bounded - unbounded,")
    print(f"95% CI from a layer bootstrap ({a.reps} reps); A = P(bounded > unbounded);")
    print("delta_min = smallest equivalence margin (max |90% CI bound|); eq. = equivalence at")
    print(f"the profile's own jitter / at {RECIPE_MARGIN} (six-recipe per-layer SD) / at {FIXED_MARGIN}.")
    print(f"\n  {'model':40s} {'nb':>3s} {'nu':>3s}  {'Delta_C':>8s}  {'A_C':>4s}  "
          f"{'Delta_rmax [95% CI]':>24s}  {'A_rmax [95% CI]':>18s}  {'d_min':>6s} {'jitter':>6s}  eq(jit/.05/.1)")
    for _, r in t.iterrows():
        print(f"  {r.model:40s} {r.n_b:3d} {r.n_u:3d}  {r.dC:+8.3f}  {r.A_C:4.2f}  "
              f"{r.dR:+7.3f} [{r.dR_lo:+.3f}, {r.dR_hi:+.3f}]  "
              f"{r.A_R:4.2f} [{r.A_R_lo:.2f}, {r.A_R_hi:.2f}]  {r.delta_min:6.3f} {r.jitter:6.3f}  "
              f"{'yes' if r.equiv_jitter else 'no ':3s}/{'yes' if r.equiv_recipe else 'no ':3s}/{'yes' if r.equiv_fixed else 'no'}")
    print(f"\n  under C:     |Delta| range {t.dC.abs().min():.3f}-{t.dC.abs().max():.3f}, "
          f"A range {t.A_C.min():.2f}-{t.A_C.max():.2f}, CI excludes 0 in "
          f"{int(((t.dC_lo > 0) | (t.dC_hi < 0)).sum())}/{len(t)}")
    print(f"  under r_max: |Delta| range {t.dR.abs().min():.3f}-{t.dR.abs().max():.3f}, "
          f"A range {t.A_R.min():.2f}-{t.A_R.max():.2f}, CI excludes 0 in "
          f"{int(((t.dR_lo > 0) | (t.dR_hi < 0)).sum())}/{len(t)}, "
          f"CI covers 0.5 for A in {int(((t.A_R_lo <= 0.5) & (t.A_R_hi >= 0.5)).sum())}/{len(t)}")
    print(f"  |Delta_rmax| / |Delta_C|: median {P['ratio_median']:.3f}")
    print(f"  equivalence at own jitter: {int(t.equiv_jitter.sum())}/{len(t)}; at margin {RECIPE_MARGIN}: "
          f"{int(t.equiv_recipe.sum())}/{len(t)}; at margin {FIXED_MARGIN}: "
          f"{int(t.equiv_fixed.sum())}/{len(t)}; delta_min median {t.delta_min.median():.3f}, "
          f"max {t.delta_min.max():.3f} ({t.loc[t.delta_min.idxmax(), 'model']})")
    print(f"\n  across models (each model's Delta_rmax one observation, n = {P['n']}):")
    print(f"    median {P['median']:+.3f} [95% CI {P['lo']:+.3f}, {P['hi']:+.3f}], 90% CI "
          f"[{P['lo90']:+.3f}, {P['hi90']:+.3f}], negative {P['neg']}, positive {P['pos']}, "
          f"max |Delta| {P['max_abs']:.3f}")
    print("    (the sign is mostly negative: bounded layers sit slightly BELOW the unbounded ones")
    print("     under r_max -- 'fill the bound' is 'to within delta_min', not 'exactly')")
    print("\n  Where the two populations are not equivalent at the jitter margin, the residual")
    print("  shift is reported as such (the note lists them); it is not hidden behind p > 0.01.")

    if a.latex:
        write_latex(t, P, a.latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
