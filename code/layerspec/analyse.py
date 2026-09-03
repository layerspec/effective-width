"""Turn measured CSVs into the answers the analysis plan asked for.

The plan (../notes/analysis-plan.md) fixed, before any data existed:
  * the primary outcome: k*(0.95)/C on conv layers that pass the sample-size
    gate;
  * an operational definition of curve "shape" -- Spearman rho against depth
    over all layers (rho_all) and over the last third (rho_late) -- so that
    "hunchback" and "monotonic rise" are decided by a number rather than by
    looking;
  * five candidate explanations for the Garg / Elmoznino-Bonner disagreement,
    each with a distinct signature in this data.

This module computes exactly those and prints a verdict per hypothesis. It
deliberately does NOT search for other patterns: anything found by reading the
numbers afterwards is exploratory and must be labelled so in the paper.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # scipy is optional; fall back to a rank correlation
    def spearmanr(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if len(ra) < 3:
            return type("R", (), {"correlation": float("nan"),
                                  "pvalue": float("nan")})()
        c = float(np.corrcoef(ra, rb)[0, 1])
        return type("R", (), {"correlation": c, "pvalue": float("nan")})()

PRIMARY = "k_star_ratio_0.95"
SHAPE_METRICS = [
    ("k_star_ratio_0.95", "k*/C (95%)"),
    ("participation_ratio_norm", "PR/C"),
    ("effective_rank_norm", "eff.rank/C"),
]
RISE, FALL = 0.5, -0.5          # thresholds fixed in the analysis plan


# ------------------------------------------------------------------ loading

def load(results_dir: str) -> tuple[dict, dict]:
    layers, conv = {}, {}
    for fn in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, fn)
        if fn.endswith("_layers.csv"):
            layers[fn[: -len("_layers.csv")]] = pd.read_csv(path)
        elif fn.endswith("_convergence.csv"):
            conv[fn[: -len("_convergence.csv")]] = pd.read_csv(path)
    return layers, conv


def conv_ok(df: pd.DataFrame) -> pd.DataFrame:
    """Conv-module rows that pass the pre-registered sample-size gate."""
    out = df[df["kind"] == "conv"]
    if "n_over_C_ok" in out.columns:
        out = out[out["n_over_C_ok"].astype(bool)]
    return out.sort_values("depth_index")


# -------------------------------------------------------------------- shape

def shape(values: np.ndarray) -> dict:
    """rho against depth order, over all layers and over the last third.

    `rho` is rank-based and therefore scale-free: a curve that slips from 0.53
    to 0.44 over its last three layers earns exactly the same rho_late as one
    that falls from 0.97 to 0.07.  The pre-registered verdict is the rho, and
    that is what the paper reports as its test -- but reporting it alone would
    equate a dip with a collapse, so the magnitude of the late change is
    carried alongside it.  These two columns were NOT pre-registered and are
    descriptive only; they qualify the verdict, they do not override it.
    """
    n = len(values)
    if n < 4:
        return {"n": n, "rho_all": np.nan, "rho_late": np.nan,
                "late_drop": np.nan, "peak_to_end": np.nan,
                "verdict": "too few layers"}

    depth = np.arange(n)
    rho_all = spearmanr(depth, values).correlation
    cut = max(3, n // 3)
    rho_late = spearmanr(depth[-cut:], values[-cut:]).correlation

    if rho_late < FALL:
        verdict = "hunchback (late collapse)"
    elif rho_all > RISE and rho_late > 0:
        verdict = "monotonic rise"
    else:
        verdict = "no clear shape"
    return {
        "n": n,
        "rho_all": rho_all,
        "rho_late": rho_late,
        # positive = the curve falls across the window
        "late_drop": float(values[-cut] - values[-1]),
        "peak_to_end": float(np.max(values) - values[-1]),
        "verdict": verdict,
    }


def shape_table(layers: dict) -> pd.DataFrame:
    rows = []
    for model, df in sorted(layers.items()):
        sub = conv_ok(df)
        if sub.empty:
            continue
        for col, label in SHAPE_METRICS:
            if col not in sub.columns:
                continue
            s = shape(sub[col].to_numpy())
            rows.append({"model": model, "metric": label, **s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- hypotheses

def h1_metric(tab: pd.DataFrame) -> str:
    """Do k* and PR disagree in shape on identical layers?

    Verdict is by MAJORITY, not by existence.  An earlier version flagged the
    hypothesis SUPPORTED if any single model disagreed, which made one
    dissenting random-init model outvote five agreeing ones.
    """
    lines = []
    disagree = total = 0
    for model, g in tab.groupby("model"):
        v = dict(zip(g.metric, g.verdict))
        k, pr = v.get("k*/C (95%)"), v.get("PR/C")
        if k and pr:
            differs = k != pr
            disagree += differs
            total += 1
            lines.append(f"    {model:22s} k*: {k:24s} PR: {pr:24s} "
                         f"-> {'DIFFER' if differs else 'agree'}")
    if not total:
        head = "no comparable models"
    elif disagree > total / 2:
        head = (f"SUPPORTED -- the statistics disagree on {disagree}/{total} "
                f"models, so the metric is a plausible explanation")
    elif disagree == 0:
        head = (f"NOT SUPPORTED -- the statistics agree on all {total} models; "
                f"the metric does not explain the disagreement")
    else:
        head = (f"NOT SUPPORTED -- the statistics agree on "
                f"{total - disagree}/{total} models; the {disagree} exception(s) "
                f"are listed and should be discussed, not generalised")
    return f"  H1 metric definition: {head}\n" + "\n".join(lines)


def h2_estimator(layers: dict) -> str:
    """Does the pooled estimator behave differently from the sampled one?"""
    lines, differ, comparable = [], 0, 0
    for model, df in sorted(layers.items()):
        sub = conv_ok(df)
        pooled_col = "pooled_" + PRIMARY
        if sub.empty or pooled_col not in sub.columns:
            continue
        ok = sub.get("pooled_n_over_C_ok")
        n_pool_ok = int(ok.astype(bool).sum()) if ok is not None else -1
        a = shape(sub[PRIMARY].to_numpy())
        b = shape(sub[pooled_col].to_numpy())
        comparable += 1
        differ += a["verdict"] != b["verdict"]
        lines.append(
            f"    {model:22s} sampled: {a['verdict']:24s} "
            f"pooled: {b['verdict']:24s} "
            f"(pooled rows passing the gate: {n_pool_ok}/{len(sub)})"
        )
    if not comparable:
        return "  H2 estimator: no pooled columns found"
    # Majority, for the same reason as H1.
    if differ > comparable / 2:
        head = (f"SUPPORTED -- pooling changes the shape on {differ}/{comparable} "
                f"models")
    elif differ == 0:
        head = "NOT SUPPORTED -- both estimators give the same shape everywhere"
    else:
        head = (f"WEAK -- pooling changes the shape on only {differ}/{comparable} "
                f"models; report per-model, do not generalise")
    return (f"  H2 estimator (pooled vs position-sampled): {head}\n"
            + "\n".join(lines)
            + "\n    NOTE pooled n is the image count, so wide layers are "
              "sample-poor by construction; read the gate column before "
              "believing a difference.")


def h5_sampling(conv: dict, layers: dict) -> str:
    """Is anything still moving at the largest available n/C?"""
    lines = []
    for model, cdf in sorted(conv.items()):
        if cdf.empty or "n_over_C" not in cdf.columns:
            continue
        sub = cdf[cdf["kind"] == "conv"] if "kind" in cdf.columns else cdf
        unstable = []
        for layer, g in sub.groupby("layer"):
            g = g.sort_values("n_over_C")
            if len(g) < 3:
                continue
            last2 = g[PRIMARY].to_numpy()[-2:]
            if abs(last2[1] - last2[0]) > 0.02:      # still moving by >2 points
                unstable.append(layer)
        total = sub["layer"].nunique()
        lines.append(f"    {model:22s} {len(unstable)}/{total} layers still "
                     f"moving at the largest n/C")
    return ("  H5 sample size: layers whose k*/C has NOT flattened "
            "(these must not be reported)\n" + "\n".join(lines))


def trained_vs_random(layers: dict) -> str:
    lines = []
    for model in sorted(layers):
        if model.endswith("_random") or f"{model}_random" not in layers:
            continue
        a = conv_ok(layers[model])[PRIMARY].to_numpy()
        b = conv_ok(layers[f"{model}_random"])[PRIMARY].to_numpy()
        m = min(len(a), len(b))
        if m < 4:
            continue
        d = a[:m] - b[:m]
        lines.append(
            f"    {model:22s} mean diff {d.mean():+.3f}  "
            f"max |diff| {np.abs(d).max():.3f}  "
            f"layers where trained is higher: {int((d > 0).sum())}/{m}"
        )
    if not lines:
        return ("  trained vs random: no matched pairs found "
                "(run the *_random models to enable this control)")
    return ("  trained vs random-init (if indistinguishable, the profile is "
            "architectural and the paper's framing changes)\n" + "\n".join(lines))


# ------------------------------------------------------------------- driver

def report(results_dir: str) -> str:
    layers, conv = load(results_dir)
    if not layers:
        raise SystemExit(f"no *_layers.csv in {results_dir}")

    tab = shape_table(layers)
    out = []
    out.append("=" * 72)
    out.append("PRE-REGISTERED ANALYSIS  --  primary outcome: k*(0.95)/C, "
               "conv layers, gate passed")
    out.append("=" * 72)
    out.append("")
    out.append("Table 2  shape of each curve (rho vs depth; last third separately)")
    out.append(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    out.append("")
    out.append("  rho_all/rho_late and the verdict are pre-registered. "
               "late_drop (value at the start of the")
    out.append("  last third minus the final value) and peak_to_end are "
               "descriptive additions, not")
    out.append("  pre-registered: rho is rank-based and scores a 0.09 dip the "
               "same as a 0.90 collapse,")
    out.append("  so a 'hunchback' verdict must be read together with its "
               "magnitude.")
    out.append("")
    out.append("-" * 72)
    out.append("HYPOTHESES")
    out.append("-" * 72)
    out.append(h1_metric(tab))
    out.append("")
    out.append(h2_estimator(layers))
    out.append("")
    out.append(h5_sampling(conv, layers))
    out.append("")
    out.append(trained_vs_random(layers))
    out.append("")
    out.append("-" * 72)
    out.append("H3 (class count) needs the CIFAR reproduction run; "
               "H4 (truncation) is read off the profile figure.")
    out.append("Anything not listed above is exploratory. Label it so.")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="also write the report here")
    args = p.parse_args(argv)

    text = report(args.results)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
