"""Publication figures.

Sized for IEEE two-column (3.5in single, 7.16in double).  Constraints applied
throughout, in this order of priority:

  * Identity is never carried by colour alone -- every series also gets a
    distinct marker and dash pattern, so the figures survive greyscale print
    and colour-vision deficiency.
  * Categorical hues are assigned in a fixed validated order and never cycled.
    Past three series we facet instead of adding a fourth hue.
  * Depth is an *ordered* variable, so where a curve exists per layer we use a
    single-hue light-to-dark ramp, not categorical colours.
  * One y-axis per panel.  Never two scales.

Figures produced:
  fig1_profile          k*/C vs depth, one panel per architecture,
                        trained vs random-init
  fig2_metrics          k*/C vs PR/C vs effective-rank/C on the same layers
                        (this is the panel that reconciles Garg with
                        Elmoznino & Bonner)
  fig3_convergence      the sample-size control: metric vs n/C, per layer
  fig4_threshold        tau sensitivity of k*/C
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical order (checked for CVD separation against a white
# surface).  Used in this order; never cycled.
SERIES = ["#2D5BA8", "#B93A2E", "#00947A", "#B58600"]
MARKERS = ["o", "s", "^", "D"]
DASHES = [(None, None), (5, 2), (1.5, 1.5), (7, 2, 1.5, 2)]

INK = "#101620"
MUTED = "#78838F"
RULE = "#D8DEE6"

# Single-hue sequential ramp for ordered (depth) encoding.
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#B7D3F6", "#2D5BA8", "#0D366B"]
)

ONE_COL = 3.5
TWO_COL = 7.16


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#EDF0F4",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.2,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.pad_inches": 0.02,
    })


def _tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _conv_layers(df: pd.DataFrame, require_ok: bool = True) -> pd.DataFrame:
    """Conv-module outputs only, adequately sampled, ordered by depth.

    Post-activation outputs are a different object (ReLU makes the
    representation non-negative, which changes the covariance structure) and
    are never mixed into the same curve.
    """
    out = df[df["kind"] == "conv"]
    if require_ok and "n_over_C_ok" in out.columns:
        out = out[out["n_over_C_ok"].astype(bool)]
    return out.sort_values("depth_index")


def _relative_depth(sub: pd.DataFrame) -> np.ndarray:
    """Map depth index to [0, 1] so architectures of different length overlay."""
    n = len(sub)
    return np.linspace(0, 1, n) if n > 1 else np.array([0.5])


def _grid(n: int) -> tuple[int, int]:
    """Rows and columns for `n` panels, preferring a full rectangle.

    A 4-panel figure laid out 2x3 leaves two empty cells and, worse, puts a
    subplot title directly under the x-axis label of the panel above it.
    Choosing 2x2 removes both problems.
    """
    if n <= 3:
        return 1, n
    if n == 4:
        return 2, 2
    ncol = 3
    return int(np.ceil(n / ncol)), ncol


# ---------------------------------------------------------------- figure 1

def fig_profile(results: dict[str, pd.DataFrame], out_dir: str,
                metric: str = "k_star_ratio_0.95") -> str:
    """k*/C against relative depth, one panel per architecture.

    Trained and random-init are two series in the same panel, which is the
    comparison that separates learned structure from architecture-plus-input
    statistics.
    """
    _style()
    archs = sorted({k.replace("_random", "") for k in results})
    archs = [a for a in archs if a in results]
    n = len(archs)
    nrow, ncol = _grid(n)

    width = min(TWO_COL, max(ONE_COL, 2.45 * ncol))
    # constrained_layout, not the default: without it a subplot title in row 2
    # is drawn on top of the x-axis label of the panel above it.
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 2.1 * nrow),
                             sharey=True, squeeze=False, constrained_layout=True)

    for i, arch in enumerate(archs):
        ax = axes[i // ncol][i % ncol]
        for j, (key, label) in enumerate(
            ((arch, "trained"), (f"{arch}_random", "random init"))
        ):
            if key not in results:
                continue
            sub = _conv_layers(results[key])
            if sub.empty:
                continue
            ax.plot(
                _relative_depth(sub), sub[metric].to_numpy(),
                color=SERIES[j], marker=MARKERS[j], dashes=DASHES[j],
                label=label, clip_on=False,
            )
        ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (2, 3)))
        ax.set_title(arch, color=INK)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 1)
        _tidy(ax)
        if i % ncol == 0:
            ax.set_ylabel(r"$k^*/C$  (95% EV)")
        ax.set_xlabel("relative depth")

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    axes[0][0].legend(frameon=False, loc="best")
    path = os.path.join(out_dir, "fig1_profile.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 2

def fig_metrics(results: dict[str, pd.DataFrame], out_dir: str) -> str:
    """Three normalised metrics on the same layers.

    This is the reconciliation panel: Garg et al. report k*(0.999), Elmoznino &
    Bonner report participation ratio.  If the two literatures disagree only
    because of the metric, that shows up here as two differently-shaped curves
    over identical layers.
    """
    _style()
    keys = [k for k in sorted(results) if not k.endswith("_random")]
    n = len(keys)
    nrow, ncol = _grid(n)

    metrics = [
        ("k_star_ratio_0.95", r"$k^*/C$ (95%)"),
        ("participation_ratio_norm", "PR$/C$"),
        ("effective_rank_norm", "eff. rank$/C$"),
    ]

    width = min(TWO_COL, max(ONE_COL, 2.45 * ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(width, 2.1 * nrow),
                             sharey=True, squeeze=False, constrained_layout=True)

    for i, key in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        sub = _conv_layers(results[key])
        if sub.empty:
            ax.axis("off")
            continue
        x = _relative_depth(sub)
        for j, (col, label) in enumerate(metrics):
            if col not in sub.columns:
                continue
            ax.plot(x, sub[col].to_numpy(), color=SERIES[j], marker=MARKERS[j],
                    dashes=DASHES[j], label=label, clip_on=False)
        ax.set_title(key, color=INK)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(0, 1)
        _tidy(ax)
        if i % ncol == 0:
            ax.set_ylabel("normalised dimensionality")
        ax.set_xlabel("relative depth")

    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    axes[0][0].legend(frameon=False, loc="best")
    path = os.path.join(out_dir, "fig2_metrics.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 3

def fig_convergence(conv: pd.DataFrame, out_dir: str, model: str,
                    metric: str = "k_star_ratio_0.95") -> str:
    """The sample-size control: metric against n/C, one line per layer.

    Depth is ordered, so layers are coloured by a single-hue light-to-dark
    ramp rather than categorical hues.  A layer whose curve has not flattened
    by the largest available n/C is not a measurement and must not be
    reported.
    """
    _style()
    sub = conv[conv["kind"] == "conv"] if "kind" in conv.columns else conv
    if sub.empty:
        raise ValueError("no conv rows in convergence frame")

    layers = sub.sort_values("depth_index")["layer"].unique()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4))

    for i, layer in enumerate(layers):
        s = sub[sub["layer"] == layer].sort_values("n_over_C")
        if len(s) < 2:
            continue
        ax.plot(s["n_over_C"], s[metric], color=DEPTH_CMAP(i / max(len(layers) - 1, 1)),
                lw=1.0, marker="o", markersize=2.2, clip_on=False)

    ax.set_xscale("log")
    ax.set_xlabel(r"samples per channel, $n/C$")
    ax.set_ylabel(r"$k^*/C$  (95% EV)")
    ax.set_ylim(0, 1.08)
    ax.axvline(50, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    ax.annotate("reporting\nthreshold", xy=(50, 0.06), xytext=(44, 0.06),
                color=MUTED, fontsize=6.5, va="bottom", ha="right")
    _tidy(ax)

    sm = plt.cm.ScalarMappable(cmap=DEPTH_CMAP,
                               norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label("relative depth", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, color=MUTED)

    ax.set_title(model, color=INK)
    path = os.path.join(out_dir, f"fig3_convergence_{model}.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 4

def fig_threshold(results: dict[str, pd.DataFrame], out_dir: str,
                  model: str) -> str:
    """tau sensitivity.  Garg et al. used 0.999, which is looser than 0.95 and
    inflates the ratio; the paper has to show how much of the shape is the
    threshold's doing."""
    _style()
    sub = _conv_layers(results[model])
    taus = [c for c in sub.columns if c.startswith("k_star_ratio_")]
    taus = sorted(taus, key=lambda c: float(c.rsplit("_", 1)[1]))

    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4), constrained_layout=True)
    x = _relative_depth(sub)
    for i, col in enumerate(taus):
        shade = 0.15 + 0.85 * (i / max(len(taus) - 1, 1))
        tau = col.rsplit("_", 1)[1]
        ax.plot(x, sub[col].to_numpy(), color=DEPTH_CMAP(shade), lw=1.2,
                marker=MARKERS[i % len(MARKERS)], markersize=2.6,
                dashes=DASHES[i % len(DASHES)],
                label=rf"$\tau={tau}$", clip_on=False)

    ax.set_xlabel("relative depth")
    ax.set_ylabel(r"$k^*/C$")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    ax.set_title(model, color=INK)
    _tidy(ax)
    path = os.path.join(out_dir, f"fig4_threshold_{model}.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


# ---------------------------------------------------------------- driver

def load(results_dir: str) -> tuple[dict, dict]:
    layers, conv = {}, {}
    for fn in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, fn)
        if fn.endswith("_layers.csv"):
            layers[fn[: -len("_layers.csv")]] = pd.read_csv(path)
        elif fn.endswith("_convergence.csv"):
            conv[fn[: -len("_convergence.csv")]] = pd.read_csv(path)
    return layers, conv


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build the paper figures.")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--allow-undersampled", action="store_true",
                   help="plot rows flagged n_over_C_ok=False (diagnostic only, "
                        "never for the paper)")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    layers, conv = load(args.results)
    if not layers:
        raise SystemExit(f"no *_layers.csv found in {args.results}")

    if args.allow_undersampled:
        global _conv_layers
        _orig = _conv_layers
        _conv_layers = lambda df, require_ok=True: _orig(df, require_ok=False)  # noqa: E731

    made = [fig_profile(layers, args.out), fig_metrics(layers, args.out)]
    for model, cdf in conv.items():
        if not cdf.empty:
            made.append(fig_convergence(cdf, args.out, model))
    for model in layers:
        if not model.endswith("_random"):
            made.append(fig_threshold(layers, args.out, model))

    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
