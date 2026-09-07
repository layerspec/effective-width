"""Figures for the reframed paper (2026-09-07) that layerspec.figures does not
make: they come from the local 6,400-image runs rather than results/*.csv.

  fig1_profile      k*(0.95)/r_max vs depth, ResNet-50: six trained recipes
                    (blue) and five random initialisations (grey), same images
  fig2_metrics      three statistics on identical layers, three representative
                    models (the 22-panel version from layerspec.figures is
                    unreadable at column width)
  fig4_threshold    tau sensitivity, VGG-16_BN (layout fix)
  fig5_trajectory   VGG-16_BN / CIFAR-10: rho vs init, rho vs final, level and
                    test accuracy against epoch

Run AFTER `python -m layerspec.figures`, which overwrites fig1/fig2 with the
first-round versions.

    cd code && python3 scripts/paper_figures.py --results ../results --out ../results/figures
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.figures import (INK, MUTED, ONE_COL, SERIES, _style, _tidy,   # noqa: E402
                               fig_metrics, fig_threshold, load, plt)
from scripts.checkpoint_analysis import _load_local, _rho                   # noqa: E402

COL = "k_star_rmax_0.95"


def fig_profile_r50(results: str, out: str) -> str:
    _style()
    base = os.path.join(results, "local6400")
    trained = sorted(glob.glob(os.path.join(base, "trained", "*resnet50*_layers.csv")))
    seeds = sorted(glob.glob(os.path.join(base, "seed*", "resnet50_random_layers.csv")))
    T = [_load_local(f) for f in trained]
    R = [_load_local(f) for f in seeds]
    T = [x[(x.kind == "conv") & x.n_over_C_ok] for x in T]
    R = [x[(x.kind == "conv") & x.n_over_C_ok] for x in R]
    L = len(T[0])
    x = np.arange(1, L + 1)
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3), constrained_layout=True)
    for i, r in enumerate(R):
        ax.plot(x, r[COL].to_numpy(), color=MUTED, lw=0.8, alpha=0.7,
                label="random init (5 seeds)" if i == 0 else None)
    for i, t in enumerate(T):
        ax.plot(x, t[COL].to_numpy(), color=SERIES[0], lw=0.9, alpha=0.85,
                label="trained (6 recipes)" if i == 0 else None)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (2, 3)))
    ax.set_xlim(1, L)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("convolutional layer (depth order)")
    ax.set_ylabel(r"$k^*(0.95)/r_{\max}$")
    _tidy(ax)
    ax.legend(frameon=False, loc="upper left")
    path = os.path.join(out, "fig1_profile.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


def fig_trajectory(results: str, out: str) -> list[str]:
    """One panel per results/trajectory*/ run; the VGG seed-0 run keeps the
    name fig5_trajectory.pdf that the paper inputs."""
    made = []
    for path_in in sorted(glob.glob(os.path.join(results, "trajectory*", "trajectory_layers.csv"))):
        name = os.path.basename(os.path.dirname(path_in))
        suffix = "" if name == "trajectory" else "_" + name.replace("trajectory_", "")
        made.append(_fig_trajectory_one(path_in, os.path.join(out, f"fig5_trajectory{suffix}.pdf")))
    return made


def _fig_trajectory_one(path_in: str, path: str) -> str:
    _style()
    d = pd.read_csv(path_in)
    d = d[(d.kind == "conv") & (d.n_over_C >= 50)].copy()
    d[COL] = d["k_star_0.95"] / d["r_max"]
    epochs = sorted(d.epoch.unique())
    P = {e: d[d.epoch == e].sort_values("depth_index")[COL].to_numpy() for e in epochs}
    acc = d.groupby("epoch")["test_acc"].first() if "test_acc" in d else None
    init, final = P[epochs[0]], P[epochs[-1]]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3), constrained_layout=True)
    ax.plot(epochs, [_rho(init, P[e]) for e in epochs], color=SERIES[0], marker="o",
            label=r"$\rho$ vs initial profile")
    ax.plot(epochs, [_rho(final, P[e]) for e in epochs], color=SERIES[1], marker="s",
            dashes=(5, 2), label=r"$\rho$ vs final profile")
    ax.plot(epochs, [np.median(P[e]) for e in epochs], color=SERIES[2], marker="^",
            dashes=(1.5, 1.5), label="median level")
    if acc is not None:
        ax.plot(epochs, [acc[e] / 100 for e in epochs], color=INK, lw=0.8, dashes=(7, 2, 1.5, 2),
                label="test accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\rho$ / level / accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(epochs))
    _tidy(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    return path


def fig_metrics_subset(results: str, out: str) -> str:
    layers, _ = load(results)
    pick = {"VGG-16 (BN)": "vgg16_bn", "ResNet-18": "resnet18",
            "ResNet-50": "timm_resnet50.tv_in1k"}
    sub = {}
    for k, v in pick.items():
        d = layers[v].copy()
        d["pr_rmax"] = d["participation_ratio"] / d["r_max"]
        d["erank_rmax"] = d["effective_rank"] / d["r_max"]
        sub[k] = d
    return fig_metrics(sub, out, metrics=[("k_star_rmax_0.95", r"$k^*/r_{\max}$ (95%)"),
                                          ("pr_rmax", r"PR$/r_{\max}$"),
                                          ("erank_rmax", r"eff. rank$/r_{\max}$")],
                       ylabel=r"statistic $/\,r_{\max}$")


def fig_threshold_vgg(results: str, out: str) -> str:
    layers, _ = load(results)
    return fig_threshold(layers, out, "vgg16_bn")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default="../results/figures")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for f in [fig_profile_r50(a.results, a.out), fig_metrics_subset(a.results, a.out),
              fig_threshold_vgg(a.results, a.out)] + fig_trajectory(a.results, a.out):
        print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
