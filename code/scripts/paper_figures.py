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


def _traj_curves(path_in: str):
    d = pd.read_csv(path_in)
    d = d[(d.kind == "conv") & (d.n_over_C >= 50)].copy()
    if "is_depthwise" in d:
        d = d[~d.is_depthwise.astype(bool)]
    d[COL] = d["k_star_0.95"] / d["r_max"]
    epochs = sorted(d.epoch.unique())
    P = {e: d[d.epoch == e].sort_values("depth_index")[COL].to_numpy() for e in epochs}
    acc = d.groupby("epoch")["test_acc"].first()
    init, final = P[epochs[0]], P[epochs[-1]]
    return (np.array(epochs), np.array([_rho(init, P[e]) for e in epochs]),
            np.array([_rho(final, P[e]) for e in epochs]), np.array([np.median(P[e]) for e in epochs]),
            np.array([acc[e] / 100 for e in epochs]))


def fig_trajectory_grid(results: str, out: str) -> str | None:
    """Fig. 5 as four panels: VGG-16 (one seed) and the three controlled
    architectures of round 3 (median over three seeds, thin lines per seed)."""
    groups = [("VGG-16 (BN), one seed", [os.path.join(results, "trajectory", "trajectory_layers.csv")]),
              ("basic ResNet [6,6,6,6], 3 seeds", sorted(glob.glob(os.path.join(results, "round3", "a2_basic52_s*", "trajectory_layers.csv")))),
              ("bottleneck ResNet-50, 3 seeds", sorted(glob.glob(os.path.join(results, "round3", "a2_bottleneck_s*", "trajectory_layers.csv")))),
              ("MobileNetV2, 3 seeds", sorted(glob.glob(os.path.join(results, "round3", "a2_mobilenetv2_s*", "trajectory_layers.csv"))))]
    if not all(paths and all(os.path.exists(p) for p in paths) for _, paths in groups):
        return None
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(2 * ONE_COL + 0.3, 4.2), constrained_layout=True, sharex=True, sharey=True)
    for ax, (title, paths) in zip(axes.ravel(), groups):
        runs = [_traj_curves(p) for p in paths]
        ep = runs[0][0]
        series = [np.median(np.stack([r[i] for r in runs]), axis=0) for i in (1, 2, 3, 4)]
        specs = [(SERIES[0], "o", None, r"$\rho$ vs initial profile"), (SERIES[1], "s", (5, 2), r"$\rho$ vs final profile"),
                 (SERIES[2], "^", (1.5, 1.5), "median level"), (INK, None, (7, 2, 1.5, 2), "test accuracy")]
        for r in runs if len(runs) > 1 else []:
            for i, (c, _, _, _) in zip((1, 2, 3, 4), specs):
                ax.plot(r[0], r[i], color=c, lw=0.5, alpha=0.35)
        for y, (c, m, dsh, lab) in zip(series, specs):
            ax.plot(ep, y, color=c, marker=m, ms=3, lw=1.2 if m else 0.9, dashes=dsh if dsh else (None, None), label=lab)
        ax.axhline(0, color=INK, lw=0.4, alpha=0.4)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlim(0, ep.max()); ax.set_ylim(-0.7, 1.02)
        _tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho$ / level / accuracy")
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=6.5)
    path = os.path.join(out, "fig5_trajectory_grid.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=150)
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


def fig_decompose(results: str, out: str) -> list[str]:
    """Per-layer decomposition for ResNet-50, trained (tv V1) and at
    initialisation: measured k*/r_max, kernel-only (white patches), and the
    orthogonalised-kernel counterfactual."""
    made = []
    for tag, fn in (("trained", "resnet50_decompose.csv"), ("random", "resnet50_random_decompose.csv")):
        path_in = os.path.join(results, "decompose", fn)
        if not os.path.exists(path_in):
            continue
        _style()
        d = pd.read_csv(path_in).sort_values("depth_index")
        x = np.arange(1, len(d) + 1)
        fig, ax = plt.subplots(figsize=(ONE_COL, 2.6), constrained_layout=True)
        ax.plot(x, d["kernel_0.95"], color=SERIES[1], marker="s", dashes=(5, 2), label=r"kernel only ($WW^\top$)")
        ax.plot(x, d["ortho_0.95"], color=SERIES[2], marker="^", dashes=(1.5, 1.5), label=r"orthogonalised kernel")
        ax.plot(x, d["data_0.95"], color=MUTED, lw=0.8, label=r"patch covariance (capped)")
        ax.plot(x, d["out_0.95"], color=SERIES[0], marker="o", label=r"measured ($W\Sigma W^\top$)")
        ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (2, 3)))
        ax.set_xlim(1, len(d)); ax.set_ylim(0, 1.05)
        ax.set_xlabel("convolutional layer (depth order)")
        ax.set_ylabel(r"$k^*(0.95)/r_{\max}$")
        ax.set_title(f"ResNet-50, {tag}", color=INK)
        _tidy(ax)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6)
        path = os.path.join(out, f"fig6_decompose_{tag}.pdf")
        fig.savefig(path); fig.savefig(path.replace(".pdf", ".png")); plt.close(fig)
        made.append(path)
    return made


def fig_blockout(results: str, out: str) -> str | None:
    """Block-level depth profile for the six ResNet-50 recipes: k*(0.999)/C at
    the conv3 (1x1 expansion) output, capped at 0.25 by r_max, against the
    same statistic at the block output (after the shortcut and the ReLU),
    which is the tensor Ansuini et al. and Elmoznino & Bonner read."""
    from layerspec.figures import load
    layers, _ = load(results)
    recipes = ["timm_resnet50.tv_in1k", "timm_resnet50.tv2_in1k", "timm_resnet50.a1_in1k",
               "timm_resnet50.a3_in1k", "timm_resnet50.gluon_in1k", "timm_resnet50.fb_ssl_yfcc100m_ft_in1k"]
    recipes = [r for r in recipes if r in layers]
    if not recipes:
        return None
    _style()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3), constrained_layout=True)
    for i, r in enumerate(recipes):
        d = layers[r]
        blk = d[d.kind == "block"].sort_values("depth_index")
        c3 = d[(d.kind == "conv") & d.layer.str.endswith(".conv3")].sort_values("depth_index")
        x = np.arange(1, len(blk) + 1)
        ax.plot(x, blk["k_star_ratio_0.999"], color=SERIES[0], lw=0.9, alpha=0.8, marker="o", markersize=2,
                label="block output" if i == 0 else None)
        ax.plot(x, c3["k_star_ratio_0.999"], color=SERIES[1], lw=0.9, alpha=0.8, marker="s", markersize=2,
                dashes=(5, 2), label="conv3 output" if i == 0 else None)
    ax.axhline(0.25, color=MUTED, lw=0.7, ls=(0, (2, 3)))
    ax.text(len(blk) + 0.3, 0.25, r"$r_{\max}/C$", color=MUTED, fontsize=6, va="center")
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (2, 3)))
    ax.set_xlim(1, len(blk)); ax.set_ylim(0, 1.05)
    ax.set_xlabel("bottleneck block (depth order)")
    ax.set_ylabel(r"$k^*(0.999)/C$")
    _tidy(ax)
    ax.legend(frameon=False, loc="center right", fontsize=6)
    path = os.path.join(out, "fig7_blockout.pdf")
    fig.savefig(path); fig.savefig(path.replace(".pdf", ".png")); plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default="../results/figures")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for f in [fig_profile_r50(a.results, a.out), fig_metrics_subset(a.results, a.out),
              fig_threshold_vgg(a.results, a.out)] + fig_trajectory(a.results, a.out) + [fig_trajectory_grid(a.results, a.out)] + fig_decompose(a.results, a.out) + [fig_blockout(a.results, a.out)]:
        if f:
            print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
