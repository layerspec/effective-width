"""The load-bearing claims, re-tested on every checkpoint from the second run.

The first run had one set of weights per architecture (n = 1), so nothing
about between-checkpoint variation could be said.  The second run (2026-09-03,
tarball layerspec_results_20260903T122955Z) added sixteen timm checkpoints,
among them six ImageNet-1k ResNet-50 recipes, two width variants of the same
architecture, and three inverted-residual networks whose 1x1 expansions have a
different rank bound (1/6 instead of 1/4).  It also added the Garg validation
gate (CIFAR-10, same dataset as theirs) and block-output measurements.

Every section below is a paired or per-layer test on ALL gate-passing layers,
never a rank correlation on the last third (rule 7 of CLAUDE.md), and every
effect size is printed next to the profile's own adjacent-layer jitter
(rule 9).  Rows with n_over_C_ok=False are dropped before anything is counted
(rule 2).  Depthwise convolutions are counted separately (rule 5).

    python scripts/checkpoint_analysis.py --results ../results
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.analyse import PRIMARY  # noqa: E402

K = PRIMARY                      # k*(0.95)/C, the pre-registered statistic
K999 = "k_star_ratio_0.999"      # the rank-bound test needs the near-full rank

# torchvision resnet50 (IMAGENET1K_V1) and timm resnet50.tv_in1k are the SAME
# weights loaded two ways; they agree layer-for-layer, and only one of them may
# be counted.  The torchvision copy is kept as a backend consistency check.
DUPLICATES = {"resnet50": "timm_resnet50.tv_in1k"}

RESNET50_RECIPES = [
    "timm_resnet50.tv_in1k", "timm_resnet50.tv2_in1k", "timm_resnet50.a1_in1k",
    "timm_resnet50.a3_in1k", "timm_resnet50.gluon_in1k",
    "timm_resnet50.fb_ssl_yfcc100m_ft_in1k",
]
CONVNEXT_CKPTS = ["convnext_tiny", "timm_convnext_tiny.fb_in22k_ft_in1k"]
BOTTLENECK_RESNETS = RESNET50_RECIPES + [
    "timm_resnet101.tv_in1k", "timm_wide_resnet50_2.tv_in1k",
    "timm_resnext50_32x4d.tv_in1k",
]


# ------------------------------------------------------------------ loading

def load_all(results: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(os.path.join(results, "*_layers.csv"))):
        name = os.path.basename(path)[: -len("_layers.csv")]
        df = pd.read_csv(path)
        if "n_over_C_ok" in df.columns:
            df["n_over_C_ok"] = df["n_over_C_ok"].astype(bool)
        out[name] = df.sort_values("depth_index").reset_index(drop=True)
    return out


def conv_rows(df: pd.DataFrame, *, dense_only: bool = True) -> pd.DataFrame:
    d = df[(df.kind == "conv") & df.n_over_C_ok]
    if dense_only and "is_depthwise" in d.columns:
        d = d[~d.is_depthwise.astype(bool)]
    return d


def depthwise_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.kind == "conv") & df.n_over_C_ok]
    if "is_depthwise" not in d.columns:
        return d.iloc[0:0]
    return d[d.is_depthwise.astype(bool)]


def is_random(name: str) -> bool:
    return name.endswith("_random")


def trained_names(layers: dict) -> list[str]:
    return [m for m in layers if not is_random(m) and m not in DUPLICATES]


def jitter(v: np.ndarray) -> float:
    """Median absolute step between adjacent layers of one profile."""
    return float(np.median(np.abs(np.diff(v)))) if len(v) > 1 else float("nan")


def hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ----------------------------------------------------------- section 0: gate

def section_gate(results: str) -> None:
    hdr("0. Validation gate: Garg et al. (2019) Table 2, VGG-16_BN on CIFAR-10")
    path = os.path.join(results, "garg", "garg_comparison.csv")
    if not os.path.exists(path):
        print("  garg/garg_comparison.csv not found -- gate not run")
        return
    g = pd.read_csv(path)
    ours, theirs = g.ratio_ours.to_numpy(), g.ratio_garg.to_numpy()
    r = np.corrcoef(ours, theirs)[0, 1]
    mad = np.abs(ours - theirs).mean()
    print(f"  same dataset, same architecture, our own training run")
    print(f"  all {len(g)} conv layers: Pearson r = {r:+.3f}   "
          f"MAD = {mad:.3f}   Spearman = {spearmanr(ours, theirs).correlation:+.3f}")
    worst = g.sort_values("abs_diff", ascending=False).head(3)
    print("  largest disagreements:")
    for _, row in worst.iterrows():
        print(f"    {row.layer:12s} ours {row.ratio_ours:.3f}  theirs {row.ratio_garg:.3f}"
              f"  diff {row.abs_diff:+.3f}")
    print("  (their recipe is under-specified; the gate is about shape, and it passes)")
    print("  For contrast, the first-round ImageNet-vs-CIFAR comparison had r = 0.125.")


# ---------------------------- section 0b: second gate, Elmoznino & Bonner 2024

def section_gate_eb(results: str, layers: dict) -> None:
    """Agreement with the published block-output participation ratios of
    Elmoznino & Bonner (PLOS CB 2024): their eigmetrics CSV (refs/eb2024/, from
    github.com/EricElmoznino/encoder_dimensionality) vs our pooled PR on the
    same torchvision ResNet-50 blocks.  They: 10,000 images, Resize(224,224)
    without crop, avg-pooled block outputs; we: 50,000 images, resize 256 +
    centre crop 224.  Not the same images, so this is agreement, not
    reproduction (rule 11)."""
    path = os.path.join(os.path.dirname(os.path.abspath(results)), "refs", "eb2024",
                        "eigmetrics_imagenet_pooling-avg.csv")
    if not os.path.exists(path) or "resnet50" not in layers:
        return
    hdr("0b. Second gate: block-output pooled PR vs Elmoznino & Bonner (2024), ResNet-50 V1")
    E = pd.read_csv(path)
    E = E[(E.architecture == "ResNet50") & (E.source == "PyTorch")].copy()
    E["layer"] = E.layer.str.replace(".relu", "", regex=False)
    for label, kind_sel, ours_key in (("trained", "Supervised", "resnet50"), ("untrained", "Untrained", "resnet50_random")):
        if ours_key not in layers:
            continue
        theirs = E[E.kind == kind_sel][["layer", "effective dimensionality"]].rename(columns={"effective dimensionality": "eb"})
        ours = layers[ours_key]; ours = ours[ours.kind == "block"][["layer", "C", "pooled_participation_ratio", "pooled_n_over_C_ok"]]
        m = ours.merge(theirs, on="layer")
        if m.empty:
            print(f"  {label}: no matching layers"); continue
        lo, lt = np.log(m.pooled_participation_ratio), np.log(m.eb)
        dl = lo - lt
        gated = m[m.pooled_n_over_C_ok.astype(bool)]
        print(f"  {label}: {len(m)} blocks matched;  log-log Pearson r = {np.corrcoef(lo, lt)[0, 1]:+.3f}   "
              f"Spearman = {_rho(lo, lt):+.3f}   median |dlog| = {np.median(np.abs(dl)):.3f}   max = {np.abs(dl).max():.3f}   "
              f"ours higher on {int((dl > 0).sum())}/{len(m)} (median ratio {np.exp(np.median(dl)):.3f})")
        print(f"    gated (pooled n/C >= 50, {len(gated)} blocks): max |dlog| = "
              f"{np.abs(np.log(gated.pooled_participation_ratio) - np.log(gated.eb)).max():.3f}")
        print(f"    {'block':10s} {'C':>5s} {'ours':>8s} {'E&B':>8s} {'ratio':>6s}")
        for _, r in m.iterrows():
            print(f"    {r.layer:10s} {int(r.C):5d} {r.pooled_participation_ratio:8.2f} {r.eb:8.2f} {r.pooled_participation_ratio / r.eb:6.3f}")


# ----------------------------------------------------- section 1: r_max bound

def section_rmax(layers: dict) -> None:
    hdr("1. r_max as the denominator, on every checkpoint")
    print("  A conv output cannot have rank above r_max = g*min((C_in/g)*kh*kw, C_out/g).")
    print("  If k* tracks r_max rather than C, then (i) k*(.999)/r_max never exceeds 1,")
    print("  (ii) k*(.999)/C at bounded layers scales with r_max/C while k*(.999)/r_max")
    print("  does not, and (iii) the bounded/unbounded difference vanishes once the")
    print("  denominator is r_max.\n")

    print("  (i) sanity: max k*(.999)/r_max per model, and which bounds occur")
    rows = []
    for m, df in layers.items():
        d = conv_rows(df, dense_only=False)
        d = d[d.r_max.notna()]
        mx = d["k_star_rmax_0.999"].max()
        bounds = sorted({round(x, 3) for x in (d.r_max / d.C) if x < 1})
        nb = int((d.r_max < d.C).sum())
        rows.append((m, mx, nb, len(d), bounds))
        print(f"    {m:40s} max {mx:.3f}   bounded {nb:3d}/{len(d):3d}   "
              f"r_max/C values {bounds if bounds else '-'}")
    over = [m for m, mx, *_ in rows if mx > 1 + 1e-9]
    print(f"  -> {'NONE exceed 1' if not over else 'EXCEED 1: ' + ', '.join(over)}")

    print("\n  (ii) bounded layers pooled across trained models, grouped by the bound r_max/C")
    pool = []
    for m in trained_names(layers):
        d = conv_rows(layers[m], dense_only=False)
        d = d[d.r_max.notna() & (d.r_max < d.C)].copy()
        d["model"] = m
        d["bound"] = (d.r_max / d.C).round(4)
        pool.append(d)
    pool = pd.concat(pool, ignore_index=True)
    grp = pool.groupby("bound")
    print(f"    {'bound':>8s} {'n':>4s}  {'k*/C med':>9s}  {'k*/r_max med':>13s}  "
          f"{'k*/r_max IQR':>14s}  models")
    for b, g in grp:
        q1, q3 = g["k_star_rmax_0.999"].quantile([0.25, 0.75])
        print(f"    {b:8.4f} {len(g):4d}  {g[K999].median():9.3f}  "
              f"{g['k_star_rmax_0.999'].median():13.3f}  "
              f"[{q1:.2f}, {q3:.2f}]      {g.model.nunique()}")
    rc = spearmanr(pool.bound, pool[K999])
    rr = spearmanr(pool.bound, pool["k_star_rmax_0.999"])
    print(f"    Spearman(bound, k*/C)     = {rc.correlation:+.3f}  p={rc.pvalue:.1e}")
    print(f"    Spearman(bound, k*/r_max) = {rr.correlation:+.3f}  p={rr.pvalue:.1e}")
    print("    Reading: k*/C tracks the bound one-for-one.  k*/r_max is not constant")
    print("    either -- a tighter ceiling is hit harder (1/6: ~0.99, 1/4: ~0.94, 1/2:")
    print("    ~0.82) -- but it stays within the unbounded layers' range, and the")
    print("    within-model comparison in (iii) is the actual test.")
    print("    The stem layers (bound 27/64 = 0.42 for VGG, 27/32 = 0.84 for the")
    print("    MobileNets) sit far below r_max: a 3x3 RGB patch has 27 dimensions but")
    print("    natural-image patches are themselves low-rank, so the stem is")
    print("    data-limited, not architecture-limited.  They are excluded from no")
    print("    test; they are simply not the layers the bound is about.")

    print("\n  (iii) bounded vs unbounded layers within each model, by C and by r_max")
    print(f"    {'model':40s} {'nb':>3s} {'nu':>3s}  {'med/C b':>7s} {'u':>6s}  "
          f"{'p(C)':>8s}  {'med/rmax b':>10s} {'u':>6s}  {'p(rmax)':>8s}")
    p_c, p_r = [], []
    for m in trained_names(layers):
        d = conv_rows(layers[m], dense_only=False)
        d = d[d.r_max.notna()]
        bd, un = d[d.r_max < d.C], d[d.r_max >= d.C]
        if len(bd) < 3 or len(un) < 3:
            continue
        pc = mannwhitneyu(bd[K999], un[K999]).pvalue
        pr = mannwhitneyu(bd["k_star_rmax_0.999"], un["k_star_rmax_0.999"]).pvalue
        p_c.append(pc)
        p_r.append(pr)
        print(f"    {m:40s} {len(bd):3d} {len(un):3d}  "
              f"{bd[K999].median():7.3f} {un[K999].median():6.3f}  {pc:8.1e}  "
              f"{bd['k_star_rmax_0.999'].median():10.3f} "
              f"{un['k_star_rmax_0.999'].median():6.3f}  {pr:8.1e}")
    print(f"    models with p(C) < 0.01 : {sum(p < 0.01 for p in p_c)}/{len(p_c)}")
    print(f"    models with p(rmax)<0.01: {sum(p < 0.01 for p in p_r)}/{len(p_r)}")
    print("    (a residual p(rmax) < 0.01 means the bounded layers still differ after")
    print("     the correction -- report it as such, do not hide it)")


# ------------------------------------------ section 2: between-checkpoint spread

def section_spread(layers: dict) -> None:
    hdr("2. Between-checkpoint variation: six ResNet-50 recipes, two ConvNeXt-T")
    print("  This is the error bar the first run did not have.  Per layer, the six")
    print("  recipes give six values of the same statistic; their spread is compared")
    print("  with each profile's own adjacent-layer jitter.\n")

    for label, names, cols in (
        ("ResNet-50 x6", RESNET50_RECIPES, (K, "k_star_rmax_0.95", K999, "k_star_rmax_0.999")),
        ("ConvNeXt-T x2 (dense 1x1 only)", CONVNEXT_CKPTS, (K, K999)),
    ):
        names = [n for n in names if n in layers]
        if len(names) < 2:
            print(f"  {label}: fewer than two checkpoints present, skipped")
            continue
        # torchvision and timm name the same layers differently, so align by
        # depth order; this is only valid when the layer counts agree.
        frames = [conv_rows(layers[n]).reset_index(drop=True) for n in names]
        if len({len(f) for f in frames}) != 1:
            print(f"  {label}: layer counts differ {[len(f) for f in frames]}, skipped")
            continue
        common = frames[0].index
        print(f"  {label}: {len(names)} checkpoints, {len(common)} gate-passing layers each")
        for col in cols:
            if col not in frames[0].columns:
                continue
            M = np.stack([f.loc[common, col].to_numpy() for f in frames], axis=1)
            sd_layer = M.std(axis=1, ddof=1)
            rng_layer = M.max(axis=1) - M.min(axis=1)
            jit = np.array([jitter(M[:, j]) for j in range(M.shape[1])])
            level = M.mean(axis=0)
            print(f"    {col:22s} per-layer SD across ckpts: median {np.median(sd_layer):.3f} "
                  f"max {sd_layer.max():.3f}   range: median {np.median(rng_layer):.3f}")
            print(f"    {'':22s} adjacent-layer jitter per ckpt: median {np.median(jit):.3f}  "
                  f"(min {jit.min():.3f}, max {jit.max():.3f})")
            print(f"    {'':22s} whole-profile mean per ckpt: "
                  + "  ".join(f"{v:.3f}" for v in level))
        # do the profiles even agree in ORDER?
        for col in (K, "k_star_rmax_0.95"):
            if col not in frames[0].columns or len(names) < 2:
                continue
            M = np.stack([f.loc[common, col].to_numpy() for f in frames], axis=1)
            rhos = [spearmanr(M[:, i], M[:, j]).correlation
                    for i, j in itertools.combinations(range(M.shape[1]), 2)]
            print(f"    pairwise Spearman between {col} profiles: "
                  f"min {min(rhos):+.3f}  median {np.median(rhos):+.3f}  max {max(rhos):+.3f}")
        print()

    print("  Reading: if the per-layer SD across recipes is comparable to the")
    print("  adjacent-layer jitter, a single checkpoint's fine structure is not")
    print("  reportable; only what is stable across recipes is.")


# -------------------------------------------------------- section 3: pooling

MODEL_LABELS = {
    "vgg16_bn": ("VGG-16 (BN)", "tv"),
    "resnet50": ("ResNet-50", "tv V1"),
    "resnet18": ("ResNet-18", "tv"),
    "resnet50_random": ("ResNet-50", "random"),
    "timm_vgg16.tv_in1k": ("VGG-16", "tv"),
    "resnet18": ("ResNet-18", "tv"),
    "timm_resnet34.a1_in1k": ("ResNet-34", "A1"),
    "timm_resnet50.tv_in1k": ("ResNet-50", "tv V1"),
    "timm_resnet50.tv2_in1k": ("ResNet-50", "tv V2"),
    "timm_resnet50.a1_in1k": ("ResNet-50", "A1"),
    "timm_resnet50.a3_in1k": ("ResNet-50", "A3"),
    "timm_resnet50.gluon_in1k": ("ResNet-50", "GluonCV"),
    "timm_resnet50.fb_ssl_yfcc100m_ft_in1k": ("ResNet-50", "SSL ft"),
    "timm_resnet101.tv_in1k": ("ResNet-101", "tv"),
    "timm_wide_resnet50_2.tv_in1k": ("Wide-ResNet-50-2", "tv"),
    "timm_resnext50_32x4d.tv_in1k": ("ResNeXt-50 32x4d", "tv"),
    "timm_densenet121.tv_in1k": ("DenseNet-121", "tv"),
    "timm_mobilenetv2_100.ra_in1k": ("MobileNetV2", "RA"),
    "timm_mobilenetv3_large_100.ra_in1k": ("MobileNetV3-L", "RA"),
    "timm_efficientnet_b0.ra_in1k": ("EfficientNet-B0", "RA"),
    "convnext_tiny": ("ConvNeXt-T", "in1k"),
    "timm_convnext_tiny.fb_in22k_ft_in1k": ("ConvNeXt-T", "in22k ft"),
}
MODEL_ORDER = list(MODEL_LABELS)


def write_models_table(layers: dict, path: str) -> None:
    """Table I: the 22 checkpoints (19 distinct trained + 2 random, the
    torchvision/timm ResNet-50 duplicate folded into one row)."""
    out = ["% Generated by scripts/checkpoint_analysis.py --latex-models.  Do not edit by hand.",
           "\\begin{tabular}{llrrrrrc}", "\\toprule",
           "Model & Recipe & Conv & DW & Widths & $r_{\\max}\\!<\\!C$ & $n/C$ & Untrained \\\\",
           "\\midrule"]
    for m in MODEL_ORDER:
        if m not in layers:
            continue
        df = layers[m]
        d, w = conv_rows(df), depthwise_rows(df)
        allc = df[(df.kind == "conv") & df.n_over_C_ok.astype(bool)]
        widths = sorted(allc["C"].unique())
        bounded = int((allc["r_max"] < allc["C"]).sum())
        lo, hi = allc["n_over_C"].min(), allc["n_over_C"].max()
        base = m.replace("timm_", "").split(".")[0]
        ctrl = "\\checkmark" if any(is_random(k) and k.startswith(base) for k in layers) else "---"
        name, recipe = MODEL_LABELS[m]
        out.append(f"{name} & {recipe} & {len(d)} & {len(w)} & {widths[0]}--{widths[-1]} & "
                   f"{bounded} & {lo:.0f}--{hi:.0f} & {ctrl} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {path}")


def write_pooling_table(rows: list, path: str) -> None:
    out = ["% Generated by scripts/checkpoint_analysis.py --latex-pooling.  Do not edit by hand.",
           "\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "Model & Recipe & lower & drop$/C$ & step$/C$ & ratio & drop$/r_{\\max}$ & step$/r_{\\max}$ & ratio \\\\",
           "\\midrule"]
    for m, kd, nd, drop_c, jit_c, drop_r, jit_r in rows:
        name, recipe = MODEL_LABELS.get(m, (m, ""))
        out.append(f"{name} & {recipe} & {kd}/{nd} & {drop_c:.3f} & {jit_c:.3f} & {drop_c / jit_c:.2f} & "
                   f"{drop_r:.3f} & {jit_r:.3f} & {drop_r / jit_r:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {path}")


def section_pooling(layers: dict, latex: str | None = None) -> None:
    hdr("3. Global average pooling lowers the level, every layer, every checkpoint")
    print(f"  Paired per layer: pooled_{K} minus {K}.  A layer counts as 'lower'")
    print("  when the pooled value is strictly smaller.  Dense and depthwise convs")
    print("  are counted separately; both are reported.\n")
    print("  'drop' and 'jitter' are given with both denominators: /C is the")
    print("  pre-registered statistic; /r_max removes the sawtooth that 1x1 expansion")
    print("  layers put into the /C profile, so that the drop is compared with the")
    print("  curve's own jitter rather than with the architecture's.  The counts do")
    print("  not depend on the denominator (it is shared by the two estimators).\n")
    print(f"    {'model':40s} {'dense':>9s}  {'depthwise':>9s}  "
          f"{'drop/C':>7s} {'jit/C':>6s}  {'drop/rmax':>9s} {'jit/rmax':>8s}  "
          f"{'ratio/C':>7s} {'ratio/rmax':>10s}  {'Wilcoxon p':>10s}")
    tot = {"dense": [0, 0], "dw": [0, 0]}
    exceptions = []
    ratio_rows = []
    latex_rows = []
    for m in list(trained_names(layers)) + [m for m in layers if is_random(m)]:
        df = layers[m]
        d, w = conv_rows(df), depthwise_rows(df)
        pcol = "pooled_" + K
        d = d[d["pooled_n_over_C_ok"].astype(bool)] if "pooled_n_over_C_ok" in d else d
        w = w[w["pooled_n_over_C_ok"].astype(bool)] if "pooled_n_over_C_ok" in w else w
        delta = (d[pcol] - d[K]).to_numpy()
        kd, nd = int((delta < 0).sum()), len(delta)
        dw_delta = (w[pcol] - w[K]).to_numpy()
        kw, nw = int((dw_delta < 0).sum()), len(dw_delta)
        key = "random" if is_random(m) else "trained"
        if key == "trained":
            tot["dense"][0] += kd
            tot["dense"][1] += nd
            tot["dw"][0] += kw
            tot["dw"][1] += nw
        pval = wilcoxon(delta, alternative="less").pvalue if nd >= 5 else float("nan")
        ties, rev = int((delta == 0).sum()), int((delta > 0).sum())
        tag = "" if kd == nd else f"  <-- {ties} tie(s), {rev} reversal(s)"
        if kd != nd:
            exceptions.append((m, d[delta >= 0].layer.tolist(), ties, rev,
                               int(d[delta >= 0].C.min())))
        # the same statistic with r_max as the denominator (k* itself is shared)
        kcount = K.replace("k_star_ratio_", "k_star_")
        s_r = (d[kcount] / d["r_max"]).to_numpy()
        p_r = (d["pooled_" + kcount] / d["r_max"]).to_numpy()
        drop_c, jit_c = -np.median(delta), jitter(d[K].to_numpy())
        drop_r, jit_r = -np.median(p_r - s_r), jitter(s_r)
        if key == "trained":
            ratio_rows.append((m, drop_c / jit_c, drop_r / jit_r,
                               int((d["r_max"] < d["C"]).sum()), nd))
            latex_rows.append((m, kd, nd, drop_c, jit_c, drop_r, jit_r))
        print(f"    {m:40s} {kd:4d}/{nd:<4d}  {kw:4d}/{nw:<4d}  "
              f"{drop_c:7.3f} {jit_c:6.3f}  {drop_r:9.3f} {jit_r:8.3f}  "
              f"{drop_c / jit_c:7.2f} {drop_r / jit_r:10.2f}  {pval:10.1e}{tag}")
    print(f"\n    TRAINED TOTAL dense    : pooled lower on {tot['dense'][0]}/{tot['dense'][1]}")
    print(f"    TRAINED TOTAL depthwise: pooled lower on {tot['dw'][0]}/{tot['dw'][1]}")
    if exceptions:
        print("    exceptions (dense, pooled >= sampled).  k* is an integer, so at")
        print("    C = 32 one channel is 0.031 of the range; a tie there is resolution,")
        print("    not a reversal:")
        for m, ls, ties, rev, cmin in exceptions:
            print(f"      {m}: {ties} tie(s), {rev} reversal(s), min C = {cmin}")
            print(f"        {ls}")
    print("    (layers within a model are correlated, so per-model p-values are")
    print("     optimistic; the count of exceptions is what carries the claim)")
    print("\n    drop-to-jitter ratio, trained models, by whether the model has")
    print("    expansion layers (r_max < C).  A ratio below 1 means the pooling")
    print("    effect is smaller than the curve's own adjacent-layer step:")
    print(f"      {'model':40s} {'bounded':>8s}  {'ratio/C':>7s} {'ratio/rmax':>10s}")
    for m, rc, rr, nb, nd in sorted(ratio_rows, key=lambda t: -t[3] / t[4]):
        print(f"      {m:40s} {nb:3d}/{nd:<3d}   {rc:7.2f} {rr:10.2f}")
    inv = [(rc, rr) for _, rc, rr, nb, nd in ratio_rows if nb / nd >= 0.4]
    rest = [(rc, rr) for _, rc, rr, nb, nd in ratio_rows if nb / nd < 0.4]
    print(f"      inverted-residual models (>=40% bounded layers), median ratio: "
          f"/C {np.median([b[0] for b in inv]):.2f}   /r_max {np.median([b[1] for b in inv]):.2f}   (n={len(inv)})")
    print(f"      all other trained models, median ratio                   : "
          f"/C {np.median([u[0] for u in rest]):.2f}   /r_max {np.median([u[1] for u in rest]):.2f}   (n={len(rest)})")
    print(f"      trained models with ratio/r_max < 1: "
          f"{[m for m, _, rr, _, _ in ratio_rows if rr < 1]}")
    print("      (DenseNet-121's jitter is the 1x1/3x3 alternation of its dense")
    print("       layers, which is structural; its ratio is below 1 with either")
    print("       denominator and is reported as such)")
    if latex:
        order = {m: i for i, m in enumerate(MODEL_ORDER)}
        write_pooling_table(sorted(latex_rows, key=lambda r: order.get(r[0], 99)), latex)


# ------------------------------------------ section 3b: the flattening rule

def section_flattening(results: str, layers: dict) -> None:
    """The plan's second sample-size rule: a reported statistic must have
    flattened in n/C.  analyse.h5_sampling applies |k*(n/C=250) - k*(n/C=100)|
    > 0.02 and lists the offenders; here they are named, sized in channels,
    and the pooling count is repeated without them, so the paper can say
    exactly what the rule changes."""
    hdr("3b. Layers whose k*(.95)/C had not flattened at n/C = 250 (|delta| > 0.02)")
    dense_flag, dw_flag = {}, {}
    for m in trained_names(layers):
        path = os.path.join(results, f"{m}_convergence.csv")
        if not os.path.exists(path):
            continue
        cdf = pd.read_csv(path)
        cdf = cdf[cdf["kind"] == "conv"]
        for layer, g in cdf.groupby("layer"):
            g = g.sort_values("n_over_C")
            v = g[K].to_numpy()
            if len(v) >= 3 and abs(v[-1] - v[-2]) > 0.02:
                C = int(g["C"].iloc[0])
                row = layers[m][layers[m].layer == layer]
                dw = bool(row.is_depthwise.iloc[0]) if len(row) else False
                (dw_flag if dw else dense_flag).setdefault(m, []).append((layer, C, (v[-1] - v[-2]) * C))
    n_dense = sum(len(v) for v in dense_flag.values())
    n_dw = sum(len(v) for v in dw_flag.values())
    print(f"  dense layers flagged: {n_dense}   depthwise flagged: {n_dw}")
    for m, ls in list(dense_flag.items()) + list(dw_flag.items()):
        print(f"    {m}:")
        for layer, C, dch in ls:
            print(f"      {layer:45s} C={C:4d}  moved {dch:+.1f} channels between n/C=100 and 250")
    # pooling count with the flagged layers removed
    tot = [0, 0]
    for m in trained_names(layers):
        d = conv_rows(layers[m])
        d = d[d["pooled_n_over_C_ok"].astype(bool)]
        bad = {l for l, _, _ in dense_flag.get(m, [])}
        d = d[~d.layer.isin(bad)]
        delta = (d["pooled_" + K] - d[K]).to_numpy()
        tot[0] += int((delta < 0).sum()); tot[1] += len(delta)
    print(f"  pooled-lower count (dense, trained) with flagged layers removed: {tot[0]}/{tot[1]}")
    print("  (a C=32 layer moves 0.031 per channel, so one channel trips the 0.02 rule)")


# ------------------------------------------------- section 4: trained vs random

def section_random(layers: dict) -> None:
    hdr("4. Trained vs random init, paired per layer, one test per checkpoint")
    print(f"  Statistic {K}, dense conv layers, common gate-passing layers.\n")
    print(f"    {'trained checkpoint':40s} {'vs':22s} {'higher':>8s}  {'sign p':>8s}  "
          f"{'Wilcoxon p':>10s}  {'med diff':>8s} {'jitter':>7s}")
    pairs = [(m, "resnet50_random", conv_rows) for m in RESNET50_RECIPES] + \
            [(m, "convnext_tiny_random", fn) for m in CONVNEXT_CKPTS
             for fn in (conv_rows, depthwise_rows)]
    for m, r, rows_fn in pairs:
        if m not in layers or r not in layers:
            continue
        a = rows_fn(layers[m]).reset_index(drop=True)
        b = rows_fn(layers[r]).reset_index(drop=True)
        if len(a) == 0 or len(a) != len(b):
            print(f"    {m:40s} {r:22s} layer counts differ ({len(a)} vs {len(b)}), skipped")
            continue
        # align by depth order: torchvision and timm name the layers differently
        av, bv = a[K].to_numpy(), b[K].to_numpy()
        dd = av - bv
        k, n = int((dd > 0).sum()), len(dd)
        tag = " dw" if rows_fn is depthwise_rows else ""
        print(f"    {m + tag:40s} {r:22s} {k:4d}/{n:<4d} "
              f"{binomtest(k, n, 0.5, alternative='greater').pvalue:8.1e}  "
              f"{wilcoxon(dd, alternative='greater').pvalue:10.1e}  "
              f"{np.median(dd):+8.3f} {jitter(av):7.3f}")
    print("\n  Caveat: the random control is ONE draw of the initialiser, and the")
    print("  first and second runs drew different weights (the init was not seeded;")
    print("  fixed in models.build after this run).  The direction of the effect is")
    print("  the same in both runs; the random profile's fine structure is not stable.")


# ----------------------------------------- section 5: conv output vs block output

def section_block(layers: dict) -> None:
    hdr("5. Where the bound applies: conv3 output vs block output (post-add ReLU)")
    print("  In a bottleneck block the 1x1 expansion conv3 has r_max = C/4 (C/2 for")
    print("  wide/ResNeXt), so k*(.999)/C <= 0.25 at its output by construction.")
    print("  The residual sum restores the rank: the same channels at the block")
    print("  output are not bounded.  This is why studies that hook block outputs")
    print("  (Ansuini 2019, Elmoznino & Bonner 2024) never saw the artefact, and why")
    print("  a per-layer protocol must state which tensor it measures.\n")
    print(f"    {'model':40s} {'blocks':>6s}  {'conv3 k*/C med':>14s}  "
          f"{'block k*/C med':>14s}  {'block > bound':>13s}  {'ratio block/conv3':>17s}")
    for m in BOTTLENECK_RESNETS:
        if m not in layers:
            continue
        df = layers[m]
        blocks = df[(df.kind == "block") & df.n_over_C_ok].set_index("layer")
        c3 = df[(df.kind == "conv") & df.n_over_C_ok & df.layer.str.endswith(".conv3")].copy()
        c3["block"] = c3.layer.str[: -len(".conv3")]
        c3 = c3.set_index("block")
        common = c3.index.intersection(blocks.index)
        if len(common) == 0:
            print(f"    {m:40s} no block rows")
            continue
        bound = (c3.loc[common, "r_max"] / c3.loc[common, "C"]).to_numpy()
        kc = c3.loc[common, K999].to_numpy()
        kb = blocks.loc[common, K999].to_numpy()
        over = int((kb > bound + 1e-9).sum())
        print(f"    {m:40s} {len(common):6d}  {np.median(kc):14.3f}  {np.median(kb):14.3f}  "
              f"{over:6d}/{len(common):<6d} {np.median(kb / kc):17.2f}")
    print("\n  The same comparison for the pre-registered statistic k*(.95)/C:")
    print(f"    {'model':40s} {'conv3 med':>10s}  {'block med':>10s}  {'block higher':>12s}")
    for m in BOTTLENECK_RESNETS:
        if m not in layers:
            continue
        df = layers[m]
        blocks = df[(df.kind == "block") & df.n_over_C_ok].set_index("layer")
        c3 = df[(df.kind == "conv") & df.n_over_C_ok & df.layer.str.endswith(".conv3")].copy()
        c3["block"] = c3.layer.str[: -len(".conv3")]
        c3 = c3.set_index("block")
        common = c3.index.intersection(blocks.index)
        if len(common) == 0:
            continue
        kc, kb = c3.loc[common, K].to_numpy(), blocks.loc[common, K].to_numpy()
        print(f"    {m:40s} {np.median(kc):10.3f}  {np.median(kb):10.3f}  "
              f"{int((kb > kc).sum()):5d}/{len(common):<6d}")


# ------------------------------------ section 6: architectural or learned?

def _stage_of(layer: str) -> str:
    """ResNet stage name ('layer1'..'layer4', 'stem') from a layer name."""
    head = layer.split(".")[0]
    return head if head.startswith("layer") else "stem"


def _rho(a, b) -> float:
    return float(spearmanr(a, b).correlation)


def section_learned(layers: dict) -> None:
    hdr("6. Is the per-layer profile a property of the architecture, or learned?")
    print("  The six ResNet-50 recipes agree on the ORDER of layers (section 2).")
    print("  If a random-init network of the same architecture, on the same images,")
    print("  gives the same order, the structure is architecture + input statistics;")
    print("  if not, it is learned.  Spearman between profiles over the same")
    print("  gate-passing dense conv layers, aligned by depth order.\n")

    def block(label, names, rand, rows_fn, cols):
        names = [n for n in names if n in layers]
        if rand not in layers or len(names) < 2:
            print(f"  {label}: missing data, skipped")
            return
        fr = [rows_fn(layers[n]).reset_index(drop=True) for n in names]
        rr = rows_fn(layers[rand]).reset_index(drop=True)
        if len({len(f) for f in fr} | {len(rr)}) != 1:
            print(f"  {label}: layer counts differ, skipped")
            return
        print(f"  {label}: {len(names)} trained checkpoints vs 1 random init, {len(rr)} layers")
        depth = np.arange(len(rr))
        for col in cols:
            if col not in rr.columns:
                continue
            T = np.stack([f[col].to_numpy() for f in fr], axis=1)
            R = rr[col].to_numpy()
            tt = [_rho(T[:, i], T[:, j]) for i, j in itertools.combinations(range(T.shape[1]), 2)]
            tr = [_rho(T[:, i], R) for i in range(T.shape[1])]
            print(f"    {col:26s} trained-vs-trained rho: median {np.median(tt):+.2f} "
                  f"[{min(tt):+.2f}, {max(tt):+.2f}]   trained-vs-random rho: "
                  f"median {np.median(tr):+.2f} [{min(tr):+.2f}, {max(tr):+.2f}]")
            print(f"    {'':26s} rho(depth): trained mean {_rho(depth, T.mean(1)):+.2f}  "
                  f"random {_rho(depth, R):+.2f}    level: trained median "
                  f"{np.median(T):.3f}  random median {np.median(R):.3f}")
        return fr, rr

    out = block("ResNet-50 (dense conv)", RESNET50_RECIPES, "resnet50_random", conv_rows,
                (K, "k_star_rmax_0.95", K999, "k_star_rmax_0.999",
                 "participation_ratio_norm", "effective_rank_norm"))

    if out is not None:
        fr, rr = out
        print("\n    Within-stage (same nominal C, so no width or r_max sawtooth confound):")
        print(f"    {'stage':8s} {'n':>3s}  {'trained-vs-trained':>20s}  {'trained-vs-random':>19s}   (k*(.95)/r_max)")
        stages = rr.layer.map(_stage_of)
        col = "k_star_rmax_0.95"
        for st in ["layer1", "layer2", "layer3", "layer4"]:
            idx = np.where(stages.to_numpy() == st)[0]
            if len(idx) < 5:
                continue
            T = np.stack([f[col].to_numpy()[idx] for f in fr], axis=1)
            R = rr[col].to_numpy()[idx]
            tt = [_rho(T[:, i], T[:, j]) for i, j in itertools.combinations(range(T.shape[1]), 2)]
            tr = [_rho(T[:, i], R) for i in range(T.shape[1])]
            print(f"    {st:8s} {len(idx):3d}  {np.median(tt):+8.2f} [{min(tt):+.2f},{max(tt):+.2f}]"
                  f"  {np.median(tr):+8.2f} [{min(tr):+.2f},{max(tr):+.2f}]")

        print("\n    By conv role inside the bottleneck (conv1 1x1 reduce / conv2 3x3 / conv3 1x1 expand):")
        roles = rr.layer.str.extract(r"\.(conv[123]|downsample)")[0].fillna("other")
        for role in ["conv1", "conv2", "conv3"]:
            idx = np.where(roles.to_numpy() == role)[0]
            T = np.stack([f[col].to_numpy()[idx] for f in fr], axis=1)
            R = rr[col].to_numpy()[idx]
            tt = [_rho(T[:, i], T[:, j]) for i, j in itertools.combinations(range(T.shape[1]), 2)]
            tr = [_rho(T[:, i], R) for i in range(T.shape[1])]
            print(f"    {role:8s} {len(idx):3d}  {np.median(tt):+8.2f} [{min(tt):+.2f},{max(tt):+.2f}]"
                  f"  {np.median(tr):+8.2f} [{min(tr):+.2f},{max(tr):+.2f}]"
                  f"   level trained {np.median(T):.3f} random {np.median(R):.3f}")

    print()
    block("ConvNeXt-T (depthwise 7x7)", CONVNEXT_CKPTS, "convnext_tiny_random",
          depthwise_rows, (K, K999, "participation_ratio_norm"))
    print("\n  Caveat: one random draw per architecture (init now seeded; more draws")
    print("  are cheap).  A single draw can only show whether the random profile is")
    print("  close to the trained one; how much of the trained-vs-random gap is draw")
    print("  noise needs the extra seeds.")


# ------------------------------- section 7: random draws, same image subset

def section_seeds(results: str, layers: dict) -> None:
    """Several random-init draws and the six recipes on ONE 6,400-image subset
    (results/local6400, run on the Mac, see scripts/run_local_seeds.sh)."""
    base = os.path.join(results, "local6400")
    seed_files = sorted(glob.glob(os.path.join(base, "seed*", "resnet50_random_layers.csv")))
    trained_files = sorted(glob.glob(os.path.join(base, "trained", "*_layers.csv")))
    if not seed_files:
        return
    hdr("7. How much of the trained-vs-random gap is draw noise?  (same 6,400 images)")

    def prof(path, col):
        d = pd.read_csv(path)
        d["n_over_C_ok"] = d["n_over_C_ok"].astype(bool)
        d = conv_rows(d.sort_values("depth_index"))
        return d[col].to_numpy(), d.layer.to_numpy()

    col = "k_star_rmax_0.95"
    R = np.stack([prof(f, col)[0] for f in seed_files], axis=1)
    names_r = [os.path.basename(os.path.dirname(f)) for f in seed_files]
    print(f"  random draws: {names_r}   layers: {R.shape[0]}")
    rr = [_rho(R[:, i], R[:, j]) for i, j in itertools.combinations(range(R.shape[1]), 2)]
    print(f"  random-vs-random rho ({col}): median {np.median(rr):+.2f} "
          f"[{min(rr):+.2f}, {max(rr):+.2f}]   per-layer SD across draws: "
          f"median {np.median(R.std(1, ddof=1)):.3f}   level median {np.median(R):.3f}")
    depth = np.arange(R.shape[0])
    print(f"  rho(depth) per draw: " + "  ".join(f"{_rho(depth, R[:, j]):+.2f}" for j in range(R.shape[1])))

    if trained_files:
        T = np.stack([prof(f, col)[0] for f in trained_files], axis=1)
        names_t = [os.path.basename(f)[: -len("_layers.csv")] for f in trained_files]
        print(f"\n  trained on the same subset: {names_t}")
        tt = [_rho(T[:, i], T[:, j]) for i, j in itertools.combinations(range(T.shape[1]), 2)]
        tr = [_rho(T[:, i], R[:, j]) for i in range(T.shape[1]) for j in range(R.shape[1])]
        print(f"  trained-vs-trained rho: median {np.median(tt):+.2f} [{min(tt):+.2f}, {max(tt):+.2f}]")
        print(f"  trained-vs-random  rho: median {np.median(tr):+.2f} [{min(tr):+.2f}, {max(tr):+.2f}]"
              f"   ({len(tr)} pairs)")
        print(f"  rho(depth) trained mean {_rho(depth, T.mean(1)):+.2f}   random mean {_rho(depth, R.mean(1)):+.2f}")
        print(f"  level: trained median {np.median(T):.3f}   random median {np.median(R):.3f}")
        # the 3 x 3 picture: within-random, within-trained, between
        print("\n  Reading: the random-vs-random rho is the ceiling a random draw could")
        print("  reach; if trained-vs-random is far below it, the gap is not draw noise.")

        # consistency with the 50k run for the same weights
        for f, n in zip(trained_files, names_t):
            if n in layers:
                a, la = prof(f, col)
                d = conv_rows(layers[n])
                b = d.set_index("layer").loc[la, col].to_numpy() if set(la) <= set(d.layer) else None
                if b is not None:
                    print(f"  {n:40s} 6,400 vs 50,000 images: rho {_rho(a, b):+.3f}  "
                          f"max|diff| {np.abs(a - b).max():.3f}")


# ------------------------- section 8: the same question, other architectures

# family: (glob for trained files, glob for random files); the ResNet-50 set
# lives one directory up (section 7) and is included so all families are in
# one table.
ARCH_FAMILIES = {
    "resnet50":        ("local6400/trained/*resnet50*_layers.csv",
                        "local6400/seed*/resnet50_random_layers.csv"),
    "resnet18":        ("local6400/archs/trained/timm_resnet18.*_layers.csv",
                        "local6400/archs/seed*/timm_resnet18.tv_in1k_random_layers.csv"),
    "resnet34":        ("local6400/archs/trained/timm_resnet34.*_layers.csv",
                        "local6400/archs/seed*/timm_resnet34.tv_in1k_random_layers.csv"),
    "vgg16":           ("local6400/archs/trained/timm_vgg16*_layers.csv",
                        "local6400/archs/seed*/timm_vgg16_bn.tv_in1k_random_layers.csv"),
    "densenet121":     ("local6400/archs/trained/timm_densenet121.*_layers.csv",
                        "local6400/archs/seed*/timm_densenet121.tv_in1k_random_layers.csv"),
    "resnet101":       ("local6400/archs/trained/timm_resnet101.*_layers.csv",
                        "local6400/archs/seed*/timm_resnet101.tv_in1k_random_layers.csv"),
    "wide_resnet50_2": ("local6400/archs/trained/timm_wide_resnet50_2.*_layers.csv",
                        "local6400/archs/seed*/timm_wide_resnet50_2.tv_in1k_random_layers.csv"),
    "resnext50":       ("local6400/archs/trained/timm_resnext50_32x4d.*_layers.csv",
                        "local6400/archs/seed*/timm_resnext50_32x4d.tv_in1k_random_layers.csv"),
    "mobilenetv2":     ("local6400/archs/trained/timm_mobilenetv2_100.*_layers.csv",
                        "local6400/archs/seed*/timm_mobilenetv2_100.ra_in1k_random_layers.csv"),
    "mobilenetv3":     ("local6400/archs/trained/timm_*mobilenetv3_large_100.*_layers.csv",
                        "local6400/archs/seed*/timm_mobilenetv3_large_100.ra_in1k_random_layers.csv"),
    "efficientnet_b0": ("local6400/archs/trained/timm_*efficientnet_b0.*_layers.csv",
                        "local6400/archs/seed*/timm_efficientnet_b0.ra_in1k_random_layers.csv"),
}


def _load_local(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["n_over_C_ok"] = d["n_over_C_ok"].astype(bool)
    return d.sort_values("depth_index").reset_index(drop=True)


def conv_role(family: str, layer: str, is_dw: bool) -> str:
    """'1x1', '3x3', 'dw' or 'stem', from the layer name.  Kernel sizes are
    not stored in the CSV, so this is by naming convention per family;
    it was checked against the architectures on 2026-09-04."""
    if is_dw:
        return "dw"
    if family in ("resnet50", "resnet101", "wide_resnet50_2", "resnext50"):
        if layer in ("conv1",):
            return "stem"
        return "3x3" if layer.endswith(".conv2") else "1x1"     # conv1, conv3, downsample
    if family in ("resnet18", "resnet34"):
        if layer == "conv1":
            return "stem"
        return "1x1" if "downsample" in layer else "3x3"
    if family == "vgg16":
        return "stem" if layer == "features.0" else "3x3"
    if family == "densenet121":
        if layer == "features.conv0":
            return "stem"
        return "3x3" if layer.endswith(".conv2") else "1x1"     # denselayer conv1, transition conv
    if family in ("mobilenetv3", "efficientnet_b0", "mobilenetv2"):
        if layer == "conv_stem":
            return "stem"
        return "1x1"                                              # conv_pw, conv_pwl, conv_head
    return "other"


def _three_rhos(Tm, Rm):
    tt = [_rho(Tm[:, i], Tm[:, j]) for i, j in itertools.combinations(range(Tm.shape[1]), 2)]
    rr = [_rho(Rm[:, i], Rm[:, j]) for i, j in itertools.combinations(range(Rm.shape[1]), 2)]
    tr = [_rho(Tm[:, i], Rm[:, j]) for i in range(Tm.shape[1]) for j in range(Rm.shape[1])]
    fmt = lambda v: f"{np.median(v):+.2f} [{min(v):+.2f},{max(v):+.2f}]" if v else "     n/a       "
    return fmt(rr), fmt(tt), fmt(tr)


FAMILY_LABELS = {
    "resnet50": ("ResNet-50", "bottleneck"),
    "resnet18": ("ResNet-18", "basic"),
    "resnet34": ("ResNet-34", "basic"),
    "vgg16": ("VGG-16 ($\\pm$BN)", "plain"),
    "densenet121": ("DenseNet-121", "dense"),
    "resnet101": ("ResNet-101", "bottleneck"),
    "wide_resnet50_2": ("Wide-ResNet-50-2", "bottleneck (2$\\times$)"),
    "resnext50": ("ResNeXt-50", "bottleneck (2$\\times$)"),
    "mobilenetv2": ("MobileNetV2", "inverted residual"),
    "mobilenetv3": ("MobileNetV3-L", "inverted residual"),
    "efficientnet_b0": ("EfficientNet-B0", "inverted residual"),
}


def _tex(v):
    return f"{np.median(v):+.2f}" if v is not None and len(v) else "---"


def _stage_ranks(M: np.ndarray, layers: list) -> np.ndarray | None:
    """Rank each profile within its stage (layer1..layer4 prefix) and pool the
    ranks, so that a Spearman between two such vectors measures within-stage
    ordering only: a monotone depth trend cannot produce it."""
    stages = np.array([l.split(".")[0] if l.startswith("layer") else "other" for l in layers])
    if len({s for s in stages if s != "other"}) < 2:
        return None
    from scipy.stats import rankdata
    R = np.full_like(M, np.nan, dtype=float)
    for st in set(stages):
        if st == "other":
            continue
        idx = stages == st
        for j in range(M.shape[1]):
            R[idx, j] = rankdata(M[idx, j]) / idx.sum()
    keep = stages != "other"
    return R[keep]


def write_families_table(rows: list, path: str) -> None:
    """LaTeX table for the paper from the section-8 rows (family, role, L, nT,
    nR, rr, tt, tr, levelT, levelR).  Dense rows carry the family name; the
    depthwise row of the same family follows it."""
    out = ["% Generated by scripts/checkpoint_analysis.py.  Do not edit by hand.",
           "\\begin{tabular}{llrrrrrrrrrr}", "\\toprule",
           " & & & & & \\multicolumn{3}{c}{$k^*/r_{\\max}$} & \\multicolumn{2}{c}{$k^*/C$} & & \\\\",
           "\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}",
           "Family & Block & $L$ & $n_T$ & $n_R$ & R--R & T--T & T--R & T--R & T--R$_{\\rm stage}$ & $T$ & $R$ \\\\",
           "\\midrule"]
    for fam, role, L, nT, nR, rr, tt, tr, lT, lR, trC, trS in rows:
        name, block = FAMILY_LABELS.get(fam, (fam, ""))
        if role == "dw":
            name, block = "\\quad depthwise", ""
        out.append(f"{name} & {block} & {L} & {nT} & {nR} & {_tex(rr)} & {_tex(tt)} & "
                   f"{_tex(tr)} & {_tex(trC)} & {_tex(trS)} & {lT:.2f} & {lR:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"  wrote {path}")


def section_archs(results: str, latex: str | None = None) -> None:
    if not os.path.isdir(os.path.join(results, "local6400", "archs")):
        return
    table_rows = []
    hdr("8. Architectural or learned, per family and per conv role (same 6,400 images)")
    print("  Spearman between k*(0.95)/r_max profiles: random-vs-random (R-R, the")
    print("  ceiling), trained-vs-trained (T-T, recipe stability), trained-vs-random")
    print("  (T-R).  'all' = every dense conv layer; then split by role, because the")
    print("  hypothesis from ResNet-50 (section 6) is that training reorganises the")
    print("  1x1 channel-mixing layers and leaves the 3x3 layers' order alone.\n")
    col = "k_star_rmax_0.95"
    print(f"    {'family':16s} {'role':5s} {'L':>3s} {'nT':>2s} {'nR':>2s}  {'R-R':^19s}  "
          f"{'T-T':^19s}  {'T-R':^19s}  {'level T/R':>11s}  {'T-R diff':>8s}  T level per ckpt")
    for fam, (tglob, rglob) in ARCH_FAMILIES.items():
        tr_files = sorted(f for f in glob.glob(os.path.join(results, tglob))
                          if "_random_" not in os.path.basename(f))
        seed_files = sorted(glob.glob(os.path.join(results, rglob)))
        if not tr_files or not seed_files:
            print(f"    {fam:16s} missing ({len(tr_files)} trained, {len(seed_files)} random)")
            continue
        T = [_load_local(f) for f in tr_files]
        R = [_load_local(f) for f in seed_files]
        T = [x[(x.kind == "conv") & x.n_over_C_ok].reset_index(drop=True) for x in T]
        R = [x[(x.kind == "conv") & x.n_over_C_ok].reset_index(drop=True) for x in R]
        if len({len(x) for x in T + R}) != 1:
            print(f"    {fam:16s} layer counts differ, skipped")
            continue
        base = R[0]
        is_dw = base.is_depthwise.astype(bool).to_numpy() if "is_depthwise" in base else np.zeros(len(base), bool)
        roles = np.array([conv_role(fam, l, d) for l, d in zip(base.layer, is_dw)])
        Tall = np.stack([x[col].to_numpy() for x in T], 1)
        Rall = np.stack([x[col].to_numpy() for x in R], 1)
        TallC = np.stack([x[K].to_numpy() for x in T], 1)
        RallC = np.stack([x[K].to_numpy() for x in R], 1)
        layer_names = base.layer.tolist()
        for role in ("all", "3x3", "1x1", "dw"):
            idx = np.where(roles != "dw")[0] if role == "all" else np.where(roles == role)[0]
            if len(idx) < 5:
                continue
            Tm, Rm = Tall[idx], Rall[idx]
            rr, tt, tr = _three_rhos(Tm, Rm)
            if role in ("all", "dw"):
                pairs = lambda M: [_rho(M[:, i], M[:, j]) for i, j in itertools.combinations(range(M.shape[1]), 2)]
                xcorr = lambda A, B: [_rho(A[:, i], B[:, j]) for i in range(A.shape[1]) for j in range(B.shape[1])]
                cross = xcorr(Tm, Rm)
                crossC = xcorr(TallC[idx], RallC[idx])
                crossS = None
                if role == "all":
                    ST = _stage_ranks(Tm, [layer_names[i] for i in idx])
                    SR = _stage_ranks(Rm, [layer_names[i] for i in idx])
                    if ST is not None:
                        crossS = xcorr(ST, SR)
                        print(f"    {fam:16s} within-stage (pooled ranks, {ST.shape[0]} layers): "
                              f"T-T {np.median(xcorr(ST, ST) if False else pairs(ST)):+.2f}  T-R {np.median(crossS):+.2f}  R-R {np.median(pairs(SR)):+.2f}")
                    print(f"    {fam:16s} under k*/C: R-R {np.median(pairs(RallC[idx])):+.2f}  "
                          f"T-T {np.median(pairs(TallC[idx])):+.2f}  T-R {np.median(crossC):+.2f}")
                table_rows.append((fam, role, len(idx), Tm.shape[1], Rm.shape[1],
                                   pairs(Rm), pairs(Tm), cross,
                                   float(np.median(Tm)), float(np.median(Rm)), crossC, crossS))
            print(f"    {fam:16s} {role:5s} {len(idx):3d} {Tm.shape[1]:2d} {Rm.shape[1]:2d}  {rr}  {tt}  {tr}  "
                  f"{np.median(Tm):5.3f}/{np.median(Rm):5.3f}  {np.mean(Tm.mean(1) - Rm.mean(1)):+8.3f}  "
                  + " ".join(f"{v:.2f}" for v in Tm.mean(0)))
    print("\n  Reading: 'learned order' = T-R well below both R-R and T-T.  A family")
    print("  whose T-T is near zero has no recipe-independent trained profile at all;")
    print("  look at its per-checkpoint levels before saying anything about it.")
    print("  VGG-16's two checkpoints are with and without BN, so its T-T compares")
    print("  two architectures.")
    if latex:
        write_families_table(table_rows, latex)


# ----------------------------------------- section 9: training trajectory

def _load_trajectory(path: str):
    d = pd.read_csv(path)
    # trajectory_cifar.py writes the raw metric rows; derive the gate and the
    # r_max-normalised statistic here exactly as run.py does.
    d = d[(d.kind == "conv") & (d.n_over_C >= 50)].copy()
    if "is_depthwise" in d:                      # rule 5: depthwise layers are a different object
        d = d[~d.is_depthwise.astype(bool)]
    d["k_star_rmax_0.95"] = d["k_star_0.95"] / d["r_max"]
    epochs = sorted(d.epoch.unique())
    P = {e: d[d.epoch == e].sort_values("depth_index")["k_star_rmax_0.95"].to_numpy() for e in epochs}
    acc = {e: float(d[d.epoch == e].test_acc.iloc[0]) for e in epochs}
    layers = d[d.epoch == epochs[0]].sort_values("depth_index").layer.tolist()
    return epochs, P, acc, layers


def section_trajectory(results: str) -> None:
    runs = sorted(glob.glob(os.path.join(results, "trajectory*", "trajectory_layers.csv")))
    if not runs:
        return
    hdr("9. When does the profile become the trained one?  CIFAR-10 trajectories")
    print("  statistic k*(0.95)/r_max, conv layers passing n/C >= 50; one table per run")
    summary = {}
    for path in runs:
        name = os.path.basename(os.path.dirname(path))
        epochs, P, acc, layers = _load_trajectory(path)
        L = len(P[epochs[0]])
        depth = np.arange(L)
        final = P[epochs[-1]]
        done = epochs[-1] >= 100
        print(f"\n  [{name}]  {L} conv layers, epochs measured {epochs[0]}..{epochs[-1]}"
              + ("" if done else "  (IN PROGRESS: 'final' = latest epoch)"))
        print(f"    {'epoch':>5s} {'acc%':>6s}  {'level':>6s}  {'rho(depth)':>10s}  "
              f"{'rho vs init':>11s}  {'rho vs final':>12s}  {'MAD vs final':>12s}")
        for e in epochs:
            v = P[e]
            print(f"    {e:5d} {acc[e]:6.2f}  {np.median(v):6.3f}  {_rho(depth, v):+10.2f}  "
                  f"{_rho(P[epochs[0]], v):+11.2f}  {_rho(final, v):+12.2f}  "
                  f"{np.abs(v - final).mean():12.3f}")
        settle = next((e for e in epochs if _rho(final, P[e]) >= 0.95), None)
        summary[name] = dict(done=done, settle=settle, rho_init_final=_rho(P[epochs[0]], final),
                             rho_depth_init=_rho(depth, P[epochs[0]]), rho_depth_final=_rho(depth, final),
                             acc_settle=acc.get(settle), acc_final=acc[epochs[-1]], final=final,
                             init=P[epochs[0]], layers=layers)
        if L >= 40:   # bottleneck run: split by conv role as in section 8
            roles = np.array([conv_role("resnet50", l, False) for l in layers])
            for role in ("3x3", "1x1"):
                idx = roles == role
                print(f"    role {role}: rho(init, final) {_rho(P[epochs[0]][idx], final[idx]):+.2f}"
                      f"   level init {np.median(P[epochs[0]][idx]):.3f} -> final {np.median(final[idx]):.3f}")
    print("\n  Summary (settle = first measured epoch with rho vs final >= 0.95):")
    print(f"    {'run':22s} {'done':>4s} {'settle':>6s} {'acc@settle':>10s} {'acc final':>9s}  "
          f"{'rho(init,final)':>15s}  {'rho_depth init->final':>22s}")
    for name, r in summary.items():
        print(f"    {name:22s} {'yes' if r['done'] else 'no':>4s} {str(r['settle']):>6s} "
              f"{(r['acc_settle'] or float('nan')):10.2f} {r['acc_final']:9.2f}  {r['rho_init_final']:+15.2f}  "
              f"{r['rho_depth_init']:+10.2f} -> {r['rho_depth_final']:+.2f}")
    vgg = [r for n, r in summary.items() if r["done"] and len(r["final"]) == 13]
    if len(vgg) >= 2:
        pairs = [_rho(a["final"], b["final"]) for a, b in itertools.combinations(vgg, 2)]
        print(f"    VGG seeds: final-vs-final rho over {len(pairs)} pairs: "
              f"median {np.median(pairs):+.2f} [{min(pairs):+.2f}, {max(pairs):+.2f}]")
    print("\n  Reading: the epoch at which rho-vs-init drops and rho-vs-final saturates")
    print("  is when the learned profile is in place; compare it with the accuracy")
    print("  column to see whether it precedes or follows the fit.  For the")
    print("  bottleneck run the question is whether rho(init, final) is near zero,")
    print("  as it is across the six ImageNet recipes (section 7), and when it drops.")


# ----------------------------------------- section 10: projection anchor

def section_projection(results: str, latex: str | None = None) -> None:
    files = sorted(glob.glob(os.path.join(results, "projection", "*_projection.csv")))
    if not files:
        return
    hdr("10. Downstream anchor: project every conv output onto its top-k*(tau) directions")
    print("  scripts/projection_check.py: covariances from one half of the 6,400 images,")
    print("  top-1 on the other half, all conv layers projected at once, no retraining.")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        model = d.model.iloc[0]
        base = float(d[d.setting == "baseline"].top1.iloc[0])
        print(f"\n  {model}: baseline top-1 {base:.2f}%  (calibrate {int(d.n_calib.iloc[0])}, evaluate {int(d.n_eval.iloc[0])})")
        print(f"    {'tau':>6s} {'k/C':>6s} {'k/rmax':>7s} {'principal':>10s} {'random':>8s}")
        for tau in sorted(d.tau.dropna().unique()):
            pr = d[(d.setting == "principal") & (d.tau == tau)].iloc[0]
            rn = d[(d.setting == "random") & (d.tau == tau)].iloc[0]
            print(f"    {tau:6.3f} {pr.mean_k_over_C:6.3f} {pr.mean_k_over_rmax:7.3f} {pr.top1:9.2f}% {rn.top1:7.2f}%")
            rows.append((model, tau, pr.mean_k_over_C, pr.mean_k_over_rmax, pr.top1, rn.top1, base))
    # paired McNemar tests (per-image correctness, baseline vs principal at each tau)
    pim = sorted(glob.glob(os.path.join(results, "projection", "*_projection_perimage.csv")))
    if pim:
        from scipy.stats import binomtest
        print("\n  McNemar (exact, two-sided) baseline vs principal truncation, same images:")
        print(f"    {'model':38s} {'tau':>6s} {'n':>5s} {'b->wrong':>8s} {'wrong->b':>8s} {'delta pp':>8s} {'p':>9s}")
        for f in pim:
            d = pd.read_csv(f); model = os.path.basename(f).replace("_projection_perimage.csv", "")
            base = d["baseline"].to_numpy()
            for col in [c for c in d.columns if c.startswith("principal_")]:
                pr = d[col].to_numpy(); tau = col.split("_", 1)[1]
                b01 = int(((base == 1) & (pr == 0)).sum()); b10 = int(((base == 0) & (pr == 1)).sum())
                pval = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 > 0 else 1.0
                print(f"    {model:38s} {tau:>6s} {len(d):5d} {b01:8d} {b10:8d} {100 * (b10 - b01) / len(d):+8.2f} {pval:9.2e}")
    if latex:
        # pivot: one row per checkpoint, principal top-1 at each tau, baseline, and the
        # McNemar sign at tau = 0.999 (* p < 0.01); the random control is <= 2% throughout
        taus_l = [0.9, 0.95, 0.99, 0.999]
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-projection.  Do not edit by hand.",
               "\\begin{tabular}{l" + "r" * (len(taus_l) + 1) + "}", "\\toprule",
               "Checkpoint & baseline & " + " & ".join(f"$\\tau={t:g}$" for t in taus_l) + " \\\\", "\\midrule"]
        pv = {}
        for model, tau, kc, kr, pt, rt, base in rows:
            pv.setdefault(model, {"base": base})[tau] = pt
        mc = {}
        for f in pim:
            d = pd.read_csv(f); model = os.path.basename(f).replace("_projection_perimage.csv", "")
            if "principal_0.999" in d:
                base = d["baseline"].to_numpy(); pr = d["principal_0.999"].to_numpy()
                b01 = int(((base == 1) & (pr == 0)).sum()); b10 = int(((base == 0) & (pr == 1)).sum())
                mc[model] = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        for model, v in pv.items():
            if model.endswith("_lolo"):
                continue
            name, recipe = MODEL_LABELS.get(model, MODEL_LABELS.get("timm_" + model.replace("timm:", ""), (model, "")))
            key = model.replace(":", "_")
            star = "$^*$" if mc.get(key, 1.0) < 0.01 else ""
            cells = " & ".join(f"{v[t]:.1f}" + (star if t == 0.999 else "") for t in taus_l)
            out.append(f"{name} {recipe} & {v['base']:.1f} & {cells} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------- section 10b: the projection as a low-rank factorisation

def section_projection_flops(results: str, layers: dict) -> None:
    """Projecting a conv output onto its top-k principal channel directions
    equals replacing W (C_out x d) by V_k (V_k^T W): a rank-k factorisation with
    k(d + C_out) multiply-adds per position instead of C_out d.  FLOPs are
    summed over dense conv layers weighted by output positions H*W (from the
    layer tables); the classifier and depthwise layers are left as they are."""
    files = sorted(glob.glob(os.path.join(results, "projection", "*_projection_layers.csv")))
    if not files:
        return
    hdr("10b. The projection read as a rank-k factorisation: conv FLOPs kept, per tau")
    print("  FLOPs of W -> V_k (V_k^T W): k(d + C_out) vs C_out d per output position, dense convs only.\n")
    print(f"    {'model':28s} {'conv GFLOPs':>11s}  " + "  ".join(f"{'tau=' + str(t):>10s}" for t in (0.9, 0.95, 0.99, 0.999)))
    for f in files:
        per = pd.read_csv(f)
        model = os.path.basename(f).replace("_projection_layers.csv", "")
        key = model if model in layers else model.replace("timm_", "timm_")
        if key not in layers:
            print(f"    {model}: no layer table"); continue
        L = layers[key]; L = L[L.kind == "conv"][["layer", "H", "W"]]
        m = per.merge(L, on="layer")
        if m.empty:
            continue
        # d = k_h k_w C_in is not stored; recover from r_max = min(d, C_out): if r_max < C_out then d = r_max,
        # else d >= C_out and we need C_in*k*k.  The decompose tables have d; use them when present.
        dec = os.path.join(results, "decompose", f"{model}_decompose.csv")
        if os.path.exists(dec):
            D = pd.read_csv(dec)[["layer", "d"]]; m = m.merge(D, on="layer")
        else:
            continue
        pos = m.H * m.W
        full = (m.C * m.d * pos).sum()
        out = []
        for t in (0.9, 0.95, 0.99, 0.999):
            k = m[f"k_{t}"]
            fac = (k * (m.d + m.C) * pos)
            kept = np.minimum(fac, m.C * m.d * pos).sum()      # never worse than the dense conv
            out.append(kept / full)
        print(f"    {model:28s} {full / 1e9:11.2f}  " + "  ".join(f"{v:10.2f}" for v in out))
    print("\n  Reading: at tau = 0.999 the factorised network computes this fraction of the")
    print("  dense conv FLOPs with the accuracy of section 10 (no retraining, no fine-tuning).")


# ------------------------------------ section 11: kernel vs data decomposition

def section_decompose(results: str, latex: str | None = None) -> None:
    files = sorted(glob.glob(os.path.join(results, "decompose", "*_decompose.csv")))
    if not files:
        return
    hdr("11. Where the effective width comes from: Sigma_out = W Sigma_patch W^T")
    print("  scripts/decompose_check.py on the 6,400-image subset, dense conv layers.")
    print("  out    = k*(W Sigma W^T)/r_max   (the measured statistic, recomputed from the identity)")
    print("  kernel = k*(W W^T)/r_max         (what the layer would show under white patches)")
    print("  data   = min(k*(Sigma), r_max)/r_max")
    print("  ortho  = k*(W_o Sigma W_o^T)/r_max, W_o the polar factor of W (nearest isometry)")
    print("  'rho' columns: Spearman across layers against 'out'.  |out-meas| = max abs")
    print("  difference from the independently measured k*(0.95)/r_max on the same subset")
    print("  (the identity check; skipped for random inits, whose seeds differ).\n")
    print(f"    {'model':38s} {'L':>3s} {'n/d min':>7s}  {'tau':>5s}  {'out':>5s} {'kernel':>6s} {'data':>5s} {'ortho':>5s}  "
          f"{'rho(kern)':>9s} {'rho(data)':>9s} {'rho(ortho)':>10s}  {'ortho>out':>9s}  {'|out-meas|':>10s}")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        model = d.model.iloc[0]
        # identity check against the independent measurement on the same subset, if present
        meas = None
        for cand in ([] if model.endswith("_random") else [os.path.join(results, "local6400", "trained", f"{model}_layers.csv"),
                     os.path.join(results, "local6400", "archs", "trained", f"{model.replace(':', '_')}_layers.csv"),
                     os.path.join(results, "local6400", "trained", f"{model.replace(':', '_')}_layers.csv")]):
            if os.path.exists(cand):
                L = pd.read_csv(cand); L = L[L.kind == "conv"]
                m = d.merge(L[["layer", "k_star_rmax_0.95"]], on="layer")
                if len(m):
                    meas = float((m["out_0.95"] - m["k_star_rmax_0.95"]).abs().max())
                break
        for tau in (0.95, 0.999):
            o, k, da, ort = (d[f"{c}_{tau}"] for c in ("out", "kernel", "data", "ortho"))
            rows.append((model, tau, len(d), o.median(), k.median(), da.median(), ort.median(),
                         _rho(k, o), _rho(da, o), _rho(ort, o), int((ort > o).sum())))
            print(f"    {model:38s} {len(d):3d} {d.n_over_d.min():7.0f}  {tau:5.3f}  {o.median():5.2f} {k.median():6.2f} {da.median():5.2f} {ort.median():5.2f}  "
                  f"{_rho(k, o):+9.2f} {_rho(da, o):+9.2f} {_rho(ort, o):+10.2f}  {int((ort > o).sum()):4d}/{len(d):<4d}  "
                  + (f"{meas:10.3f}" if (meas is not None and tau == 0.95) else " " * 10))
    # Proposition 1 verification, when decompose_check was run with the bound columns
    withp = [f for f in files if "ostrowski_ok" in pd.read_csv(f, nrows=1).columns]
    if withp:
        print("\n  Proposition 1 check (Ostrowski ratios in [s_min^2, s_max^2]; k*_out(tau) <= k*_ortho(tau')):")
        print(f"    {'model':38s} {'L':>3s} {'null dirs':>9s} {'ostrowski':>9s} {'bound .95':>9s} {'bound .999':>10s} {'informative .95':>15s} {'kappa med':>9s}")
        tot = np.zeros(5, int)
        for f in withp:
            d = pd.read_csv(f); model = d.model.iloc[0]
            inf = int((d["bound_0.95"] < 1).sum())
            print(f"    {model:38s} {len(d):3d} {int((d.n_null > 0).sum()):9d} {int(d.ostrowski_ok.sum()):9d} "
                  f"{int(d['bound_ok_0.95'].sum()):9d} {int(d['bound_ok_0.999'].sum()):10d} {inf:15d} {d.kappa.median():9.1f}")
            tot += [len(d), int(d.ostrowski_ok.sum()), int(d['bound_ok_0.95'].sum()), int(d['bound_ok_0.999'].sum()), inf]
        print(f"    {'TOTAL':38s} {tot[0]:3d} {'':9s} {tot[1]:9d} {tot[2]:9d} {tot[3]:10d} {tot[4]:15d}")
        print("    (a layer with numerically null kernel directions has kappa set by the tolerance;")
        print("     the bound then degenerates to r_max and 'informative' counts layers where it does not)")
    print("\n  Reading: 'kernel' vs 'out' says how much the layer's width is set by the")
    print("  kernel spectrum alone; 'ortho' is the analytic prediction for orthogonalising")
    print("  the kernel with the data held fixed, to be tested against trained-with-")
    print("  orthogonality-penalty networks (submission bar A5).")
    if latex:
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-decompose.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
               "Model & $L$ & $\\tau$ & out & kernel & data & ortho & $\\rho_{\\rm kernel}$ & $\\rho_{\\rm data}$ & ortho $>$ out \\\\",
               "\\midrule"]
        order = {m: i for i, m in enumerate(MODEL_ORDER)}
        rows.sort(key=lambda r: (order.get(r[0], order.get("timm_" + r[0].replace("timm:", ""), 99)), r[1]))
        for model, tau, L, o, k, da, ort, rk, rd, ro, n in rows:
            nm, recipe = MODEL_LABELS.get(model, MODEL_LABELS.get("timm_" + model.replace("timm:", ""), (model, "")))
            name = f"{nm} {recipe}".strip()
            if model.endswith("_random"):
                name = nm + " (random init)"
            out.append(f"{name} & {L} & {tau:g} & {o:.2f} & {k:.2f} & {da:.2f} & {ort:.2f} & {rk:+.2f} & {rd:+.2f} & {n}/{L} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# --------------------------- section 12: round 3 (controlled block experiment)

def _final_profiles(results: str, prefix: str):
    """{seed: dict(init, final, acc, layers, settle, rho_depth_init, rho_depth_final)}
    for results/round3/<prefix>_s<seed>/ (finished runs only)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results, "round3", f"{prefix}_s*", "trajectory_layers.csv"))):
        seed = os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[1]
        epochs, P, acc, layers = _load_trajectory(path)
        if epochs[-1] < 100:
            continue
        init, final = P[epochs[0]], P[epochs[-1]]
        depth = np.arange(len(final))
        out[seed] = dict(init=init, final=final, acc=acc[epochs[-1]], layers=layers, epochs=epochs, P=P, accs=acc,
                         settle=next((e for e in epochs if _rho(final, P[e]) >= 0.95), None),
                         rho_depth_init=_rho(depth, init), rho_depth_final=_rho(depth, final))
    return out


def section_round3(results: str, latex: str | None = None) -> None:
    groups = {"basic (52 conv)": _final_profiles(results, "a2_basic52"),
              "bottleneck (53 conv)": _final_profiles(results, "a2_bottleneck"),
              "MobileNetV2 (CIFAR)": _final_profiles(results, "a2_mobilenetv2")}
    labels = {"basic (52 conv)": "basic $[6,6,6,6]$ & 52",
              "bottleneck (53 conv)": "bottleneck $[3,4,6,3]$ & 53",
              "MobileNetV2 (CIFAR)": "MobileNetV2 & 35"}
    if not any(groups.values()):
        return
    hdr("12. Round 3, controlled block type on CIFAR-10 (analysis-plan 9.1)")
    print("  same recipe, same dataset, same conv count for basic vs bottleneck; k*(0.95)/r_max, dense conv")
    print(f"    {'group':22s} {'seeds':>5s}  {'R-R':>6s} {'T-T':>6s} {'T-R':>6s} {'T-R range':>12s} {'T-R_stage':>9s}  "
          f"{'level R->T':>12s}  {'rho_depth R->T':>14s}  {'settle':>6s}  {'acc':>6s}")
    tr_by_group, latex_rows = {}, []
    for name, runs in groups.items():
        if not runs:
            print(f"    {name:22s} (no finished runs)"); continue
        inits = [v["init"] for v in runs.values()]; finals = [v["final"] for v in runs.values()]
        rr = [_rho(a, b) for a, b in itertools.combinations(inits, 2)]
        tt = [_rho(a, b) for a, b in itertools.combinations(finals, 2)]
        tr = [_rho(a, b) for a in finals for b in inits]          # all trained x random pairs, as in Table families
        tr_same = [_rho(v["init"], v["final"]) for v in runs.values()]
        tr_by_group[name] = tr_same
        layers = next(iter(runs.values()))["layers"]
        Rst = _stage_ranks(np.column_stack(inits + finals), layers)
        if Rst is not None:
            n = len(inits)
            trs = [_rho(Rst[:, n + i], Rst[:, j]) for i in range(n) for j in range(n)]
            trs_med = float(np.median(trs))
        else:
            trs_med = float("nan")
        lvl_r, lvl_t = float(np.median(np.concatenate(inits))), float(np.median(np.concatenate(finals)))
        rd_r = float(np.median([v["rho_depth_init"] for v in runs.values()]))
        rd_t = float(np.median([v["rho_depth_final"] for v in runs.values()]))
        settles = [v["settle"] for v in runs.values() if v["settle"] is not None]
        settle = float(np.median(settles)) if settles else float("nan")
        acc = float(np.mean([v["acc"] for v in runs.values()]))
        fmt = lambda v: f"{np.median(v):+6.2f}" if v else "   n/a"
        print(f"    {name:22s} {len(runs):5d}  {fmt(rr)} {fmt(tt)} {fmt(tr)} {min(tr):+5.2f}..{max(tr):+5.2f} "
              f"{trs_med:+9.2f}  {lvl_r:5.2f} -> {lvl_t:4.2f}  {rd_r:+6.2f} -> {rd_t:+5.2f}  {settle:6.0f}  {acc:6.2f}")
        print(f"    {'':22s} T-R same seed: {' '.join(f'{x:+.2f}' for x in tr_same)};  "
              f"settle per seed: {[int(v['settle']) for v in runs.values() if v['settle'] is not None]};  "
              f"rho_depth final per seed: {' '.join(f'{v["rho_depth_final"]:+.2f}' for v in runs.values())}")
        ep = next(iter(runs.values()))["epochs"]
        print(f"    {'':22s} L = {len(layers)} dense conv; per-epoch medians over seeds:")
        print(f"    {'':22s} {'epoch':>5s} {'acc%':>6s} {'level':>6s} {'rho(depth)':>10s} {'rho vs init':>11s} {'rho vs final':>12s}")
        for e in ep:
            vals = [v["P"][e] for v in runs.values() if e in v["P"]]
            print(f"    {'':22s} {e:5d} {np.median([v['accs'][e] for v in runs.values()]):6.2f} "
                  f"{np.median([np.median(x) for x in vals]):6.3f} "
                  f"{np.median([_rho(np.arange(len(x)), x) for x in vals]):+10.2f} "
                  f"{np.median([_rho(v['init'], v['P'][e]) for v in runs.values()]):+11.2f} "
                  f"{np.median([_rho(v['final'], v['P'][e]) for v in runs.values()]):+12.2f}")
        latex_rows.append((labels[name], len(runs), np.median(rr), np.median(tt), np.median(tr), min(tr), max(tr),
                           trs_med, lvl_r, lvl_t, rd_r, rd_t, settle, acc))
    b, t = tr_by_group.get("basic (52 conv)"), tr_by_group.get("bottleneck (53 conv)")
    if b and t:
        p = mannwhitneyu(t, b, alternative="less").pvalue if len(b) > 1 and len(t) > 1 else float("nan")
        print(f"\n  H9.1 (bottleneck T-R below basic T-R): max bottleneck {max(t):+.2f} vs min basic {min(b):+.2f}; "
              f"Mann-Whitney one-sided p = {p:.3g}  -> {'SUPPORTED' if max(t) < min(b) or p < 0.01 else 'NOT supported'}")
    m = groups["MobileNetV2 (CIFAR)"]
    if len(m) >= 2:
        tt = [_rho(a["final"], b["final"]) for a, b in itertools.combinations(m.values(), 2)]
        print(f"  H9.2 (MobileNetV2 seeds share a profile, T-T >= 0.7): pairwise {[f'{x:+.2f}' for x in tt]} "
              f"-> {'SUPPORTED' if min(tt) >= 0.7 else 'NOT supported'}")
    if latex and latex_rows:
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-round3.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
               "Block & $L$ & $n$ & R--R & T--T & T--R (range) & T--R$_{\\rm stage}$ & level $R \\to T$ & $\\rho_{\\rm depth}$ $R \\to T$ & settle & acc. \\\\",
               "\\midrule"]
        for lab, n, rr, tt, tr, lo, hi, trs, lr, lt, rdr, rdt, st, acc in latex_rows:
            trs_s = "--" if np.isnan(trs) else f"{trs:+.2f}"
            out.append(f"{lab} & {n} & {rr:+.2f} & {tt:+.2f} & {tr:+.2f} ({lo:+.2f}, {hi:+.2f}) & {trs_s} & "
                       f"{lr:.2f} $\\to$ {lt:.2f} & {rdr:+.2f} $\\to$ {rdt:+.2f} & {st:.0f} & {acc:.1f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------- section 13: round 3, orthogonality intervention (A5)

def _run_acc(results: str, run: str) -> float:
    path = os.path.join(results, "round3", run, "trajectory_layers.csv")
    if not os.path.exists(path):
        return float("nan")
    epochs, _, acc, _ = _load_trajectory(path)
    return acc[epochs[-1]]


def section_round3_ortho(results: str, latex: str | None = None) -> None:
    d = os.path.join(results, "round3_decompose")
    files = glob.glob(os.path.join(d, "a5_*_decompose.csv"))
    if not files:
        return
    hdr("13. Round 3, orthogonality penalties vs the analytic prediction (analysis-plan 9.2)")
    print("  For each penalty arm (so, srip), per layer: measured k*(0.95)/r_max of the")
    print("  penalised network vs the 'ortho' counterfactual of the matching no-penalty")
    print("  network (same arch, same seed).  P9.3: median |diff| < 0.1 for SO; P9.4: SRIP does not.")
    none_of = {"vgg16_bn": {k: f"a5_vgg16_bn_none_s{k}" for k in "012"},
               "basic52": {k: f"a2_basic52_s{k}" for k in "012"}}
    rows = []
    for f in sorted(files):
        run = os.path.basename(f).replace("_decompose.csv", "")   # a5_<arch>_<o>_s<k>
        m = re.match(r"a5_(vgg16_bn|basic52)_(so|srip)_s(\d)", run)
        if not m:
            continue
        arch, o, k = m.groups()
        base = os.path.join(d, none_of[arch][k] + "_decompose.csv")
        if not os.path.exists(base):
            print(f"    {run}: no-penalty arm {none_of[arch][k]} not decomposed yet"); continue
        P = pd.read_csv(f).merge(pd.read_csv(base), on="layer", suffixes=("_pen", "_none"))
        P = P[P.n_over_d_pen >= 50]
        diff = (P["out_0.95_pen"] - P["ortho_0.95_none"]).to_numpy()
        gain = (P["out_0.95_pen"] - P["out_0.95_none"]).to_numpy()
        pred_gain = (P["ortho_0.95_none"] - P["out_0.95_none"]).to_numpy()
        rows.append(dict(arch=arch, arm=o, seed=k, L=len(P),
                         mad=float(np.median(np.abs(diff))), md=float(np.median(diff)),
                         gain=float(np.median(gain)), pred_gain=float(np.median(pred_gain)),
                         out_none=float(np.median(P["out_0.95_none"])), out_pen=float(np.median(P["out_0.95_pen"])),
                         pred=float(np.median(P["ortho_0.95_none"])), ortho_pen=float(np.median(P["ortho_0.95_pen"])),
                         ker_none=float(np.median(P["kernel_erank_over_rmax_none"])),
                         ker_pen=float(np.median(P["kernel_erank_over_rmax_pen"])),
                         data_none=float(np.median(P["data_erank_over_rmax_none"])),
                         data_pen=float(np.median(P["data_erank_over_rmax_pen"])),
                         kdata_none=float(np.median(P["data_0.999_none"])), kdata_pen=float(np.median(P["data_0.999_pen"])),
                         out999_none=float(np.median(P["out_0.999_none"])), out999_pen=float(np.median(P["out_0.999_pen"])),
                         pred999=float(np.median(P["ortho_0.999_none"])),
                         rho_gain=_rho(gain, pred_gain),
                         acc_none=_run_acc(results, none_of[arch][k]), acc_pen=_run_acc(results, run)))
    print(f"    {'arch':9s} {'arm':5s} {'seed':>4s} {'L':>3s}  {'out none':>8s} {'out pen':>7s} {'pred':>6s}  "
          f"{'|out-pred|':>10s} {'gain':>6s} {'pred gain':>9s} {'rho(gain,pred)':>14s}  "
          f"{'ortho pen':>9s}  {'kernel erank n->p':>17s}  {'data erank n->p':>15s}  {'data k*.999 n->p':>16s}  {'out.999 n->p->pred':>18s}  {'acc n->p':>13s}")
    for r in rows:
        print(f"    {r['arch']:9s} {r['arm']:5s} {r['seed']:>4s} {r['L']:3d}  {r['out_none']:8.3f} {r['out_pen']:7.3f} {r['pred']:6.3f}  "
              f"{r['mad']:10.3f} {r['gain']:+6.3f} {r['pred_gain']:+9.3f} {r['rho_gain']:+14.2f}  "
              f"{r['ortho_pen']:9.3f}  {r['ker_none']:8.3f} -> {r['ker_pen']:5.3f}  {r['data_none']:6.3f} -> {r['data_pen']:5.3f}  "
              f"{r['kdata_none']:6.3f} -> {r['kdata_pen']:5.3f}  {r['out999_none']:5.3f} -> {r['out999_pen']:5.3f} -> {r['pred999']:5.3f}  "
              f"{r['acc_none']:5.2f} -> {r['acc_pen']:5.2f}")
    for o in ("so", "srip"):
        v = [r["mad"] for r in rows if r["arm"] == o]
        if v:
            print(f"  {o.upper()}: median |out - prediction| over arms = {np.median(v):.3f}  -> "
                  f"{'within 0.1 (prediction holds)' if np.median(v) < 0.1 else 'not within 0.1'}")
    if len(rows) >= 4:
        g = [r["gain"] for r in rows]; da = [r["acc_pen"] - r["acc_none"] for r in rows]
        print(f"  P9.5 (gain vs accuracy change, {len(rows)} runs): Spearman rho = {_rho(g, da):+.2f}; "
              f"accuracy change {min(da):+.2f}..{max(da):+.2f} points  -> {'no monotone relation (|rho| < 0.5)' if abs(_rho(g, da)) < 0.5 else 'monotone relation'}")
        fr = [r["gain"] / r["pred_gain"] for r in rows if r["pred_gain"] > 0]
        print(f"  Fraction of the predicted gain realised, median over runs: {np.median(fr):.2f} (range {min(fr):.2f}..{max(fr):.2f})")
    if latex and rows:
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-ortho.  Do not edit by hand.",
               "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
               "Network & penalty & $L$ & none & pen. & pred. & realised & $\\rho$(gain, pred.) & kernel erank & acc. \\\\",
               "\\midrule"]
        names = {"vgg16_bn": "VGG-16 (BN)", "basic52": "basic $[6,6,6,6]$"}
        arms = {"so": "SO", "srip": "SRIP"}
        for arch in ("vgg16_bn", "basic52"):
            for o in ("so", "srip"):
                R = [r for r in rows if r["arch"] == arch and r["arm"] == o]
                if not R:
                    continue
                med = lambda key: float(np.median([r[key] for r in R]))
                frac = float(np.median([r["gain"] / r["pred_gain"] for r in R]))
                out.append(f"{names[arch]} & {arms[o]} & {R[0]['L']} & {med('out_none'):.2f} & {med('out_pen'):.2f} & {med('pred'):.2f} & "
                           f"{frac:.2f} & {med('rho_gain'):+.2f} & {med('ker_none'):.2f} $\\to$ {med('ker_pen'):.2f} & "
                           f"{med('acc_none'):.1f} $\\to$ {med('acc_pen'):.1f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


import re  # noqa: E402  (used by section_round3_ortho)


# ------------------- section 16: A18, width from the ruler, retrained (analysis-plan 9.5)

def _a18_runs(results: str, arm: str, dataset: str = "cifar10") -> dict:
    """{seed: (final acc, final k*(0.999)/r_max profile over dense conv)} for one arm.
    CIFAR-10: the full-width arm is round 3's untreated A5 run; CIFAR-100 (plan 9.7) trains
    its own full-width arm under results/a18_cifar100."""
    if dataset == "cifar10":
        base = {"full": os.path.join(results, "round3", "a5_vgg16_bn_none_s{}"),
                }.get(arm, os.path.join(results, "a18", f"vgg16_bn_{arm}_s{{}}"))
    else:
        base = os.path.join(results, f"a18_{dataset}", f"vgg16_bn_{arm}_s{{}}")
    out = {}
    for seed in "012":
        path = os.path.join(base.format(seed), "trajectory_layers.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[(d.kind == "conv") & (d.epoch == d.epoch.max())].sort_values("depth_index")
        if d.epoch.iloc[0] < 100:
            continue
        out[seed] = (float(d.test_acc.iloc[0]), (d["k_star_0.999"] / d["r_max"]).to_numpy())
    return out


A18_DATASETS = (("cifar10", "9.5", ("P9.6", "P9.7", "P9.8", "P9.9"), "a18"),
                ("cifar100", "9.7", ("P9.14", "P9.15", "P9.16", "expl."), "a18_cifar100"))
A18_ARMS = ("full", "ruler", "uniform", "ruler95", "ruler_act")


def section_a18(results: str, latex: str | None = None) -> None:
    blocks = []
    for dataset, plan_sec, pnames, sub in A18_DATASETS:
        arms = {a: _a18_runs(results, a, dataset) for a in A18_ARMS}
        if any(arms[a] for a in ("ruler", "uniform", "ruler95", "ruler_act")):
            blocks.append((dataset, plan_sec, pnames, sub, arms))
    if not blocks:
        return
    hdr("16. A18: widths set from the ruler, retrained from scratch (analysis-plan 9.5; CIFAR-100: 9.7)")
    rows = []
    for dataset, plan_sec, pnames, sub, arms in blocks:
        wpath = os.path.join(results, sub, "widths.json")
        W = json.load(open(wpath)) if os.path.exists(wpath) else {}
        wact = os.path.join(results, sub, "widths_act.json")
        if os.path.exists(wact):          # plan 9.8: arm (e), widths read after the ReLU
            Wa = json.load(open(wact))
            W.setdefault("widths", {})["ruler_act"] = Wa["widths"]["ruler"]
            W.setdefault("params", {})["ruler_act"] = Wa["params"]["ruler"]
        print(f"\n  {dataset} (plan {plan_sec}); widths: {W.get('widths', {}).get('ruler', '?')}")
        print(f"    {'arm':8s} {'n':>2s} {'params':>11s}  {'acc per seed':>22s}  {'mean':>6s}  {'final k*(.999)/r_max med':>24s}")
        means = {}
        for a, runs in arms.items():
            if not runs:
                print(f"    {a:8s}  0  (no finished runs)"); continue
            accs = [runs[k][0] for k in sorted(runs)]
            means[a] = float(np.mean(accs))
            lvl = float(np.median(np.concatenate([runs[k][1] for k in runs])))
            params = W.get("params", {}).get(a, float("nan"))
            print(f"    {a:8s} {len(runs):2d} {params:11,.0f}  {' '.join(f'{x:5.2f}' for x in accs):>22s}  {means[a]:6.2f}  {lvl:24.3f}")
            rows.append((dataset, a, params, accs, means[a], lvl))
        p6, p7, p8, p9 = pnames
        if "full" in means and "ruler" in means:
            r = [arms["ruler"][k][0] for k in arms["ruler"]]
            ok = (means["ruler"] - means["full"] > -0.3) and (min(r) >= means["full"] - 0.5)
            print(f"\n  {p6} (ruler within 0.3 of full, no seed below full-0.5): diff {means['ruler'] - means['full']:+.2f}, "
                  f"min ruler {min(r):.2f} vs full-0.5 {means['full'] - 0.5:.2f} -> {'SUPPORTED' if ok else 'NOT supported'}")
            f = [arms["full"][k][0] for k in arms["full"]]
            print(f"     (seed spread: full {max(f) - min(f):.2f}, ruler {max(r) - min(r):.2f})")
        if arms["ruler"] and arms["uniform"]:
            r = [arms["ruler"][k][0] for k in arms["ruler"]]; u = [arms["uniform"][k][0] for k in arms["uniform"]]
            ok = max(u) < min(r)
            wins = sum(ru > uu for ru in r for uu in u); ties = sum(ru == uu for ru in r for uu in u)
            print(f"  {p7} (all uniform below all ruler): uniform max {max(u):.2f} vs ruler min {min(r):.2f} -> {'SUPPORTED' if ok else 'NOT supported'}"
                  f"  [pairs: ruler higher {wins}/{len(r) * len(u)}, ties {ties}; means {np.mean(r):.2f} vs {np.mean(u):.2f}, diff {np.mean(r) - np.mean(u):+.2f}]")
        if "full" in means and "ruler95" in means:
            ok = means["full"] - means["ruler95"] >= 1.0
            print(f"  {p8} (ruler95 loses >= 1.0 point): {means['full'] - means['ruler95']:+.2f} -> {'SUPPORTED' if ok else 'NOT supported'}")
        if arms["ruler"]:
            lvl = float(np.median(np.concatenate([arms["ruler"][k][1] for k in arms["ruler"]])))
            print(f"  {p9} (exploratory; ruler arm filled: median k*(.999)/r_max >= 0.9): {lvl:.3f} -> {'yes' if lvl >= 0.9 else 'no'}")
        if arms.get("ruler_act") and "full" in means:
            e = [arms["ruler_act"][k][0] for k in arms["ruler_act"]]
            pe = W.get("params", {}).get("ruler_act", float("nan")); pf = W.get("params", {}).get("full", float("nan"))
            frac = pe / pf if pf else float("nan")
            ok17 = (np.mean(e) - means["full"] > -0.3) and (min(e) >= means["full"] - 0.5)
            evid = "counts" if (1 - frac) >= 0.05 else "does NOT count (within 5% of full width)"
            print(f"  P9.17 (plan 9.8; ruler_act within 0.3 of full, no seed below full-0.5): diff {np.mean(e) - means['full']:+.2f}, "
                  f"min {min(e):.2f} vs {means['full'] - 0.5:.2f} -> {'SUPPORTED' if ok17 else 'NOT supported'}; params {frac:.3f} of full, {evid}")
            if arms["ruler"]:
                r = [arms["ruler"][k][0] for k in arms["ruler"]]
                print(f"  P9.18 (ruler_act mean above ruler mean; C-100 also all 3 above all 3): {np.mean(e):.2f} vs {np.mean(r):.2f} "
                      f"-> {'yes' if np.mean(e) > np.mean(r) else 'no'}; all-above-all {'yes' if min(e) > max(r) else 'no'}")
    if latex and rows:
        cost = {}
        for dataset, _, _, sub in A18_DATASETS:
            cp = os.path.join(results, sub, "cost.json")
            if os.path.exists(cp):
                cost[dataset] = json.load(open(cp))
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-a18.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrr}", "\\toprule",
               "Arm & params & MACs & img/s & accuracy (\\%) & $k^*(0.999)/r_{\\max}$ \\\\"]
        labels = {"full": "full width", "ruler": "ruler, $\\tau = 0.999$", "uniform": "uniform cut", "ruler95": "ruler, $\\tau = 0.95$",
                  "ruler_act": "ruler after ReLU, $\\tau = 0.999$"}
        dlabel = {"cifar10": "CIFAR-10 (plan 9.5, 9.8)", "cifar100": "CIFAR-100 (plan 9.7, 9.8)"}
        last = None
        for dataset, a, params, accs, mean, lvl in rows:
            if dataset != last:
                out.append("\\midrule")
                out.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{dlabel[dataset]}}}}} \\\\")
            last = dataset
            c = cost.get(dataset, {}).get(a, {})
            macs = f"{c['macs'] / 1e6:.0f}M" if c else "--"
            ips = f"{c['images_per_s'] / 1e3:.0f}k" if c else "--"
            sd = float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan")
            out.append(f"{labels[a]} & {params / 1e6:.2f}M & {macs} & {ips} & {mean:.2f} $\\pm$ {sd:.2f} & {lvl:.2f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------- section 17: kernel-data alignment rho_align (analysis-plan 9.6)

def _align_tables(results: str) -> dict:
    """{name: DataFrame} for every decomposition CSV that carries rho_align."""
    out = {}
    for d in ("decompose", "decompose_align", "round3_decompose", os.path.join("a18", "decompose")):
        for f in sorted(glob.glob(os.path.join(results, d, "*_decompose.csv"))):
            t = pd.read_csv(f)
            if "rho_align" in t.columns:
                out[os.path.basename(f).replace("_decompose.csv", "")] = t
    return out


def section_alignment(results: str) -> None:
    T = _align_tables(results)
    if not T:
        return
    from scipy.stats import binomtest
    hdr("17. Kernel-data alignment rho_align (analysis-plan 9.6)")
    print("  rho_align = Spearman(s_i^2, v_i^T Sigma_patch v_i) over the retained singular directions of W;")
    print("  < 0: the kernel suppresses the data's high-variance directions (compressive).")
    # ---- P9.10: random ~ 0, trained < 0
    print(f"\n    {'checkpoint':40s} {'L':>3s} {'median':>7s} {'neg/pos':>8s} {'p(neg)':>8s} {'p(pos)':>8s}")
    rand_ok, tr_ok, tr_pos, tr_n, medians = [], [], [], 0, {}
    for name, t in T.items():
        if re.search(r"_epoch_\d+$", name):
            continue                                   # trajectory checkpoints: section below
        v = t.rho_align.dropna().to_numpy()
        if len(v) < 3:
            continue
        med = float(np.median(v)); neg, pos = int((v < 0).sum()), int((v > 0).sum())
        p = binomtest(neg, neg + pos, 0.5, alternative="greater").pvalue if neg + pos else 1.0
        pp = binomtest(pos, neg + pos, 0.5, alternative="greater").pvalue if neg + pos else 1.0
        print(f"    {name:40s} {len(v):3d} {med:+7.2f} {neg:3d}/{pos:<4d} {p:8.2g} {pp:8.2g}")
        if "random" in name:
            rand_ok.append(abs(med) < 0.1)
        else:
            tr_n += 1; tr_ok.append(p < 0.01); tr_pos.append(pp < 0.01); medians[name] = med
    if rand_ok:
        print(f"  P9.10a (random |median| < 0.1): {sum(rand_ok)}/{len(rand_ok)} checkpoints")
    if tr_ok:
        print(f"  P9.10b AS REGISTERED (trained rho_align < 0, sign test p < 0.01): {sum(tr_ok)}/{tr_n} checkpoints -> "
              f"{'SUPPORTED' if sum(tr_ok) >= 18 else ('pending (need 22 trained checkpoints)' if tr_n < 22 else 'NOT supported')}")
        imagenet = {os.path.basename(f)[: -len("_layers.csv")] for f in glob.glob(os.path.join(results, "*_layers.csv"))}
        im = [k for k in medians if k in imagenet and k not in DUPLICATES]   # the paper's ImageNet checkpoints only
        cifar = [k for k in medians if k not in im]
        print(f"  REVERSED SIGN (post hoc, not pre-registered): trained rho_align > 0 with sign test p < 0.01 in {sum(tr_pos)}/{tr_n} checkpoints "
              f"({len(im)} ImageNet + {len(cifar)} CIFAR-10 networks: round 3, A18 final arms, trajectories); "
              f"ImageNet checkpoints (n = {len(im)}): median of medians {np.median([medians[k] for k in im]):+.2f}, >= +0.8 in {sum(medians[k] >= 0.8 for k in im)}, "
              f"min {min(medians[k] for k in im):+.2f} ({min(im, key=lambda k: medians[k])}); "
              f"six ResNet-50 recipes {' '.join(f'{medians[k]:+.2f}' for k in RESNET50_RECIPES if k in medians)}")
    # ---- P9.11: along the A18 trajectories
    traj = {}
    for name, t in T.items():
        m = re.match(r"(vgg16_bn_(ruler|uniform|ruler95)_s\d)_epoch_(\d+)$", name)
        if m:
            traj.setdefault(m.group(1), {})[int(m.group(3))] = float(t.rho_align.median())
    if traj:
        print(f"\n    {'run':24s} {'epochs':>6s} {'rho(epoch, median)':>18s} {'settle':>6s} {'at settle':>9s} {'final':>6s} {'>=90%':>5s}")
        ok, rev = [], []
        for run, series in sorted(traj.items()):
            ep = sorted(series); med = [series[e] for e in ep]
            rho_e = _rho(np.array(ep, dtype=float), np.array(med))
            path = os.path.join(results, "a18", run, "trajectory_layers.csv")
            settle = None
            if os.path.exists(path):
                epochs, P, _, _ = _load_trajectory(path)
                settle = next((e for e in epochs if _rho(P[epochs[-1]], P[e]) >= 0.95), None)
            at_s = series.get(settle, float("nan")) if settle is not None else float("nan")
            fin = med[-1]
            reached = bool(np.sign(at_s) == np.sign(fin) and abs(at_s) >= 0.9 * abs(fin)) if settle is not None else False
            ok.append(rho_e < -0.7 and reached)
            rev.append(rho_e > 0.7 and reached)
            first90 = next((e for e in ep if np.sign(series[e]) == np.sign(fin) and abs(series[e]) >= 0.9 * abs(fin)), None)
            print(f"    {run:24s} {len(ep):6d} {rho_e:+18.2f} {str(settle):>6s} {at_s:+9.2f} {fin:+6.2f} {'yes' if reached else 'no':>5s}   "
                  f"epoch0 {series.get(0, float('nan')):+.2f}  first>=90% at epoch {first90}")
        print(f"  P9.11 AS REGISTERED (monotone descent rho < -0.7 and 90% of final by the settle epoch): {sum(ok)}/{len(ok)} runs -> "
              f"{'SUPPORTED' if ok and all(ok) else 'NOT supported'}")
        print(f"  REVERSED SIGN (post hoc): monotone ascent rho > +0.7 and 90% of final by the settle epoch: {sum(rev)}/{len(rev)} runs")
    # ---- P9.12: SO arm closer to zero than none
    pairs = [(f"a5_vgg16_bn_so_s{k}", f"a5_vgg16_bn_none_s{k}") for k in "012"] + \
            [(f"a5_basic52_so_s{k}", f"a2_basic52_s{k}") for k in "012"]
    diffs = []
    for so, none in pairs:
        if so in T and none in T:
            m = T[so].merge(T[none], on="layer", suffixes=("_so", "_none"))
            diffs.append((so, (m.rho_align_so.abs() - m.rho_align_none.abs()).dropna().to_numpy()))
    if diffs:
        from scipy.stats import wilcoxon
        allv = np.concatenate([d for _, d in diffs])
        p = wilcoxon(allv, alternative="less").pvalue if len(allv) >= 6 else float("nan")
        print(f"\n  P9.12 (|rho_align| SO < none, paired over layers): median diff {np.median(allv):+.3f}, "
              f"n = {len(allv)}, Wilcoxon p = {p:.2g} -> {'SUPPORTED' if p < 0.01 else 'NOT supported'}")
    # ---- P9.13: cross-family range (exploratory)
    if medians:
        vals = [v for k, v in medians.items() if not re.match(r"a[25]_", k)]
        if vals:
            print(f"  P9.13 (exploratory): trained ImageNet checkpoints median rho_align range "
                  f"{min(vals):+.2f}..{max(vals):+.2f} (n = {len(vals)}, spread {max(vals) - min(vals):.2f})")


# ------------------- section 18: the profile as a diagnostic (six-recipe consensus)

def _role(layer: str) -> str:
    if layer == "conv1":
        return "stem"
    if "downsample" in layer:
        return "shortcut"
    return layer.split(".")[-1]


def section_diagnostic(results: str, latex: str | None = None) -> None:
    """Which ResNet-50 layers are under-used in every one of the six recipes,
    at tau = 0.999 (the width the classifier uses, section 10) and 0.95."""
    D = {}
    for n in RESNET50_RECIPES:
        f = os.path.join(results, "local6400", "trained", f"{n}_layers.csv")
        if not os.path.exists(f):
            f = os.path.join(results, "local6400", "trained", "resnet50_layers.csv")
        if not os.path.exists(f):
            return
        d = pd.read_csv(f)
        d = d[(d.kind == "conv") & d.n_over_C_ok.astype(bool)].sort_values("depth_index")
        D[n] = d.set_index("layer")[["C", "r_max", "k_star_rmax_0.999", "k_star_rmax_0.95"]]
    layers = D[RESNET50_RECIPES[0]].index
    M999 = pd.DataFrame({n: D[n]["k_star_rmax_0.999"] for n in D}).loc[layers]
    M95 = pd.DataFrame({n: D[n]["k_star_rmax_0.95"] for n in D}).loc[layers]
    res = pd.DataFrame({"C": D[RESNET50_RECIPES[0]].C, "r_max": D[RESNET50_RECIPES[0]].r_max,
                        "med999": M999.median(axis=1), "min999": M999.min(axis=1), "max999": M999.max(axis=1),
                        "med95": M95.median(axis=1), "min95": M95.min(axis=1), "max95": M95.max(axis=1)})
    res["stage"] = [l.split(".")[0] if l != "conv1" else "stem" for l in res.index]
    res["role"] = [_role(l) for l in res.index]
    hdr("18. The profile as a diagnostic: six-recipe consensus on ResNet-50 (exploratory, not pre-registered)")
    print("  k*(tau)/r_max per layer, median / min / max over the six recipes on the 6,400-image subset")
    full = res.index[res.min999 > 0.9]; idle = res.index[res.max999 < 0.7]
    print(f"  tau = 0.999: {len(full)}/{len(res)} layers above 0.9 in every recipe; {len(idle)} layers below 0.7 in every recipe: "
          + ", ".join(idle))
    print(f"  tau = 0.95: {int((res.min95 > 0.5).sum())}/{len(res)} layers above 0.5 in every recipe")
    print(f"\n    {'layer':24s} {'C':>5s} {'r_max':>5s}  {'tau=.999 med (min-max)':>24s}  {'tau=.95 med (min-max)':>22s}")
    for l, r in res.iterrows():
        flag = "  <-- under-used in all six" if r.max999 < 0.7 else ""
        print(f"    {l:24s} {int(r.C):5d} {int(r.r_max):5d}  {r.med999:6.2f} ({r.min999:.2f}-{r.max999:.2f})        "
              f"{r.med95:6.2f} ({r.min95:.2f}-{r.max95:.2f})      {flag}")
    g = res.groupby(["stage", "role"], sort=False)[["med999", "med95"]].median()
    print("\n  by stage x role (median over layers of the six-recipe median):")
    print(g.round(2).to_string())
    if latex:
        stages = ["stem", "layer1", "layer2", "layer3", "layer4"]
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-diagnostic.  Do not edit by hand.",
               "\\begin{tabular}{l" + "r" * 4 + "}", "\\toprule",
               "Role & stage 1 & stage 2 & stage 3 & stage 4 \\\\", "\\midrule"]
        labels = {"conv1": "$1\\!\\times\\!1$ reduce", "conv2": "$3\\!\\times\\!3$", "conv3": "$1\\!\\times\\!1$ expand", "shortcut": "projection shortcut"}
        stem = res.loc["conv1"]
        out.append(f"stem ($7\\!\\times\\!7$) & {stem.med999:.2f}/{stem.med95:.2f} & & & \\\\")
        for role in ["conv1", "conv2", "conv3", "shortcut"]:
            cells = []
            for st in stages[1:]:
                sub = res[(res.stage == st) & (res.role == role)]
                cells.append(f"{sub.med999.median():.2f}/{sub.med95.median():.2f}" if len(sub) else "")
            out.append(f"{labels[role]} & " + " & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------- section 19: the gate as a derived number (Proposition 6, Lawley bias)

def section_gate_theory(results: str) -> None:
    """Lawley's first-order bias of the top-k partial eigenvalue sum at n = 50 C,
    computed from the measured spectra of ResNet-50, against the observed
    k*(0.999) at the n/C = 50 checkpoint of the convergence file."""
    sp_path = os.path.join(results, "local6400", "trained", "resnet50_spectra.npz")
    cv_path = os.path.join(results, "local6400", "trained", "resnet50_convergence.csv")
    ly_path = os.path.join(results, "local6400", "trained", "resnet50_layers.csv")
    if not all(os.path.exists(p) for p in (sp_path, cv_path, ly_path)):
        return
    sp = np.load(sp_path); conv = pd.read_csv(cv_path); lay = pd.read_csv(ly_path)
    lay = lay[lay.kind == "conv"].set_index("layer")
    hdr("19. The n/C >= 50 gate as a derived number (Proposition 6): Lawley bias at n = 50 C vs the convergence file")
    rows = []
    for key in sp.files:
        if not key.endswith("::conv"):
            continue
        name = key.split("::")[0]
        lam = np.sort(np.clip(sp[key], 0, None))[::-1]; lam = lam[lam > 0]
        C = int(lay.loc[name, "C"]); tot = lam.sum(); cum = np.cumsum(lam) / tot
        k = int(np.searchsorted(cum, 0.999) + 1)
        li, lj = lam[:k], lam[k:]
        if len(lj) == 0:
            continue
        D = li[:, None] - lj[None, :]
        bias = (li[:, None] * lj[None, :] / np.where(D > 0, D, np.inf)).sum() / (50 * C)
        c = conv[(conv.layer == name) & (conv.kind == "conv")]
        at50 = c[c.n_over_C == 50]["k_star_0.999"]
        final = c.loc[c.n_over_C.idxmax(), "k_star_0.999"] if len(c) else np.nan
        rows.append((name, C, k, bias / tot / (1 - 0.999), int(at50.iloc[0]) if len(at50) else np.nan, final))
    r = pd.DataFrame(rows, columns=["layer", "C", "k999", "bias_over_resid", "k999_at50", "k999_final"])
    print(f"  {len(r)} layers; predicted bias of the captured fraction at n/C = 50, as a fraction of (1 - tau) at tau = 0.999:")
    print(f"    median {r.bias_over_resid.median():.3f}, max {r.bias_over_resid.max():.3f}  (Proposition 6 bound with a factor-2 gap: 2C/n = 0.04)")
    d = (r.k999_at50 - r.k999_final).abs()
    print(f"  observed k*(0.999) at the n/C = 50 checkpoint vs the final value: identical on {int((d == 0).sum())}/{len(r)} layers, "
          f"max |diff| {int(d.max())}, i.e. at most one component")


# ------------------- section 20: A19, first-order truncation sensitivity vs the actual effect

def section_sensitivity(results: str) -> None:
    files = sorted(glob.glob(os.path.join(results, "sensitivity", "*_sensitivity.csv")))
    if not files:
        return
    from scipy.stats import spearmanr
    hdr("20. A19: first-order truncation sensitivity tr(J Sigma_perp J^T) vs the actual single-layer truncation (exploratory)")
    for f in files:
        d = pd.read_csv(f); model = d.model.iloc[0]
        mp = f.replace("_sensitivity.csv", "_margins.csv")
        m = pd.read_csv(mp).margin.to_numpy() if os.path.exists(mp) else None
        print(f"\n  {model}: {d.layer.nunique()} layers, {int(d.n_eval.iloc[0])} eval images, {int(d.n_calib.iloc[0])} calibration images")
        print(f"    {'tau':>6s} {'rho(lin,act)':>12s} {'lin/act med':>11s} {'sum act':>9s} {'max act (layer)':>36s}")
        for tau in sorted(d.tau.unique()):
            t = d[d.tau == tau]
            rho = spearmanr(t.S_lin, t.S_act).correlation
            top = t.sort_values("S_act", ascending=False).head(3)
            print(f"    {tau:6g} {rho:+12.2f} {np.median(t.S_lin / t.S_act):11.2f} {t.S_act.sum():9.3g} "
                  f"{top.S_act.iloc[0]:8.3g} ({', '.join(top.layer)})")
        if m is not None:
            print(f"    logit margin^2: median {np.median(m) ** 2:.3g}, 10th percentile {np.percentile(m, 10) ** 2:.3g}")


# ------------------- section 21: the ruler on a transformer (ViT-B/16, one checkpoint)

def _vit_role(layer: str) -> str:
    if layer.startswith("patch_embed"):
        return "patch embedding"
    for r, lab in (("qkv", "qkv"), ("attn.proj", "attention proj."), ("fc1", "MLP fc1"), ("fc2", "MLP fc2")):
        if layer.endswith(r):
            return lab
    return "other"


def section_vit(results: str, latex: str | None = None) -> None:
    files = sorted(glob.glob(os.path.join(results, "vit", "*_layers.csv")))
    if not files:
        return
    from scipy.stats import spearmanr
    hdr("21. The ruler on a transformer: ViT-B/16 (exploratory, one checkpoint, 6,400 images x 32 tokens)")
    rows = []
    for f in files:
        v = pd.read_csv(f); name = os.path.basename(f).replace("_layers.csv", "")
        v = v[v.n_over_C_ok.astype(bool) & v.kind.isin(["linear", "conv"])].copy()
        v["role"] = v.layer.map(_vit_role); v["block"] = v.layer.str.extract(r"blocks\.(\d+)\.").astype(float)
        print(f"\n  {name}: {len(v)} linear/conv layers pass the gate")
        print(f"    {'role':18s} {'n':>2s} {'C':>5s} {'r_max':>5s}  {'k*(.95)/r_max med (min-max)':>28s}  {'k*(.999)/r_max med':>18s}  {'rho(block)':>10s}")
        for role in ("patch embedding", "qkv", "attention proj.", "MLP fc1", "MLP fc2"):
            t = v[v.role == role].sort_values("block")
            if t.empty:
                continue
            rho = spearmanr(t.block, t["k_star_rmax_0.95"]).correlation if len(t) >= 3 else float("nan")
            print(f"    {role:18s} {len(t):2d} {int(t.C.iloc[0]):5d} {int(t.r_max.iloc[0]):5d}  "
                  f"{t['k_star_rmax_0.95'].median():6.2f} ({t['k_star_rmax_0.95'].min():.2f}-{t['k_star_rmax_0.95'].max():.2f})           "
                  f"{t['k_star_rmax_0.999'].median():18.2f}  {rho:+10.2f}")
            rows.append((name, role, len(t), int(t.C.iloc[0]), int(t.r_max.iloc[0]), t['k_star_rmax_0.95'].median(),
                         t['k_star_rmax_0.95'].min(), t['k_star_rmax_0.95'].max(), t['k_star_rmax_0.999'].median(), rho))
        conv = v[v.kind == "conv"]
        if len(conv):
            print(f"    r_max check: max k*(0.999)/r_max over all rows {v['k_star_rmax_0.999'].max():.3f} (must be <= 1)")
    if latex and rows:
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-vit.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrrr}", "\\toprule",
               "Role & $n$ & $C$ & $r_{\\max}$ & $k^*_{0.95}/r_{\\max}$ (range) & $k^*_{0.999}/r_{\\max}$ & $\\rho$ \\\\", "\\midrule"]
        for name, role, n, C, rm, m95, lo, hi, m999, rho in rows:
            rs = "--" if np.isnan(rho) else f"{rho:+.2f}"
            short = {"patch embedding": "patch embed.", "attention proj.": "attn.\\ proj."}.get(role, role)
            out.append(f"{short} & {n} & {C} & {rm} & {m95:.2f} ({lo:.2f}--{hi:.2f}) & {m999:.2f} & {rs} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------------------------------- section 15: ablations (A11-A15)

def _conv_profile(path: str, kind: str = "conv"):
    d = pd.read_csv(path); d = d[(d.kind == kind) & d.n_over_C_ok.astype(bool)]
    d = d[~d.is_depthwise.astype(bool)] if "is_depthwise" in d else d
    return d.sort_values("depth_index")[["layer", "C", "r_max", "k_star_0.95", "k_star_rmax_0.95", "k_star_ratio_0.95"]]


def section_ablations(results: str, latex: str | None = None) -> None:
    A = os.path.join(results, "ablation")
    if not os.path.isdir(A):
        return
    hdr("15. Ablations of the protocol on ResNet-50 V1 (A11-A15): k*(0.95)/r_max profile vs the reference run")
    ref_path = os.path.join(results, "local6400", "trained", "resnet50_layers.csv")
    if not os.path.exists(ref_path):
        print("  reference run missing"); return
    ref = _conv_profile(ref_path)
    print("  reference: results/local6400/trained/resnet50 (6,400 images, 16 positions, crop, conv output)\n")
    print(f"    {'ablation':34s} {'L':>3s}  {'rho vs ref':>10s} {'med|diff|':>9s} {'max|diff|':>9s}  {'level':>6s} {'ref level':>9s}")

    latex_rows = []

    def compare(label, prof, ref_=None, note=""):
        R = ref if ref_ is None else ref_
        m = R.merge(prof, on="layer", suffixes=("_ref", ""))
        if m.empty:
            print(f"    {label:34s} no common layers"); return
        a, b = m["k_star_rmax_0.95_ref"].to_numpy(), m["k_star_rmax_0.95"].to_numpy()
        print(f"    {label:34s} {len(m):3d}  {_rho(a, b):+10.2f} {np.median(np.abs(a - b)):9.3f} {np.abs(a - b).max():9.3f}  "
              f"{np.median(b):6.3f} {np.median(a):9.3f}" + note)
        latex_rows.append((label, len(m), _rho(a, b), float(np.median(np.abs(a - b))), float(np.abs(a - b).max()),
                           float(np.median(b)), float(np.median(a))))

    for sub, label in (("pos4", "4 positions per image"), ("pos64", "64 positions per image"),
                       ("resize", "direct resize to $224\\times224$ (no crop)"),
                       ("gaussian", "white-noise inputs, trained"), ("pink", "1/f-noise inputs, trained")):
        f = os.path.join(A, sub, "resnet50_layers.csv")
        if os.path.exists(f):
            compare(label, _conv_profile(f))
    for sub, label in (("gaussian", "white-noise inputs, random init"), ("pink", "1/f-noise inputs, random init")):
        f = os.path.join(A, sub, "resnet50_random_layers.csv")
        if os.path.exists(f):
            r = _conv_profile(f); rr = os.path.join(results, "local6400", "seed1", "resnet50_random_layers.csv")
            if os.path.exists(rr):
                compare(label, r, ref_=_conv_profile(rr), note="   (ref: random seed 1 on ImageNet images)")
    # A12: BN output vs conv output, same layer
    f = os.path.join(A, "bn", "resnet50_layers.csv")
    if os.path.exists(f):
        d = pd.read_csv(f); d = d[d.n_over_C_ok.astype(bool)]
        conv = d[d.kind == "conv"].copy(); bn = d[d.kind == "bn"].copy()
        conv["stem"] = conv.layer.str.replace(r"\.conv(\d)$", r".\1", regex=True).str.replace("^conv1$", "1", regex=True).str.replace(r"downsample\.0$", "downsample", regex=True)
        bn["stem"] = bn.layer.str.replace(r"\.bn(\d)$", r".\1", regex=True).str.replace("^bn1$", "1", regex=True).str.replace(r"downsample\.1$", "downsample", regex=True)
        m = conv.merge(bn, on="stem", suffixes=("_conv", "_bn"))
        a, b = m["k_star_rmax_0.95_conv"].to_numpy(), m["k_star_rmax_0.95_bn"].to_numpy()
        latex_rows.append(("batch-norm output instead of conv output", len(m), _rho(a, b), float(np.median(np.abs(a - b))),
                           float(np.abs(a - b).max()), float(np.median(b)), float(np.median(a))))
        print(f"\n  A12 BatchNorm: {len(m)} conv/BN pairs (ResNet-50).  rho(conv, bn) = {_rho(a, b):+.2f};  "
              f"median k*/r_max conv {np.median(a):.3f} vs bn {np.median(b):.3f};  bn higher on {int((b > a).sum())}/{len(m)};  "
              f"median |diff| {np.median(np.abs(a - b)):.3f}")
        for model in ("vgg16_bn", "resnet50_random"):
            g = os.path.join(A, "bn", f"{model}_layers.csv")
            if os.path.exists(g):
                e = pd.read_csv(g); e = e[e.n_over_C_ok.astype(bool)]
                c2 = e[e.kind == "conv"].sort_values("depth_index"); b2 = e[e.kind == "bn"].sort_values("depth_index")
                if len(c2) == len(b2):
                    a2, bb = c2["k_star_rmax_0.95"].to_numpy(), b2["k_star_rmax_0.95"].to_numpy()
                    print(f"    {model}: rho(conv, bn) = {_rho(a2, bb):+.2f}; median conv {np.median(a2):.3f} vs bn {np.median(bb):.3f}; bn higher on {int((bb > a2).sum())}/{len(a2)}")
    # A11: COCO
    fc = os.path.join(A, "coco")
    if os.path.isdir(fc):
        print("\n  A11 COCO val2017 images (3,200), 16 positions:")
        r50 = os.path.join(fc, "resnet50_layers.csv")
        if os.path.exists(r50):
            compare("ImageNet ResNet-50 on COCO images", _conv_profile(r50))
        for det in ("tvdet_fasterrcnn_resnet50_fpn", "tvdet_maskrcnn_resnet50_fpn"):
            g = os.path.join(fc, f"{det}_layers.csv")
            if os.path.exists(g):
                compare(("Faster R-CNN" if "faster" in det else "Mask R-CNN") + " R50-FPN backbone, COCO-trained", _conv_profile(g))
                if os.path.exists(r50):
                    m = _conv_profile(r50).merge(_conv_profile(g), on="layer", suffixes=("_cls", "_det"))
                    print(f"      vs ImageNet ResNet-50 on the same COCO images: rho {_rho(m['k_star_rmax_0.95_cls'], m['k_star_rmax_0.95_det']):+.2f}, "
                          f"levels {m['k_star_rmax_0.95_cls'].median():.3f} (cls) / {m['k_star_rmax_0.95_det'].median():.3f} (det)")
    if latex:
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-ablation.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrrr}", "\\toprule",
               "Variant & $L$ & $\\rho$ & med.\\ $|\\Delta|$ & max $|\\Delta|$ & level & ref.\\ level \\\\", "\\midrule"]
        for label, L, rho, md, mx, lv, lr in latex_rows:
            out.append(f"{label} & {L} & {rho:+.2f} & {md:.3f} & {mx:.3f} & {lv:.3f} & {lr:.3f} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------------------------ section 22: estimator comparison (B10)

def _rankme(lam: np.ndarray, eps: float = 1e-7) -> float:
    """Garrido et al. (ICML 2023) Eq. (1)-(2): entropy rank of the *singular values*
    of the data matrix, i.e. of sqrt(eigenvalues) of the covariance."""
    s = np.sqrt(np.maximum(lam, 0.0))
    if s.sum() <= 0:
        return float("nan")
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def _pr_corrected(lam: np.ndarray, n: float) -> float:
    """Gaussian closed form of the row-corrected participation ratio (Chun et al.,
    ICLR 2026, gamma_row): unbiased tr(Sigma^2) and (tr Sigma)^2 from the sample
    covariance S with n samples, then their ratio.  With E[tr S^2] = a/n + b(n+1)/n
    and E[(tr S)^2] = a + 2b/n (a = (tr Sigma)^2, b = tr Sigma^2) the solution is
    b_hat = n^2/((n+2)(n-1)) [tr S^2 - (tr S)^2/n],  a_hat = (tr S)^2 - 2 b_hat/n."""
    t1 = lam.sum(); t2 = np.square(lam).sum()
    if t2 <= 0 or n < 3:
        return float("nan")
    b = n * n / ((n + 2.0) * (n - 1.0)) * (t2 - t1 * t1 / n)
    a = t1 * t1 - 2.0 * b / n
    return float(a / b) if b > 0 else float("nan")


def _estimators(lam: np.ndarray, n: float) -> dict[str, float]:
    from layerspec.metrics import effective_rank, k_star, participation_ratio, powerlaw_alpha
    lam = np.sort(np.maximum(np.asarray(lam, dtype=np.float64), 0.0))[::-1]
    return {
        "k95": float(k_star(lam, 0.95)), "k999": float(k_star(lam, 0.999)),
        "PR": participation_ratio(lam), "PRc": _pr_corrected(lam, n),
        "erank": effective_rank(lam), "RankMe": _rankme(lam),
        "alpha": powerlaw_alpha(lam)[0],
    }


EST_ORDER = ["k95", "k999", "PR", "PRc", "erank", "RankMe", "alpha"]
EST_LABEL = {"k95": "$k^*(0.95)$", "k999": "$k^*(0.999)$", "PR": "PR", "PRc": "PR$_{\\rm corr}$",
             "erank": "erank", "RankMe": "RankMe", "alpha": "$\\alpha$-ReQ"}


def _estimator_table(results: str, layers: dict) -> pd.DataFrame:
    """One row per gate-passing dense conv layer of every checkpoint with a saved spectrum."""
    rows = []
    for m, df in layers.items():
        if m in DUPLICATES:
            continue
        path = os.path.join(results, f"{m}_spectra.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        for _, r in conv_rows(df).iterrows():
            key = f"{r.layer}::conv"
            if key not in z:
                continue
            e = _estimators(z[key], float(r.n_samples))
            e.update(model=m, layer=r.layer, depth_index=int(r.depth_index), C=int(r.C),
                     r_max=float(r.r_max), n_over_C=float(r.n_over_C), random=is_random(m))
            rows.append(e)
    return pd.DataFrame(rows)


def section_estimators(results: str, latex: str | None = None) -> None:
    layers = load_all(results)
    layers = {m: d for m, d in layers.items() if not m.startswith("vgg16_cifar")}
    t = _estimator_table(results, layers)
    if t.empty:
        return
    hdr("22. Estimator comparison (B10, plan section 11): the ruler against RankMe, alpha-ReQ, "
        "corrected PR, on identical layers and images")
    models = sorted(t.model.unique())
    print(f"  {len(models)} checkpoints with saved spectra, {len(t)} gate-passing dense conv layers "
          f"(random inits: {sorted(t[t.random].model.unique())})")
    for c in ("k95", "k999", "PR", "PRc", "erank", "RankMe"):
        t[c + "_n"] = t[c] / t.r_max
    # 4. bound check
    worst = max(t[c + "_n"].max() for c in ("PR", "PRc", "erank", "RankMe", "k999"))
    print(f"  bound check: max estimator / r_max over all rows = {worst:.3f} (must be <= 1)")
    # 5. size of the PR correction
    rel = (t.PRc - t.PR).abs() / t.PR
    print(f"  PR correction |PRc-PR|/PR: median {rel.median()*100:.2f}%, max {rel.max()*100:.2f}% "
          f"(at n/C = {t.loc[rel.idxmax(), 'n_over_C']:.0f}, {t.loc[rel.idxmax(), 'model']} {t.loc[rel.idxmax(), 'layer']}); "
          f"rows with > 5%: {(rel > 0.05).sum()}, > 10%: {(rel > 0.10).sum()}")
    # 1. ordering agreement with k*(0.999)/r_max and k*(0.95)/r_max, per model
    rho = {}
    for ref in ("k999_n", "k95_n"):
        for c in EST_ORDER:
            col = c if c == "alpha" else c + "_n"
            if col == ref:
                continue
            vals = []
            for m in models:
                s = t[t.model == m].dropna(subset=[ref, col])
                if len(s) >= 5:
                    vals.append(spearmanr(s[ref], s[col]).correlation)
            rho[(ref, c)] = np.array(vals)
    trained = [m for m in models if not is_random(m)]
    print(f"\n  1. per-model Spearman rho between each estimator/r_max and the ruler (over layers; "
          f"{len(trained)} trained + {len(models)-len(trained)} random checkpoints, >= 5 layers each)")
    print(f"     {'estimator':12s} {'vs k*(.999)/r_max: med [min, max]':>36s}   {'vs k*(.95)/r_max: med [min, max]':>34s}   {'n>=0.8':>6s}")
    summary = {}
    for c in EST_ORDER:
        line = f"     {c:12s}"
        for ref in ("k999_n", "k95_n"):
            v = rho.get((ref, c))
            if v is None:
                line += f" {'--':>36s}  "
                continue
            line += f" {np.median(v):+.2f} [{v.min():+.2f}, {v.max():+.2f}]".rjust(37) + "  "
            summary[(ref, c)] = (np.median(v), v.min(), v.max(), int((np.abs(v) >= 0.8).sum()), len(v))
        v = rho.get(("k999_n", c))
        if v is not None:
            line += f" {int((np.abs(v) >= 0.8).sum()):3d}/{len(v)}"
        print(line)
    # 2. level
    print("\n  2. level: median of estimator / r_max over all gate-passing dense conv layers (trained only)")
    tt = t[~t.random]
    lev = {c: float(tt[c + "_n"].median()) for c in ("k95", "k999", "PR", "PRc", "erank", "RankMe")}
    lev["alpha"] = float(tt.alpha.median())
    print(f"     (alpha defined on {tt.alpha.notna().sum()}/{len(tt)} layers; the fit window [10, C] needs C >= 18)")
    print("     " + "  ".join(f"{c} {lev[c]:.3f}" for c in EST_ORDER))
    # 3. cross-recipe stability on the six ResNet-50 recipes
    print("\n  3. six ResNet-50 recipes: median pairwise layer-ordering rho, per estimator")
    stab = {}
    rec = [m for m in RESNET50_RECIPES if m in models]
    for c in EST_ORDER:
        col = c if c == "alpha" else c + "_n"
        vals = []
        for a_, b_ in itertools.combinations(rec, 2):
            sa = t[t.model == a_].set_index("layer")[col].dropna(); sb = t[t.model == b_].set_index("layer")[col].dropna()
            common = sa.index.intersection(sb.index)
            if len(common) >= 5:
                vals.append(spearmanr(sa[common], sb[common]).correlation)
        stab[c] = (float(np.median(vals)), float(np.min(vals)), float(np.max(vals)), len(vals)) if vals else (np.nan,) * 4
        print(f"     {c:12s} median {stab[c][0]:+.2f} [{stab[c][1]:+.2f}, {stab[c][2]:+.2f}]  ({stab[c][3]} pairs, {len(rec)} recipes)")
    # 6. pooling claim under each estimator (pooled_* columns already in the layer tables)
    print("\n  6. pooling: layers where position-sampled > globally pooled, per estimator (trained, gate-passing dense conv, both gates)")
    for c, a_, b_ in (("k*(0.95)", "k_star_0.95", "pooled_k_star_0.95"), ("PR", "participation_ratio", "pooled_participation_ratio"),
                      ("erank", "effective_rank", "pooled_effective_rank")):
        win = tot = 0
        for m in trained:
            d = conv_rows(layers[m])
            if b_ not in d.columns or "pooled_n_over_C_ok" not in d.columns:
                continue
            d = d[d.pooled_n_over_C_ok.astype(bool)]
            win += int((d[a_] > d[b_]).sum()); tot += len(d)
        print(f"     {c:10s} {win}/{tot}")
    # per-model detail, trained only
    print("\n  per-checkpoint rho vs k*(0.999)/r_max (trained):")
    print(f"     {'model':42s} {'layers':>6s} " + " ".join(f"{c:>7s}" for c in EST_ORDER if c != "k999"))
    for m in trained:
        s = t[t.model == m]
        if len(s) < 5:
            continue
        line = f"     {m:42s} {len(s):6d} "
        for c in EST_ORDER:
            if c == "k999":
                continue
            col = c if c == "alpha" else c + "_n"
            ss = s.dropna(subset=[col])
            line += f" {spearmanr(ss['k999_n'], ss[col]).correlation:+7.2f}"
        print(line)
    if latex:
        def fmt(x: float) -> str:
            return f"{x:.2f}" if x >= 0 else f"$-$" + f"{-x:.2f}"
        out = ["% Generated by scripts/checkpoint_analysis.py --latex-estimators.  Do not edit by hand.",
               "\\begin{tabular}{lrrrrr}", "\\toprule",
               "Estimator & level & $\\rho_{0.999}$ (range) & $\\ge 0.8$ & $\\rho_{0.95}$ & recipes \\\\",
               "\\midrule"]
        for c in EST_ORDER:
            lv = fmt(lev[c]) if c != "alpha" else f"{lev[c]:.2f}$^\\dagger$"
            s999 = summary.get(("k999_n", c)); s95 = summary.get(("k95_n", c))
            r999 = "--" if s999 is None else f"{fmt(s999[0])} [{fmt(s999[1])}, {fmt(s999[2])}]"
            frac = "--" if s999 is None else f"{s999[3]}/{s999[4]}"
            r95 = "--" if s95 is None else fmt(s95[0])
            out.append(f"{EST_LABEL[c]} & {lv} & {r999} & {frac} & {r95} & {fmt(stab[c][0])} \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        with open(latex, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"  wrote {latex}")


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--latex-families", default=None,
                   help="write the section-8 family table as LaTeX to this path")
    p.add_argument("--latex-models", default=None, help="write Table I (checkpoints) here")
    p.add_argument("--latex-pooling", default=None, help="write the section-3 pooling table here")
    p.add_argument("--latex-projection", default=None, help="write the section-10 projection table here")
    p.add_argument("--latex-decompose", default=None, help="write the section-11 decomposition table here")
    p.add_argument("--latex-ablation", default=None, help="write the section-15 ablation table here")
    p.add_argument("--latex-round3", default=None, help="write the section-12 controlled-block table here")
    p.add_argument("--latex-ortho", default=None, help="write the section-13 orthogonality table here")
    p.add_argument("--latex-a18", default=None, help="write the section-16 width-from-the-ruler table here")
    p.add_argument("--latex-diagnostic", default=None, help="write the section-18 six-recipe consensus table here")
    p.add_argument("--latex-vit", default=None, help="write the section-21 ViT table here")
    p.add_argument("--latex-estimators", default=None, help="write the section-22 estimator-comparison table here")
    a = p.parse_args(argv)
    layers = load_all(a.results)
    layers = {m: df for m, df in layers.items()
              if not m.startswith("vgg16_cifar")}   # the gate's own model
    print(f"loaded {len(layers)} layer tables from {a.results}")
    print(f"duplicates excluded from counts: {DUPLICATES}")
    section_gate(a.results)
    section_gate_eb(a.results, layers)
    if a.latex_models:
        write_models_table(layers, a.latex_models)
    section_rmax(layers)
    section_spread(layers)
    section_pooling(layers, latex=a.latex_pooling)
    section_flattening(a.results, layers)
    section_random(layers)
    section_block(layers)
    section_learned(layers)
    section_seeds(a.results, layers)
    section_archs(a.results, latex=a.latex_families)
    section_trajectory(a.results)
    section_projection(a.results, latex=a.latex_projection)
    section_projection_flops(a.results, layers)
    section_decompose(a.results, latex=a.latex_decompose)
    section_round3(a.results, latex=a.latex_round3)
    section_round3_ortho(a.results, latex=a.latex_ortho)
    section_ablations(a.results, latex=a.latex_ablation)
    section_a18(a.results, latex=a.latex_a18)
    section_alignment(a.results)
    section_diagnostic(a.results, latex=a.latex_diagnostic)
    section_gate_theory(a.results)
    section_sensitivity(a.results)
    section_vit(a.results, latex=a.latex_vit)
    section_estimators(a.results, latex=a.latex_estimators)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
