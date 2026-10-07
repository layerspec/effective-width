"""Order statistics that do not depend on the normaliser or on a level shift between stages.

Hostile review 2026-10-07 (findings 1-3, 8, 9): the all-layer Spearman between a trained and an initial
k*(tau)/r_max profile ranks layers of different stages and roles against each other, so a level shift
between stages, the choice of denominator (C or r_max) and the role mix all move it.  Within one stage
(same spatial grid H x W) and one kernel role (1x1 or 3x3) the ratio r_max/C is constant, so the ranks --
and any rank correlation -- are the same under either denominator.  This script reports, per run:

  * per (stage, role) Spearman between the initial and the final profile (n >= MIN_N layers);
  * the pooled within-(stage, role) rank correlation over all gated dense convolutions (one number);
  * the all-layer Spearman for reference, and the permutation null standard error 1/sqrt(n - 1).

Controlled runs are compared with their own initialisation (epoch 0 of the same seed).  Public families are
compared with the library-default random draw (timm, last BN gamma = 0 for the timm-named families) and,
where measured, with the gamma = 1 draw (results/bplus_controls/gamma1) and the gamma = 0 draw of ResNet-50
(results/init_axis/gamma0).

Usage:  cd code && python scripts/order_within_stage.py --results ../results [--out ../results/order_within_stage]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TAU = "k_star_0.95"
MIN_N = 4

# --------------------------------------------------------------------------- roles and stages

BOTTLENECK = ("resnet50", "resnet101", "wide_resnet50_2", "resnext50_32x4d", "bottleneck", "a28", "a31")
BASIC = ("resnet18", "resnet34", "basic52", "basic6666")


def family_of(run: str) -> str:
    r = run.lower()
    for key in ("wide_resnet50_2", "resnext50_32x4d", "resnet101", "resnet50", "resnet34", "resnet18",
                "basic6666", "basic52", "bottleneck", "mobilenetv2", "vgg16", "densenet121"):
        if key in r:
            return key
    if "/a28/" in r or r.startswith("a28") or "/a31/" in r or r.startswith("a31"):
        return "resnet50"
    return "unknown"


def role_of(family: str, layer: str) -> str:
    """'1x1', '3x3' or 'other' from the family's block type and the layer name."""
    name = layer.split("/")[-1]
    if "downsample" in name:
        return "1x1"
    if family in BOTTLENECK or family in ("a28", "a31"):
        if re.search(r"\.conv2$", name):
            return "3x3"
        if re.search(r"\.conv[13]$", name):
            return "1x1"
        return "3x3" if name in ("conv1",) else "other"
    if family in BASIC:
        return "3x3" if re.search(r"\.conv[12]$", name) or name == "conv1" else "other"
    if family == "vgg16":
        return "3x3" if name.startswith("features.") else "other"
    if family == "mobilenetv2":
        if re.search(r"conv\.(1|2)$", name) or name == "features.18.0":
            return "1x1"
        return "3x3" if name == "features.0.0" else "other"
    if family == "densenet121":
        if name.endswith("conv1") or "transition" in name:
            return "1x1"
        return "3x3" if name.endswith("conv2") or name == "features.conv0" else "other"
    return "other"


def stage_of(row: pd.Series) -> str:
    """Stage = spatial grid of the layer's output (H x W), which is what a ResNet stage is."""
    if "H" in row and not pd.isna(row["H"]):
        return f"{int(row['H'])}x{int(row['W'])}"
    head = str(row["layer"]).split(".")[0]
    return head if head.startswith("layer") else "stem"


# --------------------------------------------------------------------------- loading

def gated(df: pd.DataFrame, family: str) -> pd.DataFrame:
    d = df.copy()
    if "kind" in d.columns:
        d = d[d["kind"] == "conv"]
    if "is_depthwise" in d.columns:
        d = d[d["is_depthwise"] == False]  # noqa: E712
    if "n_over_C_ok" in d.columns:
        d = d[d["n_over_C_ok"] == True]  # noqa: E712
    elif "n_over_C" in d.columns:
        d = d[d["n_over_C"] >= 50]
    d = d[d[TAU].notna() & d["r_max"].notna()]
    d["v"] = d[TAU] / d["r_max"]
    d["stage"] = d.apply(stage_of, axis=1)
    d["role"] = d["layer"].map(lambda l: role_of(family, l))
    return d.set_index("layer")[["v", "stage", "role", "C"]]


def epoch_files(run_dir: str) -> list[str]:
    fs = sorted(glob.glob(os.path.join(run_dir, "epoch_*_layers.csv")),
                key=lambda f: int(re.search(r"epoch_(\d+)_", f).group(1)))
    return fs


# --------------------------------------------------------------------------- statistics

def pooled_rank_rho(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, int]:
    """Spearman between profiles after ranking each within its (stage, role) cell; cells with < 2 layers
    are dropped.  Invariant to any per-cell monotone transform (so to the denominator and to level shifts)."""
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    keys = list(zip(a["stage"], a["role"]))
    ra, rb = np.full(len(common), np.nan), np.full(len(common), np.nan)
    for key in set(keys):
        idx = np.array([k == key for k in keys])
        if idx.sum() < 2 or key[1] == "other":
            continue
        ra[idx] = rankdata(a["v"].values[idx]) / idx.sum()
        rb[idx] = rankdata(b["v"].values[idx]) / idx.sum()
    keep = ~np.isnan(ra)
    if keep.sum() < MIN_N:
        return float("nan"), int(keep.sum())
    return float(spearmanr(ra[keep], rb[keep]).correlation), int(keep.sum())


def per_cell(a: pd.DataFrame, b: pd.DataFrame) -> list[dict]:
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    out = []
    for (st, role), g in b.groupby(["stage", "role"], sort=False):
        if role == "other" or len(g) < MIN_N:
            continue
        rho = float(spearmanr(a.loc[g.index, "v"], g["v"]).correlation)
        out.append({"stage": st, "role": role, "n": len(g), "rho": rho, "null_se": 1 / np.sqrt(len(g) - 1)})
    for role, g in b.groupby("role", sort=False):
        if role == "other" or len(g) < MIN_N:
            continue
        rho = float(spearmanr(a.loc[g.index, "v"], g["v"]).correlation)
        out.append({"stage": "all", "role": role, "n": len(g), "rho": rho, "null_se": 1 / np.sqrt(len(g) - 1)})
    return out


def compare(label: str, group: str, family: str, init: pd.DataFrame, final: pd.DataFrame, rows: list, reference: str):
    common = init.index.intersection(final.index)
    allrho = float(spearmanr(init.loc[common, "v"], final.loc[common, "v"]).correlation) if len(common) >= MIN_N else float("nan")
    pooled, npool = pooled_rank_rho(init, final)
    rows.append({"group": group, "run": label, "family": family, "reference": reference, "stage": "ALL", "role": "all",
                 "n": len(common), "rho": allrho, "null_se": 1 / np.sqrt(max(len(common) - 1, 1))})
    rows.append({"group": group, "run": label, "family": family, "reference": reference, "stage": "pooled", "role": "within",
                 "n": npool, "rho": pooled, "null_se": 1 / np.sqrt(max(npool - 1, 1))})
    for c in per_cell(init, final):
        rows.append({"group": group, "run": label, "family": family, "reference": reference, **c})


# --------------------------------------------------------------------------- drivers

CONTROLLED = [  # (group, glob of run dirs)
    ("A17 ImageNet-1k", "a17/*_s[0-9]"),
    ("A29b ImageNet-1k depth", "a29b/*_s[0-9]"),
    ("A28 ImageNet factors", "a28/k*_s[0-9]"),
    ("A31 ImageNet class count", "a31/k*_s[0-9]"),
    ("round 3 CIFAR-10", "round3/a2_*_s[0-9]"),
    ("round 3 CIFAR-10 VGG", "round3/a5_vgg16_bn_none_s[0-9]"),
]

PUBLIC = {  # family -> (trained glob(s), default-draw glob, gamma1 glob, gamma0 glob)
    "resnet50": (["timm_resnet50.*_layers.csv", "resnet50_layers.csv"], "local6400/seed[1235]/resnet50_random_layers.csv",
                 None, "init_axis/gamma0/*/*resnet50*random_layers.csv"),
    "resnet18": (["local6400/archs/trained/timm_resnet18.*_layers.csv"], "local6400/archs/seed[123]/timm_resnet18.*random_layers.csv",
                 "bplus_controls/gamma1/seed[123]/timm_resnet18.*random_layers.csv", None),
    "resnet34": (["local6400/archs/trained/timm_resnet34.*_layers.csv"], "local6400/archs/seed[123]/timm_resnet34.*random_layers.csv",
                 "bplus_controls/gamma1/seed[123]/timm_resnet34.*random_layers.csv", None),
    "resnet101": (["local6400/archs/trained/timm_resnet101.*_layers.csv"], "local6400/archs/seed[123]/timm_resnet101.*random_layers.csv",
                  "bplus_controls/gamma1/seed[123]/timm_resnet101.*random_layers.csv", None),
    "wide_resnet50_2": (["local6400/archs/trained/timm_wide_resnet50_2.*_layers.csv"], "local6400/archs/seed[123]/timm_wide_resnet50_2.*random_layers.csv",
                        "bplus_controls/gamma1/seed[123]/timm_wide_resnet50_2.*random_layers.csv", None),
    "resnext50_32x4d": (["local6400/archs/trained/timm_resnext50_32x4d.*_layers.csv"], "local6400/archs/seed[123]/timm_resnext50_32x4d.*random_layers.csv",
                        "bplus_controls/gamma1/seed[123]/timm_resnext50_32x4d.*random_layers.csv", None),
}


def run_controlled(results: str, rows: list) -> None:
    for group, pat in CONTROLLED:
        for run_dir in sorted(glob.glob(os.path.join(results, pat))):
            fs = epoch_files(run_dir)
            if len(fs) < 2 or not os.path.isdir(run_dir):
                continue
            rel = os.path.relpath(run_dir, results)
            fam = family_of(rel)
            init = gated(pd.read_csv(fs[0]), fam)
            final = gated(pd.read_csv(fs[-1]), fam)
            compare(rel, group, fam, init, final, rows, reference="own init (epoch 0)")


def run_public(results: str, rows: list) -> None:
    for fam, (trained_globs, default_glob, g1_glob, g0_glob) in PUBLIC.items():
        trained = sorted(f for g in trained_globs for f in glob.glob(os.path.join(results, g)) if "random" not in f and "_layers.csv" in f)
        refs = [("default draw", f) for f in sorted(glob.glob(os.path.join(results, default_glob)))]
        if g1_glob:
            refs += [("gamma=1 draw", f) for f in sorted(glob.glob(os.path.join(results, g1_glob)))]
        if g0_glob:
            refs += [("gamma=0 draw", f) for f in sorted(glob.glob(os.path.join(results, g0_glob)))]
        for tf in trained:
            final = gated(pd.read_csv(tf), fam)
            for refname, rf in refs:
                init = gated(pd.read_csv(rf), fam)
                label = os.path.basename(tf).replace("_layers.csv", "") + " vs " + os.path.relpath(rf, results).split("/")[-2]
                compare(label, "public " + fam, fam, init, final, rows, reference=refname)


def summarise(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    for (group, fam), g in df.groupby(["group", "family"], sort=False):
        print(f"\n== {group} / {fam}")
        piv = g[g.stage.isin(["ALL", "pooled"]) | (g.stage == "all")].pivot_table(
            index=["run", "reference"], columns=["stage", "role"], values="rho", aggfunc="first")
        print(piv.round(2).to_string())
        cells = g[~g.stage.isin(["ALL", "pooled", "all"])]
        if len(cells):
            piv2 = cells.pivot_table(index=["run", "reference"], columns=["stage", "role"], values="rho", aggfunc="first")
            print(piv2.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: list = []
    run_controlled(a.results, rows)
    run_public(a.results, rows)
    df = pd.DataFrame(rows)
    out = a.out or os.path.join(a.results, "order_within_stage")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "table.csv"), index=False)
    summarise(df)
    print(f"\nwrote {os.path.join(out, 'table.csv')}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
