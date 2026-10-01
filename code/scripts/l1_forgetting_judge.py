"""L1 map, item 1 (plan 9.27 C; P9.85 / P9.86): does the initialisation profile forget the input spectrum with depth?
For each family and random seed, the random network's k*/r_max profile on CIFAR-10 images (results/apriori/cifar10/seed<s>) is
compared with the same weights' profile on ImageNet images (results/local6400/archs/seed<s>; ResNet-50: results/local6400/seed<s>):
    rel|D|_l = |k_cifar - k_imagenet| / k_imagenet     (dense conv outputs passing n/C >= 50 on both image sets)
per layer, at tau = 0.95 and 0.999; medians by depth third and Spearman(depth, rel|D|).

    python scripts/l1_forgetting_judge.py --results ../results
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAMILIES = {"resnet50": "resnet50_random", "resnet18": "timm_resnet18.tv_in1k_random", "vgg16_bn": "timm_vgg16_bn.tv_in1k_random",
            "densenet121": "timm_densenet121.tv_in1k_random", "mobilenetv2": "timm_mobilenetv2_100.ra_in1k_random",
            "efficientnet_b0": "timm_efficientnet_b0.ra_in1k_random"}
FORGET, KEEP = ("vgg16_bn", "densenet121", "mobilenetv2"), ("resnet18", "resnet50")
SEEDS = (1, 2, 3)
TAUS = ("0.95", "0.999")


def _profile(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[(d.kind == "conv") & (d.n_over_C >= 50)]
    if "is_depthwise" in d:
        d = d[~d.is_depthwise.astype(bool)]
    out = d[["layer", "depth_index"]].copy()
    for t in TAUS:
        out[f"k{t}"] = d[f"k_star_{t}"] / d["r_max"]
    return out


def _paths(results: str, family: str, seed: int) -> tuple[str, str]:
    name = FAMILIES[family]
    cifar = os.path.join(results, "apriori", "cifar10", f"seed{seed}", f"{name}_layers.csv")
    imnet = os.path.join(results, "local6400", f"seed{seed}", f"{name}_layers.csv") if family == "resnet50" \
        else os.path.join(results, "local6400", "archs", f"seed{seed}", f"{name}_layers.csv")
    return cifar, imnet


def stats(results: str, family: str, seed: int) -> dict | None:
    cifar, imnet = _paths(results, family, seed)
    if not (os.path.exists(cifar) and os.path.exists(imnet)):
        return None
    c, i = _profile(cifar), _profile(imnet)
    m = c.merge(i, on="layer", suffixes=("_c", "_i")).sort_values("depth_index_c")
    out = dict(family=family, seed=seed, L=len(m))
    depth = np.arange(len(m))
    thirds = np.minimum(depth * 3 // max(len(m), 1), 2)
    for t in TAUS:
        rel = np.abs(m[f"k{t}_c"] - m[f"k{t}_i"]) / m[f"k{t}_i"]
        out[f"rel{t}"] = [float(np.median(rel[thirds == j])) for j in range(3)]
        out[f"rho{t}"] = float(spearmanr(depth, rel).correlation)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    print("==== L1 item 1 (plan 9.27 C): does the initialisation profile forget the input spectrum?  relative |difference| CIFAR-10 vs ImageNet images ====")
    print(f"  {'family':16s} seed  L   tau=0.95 first/mid/last   rho     tau=0.999 first/mid/last   rho")
    S = {}
    for fam in FAMILIES:
        for s in SEEDS:
            r = stats(a.results, fam, s)
            if r is None:
                continue
            S[(fam, s)] = r
            print(f"  {fam:16s} {s}   {r['L']:3d}   {' / '.join(f'{x:.2f}' for x in r['rel0.95'])}     {r['rho0.95']:+.2f}    "
                  f"{' / '.join(f'{x:.2f}' for x in r['rel0.999'])}     {r['rho0.999']:+.2f}")
    V = {}
    if all((f, s) in S for f in FORGET for s in SEEDS):
        V["P9.85"] = all(S[(f, s)]["rho0.999"] <= -0.5 and S[(f, s)]["rel0.999"][2] <= 0.05 for f in FORGET for s in SEEDS)
        print(f"  P9.85 (VGG-16_BN, DenseNet-121, MobileNetV2: tau=0.999 rho <= -0.5 and last-third median <= 0.05, 9/9) -> {'holds' if V['P9.85'] else 'fails'}")
    if all((f, s) in S for f in KEEP for s in SEEDS):
        V["P9.86"] = all(S[(f, s)]["rel0.95"][2] >= 0.3 and S[(f, s)]["rho0.999"] >= -0.5 for f in KEEP for s in SEEDS)
        print(f"  P9.86 (ResNet-18, ResNet-50: tau=0.95 last-third median >= 0.3 and tau=0.999 rho >= -0.5, 6/6) -> {'holds' if V['P9.86'] else 'fails'}")
    if len(V) == 2:
        print("  Registered consequence: " + ("links 1' and 5 get the three-seed sentence and a claims-table row" if all(V.values())
                                             else "stays exploratory (single seed); the failing families are reported"))
    elif not V:
        print("  seeds incomplete -- not judged yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
