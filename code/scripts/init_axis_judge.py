#!/usr/bin/env python
"""Verdicts of A33 (analysis-plan 9.23), written before its data: the initialisation of the reference network as an
axis of the measurement.  Profiles are k*(0.95)/r_max over the gate-passing dense convolutions (ResNet families) or
linear layers of the twelve blocks (transformers).

  P9.68  ResNet-50: median T-R of the six public weights against the gamma = 0 networks is at least +0.12
         (0.20 above the -0.08 measured against gamma = 1)
  P9.69  the control sets the number, not the match: in at least 5 of the 6 ResNet families the shift
         T-R(gamma = 0) - T-R(gamma = 1) has one sign for the torchvision-recipe weights (trained from gamma = 1)
         and the timm-recipe weights (trained from gamma = 0)
  P9.70  (a) the LayerScale networks at 1e-4 repeat those at 1e-6 measured in A30 (deit3_small: rho >= 0.98 between
         the two groups, median level within 0.01); (b) the DeiT-III weight of each family reads an all-layer T-R
         against its LayerScale networks at least 0.15 below that against the default networks
  P9.71  (a) the JAX-initialised networks read a lower level than the default ones by at least 0.03, both families;
         (b) P9.51 under faithful references: every weight against the networks of its own initialisation, median
         T-R >= 0.5 over all layers and >= 0.8 within qkv, attention projection and fc1, both families

    python scripts/init_axis_judge.py --results ../results
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bplus_controls_judge import aligned, profile, rho   # noqa: E402
from scripts.checkpoint_analysis import A30_FAMILIES, A30_ROLES, ARCH_FAMILIES, _a30_profile, _vit_role   # noqa: E402

RESNET_FAMILIES = ("resnet18", "resnet34", "resnet50", "resnet101", "wide_resnet50_2", "resnext50")
GAMMA1_NAME = {"resnet18": "resnet18.tv_in1k", "resnet34": "resnet34.tv_in1k", "resnet101": "resnet101.tv_in1k",
               "wide_resnet50_2": "wide_resnet50_2.tv_in1k", "resnext50": "resnext50_32x4d.tv_in1k"}
# which initialisation a recipe trained from (plan 9.23, provenance table); weights of other recipes are left out of P9.69
TORCHVISION_RECIPES = ("tv_in1k", "tv2_in1k")             # torchvision reference script: zero_init_residual = False
TIMM_RECIPES = ("a1_in1k", "a2_in1k", "a3_in1k", "a1h_in1k", "ra_in1k", "racm_in1k")   # timm train.py: zero_init_last = True
FAITHFUL_VIT = {"augreg_in1k": "jax", "augreg_in21k_ft_in1k": "jax", "orig_in21k_ft_in1k": "jax", "sam_in1k": "jax",
                "fb_in1k": "default", "fb_in22k_ft_in1k": "layerscale"}


def recipe_group(path: str):
    name = os.path.basename(path).replace("_layers.csv", "")
    if name == "resnet50":                                # the torchvision V1 weight of the first round
        return "torchvision"
    tag = name.split(".", 1)[1] if "." in name else ""
    return "torchvision" if tag in TORCHVISION_RECIPES else "timm" if tag in TIMM_RECIPES else None


def resnet_controls(results: str, fam: str):
    """(files of the trained weights, gamma = 0 networks, gamma = 1 networks) of one ResNet family."""
    t_glob, r_glob = ARCH_FAMILIES[fam]
    trained = sorted(glob.glob(os.path.join(results, t_glob)))
    if fam == "resnet50":                                 # the public random networks of ResNet-50 are torchvision's
        g1 = sorted(glob.glob(os.path.join(results, r_glob)))
        g0 = sorted(glob.glob(os.path.join(results, "init_axis", "gamma0", "seed*", "timm_resnet50.tv_in1k_random_layers.csv")))
    else:
        g0 = sorted(glob.glob(os.path.join(results, r_glob)))
        g1 = sorted(glob.glob(os.path.join(results, "bplus_controls", "gamma1", "seed*", f"timm_{GAMMA1_NAME[fam]}_random_layers.csv")))
    return trained, g0, g1


def med_tr(T, R) -> float:
    return float(np.median([rho(t, r) for t in T for r in R]))


def resnets(results: str) -> dict:
    print("==== P9.68 / P9.69: the six ResNet families against both references (plan 9.23) ====")
    print(f"  {'family':16s} {'T':>2s} {'g0':>2s} {'g1':>2s}  {'T-R g=0':>8s} {'T-R g=1':>8s} {'shift':>6s}   "
          f"{'torchvision recipes (n): g0, g1, shift':>40s}   {'timm recipes (n): g0, g1, shift':>34s}")
    rows, V = [], {}
    for fam in RESNET_FAMILIES:
        files, g0, g1 = resnet_controls(results, fam)
        if not files or len(g0) < 3 or len(g1) < 2:
            print(f"  {fam:16s} {len(files):2d} {len(g0):2d} {len(g1):2d}  not complete"); continue
        A = aligned([profile(f) for f in files + g0 + g1])
        T, R0, R1 = A[:len(files)], A[len(files):len(files) + len(g0)], A[len(files) + len(g0):]
        row = dict(fam=fam, tr0=med_tr(T, R0), tr1=med_tr(T, R1))
        for grp in ("torchvision", "timm"):
            sel = [t for t, f in zip(T, files) if recipe_group(f) == grp]
            row[grp] = (len(sel), med_tr(sel, R0), med_tr(sel, R1)) if sel else None
        rows.append(row)
        cell = lambda g: "      --" if g is None else f"({g[0]}) {g[1]:+.2f} {g[2]:+.2f} {g[1] - g[2]:+.2f}"   # noqa: E731
        print(f"  {fam:16s} {len(files):2d} {len(g0):2d} {len(g1):2d}  {row['tr0']:+8.2f} {row['tr1']:+8.2f} {row['tr0'] - row['tr1']:+6.2f}   "
              f"{cell(row['torchvision']):>40s}   {cell(row['timm']):>34s}")
    r50 = next((r for r in rows if r["fam"] == "resnet50"), None)
    if r50 is not None:
        V["P9.68"] = bool(r50["tr0"] >= 0.12)
        print(f"  P9.68 (ResNet-50 against gamma = 0: median T-R >= +0.12) -> {'holds' if V['P9.68'] else 'fails'} ({r50['tr0']:+.2f}; against gamma = 1 {r50['tr1']:+.2f})")
    both = [r for r in rows if r["torchvision"] and r["timm"]]
    if len(rows) == len(RESNET_FAMILIES) and len(both) == len(RESNET_FAMILIES):
        same = [np.sign(r["torchvision"][1] - r["torchvision"][2]) == np.sign(r["timm"][1] - r["timm"][2]) for r in both]
        gap = [abs((r["torchvision"][1] - r["torchvision"][2]) - (r["timm"][1] - r["timm"][2])) for r in both]
        V["P9.69"] = bool(sum(same) >= 5)
        print(f"  P9.69 (the shift has one sign for both recipe groups in >= 5 of 6 families) -> {'holds' if V['P9.69'] else 'fails'} "
              f"({sum(same)}/6; |difference of the two shifts| median {np.median(gap):.2f}, largest {max(gap):.2f})")
        print("  Registered consequence: " + ("T-R belongs to the pair (trained profile, reference); the faithful reference has no special "
              "agreement with its weights, and both references are reported." if V["P9.69"] else
              "the faithful reference matters: T-R against the initialisation a weight was trained from is the number reported first."))
    return V


def _vit_profiles(results: str, fam: str):
    weights, rand_glob = A30_FAMILIES[fam]
    root = os.path.join(results, "a30")
    arch = rand_glob.replace("_random_s*", "")
    deit3 = arch.replace("vit_", "deit3_")
    T = {w: _a30_profile(f) for w in weights if os.path.exists(f := os.path.join(root, f"{w}_layers.csv"))}
    R = {"default": [_a30_profile(f) for f in sorted(glob.glob(os.path.join(root, rand_glob + "_layers.csv")))],
         "jax": [_a30_profile(f) for f in sorted(glob.glob(os.path.join(results, "init_axis", "jax", f"{arch}_random_s*_layers.csv")))],
         "layerscale": [_a30_profile(f) for f in sorted(glob.glob(os.path.join(results, "init_axis", "layerscale", f"{deit3}_random_s*_layers.csv")))],
         "layerscale_1e-6": [_a30_profile(f) for f in sorted(glob.glob(os.path.join(root, f"{deit3}_random_s*_layers.csv")))]}
    return T, R


def transformers(results: str) -> dict:
    print("\n==== P9.70 / P9.71: the two transformer families against three references (plan 9.23) ====")
    V, fam_rows = {}, {}
    for fam in A30_FAMILIES:
        T, R = _vit_profiles(results, fam)
        if len(T) < len(A30_FAMILIES[fam][0]) or any(len(R[k]) < 3 for k in ("default", "jax", "layerscale")):
            print(f"  {fam}: {len(T)} weights; references " + ", ".join(f"{k} {len(v)}" for k, v in R.items()) + " -- not complete"); continue
        every = list(T.values()) + [p for v in R.values() for p in v]
        layers = sorted(set.intersection(*[set(p) for p in every]), key=lambda lay: (int(lay.split(".")[1]), lay))
        vec = lambda p, sel=layers: np.array([p[lay] for lay in sel])                                  # noqa: E731
        level = {k: float(np.median([np.median(vec(p)) for p in v])) for k, v in R.items() if v}
        rr = {k: float(np.median([rho(vec(a), vec(b)) for a, b in itertools.combinations(v, 2)])) for k, v in R.items() if len(v) > 1}
        print(f"  {fam}: {len(layers)} layers; level of the references " + ", ".join(f"{k} {v:.3f}" for k, v in level.items())
              + "; R-R " + ", ".join(f"{k} {v:+.2f}" for k, v in rr.items()))
        print("      between references (median rho): " + ", ".join(
            f"{a} x {b} {np.median([rho(vec(x), vec(y)) for x in R[a] for y in R[b]]):+.2f}"
            for a, b in itertools.combinations([k for k in R if R[k]], 2)))
        print(f"      {'weight':42s} {'trained from':>12s}  " + "  ".join(f"{k:>10s}" for k in ("default", "jax", "layerscale")))
        faithful_pairs = {g: [] for g in ("all",) + A30_ROLES}
        row = {}
        for w, p in T.items():
            own = FAITHFUL_VIT[w.split(".", 1)[1]]
            if w.startswith("deit3") and own == "default":
                own = "layerscale"                                  # deit3 ... fb_in1k: DeiT-III, LayerScale
            tr = {k: float(np.median([rho(vec(p), vec(r)) for r in R[k]])) for k in ("default", "jax", "layerscale")}
            row[w] = (own, tr)
            print(f"      {w:42s} {own:>12s}  " + "  ".join(f"{tr[k]:+10.2f}" + ("*" if k == own else " ") for k in ("default", "jax", "layerscale")))
            for g in faithful_pairs:
                sel = layers if g == "all" else [lay for lay in layers if _vit_role(lay) == g]
                faithful_pairs[g] += [rho(vec(p, sel), vec(r, sel)) for r in R[own]]
        fmed = {g: float(np.median(v)) for g, v in faithful_pairs.items()}
        print("      faithful references (*): median T-R " + ", ".join(f"{g} {v:+.2f}" for g, v in fmed.items()))
        d3 = [(tr["layerscale"], tr["default"]) for w, (own, tr) in row.items() if w.startswith("deit3")]
        same = None
        if R["layerscale_1e-6"]:
            a, b = R["layerscale"], R["layerscale_1e-6"]
            same = (float(np.median([rho(vec(x), vec(y)) for x in a for y in b])), abs(level["layerscale"] - level["layerscale_1e-6"]))
        fam_rows[fam] = dict(level=level, fmed=fmed, d3=d3, same=same)
    if len(fam_rows) == len(A30_FAMILIES):
        sm = [f["same"] for f in fam_rows.values() if f["same"] is not None]
        a_ok = bool(sm) and all(r >= 0.98 and dl <= 0.01 for r, dl in sm)
        b_ok = all(all(ls <= df - 0.15 for ls, df in f["d3"]) and f["d3"] for f in fam_rows.values())
        V["P9.70a"], V["P9.70b"] = bool(a_ok), bool(b_ok)
        print(f"  P9.70 (a) (LayerScale 1e-4 repeats 1e-6: rho >= 0.98, level within 0.01) -> {'holds' if a_ok else 'fails'} "
              + "; ".join(f"rho {r:+.3f}, level difference {dl:.3f}" for r, dl in sm))
        print(f"  P9.70 (b) (every DeiT-III weight: T-R against LayerScale at least 0.15 below that against default) -> {'holds' if b_ok else 'fails'} "
              + "; ".join(f"{ls:+.2f} vs {df:+.2f}" for f in fam_rows.values() for ls, df in f["d3"]))
        V["P9.71a"] = bool(all(f["level"]["jax"] <= f["level"]["default"] - 0.03 for f in fam_rows.values()))
        V["P9.71b"] = bool(all(f["fmed"]["all"] >= 0.5 and all(f["fmed"][r] >= 0.8 for r in ("qkv", "attention proj.", "MLP fc1"))
                          for f in fam_rows.values()))
        print(f"  P9.71 (a) (JAX networks read a level at least 0.03 below the default ones, both families) -> {'holds' if V['P9.71a'] else 'fails'} "
              + "; ".join(f"{f['level']['jax']:.3f} vs {f['level']['default']:.3f}" for f in fam_rows.values()))
        print(f"  P9.71 (b) (P9.51 under faithful references, both families) -> {'holds' if V['P9.71b'] else 'fails'}")
        print("  Registered consequence: " + ("the transformer statement of P9.51 stands under the references the weights were trained from."
              if V["P9.71b"] else "'keeps the order of its initialisation' is stated per reference; the family table carries one column per "
              "initialisation and the faithful one is marked."))
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    resnets(a.results)
    transformers(a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
