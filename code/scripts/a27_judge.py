#!/usr/bin/env python
"""A27 (analysis-plan 9.16): VGG-16 (BN, ImageNet head) trained from scratch on the ten Imagenette classes with the
A26 recipe; the post-ReLU tau = 0.999 ruler of each seed is in results/a27/widths_act_s<seed>.json.

Written 2026-09-29, before any A27 run finished, so the readings below are fixed ahead of the data:
  P9.39  each of the last three conv has k*(0.999)/C < 0.7, in every seed          (their mean is printed, descriptive)
  P9.40  params(ruler) / params(full) < 0.85 in every seed, classifier included    (as registered; the 7x7x512 -> 4096
         classifier holds 3/4 of the parameters, so the conv-only fraction is printed beside it, descriptive)
  P9.41  each of the first five conv has k*(0.999)/C >= 0.9, in every seed
A layer that fails the n/C >= 50 gate is not reported and its prediction is left unjudged for that seed.

    python scripts/a27_judge.py --results ../results
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LAST, FIRST = 3, 5
N_SEEDS = 3


def _conv_params(widths, c_in=3, k=3):
    """conv weights + bias + BN weight/bias of a plain 3x3 stack (the part of VGG-16 the ruler resizes)."""
    n = 0
    for w in widths:
        n += c_in * w * k * k + w + 2 * w
        c_in = w
    return n


def read_seed(path):
    d = json.load(open(path))
    layers = d["layers"]
    ratio = np.array([r["k_star_0.999"] / r["C"] if r["ok"] else np.nan for r in layers])
    return dict(seed=int(re.search(r"_s(\d+)\.json$", path).group(1)), ratio=ratio,
                params=d["params"]["ruler"] / d["params"]["full"],
                conv_params=_conv_params(d["widths"]["ruler"]) / _conv_params(d["widths"]["full"]),
                widths=d["widths"]["ruler"])


def _top1(results, seed):
    f = os.path.join(results, "a27", f"vgg16_bn_in10_s{seed}", "result.json")
    return json.load(open(f)).get("final_top1", float("nan")) if os.path.exists(f) else float("nan")


def verdict(per_seed):
    """'holds' / 'fails' / 'unjudged' from one True/False/None per seed; a registered 3/3 needs all three seeds."""
    if any(v is None for v in per_seed) or len(per_seed) < N_SEEDS:
        return "fails" if any(v is False for v in per_seed) else "unjudged"
    return "holds" if all(per_seed) else "fails"


def judge(runs):
    def each(x, test):
        return None if np.isnan(x).any() else bool(test(x).all())
    return {"P9.39": verdict([each(r["ratio"][-LAST:], lambda x: x < 0.7) for r in runs]),
            "P9.40": verdict([r["params"] < 0.85 for r in runs]),
            "P9.41": verdict([each(r["ratio"][:FIRST], lambda x: x >= 0.9) for r in runs])}


def section(results):
    files = sorted(glob.glob(os.path.join(results, "a27", "widths_act_s*.json")))
    print("==== A27: VGG-16 (BN) on ten-class ImageNet, post-ReLU ruler at tau 0.999 (analysis-plan 9.16) ====")
    if not files:
        print("  no ruler read yet"); return None
    runs = [read_seed(f) for f in files]
    print(f"  {'seed':>4s} {'top1':>6s} {'params':>7s} {'conv only':>9s}  {'first five':>30s}  {'last three':>18s} {'mean':>5s}")
    for r in runs:
        fmt = lambda x: " ".join("  n/a" if np.isnan(v) else f"{v:5.2f}" for v in x)   # noqa: E731
        print(f"  {r['seed']:4d} {_top1(results, r['seed']):6.2f} {r['params']:7.3f} {r['conv_params']:9.3f}  "
              f"{fmt(r['ratio'][:FIRST]):>30s}  {fmt(r['ratio'][-LAST:]):>18s} {np.nanmean(r['ratio'][-LAST:]):5.2f}")
        print(f"       widths {r['widths']}")
    v = judge(runs)
    n = len(runs)
    print(f"  P9.39 (each of the last three conv < 0.7, {N_SEEDS}/{N_SEEDS} seeds): {v['P9.39']}  [{n} seed(s) read]")
    print(f"  P9.40 (ruler keeps < 0.85 of the parameters; ResNet-18 on the same task 0.95-0.97): {v['P9.40']}")
    print(f"  P9.41 (each of the first five conv >= 0.9): {v['P9.41']}")
    if "unjudged" not in (v["P9.39"], v["P9.40"]):
        if v["P9.39"] == "holds" and v["P9.40"] == "holds":
            print("  Registered consequence: a plain network's late layers collapse on a few-class task, a residual stream's do not.")
        elif v["P9.39"] == "fails":
            print("  Registered consequence: idle width stays an observation on VGG-16 / CIFAR-10; the plain-vs-residual reading is withdrawn.")
        else:
            print("  Registered consequence: none fits (P9.39 holds, P9.40 fails); report both numbers as measured.")
    return v


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    section(a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
