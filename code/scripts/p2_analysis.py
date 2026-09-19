#!/usr/bin/env python
"""P2 / P2b / P3 (notes/controller-criterion-2026-09-16.md §8): one training run with the width controller
(layerspec.realloc inside trajectory_cifar.py) against A18's retrained arms (checkpoint_analysis.txt §16).

Runs live in results/p2/<run>/ (trajectory_layers.csv with epoch/test_acc, realloc.json with the actions).
Arms are parsed from results/checkpoint_analysis.txt §16 so the numbers stay in one place.

    python scripts/p2_analysis.py --results ../results
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# run-name prefix -> (dataset, budget, arm to beat by >= -0.3, arm that must be strictly below)
GROUPS = {
    "c10_b673":    ("cifar10",  0.673, "ruler_act", "uniform"),   # P2, post-activation ruler budget
    "c10_b345":    ("cifar10",  0.345, "ruler",     "uniform"),   # P2, conv-output ruler budget
    "c10_fixed339": ("cifar10", 0.339, "ruler",     "uniform"),   # P2b: start from the uniform arm, conserve its budget
    "c100_b673":   ("cifar100", 0.673, None,        None),        # P3-type stress test: CIFAR-100 has no tau=0.999 slack
    "c10_b673_nogrow": ("cifar10", 0.673, "ruler_act", "uniform"),  # P2c ablation: shrink-only (--realloc-grow-cap 0)
    "c10_narrow341": ("cifar10", 0.341, "ruler",     "uniform"),    # P2d: wrong start (front narrow, back wide), conserved
    "c10_b673_cap50":  ("cifar10", 0.673, "ruler_act", "uniform"),  # P2e: grow-cap 0.5
    "c10_b673_cap100": ("cifar10", 0.673, "ruler_act", "uniform"),  # P2e: grow-cap 1.0
}


def parse_arms(path):
    """{dataset: {arm: (params, [acc per seed], mean)}} from §16 of checkpoint_analysis.txt."""
    arms, ds = {}, None
    for line in open(path):
        m = re.match(r"\s+(cifar10|cifar100) \(plan", line)
        if m:
            ds = m.group(1); arms[ds] = {}; continue
        m = re.match(r"\s+(full|ruler|uniform|ruler95|ruler_act)\s+(\d+)\s+([\d,]+)\s+(.*)$", line)
        if re.match(r"\s*\d+\. ", line) and ds:           # next numbered section: stop
            break
        if m and ds:
            try:
                nums = [float(x) for x in m.group(4).split()]  # acc per seed ..., mean, final level
            except ValueError:
                continue
            arms[ds][m.group(1)] = (int(m.group(3).replace(",", "")), nums[:-2], nums[-2])
    return arms


def read_run(d):
    tl = os.path.join(d, "trajectory_layers.csv")
    if not os.path.exists(tl):
        return None
    t = pd.read_csv(tl)
    last = int(t.epoch.max())
    acc = float(t.loc[t.epoch == last, "test_acc"].iloc[0])
    rl = os.path.join(d, "realloc.json")
    actions = json.load(open(rl)) if os.path.exists(rl) else []
    return dict(epoch=last, acc=acc, done=os.path.exists(os.path.join(d, "epoch_100_layers.csv")), actions=actions,
                widths=actions[-1]["after"] if actions else None,
                frac=actions[-1]["actual_fraction"] if actions else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    a = ap.parse_args()
    arms = parse_arms(os.path.join(a.results, "checkpoint_analysis.txt"))
    ruler_widths = {}
    for ds, fn in (("cifar10", "widths_act.json"), ):
        p = os.path.join(a.results, "a18", fn)
        if os.path.exists(p):
            ruler_widths[ds] = json.load(open(p))["widths"]["ruler"]
    for prefix, (ds, budget, beat, above) in GROUPS.items():
        runs = {}
        for d in sorted(glob.glob(os.path.join(a.results, "p2", prefix + "_s*"))):
            r = read_run(d)
            if r:
                runs[os.path.basename(d)] = r
        if not runs:
            continue
        print(f"\n== {prefix}  ({ds}, budget {budget:.3f} of full)")
        for name, r in runs.items():
            st = "done" if r["done"] else f"epoch {r['epoch']}"
            ev = "; ".join(f"ep{x['epoch']}: {x['params_before']/1e6:.2f}M->{x['params_after']/1e6:.2f}M"
                           + ("" if x["shrink_only"] else " grow+shrink") for x in r["actions"])
            print(f"  {name:16s} {st:9s} acc {r['acc']:6.2f}  frac {r['frac'] if r['frac'] is not None else float('nan'):.3f}  [{ev}]")
            if r["widths"]:
                line = f"      widths {r['widths']}"
                if ds in ruler_widths and beat == "ruler_act":
                    rho = pd.Series(r["widths"]).corr(pd.Series(ruler_widths[ds]), method="spearman")
                    line += f"  (Spearman vs ruler_act widths {rho:+.2f})"
                print(line)
        done = {n: r["acc"] for n, r in runs.items() if r["done"]}
        if not done or ds not in arms:
            continue
        accs = list(done.values()); mean = float(np.mean(accs))
        A = arms[ds]
        print(f"  controller: n={len(accs)} mean {mean:.2f} (min {min(accs):.2f}, spread {max(accs)-min(accs):.2f})")
        for arm, (params, per_seed, m) in A.items():
            print(f"    vs {arm:9s} {params/1e6:6.2f}M  {' '.join(f'{x:5.2f}' for x in per_seed)}  mean {m:.2f}  diff {mean - m:+.2f}")
        if beat:
            ok1 = mean >= A[beat][2] - 0.3
            ok2 = mean > A[above][2]
            allabove = min(accs) > max(A[above][1])
            print(f"  P2 ({prefix}): controller >= {beat} - 0.3 -> {'yes' if ok1 else 'NO'} ({mean:.2f} vs {A[beat][2]-0.3:.2f}); "
                  f"> {above} -> {'yes' if ok2 else 'NO'}; every seed above every {above} seed -> {'yes' if allabove else 'no'}"
                  f"{'' if len(accs) >= 3 else '  [fewer than 3 seeds: provisional]'}")
        else:
            full = A["full"][2]
            print(f"  P3-type ({prefix}): controller vs full {mean - full:+.2f} (arms have no slack at tau=0.999; "
                  f"ruler_act {A['ruler_act'][2]:.2f}, uniform {A['uniform'][2]:.2f})")


if __name__ == "__main__":
    main()
