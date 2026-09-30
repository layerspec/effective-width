#!/usr/bin/env python
"""Verdicts of A34 (analysis-plan 9.24), written before its data.  nr = numerical rank at 1e-3 of the largest
eigenvalue (Masarczyk et al. 2023), k999 = k*(0.999); both over C, post-ReLU, VGG-16_BN; "late" = the last three
convolutions.

  P9.72  end state, CIFAR-10 full network, every seed: each late layer has nr/C <= 0.05 and k999/C >= 0.15
  P9.73  class count: at the last convolution the seed-median of nr/C does not decrease along K = 13, 20, 50, 100
         (and that of k999/C does not either)
  P9.74  first steps: (a) their recipe, every seed: nr of each late layer at step 75 is at most half of its value at
         step 0; (b) our recipe: the same by step 390; (c) both recipes, every seed: k999/C of each late layer at
         step 1950 is still above the end-state value of that layer (seed-median, CIFAR-10 full) by 0.2
  P9.75  the two arms of A25 (640 images per class; 5,000 against 39,000 steps): seed-median nr/C of each late layer
         differs by at most 0.02 between the arms, while k999/C differs by at least 0.15 in two of the three

    python scripts/two_rulers_judge.py --results ../results
"""
import argparse
import os

import numpy as np
import pandas as pd

LATE = (10, 11, 12)
KS = (("c100_k13", 13), ("c100_k20", 20), ("c100_k50", 50), ("c100_k100", 100))


def _ratio(d, col):
    return (d[col] / d.C).to_numpy()


def end_state(path: str) -> dict:
    V = {}
    if not os.path.exists(path):
        print("  end_state.csv not found"); return V
    d = pd.read_csv(path)
    d = d[d.ok.astype(bool)]
    d["nr_C"], d["k999_C"] = d.nr / d.C, d.k999 / d.C
    print("==== A34, end state: seed-median over C, the 13 convolutions (post-ReLU) ====")
    for arm, g in d.groupby("arm", sort=False):
        m = g.groupby("index")[["nr_C", "k999_C"]].median()
        print(f"  {arm:15s} seeds {g.seed.nunique()}  nr/C   " + " ".join(f"{x:.3f}" for x in m.nr_C))
        print(f"  {'':15s}          k999/C " + " ".join(f"{x:.3f}" for x in m.k999_C))
    full = d[(d.arm == "c10_full") & d["index"].isin(LATE)]
    if full.seed.nunique() >= 3 and len(full) == 3 * full.seed.nunique():
        V["P9.72"] = bool((full.nr_C <= 0.05).all() and (full.k999_C >= 0.15).all())
        print(f"  P9.72 (CIFAR-10 full, late layers, every seed: nr/C <= 0.05 and k999/C >= 0.15) -> "
              f"{'holds' if V['P9.72'] else 'fails'}  nr/C {full.nr_C.min():.3f}..{full.nr_C.max():.3f}, "
              f"k999/C {full.k999_C.min():.3f}..{full.k999_C.max():.3f}")
    last = d[d["index"] == 12]
    med = {k: last[last.arm == arm][["nr_C", "k999_C"]].median() for arm, k in KS if (last.arm == arm).sum() >= 3}
    if len(med) == len(KS):
        nr = [med[k].nr_C for _, k in KS]; kk = [med[k].k999_C for _, k in KS]
        V["P9.73"] = bool(all(b >= a for a, b in zip(nr, nr[1:])) and all(b >= a for a, b in zip(kk, kk[1:])))
        print(f"  P9.73 (last convolution, K = 13/20/50/100, both rulers non-decreasing) -> {'holds' if V['P9.73'] else 'fails'}  "
              f"nr/C {' '.join(f'{x:.3f}' for x in nr)};  k999/C {' '.join(f'{x:.3f}' for x in kk)}")
    a, b = d[d.arm == "c10_pc640"], d[d.arm == "c10_pc640_e780"]
    if a.seed.nunique() >= 3 and b.seed.nunique() >= 3:
        ma = a[a["index"].isin(LATE)].groupby("index")[["nr_C", "k999_C"]].median()
        mb = b[b["index"].isin(LATE)].groupby("index")[["nr_C", "k999_C"]].median()
        dn, dk = (ma.nr_C - mb.nr_C).abs(), (ma.k999_C - mb.k999_C).abs()
        V["P9.75"] = bool((dn <= 0.02).all() and (dk >= 0.15).sum() >= 2)
        print(f"  P9.75 (A25 arms, late layers: nr/C within 0.02, k999/C apart by 0.15 in two of three) -> "
              f"{'holds' if V['P9.75'] else 'fails'}  |d nr/C| {' '.join(f'{x:.3f}' for x in dn)};  |d k999/C| {' '.join(f'{x:.3f}' for x in dk)}")
    return V


def early(path: str, end_path: str) -> dict:
    V = {}
    if not os.path.exists(path):
        print("  early.csv not found"); return V
    d = pd.read_csv(path)
    d = d[d["index"].isin(LATE)].copy()
    d["nr_C"], d["k999_C"] = d.nr / d.C, d.k999 / d.C
    print("\n==== A34, first steps of training: late layers, seed-median over C ====")
    for recipe, g in d.groupby("recipe", sort=False):
        for step, h in g.groupby("step"):
            m = h.groupby("index")[["nr_C", "k999_C"]].median()
            print(f"  {recipe:7s} step {int(step):5d}  lr {h.lr.median():.4f}  nr/C {' '.join(f'{x:.3f}' for x in m.nr_C)}   "
                  f"k999/C {' '.join(f'{x:.3f}' for x in m.k999_C)}")

    def collapsed(recipe, step):
        g = d[d.recipe == recipe]
        if g.seed.nunique() < 3 or step not in set(g.step):
            return None
        a = g[g.step == 0].set_index(["seed", "index"]).nr
        b = g[g.step == step].set_index(["seed", "index"]).nr
        return bool((b.reindex(a.index) <= 0.5 * a).all())
    for key, recipe, step in (("P9.74a", "theirs", 75), ("P9.74b", "ours", 390)):
        v = collapsed(recipe, step)
        if v is not None:
            V[key] = v
            print(f"  {key} ({recipe} recipe: nr of every late layer at step {step} at most half of step 0, every seed) -> {'holds' if v else 'fails'}")
    if os.path.exists(end_path) and 1950 in set(d.step) and all(d[d.recipe == r].seed.nunique() >= 3 for r in ("ours", "theirs")):
        e = pd.read_csv(end_path)
        e = e[(e.arm == "c10_full") & e["index"].isin(LATE)]
        if e.seed.nunique() >= 3:
            final = (e.k999 / e.C).groupby(e["index"]).median()
            g = d[d.step == 1950]
            V["P9.74c"] = bool((g.k999_C.to_numpy() >= final.reindex(g["index"]).to_numpy() + 0.2).all())
            print(f"  P9.74c (k999/C of every late layer at step 1950 above its end-state value by 0.2, both recipes, every seed) -> "
                  f"{'holds' if V['P9.74c'] else 'fails'}  end state {' '.join(f'{x:.3f}' for x in final)}; "
                  f"step 1950 {g.k999_C.min():.3f}..{g.k999_C.max():.3f}")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    a = p.parse_args(argv)
    root = os.path.join(a.results, "two_rulers")
    V = end_state(os.path.join(root, "end_state.csv"))
    V.update(early(os.path.join(root, "early.csv"), os.path.join(root, "end_state.csv")))
    if all(k in V for k in ("P9.72", "P9.74c")) and ("P9.74a" in V or "P9.74b" in V):
        if V["P9.72"] and V["P9.74c"] and (V.get("P9.74a") or V.get("P9.74b")):
            print("  Registered consequence: two readings of one spectrum. The numerical rank falls in the first steps; the width at "
                  "tau = 0.999 keeps contracting long after.")
        elif not V["P9.72"]:
            print("  Registered consequence: the collapse of the numerical rank is not reproduced on this object; the sentence on "
                  "optimisation length is written with that stated.")
        else:
            print("  Registered consequence: none fits; the numbers are reported as measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
