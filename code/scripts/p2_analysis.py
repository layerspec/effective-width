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
    "c10_narrow341_asis": ("cifar10", 0.341, None, None),           # P2f: the P2d start trained as is (no controller)
    "c10_apriori673": ("cifar10", 0.673, "ruler_act", "uniform"),   # P2h A': controller acts at epoch 0 too (a-priori start)
    "c10_p2widths":   ("cifar10", 0.673, "ruler_act", "uniform"),   # P2h A'': P2 final widths trained from scratch
    "c10_b673_randgrow": ("cifar10", 0.673, "ruler_act", "uniform"), # P2h A'': grow along random directions
    "c10_f88":        ("cifar10", 0.88,  "ruler_act", "uniform"),   # P2g: FLOPs-conserving, ruler_act's MACs
    "c10_f60":        ("cifar10", 0.60,  "ruler",     "uniform"),   # P2g: FLOPs-conserving, ruler's MACs
    "c10_ffixed342":  ("cifar10", 0.342, "ruler",     "uniform"),   # P2g: from the uniform net at conserved MACs
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
                frac=actions[-1]["actual_fraction"] if actions else None,
                macs=actions[-1].get("macs_fraction") if actions else None)


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
            mf = f"  macs {r['macs']:.3f}" if r.get("macs") is not None else ""
            print(f"  {name:16s} {st:9s} acc {r['acc']:6.2f}  frac {r['frac'] if r['frac'] is not None else float('nan'):.3f}{mf}  [{ev}]")
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
        elif prefix == "c10_narrow341_asis":
            ctrl = {os.path.basename(d): read_run(d) for d in sorted(glob.glob(os.path.join(a.results, "p2", "c10_narrow341_s*")))}
            pairs = [(ctrl[n]["acc"], done[n.replace("narrow341_", "narrow341_asis_")]) for n in ctrl
                     if ctrl[n] and ctrl[n]["done"] and n.replace("narrow341_", "narrow341_asis_") in done]
            if pairs:
                diffs = [c - x for c, x in pairs]
                print(f"  P2d vs as-is (same seed): controller {np.mean([c for c, _ in pairs]):.2f} vs as-is {mean:.2f}, "
                      f"paired diff mean {np.mean(diffs):+.2f} (min {min(diffs):+.2f}), {sum(d > 0 for d in diffs)}/{len(diffs)} positive")
        else:
            full = A["full"][2]
            print(f"  P3-type ({prefix}): controller vs full {mean - full:+.2f} (arms have no slack at tau=0.999; "
                  f"ruler_act {A['ruler_act'][2]:.2f}, uniform {A['uniform'][2]:.2f})")
    r20_section(a.results)



# ---------------------------------------------------------------------------------------------------------------------
# P2i: ResNet-20 (block-internal widths; notes/controller-criterion-2026-09-16.md, "預登記 P2i"). Baseline = results/r20;
# compute from realloc.json (macs_before/after) with the 3x forward convention of p2_flops.py; uniform arms from widths.
R20_PLANES, R20_RES, R20_N_TRAIN, R20_EPOCHS, R20_CALIB = (16, 32, 64), (32, 16, 8), 50_000, 100, 6_400
R20_FULL_INNER = [16] * 3 + [32] * 3 + [64] * 3
R20_GROUPS = {   # prefix -> (control prefix, training-cost fraction the arm is designed for)
    "r20_f47": ("r20_ufixed47", 0.55),
    "r20_apriori55": ("r20_ufixed55", 0.55),
}


def r20_macs(inner):
    """Forward MACs per image of CifarResNet(3) with the given 9 block-internal widths (matches trajectory_cifar.model_macs)."""
    total, b = 9 * 3 * 16 * 32 * 32, 0
    for s, (planes, res) in enumerate(zip(R20_PLANES, R20_RES)):
        for j in range(3):
            cin = R20_PLANES[s - 1] if (j == 0 and s > 0) else planes
            total += 9 * cin * inner[b] * res * res + 9 * inner[b] * planes * res * res
            b += 1
    return total + 64 * 10


def r20_train_cost(actions, widths):
    """Training MACs of a run as a fraction of the full-width run (3x forward per image; calibration passes added)."""
    full = 3 * r20_macs(R20_FULL_INNER) * R20_N_TRAIN * R20_EPOCHS
    if not actions:
        return 3 * r20_macs(widths) * R20_N_TRAIN * R20_EPOCHS / full
    sched, prev, w = [], 0, actions[0]["before"]
    for a in actions:
        sched.append((a["epoch"] - prev, w)); prev, w = a["epoch"], a["after"]
    sched.append((R20_EPOCHS - prev, w))
    calib = sum(r20_macs(a["before"]) * R20_CALIB * (1 if a["shrink_only"] else 2) for a in actions)
    return (sum(3 * r20_macs(w) * R20_N_TRAIN * n for n, w in sched) + calib) / full


def r20_widths_of(run_dir, r):
    if r["widths"]:
        return r["widths"]
    st = os.path.join(run_dir, "epoch_100_layers.csv") if r["done"] else None
    if st and os.path.exists(st):
        t = pd.read_csv(st)
        t = t[t.layer.str.endswith("conv1") & t.layer.str.startswith("layers.")].sort_values("depth_index")
        if len(t) == 9:
            return [int(c) for c in t.C]
    return None


def r20_section(results):
    base = {}
    for d in sorted(glob.glob(os.path.join(results, "r20", "cifar10_full_s*"))):
        r = read_run(d)
        if r and r["done"]:
            base[os.path.basename(d)] = r["acc"]
    runs = {}
    for d in sorted(glob.glob(os.path.join(results, "p2", "r20_*_s*"))):
        r = read_run(d)
        if r:
            r["widths"] = r20_widths_of(d, r)
            runs[os.path.basename(d)] = r
    if not runs:
        return
    print("\n==== P2i: ResNet-20 CIFAR-10 (block-internal widths, FLOPs-conserving) ====")
    if base:
        print(f"  full width: {' '.join(f'{v:.2f}' for v in base.values())}  mean {np.mean(list(base.values())):.2f}")
    for name, r in runs.items():
        st = "done" if r["done"] else f"epoch {r['epoch']}"
        w = r["widths"]
        mi = f"{r20_macs(w) / r20_macs(R20_FULL_INNER):.3f}" if w else "  ?  "
        tc = f"{r20_train_cost(r['actions'], w):.3f}" if w else "  ?  "
        print(f"  {name:18s} {st:9s} acc {r['acc']:6.2f}  inference MACs {mi}  training MACs {tc}  widths {w}")
    full_mean = np.mean(list(base.values())) if base else None
    for prefix, (ctrl_prefix, _) in R20_GROUPS.items():
        arm = {n: r for n, r in runs.items() if n.startswith(prefix + "_s") and r["done"]}
        ctrl = {n: r for n, r in runs.items() if n.startswith(ctrl_prefix + "_s") and r["done"]}
        if not arm:
            continue
        a = [r["acc"] for r in arm.values()]; c = [r["acc"] for r in ctrl.values()]
        line = f"  {prefix}: n={len(a)} mean {np.mean(a):.2f} (spread {max(a) - min(a):.2f})"
        if c:
            line += f"  vs {ctrl_prefix} n={len(c)} mean {np.mean(c):.2f}  diff {np.mean(a) - np.mean(c):+.2f}" \
                    f"  every seed above every control seed: {'yes' if min(a) > max(c) else 'no'}"
        if full_mean is not None:
            line += f"  vs full {np.mean(a) - full_mean:+.2f}"
        print(line)
        ws = [r["widths"] for r in arm.values() if r["widths"]]
        if len(ws) >= 2:
            spread = np.max(np.array(ws), axis=0) - np.min(np.array(ws), axis=0)
            stage = [np.mean([w[i] for i in range(3 * s, 3 * s + 3)]) / R20_PLANES[s] for w in ws for s in range(3)]
            stage = np.array(stage).reshape(len(ws), 3).mean(axis=0)
            print(f"    shape: per-site seed spread max {int(spread.max())} channels; mean width fraction per stage "
                  f"{' '.join(f'{x:.2f}' for x in stage)}  (P-r4: stage 1 <= 0.5, stage 3 >= 0.6)")
    if "r20_f47" in {n.rsplit('_s', 1)[0] for n in runs} and full_mean is not None:
        a = [r["acc"] for n, r in runs.items() if n.startswith("r20_f47_s") and r["done"]]
        c = [r["acc"] for n, r in runs.items() if n.startswith("r20_ufixed47_s") and r["done"]]
        if len(a) >= 3 and len(c) >= 3:
            print(f"  P-r1 (f47 >= ufixed47 + 0.3): {'yes' if np.mean(a) >= np.mean(c) + 0.3 else 'NO'};  "
                  f"P-r2 (f47 >= full - 0.5): {'yes' if np.mean(a) >= full_mean - 0.5 else 'NO'}")
        a2 = [r["acc"] for n, r in runs.items() if n.startswith("r20_apriori55_s") and r["done"]]
        c2 = [r["acc"] for n, r in runs.items() if n.startswith("r20_ufixed55_s") and r["done"]]
        if len(a2) >= 3 and len(c2) >= 3 and len(a) >= 3:
            print(f"  P-r3 (apriori55 >= ufixed55 + 0.2 and >= f47 - 0.3): "
                  f"{'yes' if (np.mean(a2) >= np.mean(c2) + 0.2 and np.mean(a2) >= np.mean(a) - 0.3) else 'NO'}")


if __name__ == "__main__":
    main()
