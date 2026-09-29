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
    # P2j sensitivity scan (2026-09-23, hostile review S4: the controller's own settings), seed 0, budget 0.673 params
    "c10_b673_ep1030":   ("cifar10", 0.673, "ruler_act", "uniform"),  # triggers at 10/30 instead of 15/40
    "c10_b673_ep2050":   ("cifar10", 0.673, "ruler_act", "uniform"),  # triggers at 20/50
    "c10_b673_ep15":     ("cifar10", 0.673, "ruler_act", "uniform"),  # one trigger at 15
    "c10_b673_ep154070": ("cifar10", 0.673, "ruler_act", "uniform"),  # three triggers 15/40/70
    "c10_b673_calib1600": ("cifar10", 0.673, "ruler_act", "uniform"), # 1,600 calibration images instead of 6,400
    "c10_b673_minw16":   ("cifar10", 0.673, "ruler_act", "uniform"),  # min width 16 instead of 8
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


# ------------------------------------------------------------------ P2k: transformer arm (2026-09-23)

def vit_section(results):
    """CIFAR-10 ViT (scripts/trajectory_vit.py): full / ruler / uniform / controller / a-priori, P-v1..v4."""
    root = os.path.join(results, "vit_c10")
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        r = read_run(d)
        if r:
            wj = os.path.join(d, "widths.json")
            if r["widths"] is None and os.path.exists(wj):
                r["widths"] = json.load(open(wj))["widths"]
            runs[os.path.basename(d)] = r
    if not runs:
        return
    print("\n==== P2k: CIFAR-10 ViT (six MLP hidden widths; parameter budget = FLOPs budget on the sites) ====")
    wj = os.path.join(root, "widths.json")
    if os.path.exists(wj):
        W = json.load(open(wj))
        print(f"  ruler (tau {W['tau']}): {W['ruler']}  fraction {W['fraction']['ruler']:.3f}; uniform {W['uniform']}")
        print(f"  P-v1 (ruler fraction < 0.9): {'yes' if W['fraction']['ruler'] < 0.9 else 'NO'}"
              f"{' (>= 0.95: no idle MLP width)' if W['fraction']['ruler'] >= 0.95 else ''}")
    for name, r in runs.items():
        st = "done" if r["done"] else f"epoch {r['epoch']}"
        print(f"  {name:14s} {st:9s} acc {r['acc']:6.2f}  frac {r['frac'] if r['frac'] is not None else float('nan'):.3f}"
              f"  widths {r['widths']}")
    acc = lambda pre: [r["acc"] for n, r in runs.items() if n.startswith(pre + "_s") and r["done"]]  # noqa: E731
    full, ruler, uni, b, ap = (acc(k) for k in ("full", "ruler", "uniform", "b", "apriori"))
    for lab, v in (("full", full), ("ruler", ruler), ("uniform", uni), ("controller", b), ("a-priori", ap)):
        if v:
            print(f"  {lab:10s} n={len(v)} {' '.join(f'{x:.2f}' for x in v)}  mean {np.mean(v):.2f}")
    if b and ruler and uni:
        print(f"  P-v2 (b >= ruler - 0.3): {'yes' if np.mean(b) >= np.mean(ruler) - 0.3 else 'NO'} ({np.mean(b):.2f} vs {np.mean(ruler):.2f});"
              f"  every b seed above every uniform seed: {'yes' if min(b) > max(uni) else 'no'} ({min(b):.2f} vs {max(uni):.2f})"
              f"{'' if len(b) >= 3 and len(uni) >= 3 else '  [fewer than 3 seeds: provisional]'}")
    if ap and b:
        print(f"  P-v3 (|apriori - b| <= 0.3): {'yes' if abs(np.mean(ap) - np.mean(b)) <= 0.3 else 'NO'} ({np.mean(ap):.2f} vs {np.mean(b):.2f})")
    ws = [r["widths"] for n, r in runs.items() if n.startswith("b_s") and r["done"] and r["widths"]]
    if ws:
        A = np.array(ws)
        print(f"  P-v4 shape (controller widths, mean over seeds): {[int(x) for x in A.mean(axis=0)]}"
              f"{'  per-site seed spread max ' + str(int((A.max(0) - A.min(0)).max())) if len(ws) >= 2 else ''}")


# ------------------------------------------------------------------ P2m: ImageNet-1k ResNet-18 (2026-09-24)
R18_MIDS = (2, 3, 5, 6, 8, 9, 11, 12)          # positions of the 8 block-internal widths in the 13-vector
R18_FULL = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512)


def _r18_run(d):
    """result.json (done) or the last epochs.csv row (running) of a train_imagenet_r18.py run, plus realloc.json."""
    rj, ec, rl = (os.path.join(d, f) for f in ("result.json", "epochs.csv", "realloc.json"))
    r = None
    if os.path.exists(rj):
        j = json.load(open(rj))
        r = dict(done=True, epoch=j["epochs"], acc=j["final_top1"], widths=j["widths"],
                 params_fraction=j.get("params_fraction"), epochs=j["epochs"])
    elif os.path.exists(ec):
        df = pd.read_csv(ec)
        if len(df):
            r = dict(done=False, epoch=int(df.epoch.iloc[-1]), acc=float(df.top1.iloc[-1]), widths=None,
                     params_fraction=None, epochs=90)
    if r is None:
        return None
    r["actions"] = json.load(open(rl)) if os.path.exists(rl) else []
    r["macs_fraction"] = r["actions"][-1]["macs_fraction"] if r["actions"] else None
    return r


def _r18_train_cost(actions, epochs):
    """Training compute in units of one full-width training: the MACs fraction in force at each epoch, averaged."""
    frac, at = 1.0, {a["epoch"]: a["macs_fraction"] for a in actions}
    total = 0.0
    if 0 in at:
        frac = at[0]
    for ep in range(1, epochs + 1):
        total += frac
        if ep in at:
            frac = at[ep]
    return total / epochs


def p2m_section(results):
    """ImageNet-1k ResNet-18: controller (f60, 0.60 MACs) vs uniform (ufixed60) vs A23 full width; P-m1..m4
    (notes/controller-criterion-2026-09-16.md, section P2m).  2 vs 2: descriptive only."""
    runs = {os.path.basename(d): _r18_run(d) for d in sorted(glob.glob(os.path.join(results, "p2m", "r18_*_s*")))}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    full = [_r18_run(d) for d in sorted(glob.glob(os.path.join(results, "a18_imagenet", "resnet18_full_s*")))]
    full = [r["acc"] for r in full if r and r["done"]]
    print("\n==== P2m: ImageNet-1k ResNet-18, FLOPs-conserving controller (8 block-internal widths) ====")
    if full:
        print(f"  A23 full width: {' '.join(f'{x:.2f}' for x in full)}  mean {np.mean(full):.2f}  spread {max(full) - min(full):.2f}")
    for name, r in runs.items():
        st = "done" if r["done"] else f"epoch {r['epoch']}"
        extra = ""
        if r["actions"]:
            extra = (f"  macs {r['macs_fraction']:.3f}  training compute {_r18_train_cost(r['actions'], r['epochs']):.2f} of full  "
                     + "[" + "; ".join(f"ep{x['epoch']}: {x['macs_fraction']:.3f}" + ("" if x["shrink_only"] else " grow+shrink")
                                        for x in r["actions"]) + "]")
        pf = f"  params {r['params_fraction']:.3f}" if r["params_fraction"] is not None else ""
        print(f"  {name:18s} {st:9s} top1 {r['acc']:6.2f}{pf}{extra}")
        if r["widths"]:
            print(f"      widths {r['widths']}")
    acc = lambda pre: [r["acc"] for n, r in runs.items() if n.startswith(pre + "_s") and r["done"]]  # noqa: E731
    f60, uni, ap = acc("r18_f60"), acc("r18_ufixed60"), acc("r18_apriori60")
    prov = "" if len(f60) >= 2 and len(uni) >= 2 else "  [fewer than 2 seeds: provisional]"
    if f60 and uni:
        print(f"  P-m1 (f60 >= ufixed60 + 0.5, mean) -> {'yes' if np.mean(f60) >= np.mean(uni) + 0.5 else 'NO'} "
              f"({np.mean(f60):.2f} vs {np.mean(uni):.2f}, diff {np.mean(f60) - np.mean(uni):+.2f}); "
              f"every f60 seed above every ufixed60 seed: {'yes' if min(f60) > max(uni) else 'no'} "
              f"(2 vs 2: p = 1/6 under the null, descriptive){prov}")
    if f60 and full:
        print(f"  P-m2 (f60 >= full - 1.0) -> {'yes' if np.mean(f60) >= np.mean(full) - 1.0 else 'NO'} "
              f"({np.mean(f60):.2f} vs {np.mean(full):.2f}, diff {np.mean(f60) - np.mean(full):+.2f}){prov}")
    if ap and f60:
        print(f"  a-priori arm vs f60: {np.mean(ap):.2f} vs {np.mean(f60):.2f} ({np.mean(ap) - np.mean(f60):+.2f})")
    ws = np.array([r["widths"] for n, r in runs.items() if n.startswith("r18_f60_s") and r["done"] and r["widths"]], dtype=float)
    if len(ws):
        rel = ws.mean(axis=0) / np.array(R18_FULL)
        s1, s4 = rel[[2, 3]], rel[[11, 12]]
        print(f"  P-m3 shape (mid / full, mean over seeds): stage1 {s1.round(2).tolist()} stage2 {rel[[5, 6]].round(2).tolist()} "
              f"stage3 {rel[[8, 9]].round(2).tolist()} stage4 {s4.round(2).tolist()} -> "
              f"stage1 <= 0.5 -> {'yes' if s1.max() <= 0.5 else 'NO'}; stage4 >= 0.7 -> {'yes' if s4.min() >= 0.7 else 'NO'}")
        if len(ws) >= 2:
            spread = (ws.max(0) - ws.min(0))[list(R18_MIDS)] / ws.mean(0)[list(R18_MIDS)]
            print(f"  P-m4 (per-site relative seed spread <= 10%) -> {'yes' if spread.max() <= 0.10 else 'NO'} "
                  f"(max {100 * spread.max():.1f}%, per site {[f'{100 * x:.0f}%' for x in spread]})")


# ------------------------------------------------------------------ P2l: shrink-only (FLOPs) and the x4-wide ResNet-20 (2026-09-25)
def _p2l_run(d):
    """acc at epoch 100, last macs_fraction from realloc.json, and the conv1-output k*(0.999)/C per stage (thirds)."""
    f = os.path.join(d, "epoch_100_layers.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    conv = df[(df.kind == "conv") & (df.epoch == 100)].sort_values("depth_index")
    sites = conv[conv.layer.str.contains(r"\.conv1$", regex=True)]
    r = (sites["k_star_0.999"] / sites.C).to_numpy()
    thirds = [float(np.mean(x)) for x in np.array_split(r, 3)] if len(r) >= 3 else []
    rl = os.path.join(d, "realloc.json")
    act = json.load(open(rl)) if os.path.exists(rl) else []
    return dict(acc=float(conv.test_acc.iloc[0]), macs=act[-1]["macs_fraction"] if act else None,
                widths=[int(c) for c in sites.C], stages=thirds)


def p2l_section(results):
    """P-l1..l4 (notes/controller-criterion-2026-09-16.md, section P2l)."""
    arms = {}
    for pre in ("c10_f60_nogrow", "c10_f60", "r20w4_full", "r20w4_f60", "r20w4_ufixed60"):
        runs = {os.path.basename(d): _p2l_run(d) for d in sorted(glob.glob(os.path.join(results, "p2", pre + "_s*")))}
        arms[pre] = {k: v for k, v in runs.items() if v}
    if not any(arms[k] for k in ("c10_f60_nogrow", "r20w4_full", "r20w4_f60", "r20w4_ufixed60")):
        return
    print("\n==== P2l: FLOPs-conserving shrink-only arm (VGG-16) and the x4-wide ResNet-20 (P-l1..l4) ====")
    for pre, runs in arms.items():
        for name, r in runs.items():
            st = [f"{x:.2f}" for x in r["stages"]]
            print(f"  {name:22s} acc {r['acc']:6.2f}" + (f"  macs {r['macs']:.3f}" if r["macs"] is not None else "")
                  + f"  conv1-out k*(.999)/C per stage {st}  widths {r['widths']}")
    acc = lambda pre: [r["acc"] for r in arms[pre].values()]   # noqa: E731
    # registered reference: f60 at the seeds of the shrink-only arm (three seeds, 93.60); the P2j seeds 3-4 of f60 have
    # no shrink-only counterpart and enter only the descriptive all-seed mean
    pairs = [(r["acc"], arms["c10_f60"][k.replace("_nogrow", "")]["acc"]) for k, r in arms["c10_f60_nogrow"].items()
             if k.replace("_nogrow", "") in arms["c10_f60"]]
    if pairs:
        ng, f60 = np.array(pairs).T
        d = ng.mean() - f60.mean()
        print(f"  P-l1 (nogrow <= f60 - 0.2: growing contributes under a FLOPs budget) -> {'yes' if d <= -0.2 else 'NO'} "
              f"({ng.mean():.2f} vs {f60.mean():.2f}, diff {d:+.2f}; same seeds, n={len(pairs)}; "
              f"per seed {' '.join(f'{x:+.2f}' for x in ng - f60)})"
              + ("  [>= -0.1: no measurable contribution of the grow operator]" if d >= -0.1 else ""))
        allf = acc("c10_f60")
        if len(allf) > len(pairs):
            print(f"       descriptive: against all {len(allf)} f60 seeds ({np.mean(allf):.2f}) the difference is {ng.mean() - np.mean(allf):+.2f}")
    full = arms["r20w4_full"]
    if full:
        s1 = np.mean([r["stages"][0] for r in full.values()]); s3 = np.mean([r["stages"][2] for r in full.values()])
        print(f"  P-l2 (x4 full: stage-3 conv1-out ruler < 0.7 and stage-1 >= 0.9: idle width appears when widened) -> "
              f"{'yes' if s3 < 0.7 and s1 >= 0.9 else 'NO'} (stage 1 {s1:.2f}, stage 3 {s3:.2f}; n={len(full)})")
    c, u, fw = acc("r20w4_f60"), acc("r20w4_ufixed60"), acc("r20w4_full")
    if c and u:
        print(f"  P-l3 (x4 controller >= uniform + 0.3 and every seed above every uniform seed) -> "
              f"{'yes' if np.mean(c) >= np.mean(u) + 0.3 and min(c) > max(u) else 'NO'} "
              f"({np.mean(c):.2f} vs {np.mean(u):.2f}, diff {np.mean(c) - np.mean(u):+.2f}; min {min(c):.2f} vs max {max(u):.2f}; n={len(c)}/{len(u)})")
    if c and fw:
        print(f"  P-l4 (x4 controller >= full - 0.5 at 0.60 MACs) -> {'yes' if np.mean(c) >= np.mean(fw) - 0.5 else 'NO'} "
              f"({np.mean(c):.2f} vs {np.mean(fw):.2f}, diff {np.mean(c) - np.mean(fw):+.2f})")


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
    vit_section(a.results)
    p2m_section(a.results)
    p2l_section(a.results)



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
