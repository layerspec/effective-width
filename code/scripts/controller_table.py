"""The main-text table of 'a width read from the covariance, used three ways' (2026-09-26, route C):
fold without retraining (P1), retrain at the ruler's widths (A18) and reach the widths in one training
with the controller (P2g), plus the uniform controls, on VGG-16 (BN) CIFAR-10; and ResNet-20 CIFAR-10
(P2i) where the ruler reads full width.  Columns: inference MACs and parameters as fractions of the full
network, training compute in units of one full-width training (the MACs fraction in force at each epoch,
averaged; a retrained arm counts the full training it was read from), accuracy as mean +- SD over seeds.

    python scripts/controller_table.py --results ../results --out ../paper/table_controller.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.p2_flops import FULL, macs                       # noqa: E402
from scripts.p2_analysis import _r18_train_cost, r20_macs      # noqa: E402

R20_FULL = [16] * 3 + [32] * 3 + [64] * 3


def _vgg_params(w):
    cin, n = 3, 0
    for c in w:
        n += 9 * cin * c + 2 * c
        cin = c
    return n + w[-1] * 10 + 10


def _accs(pattern):
    out = []
    for d in sorted(glob.glob(pattern)):
        tl = os.path.join(d, "trajectory_layers.csv")
        if not os.path.exists(os.path.join(d, "epoch_100_layers.csv")) or not os.path.exists(tl):
            continue
        t = pd.read_csv(tl)
        out.append(float(t.loc[t.epoch == t.epoch.max(), "test_acc"].iloc[0]))
    return out


def _r20_accs(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "epoch_100_layers.csv"))):
        out.append(float(pd.read_csv(f).test_acc.iloc[0]))
    return out


def _trained_widths(pattern):
    """The conv widths an arm was actually trained at, read from its trajectory file (depth order at the
    last epoch); None when no finished run is found.  The widths files list arms that were never trained
    (widths_act.json's 67%-parameter uniform arm), so the costs come from the trained network."""
    for d in sorted(glob.glob(pattern)):
        tl = os.path.join(d, "trajectory_layers.csv")
        if not os.path.exists(os.path.join(d, "epoch_100_layers.csv")) or not os.path.exists(tl):
            continue
        t = pd.read_csv(tl)
        t = t[(t.epoch == t.epoch.max()) & (t.kind == "conv")]
        if "depth_index" in t.columns:
            t = t.sort_values("depth_index")
        w = [int(c) for c in t.C]
        if len(w) == len(FULL):
            return w
    return None


def _row(arm, dataset, accs, macs_frac, params_frac, train, note=""):
    if not accs:
        return None
    return dict(arm=arm, dataset=dataset, acc=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                n=len(accs), macs=float(macs_frac), params=float(params_frac), train=float(train), note=note)


def _controller_rows(results, prefix, arm, dataset, macs_of_widths, params_of_widths, full_params):
    accs, mf, pf, tr = [], [], [], []
    for d in sorted(glob.glob(os.path.join(results, "p2", prefix + "_s*"))):
        a = _accs(d)
        rl = os.path.join(d, "realloc.json")
        if not a or not os.path.exists(rl):
            continue
        acts = json.load(open(rl))
        accs += a
        mf.append(acts[-1]["macs_fraction"])
        pf.append(params_of_widths(acts[-1]["after"]) / full_params)
        tr.append(_r18_train_cost(acts, 100))
    if not accs:
        return None
    return _row(arm, dataset, accs, np.mean(mf), np.mean(pf), np.mean(tr))


def build_rows(results):
    rows = []
    vgg_full_macs, vgg_full_params = macs(FULL), _vgg_params(FULL)
    ds = "VGG-16 (BN), CIFAR-10"
    rows.append(_row("full width", ds, _accs(os.path.join(results, "round3", "a5_vgg16_bn_none_s*")), 1.0, 1.0, 1.0))
    # fold without retraining: seed-0 fold plus any seed folders
    folds = {}
    for f in [os.path.join(results, "cssp", "fold_results.csv")] + sorted(glob.glob(os.path.join(results, "cssp", "seed*", "fold_results.csv"))):
        if os.path.exists(f):
            for _, r in pd.read_csv(f).iterrows():
                folds.setdefault(r.scheme, []).append(r)
    names = {"kstar_0.999": "fold, $\\tau = 0.999$", "kstar_0.99": "fold, $k^*(0.99)$", "a18_ruler_act_0.95": "fold, $\\tau = 0.95$"}
    for scheme, label in names.items():
        if scheme in folds:
            rs = folds[scheme]
            w = [int(x) for x in str(rs[0].widths).split()]
            rows.append(_row(label, ds, [float(r.acc_fold) for r in rs], macs(w) / vgg_full_macs, float(rs[0].param_frac), 1.0))
    wj = os.path.join(results, "a18", "widths_act.json")
    if os.path.exists(wj):
        W = json.load(open(wj))["widths"]
        for key, arm, label in (("ruler", "vgg16_bn_ruler_act", "retrain at the ruler's widths"), ("uniform", "vgg16_bn_uniform", "uniform cut, retrained")):
            pat = os.path.join(results, "a18", arm + "_s*")
            w = _trained_widths(pat) or W[key]; mfrac = macs(w) / vgg_full_macs
            rows.append(_row(label, ds, _accs(pat), mfrac, _vgg_params(w) / vgg_full_params, 1.0 + mfrac))
    for prefix, arm in (("c10_f88", "controller, 0.88 MACs"), ("c10_f60", "controller, 0.60 MACs"),
                        ("c10_f60_nogrow", "controller, 0.60 MACs, shrink only"), ("c10_ffixed342", "controller, 0.34 MACs")):
        r = _controller_rows(results, prefix, arm, ds, lambda w: macs(w) / vgg_full_macs, _vgg_params, vgg_full_params)
        if r:
            rows.append(r)
    ds = "ResNet-20, CIFAR-10"
    r20_full_macs = r20_macs(R20_FULL)
    r20_params = lambda inner: sum(9 * cin * m + 9 * m * pl for cin, m, pl in zip(  # noqa: E731
        [16, 16, 16, 16, 32, 32, 32, 64, 64], inner, [16] * 3 + [32] * 3 + [64] * 3))
    rows.append(_row("full width", ds, _r20_accs(os.path.join(results, "r20", "cifar10_full_s*")), 1.0, 1.0, 1.0))
    for prefix, arm in (("r20_f47", "controller, 0.47 MACs"), ("r20_ufixed47", "uniform, 0.47 MACs")):
        accs = _r20_accs(os.path.join(results, "p2", prefix + "_s*"))
        dirs = sorted(glob.glob(os.path.join(results, "p2", prefix + "_s*")))
        if not accs:
            continue
        rl = [json.load(open(os.path.join(d, "realloc.json"))) for d in dirs if os.path.exists(os.path.join(d, "realloc.json"))]
        if rl:
            rows.append(_row(arm, ds, accs, np.mean([a[-1]["macs_fraction"] for a in rl]),
                             np.mean([r20_params(a[-1]["after"]) / r20_params(R20_FULL) for a in rl]),
                             np.mean([_r18_train_cost(a, 100) for a in rl])))
        else:
            inner = [7, 7, 7, 15, 15, 15, 30, 30, 30]
            mfrac = r20_macs(inner) / r20_full_macs
            rows.append(_row(arm, ds, accs, mfrac, r20_params(inner) / r20_params(R20_FULL), mfrac))
    return [r for r in rows if r]


def render(rows):
    out = ["% Generated by scripts/controller_table.py.  Do not edit by hand.",
           "\\begin{tabular}{@{}lrrrl@{}}", "\\toprule",
           "Arm & MACs & params & training & accuracy (\\%) \\\\", "\\midrule"]
    last = None
    for r in rows:
        if r["dataset"] != last:
            if last is not None:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{r['dataset']}}}}} \\\\")
            last = r["dataset"]
        acc = f"{r['acc']:.2f} $\\pm$ {r['sd']:.2f}" if r["n"] > 1 else f"{r['acc']:.2f}"
        out.append(f"{r['arm']} & {r['macs']:.2f} & {r['params']:.2f} & {r['train']:.2f} & {acc} ($n={r['n']}$) \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    tex = render(build_rows(a.results))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(tex)
        print("wrote", a.out)
    else:
        print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
