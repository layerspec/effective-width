"""B9 with the number of positions changed (plan 9.30 A, P-b9.9 .. P-b9.11).

Lemma B9.0 caps the tau-width of a collapsed layer at P*K - 1 (P spatial positions, K classes).  Plan 10.28 varied K at
P = 2 x 2; here P is changed: VGG-16_BN on CIFAR-10 exactly as scripts/trajectory_cifar.py trains it, except that the
network upsamples its 32-pixel input to --px pixels in its forward pass (48 px: the last block works on 3 x 3 = 9
positions, P*K = 90).  A wrapper, so the trainer is not edited.

    python scripts/position_sweep.py train --px 48 --seed 0 --device mps --out ../results/position_sweep/px48_s0
    python scripts/position_sweep.py measure --px 48 --run ../results/position_sweep/px48_s0 --name px48_s0
    python scripts/position_sweep.py measure --px 32 --run ../results/wd_sweep/wd5e-4_s0 --name px32_s0     # instrument check
    python scripts/position_sweep.py judge --results ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.position_collapse_judge import BLOCK5, RATIO_MIN, cliff, k_star      # noqa: E402

K = 10
N_IMAGES = 6400


def build(px: int, num_classes: int = 10):
    """The CIFAR VGG-16_BN of the trainer; at px != 32 its forward upsamples the input first (same parameters, same names)."""
    from scripts.reproduce_garg import build_vgg16_bn_cifar
    m = build_vgg16_bn_cifar(num_classes, "small")
    if px != 32:
        inner = m.forward
        m.forward = lambda x: inner(F.interpolate(x, size=(px, px), mode="bilinear", align_corners=False))
    return m


def train(a, rest) -> int:
    import scripts.trajectory_cifar as tc
    real = tc.ARCHS["vgg16_bn"]
    tc.ARCHS["vgg16_bn"] = lambda nc=10: build(a.px, nc)
    try:
        rc = tc.main(["--arch", "vgg16_bn", "--dataset", "cifar10", "--seed", str(a.seed), "--device", a.device,
                      "--save-checkpoints", "--schedule", "0", "50", "100", "--out", a.out] + rest)
    finally:
        tc.ARCHS["vgg16_bn"] = real
    if os.path.isdir(a.out):
        json.dump({"px": a.px, "seed": a.seed}, open(os.path.join(a.out, "position_sweep.json"), "w"))
    return rc


def cell_statistics(v: np.ndarray, cell: np.ndarray, n_cells: int) -> dict:
    """v: (n, C) float64 samples, cell: (n,) cell index.  Spectrum of the covariance, the within-cell variance fraction
    w_cp, and w_out, the fraction of the variance outside the span of the centred cell means."""
    mu = v.mean(0)
    x = v - mu
    S = x.T @ x / len(x)
    tot = float(np.trace(S))
    means, counts = [], []
    for c in range(n_cells):
        sel = cell == c
        if sel.any():
            means.append(x[sel].mean(0)); counts.append(int(sel.sum()))
    Mc = np.stack(means); w = np.asarray(counts, dtype=np.float64) / len(x)
    B = (Mc * w[:, None]).T @ Mc
    U, s, _ = np.linalg.svd(Mc.T, full_matrices=False)
    Q = U[:, s > 1e-10 * s[0]]
    inside = float(np.trace(Q.T @ S @ Q))
    return dict(eig=np.linalg.eigvalsh(S)[::-1], w_cp=1.0 - float(np.trace(B)) / tot, w_out=1.0 - inside / tot,
                rank_means=int(Q.shape[1]), k_means=k_star(np.linalg.eigvalsh(B), 0.999))


@torch.no_grad()
def measure(a) -> int:
    from scripts.measure_stationarity import labelled_loader
    out = os.path.join(a.results, "position_sweep"); os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"{a.name}.json")
    if os.path.exists(dst):
        print(f"{a.name}: already measured"); return 0
    ck = os.path.join(a.run, "epoch_100.pt")
    model = build(a.px); model.load_state_dict(torch.load(ck, map_location="cpu")); model.eval()
    acts: dict = {}
    mods = dict(model.named_modules())
    hooks = [mods[n].register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o)) for n in BLOCK5]
    V = {n: [] for n in BLOCK5}; cells = {n: [] for n in BLOCK5}; P = {}
    correct = total = 0
    loader, _ = labelled_loader(a.data_root, "cifar10", None, None, n_images=N_IMAGES, seed=0)
    for x, y in loader:
        logits = model(x)
        correct += int((logits.argmax(1) == y).sum()); total += len(y)
        for n in BLOCK5:
            o = acts[n]; b, c, h, w = o.shape; P[n] = h * w
            V[n].append(o.permute(0, 2, 3, 1).reshape(b * h * w, c).double().numpy())
            pos = torch.arange(h * w).repeat(b)
            cells[n].append((y.repeat_interleave(h * w) * (h * w) + pos).numpy())
    for h_ in hooks:
        h_.remove()
    rows = []
    for n in BLOCK5:
        st = cell_statistics(np.concatenate(V[n]), np.concatenate(cells[n]), P[n] * K)
        g, ratio = cliff(st["eig"], K)
        rows.append(dict(layer=n, P=P[n], PK=P[n] * K, C=len(st["eig"]), g=g, ratio=ratio, k95=k_star(st["eig"], 0.95),
                         k999=k_star(st["eig"], 0.999), w_cp=st["w_cp"], w_out=st["w_out"], rank_means=st["rank_means"],
                         k_means=st["k_means"], n=int(sum(len(c) for c in cells[n])), eig=st["eig"][:200].tolist()))
        print(f"  {a.name} {n}: P {P[n]}  cliff {g} (ratio {ratio:.1f})  k*(0.999) {rows[-1]['k999']}  w_cp {st['w_cp']:.4f}  w_out {st['w_out']:.5f}", flush=True)
    accf = os.path.join(a.run, "epoch_100_layers.csv")
    test_acc = None
    if os.path.exists(accf):
        import pandas as pd
        test_acc = float(pd.read_csv(accf).test_acc.iloc[0])
    json.dump(dict(name=a.name, px=a.px, run=a.run, train_acc_6400=correct / total, test_acc=test_acc, layers=rows), open(dst, "w"))
    return 0


def judge(results: str) -> dict:
    R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(results, "position_sweep", "px*_s*.json")))}
    V: dict = {}
    print("==== B9 with the number of positions changed (plan 9.30 A) ====")
    for name, r in R.items():
        print(f"  {name}: px {r['px']}  test acc {r['test_acc']}  " + "  ".join(
            f"{l['layer'].split('.')[1]}: P {l['P']} cliff {l['g']}({l['ratio']:.1f}) k999 {l['k999']} w_out {l['w_out']:.4f}" for l in r["layers"]))
    at = lambda l: l["g"] in (l["PK"] - 1, l["PK"])                                   # noqa: E731
    ctrl = [l for n, r in R.items() if r["px"] == 32 for l in r["layers"]]
    if len(ctrl) == 6:
        n = sum(at(l) and l["ratio"] >= RATIO_MIN for l in ctrl)
        V["P-b9.9"] = n >= 5
        print(f"  P-b9.9 (instrument check at 32 px: cliff at 39 or 40 with ratio >= {RATIO_MIN:g}, >= 5/6) -> {'holds' if V['P-b9.9'] else 'fails'} ({n}/6)")
    new = [l for n, r in R.items() if r["px"] == 48 for l in r["layers"]]
    if len(new) == 6 and V.get("P-b9.9"):
        cl = [l for l in new if l["ratio"] >= RATIO_MIN]
        V["P-b9.10"] = len(cl) >= 4 and all(at(l) for l in cl)
        V["P-b9.11"] = all(l["k999"] <= l["PK"] for l in new)
        print(f"  P-b9.10 (48 px: every layer with a cliff of ratio >= 3 has it at 89 or 90, and >= 4/6 layers have one) -> "
              f"{'holds' if V['P-b9.10'] else 'fails'} ({sum(at(l) for l in cl)}/{len(cl)} cliffs at 89/90; {len(cl)}/6 layers with a cliff)")
        print(f"  P-b9.11 (48 px: k*(0.999) <= 90, 6/6) -> {'holds' if V['P-b9.11'] else 'fails'} ({sum(l['k999'] <= l['PK'] for l in new)}/6)")
    elif len(new) == 6:
        print("  P-b9.10 / P-b9.11 not judged: the instrument check did not pass")
    json.dump({"verdicts": V}, open(os.path.join(results, "position_sweep", "judgement.json"), "w"), indent=1)
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("train", "measure", "judge"))
    p.add_argument("--px", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--results", default="../results")
    p.add_argument("--data-root", default="../data")
    a, rest = p.parse_known_args(argv)
    if a.mode == "train":
        return train(a, rest)
    if a.mode == "measure":
        return measure(a)
    judge(a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
