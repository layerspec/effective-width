"""P1 of the reallocation controller (notes/controller-criterion-2026-09-16.md, Proposition A):
removing CHANNELS of a post-activation tensor is column-subset selection, not subspace truncation.

For every post-activation output a_l of the trained VGG-16 (BN) on CIFAR-10 (the A18 seed-0
full-width network), Sigma_a is accumulated over all positions of calibration images; then, at
inference and without retraining, each layer is either

  subspace   a <- mean + U_m U_m^T (a - mean)              (Proposition 2, keeps C channels)
  cssp       a_S kept, a_{S^c} <- mean_{S^c} + B^T (a_S - mean_S), B = Sigma[S,S]^+ Sigma[S,S^c]
             with S the greedy (pivoted-Cholesky) column subset of size m       (Proposition A)

The cssp map is exactly what folding the removed channels into the next convolution computes,
so its accuracy is the accuracy of a network that HAS m channels at that layer.  Reported per
layer and per m: residual fraction tr(R_S)/tr(Sigma) against the subspace residual 1 - F(m),
their ratio (Proposition A says it lies in [1, m+1]), and test accuracy for both maps; and for
all layers at once at the A18 post-activation ruler widths (tau = 0.999 and 0.95).

    python scripts/cssp_check.py --device mps --out ../results/cssp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reproduce_garg import CIFAR_STATS, evaluate                 # noqa: E402
from scripts.trajectory_cifar import VGG16_WIDTHS, build_vgg16_bn_cifar_widths  # noqa: E402

RELU_LAYERS = ["features.2", "features.5", "features.9", "features.12", "features.16", "features.19",
               "features.22", "features.26", "features.29", "features.32", "features.36", "features.39",
               "features.42"]


def loaders(root: str, n_calib: int, batch_size: int, workers: int, seed: int):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    mean, std, _ = CIFAR_STATS["cifar10"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(root, train=True, download=False, transform=tf)
    te = datasets.CIFAR10(root, train=False, download=False, transform=tf)
    g = np.random.default_rng(seed)
    idx = g.choice(len(tr), size=n_calib, replace=False)
    calib = DataLoader(Subset(tr, idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=workers)
    test = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=workers)
    return calib, test


def load_model(path: str, device: str) -> nn.Module:
    m = build_vgg16_bn_cifar_widths(VGG16_WIDTHS, num_classes=10)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    m.load_state_dict(sd)
    return m.to(device).eval()


class Stats:
    """Per-layer mean and covariance over all positions, float64 on CPU."""

    def __init__(self):
        self.n = 0
        self.s = None
        self.ss = None

    def update(self, a: torch.Tensor):
        x = a.detach().permute(0, 2, 3, 1).reshape(-1, a.shape[1]).to("cpu").double()  # cast after leaving MPS
        self.n += x.shape[0]
        self.s = x.sum(0) if self.s is None else self.s + x.sum(0)
        self.ss = x.T @ x if self.ss is None else self.ss + x.T @ x

    def finish(self):
        mean = self.s / self.n
        cov = self.ss / self.n - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        return mean, cov


def greedy_cssp(cov: torch.Tensor, m: int):
    """Pivoted Cholesky on the covariance: pick the channel with the largest residual diagonal.
    Returns the ordered pivots and the residual trace after each pick (index k = trace after k picks)."""
    C = cov.shape[0]
    R = cov.clone()
    order, traces = [], [float(torch.trace(R))]
    for _ in range(m):
        d = torch.diagonal(R).clone()
        for j in order:
            d[j] = -1.0
        j = int(torch.argmax(d))
        if d[j] <= 1e-12 * traces[0]:
            break
        r = R[:, j] / R[j, j]
        R = R - torch.outer(r, R[j, :])
        R = 0.5 * (R + R.T)
        order.append(j)
        traces.append(max(float(torch.trace(R)), 0.0))
    return order, traces


def cssp_map(mean: torch.Tensor, cov: torch.Tensor, S: list[int], device: str):
    C = cov.shape[0]
    keep = torch.tensor(sorted(S))
    drop = torch.tensor([j for j in range(C) if j not in set(S)])
    if len(drop) == 0:
        return None
    Sss = cov[keep][:, keep]
    B = torch.linalg.pinv(Sss, rtol=1e-8) @ cov[keep][:, drop]       # |S| x |drop|
    mk, md = mean[keep], mean[drop]
    c = (md - B.T @ mk).to(device, torch.float32)                   # constant offset of the reconstruction
    Bt = B.T.to(device, torch.float32)                              # |drop| x |S|
    keep, drop = keep.to(device), drop.to(device)

    def f(a):
        out = a.clone()
        ak = a[:, keep]                                             # N x |S| x H x W
        rec = torch.einsum("dk,nkhw->ndhw", Bt, ak) + c[None, :, None, None]
        out[:, drop] = rec
        return out
    return f


def subspace_map(mean: torch.Tensor, cov: torch.Tensor, m: int, device: str):
    evals, evecs = np.linalg.eigh(cov.numpy())
    U = torch.from_numpy(np.ascontiguousarray(evecs[:, -m:])).to(device, torch.float32)  # C x m
    mu = mean.to(device, torch.float32)

    def f(a):
        z = a - mu[None, :, None, None]
        p = torch.einsum("cm,nchw->nmhw", U, z)
        return mu[None, :, None, None] + torch.einsum("cm,nmhw->nchw", U, p)
    return f


class Patch:
    """Register hooks that replace the outputs of named modules with given maps."""

    def __init__(self, model: nn.Module, maps: dict):
        self.h = []
        mods = dict(model.named_modules())
        for name, fn in maps.items():
            self.h.append(mods[name].register_forward_hook(lambda mod, inp, out, fn=fn: fn(out)))

    def remove(self):
        for h in self.h:
            h.remove()


def kstar(evals_desc: np.ndarray, tau: float) -> int:
    f = np.cumsum(evals_desc) / evals_desc.sum()
    return int(np.searchsorted(f, tau) + 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default="../results/round3_pt/a5_vgg16_bn_none_s0/vgg16_bn_cifar10_final.pt")
    p.add_argument("--widths-act", default="../results/a18/widths_act.json")
    p.add_argument("--data-root", default="../data")
    p.add_argument("--device", default="mps")
    p.add_argument("--n-calib", type=int, default=6400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--taus", type=float, nargs="+", default=[0.95, 0.99, 0.999])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/cssp")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    model = load_model(a.checkpoint, a.device)
    calib, test = loaders(a.data_root, a.n_calib, a.batch_size, a.workers, a.seed)
    mods = dict(model.named_modules())

    # ---- calibration: mean and covariance of every post-activation output over all positions
    stats = {n: Stats() for n in RELU_LAYERS}
    hs = [mods[n].register_forward_hook(lambda mod, inp, out, n=n: stats[n].update(out)) for n in RELU_LAYERS]
    t0 = time.time()
    with torch.no_grad():
        for x, _ in calib:
            model(x.to(a.device))
    for h in hs:
        h.remove()
    moments = {n: stats[n].finish() for n in RELU_LAYERS}
    print(f"calibration on {a.n_calib} training images, all positions: {time.time() - t0:.0f}s", flush=True)

    with torch.no_grad():
        acc_full = evaluate(model, test, a.device)
    print(f"full network test accuracy {acc_full:.2f}%", flush=True)

    # ---- per-layer spectra, greedy pivots and residual curves
    spec, pivots, traces = {}, {}, {}
    for n in RELU_LAYERS:
        mean, cov = moments[n]
        cn = cov.numpy()
        if not np.isfinite(cn).all():
            raise RuntimeError(f"{n}: covariance has non-finite entries (n={stats[n].n})")
        ev = np.clip(np.linalg.eigvalsh(cn)[::-1], 0, None)
        spec[n] = ev
        order, tr = greedy_cssp(cov, cov.shape[0])
        pivots[n], traces[n] = order, np.array(tr)

    rows = []
    # ---- (ii) one layer at a time, m = k*(tau) of Sigma_a, both maps
    for n in RELU_LAYERS:
        mean, cov = moments[n]
        C = cov.shape[0]
        ev = spec[n]
        for tau in a.taus:
            m = min(kstar(ev, tau), C)
            sub_resid = float(ev[m:].sum() / ev.sum()) if m < C else 0.0
            cssp_resid = float(traces[n][m] / traces[n][0]) if m < len(traces[n]) else 0.0
            res = {"layer": n, "C": C, "tau": tau, "m": m, "subspace_resid": sub_resid, "cssp_resid": cssp_resid,
                   "ratio": (cssp_resid / sub_resid) if sub_resid > 0 else np.nan}
            for mode in ("subspace", "cssp"):
                if m >= C:
                    res[f"acc_{mode}"] = acc_full
                    continue
                fn = subspace_map(mean, cov, m, a.device) if mode == "subspace" \
                    else cssp_map(mean, cov, pivots[n][:m], a.device)
                pt = Patch(model, {n: fn})
                with torch.no_grad():
                    res[f"acc_{mode}"] = evaluate(model, test, a.device)
                pt.remove()
            rows.append(res)
            print(f"  {n:12s} C={C:3d} tau={tau:5.3f} m={m:3d}  resid sub {sub_resid:.4f} cssp {cssp_resid:.4f} "
                  f"(x{res['ratio']:.2f})  acc sub {res['acc_subspace']:.2f} cssp {res['acc_cssp']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "cssp_per_layer.csv"), index=False)

    # ---- (i) all layers at once at the A18 post-activation ruler widths, and at k*(tau) of these spectra
    widths = json.load(open(a.widths_act))["widths"]
    schemes = {"a18_ruler_act_0.999": dict(zip(RELU_LAYERS, widths["ruler"])),
               "a18_ruler_act_0.95": dict(zip(RELU_LAYERS, widths["ruler95"]))}
    for tau in a.taus:
        schemes[f"kstar_{tau}"] = {n: min(kstar(spec[n], tau), spec[n].shape[0]) for n in RELU_LAYERS}
    summary = []
    for name, wd in schemes.items():
        for mode in ("subspace", "cssp"):
            maps = {}
            tot_sub = tot_cssp = 0.0
            for n in RELU_LAYERS:
                mean, cov = moments[n]
                C = cov.shape[0]
                m = int(wd[n])
                if m >= C:
                    continue
                maps[n] = subspace_map(mean, cov, m, a.device) if mode == "subspace" \
                    else cssp_map(mean, cov, pivots[n][:m], a.device)
                tot_sub += float(spec[n][m:].sum() / spec[n].sum())
                tot_cssp += float(traces[n][m] / traces[n][0])
            pt = Patch(model, maps)
            with torch.no_grad():
                acc = evaluate(model, test, a.device)
            pt.remove()
            summary.append({"scheme": name, "mode": mode, "acc": acc, "acc_full": acc_full,
                            "sum_subspace_resid": tot_sub, "sum_cssp_resid": tot_cssp,
                            "widths": " ".join(str(int(wd[n])) for n in RELU_LAYERS)})
            print(f"all layers {name:22s} {mode:8s} acc {acc:6.2f}  (full {acc_full:.2f})  "
                  f"sum resid sub {tot_sub:.3f} cssp {tot_cssp:.3f}", flush=True)
    pd.DataFrame(summary).to_csv(os.path.join(a.out, "cssp_all_layers.csv"), index=False)

    # residual curves for the paper figure
    np.savez(os.path.join(a.out, "cssp_curves.npz"),
             **{f"{n}_eig": spec[n] for n in RELU_LAYERS}, **{f"{n}_trace": traces[n] for n in RELU_LAYERS},
             **{f"{n}_pivots": np.array(pivots[n]) for n in RELU_LAYERS})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
