"""The ReLU second-moment map with a Gaussian copula and the empirical marginals of the pre-activation
(2026-09-27; feasibility note section C, route 1).  The Gaussian recursion (apriori_relu_check.py) predicts
Cov(relu z) from (E z, Cov z) as if z were Gaussian and overestimates the effective width after the first layer.
Here the pairwise dependence is still a Gaussian copula with the measured correlation, but each channel's marginal
is its empirical distribution: if this closes the gap, the non-Gaussianity that matters is marginal (kurtosis, mean
shift) and a cumulant-corrected recursion can exist; if it does not, it is the joint structure.

    python scripts/relu_copula_check.py timm:vgg16_bn.tv_in1k_random --inputs matched --seed 1 --max-batches 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layerspec import api                                                       # noqa: E402
from scripts.apriori_check import make_loader, OUT, stem                        # noqa: E402
from scripts.apriori_relu_check import gaussian_relu_cov, subsample, k_star     # noqa: E402

GRID = np.linspace(-5.0, 5.0, 1001)


def marginal_tables(z: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """(C, G): r_c(t) = relu(F_c^{-1}(Phi(t))) on the standard-normal grid, F_c the empirical CDF of channel c."""
    p = norm.cdf(grid)
    q = np.quantile(np.asarray(z, dtype=np.float64), p, axis=0).T          # (C, G)
    return np.clip(q, 0.0, None)


def copula_relu_cov(R: np.ndarray, tab: np.ndarray, nodes: int = 40, grid: np.ndarray = GRID) -> np.ndarray:
    """Cov(relu z) when (z_i, z_j) have a Gaussian copula with correlation R_ij and the marginals of `tab`."""
    R = np.clip(np.asarray(R, dtype=np.float64), -1.0, 1.0)
    C, G = tab.shape
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    t0, dt = grid[0], grid[1] - grid[0]

    def lookup(j_idx, arg):                     # tab[j] linearly interpolated at arg, arg shape (..., ) matching j_idx broadcast
        u = np.clip((arg - t0) / dt, 0, G - 1 - 1e-9)
        i0 = np.floor(u).astype(int); f = u - i0
        return tab[j_idx, i0] * (1 - f) + tab[j_idx, i0 + 1] * f

    r_at_x = np.stack([np.interp(x, grid, tab[c]) for c in range(C)])   # (C, A)
    m = r_at_x @ w                                                       # E relu(z_c)
    E = np.empty((C, C))
    jj = np.arange(C)[:, None, None]
    for i in range(C):
        rho = R[i][:, None, None]                                        # (C,1,1)
        arg = rho * x[None, :, None] + np.sqrt(np.clip(1 - rho ** 2, 0, None)) * x[None, None, :]   # (C, A, B)
        rj = lookup(jj, arg)                                             # (C, A, B)
        inner = rj @ w                                                   # (C, A): E over v
        E[i] = (r_at_x[i][None, :] * inner) @ w                          # E over u
    return E - np.outer(m, m)


class _Acc:
    def __init__(self):
        self.n, self.s1, self.s2 = 0, None, None

    def add(self, x):
        x = x.to(torch.float64)
        s1, s2 = x.sum(0).cpu(), (x.T @ x).cpu()
        self.n += x.shape[0]
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2

    def mean_cov(self):
        m = self.s1 / self.n
        return m, self.s2 / self.n - torch.outer(m, m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--inputs", default="matched")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--max-pos", type=int, default=1024)
    ap.add_argument("--reservoir", type=int, default=60000, help="rows kept per ReLU for the marginals")
    ap.add_argument("--out", default=os.path.join(OUT, "relu_copula"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    model = api.load_model(a.model, seed=a.seed).to(a.device).eval()
    relus = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ReLU) and n.startswith("features")]
    acc_z, acc_a, res = {n: _Acc() for n, _ in relus}, {n: _Acc() for n, _ in relus}, {n: [] for n, _ in relus}
    kept = {n: 0 for n, _ in relus}
    gen = torch.Generator().manual_seed(a.seed)

    def pre(name):
        def f(mod, inp):
            z = subsample(inp[0].detach(), a.max_pos)
            acc_z[name].add(z)
            if kept[name] < a.reservoir:
                take = min(z.shape[0], a.reservoir - kept[name])
                idx = torch.randperm(z.shape[0], generator=gen)[:take]
                res[name].append(z[idx.to(z.device)].cpu().numpy()); kept[name] += take
        return f

    def post(name):
        def f(mod, inp, out):
            acc_a[name].add(subsample(out.detach(), a.max_pos))
        return f
    hs = [m.register_forward_pre_hook(pre(n)) for n, m in relus] + [m.register_forward_hook(post(n)) for n, m in relus]
    loader = make_loader(a.inputs, a.seed)
    t0 = time.time()
    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            if b >= a.max_batches:
                break
            model(x.to(a.device))
    for h in hs:
        h.remove()
    print(f"probe {time.time() - t0:.0f}s over {b} batches", flush=True)
    rows = []
    for n, _ in relus:
        mz, Sz = acc_z[n].mean_cov(); _, Sa = acc_a[n].mean_cov()
        z = np.concatenate(res[n])
        Sg = gaussian_relu_cov(mz, Sz)[0].numpy()
        s = np.sqrt(np.clip(np.diag(Sz.numpy()), 1e-12, None)); R = Sz.numpy() / np.outer(s, s)
        t1 = time.time(); Sc = copula_relu_cov(R, marginal_tables(z)); tc = time.time() - t1
        C = len(mz)
        kurt = float(np.mean(((z - z.mean(0)) / z.std(0)) ** 4) - 3)
        ks = lambda M, t: k_star(torch.as_tensor(np.asarray(M)), (t,))[t]   # noqa: E731
        rows.append(dict(relu=n, C=C, meas=ks(Sa, 0.95), gauss=ks(Sg, 0.95), copula=ks(Sc, 0.95),
                         meas_999=ks(Sa, 0.999), gauss_999=ks(Sg, 0.999), copula_999=ks(Sc, 0.999),
                         kurtosis=kurt, mean_over_sd=float(np.mean(np.abs(mz.numpy()) / s)), rows=len(z), seconds=round(tc)))
        r = rows[-1]
        print(f"{n:12s} C {C:4d}  k*(.95)/C meas {r['meas']/C:.3f} gauss {r['gauss']/C:.3f} copula {r['copula']/C:.3f}   "
              f"k*(.999)/C meas {r['meas_999']/C:.3f} gauss {r['gauss_999']/C:.3f} copula {r['copula_999']/C:.3f}   kurt {kurt:+.2f}  ({tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(a.out, f"{stem(a.model)}_{a.inputs}_seed{a.seed}.csv")
    df.to_csv(path, index=False); print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
