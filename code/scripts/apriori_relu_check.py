#!/usr/bin/env python
"""A-priori profile feasibility, step 1 (notes/apriori-plan-2026-09-16.md):
how accurate is the Gaussian + ReLU second-moment map at finite width?

For every ReLU of a random-init network, accumulate the mean and covariance
of its input z (the BN output; at initialisation BN in eval mode is the
identity) and of its output a = relu(z) over the images.  Predict Cov(a) from
(E z, Cov z) under the assumption that z is Gaussian, and compare the
effective width k*(tau) of the predicted and the measured Cov(a).  The images
can be the real ones or the matched Gaussian surrogate (the theory's own
input); with the surrogate the only approximation tested is Gaussianity of z
after the first layer.

    python scripts/apriori_relu_check.py timm:vgg16_bn.tv_in1k_random --inputs matched --seed 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layerspec import api  # noqa: E402
from scripts.apriori_check import make_loader, OUT, stem  # noqa: E402

_SQRT2 = math.sqrt(2.0)


def _phi(x):
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + torch.erf(x / _SQRT2))


def relu_mean(mu, tau):
    """E relu(W), W ~ N(mu, tau^2)."""
    tau = tau.clamp_min(1e-12)
    return mu * _Phi(mu / tau) + tau * _phi(mu / tau)


def gaussian_relu_cov(m, S, nodes: int = 80):
    """Cov(relu z) for z ~ N(m, S), via 1-D Gauss-Hermite quadrature per pair.
    Write z_j = m_j + s_j (rho_ij u + sqrt(1-rho_ij^2) v); the inner expectation
    over v is closed-form, the outer over u is quadrature."""
    m = m.to(torch.float64)
    S = S.to(torch.float64)
    s = S.diagonal().clamp_min(0).sqrt()
    rho = (S / (s[:, None] * s[None, :]).clamp_min(1e-30)).clamp(-1, 1)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)      # probabilists' Hermite: weight exp(-u^2/2)
    x = torch.as_tensor(x, dtype=torch.float64)
    w = torch.as_tensor(w / w.sum(), dtype=torch.float64)
    C = S.shape[0]
    M2 = torch.zeros(C, C, dtype=torch.float64)
    tau = s[None, :] * torch.sqrt((1 - rho * rho).clamp_min(0))      # (i, j)
    for uk, wk in zip(x, w):
        ri = torch.relu(m + s * uk)                                   # (i,)
        mu_j = m[None, :] + s[None, :] * rho * uk                     # (i, j)
        M2 += wk * ri[:, None] * relu_mean(mu_j, tau)
    M2 = 0.5 * (M2 + M2.T)
    ma = relu_mean(m, s)
    return M2 - ma[:, None] * ma[None, :], ma


def k_star(S, taus=(0.9, 0.95, 0.99)):
    ev = torch.linalg.eigvalsh(S.to(torch.float64)).clamp_min(0).flip(0)
    tot = ev.sum()
    if tot <= 0:
        return {t: 0 for t in taus}
    cum = ev.cumsum(0) / tot
    return {t: int((cum < t).sum()) + 1 for t in taus}


class Moments:
    def __init__(self):
        self.n = 0
        self.s1 = None
        self.s2 = None
        self.s4 = None                   # per-channel sum of x^4 (channel kurtosis, elliptical diagnostic)

    def add(self, x):                    # x: (N, C); per-batch products on the device, sums in float64 on the CPU
        x = x.to(torch.float32)
        s1 = x.sum(0).cpu().to(torch.float64)
        s2 = (x.T @ x).cpu().to(torch.float64)
        s4 = (x ** 4).sum(0).cpu().to(torch.float64)
        if self.s1 is None:
            self.s1, self.s2, self.s4 = torch.zeros_like(s1), torch.zeros_like(s2), torch.zeros_like(s4)
        self.n += x.shape[0]
        self.s1 += s1
        self.s2 += s2
        self.s4 += s4

    def mean_cov(self):
        m = self.s1 / self.n
        return m.cpu(), (self.s2 / self.n - m[:, None] * m[None, :]).cpu()


def subsample(t, max_pos=1024):
    """(B, C, H, W) -> (N, C) with at most max_pos positions per image (regular grid)."""
    B, C, H, W = t.shape
    step = max(1, int(math.ceil(math.sqrt(H * W / max_pos))))
    t = t[:, :, ::step, ::step]
    return t.permute(0, 2, 3, 1).reshape(-1, C)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--inputs", default="matched", choices=["matched", "real", "pink", "white"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file names (e.g. a --max-batches subset)")
    ap.add_argument("--save-moments", action="store_true",
                    help="also write <stem>_<inputs>_seed<s>_moments.npz with E z, Cov z, E a, Cov a, E z^4 per ReLU")
    args = ap.parse_args(argv)

    model = api.load_model(args.model, seed=args.seed).eval().to(args.device)
    loader = make_loader(args.inputs, args.seed)
    relus = [(n, mod) for n, mod in model.named_modules() if isinstance(mod, nn.ReLU)]
    mom_z, mom_a = {}, {}
    order = []

    def pre(name):
        def f(mod, inp):
            z = inp[0]
            if z.dim() != 4 or z.shape[1] > 2048:
                return
            if name not in mom_z:
                mom_z[name], mom_a[name] = Moments(), Moments()
                order.append(name)
            mom_z[name].add(subsample(z))          # before an in-place ReLU overwrites it
        return f

    def post(name):
        def f(mod, inp, out):
            if name in mom_z:
                mom_a[name].add(subsample(out))
        return f

    handles = [mod.register_forward_pre_hook(pre(n)) for n, mod in relus] + \
              [mod.register_forward_hook(post(n)) for n, mod in relus]
    t0 = time.time()
    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            if args.max_batches and b >= args.max_batches:
                break
            model(x.to(args.device))
    for h in handles:
        h.remove()
    print(f"{args.model} on {args.inputs} seed {args.seed}: {len(order)} ReLUs, {time.time() - t0:.0f}s")

    rows = []
    saved = {}
    print(f"{'relu':24s} {'C':>4s}  {'k*z .95':>7s}  {'k*a meas':>8s} {'k*a pred':>8s}  {'meas/C':>6s} {'pred/C':>6s}  "
          f"{'k*a .99 meas/pred':>17s}  {'relF':>5s}  {'skew z':>6s} {'kurt z':>6s}")
    for key in order:
        mz, Sz = mom_z[key].mean_cov()
        ma, Sa = mom_a[key].mean_cov()
        Sp, mp = gaussian_relu_cov(mz, Sz)
        C = Sz.shape[0]
        kz, ka, kp = k_star(Sz), k_star(Sa), k_star(Sp)
        rel = float((Sa - Sp).norm() / Sa.norm())
        if args.save_moments:
            saved[key] = dict(mz=mz.numpy(), Sz=Sz.numpy(), ma=ma.numpy(), Sa=Sa.numpy(), Sp=Sp.numpy(), mp=mp.numpy(),
                              m4z=(mom_z[key].s4 / mom_z[key].n).numpy(), n=mom_z[key].n)
        rows.append(dict(model=args.model, inputs=args.inputs, seed=args.seed, relu=key, C=C, n=mom_z[key].n,
                         kz95=kz[0.95], ka95=ka[0.95], kp95=kp[0.95], ka99=ka[0.99], kp99=kp[0.99],
                         ka90=ka[0.90], kp90=kp[0.90], rel_frob=rel,
                         mean_a_meas=float(ma.mean()), mean_a_pred=float(mp.mean())))
        print(f"{key:24s} {C:4d}  {kz[0.95]:7d}  {ka[0.95]:8d} {kp[0.95]:8d}  {ka[0.95]/C:6.3f} {kp[0.95]/C:6.3f}  "
              f"{ka[0.99]:8d}/{kp[0.99]:<8d}  {rel:5.3f}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "relu"), exist_ok=True)
    path = os.path.join(OUT, "relu", f"{stem(args.model)}_{args.inputs}_seed{args.seed}{args.tag}.csv")
    df.to_csv(path, index=False)
    if args.save_moments:
        mpath = path.replace(".csv", "_moments.npz")
        np.savez(mpath, keys=np.array(list(saved)), **{f"{k}__{f}": v for k, d in saved.items() for f, v in d.items()})
        print(f"-> {mpath}")
    d95 = (df.kp95 - df.ka95) / df.C
    print(f"\nk*(0.95)/C pred - meas: median {d95.median():+.3f}, max |.| {d95.abs().max():.3f}; "
          f"Spearman(meas, pred) over layers {df.ka95.corr(df.kp95, method='spearman'):+.2f}; "
          f"rel. Frobenius median {df.rel_frob.median():.3f}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
