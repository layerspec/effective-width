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


def predicted_kurtosis(E_e: float, E_e2: float, pr: float) -> float:
    """Excess kurtosis of a channel z = w^T x, w i.i.d. Gaussian, averaged over w.  For fixed x,
    E_w (w^T x)^4 = 3 sigma_w^4 |x|^4 and E_w (w^T Sigma w)^2 = sigma_w^4 ((tr Sigma)^2 + 2 tr Sigma^2), so with
    pr = (tr Sigma)^2 / tr Sigma^2 the participation ratio of the patch covariance,
        kurt ~ 3 E|x|^4 / ((E|x|^2)^2 (1 + 2/pr)) - 3,
    which is exact for a Gaussian scale mixture x = s g (kurt = 3 E s^4 / (E s^2)^2 - 3)."""
    return 3.0 * E_e2 / (E_e ** 2 * (1.0 + 2.0 / pr)) - 3.0


def matched_tables(mean, sd, kurt, grid: np.ndarray = GRID) -> np.ndarray:
    """(C, G): r_c(t) = relu(mean_c + sd_c Q_c(Phi(t))) with Q_c the unit-variance Student-t quantile whose excess
    kurtosis is kurt_c (nu = 4 + 6/kurt; Gaussian when kurt <= 0.05)."""
    from scipy.stats import t as student
    mean, sd, kurt = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (mean, sd, kurt))
    p = norm.cdf(grid)
    out = np.empty((len(mean), len(grid)))
    for c in range(len(mean)):
        k = float(kurt[c])
        if k <= 0.05:
            q = norm.ppf(p)
        else:
            nu = 4.0 + 6.0 / k
            q = student.ppf(p, df=nu) * np.sqrt((nu - 2.0) / nu)
        out[c] = np.clip(mean[c] + sd[c] * q, 0.0, None)
    return out


def cornish_fisher_tables(mean, sd, skew, kurt, grid: np.ndarray = GRID) -> np.ndarray:
    """(C, G): r_c(t) = relu(mean_c + sd_c q_c(t)) with q_c the Cornish-Fisher quantile of a variable with skewness
    skew_c and excess kurtosis kurt_c: q = z + (z^2 - 1) g1/6 + (z^3 - 3z) g2/24 - (2z^3 - 5z) g1^2/36, z = t,
    made monotone by a running maximum."""
    mean, sd, skew, kurt = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (mean, sd, skew, kurt))
    z = grid[None, :]
    g1, g2 = skew[:, None], kurt[:, None]
    q = z + (z ** 2 - 1) * g1 / 6 + (z ** 3 - 3 * z) * g2 / 24 - (2 * z ** 3 - 5 * z) * g1 ** 2 / 36
    q = np.maximum.accumulate(q, axis=1)
    return np.clip(mean[:, None] + sd[:, None] * q, 0.0, None)


def _patch_energy(x: torch.Tensor, conv: nn.Conv2d, max_pos: int) -> torch.Tensor:
    """|patch|^2 at the subsampled positions of the conv's input x (B, C, H, W) -> (N,): the channel-summed squares
    box-filtered over the k x k window (zero padding, as unfold would), O(HW) instead of O(HW d)."""
    if conv.dilation != (1, 1):
        raise NotImplementedError("dilated convolutions")
    sq = (x.float() ** 2).sum(1, keepdim=True)
    kh, kw = conv.kernel_size
    e = torch.nn.functional.avg_pool2d(sq, (kh, kw), stride=conv.stride, padding=conv.padding, count_include_pad=True) * (kh * kw)
    return subsample(e, max_pos).reshape(-1)


class _Acc:
    def __init__(self):
        self.n, self.s1, self.s2 = 0, None, None

    def add(self, x):
        x = x.float()                                   # products on the device in float32 (MPS has no float64)
        s1, s2 = x.sum(0).cpu().double(), (x.T @ x).cpu().double()
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
    ap.add_argument("--patch-spectra", default=os.path.join(OUT, "..", "freeprob", "vgg16_bn_random_patch_spectra.npz"),
                    help="per-conv patch spectra (freeprob_check.py) for the participation ratio in the kurtosis prediction")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    model = api.load_model(a.model, seed=a.seed).to(a.device).eval()
    relus = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ReLU) and n.startswith("features")]
    mods = dict(model.named_modules())
    conv_of = {}                                                        # ReLU name -> the conv whose patches feed it (VGG: i-2)
    for n, _ in relus:
        i = int(n.split(".")[1]); cand = f"features.{i - 2}"
        if isinstance(mods.get(cand), nn.Conv2d):
            conv_of[n] = cand
    energy = {n: [0, 0.0, 0.0] for n in conv_of}                        # count, sum e, sum e^2
    pr_of = {}                                                          # conv name -> participation ratio of its patch covariance
    if a.patch_spectra and os.path.exists(a.patch_spectra):
        sp = np.load(a.patch_spectra)
        for k in sp.files:
            lam = np.clip(sp[k], 0, None); pr_of[k.replace("_", ".")] = float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-300))
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
    def conv_pre(name):
        def f(mod, inp):
            e = _patch_energy(inp[0].detach(), mod, a.max_pos).cpu().double()
            energy[name][0] += e.numel(); energy[name][1] += float(e.sum()); energy[name][2] += float((e ** 2).sum())
        return f
    hs = [m.register_forward_pre_hook(pre(n)) for n, m in relus] + [m.register_forward_hook(post(n)) for n, m in relus]
    hs += [mods[c].register_forward_pre_hook(conv_pre(n)) for n, c in conv_of.items()]
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
        zc = z - z.mean(0); kurt_c = (zc ** 4).mean(0) / (zc ** 2).mean(0) ** 2 - 3
        kurt = float(kurt_c.mean())
        kurt_pred = float("nan")
        if n in energy and energy[n][0] > 0 and conv_of[n] in pr_of:
            cnt, s1, s2 = energy[n]; kurt_pred = predicted_kurtosis(s1 / cnt, s2 / cnt, pr_of[conv_of[n]])
        Sm = copula_relu_cov(R, matched_tables(z.mean(0), z.std(0), kurt_c))            # parametric marginals, measured kurtosis
        skew_c = (zc ** 3).mean(0) / (zc ** 2).mean(0) ** 1.5
        Scf = copula_relu_cov(R, cornish_fisher_tables(z.mean(0), z.std(0), skew_c, kurt_c))   # four cumulants
        Sp = copula_relu_cov(R, matched_tables(z.mean(0), z.std(0), np.full(C, kurt_pred))) if np.isfinite(kurt_pred) else Sm
        ks = lambda M, t: k_star(torch.as_tensor(np.asarray(M)), (t,))[t]   # noqa: E731
        rows.append(dict(relu=n, C=C, meas=ks(Sa, 0.95), gauss=ks(Sg, 0.95), copula=ks(Sc, 0.95), matched=ks(Sm, 0.95), energy=ks(Sp, 0.95), cf=ks(Scf, 0.95),
                         meas_999=ks(Sa, 0.999), gauss_999=ks(Sg, 0.999), copula_999=ks(Sc, 0.999), matched_999=ks(Sm, 0.999), energy_999=ks(Sp, 0.999), cf_999=ks(Scf, 0.999),
                         kurtosis=kurt, kurtosis_pred=kurt_pred, skewness=float(skew_c.mean()), mean_over_sd=float(np.mean(np.abs(mz.numpy()) / s)), rows=len(z), seconds=round(tc)))
        r = rows[-1]
        print(f"{n:12s} C {C:4d}  k*(.95)/C meas {r['meas']/C:.3f} gauss {r['gauss']/C:.3f} copula {r['copula']/C:.3f} matched {r['matched']/C:.3f} energy {r['energy']/C:.3f}   "
              f"k*(.999)/C meas {r['meas_999']/C:.3f} gauss {r['gauss_999']/C:.3f} copula {r['copula_999']/C:.3f} matched {r['matched_999']/C:.3f} energy {r['energy_999']/C:.3f}   "
              f"cf {r['cf']/C:.3f}/{r['cf_999']/C:.3f}   kurt meas {kurt:+.2f} pred {kurt_pred:+.2f} skew {r['skewness']:+.2f}  ({tc:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(a.out, f"{stem(a.model)}_{a.inputs}_seed{a.seed}.csv")
    df.to_csv(path, index=False); print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
