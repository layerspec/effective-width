"""Free-probability check of the untrained profile (2026-09-26; hostile review 09-25 item 1, "make (i) a theorem").

At initialisation every dense convolution has Sigma_out = W Sigma_patch W^T with W's entries i.i.d. Gaussian
(Kaiming), so the empirical spectral distribution of Sigma_out is, in the proportional limit C, d -> infinity,
the free multiplicative convolution of the patch spectrum with a Marchenko-Pastur law: with y = d / C and H the
spectral distribution of Sigma_patch (d eigenvalues, zeros included), the Stieltjes transform of the law of
B = (1/C) W~ Sigma_patch W~^T solves Silverstein's equation
        m(z) = -1 / ( z - y * int t dH(t) / (1 + t m(z)) ),
and k*(tau) of the conv output is the number of eigenvalues above the level below which a fraction 1 - tau of
the eigenvalue mass lies.  This makes the level of the untrained profile a computable function of the input
spectrum, layer by layer -- the statement the a-priori section lacked.  The script solves the equation
numerically, checks it against a Monte Carlo over W draws, and compares both with the measured output width
of a random-initialised network whose patch covariances are accumulated by layerspec.api.decompose.

    python scripts/freeprob_check.py --model timm:vgg16_bn.tv_in1k_random --data ../data/imagenet_val_6400 \\
        --limit 1600 --device mps --out ../results/freeprob/vgg16_bn.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.decomposition import kstar   # noqa: E402


def kstar_from_spectrum(lam, tau: float) -> int:
    return int(kstar(np.asarray(lam, dtype=float), tau))


def _silverstein_density(lam: np.ndarray, C: int, n_grid: int = 1500, eta_rel: float = 1e-3, iters: int = 60):
    """Density of the law of (1/C) W~ diag(lam) W~^T (C x C, W~ i.i.d. N(0,1), lam of length d) on a log grid,
    from Silverstein's equation  m = -1 / (z - y * int t dH(t) / (1 + t m)),  y = d / C, H = ESD of diag(lam).
    Solved by Newton's method on g(m) = m + 1 / (z - y I(m)), vectorised over the grid, warm-started from a
    damped fixed-point pass (the fixed point alone converges too slowly near an isolated top eigenvalue)."""
    lam = np.asarray(lam, dtype=float)
    d = len(lam)
    y = d / C
    t = lam[lam > 0]
    if len(t) == 0:
        return np.array([1.0]), np.array([0.0])
    hi = t.max() * (1 + np.sqrt(y)) ** 2 * 1.5 * (d / max(len(t), 1))
    lo = max(t.min() * 1e-3, hi * 1e-9)
    x = np.geomspace(lo, hi, n_grid)
    z = x + 1j * eta_rel * x
    w = 1.0 / d

    def I_and_dI(m):
        den = 1.0 + t[None, :] * m[:, None]
        I = w * np.sum(t[None, :] / den, axis=1)
        dI = -w * np.sum(t[None, :] ** 2 / den ** 2, axis=1)
        return I, dI

    m = -1.0 / z
    for _ in range(200):                                  # damped fixed point: a safe start
        I, _ = I_and_dI(m)
        m = 0.5 * m + 0.5 * (-1.0 / (z - y * I))
    for _ in range(iters):                                # Newton on g(m) = m + 1/(z - y I(m))
        I, dI = I_and_dI(m)
        den = z - y * I
        g = m + 1.0 / den
        dg = 1.0 + y * dI / den ** 2
        step = g / dg
        m_new = m - step
        bad = m_new.imag <= 0                             # keep the physical branch
        m = np.where(bad, m - 0.3 * step, m_new)
        if np.max(np.abs(step)) < 1e-12 * max(1.0, np.max(np.abs(m))):
            break
    f = np.maximum(m.imag / np.pi, 0.0)
    return x, f


def mp_eigenvalues(lam, C: int) -> np.ndarray:
    """The C 'typical' eigenvalues of MP_{d/C} x law(lam): the quantiles of the numerical density's counting
    function at (i - 1/2) / n for the n nonzero eigenvalues (an isolated eigenvalue appears as a narrow bump
    whose count is 1/C, so quantiles locate it while the smeared density's moments would not); the rest are 0."""
    lam = np.asarray(lam, dtype=float)
    n = min(C, int(np.sum(lam > 0)))
    if n == 0:
        return np.zeros(C)
    x, f = _silverstein_density(lam, C)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
    cum = cum / cum[-1]
    q = (np.arange(n) + 0.5) / n
    ev = np.interp(q, cum, x)
    return np.sort(np.concatenate([ev, np.zeros(C - n)]))[::-1]


def mp_convolve_kstar(lam, C: int, tau: float) -> int:
    """k*(tau) of the free multiplicative convolution MP_{d/C} x law(lam)."""
    return kstar_from_spectrum(mp_eigenvalues(lam, C), tau)


def mc_kstar(lam, C: int, tau: float, draws: int = 8, seed: int = 0) -> list[int]:
    """Monte Carlo: k*(tau) of W~ diag(lam) W~^T over `draws` Gaussian W~."""
    lam = np.asarray(lam, dtype=float)
    rng = np.random.default_rng(seed)
    out = []
    root = np.sqrt(np.clip(lam, 0, None))
    for _ in range(draws):
        W = rng.standard_normal((C, len(lam)))
        s = np.linalg.svd(W * root[None, :], compute_uv=False)
        out.append(kstar_from_spectrum(s ** 2, tau))
    return out


def check_model(name: str, data: str, limit: int, device: str, positions: int, seed: int, taus=(0.95, 0.999)) -> pd.DataFrame:
    from layerspec import api
    model = api.load_model(name, seed=seed)
    loader = api.image_loader(data, batch_size=32, workers=2, image_size=224, limit=limit, seed=seed)
    dec = api.decompose(model, loader, device=device, positions=positions, taus=taus, seed=seed)
    rows, spectra = [], {}
    for _, r in dec.table.iterrows():
        Sig = dec.patch_covariance[r.layer]
        lam = np.sort(np.linalg.eigvalsh((Sig + Sig.T) / 2))[::-1].clip(0, None)
        spectra[r.layer] = lam
        C, d, rmax = int(r.C_out), int(r.d), int(r.r_max)
        row = dict(layer=r.layer, depth_index=int(r.depth_index), C=C, d=d, r_max=rmax, n_over_d=float(r.n_over_d))
        for tau in taus:
            tk = f"{tau:g}".replace("0.", "")
            meas = float(r[f"out_{tau:g}".replace("out_0.", "out_0.")]) if f"out_{tau:g}" in r else None
            key = [c for c in dec.table.columns if c.startswith("out_") and abs(float(c[4:]) - tau) < 1e-9][0]
            meas_k = float(r[key]) * rmax
            pred_k = mp_convolve_kstar(lam, C, tau)
            mc = mc_kstar(lam, C, tau, draws=4, seed=seed)
            row.update({f"measured_{tk}": meas_k, f"silverstein_{tk}": pred_k, f"mc_{tk}": float(np.mean(mc)),
                        f"mc_sd_{tk}": float(np.std(mc)), f"data_{tk}": kstar_from_spectrum(lam, tau)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)
    df.attrs["spectra"] = spectra
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="../data/imagenet_val_6400")
    ap.add_argument("--limit", type=int, default=1600)
    ap.add_argument("--device", default=None)
    ap.add_argument("--positions", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = check_model(a.model, a.data, a.limit, a.device, a.positions, a.seed)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.round(1).to_string(index=False))
    for tk in ("95", "999"):
        dm = (df[f"silverstein_{tk}"] - df[f"measured_{tk}"]).abs()
        dc = (df[f"mc_{tk}"] - df[f"measured_{tk}"]).abs()
        rel = dm / df.r_max
        print(f"tau=0.{tk}: |Silverstein - measured| median {dm.median():.1f} channels (max {dm.max():.0f}); "
              f"in k*/r_max median {rel.median():.3f} (max {rel.max():.3f}); |MC - measured| median {dc.median():.1f}; "
              f"Spearman(pred, measured) {df[f'silverstein_{tk}'].corr(df[f'measured_{tk}'], method='spearman'):+.3f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        np.savez_compressed(a.out.replace(".csv", "_patch_spectra.npz"), **{k.replace(".", "_"): v for k, v in df.attrs["spectra"].items()})
        print("wrote", a.out, "and the patch spectra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
