"""Plan 9.50 (P9.118-P9.119; proofs T2.1-T2.3): a computable interval for the surrogate width of the activation step.

Given the convolution-step covariance G (C x C) and the pre-activation mean m, the Gaussian surrogate of the ReLU step is
Sigma_a^G = sum_j (mu_j mu_j^T) o R^{oj} (T1.2).  This script computes, per stored row,

  k_lo  = k_{tau/kappa}(G)                        Theorem T2.2 (kappa = 1 at zero mean)
  k_surr = k_tau(Sigma_a^G)                       the surrogate itself (Hermite to k = 24)
  k_hi  = min_k (k + |S|)                         Prop. T2.3: greedy cover by product vectors of R's top eigenvectors

and judges k_lo <= k_surr <= k_hi (a theorem; a violation is a bug) and whether the interval has content.

    cd code && python scripts/t2_bounds.py --results ../results
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relu_step_hermite import KMAX, hermite_coeffs  # noqa: E402
from t1_kernel_reading import VGG_MOMENTS, correlation, eig, kstar, surrogate  # noqa: E402

TAUS = (0.95, 0.999)
TOPK = {2: 48, 3: 20, 4: 12}        # eigenvectors of R used for the j-fold product vectors


def coefficients(mean_z, cov_z, kmax=KMAX):
    R, s = correlation(cov_z)
    mu = hermite_coeffs(np.asarray(mean_z, float), s, kmax)          # (C, kmax+1); mu[:, j] already divided by sqrt(j!)
    return R, s, mu


def kappa(mu, s, trG):
    """Theorem T2.2: [sum_j max_c (mu_j(c)/s_c)^2] / [sum_j |mu_j|^2 / tr G]  (j >= 1)."""
    lam = mu[:, 1:] / s[:, None]
    num = float((lam ** 2).max(axis=0).sum())
    den = float((mu[:, 1:] ** 2).sum() / trG)
    return num / den


def lower_bound(G, mu, s, tau):
    k = kappa(mu, s, float(np.trace(G)))
    return kstar(eig(G), tau / k), k


def lower_bound_per_term(G, R, mu, s, tau):
    """Exploratory (added after plan 10.54 was read): per-term Ky Fan.  By Lemma T2.1, S_k(Sigma_a^G) <= sum_j S_k(Lambda_j G Lambda_j),
    so f_k(Sigma_a^G) <= F(k) / tr Sigma_a^G with F(k) = sum_j S_k(Lambda_j G Lambda_j); k_lo2 = min{k : F(k)/tr >= tau}."""
    C = G.shape[0]
    F = np.zeros(C)
    for j in range(1, mu.shape[1]):
        lam = mu[:, j] / s
        if not np.any(lam):
            continue
        Mj = (lam[:, None] * G) * lam[None, :]
        F += np.cumsum(np.clip(np.sort(eig(Mj))[::-1], 0, None))
    total = float((mu[:, 1:] ** 2).sum())
    return int(np.searchsorted(F / total, tau, side="left") + 1)


def product_energies(R, mu, topk=TOPK):
    """Prop. T2.3: energies E_r = mult(r) * prod lambda * |mu_j o u_{i1} o ... o u_{ij}|^2 over multisets r of top eigenvectors, j = 2..4."""
    lam, U = np.linalg.eigh(R)
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0, None), U[:, order]
    out = []
    for j, K in topk.items():
        if mu.shape[1] <= j:
            continue
        K = min(K, R.shape[0])
        for r in itertools.combinations_with_replacement(range(K), j):
            h = mu[:, j].copy()
            pl = 1.0
            for i in r:
                h = h * U[:, i]
                pl *= lam[i]
            mult = math.factorial(j)
            for _, grp in itertools.groupby(r):
                mult //= math.factorial(len(list(grp)))
            out.append((j, mult * pl * float(h @ h)))
    return out


GENERIC_TOPK = {3: 64, 4: 24}


def generic_prediction(L, R, mu, tau, topk=GENERIC_TOPK):
    """Plan 9.51 (P9.120): spectrum-only prediction.  Product vectors taken orthogonal with generic-position energies
    E_r = mult(r) prod_t (lambda_{i_t}/C) |mu_j|^2 (all ordered tuples for j = 2; top eigenvalues for j = 3, 4); predicted spectrum =
    spec(L) u {E_r}; returns its tau-count (the total trace is the exact sum_j |mu_j|^2, so an unreachable tau counts every element)."""
    C = R.shape[0]
    lam = np.clip(np.sort(np.linalg.eigvalsh(R))[::-1], 0, None) / C
    parts = [np.clip(eig(L), 0, None)]
    if mu.shape[1] > 2:
        parts.append(float(mu[:, 2] @ mu[:, 2]) * np.outer(lam, lam).ravel())          # every ordered pair, multiplicity included
    for j, K in topk.items():
        if mu.shape[1] <= j or not np.any(mu[:, j]):
            continue
        K = min(K, C)
        w = float(mu[:, j] @ mu[:, j])
        for r in itertools.combinations_with_replacement(range(K), j):
            mult = math.factorial(j)
            for _, grp in itertools.groupby(r):
                mult //= math.factorial(len(list(grp)))
            parts.append(np.array([mult * w * float(np.prod(lam[list(r)]))]))
    e = np.sort(np.concatenate(parts))[::-1]
    total = float((mu[:, 1:] ** 2).sum())
    cum = np.cumsum(e) / total
    idx = int(np.searchsorted(cum, tau, side="left"))
    return min(idx + 1, len(e))


def upper_bound(L, energies, total, tau):
    """min over k of k + (number of greedy product vectors needed): S_k(L) + sum_S E >= tau * total."""
    e = np.sort(np.array([v for _, v in energies]))[::-1] if energies else np.zeros(0)
    cum_e = np.concatenate([[0.0], np.cumsum(e)])
    ev = np.clip(np.sort(eig(L))[::-1], 0, None)
    cum_l = np.concatenate([[0.0], np.cumsum(ev)])
    need = tau * total
    C = len(ev)
    best, best_k, best_s = C, C, 0           # rank C: the projection is the identity, so k_tau <= C always
    for k in range(len(cum_l)):
        rest = need - cum_l[k]
        if rest <= 0:
            n_s = 0
        else:
            idx = int(np.searchsorted(cum_e, rest, side="left"))
            if idx >= len(cum_e):
                continue
            n_s = idx
        if k + n_s < best:
            best, best_k, best_s = k + n_s, k, n_s
    return best, best_k, best_s


def analyse_row(mean_z, cov_z, mean_a, cov_a) -> dict:
    G = np.asarray(cov_z, float)
    R, s, mu = coefficients(mean_z, G)
    full, lin, _, _ = surrogate(mean_z, G)
    ef, eg = eig(full), eig(G)
    total = float((mu[:, 1:] ** 2).sum())
    energies = product_energies(R, mu)
    out = dict(C=G.shape[0], n_product_vectors=len(energies))
    for tau in TAUS:
        t = f"{tau:g}"
        klo, kap = lower_bound(G, mu, s, tau)
        klo2 = lower_bound_per_term(G, R, mu, s, tau)
        khi, k_part, s_part = upper_bound(lin, energies, total, tau)
        ks = kstar(ef, tau)
        kp = generic_prediction(lin, R, mu, tau)
        out.update({f"k_pred@{t}": kp, f"pred_err@{t}": (kp - ks) / ks,f"kappa": kap, f"k_G@{t}": kstar(eg, tau), f"k_lo@{t}": klo, f"k_lo2@{t}": klo2, f"lo2_ratio@{t}": klo2 / ks, f"lo2_ok@{t}": bool(klo2 <= ks),
                    f"k_lin@{t}": kstar(eig(lin), tau), f"k_surr@{t}": ks, f"k_hi@{t}": khi,
                    f"k_hi_lin@{t}": k_part, f"k_hi_prod@{t}": s_part, f"k_a@{t}": kstar(eig(np.asarray(cov_a, float)), tau),
                    f"sandwich_ok@{t}": bool(klo <= ks <= khi), f"hi_ratio@{t}": khi / ks, f"lo_ratio@{t}": klo / ks})
    return out


def load_rows(results: str) -> pd.DataFrame:
    rows = []
    p = os.path.join(results, VGG_MOMENTS)
    if os.path.exists(p):
        z = np.load(p)
        for key in z["keys"]:
            r = analyse_row(z[f"{key}__mz"], z[f"{key}__Sz"], z[f"{key}__ma"], z[f"{key}__Sa"])
            r.update(group="vgg", tag="vgg16_bn_random_s1_matched", layer=str(key))
            rows.append(r)
    for path in sorted(glob.glob(os.path.join(results, "relu_step", "*.npz"))):
        z = np.load(path)
        tag = os.path.basename(path).replace(".npz", "")
        names = sorted({k.split("::")[0] for k in z.files}, key=lambda n: int(n.split(".")[1]))
        for nme in names:
            r = analyse_row(z[f"{nme}::mean_y"], z[f"{nme}::cov_y"], z[f"{nme}::mean_z"], z[f"{nme}::cov_z"])
            r.update(group="stream_init" if "random" in tag else "stream_trained", tag=tag, layer=nme)
            rows.append(r)
    return pd.DataFrame(rows)


def judge(df: pd.DataFrame) -> tuple[dict, str]:
    lines = []
    sw = int(sum(df[f"sandwich_ok@{t:g}"].sum() for t in TAUS)); sw_n = len(df) * len(TAUS)
    lines.append(f"sanity: k_lo <= k_surr <= k_hi in {sw}/{sw_n} (theorem; a miss is a bug)")
    n118 = int((df["hi_ratio@0.95"] <= 1.3).sum()); n119 = int((df["lo_ratio@0.95"] >= 0.4).sum())
    lines.append(f"P9.118: k_hi <= 1.3 k_surr at tau=0.95 in {n118}/{len(df)} (need >= 20) -> {'holds' if n118 >= 20 else 'fails'}")
    lines.append(f"P9.119: k_lo >= 0.4 k_surr at tau=0.95 in {n119}/{len(df)} (need >= 20) -> {'holds' if n119 >= 20 else 'fails'}")
    if "k_pred@0.95" in df:
        n120 = int((df["pred_err@0.95"].abs() <= 0.15).sum())
        lines.append(f"P9.120: |k_pred - k_surr|/k_surr <= 0.15 at tau=0.95 in {n120}/{len(df)} (need >= 20) -> {'holds' if n120 >= 20 else 'fails'}; "
                     f"signed error median {df['pred_err@0.95'].median():+.2f} [{df['pred_err@0.95'].min():+.2f}, {df['pred_err@0.95'].max():+.2f}]; "
                     f"@0.999 median {df['pred_err@0.999'].median():+.2f}; "
                     + "; ".join(f"{g}: median {d['pred_err@0.95'].median():+.2f}" for g, d in df.groupby("group", sort=False)))
    if "k_lo2@0.95" in df:
        ok2 = int(sum(df[f"lo2_ok@{t:g}"].sum() for t in TAUS))
        lines.append(f"exploratory (post hoc, per-term Ky Fan lower bound k_lo2): valid in {ok2}/{sw_n}; k_lo2/k_surr@0.95 median {df['lo2_ratio@0.95'].median():.2f} "
                     f"[{df['lo2_ratio@0.95'].min():.2f}, {df['lo2_ratio@0.95'].max():.2f}]; >= 0.4 in {int((df['lo2_ratio@0.95'] >= 0.4).sum())}/{len(df)}; "
                     f"k_lo2 >= k_G in {int((df['k_lo2@0.95'] >= df['k_G@0.95']).sum())}/{len(df)}; k_lin <= k_surr in {int((df['k_lin@0.95'] <= df['k_surr@0.95']).sum())}/{len(df)}")
    for g, d in df.groupby("group", sort=False):
        lines.append(f"exploratory {g}: kappa median {d.kappa.median():.2f} [{d.kappa.min():.2f}, {d.kappa.max():.2f}]; "
                     f"hi/surr@0.95 median {d['hi_ratio@0.95'].median():.2f} [{d['hi_ratio@0.95'].min():.2f}, {d['hi_ratio@0.95'].max():.2f}]; "
                     f"lo/surr@0.95 median {d['lo_ratio@0.95'].median():.2f} [{d['lo_ratio@0.95'].min():.2f}, {d['lo_ratio@0.95'].max():.2f}]; "
                     f"hi/surr@0.999 median {d['hi_ratio@0.999'].median():.2f}; lo/surr@0.999 median {d['lo_ratio@0.999'].median():.2f}; "
                     f"k_hi covers measured k_a @0.95 in {int((d['k_hi@0.95'] >= d['k_a@0.95']).sum())}/{len(d)}; "
                     f"greedy split (lin part, product vectors) median ({d['k_hi_lin@0.95'].median():.0f}, {d['k_hi_prod@0.95'].median():.0f})")
    verdict = dict(sandwich=sw, sandwich_n=sw_n, p118_rows=n118, p118=bool(n118 >= 20), p119_rows=n119, p119=bool(n119 >= 20))
    if "k_pred@0.95" in df:
        verdict.update(p120_rows=int((df["pred_err@0.95"].abs() <= 0.15).sum()), p120=bool((df["pred_err@0.95"].abs() <= 0.15).sum() >= 20))
    return verdict, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "t2_bounds")
    os.makedirs(out, exist_ok=True)
    df = load_rows(a.results)
    df.to_csv(os.path.join(out, "rows.csv"), index=False)
    verdict, text = judge(df)
    json.dump(verdict, open(os.path.join(out, "judgement.json"), "w"), indent=1)
    open(os.path.join(out, "judgement.txt"), "w").write(text + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    show = ["group", "layer", "C", "kappa", "k_G@0.95", "k_lo@0.95", "k_lo2@0.95", "k_lin@0.95", "k_surr@0.95", "k_hi@0.95", "k_hi_lin@0.95", "k_hi_prod@0.95", "k_a@0.95", "k_pred@0.95",
            "lo2_ratio@0.95", "hi_ratio@0.95", "pred_err@0.95", "k_lo2@0.999", "k_surr@0.999", "k_hi@0.999"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
