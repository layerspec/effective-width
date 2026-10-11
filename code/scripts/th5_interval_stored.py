"""Theorem M.2 / Proposition Th5' (notes/proofs-2026-09-29.md, sections M and Th5) on stored input spectra.

Plan section 9.55. For every dense convolution of the ten random families in results/freeprob/, the law of the
convolution-step covariance W Sigma_p W^T depends on the data only through spec(Sigma_p) (Proposition S6). From the
stored spectrum we draw the law, build the Monte Carlo tail curve hbar_k = E R_k / E T^2, bound sd(R_k) by the
Poincare inequality (first inequality of Th5'(1): Var R_k <= 4 E tr(M_perp Sigma M_perp^T), c_P = 1 for Gaussian
kernels), take q = sd(T^2)/E T^2 exactly, and form the Chebyshev interval [k_minus, k_plus] of Th5'(4) at u = 4.5
(guarantee 1 - 4/u^2 = 0.802). The measured k*(tau) of the drawn kernel (results/freeprob/*_random.csv) is one draw
of that law, so the interval must cover it at least 80% of the time (P9.126); whether the interval is informative is
P9.127. Two reference intervals are also written: one with the Monte Carlo sd of R_k instead of the bound (not a
theorem) and one with the looser lambda_1 bound (second inequality of Th5'(1)).

Usage (from code/):  python scripts/th5_interval_stored.py --results ../results [--draws 200] [--families vgg16_bn]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.decomposition import kstar  # noqa: E402

TAUS = (0.95, 0.999)
U = 4.5
C_P = 1.0  # Gaussian Poincare constant


def npz_key(layer: str) -> str:
    """freeprob_check.py stores the spectrum of layer 'a.b' under the key 'a_b'."""
    return layer.replace(".", "_")


def draw_tail_statistics(lam: np.ndarray, C: int, draws: int, rng: np.random.Generator):
    """Sample W Sigma_p W^T for standard Gaussian W (C x d). Returns, per draw, the eigenvalues (descending, padded
    to r = min(C, d)), the tail sums R_k (k = 0..r), the Poincare tail quantity G2_k = 4 sum_{i>k} e_i <v_i^2, lam>
    (k = 0..r), T^2 and k*(tau) for each tau in TAUS."""
    lam = np.clip(np.asarray(lam, dtype=float), 0, None)
    d = len(lam)
    r = min(C, d)
    root = np.sqrt(lam)
    E = np.empty((draws, r))
    R = np.empty((draws, r + 1))
    G2 = np.empty((draws, r + 1))
    K = np.empty((draws, len(TAUS)), dtype=int)
    for t in range(draws):
        W = rng.standard_normal((C, d))
        _, s, vt = np.linalg.svd(W * root[None, :], full_matrices=False)
        e = s ** 2  # descending
        w = e * (vt ** 2 @ lam)  # sigma_i^2 * sum_j lam_j v_i(j)^2
        rev_e = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]])  # R_k = sum_{i>k} e_i, k = 0..r
        rev_w = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
        E[t] = e
        R[t] = rev_e
        G2[t] = 4.0 * rev_w
        K[t] = [kstar(e, tau) for tau in TAUS]
    return E, R, G2, R[:, 0], K


def chebyshev_interval(hbar: np.ndarray, r_k: np.ndarray, q: float, u: float, tau: float, r: int):
    """Th5'(4). hbar, r_k indexed k = 0..r. Returns (k_minus, k_plus); k_plus = r when the upper curve never drops
    below 1 - tau (vacuous), and also when u q >= 1."""
    ks = np.arange(1, r + 1)
    h = hbar[1:]
    rk = r_k[1:]
    lower = h * (1.0 - u * rk) / (1.0 + u * q)
    hit = np.nonzero(lower <= 1.0 - tau)[0]
    k_minus = int(ks[hit[0]]) if hit.size else r
    if 1.0 - u * q <= 0:
        return k_minus, r
    upper = h * (1.0 + u * rk) / (1.0 - u * q)
    hit = np.nonzero(upper <= 1.0 - tau)[0]
    k_plus = int(ks[hit[0]]) if hit.size else r
    return k_minus, k_plus


def interval_from_spectrum(lam, C: int, u: float = U, draws: int = 200, seed: int = 0) -> dict:
    lam = np.clip(np.asarray(lam, dtype=float), 0, None)
    rng = np.random.default_rng(seed)
    E, R, G2, T2, K = draw_tail_statistics(lam, C, draws, rng)
    r = E.shape[1]
    ER = R.mean(0)
    ET2 = T2.mean()
    hbar = ER / ET2
    with np.errstate(divide="ignore", invalid="ignore"):
        r_poinc = np.where(ER > 0, np.sqrt(C_P * G2.mean(0)) / ER, 0.0)  # Var R_k <= c_P E G2_k
        r_lam = np.where(ER > 0, np.sqrt(4.0 * C_P * lam.max() / ER), 0.0)  # Var R_k <= 4 c_P lambda_1 E R_k
        r_mc = np.where(ER > 0, R.std(0) / ER, 0.0)
    q_exact = float(np.sqrt(2.0 * np.sum(lam ** 2) / (C * lam.sum() ** 2)))
    q_mc = float(T2.std() / ET2)
    out = dict(r=r, q=q_exact, q_mc=q_mc)
    for j, tau in enumerate(TAUS):
        kmc = K[:, j]
        km, kp = chebyshev_interval(hbar, r_poinc, q_exact, u, tau, r)
        km_mc, kp_mc = chebyshev_interval(hbar, r_mc, q_mc, u, tau, r)
        km_l, kp_l = chebyshev_interval(hbar, r_lam, q_exact, u, tau, r)
        kk = int(np.clip(round(kmc.mean()), 1, r))
        out[tau] = dict(mc_mean=float(kmc.mean()), mc_sd=float(kmc.std()), k_minus=km, k_plus=kp,
                        k_minus_mc=km_mc, k_plus_mc=kp_mc, k_minus_lam=km_l, k_plus_lam=kp_l,
                        r_poinc_at_k=float(r_poinc[kk]), r_mc_at_k=float(r_mc[kk]), hbar_at_k=float(hbar[kk]))
    return out


def analyse_family(spectra_path: str, csv_path: str, draws: int, seed: int, limit: int | None = None) -> pd.DataFrame:
    fam = os.path.basename(csv_path).replace("_random.csv", "")
    z = np.load(spectra_path)
    df = pd.read_csv(csv_path)
    rows = []
    for i, rr in enumerate(df.itertuples(index=False)):
        if limit is not None and i >= limit:
            break
        key = npz_key(rr.layer)
        if key not in z:
            print(f"[warn] {fam}: no spectrum for {rr.layer}", file=sys.stderr)
            continue
        lam = z[key]
        C, rmax = int(rr.C), int(rr.r_max)
        res = interval_from_spectrum(lam, C, draws=draws, seed=seed + i)
        for tau in TAUS:
            o = res[tau]
            meas = int(getattr(rr, f"measured_{str(tau)[2:]}"))
            rows.append(dict(family=fam, layer=rr.layer, C=C, d=int(rr.d), r_max=rmax, n_over_d=float(rr.n_over_d), tau=tau,
                             measured=meas, mc_mean=o["mc_mean"], mc_sd=o["mc_sd"],
                             k_minus=o["k_minus"], k_plus=o["k_plus"], k_minus_mc=o["k_minus_mc"], k_plus_mc=o["k_plus_mc"],
                             k_minus_lam=o["k_minus_lam"], k_plus_lam=o["k_plus_lam"],
                             q=res["q"], q_mc=res["q_mc"], r_poinc_at_k=o["r_poinc_at_k"], r_mc_at_k=o["r_mc_at_k"],
                             inside=int(o["k_minus"] <= meas <= o["k_plus"]),
                             inside_mc=int(o["k_minus_mc"] <= meas <= o["k_plus_mc"]),
                             inside_lam=int(o["k_minus_lam"] <= meas <= o["k_plus_lam"]),
                             halfwidth_over_rmax=(o["k_plus"] - o["k_minus"]) / 2.0 / rmax,
                             halfwidth_lam_over_rmax=(o["k_plus_lam"] - o["k_minus_lam"]) / 2.0 / rmax,
                             degenerate=int(o["k_plus"] >= rmax), degenerate_lam=int(o["k_plus_lam"] >= rmax)))
    return pd.DataFrame(rows)


def judge(rows: pd.DataFrame, u: float = U) -> tuple[dict, str]:
    lines = [f"plan 9.55 judgement (u = {u}, guarantee 1 - 4/u^2 = {1 - 4 / u ** 2:.3f}); layers = {rows.layer.nunique()} x families = {rows.family.nunique()}"]
    out = {}
    for tau in TAUS:
        s = rows[rows.tau == tau]
        cov = s.inside.mean()
        out[f"coverage@{tau}"] = float(cov)
        out[f"coverage_mc@{tau}"] = float(s.inside_mc.mean())
        out[f"coverage_lam@{tau}"] = float(s.inside_lam.mean())
        out[f"median_halfwidth_over_rmax@{tau}"] = float(s.halfwidth_over_rmax.median())
        out[f"frac_nondegenerate@{tau}"] = float(1 - s.degenerate.mean())
        out[f"frac_nondegenerate_lam@{tau}"] = float(1 - s.degenerate_lam.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ((s.k_plus - s.k_minus) / 2.0 / s.mc_sd.replace(0, np.nan)).median()
        out[f"median_halfwidth_over_mcsd@{tau}"] = float(ratio)
        sub = s[s.n_over_d >= 50]
        out[f"coverage_n_over_d50@{tau}"] = float(sub.inside.mean()) if len(sub) else float("nan")
        lines.append(f"tau={tau}: coverage {cov:.3f} ({int(s.inside.sum())}/{len(s)}); MC-sd interval {out[f'coverage_mc@{tau}']:.3f}; "
                     f"lambda_1 interval {out[f'coverage_lam@{tau}']:.3f}; median halfwidth/r_max {out[f'median_halfwidth_over_rmax@{tau}']:.3f}; "
                     f"non-degenerate {out[f'frac_nondegenerate@{tau}']:.3f} (lambda_1 version {out[f'frac_nondegenerate_lam@{tau}']:.3f}); "
                     f"median halfwidth/MC sd(k*) {ratio:.2f}; n/d>=50 subset coverage {out[f'coverage_n_over_d50@{tau}']:.3f}")
        for fam, g in s.groupby("family"):
            lines.append(f"   {fam:32s} L={len(g):3d} coverage {g.inside.mean():.3f}  median halfwidth/r_max {g.halfwidth_over_rmax.median():.3f}  "
                         f"non-degenerate {1 - g.degenerate.mean():.2f}  median |measured - mc_mean| {np.median(np.abs(g.measured - g.mc_mean)):.1f} ch")
    p126 = all(out[f"coverage@{tau}"] >= 0.80 for tau in TAUS)
    p127 = out["median_halfwidth_over_rmax@0.95"] <= 0.10 and out["frac_nondegenerate@0.95"] >= 0.80
    out["P9.126"] = bool(p126)
    out["P9.127"] = bool(p127)
    lines.append(f"P9.126 (coverage >= 0.80 at both tau): {'HOLDS' if p126 else 'FAILS'}")
    lines.append(f"P9.127 (tau=0.95: median halfwidth <= 0.10 r_max and k_plus < r_max in >= 80%): {'HOLDS' if p127 else 'FAILS'}")
    return out, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=None, help="substrings of family names to keep")
    ap.add_argument("--limit", type=int, default=None, help="layers per family (smoke test)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = a.out or os.path.join(a.results, "th5_interval")
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for spec in sorted(glob.glob(os.path.join(a.results, "freeprob", "*_random_patch_spectra.npz"))):
        csv_path = spec.replace("_patch_spectra.npz", ".csv")
        fam = os.path.basename(csv_path).replace("_random.csv", "")
        if a.families and not any(f in fam for f in a.families):
            continue
        if not os.path.exists(csv_path):
            print(f"[warn] missing {csv_path}", file=sys.stderr)
            continue
        t0 = time.time()
        df = analyse_family(spec, csv_path, a.draws, a.seed, a.limit)
        frames.append(df)
        s95 = df[df.tau == 0.95]
        print(f"{fam}: {len(s95)} layers, coverage@0.95 {s95.inside.mean():.3f}, median halfwidth/r_max {s95.halfwidth_over_rmax.median():.3f}, {time.time() - t0:.0f} s", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    verdict, text = judge(rows)
    with open(os.path.join(out_dir, "judgement.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    with open(os.path.join(out_dir, "judgement.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
