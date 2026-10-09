"""Plan 9.49 (P9.116-P9.117; proofs T1.1-T1.4): the kernel-matrix reading of the activation step, on stored moments.

Every row is one ReLU: a pre-activation mean/covariance (m_z, Sigma_z) and the measured post-ReLU mean/covariance (m_a, Sigma_a).
For each row the script computes

  centring (Lemma T1.3)   k_tau(Sigma_a), k_tau(M) with M = Sigma_a + m_a m_a^T, beta = |m_a|^2 / tr Sigma_a, tau'' = tau - (1-tau) beta,
                          and the two-sided bound  k_tau''(Sigma_a) <= k_tau(M) <= k_tau''(Sigma_a) + 1   (exact; a violation is a bug)
  angles (Lemma T1.4)     RMS off-diagonal correlation of Sigma_z, r_eff_hat = 1 / mean(rho^2), and for a conv-step Gram G = W Sigma_p W^T
                          the moment estimate 1 / r_eff(Sigma_p) = (C tr G^2 / (tr G)^2 - 1) / (C + 1)
  surrogate (Prop. T1.2)  the Gaussian surrogate Sigma_a^G = sum_{k>=1} (mu_k mu_k^T) o R^{ok} (Hermite to k = 24, coefficients with the mean),
                          its linear part (k = 1 only), the Gaussian-equivalence form (linear + diagonal remainder), and the four widths.

Inputs (all stored, nothing is re-measured):
  results/apriori/relu/timm_vgg16_bn.tv_in1k_random_matched_seed1_b40_moments.npz   13 VGG-16 BN ReLUs (random seed 1, matched inputs)
  results/relu_step/*.npz                                                            residual-stream blocks (cov_y/mean_y -> cov_z/mean_z)

    cd code && python scripts/t1_kernel_reading.py --results ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relu_step_hermite import KMAX, hermite_coeffs  # noqa: E402

TAUS = (0.95, 0.999)
VGG_MOMENTS = "apriori/relu/timm_vgg16_bn.tv_in1k_random_matched_seed1_b40_moments.npz"
RHO_MIN = 0.15          # P9.116: near-orthogonal regime fails when the RMS correlation is at least this
REFF_MAX = 45.0         # equivalently r_eff <= 1 / RHO_MIN^2


def kstar(eigs: np.ndarray, tau: float) -> int:
    """min{k : S_k / tr >= tau}, the counting convention of apriori_relu_check.k_star (tau <= 0 gives 1)."""
    e = np.clip(np.sort(np.asarray(eigs, float))[::-1], 0, None)
    tot = e.sum()
    if tot <= 0:
        return 0
    cum = np.cumsum(e) / tot
    return int((cum < tau).sum()) + 1


def eig(S: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (S + S.T))


def correlation(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    R = cov / np.outer(s, s)
    return np.clip(R, -1.0, 1.0), s


def surrogate(mean_z: np.ndarray, cov_z: np.ndarray, kmax: int = KMAX):
    """Gaussian surrogate of the ReLU step (Prop. T1.2 with the mean): returns (Sigma_a^G, linear part, Gaussian-equivalence form, m^G)."""
    R, s = correlation(cov_z)
    mu = hermite_coeffs(np.asarray(mean_z, float), s, kmax)          # (C, kmax+1), normalised Hermite basis (1/sqrt(k!) included)
    lin = np.outer(mu[:, 1], mu[:, 1]) * R
    full = lin.copy()
    Rk = R.copy()
    for k in range(2, kmax + 1):
        Rk = Rk * R
        full += np.outer(mu[:, k], mu[:, k]) * Rk
    rem_diag = (mu[:, 2:] ** 2).sum(axis=1)
    ge = lin + np.diag(rem_diag)
    return full, lin, ge, mu[:, 0]


def centring(cov_a: np.ndarray, mean_a: np.ndarray, tau: float) -> dict:
    """Lemma T1.3 on one row."""
    M = cov_a + np.outer(mean_a, mean_a)
    tr = float(np.trace(cov_a))
    beta = float(mean_a @ mean_a) / tr
    tau2 = tau - (1.0 - tau) * beta
    ka, kM, ka2 = kstar(eig(cov_a), tau), kstar(eig(M), tau), kstar(eig(cov_a), tau2)
    return dict(k_a=ka, k_M=kM, beta=beta, tau2=tau2, k_a_tau2=ka2, lemma_ok=bool(ka2 <= kM <= ka2 + 1))


def angles(cov_z: np.ndarray) -> dict:
    R, _ = correlation(cov_z)
    C = R.shape[0]
    off = R[~np.eye(C, dtype=bool)]
    m2 = float(np.mean(off ** 2))
    G2 = float(np.trace(cov_z @ cov_z))
    G1 = float(np.trace(cov_z))
    inv_reff_p = (C * G2 / G1 ** 2 - 1.0) / (C + 1.0)
    return dict(rms_rho=float(np.sqrt(m2)), mean_rho2=m2, reff_hat=1.0 / m2 if m2 > 0 else np.inf,
                reff_p_hat=(1.0 / inv_reff_p) if inv_reff_p > 0 else np.inf)


def analyse_row(mean_z, cov_z, mean_a, cov_a, stored_surrogate=None) -> dict:
    full, lin, ge, m_g = surrogate(mean_z, cov_z)
    out = dict(C=cov_z.shape[0])
    out.update(angles(cov_z))
    ef, el, eg, ea = eig(full), eig(lin), eig(ge), eig(cov_a)
    for tau in TAUS:
        t = f"{tau:g}"
        c = centring(cov_a, mean_a, tau)
        out.update({f"k_a@{t}": c["k_a"], f"k_M@{t}": c["k_M"], f"beta": c["beta"], f"tau2@{t}": c["tau2"],
                    f"k_a_tau2@{t}": c["k_a_tau2"], f"lemma_ok@{t}": c["lemma_ok"],
                    f"k_lin@{t}": kstar(el, tau), f"k_surr@{t}": kstar(ef, tau), f"k_ge@{t}": kstar(eg, tau)})
        out[f"order_ok@{t}"] = bool(out[f"k_lin@{t}"] <= out[f"k_a@{t}"] <= out[f"k_surr@{t}"] <= out[f"k_ge@{t}"])
    out["beta_over_const"] = out["beta"] * (np.pi - 1.0)
    out["snr_z"] = float(np.median(np.abs(mean_z) / np.sqrt(np.clip(np.diag(cov_z), 1e-300, None))))
    if stored_surrogate is not None:
        out["surr_xcheck"] = float(np.linalg.norm(full - stored_surrogate) / np.linalg.norm(stored_surrogate))
    return out


def load_rows(results: str) -> pd.DataFrame:
    rows = []
    p = os.path.join(results, VGG_MOMENTS)
    if os.path.exists(p):
        z = np.load(p)
        for key in z["keys"]:
            r = analyse_row(z[f"{key}__mz"], z[f"{key}__Sz"], z[f"{key}__ma"], z[f"{key}__Sa"], z[f"{key}__Sp"])
            r.update(group="vgg", tag="vgg16_bn_random_s1_matched", layer=str(key), n=int(z[f"{key}__n"]))
            rows.append(r)
    for path in sorted(glob.glob(os.path.join(results, "relu_step", "*.npz"))):
        z = np.load(path)
        tag = os.path.basename(path).replace(".npz", "")
        names = sorted({k.split("::")[0] for k in z.files}, key=lambda n: int(n.split(".")[1]))
        for nme in names:
            r = analyse_row(z[f"{nme}::mean_y"], z[f"{nme}::cov_y"], z[f"{nme}::mean_z"], z[f"{nme}::cov_z"])
            r.update(group="stream_init" if "random" in tag else "stream_trained", tag=tag, layer=nme, n=-1)
            rows.append(r)
    return pd.DataFrame(rows)


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(a.corr(b))


def judge(df: pd.DataFrame) -> tuple[dict, str]:
    vgg = df[df.group == "vgg"]
    init = df[df.group == "stream_init"]
    tr = df[df.group == "stream_trained"]
    lines = []
    lemma = int(sum(df[f"lemma_ok@{t:g}"].sum() for t in TAUS))
    lemma_n = len(df) * len(TAUS)
    xck = vgg["surr_xcheck"] if "surr_xcheck" in vgg else pd.Series(dtype=float)
    xck_ok = int((xck <= 0.02).sum())
    lines.append(f"sanity: Lemma T1.3 bounds hold {lemma}/{lemma_n}; surrogate cross-check (Hermite vs stored Gauss-Hermite, rel. Frobenius <= 0.02) {xck_ok}/{len(xck)}"
                 + (f" (max {xck.max():.4f})" if len(xck) else ""))
    n_rho = int((df.rms_rho >= RHO_MIN).sum())
    vgg_reff = int((vgg.reff_p_hat <= REFF_MAX).sum())
    p116 = n_rho >= 25 and vgg_reff == len(vgg)
    lines.append(f"P9.116: RMS rho >= {RHO_MIN} in {n_rho}/{len(df)} rows (need >= 25); VGG r_eff_hat(Sigma_p) <= {REFF_MAX:g} in {vgg_reff}/{len(vgg)} (need all) -> {'holds' if p116 else 'fails'}")
    ov = int(vgg["order_ok@0.95"].sum())
    oi = int(init["order_ok@0.95"].sum())
    p117 = ov >= 11 and oi >= 10
    lines.append(f"P9.117: linear <= measured <= surrogate <= GE at tau=0.95 in VGG {ov}/{len(vgg)} (need >= 11), init streams {oi}/{len(init)} (need >= 10) -> {'holds' if p117 else 'fails'}; "
                 f"trained streams (described only) {int(tr['order_ok@0.95'].sum())}/{len(tr)}")
    expl = {}
    if len(vgg) >= 5:
        expl["spearman_rho2_vs_ge_minus_surr"] = spearman(vgg.mean_rho2, (vgg["k_ge@0.95"] - vgg["k_surr@0.95"]) / vgg.C)
        lines.append(f"exploratory: Spearman(mean rho^2, (k_GE - k_surr)/C) over {len(vgg)} VGG layers = {expl['spearman_rho2_vs_ge_minus_surr']:+.2f}")
    for g, d in (("vgg", vgg), ("stream_init", init), ("stream_trained", tr)):
        if len(d):
            lines.append(f"exploratory {g}: beta/(1/(pi-1)) median {d.beta_over_const.median():.2f} [{d.beta_over_const.min():.2f}, {d.beta_over_const.max():.2f}]; "
                         f"tau''@0.95 median {d['tau2@0.95'].median():.3f}; k_M - k_a @0.95 median {(d['k_M@0.95'] - d['k_a@0.95']).median():+.0f}; "
                         f"RMS rho median {d.rms_rho.median():.3f} [{d.rms_rho.min():.3f}, {d.rms_rho.max():.3f}]; |mean|/std of z median {d.snr_z.median():.2f}")
    verdict = dict(lemma_ok=lemma, lemma_n=lemma_n, xcheck_ok=xck_ok, xcheck_n=int(len(xck)), p116_rows=n_rho, p116_vgg_reff=vgg_reff, p116=bool(p116),
                   p117_vgg=ov, p117_init=oi, p117=bool(p117), **expl)
    return verdict, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "t1_kernel")
    os.makedirs(out, exist_ok=True)
    df = load_rows(a.results)
    cols = ["group", "tag", "layer", "C", "n", "snr_z", "beta", "beta_over_const", "rms_rho", "reff_hat", "reff_p_hat", "surr_xcheck"] + \
           [f"{k}@{t:g}" for t in TAUS for k in ("k_lin", "k_a", "k_surr", "k_ge", "k_M", "tau2", "k_a_tau2", "lemma_ok", "order_ok")]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(os.path.join(out, "rows.csv"), index=False)
    verdict, text = judge(df)
    with open(os.path.join(out, "judgement.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    with open(os.path.join(out, "judgement.txt"), "w") as f:
        f.write(text + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    show = ["group", "layer", "C", "snr_z", "beta", "rms_rho", "reff_p_hat", "k_lin@0.95", "k_a@0.95", "k_surr@0.95", "k_ge@0.95", "k_M@0.95", "tau2@0.95", "k_a_tau2@0.95", "order_ok@0.95"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
