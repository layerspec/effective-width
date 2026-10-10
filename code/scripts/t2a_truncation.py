"""Plan 9.54 (P9.124-P9.125): truncation of the diagonal tuples under the correlation constraint (model B'' of R^{o2}).

Lemma T2a.1 (exact): R a correlation matrix, R = sum_i lam_i u_i u_i^T.  Then lam_i u_i(c)^2 <= R_cc = 1, hence
    ||u_i o u_i'||^2 <= 1 / lam_i   and   lam_i lam_i' ||u_i o u_i'||^2 <= min(lam_i, lam_i').
The Haar value 3 lam_i^2 / C of model B' violates it when lam_i > C / 3.

Model B'' (constrained Haar, first-order conditioning): for fixed c the row (u_i(c)^2)_i obeys sum_i u_i(c)^2 = 1 and
sum_i lam_i u_i(c)^2 = 1.  Treating the Haar fluctuations x_i = u_i(c)^2 - 1/C as isotropic Gaussian with variance
sigma^2 = 2 (C - 1) / (C^2 (C + 2)) (which already encodes the row constraint: the Haar row is a uniform unit vector, so the
x_i live in the hyperplane sum_i x_i = 0 with covariance ~ sigma^2 (C/(C-1)) (I - 11^T/C)) and conditioning on the remaining
constraint sum_i (lam_i - 1) x_i = 0 gives Var(u_i(c)^2) ~ sigma^2 (1 - pi_i), pi_i = (C/(C-1)) (lam_i - 1)^2 / (tr R^2 - C), so
    a_i := C ||u_i o u_i||^2  ~  p_i := 1 + C^2 sigma^2 (1 - pi_i)        (Haar: 3C/(C+2); rank one: 1).
B'' = B' with ||r_i||^2 = 2/C replaced by (p_i - 1)/C in the (K+1) block and in the beyond-K lump.

    cd code && python scripts/t2a_truncation.py --results ../results [--draws 4 --draws-large 2]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_kernel_reading import VGG_MOMENTS, eig, kstar, surrogate  # noqa: E402
from t2_bounds import coefficients  # noqa: E402
from t2a_free_model import JMAX_COMP, compress, compressed_powers  # noqa: E402

TAUS = (0.95, 0.999)
KTOP = 48


def haar_value(C: int) -> float:
    """E[C sum_c u(c)^4] for a uniform unit vector in R^C."""
    return 3.0 * C / (C + 2)


def predicted_tuple_norm(lamR: np.ndarray) -> np.ndarray:
    """p_i = C ||u_i o u_i||^2 under the constrained-Haar model (plan 9.54)."""
    lamR = np.asarray(lamR, float)
    C = len(lamR)
    sigma2 = 2.0 * (C - 1) / (C ** 2 * (C + 2))
    denom = float((lamR ** 2).sum()) - C
    quad = (lamR - 1.0) ** 2
    ratio = np.zeros_like(lamR) if denom <= 1e-12 else quad / denom
    pi = (C / (C - 1.0)) * ratio
    return 1.0 + C ** 2 * sigma2 * np.clip(1.0 - pi, 0.0, None)


def actual_tuple_norm(R: np.ndarray, K: int = KTOP) -> tuple[np.ndarray, np.ndarray]:
    """(lam_i, a_i = C sum_c u_i(c)^4) for the top K eigenvectors of R."""
    w, V = np.linalg.eigh(R)
    order = np.argsort(w)[::-1][:K]
    lam = np.clip(w[order], 0, None)
    U = V[:, order]
    return lam, R.shape[0] * (U ** 4).sum(axis=0)


def hadamard_square_variants(rng, lamR, K=KTOP):
    """Model B' and model B'' of spec(R^{o2}) from one shared off-diagonal compression draw.  Returns (M_Bprime, M_Bdprime)."""
    C = len(lamR)
    K = min(K, C - 1)
    big = np.outer(lamR, lamR)
    np.fill_diagonal(big, 0.0)
    Moff = compress(rng, big.ravel(), C)
    Q, _ = np.linalg.qr(rng.standard_normal((C, K + 1)))
    p = predicted_tuple_norm(lamR)
    out, tr_ratio = [], []
    for rnorm2 in (np.full(C, 2.0 / C), (p - 1.0) / C):
        B = np.zeros((K + 1, K + 1))
        B[0, 0] = float((lamR ** 2).sum()) / C
        for i in range(K):
            B[i + 1, i + 1] = lamR[i] ** 2 * rnorm2[i]
            B[0, i + 1] = B[i + 1, 0] = lamR[i] ** 2 * np.sqrt(rnorm2[i]) / np.sqrt(C)
        M = Moff + Q @ B @ Q.T
        lump = float(((1.0 / C + rnorm2[K:]) * lamR[K:] ** 2).sum())
        M = M + np.eye(C) * lump / C
        tr_ratio.append(float(np.trace(M) / C))
        out.append(M * (C / np.trace(M)))
    return out[0], out[1], dict(tr_ratio_Bp=tr_ratio[0], tr_ratio_Bpp=tr_ratio[1])


def models(rng, mean_z, cov_z, jmax=JMAX_COMP):
    """Full surrogate spectra under B' and B'' (shared draws for j = 3, 4 and for the off-diagonal part of j = 2)."""
    R, s, mu = coefficients(mean_z, cov_z)
    lamR = np.clip(np.sort(eig(R))[::-1], 0, None)
    _, lin, _, _ = surrogate(mean_z, cov_z)
    comp = compressed_powers(rng, lamR, jmax)
    Mp, Mpp, diag = hadamard_square_variants(rng, lamR)
    outs = [diag]
    for M2 in (Mp, Mpp):
        total = lin.copy()
        for j, M in comp.items():
            Mj = M2 if j == 2 else M
            total += (mu[:, j][:, None] * Mj) * mu[:, j][None, :]
        total += np.diag((mu[:, jmax + 1:] ** 2).sum(axis=1))
        outs.append(total)
    return outs[1], outs[2], outs[0]


def analyse_row(mean_z, cov_z, n_draws, seed=0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    G = np.asarray(cov_z, float)
    C = G.shape[0]
    full, _, _, _ = surrogate(mean_z, G)
    ef = np.sort(eig(full))[::-1]
    R, _, _ = coefficients(mean_z, G)
    lam, a = actual_tuple_norm(R)
    lamR_full = np.clip(np.sort(eig(R))[::-1], 0, None)
    p = predicted_tuple_norm(lamR_full)[: len(lam)]
    h = haar_value(C)
    vec = pd.DataFrame(dict(i=np.arange(len(lam)), lam=lam, lam_over_C=lam / C, a=a, p=p, haar=np.full(len(lam), h)))
    out = dict(C=C, lam1_over_C=float(lam[0] / C), n_strong=int((lam > C / 3).sum()), n_medium=int(((lam > C / 10) & (lam <= C / 3)).sum()),
               r_eff_R=float(C ** 2 / (lamR_full ** 2).sum()))
    kp = {t: [] for t in TAUS}; kpp = {t: [] for t in TAUS}; top_p, top_pp, trs = [], [], []
    for _ in range(n_draws):
        Tp, Tpp, diag = models(rng, mean_z, G)
        trs.append(diag)
        ep, epp = np.sort(eig(Tp))[::-1], np.sort(eig(Tpp))[::-1]
        top_p.append(ep[:5]); top_pp.append(epp[:5])
        for t in TAUS:
            kp[t].append(kstar(ep, t)); kpp[t].append(kstar(epp, t))
    for t in TAUS:
        tt = f"{t:g}"
        ks = kstar(ef, t)
        out.update({f"k_surr@{tt}": ks, f"kBp_med@{tt}": float(np.median(kp[t])), f"Bp_err@{tt}": float((np.median(kp[t]) - ks) / ks),
                    f"kBpp_med@{tt}": float(np.median(kpp[t])), f"Bpp_err@{tt}": float((np.median(kpp[t]) - ks) / ks)})
    out["tr_ratio_Bp"] = float(np.mean([d["tr_ratio_Bp"] for d in trs])); out["tr_ratio_Bpp"] = float(np.mean([d["tr_ratio_Bpp"] for d in trs]))
    out["top5_ratio_Bp_over_true"] = ";".join(f"{v:.2f}" for v in np.mean(top_p, axis=0) / ef[:5])
    out["top5_ratio_Bpp_over_true"] = ";".join(f"{v:.2f}" for v in np.mean(top_pp, axis=0) / ef[:5])
    return out, vec


def load_rows(results, n_draws, n_draws_large, large=512, groups=None):
    rows, vecs = [], []
    p = os.path.join(results, VGG_MOMENTS)
    if os.path.exists(p) and (groups is None or "vgg" in groups):
        z = np.load(p)
        for key in z["keys"]:
            C = z[f"{key}__Sz"].shape[0]
            t0 = time.time()
            r, v = analyse_row(z[f"{key}__mz"], z[f"{key}__Sz"], n_draws_large if C > large else n_draws)
            r.update(group="vgg", tag="vgg16_bn_random_s1_matched", layer=str(key), sec=round(time.time() - t0))
            v = v.assign(group="vgg", layer=str(key), C=C)
            rows.append(r); vecs.append(v)
            print(f"vgg {key} C={C}: k_surr {r['k_surr@0.95']} B' {r['kBp_med@0.95']:.0f} B'' {r['kBpp_med@0.95']:.0f} lam1/C {r['lam1_over_C']:.2f} strong {r['n_strong']} ({r['sec']}s)", flush=True)
    for path in sorted(glob.glob(os.path.join(results, "relu_step", "*.npz"))):
        tag = os.path.basename(path).replace(".npz", "")
        group = "stream_init" if "random" in tag else "stream_trained"
        if groups is not None and group not in groups:
            continue
        z = np.load(path)
        names = sorted({k.split("::")[0] for k in z.files}, key=lambda n: int(n.split(".")[1]))
        for nme in names:
            C = z[f"{nme}::cov_y"].shape[0]
            t0 = time.time()
            r, v = analyse_row(z[f"{nme}::mean_y"], z[f"{nme}::cov_y"], n_draws_large if C > large else n_draws)
            r.update(group=group, tag=tag, layer=nme, sec=round(time.time() - t0))
            v = v.assign(group=group, layer=f"{tag}:{nme}", C=C)
            rows.append(r); vecs.append(v)
            print(f"{tag} {nme} C={C}: k_surr {r['k_surr@0.95']} B' {r['kBp_med@0.95']:.0f} B'' {r['kBpp_med@0.95']:.0f} lam1/C {r['lam1_over_C']:.2f} strong {r['n_strong']} ({r['sec']}s)", flush=True)
    return pd.DataFrame(rows), pd.concat(vecs, ignore_index=True)


def judge(df: pd.DataFrame, vec: pd.DataFrame) -> tuple[dict, str]:
    vgg, init, tr = df[df.group == "vgg"], df[df.group == "stream_init"], df[df.group == "stream_trained"]
    lines = []
    f2 = vgg[vgg.layer == "features.2"]
    f2_err = float(f2["Bpp_err@0.95"].abs().iloc[0]) if len(f2) else float("nan")
    f2_bp = float(f2["Bp_err@0.95"].abs().iloc[0]) if len(f2) else float("nan")
    a_ok = len(f2) == 1 and f2_err <= 0.25 and f2_err < f2_bp
    v_ok = int((vgg["Bpp_err@0.95"].abs() <= 0.10).sum()); i_ok = int((init["Bpp_err@0.95"].abs() <= 0.10).sum())
    p124 = a_ok and v_ok >= 12 and i_ok >= 9
    lines.append(f"P9.124 (model B''): features.2 |err| {f2_err:.2f} (need <= 0.25 and < B' {f2_bp:.2f}) -> {'ok' if a_ok else 'fails'}; "
                 f"|err| <= 0.10 in VGG {v_ok}/{len(vgg)} (need >= 12), init streams {i_ok}/{len(init)} (need >= 9); trained {int((tr['Bpp_err@0.95'].abs() <= 0.10).sum())}/{len(tr)} -> {'holds' if p124 else 'fails'}; "
                 f"B'' signed err median vgg {vgg['Bpp_err@0.95'].median():+.2f}, init {init['Bpp_err@0.95'].median():+.2f}, trained {tr['Bpp_err@0.95'].median():+.2f}; "
                 f"B' same rows: vgg {int((vgg['Bp_err@0.95'].abs() <= 0.10).sum())}/{len(vgg)}, init {int((init['Bp_err@0.95'].abs() <= 0.10).sum())}/{len(init)}")
    vi = vec[vec.group.isin(["vgg", "stream_init"])]
    strong = vi[vi.lam > vi.C / 3]
    dp, dh = (strong.p - strong.a).abs(), (strong.haar - strong.a).abs()
    n_s = len(strong)
    s_closer = float((dp < dh).mean()) if n_s else float("nan"); s_half = float((dp <= 0.5).mean()) if n_s else float("nan")
    a_part = (n_s >= 5) and s_closer >= 0.90 and s_half >= 0.80
    vv = vec[vec.group == "vgg"]
    v_half = float(((vv.p - vv.a).abs() <= 0.5).mean())
    b_part = v_half >= 0.80
    p125 = (a_part if n_s >= 5 else True) and b_part
    lines.append(f"P9.125 (eigenvector level): strong spikes (lam > C/3, 25 init rows) n = {n_s}{' (< 5: void, described only)' if n_s < 5 else ''}: "
                 f"B'' closer than Haar {s_closer:.2f} (need >= 0.90), |p - a| <= 0.5 {s_half:.2f} (need >= 0.80); "
                 f"VGG top-K all: |p - a| <= 0.5 {v_half:.2f} (need >= 0.80) -> {'holds' if p125 else 'fails'}")
    if n_s:
        lines.append("strong spikes: " + "; ".join(f"{r.layer} i={int(r.i)} lam/C {r.lam_over_C:.2f} a {r.a:.2f} p {r.p:.2f} haar {r.haar:.2f}" for r in strong.itertuples()))
    med = vi[(vi.lam > vi.C / 10) & (vi.lam <= vi.C / 3)]
    lines.append(f"exploratory medium spikes (C/10 < lam <= C/3) n = {len(med)}: |p - a| median {(med.p - med.a).abs().median():.2f}, |haar - a| median {(med.haar - med.a).abs().median():.2f}; "
                 f"bulk (lam <= C/10) VGG |p - a| median {((vv[vv.lam <= vv.C / 10].p - vv[vv.lam <= vv.C / 10].a).abs()).median():.2f}")
    for g, d in vec.groupby("group", sort=False):
        lines.append(f"exploratory {g}: a/haar median {(d.a / d.haar).median():.2f} [{(d.a / d.haar).min():.2f}, {(d.a / d.haar).max():.2f}]; a/p median {(d.a / d.p).median():.2f}")
    for g, d in df.groupby("group", sort=False):
        lines.append(f"exploratory {g}: lam1/C median {d.lam1_over_C.median():.2f} [{d.lam1_over_C.min():.2f}, {d.lam1_over_C.max():.2f}], strong per row {d.n_strong.tolist()}; "
                     f"B''@0.999 err median {d['Bpp_err@0.999'].median():+.2f}, B'@0.999 {d['Bp_err@0.999'].median():+.2f}; pre-rescale trace/C B' {d.tr_ratio_Bp.median():.2f} B'' {d.tr_ratio_Bpp.median():.2f}; top-5 ratio B''/true (first row) {d['top5_ratio_Bpp_over_true'].iloc[0]} (B' {d['top5_ratio_Bp_over_true'].iloc[0]})")
    verdict = dict(p124=bool(p124), p124_f2_err=f2_err, p124_vgg=v_ok, p124_init=i_ok, p125=bool(p125), p125_n_strong=n_s, p125_strong_closer=s_closer, p125_strong_half=s_half, p125_vgg_half=v_half)
    return verdict, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--draws-large", type=int, default=2)
    ap.add_argument("--groups", default=None, help="comma list of vgg,stream_init,stream_trained")
    ap.add_argument("--judge", action="store_true", help="only re-judge the stored rows.csv / eigvec_tuples.csv")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "t2a_truncation")
    os.makedirs(out, exist_ok=True)
    if a.judge:
        df = pd.read_csv(os.path.join(out, "rows.csv")); vec = pd.read_csv(os.path.join(out, "eigvec_tuples.csv"))
    else:
        df, vec = load_rows(a.results, a.draws, a.draws_large, groups=None if a.groups is None else set(a.groups.split(",")))
        df.to_csv(os.path.join(out, "rows.csv"), index=False); vec.to_csv(os.path.join(out, "eigvec_tuples.csv"), index=False)
    verdict, text = judge(df, vec)
    json.dump(verdict, open(os.path.join(out, "judgement.json"), "w"), indent=1)
    open(os.path.join(out, "judgement.txt"), "w").write(text + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    show = ["group", "layer", "C", "lam1_over_C", "n_strong", "r_eff_R", "k_surr@0.95", "kBp_med@0.95", "Bp_err@0.95", "kBpp_med@0.95", "Bpp_err@0.95", "k_surr@0.999", "kBp_med@0.999", "kBpp_med@0.999", "tr_ratio_Bp", "tr_ratio_Bpp", "top5_ratio_Bpp_over_true", "sec"]
    print(df[[c for c in show if c in df.columns]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
