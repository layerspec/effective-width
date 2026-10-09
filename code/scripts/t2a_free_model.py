"""Plan 9.52 (P9.121-P9.122): the activation step as a compression of tensor powers of the convolution step.

Model A (exact rotation law): G = W Sigma_p W^T has Haar eigenvectors, so (G, m) -> (O G O^T, O m) with Haar O leaves the law
of k_tau(Sigma_a^G) unchanged; rotating the stored (G, m) gives the law's spread (exact for the VGG rows, a test for the streams).

Model B (spectrum-only candidate formula): replace every Hadamard power R^{oj} (j = 2, 3, 4) by a Haar compression of spec(R)^{(x)j}
from dimension C^j to C, done iteratively (C^2 -> C at each step, Haar isometry approximated by i.i.d. Gaussian rows, streamed);
Hermite orders j >= 5 are kept as a diagonal; mu_j and the linear part use the row's own (m, s).

    cd code && python scripts/t2a_free_model.py --results ../results [--rot 8 --draws 4 --draws-large 2]
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
from t1_kernel_reading import VGG_MOMENTS, correlation, eig, kstar, surrogate  # noqa: E402
from t2_bounds import coefficients  # noqa: E402

TAUS = (0.95, 0.999)
JMAX_COMP = 4


def haar(rng, n):
    Q, Rr = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(Rr))[None, :]


def compress(rng, big, C, chunk=2048):
    """Haar-type compression of diag(big) (dimension len(big)) to C dimensions: (1/N) sum_t big_t g_t g_t^T, g_t ~ N(0, I_C), N = len(big)."""
    N = len(big)
    M = np.zeros((C, C))
    for a in range(0, N, chunk):
        b = big[a:a + chunk]
        Gm = rng.standard_normal((len(b), C))
        M += (Gm * b[:, None]).T @ Gm
    return M / N


def compressed_powers(rng, lamR, jmax=JMAX_COMP):
    """Iterative compression: returns {j: M_j} with spec(M_j) modelling spec(R^{oj}), M_j in a random basis, diag ~ 1."""
    C = len(lamR)
    out = {}
    cur = np.asarray(lamR, float)
    for j in range(2, jmax + 1):
        big = np.outer(cur, lamR).ravel()
        M = compress(rng, big, C)
        out[j] = M
        cur = np.clip(eig(M), 0, None)[::-1]
    return out


def model_b(rng, mean_z, cov_z, jmax=JMAX_COMP):
    R, s, mu = coefficients(mean_z, cov_z)
    lamR = np.clip(np.sort(eig(R))[::-1], 0, None)
    _, lin, _, _ = surrogate(mean_z, cov_z)
    comp = compressed_powers(rng, lamR, jmax)
    total = lin.copy()
    for j, M in comp.items():
        total += (mu[:, j][:, None] * M) * mu[:, j][None, :]
    rem = (mu[:, jmax + 1:] ** 2).sum(axis=1)
    total += np.diag(rem)
    # simplified variant: j = 2 compressed, j >= 3 diagonal
    simple = lin + (mu[:, 2][:, None] * comp[2]) * mu[:, 2][None, :] + np.diag((mu[:, 3:] ** 2).sum(axis=1))
    return total, simple


def hadamard_square_bprime(rng, lamR, K=48):
    """Plan 9.53 / proofs T2a (b)-(c): model B' of spec(R^{o2}).  Off-diagonal tuples {lam_i lam_i'}_{i != i'} by Haar compression;
    diagonal tuples u_i o u_i = 1/C + r_i as an explicit (K+1)x(K+1) block in span(e, r_1..r_K): M_ee = tr R^2 / C, M_ii = 2 lam_i^2 / C,
    M_ei = sqrt(2) lam_i^2 / C; diagonal tuples beyond K as a diagonal lump (3 lam_i^2 / C each); finally rescaled to the exact trace C."""
    C = len(lamR)
    K = min(K, C - 1)
    big = np.outer(lamR, lamR)
    np.fill_diagonal(big, 0.0)
    M = compress(rng, big.ravel(), C)
    B = np.zeros((K + 1, K + 1))
    B[0, 0] = float((lamR ** 2).sum()) / C
    for i in range(K):
        B[i + 1, i + 1] = 2.0 * lamR[i] ** 2 / C
        B[0, i + 1] = B[i + 1, 0] = np.sqrt(2.0) * lamR[i] ** 2 / C
    Q, _ = np.linalg.qr(rng.standard_normal((C, K + 1)))
    M += Q @ B @ Q.T
    M += np.eye(C) * (3.0 * float((lamR[K:] ** 2).sum()) / C) / C
    return M * (C / np.trace(M))


def model_bprime(rng, mean_z, cov_z, jmax=JMAX_COMP):
    R, s, mu = coefficients(mean_z, cov_z)
    lamR = np.clip(np.sort(eig(R))[::-1], 0, None)
    _, lin, _, _ = surrogate(mean_z, cov_z)
    comp = compressed_powers(rng, lamR, jmax)
    comp[2] = hadamard_square_bprime(rng, lamR)
    total = lin.copy()
    for j, M in comp.items():
        total += (mu[:, j][:, None] * M) * mu[:, j][None, :]
    total += np.diag((mu[:, jmax + 1:] ** 2).sum(axis=1))
    return total


def model_a(rng, mean_z, cov_z, n_rot, taus=TAUS):
    ks = {t: [] for t in taus}
    C = cov_z.shape[0]
    for _ in range(n_rot):
        O = haar(rng, C)
        Gp = O @ cov_z @ O.T
        mp = O @ np.asarray(mean_z, float)
        full, _, _, _ = surrogate(mp, Gp)
        e = eig(full)
        for t in taus:
            ks[t].append(kstar(e, t))
    return ks


def analyse_row(mean_z, cov_z, n_rot, n_draws, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    G = np.asarray(cov_z, float)
    full, _, _, _ = surrogate(mean_z, G)
    ef = np.sort(eig(full))[::-1]
    out = dict(C=G.shape[0])
    ka = model_a(rng, mean_z, G, n_rot)
    kb = {t: [] for t in TAUS}; kb2 = {t: [] for t in TAUS}; kbp = {t: [] for t in TAUS}; tops = []; topsp = []
    for _ in range(n_draws):
        tot, simple = model_b(rng, mean_z, G)
        totp = model_bprime(rng, mean_z, G)
        et, es, ep = np.sort(eig(tot))[::-1], eig(simple), np.sort(eig(totp))[::-1]
        tops.append(et[:5]); topsp.append(ep[:5])
        for t in TAUS:
            kb[t].append(kstar(et, t)); kb2[t].append(kstar(es, t)); kbp[t].append(kstar(ep, t))
    for t in TAUS:
        tt = f"{t:g}"
        ks = kstar(ef, t)
        a = np.array(ka[t]); b = np.array(kb[t]); b2 = np.array(kb2[t])
        bp = np.array(kbp[t])
        out.update({f"kBp_med@{tt}": float(np.median(bp)), f"Bp_err@{tt}": float((np.median(bp) - ks) / ks)})
        out.update({f"k_surr@{tt}": ks, f"kA_min@{tt}": int(a.min()), f"kA_max@{tt}": int(a.max()), f"kA_med@{tt}": float(np.median(a)), f"kA_sd@{tt}": float(a.std()),
                    f"A_in_range@{tt}": bool(a.min() <= ks <= a.max()), f"A_err@{tt}": float((np.median(a) - ks) / ks),
                    f"kB_med@{tt}": float(np.median(b)), f"B_err@{tt}": float((np.median(b) - ks) / ks),
                    f"kB2_med@{tt}": float(np.median(b2)), f"B2_err@{tt}": float((np.median(b2) - ks) / ks)})
    top_true = ef[:5]; top_b = np.mean(tops, axis=0)
    out["top5_ratio_B_over_true"] = ";".join(f"{v:.2f}" for v in top_b / top_true)
    out["top5_ratio_Bp_over_true"] = ";".join(f"{v:.2f}" for v in np.mean(topsp, axis=0) / top_true)
    return out


def load_rows(results, n_rot, n_draws, n_draws_large, large=512, groups=None):
    rows = []
    p = os.path.join(results, VGG_MOMENTS)
    if os.path.exists(p) and (groups is None or "vgg" in groups):
        z = np.load(p)
        for key in z["keys"]:
            C = z[f"{key}__Sz"].shape[0]
            t0 = time.time()
            r = analyse_row(z[f"{key}__mz"], z[f"{key}__Sz"], n_rot, n_draws_large if C > large else n_draws)
            r.update(group="vgg", tag="vgg16_bn_random_s1_matched", layer=str(key), sec=round(time.time() - t0))
            rows.append(r); print(f"vgg {key} C={C}: k_surr {r['k_surr@0.95']} A med {r['kA_med@0.95']:.0f} [{r['kA_min@0.95']},{r['kA_max@0.95']}] B med {r['kB_med@0.95']:.0f} B' med {r['kBp_med@0.95']:.0f} ({r['sec']}s)", flush=True)
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
            r = analyse_row(z[f"{nme}::mean_y"], z[f"{nme}::cov_y"], n_rot, n_draws_large if C > large else n_draws)
            r.update(group=group, tag=tag, layer=nme, sec=round(time.time() - t0))
            rows.append(r); print(f"{tag} {nme} C={C}: k_surr {r['k_surr@0.95']} A med {r['kA_med@0.95']:.0f} [{r['kA_min@0.95']},{r['kA_max@0.95']}] B med {r['kB_med@0.95']:.0f} B' med {r['kBp_med@0.95']:.0f} ({r['sec']}s)", flush=True)
    return pd.DataFrame(rows)


def judge(df: pd.DataFrame) -> tuple[dict, str]:
    vgg, init, tr = df[df.group == "vgg"], df[df.group == "stream_init"], df[df.group == "stream_trained"]
    lines = []
    a_v = int(vgg["A_in_range@0.95"].sum()); a_i = int((init["A_err@0.95"].abs() <= 0.10).sum())
    p121 = a_v >= 12 and a_i >= 9
    lines.append(f"P9.121 (model A): VGG k_surr inside the rotation range {a_v}/{len(vgg)} (need >= 12; theorem); init streams |median_A - k_surr|/k_surr <= 0.10 in {a_i}/{len(init)} (need >= 9) -> {'holds' if p121 else 'fails'}; "
                 f"trained streams {int((tr['A_err@0.95'].abs() <= 0.10).sum())}/{len(tr)}; VGG sd_A median {vgg['kA_sd@0.95'].median():.1f} channels")
    b_v = int((vgg["B_err@0.95"].abs() <= 0.10).sum()); b_i = int((init["B_err@0.95"].abs() <= 0.10).sum())
    p122 = b_v >= 10 and b_i >= 9
    lines.append(f"P9.122 (model B): |k_B - k_surr|/k_surr <= 0.10 in VGG {b_v}/{len(vgg)} (need >= 10), init streams {b_i}/{len(init)} (need >= 9) -> {'holds' if p122 else 'fails'}; "
                 f"trained {int((tr['B_err@0.95'].abs() <= 0.10).sum())}/{len(tr)}")
    if "Bp_err@0.95" in df:
        bp_v = int((vgg["Bp_err@0.95"].abs() <= 0.10).sum()); bp_i = int((init["Bp_err@0.95"].abs() <= 0.10).sum())
        small = vgg[vgg.C <= 128]
        better = int((small["Bp_err@0.95"].abs() < small["B_err@0.95"].abs()).sum())
        p123 = bp_v >= 10 and bp_i >= 9 and better == len(small)
        lines.append(f"P9.123 (model B'): |k_B' - k_surr|/k_surr <= 0.10 in VGG {bp_v}/{len(vgg)} (need >= 10), init streams {bp_i}/{len(init)} (need >= 9); "
                     f"B' strictly closer than B on the C <= 128 VGG rows {better}/{len(small)} (need all) -> {'holds' if p123 else 'fails'}; "
                     f"B' signed err median vgg {vgg['Bp_err@0.95'].median():+.2f}, init {init['Bp_err@0.95'].median():+.2f}, trained {tr['Bp_err@0.95'].median():+.2f}; "
                     f"top-5 ratio B'/true (first VGG row) {vgg['top5_ratio_Bp_over_true'].iloc[0]}")
        verdict_extra = dict(p123_vgg=bp_v, p123_init=bp_i, p123_better=better, p123=bool(p123))
    else:
        verdict_extra = {}
    for g, d in df.groupby("group", sort=False):
        lines.append(f"exploratory {g}: B signed err@0.95 median {d['B_err@0.95'].median():+.2f} [{d['B_err@0.95'].min():+.2f}, {d['B_err@0.95'].max():+.2f}]; "
                     f"B@0.999 median {d['B_err@0.999'].median():+.2f}; simplified B2@0.95 median {d['B2_err@0.95'].median():+.2f}; A@0.999 median {d['A_err@0.999'].median():+.2f}; "
                     f"top-5 ratio B/true (first row) {d['top5_ratio_B_over_true'].iloc[0]}")
    verdict = dict(p121_vgg=a_v, p121_init=a_i, p121=bool(p121), p122_vgg=b_v, p122_init=b_i, p122=bool(p122), **verdict_extra)
    return verdict, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rot", type=int, default=8)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--draws-large", type=int, default=2)
    ap.add_argument("--groups", default=None, help="comma list of vgg,stream_init,stream_trained")
    ap.add_argument("--judge", action="store_true", help="only re-judge the stored rows.csv")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "t2a_free_model")
    os.makedirs(out, exist_ok=True)
    if a.judge:
        df = pd.read_csv(os.path.join(out, "rows.csv"))
    else:
        df = load_rows(a.results, a.rot, a.draws, a.draws_large, groups=None if a.groups is None else set(a.groups.split(",")))
        df.to_csv(os.path.join(out, "rows.csv"), index=False)
    verdict, text = judge(df)
    json.dump(verdict, open(os.path.join(out, "judgement.json"), "w"), indent=1)
    open(os.path.join(out, "judgement.txt"), "w").write(text + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    show = ["group", "layer", "C", "k_surr@0.95", "kA_min@0.95", "kA_med@0.95", "kA_max@0.95", "kB_med@0.95", "B_err@0.95", "kB2_med@0.95", "kBp_med@0.95", "Bp_err@0.95", "k_surr@0.999", "kA_med@0.999", "kB_med@0.999", "top5_ratio_B_over_true", "sec"]
    print(df[[c for c in show if c in df.columns]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
