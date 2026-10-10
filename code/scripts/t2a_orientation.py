"""Plan 10.58 (post hoc, exploratory; written after the first four rows of plan 9.54 were read): is the small-C gap of model B'
a spectrum error or an orientation error?

Controls per row (tau = 0.95, 0.999):
  k_surr      true surrogate width;
  k_rotRoR    same surrogate with the TRUE spectrum of R^{o2} placed in a Haar basis (orientation removed, spectrum kept);
  k_peel      exact spike terms with the real u_1 (lam_1^2 (u_1 o u_1)(u_1 o u_1)^T + 2 lam_1 D_{u_1} T D_{u_1}) + spec(T o T) in a Haar basis;
  k_RoR       tau-width of the true R^{o2} alone, against k_RoR_Bp = width of model B' spectrum of R^{o2} alone (is B' right as a spectrum?);
  cos(m, u_1), corr(mu_2^2, u_1^2), median |m/s|   (why the orientation is not generic).

    cd code && python scripts/t2a_orientation.py --results ../results [--rot 4]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_kernel_reading import VGG_MOMENTS, eig, kstar, surrogate  # noqa: E402
from t2_bounds import coefficients  # noqa: E402
from t2a_free_model import hadamard_square_bprime, haar  # noqa: E402

TAUS = (0.95, 0.999)


def rotated(rng, spec, C):
    O = haar(rng, C)
    return (O * np.clip(spec, 0, None)[None, :]) @ O.T


def analyse_row(mean_z, cov_z, n_rot=4, seed=0, with_bprime=True) -> dict:
    rng = np.random.default_rng(seed)
    m = np.asarray(mean_z, float); G = np.asarray(cov_z, float)
    R, s, mu = coefficients(m, G); C = R.shape[0]
    full, lin, _, _ = surrogate(m, G)
    w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]; w, V = np.clip(w[o], 0, None), V[:, o]
    u1 = V[:, 0]
    RR = R * R
    j2 = (mu[:, 2][:, None] * RR) * mu[:, 2][None, :]
    rest = full - lin - j2
    T = R - w[0] * np.outer(u1, u1)
    spike = w[0] ** 2 * np.outer(u1 * u1, u1 * u1) + 2 * w[0] * (u1[:, None] * T) * u1[None, :]
    specRR, specTT = eig(RR), eig(T * T)
    krot = {t: [] for t in TAUS}; kpeel = {t: [] for t in TAUS}
    for _ in range(n_rot):
        M = rotated(rng, specRR, C)
        e1 = eig(lin + (mu[:, 2][:, None] * M) * mu[:, 2][None, :] + rest)
        TT = rotated(rng, specTT, C)
        e2 = eig(lin + (mu[:, 2][:, None] * (spike + TT)) * mu[:, 2][None, :] + rest)
        for t in TAUS:
            krot[t].append(kstar(e1, t)); kpeel[t].append(kstar(e2, t))
    out = dict(C=C, lam1_over_C=float(w[0] / C), cos_m_u1=float(abs(np.dot(m / np.linalg.norm(m), u1))),
               corr_mu2sq_u1sq=float(np.corrcoef(mu[:, 2] ** 2, u1 ** 2)[0, 1]), med_abs_m_over_s=float(np.median(np.abs(m / s))),
               tr_lin=float(np.trace(lin)), tr_j2=float(np.trace(j2)), tr_rest=float(np.trace(rest)))
    ef = eig(full); el = eig(lin)
    if with_bprime:
        Mb = hadamard_square_bprime(rng, w)
        eb = eig(Mb)
    for t in TAUS:
        tt = f"{t:g}"
        out.update({f"k_surr@{tt}": kstar(ef, t), f"k_lin@{tt}": kstar(el, t), f"k_rotRoR@{tt}": float(np.median(krot[t])),
                    f"k_peel@{tt}": float(np.median(kpeel[t])), f"k_RoR@{tt}": kstar(specRR, t)})
        if with_bprime:
            out[f"k_RoR_Bp@{tt}"] = kstar(eb, t)
    if with_bprime:
        out["top5_RoR_Bp_over_true"] = ";".join(f"{v:.2f}" for v in np.sort(eb)[::-1][:5] / np.sort(specRR)[::-1][:5])
    return out


def load_rows(results, n_rot, groups=None):
    rows = []
    p = os.path.join(results, VGG_MOMENTS)
    if os.path.exists(p) and (groups is None or "vgg" in groups):
        z = np.load(p)
        for key in z["keys"]:
            t0 = time.time()
            r = analyse_row(z[f"{key}__mz"], z[f"{key}__Sz"], n_rot)
            r.update(group="vgg", tag="vgg16_bn_random_s1_matched", layer=str(key), sec=round(time.time() - t0)); rows.append(r)
            print(f"vgg {key} C={r['C']}: k_surr {r['k_surr@0.95']} rotRoR {r['k_rotRoR@0.95']:.0f} peel {r['k_peel@0.95']:.0f} RoR {r['k_RoR@0.95']} B'RoR {r['k_RoR_Bp@0.95']} ({r['sec']}s)", flush=True)
    for path in sorted(glob.glob(os.path.join(results, "relu_step", "*.npz"))):
        tag = os.path.basename(path).replace(".npz", "")
        group = "stream_init" if "random" in tag else "stream_trained"
        if groups is not None and group not in groups:
            continue
        z = np.load(path)
        names = sorted({k.split("::")[0] for k in z.files}, key=lambda n: int(n.split(".")[1]))
        for nme in names:
            t0 = time.time()
            r = analyse_row(z[f"{nme}::mean_y"], z[f"{nme}::cov_y"], n_rot)
            r.update(group=group, tag=tag, layer=nme, sec=round(time.time() - t0)); rows.append(r)
            print(f"{tag} {nme} C={r['C']}: k_surr {r['k_surr@0.95']} rotRoR {r['k_rotRoR@0.95']:.0f} peel {r['k_peel@0.95']:.0f} RoR {r['k_RoR@0.95']} B'RoR {r['k_RoR_Bp@0.95']} ({r['sec']}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> str:
    lines = []
    for t in TAUS:
        tt = f"{t:g}"
        rot = (df[f"k_rotRoR@{tt}"] - df[f"k_surr@{tt}"]) / df[f"k_surr@{tt}"]
        peel = (df[f"k_peel@{tt}"] - df[f"k_surr@{tt}"]) / df[f"k_surr@{tt}"]
        small, big = df[df.C <= 128], df[df.C >= 256]
        lines.append(f"tau={tt}: rotated-true-spectrum err: C<=128 {rot[df.C <= 128].round(2).tolist()}, C>=256 median {rot[df.C >= 256].median():+.3f} [{rot[df.C >= 256].min():+.3f}, {rot[df.C >= 256].max():+.3f}]; "
                     f"peel err: C<=128 {peel[df.C <= 128].round(2).tolist()}, C>=256 median {peel[df.C >= 256].median():+.3f}")
        if f"k_RoR_Bp@{tt}" in df:
            sp = (df[f"k_RoR_Bp@{tt}"] - df[f"k_RoR@{tt}"]) / df[f"k_RoR@{tt}"]
            lines.append(f"tau={tt}: B' spectrum of R^(o2) alone vs true: err median {sp.median():+.3f} [{sp.min():+.3f}, {sp.max():+.3f}]; |err| <= 0.10 in {int((sp.abs() <= 0.10).sum())}/{len(df)}; C<=128 {sp[df.C <= 128].round(2).tolist()}")
    lines.append(f"cos(m,u1): median {df.cos_m_u1.median():.2f} [{df.cos_m_u1.min():.2f}, {df.cos_m_u1.max():.2f}] (n={len(df)}); "
                 f"corr(mu2^2,u1^2): C<=128 {df[df.C <= 128].corr_mu2sq_u1sq.round(2).tolist()}, C>=256 median {df[df.C >= 256].corr_mu2sq_u1sq.median():+.2f}; "
                 f"median |m/s|: C<=128 {df[df.C <= 128].med_abs_m_over_s.round(2).tolist()}, C>=256 median {df[df.C >= 256].med_abs_m_over_s.median():.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rot", type=int, default=4)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "t2a_orientation")
    os.makedirs(out, exist_ok=True)
    if a.summarise:
        df = pd.read_csv(os.path.join(out, "rows.csv"))
    else:
        df = load_rows(a.results, a.rot, None if a.groups is None else set(a.groups.split(",")))
        df.to_csv(os.path.join(out, "rows.csv"), index=False)
    text = summarise(df)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
    show = ["group", "layer", "C", "lam1_over_C", "k_surr@0.95", "k_lin@0.95", "k_rotRoR@0.95", "k_peel@0.95", "k_RoR@0.95", "k_RoR_Bp@0.95", "k_surr@0.999", "k_rotRoR@0.999", "k_peel@0.999", "cos_m_u1", "corr_mu2sq_u1sq", "med_abs_m_over_s", "top5_RoR_Bp_over_true", "sec"]
    print(df[[c for c in show if c in df.columns]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
