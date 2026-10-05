"""B10 (plan 9.31): the transfer matrix between the memory-window gradient Gram and the output covariance.

With Gamma_win = U diag(gamma) U^T restricted to its head (the top H = k*(0.999) directions),
    M~ = diag(gamma)^{-1/2} U^T Sigma_out U diag(gamma)^{-1/2}        (H x H, PSD)
so that Sigma_out = U gamma^{1/2} M~ gamma^{1/2} U^T on range(U).  Its diagonal is the per-direction energy ratio
(exponent-one observation of plan 10.30); any narrowing of Sigma_out relative to Gamma must sit in its off-diagonal part.
Existing matrices only (results/gradwindow/<run>_epoch_*.npz and <run>_weights.json, as measure_gradwindow.py wrote them).

    python scripts/transfer_matrix.py --results ../results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FULL = ("wd5e-5_s0", "wd5e-5_s1", "wd5e-4_s0", "wd5e-4_s1", "wd5e-3_s0", "wd5e-3_s1")
EXTRA = ("wd1e-3_s0", "wd1e-3_s1", "wd2e-3_s0", "wd2e-3_s1", "vgg16_bn_ruler_s0", "vgg16_bn_ruler_s1", "vgg16_bn_ruler_s2")
LATE = ("features.34", "features.37", "features.40")
EARLY = ("features.0", "features.3", "features.7", "features.10")       # C <= 128
TAU = 0.999


def k_star(eig: np.ndarray, tau: float = TAU) -> int:
    e = np.clip(np.sort(np.asarray(eig, dtype=np.float64))[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-300), tau) + 1)


def window_gram(results: str, run: str) -> tuple[dict, dict]:
    """(Gamma_win per layer, Sigma_out per layer) from the saved per-epoch matrices and the window weights."""
    w = json.load(open(os.path.join(results, "gradwindow", f"{run}_weights.json")))["weights"]
    files = {int(re.search(r"epoch_(\d+)\.npz$", f).group(1)): f for f in glob.glob(os.path.join(results, "gradwindow", f"{run}_epoch_*.npz"))}
    T = max(files)
    Z = {E: np.load(files[E]) for E in files}
    layers = sorted({k.split("::")[0] for k in Z[T].files}, key=lambda s: int(s.split(".")[1]))
    win = {n: sum(float(w[str(E)]) * Z[E][n + "::grad"].astype(np.float64) for E in files if str(E) in w) for n in layers}
    sig = {n: Z[T][n + "::out"].astype(np.float64) for n in layers}
    return win, sig


def transfer(G: np.ndarray, S: np.ndarray) -> dict:
    """M~ on the head of G, its spectrum, and the widths of Sigma, Gamma, the diagonal-only and the off-diagonal reconstructions."""
    G = 0.5 * (G + G.T); S = 0.5 * (S + S.T)
    gam, U = np.linalg.eigh(G)
    order = np.argsort(-gam); gam = np.clip(gam[order], 0, None); U = U[:, order]
    H = k_star(gam)
    Uh, gh = U[:, :H], gam[:H]
    R = Uh.T @ S @ Uh                                     # Sigma in the head basis
    Mt = R / np.sqrt(np.outer(gh, gh))                    # transfer matrix
    d = np.diag(Mt)
    D = np.diag(d)
    diag_only = np.sqrt(np.outer(gh, gh)) * D             # = gamma^{1/2} D gamma^{1/2}
    eS = np.clip(np.linalg.eigvalsh(S)[::-1], 0, None)
    eM = np.clip(np.linalg.eigvalsh(Mt)[::-1], 0, None)
    eD = np.clip(np.linalg.eigvalsh(diag_only)[::-1], 0, None)
    eR = np.clip(np.linalg.eigvalsh(R)[::-1], 0, None)
    off = Mt - D
    return dict(H=H, k_sigma=k_star(eS), k_gamma=H, k_M=k_star(eM), k_diag=k_star(eD), k_head=k_star(eR),
                diag_cv=float(d.std() / max(d.mean(), 1e-300)), diag_slope=float(np.polyfit(np.log(gh), np.log(np.maximum(d, 1e-300)), 1)[0]),
                off_frac=float((off * off).sum() / max((Mt * Mt).sum(), 1e-300)), cond_M=float(eM[0] / max(eM[eM > 0][-1], 1e-300)),
                head_energy=float(eR.sum() / max(eS.sum(), 1e-300)))


def table(results: str, runs) -> pd.DataFrame:
    rows = []
    for run in runs:
        if not os.path.exists(os.path.join(results, "gradwindow", f"{run}_weights.json")):
            continue
        win, sig = window_gram(results, run)
        for n in win:
            r = transfer(win[n], sig[n]); r.update(run=run, layer=n, C=win[n].shape[0]); rows.append(r)
    return pd.DataFrame(rows)


def judge(results: str, out: str | None = None) -> dict:
    out = out or os.path.join(results, "transfer_matrix"); os.makedirs(out, exist_ok=True)
    df = table(results, FULL + EXTRA)
    if df.empty:
        print("no gradwindow matrices found"); return {}
    df.to_csv(os.path.join(out, "transfer.csv"), index=False)
    print("==== B10 (plan 9.31): the transfer matrix M~ between Gamma_win and Sigma_out ====")
    print(f"    {'run':20s} {'layer':12s} {'C':>4s} {'H':>4s} {'k(S)':>5s} {'k(M~)':>6s} {'k(D)':>5s} {'k(S|head)':>9s} {'diagCV':>7s} {'slope':>6s} {'off':>5s} {'headE':>6s}")
    for _, r in df.iterrows():
        if r.layer in LATE or r.layer in EARLY[:2]:
            print(f"    {r.run:20s} {r.layer:12s} {r.C:4d} {r.H:4d} {r.k_sigma:5d} {r.k_M:6d} {r.k_diag:5d} {r.k_head:9d} {r.diag_cv:7.2f} {r.diag_slope:+6.2f} {r.off_frac:5.2f} {r.head_energy:6.2f}")
    V: dict = {}
    full = df[df.run.isin(FULL)]
    late = full[full.layer.isin(LATE)]
    early = full[full.layer.isin(EARLY)]
    if len(late) == 18:
        n1 = int(((late.k_diag / late.k_gamma).between(0.8, 1.25)).sum())
        V["P-b10.1"] = n1 >= 16
        print(f"  P-b10.1 (diagonal-only reconstruction keeps Gamma's width: k(D)/k(Gamma) in [0.8, 1.25], >= 16/18 late layers) -> {'holds' if V['P-b10.1'] else 'fails'} ({n1}/18)")
        n2a = int((late.k_M / late.H <= 0.5).sum()); n2b = int(((late.k_sigma / late.k_M).between(0.5, 2.0)).sum())
        V["P-b10.2"] = n2a >= 16 and n2b >= 14
        print(f"  P-b10.2 (the narrowing sits in M~: k(M~)/H <= 0.5 in >= 16/18 and k(Sigma)/k(M~) in [0.5, 2] in >= 14/18 late layers) -> {'holds' if V['P-b10.2'] else 'fails'} ({n2a}/18, {n2b}/18)")
    if len(early) == 24:
        n3 = int((early.k_M / early.H >= 0.8).sum())
        V["P-b10.3"] = n3 >= 20
        print(f"  P-b10.3 (early wide layers: M~ nearly scalar, k(M~)/H >= 0.8, >= 20/24) -> {'holds' if V['P-b10.3'] else 'fails'} ({n3}/24)")
    med = {}
    for lam in ("5e-5", "5e-4", "2e-3"):
        for s in (0, 1):
            d = df[(df.run == f"wd{lam}_s{s}") & df.layer.isin(LATE)]
            if len(d) == 3:
                med[(lam, s)] = float(np.median(d.k_M / d.H))
    if len(med) == 6:
        V["P-b10.4"] = all(med[("5e-5", s)] > med[("5e-4", s)] > med[("2e-3", s)] for s in (0, 1))
        print(f"  P-b10.4 (coherence grows with lambda: median k(M~)/H over the late layers 5e-5 > 5e-4 > 2e-3, both seeds) -> {'holds' if V['P-b10.4'] else 'fails'} "
              + " ".join(f"{k[0]}/s{k[1]}:{v:.2f}" for k, v in sorted(med.items(), key=lambda kv: (float(kv[0][0]), kv[0][1]))))
    json.dump({"verdicts": V}, open(os.path.join(out, "judgement.json"), "w"), indent=1)
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    judge(a.results, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
