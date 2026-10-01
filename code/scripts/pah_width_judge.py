"""B8 (plan 9.28; notes/crh-prior-art-2026-10-02.md): is the output covariance of a conv-BN layer a power of its
output-gradient Gram, and does the power account for the gap between their tau-widths?

Ziyin, Chuang, Galanti and Poggio (ICLR 2025), Theorem 2, phase 8: backward representation-weight alignment
(W^T W prop. Sigma_patch) and forward gradient-weight alignment (W W^T prop. Gamma) give Sigma_out prop. (W W^T)^2 prop. Gamma^2.
Lemma B8.0 (elementary): k*_tau(lambda^alpha) is non-increasing in alpha.  Together: k*_tau(Sigma_out) = k*_tau(Gamma^alpha).

Existing data only (no forward pass, no training):
    results/gradwindow/<run>_epoch_100.npz   per layer the matrices `out` (Sigma_out) and `grad` (Gamma) of one checkpoint
    results/wd_sweep/<run>/epoch_100.pt      the weights of the same checkpoint (three-matrix version)
    results/agop/<net>_directions.npz        ImageNet public weights: gain = s^2, sigma = v^T Sigma_patch v, gamma = u^T Gamma u
    results/gradgram/*_spectra.npz           eigenvalues only (rank-matched description)

    python scripts/pah_width_judge.py --results ../results            # P-b8.1 .. P-b8.4
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TAUS = (0.95, 0.999)
HEAD_TAU = 0.999                 # directions used in the fits: the head carrying this fraction of the trace
WIDE = 100                       # layers judged: at least this many channels (as in P-b7.1')
FULL_RUNS = [f"wd{lam}_s{s}" for lam in ("5e-5", "5e-4", "5e-3") for s in (0, 1)]
REF_LAM = "5e-4"
IMAGENET = ("vgg16_bn", "resnet18")


def k_star(eig: np.ndarray, tau: float) -> int:
    """Smallest k whose leading eigenvalues carry a fraction tau of the trace (measure_gradgram.k_star)."""
    e = np.clip(np.sort(np.asarray(eig, dtype=np.float64))[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-300), tau) + 1)


def power_width(eig: np.ndarray, alpha: float, tau: float) -> int:
    """k*_tau of the spectrum raised to the power alpha (scale-free: the spectrum is normalised by its maximum first)."""
    e = np.clip(np.sort(np.asarray(eig, dtype=np.float64))[::-1], 0, None)
    if e[0] <= 0:
        return 1
    return k_star((e / e[0]) ** alpha, tau)


def head(weights: np.ndarray, tau: float = HEAD_TAU) -> np.ndarray:
    """Indices (in decreasing order of weight) of the directions carrying a fraction tau of the total, positive ones only."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 0, None)
    order = np.argsort(-w)
    k = k_star(w, tau)
    idx = order[:k]
    return idx[w[idx] > 1e-12 * max(w[order[0]], 1e-300)]


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log y on log x over the entries where both are positive; nan with fewer than 3 points."""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    ok = (x > 0) & (y > 0)
    if ok.sum() < 3 or np.ptp(np.log(x[ok])) == 0:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def _spearman(a, b) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def basis_stats(S: np.ndarray, G: np.ndarray, rng: np.random.Generator) -> dict:
    """Sigma_out read in the eigenbasis of Gamma: rank alignment, co-diagonality, exponent, and the widths it implies."""
    S = 0.5 * (np.asarray(S, dtype=np.float64) + np.asarray(S, dtype=np.float64).T)
    G = 0.5 * (np.asarray(G, dtype=np.float64) + np.asarray(G, dtype=np.float64).T)
    gam, U = np.linalg.eigh(G)
    order = np.argsort(-gam); gam = np.clip(gam[order], 0, None); U = U[:, order]
    h = np.clip(np.einsum("ji,jk,ki->i", U, S, U), 0, None)
    fro2 = float((S * S).sum())
    Q, _ = np.linalg.qr(rng.standard_normal(S.shape))
    h_null = np.einsum("ji,jk,ki->i", Q, S, Q)
    I = head(gam)
    alpha = loglog_slope(gam[I], h[I])
    eS = np.linalg.eigvalsh(S)
    row = {"C": S.shape[0], "n_head": int(len(I)), "rho_b": _spearman(gam[I], h[I]),
           "codiag": float((h * h).sum() / max(fro2, 1e-300)), "codiag_null": float((h_null * h_null).sum() / max(fro2, 1e-300)),
           "alpha_hat": alpha}
    for tau in TAUS:
        row[f"kS_{tau}"] = k_star(eS, tau)
        row[f"kG1_{tau}"] = power_width(gam, 1.0, tau)
        row[f"kG2_{tau}"] = power_width(gam, 2.0, tau)
        row[f"kGhat_{tau}"] = power_width(gam, alpha, tau) if np.isfinite(alpha) and alpha > 0 else np.nan
    return row


def three_matrix(W: np.ndarray, S: np.ndarray, G: np.ndarray) -> dict:
    """Exponents among H_b = Sigma_out, Z_b = W W^T and G_b = Gamma read along the left singular vectors of W."""
    W = np.asarray(W, dtype=np.float64).reshape(W.shape[0], -1)
    U, s, _ = np.linalg.svd(W, full_matrices=False)
    gain = s * s
    hS = np.clip(np.einsum("ji,jk,ki->i", U, S, U), 0, None)
    hG = np.clip(np.einsum("ji,jk,ki->i", U, G, U), 0, None)
    return exponents(gain, hS, hG)


def exponents(gain: np.ndarray, hS: np.ndarray, hG: np.ndarray) -> dict:
    """a_HZ: log energy of Sigma_out on log gain; a_ZG: log gain on log energy of Gamma; a_HG: their composition measured."""
    I = head(gain)
    return {"a_HZ": loglog_slope(gain[I], hS[I]), "a_ZG": loglog_slope(hG[I], gain[I]), "a_HG": loglog_slope(hG[I], hS[I]),
            "rho_ZG": _spearman(gain[I], hG[I]), "rho_HZ": _spearman(gain[I], hS[I]), "n_dir": int(len(I))}


def _layers(z) -> list[str]:
    names = [k.rsplit("::", 1)[0] for k in z.files if k.endswith("::out")]
    return sorted(names, key=lambda n: int(re.search(r"(\d+)$", n).group(1)) if re.search(r"(\d+)$", n) else 0)


def run_rows(results: str, run: str, epoch: int = 100) -> list[dict]:
    path = os.path.join(results, "gradwindow", f"{run}_epoch_{epoch:03d}.npz")
    if not os.path.exists(path):
        return []
    z = np.load(path)
    sd = None
    for cand in (os.path.join(results, "wd_sweep", run, f"epoch_{epoch:03d}.pt"), os.path.join(results, "a18_pt", run, f"epoch_{epoch:03d}.pt")):
        if os.path.exists(cand):
            import torch
            sd = torch.load(cand, map_location="cpu", weights_only=False)
            sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
            break
    wpath = os.path.join(results, "gradwindow", f"{run}_weights.json")
    wE = json.load(open(wpath))["weights"] if os.path.exists(wpath) else None
    rng = np.random.default_rng(0)
    rows = []
    for depth, name in enumerate(_layers(z)):
        S, G = z[f"{name}::out"], z[f"{name}::grad"]
        row = {"run": run, "layer": name, "depth": depth, **basis_stats(S, G, rng)}
        if sd is not None and f"{name}.weight" in sd:
            row.update(three_matrix(sd[f"{name}.weight"].double().numpy(), S, G))
        if wE:                                                   # descriptive: the memory-window Gram in place of the final one
            Gw = 0
            for e, w in wE.items():
                pe = os.path.join(results, "gradwindow", f"{run}_epoch_{int(e):03d}.npz")
                if not os.path.exists(pe):
                    Gw = None; break
                Gw = Gw + w * np.load(pe)[f"{name}::grad"].astype(np.float64)
            if Gw is not None:
                b = basis_stats(S, Gw, rng)
                row.update({"alpha_hat_win": b["alpha_hat"], "kG1_win_0.999": b["kG1_0.999"], "kG2_win_0.999": b["kG2_0.999"]})
        rows.append(row)
    return rows


def d_alpha(df: pd.DataFrame, col: str, tau: float = 0.999) -> float:
    w = df[(df.C >= WIDE) & df[f"{col}_{tau}"].notna()]
    return float(np.median(np.abs(np.log(w[f"kS_{tau}"]) - np.log(w[f"{col}_{tau}"])))) if len(w) else float("nan")


def imagenet_exponents(results: str, net: str) -> pd.DataFrame:
    path = os.path.join(results, "agop", f"{net}_directions.npz")
    if not os.path.exists(path):
        return pd.DataFrame()
    z = np.load(path)
    rows = []
    for name in dict.fromkeys(k.rsplit("::", 1)[0] for k in z.files):
        gain, sigma, gamma = z[f"{name}::gain"], z[f"{name}::sigma"], z[f"{name}::gamma"]
        rows.append({"net": net, "layer": name, **exponents(gain, gain * sigma, gamma)})          # u^T Sigma_out u = s^2 sigma (identity)
    return pd.DataFrame(rows)


def rank_matched(results: str) -> pd.DataFrame:
    """gradgram spectra (eigenvalues only): widths of the output against the gradient spectrum to the powers 1 and 2, last three layers."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results, "gradgram", "*_spectra.npz"))):
        z = np.load(path)
        names = _layers(z)[-3:]
        for name in names:
            eo, eg = z[f"{name}::out"], z[f"{name}::grad"]
            rows.append({"ckpt": os.path.basename(path).replace("_spectra.npz", ""), "layer": name,
                         **{f"kS_{t}": k_star(eo, t) for t in TAUS}, **{f"kG1_{t}": power_width(eg, 1, t) for t in TAUS},
                         **{f"kG2_{t}": power_width(eg, 2, t) for t in TAUS}})
    return pd.DataFrame(rows)


def judge(results: str, out: str | None = None) -> dict:
    out = out or os.path.join(results, "pah")
    os.makedirs(out, exist_ok=True)
    runs = FULL_RUNS + [f"vgg16_bn_ruler_s{s}" for s in (0, 1, 2)]
    df = pd.DataFrame([r for run in runs for r in run_rows(results, run)])
    V: dict = {}
    print("==== B8 (plan 9.28): Sigma_out as a power of the output-gradient Gram on conv-BN layers ====")
    if df.empty:
        print("  no gradwindow matrices found"); return V
    df.to_csv(os.path.join(out, "pah_layers.csv"), index=False)
    present = [r for r in FULL_RUNS if (df.run == r).any()]
    print(f"  full-width runs present: {len(present)}/6 {present}")
    print(f"    {'run':20s} {'rho_b':>6s} {'codiag':>7s} {'null':>6s} {'alpha (all wide)':>18s} {'alpha last3':>11s} {'d_1':>5s} {'d_2':>5s} {'d_hat':>6s} "
          f"{'d_1@.95':>7s} {'d_2@.95':>7s} {'a_HZ':>5s} {'a_ZG':>5s} {'alpha_win':>9s}")
    S = {}
    for run in [r for r in runs if (df.run == r).any()]:
        d = df[df.run == run]; w = d[d.C >= WIDE]
        last3 = d.sort_values("depth").tail(3)
        S[run] = dict(rho_b=w.rho_b.median(), codiag=w.codiag.median(), null=w.codiag_null.median(), a_min=w.alpha_hat.min(), a_max=w.alpha_hat.max(),
                      a_med=w.alpha_hat.median(), a_last3=last3.alpha_hat.median(), d1=d_alpha(d, "kG1"), d2=d_alpha(d, "kG2"), dhat=d_alpha(d, "kGhat"),
                      d1_95=d_alpha(d, "kG1", 0.95), d2_95=d_alpha(d, "kG2", 0.95),
                      a_HZ=w.a_HZ.median() if "a_HZ" in w else np.nan, a_ZG=w.a_ZG.median() if "a_ZG" in w else np.nan,
                      a_win=w.alpha_hat_win.median() if "alpha_hat_win" in w else np.nan)
        s = S[run]
        print(f"    {run:20s} {s['rho_b']:+6.2f} {s['codiag']:7.2f} {s['null']:6.2f} {s['a_med']:6.2f} [{s['a_min']:.2f},{s['a_max']:.2f}] {s['a_last3']:11.2f} "
              f"{s['d1']:5.2f} {s['d2']:5.2f} {s['dhat']:6.2f} {s['d1_95']:7.2f} {s['d2_95']:7.2f} {s['a_HZ']:5.2f} {s['a_ZG']:5.2f} {s['a_win']:9.2f}")
    im = pd.concat([imagenet_exponents(results, n) for n in IMAGENET + tuple(f"{n}_random" for n in IMAGENET)], ignore_index=True)
    IM = {}
    if not im.empty:
        im.to_csv(os.path.join(out, "pah_imagenet.csv"), index=False)
        print("  ImageNet public weights, exponents along the singular directions of W (layer medians)")
        for net, d in im.groupby("net", sort=False):
            IM[net] = dict(a_HZ=d.a_HZ.median(), a_ZG=d.a_ZG.median(), a_HG=d.a_HG.median())
            print(f"    {net:20s} a_HZ {IM[net]['a_HZ']:5.2f}   a_ZG {IM[net]['a_ZG']:5.2f}   a_HG {IM[net]['a_HG']:5.2f}   layers {len(d)}")
    rm = rank_matched(results)
    if not rm.empty:
        rm.to_csv(os.path.join(out, "pah_rank_matched.csv"), index=False)
        for t in TAUS:
            e1 = np.abs(np.log(rm[f"kS_{t}"]) - np.log(rm[f"kG1_{t}"])); e2 = np.abs(np.log(rm[f"kS_{t}"]) - np.log(rm[f"kG2_{t}"]))
            print(f"  rank-matched, 21 checkpoints x last three layers, tau={t}: median |dlog| power 1 {e1.median():.2f}, power 2 {e2.median():.2f}; "
                  f"power 2 closer in {int((e2 < e1).sum())}/{len(rm)}")
    if len(present) == 6:
        ref = [f"wd{REF_LAM}_s{s}" for s in (0, 1)]
        in_range = lambda x, lo, hi: bool(np.isfinite(x) and lo <= x <= hi)                                   # noqa: E731
        V["P-b8.1"] = all(S[r]["rho_b"] >= 0.8 and S[r]["codiag"] >= 0.5 for r in FULL_RUNS)
        V["P-b8.2"] = all(S[r]["a_min"] >= 1 / 3 and S[r]["a_max"] <= 3 for r in FULL_RUNS) and all(in_range(S[r]["a_last3"], 1.5, 2.5) for r in ref)
        V["P-b8.3"] = all(S[r]["d2"] < S[r]["d1"] for r in FULL_RUNS) and all(S[r]["d2"] <= 0.35 for r in ref)
        V["P-b8.4"] = (all(in_range(S[r]["a_HZ"], 1.5, 2.5) and in_range(S[r]["a_ZG"], 0.67, 1.5) for r in ref)
                       and all(n in IM and in_range(IM[n]["a_HZ"], 1.5, 2.5) and in_range(IM[n]["a_ZG"], 0.67, 1.5) for n in IMAGENET))
        say = lambda k: "holds" if V[k] else "fails"                                                          # noqa: E731
        print(f"  P-b8.1 (shared basis: wide-layer median Spearman >= +0.8 and co-diagonal fraction >= 0.5, 6/6) -> {say('P-b8.1')}")
        print(f"  P-b8.2 (exponent in [1/3, 3] in every wide layer, 6/6; last-three median in [1.5, 2.5] at lambda 5e-4, 2/2) -> {say('P-b8.2')}")
        print(f"  P-b8.3 (width from Gamma^2 closer than from Gamma, 6/6; within 0.35 in log at lambda 5e-4, 2/2) -> {say('P-b8.3')}")
        print(f"  P-b8.4 (a_HZ in [1.5, 2.5] and a_ZG in [0.67, 1.5]: lambda 5e-4 runs 2/2 and ImageNet VGG-16/ResNet-18 2/2) -> {say('P-b8.4')}")
    else:
        print("  verdicts withheld until the six full-width runs are measured")
    json.dump({"verdicts": V, "runs": S, "imagenet": IM}, open(os.path.join(out, "pah_judgement.json"), "w"), indent=1, default=float)
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
