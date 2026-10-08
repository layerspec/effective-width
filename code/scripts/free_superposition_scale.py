"""Target (d2), plan 9.44: does the gap between the free-convolution prediction and the measured
stream width at the residual addition shrink with the channel count C?

For every block of the gamma = 1 random networks measured in 9.21 / 9.34 (x = stream entering the
addition, f = branch, y = x + f before the ReLU; spectra stored per block) we compute

    k_free  = k_tau of diag(lam_x) + O diag(lam_f) O^T averaged over Haar O    (random-rotation superposition)
    delta   = 1 - k_tau(y measured) / k_free

together with the exact bracket that needs no assumption on the relative position of the two
eigenbases (cross terms set to zero):

    lower   = min{k : tail_x(k) + tail_f(k) <= (1 - tau) (tr x + tr f)}     (Ky Fan norm triangle inequality)
    upper   = min{k1 + k2 : tail_x(k1) + tail_f(k2) <= (1 - tau) (tr x + tr f)}   (Lemma B11.0)

Outputs results/free_scale/blocks.csv and, with --judge, P9.105 / P9.106.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

FILES = {
    "resnet18": "residual/timm_resnet18.tv_in1k_random_s1",
    "resnet34": "residual_sign/timm_resnet34.tv_in1k_random_s1",
    "resnet50": "residual/timm_resnet50.tv_in1k_random_s1",
    "resnext50": "residual_sign/timm_resnext50_32x4d.tv_in1k_random_s1",
}
FAMILY = {"resnet18": "basic", "resnet34": "basic", "resnet50": "bottleneck", "resnext50": "grouped"}
GATE = 50.0


def k_star(eig: np.ndarray, tau: float) -> int:
    e = np.clip(np.sort(np.asarray(eig, dtype=float))[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / e.sum(), tau) + 1)


def tails(eig: np.ndarray) -> np.ndarray:
    """tail[k] = sum of eigenvalues beyond the top k, k = 0..C."""
    e = np.clip(np.sort(np.asarray(eig, dtype=float))[::-1], 0, None)
    c = np.concatenate([[0.0], np.cumsum(e)])
    return e.sum() - c


def exact_bracket(lx: np.ndarray, lf: np.ndarray, tau: float) -> tuple[int, int]:
    """Assumption-free bounds on k_tau(Sigma_x + Sigma_f) when the cross terms vanish."""
    tx, tf = tails(lx), tails(lf)
    budget = (1.0 - tau) * (tx[0] + tf[0])
    c = len(tx) - 1
    lower = int(np.argmax(tx + tf <= budget))          # first k with tail_x(k) + tail_f(k) <= budget
    # upper: minimise k1 + k2 subject to tail_x(k1) + tail_f(k2) <= budget
    best = 2 * c
    for k1 in range(c + 1):
        rem = budget - tx[k1]
        if rem < 0:
            continue
        k2 = int(np.argmax(tf <= rem))
        best = min(best, k1 + k2)
    return max(lower, 1), max(best, 1)


def haar(c: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((c, c)))
    return q * np.sign(np.diag(r))


def free_width(lx: np.ndarray, lf: np.ndarray, tau: float, draws: int, rng: np.random.Generator,
               taus_extra: tuple[float, ...] = ()) -> dict:
    lx = np.clip(np.sort(np.asarray(lx, dtype=float))[::-1], 0, None)
    lf = np.clip(np.sort(np.asarray(lf, dtype=float))[::-1], 0, None)
    out = {tau: [], **{t: [] for t in taus_extra}}
    for _ in range(draws):
        o = haar(len(lx), rng)
        m = np.diag(lx) + (o * lf) @ o.T
        ev = np.linalg.eigvalsh(m)
        for t in out:
            out[t].append(k_star(ev, t))
    return {t: (float(np.mean(v)), float(np.std(v))) for t, v in out.items()}


def compute(results: str, tau: float = 0.95, draws: int = 5, seed: int = 0, models=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, stem in FILES.items():
        if models and name not in models:
            continue
        npz = os.path.join(results, stem + "_spectra.npz")
        csv = os.path.join(results, stem + "_blocks.csv")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        meta = pd.read_csv(csv).set_index("block") if os.path.exists(csv) else None
        for key in [k for k in z.keys() if k.endswith("::x")]:
            blk = key[:-3]
            lx, lf, ly = z[blk + "::x"], z[blk + "::f"], z[blk + "::y"]
            c = len(lx)
            kx, kf, ky = k_star(lx, tau), k_star(lf, tau), k_star(ly, tau)
            fw = free_width(lx, lf, tau, draws, rng, taus_extra=(0.999,))
            kfree, kfree_sd = fw[tau]
            lo, up = exact_bracket(lx, lf, tau)
            row = dict(model=name, family=FAMILY[name], block=blk, stage=int(blk[5]), block_index=int(blk.split(".")[1]),
                       C=c, k_x=kx, k_f=kf, k_y=ky, k_free=kfree, k_free_sd=kfree_sd,
                       delta=1.0 - ky / kfree, abs_delta=abs(1.0 - ky / kfree),
                       lower=lo, upper=up, free_in_bracket=bool(lo - 1 <= kfree <= up + 1),
                       k_y999=k_star(ly, 0.999), k_free999=fw[0.999][0],
                       energy_f_over_x=float(np.clip(lf, 0, None).sum() / np.clip(lx, 0, None).sum()))
            if meta is not None and blk in meta.index:
                row["downsample"] = bool(meta.loc[blk, "downsample"])
                row["ok"] = bool(float(meta.loc[blk, "n_over_2C"]) >= GATE)
            else:
                row["downsample"] = False
                row["ok"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def judge(df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    out = {}
    d = df[df.ok]
    print("P9.105 (within each ResNet family, the stage median |delta| falls with C: Spearman <= -0.8 over 4 stages and widest <= half of narrowest)")
    passes = []
    for m in ["resnet18", "resnet34", "resnet50"]:
        g = d[d.model == m].groupby("stage").agg(C=("C", "first"), med=("abs_delta", "median"), n=("abs_delta", "size")).reset_index()
        if len(g) < 4:
            print(f"  {m}: only {len(g)} stages"); passes.append(False); continue
        rho = spearmanr(g.C, g.med).correlation
        ratio = g.med.iloc[-1] / g.med.iloc[0] if g.med.iloc[0] > 0 else np.inf
        ok = (rho <= -0.8) and (ratio <= 0.5)
        passes.append(bool(ok))
        print(f"  {m}: " + ", ".join(f"C={int(r.C)}:{r.med:+.3f}(n={int(r.n)})" for r in g.itertuples()) + f"  rho={rho:+.2f} ratio={ratio:.2f} -> {'holds' if ok else 'fails'}")
    out["P9.105"] = sum(passes) >= 2
    print(f"  -> {sum(passes)}/3 families: P9.105 {'HOLDS' if out['P9.105'] else 'FAILS'}")
    g = d[d.model == "resnext50"].groupby("stage").agg(C=("C", "first"), med=("abs_delta", "median")).reset_index()
    if len(g):
        print("  resnext50 (descriptive): " + ", ".join(f"C={int(r.C)}:{r.med:+.3f}" for r in g.itertuples()))
    print("P9.106 (same stage index: ResNet-50 median |delta| < ResNet-34, 4/4)")
    a = d[d.model == "resnet50"].groupby("stage").abs_delta.median()
    b = d[d.model == "resnet34"].groupby("stage").abs_delta.median()
    wins = [(s, float(a.get(s, np.nan)), float(b.get(s, np.nan))) for s in sorted(set(a.index) & set(b.index))]
    n_win = sum(1 for _, x, y in wins if x < y)
    for s, x, y in wins:
        print(f"  stage {s}: resnet50 {x:.3f} vs resnet34 {y:.3f} -> {'<' if x < y else '>='}")
    out["P9.106"] = n_win == 4 and len(wins) == 4
    print(f"  -> {n_win}/{len(wins)}: P9.106 {'HOLDS' if out['P9.106'] else 'FAILS'}")
    print(f"sanity: free value inside the exact bracket in {int(d.free_in_bracket.sum())}/{len(d)} blocks; "
          f"signed delta median by family: " + ", ".join(f"{f}:{v:+.3f}" for f, v in d.groupby('family').delta.median().items()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default="../results/free_scale")
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "blocks.csv")
    if a.judge and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = compute(a.results, a.tau, a.draws, a.seed)
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df)} blocks)")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(df[["model", "block", "C", "k_x", "k_f", "k_y", "k_free", "lower", "upper", "delta", "energy_f_over_x", "downsample"]].round(3).to_string(index=False))
    if a.judge:
        judge(df)


if __name__ == "__main__":
    main()
