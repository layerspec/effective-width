"""B9, part B (plan 9.29): split the position-sampled covariance of the late convolutions by (class, position) cells.

For every checkpoint of the A34 arms (VGG-16_BN on CIFAR-10 and K-class subsets of CIFAR-100), on 6,400 training images of its
own task, BN in training mode, ALL positions of the map (block 5: 2 x 2, P = 4; block 4: 4 x 4, P = 16):

    Sigma = B_cp + W_cp      cells = class x position  (rank B_cp <= P K - 1)
            B_c               cells = class             (rank <= K - 1)
            B_p               cells = position          (rank <= P - 1)
    w_cp = tr W_cp / tr Sigma,  w_c = 1 - tr B_c / tr Sigma,  w_p = 1 - tr B_p / tr Sigma

Lemma B9.0: w_cp <= 1 - tau  =>  k*_tau(Sigma) <= P K - 1; sharper (same proof): it is enough that the variance outside the span\nof the cell means, w_out = tr((I - Pi) Sigma) / tr Sigma, is <= 1 - tau.  Forward passes only, no training.

    python scripts/measure_position_cells.py --results ../results --data ../data --device cpu
    python scripts/measure_position_cells.py --results ../results --judge          # P-b9.4 .. P-b9.6
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_stationarity import ARMS, SEEDS, labelled_loader      # noqa: E402
from scripts.position_collapse_judge import cliff, k_star                   # noqa: E402

LAYERS = ("features.24", "features.27", "features.30", "features.34", "features.37", "features.40")
BLOCK5 = LAYERS[3:]
TRAINED_K10 = ("c10_full", "c10_pc640_e780")


def cell_decomposition(X: np.ndarray, cls: np.ndarray, pos: np.ndarray, chunk: int = 20000) -> dict:
    """X: (n, C) samples (float32 is fine) with integer class and position labels.  Trace fractions and widths of the parts."""
    n, C = X.shape
    m = np.zeros(C); M2 = np.zeros((C, C))
    for a in range(0, n, chunk):
        xb = X[a:a + chunk].astype(np.float64)
        m += xb.sum(0); M2 += xb.T @ xb
    m /= n
    S = M2 / n - np.outer(m, m)

    def between(labels):
        vals, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind="stable")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        sums = np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
        D = sums / counts[:, None] - m
        return (D * (counts / n)[:, None]).T @ D
    K, P = int(cls.max()) + 1, int(pos.max()) + 1
    B_cp, B_c, B_p = between(cls * P + pos), between(cls), between(pos)
    tr = np.trace(S)
    eS = np.linalg.eigvalsh(S); eB, UB = np.linalg.eigh(B_cp)
    r = min(P * K - 1, C)                                            # span of the cell means (rank <= PK - 1)
    Pi = UB[:, np.argsort(-eB)[:r]]
    w_out = float(1 - np.trace(Pi.T @ S @ Pi) / tr)                  # variance outside that span: the quantity Ky Fan needs
    g, ratio = cliff(eS, min(K, len(eS) // 2))
    return {"C": C, "n": n, "K": K, "P": P, "PK": P * K,
            "w_cp": float(1 - np.trace(B_cp) / tr), "w_c": float(1 - np.trace(B_c) / tr), "w_p": float(1 - np.trace(B_p) / tr), "w_out": w_out,
            "k0.95": k_star(eS, 0.95), "k0.999": k_star(eS, 0.999), "kB_cp_0.999": k_star(eB, 0.999), "kB_c_0.999": k_star(np.linalg.eigvalsh(B_c), 0.999),
            "cliff_g": g, "cliff_ratio": ratio}


@torch.no_grad()
def measure_checkpoint(ckpt: str, dataset: str, kw: dict, n_classes: int, data: str, device: str) -> pd.DataFrame:
    from scripts.reproduce_garg import build_vgg16_bn_cifar
    model = build_vgg16_bn_cifar(n_classes, "small")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    model.to(device).train()                                     # batch statistics, as in measure_gradgram
    loader, K = labelled_loader(data, dataset, kw.get("classes"), kw.get("per_class"))
    mods = dict(model.named_modules())
    store = {n: [] for n in LAYERS}
    hooks = [mods[n].register_forward_hook(lambda m, i, o, n=n: store[n].append(o.detach().to("cpu", torch.float32))) for n in LAYERS]
    labels = []
    for x, y in loader:
        model(x.to(device)); labels.append(torch.as_tensor(y))
    for h in hooks:
        h.remove()
    y = torch.cat(labels).numpy()
    rows = []
    for n in LAYERS:
        t = torch.cat(store[n])                                  # (N, C, H, W)
        N, C, H, W = t.shape
        X = t.permute(0, 2, 3, 1).reshape(N * H * W, C).numpy()
        cls = np.repeat(y, H * W); pos = np.tile(np.arange(H * W), N)
        rows.append({"layer": n, "H": H, **cell_decomposition(X, cls, pos)})
        store[n] = None
    return pd.DataFrame(rows)


def run(a) -> None:
    out_dir = os.path.join(a.results, "position_collapse"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "cells.csv")
    for arm, (dataset, kw, pattern, n_classes) in ARMS.items():
        for s in SEEDS:
            old = pd.read_csv(out) if os.path.exists(out) else None
            if old is not None and ((old.arm == arm) & (old.seed == s)).any():
                continue
            ckpt = os.path.join(a.results, pattern.format(s=s))
            if not os.path.exists(ckpt):
                print(f"=== {arm} seed {s}: missing {ckpt} ===", flush=True); continue
            t0 = time.time()
            df = measure_checkpoint(ckpt, dataset, kw, n_classes, a.data, a.device)
            df.insert(0, "seed", s); df.insert(0, "arm", arm)
            (df if old is None else pd.concat([old, df], ignore_index=True)).to_csv(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            last = df.iloc[-1]
            print(f"=== {arm} seed {s}: features.40 w_cp {last.w_cp:.4f} w_c {last.w_c:.3f} w_out {last.w_out:.5f} k0.999 {last['k0.999']} kB {last['kB_cp_0.999']} "
                  f"({time.time() - t0:.0f} s) ===", flush=True)


def judge(results: str) -> dict:
    path = os.path.join(results, "position_collapse", "cells.csv")
    V: dict = {}
    print("==== B9 part B (plan 9.29): the covariance split by (class, position) cells ====")
    if not os.path.exists(path):
        print("  no measurements"); return V
    df = pd.read_csv(path)
    print(f"    {'arm':16s} {'layer':12s} {'P':>3s} {'PK':>5s}  w_cp (seeds)            w_c (seeds)             w_p     k*(0.999) Sigma   k*(0.999) B_cp   cliff")
    for (arm, layer), d in df.groupby(["arm", "layer"], sort=False):
        d = d.sort_values("seed")
        print(f"    {arm:16s} {layer:12s} {d.P.iloc[0]:3d} {d.PK.iloc[0]:5d}  {' '.join(f'{v:.4f}' for v in d.w_cp):22s}  {' '.join(f'{v:.3f}' for v in d.w_c):22s} "
              f"{d.w_p.median():6.3f}   {' '.join(map(str, d['k0.999'])):14s}   {' '.join(map(str, d['kB_cp_0.999'])):14s}  {' '.join(f'{g}({r:.0f})' for g, r in zip(d.cliff_g, d.cliff_ratio))}")
    t = df[df.arm.isin(TRAINED_K10) & df.layer.isin(BLOCK5)]
    if len(t) == 18:
        last = t[t.layer == "features.40"]
        V["P-b9.4"] = bool((last.w_cp <= 0.02).all()) and int((t.w_cp <= 0.10).sum()) >= 15
        V["P-b9.5"] = int((t.w_c >= 5 * t.w_cp).sum()) >= 15
        V["P-b9.6"] = int((t["kB_cp_0.999"] >= 36).sum()) >= 15
        say = lambda k: "holds" if V[k] else "fails"                                                          # noqa: E731
        print(f"  P-b9.4 (K = 10 trained: last conv w_cp <= 0.02 in 6/6 [{int((last.w_cp <= 0.02).sum())}/6]; block 5 w_cp <= 0.10 in >= 15/18 [{int((t.w_cp <= 0.10).sum())}/18]) -> {say('P-b9.4')}")
        print(f"  P-b9.5 (class-only cells leave at least 5 times the within-cell variance, >= 15/18 [{int((t.w_c >= 5 * t.w_cp).sum())}/18]) -> {say('P-b9.5')}")
        print(f"  P-b9.6 (cell means linearly independent: k*(0.999) of B_cp >= 36, >= 15/18 [{int((t['kB_cp_0.999'] >= 36).sum())}/18]) -> {say('P-b9.6')}")
    else:
        print(f"  verdicts withheld: {len(t)}/18 layers of the K = 10 trained arms measured")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="../results")
    p.add_argument("--data", default="../data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--judge", action="store_true")
    a = p.parse_args(argv)
    if a.judge:
        judge(a.results)
    else:
        run(a); judge(a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
