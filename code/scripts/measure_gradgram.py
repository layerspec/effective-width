"""B7 (theory-chain note, 2026-09-30): is the late-layer cap kappa the width of the position-resolved output gradients?
For every convolution of the 21 CIFAR checkpoints of A35 (VGG-16_BN; CIFAR-10 full, the class-count arms of A24, the long
arm of A25), on 3,200 of the task's own training images and 16 random positions per image (n/C = 100 at C = 512):

  Sigma_out  channel covariance of the convolution output at the sampled positions (the object of the paper)
  Gamma      second-moment matrix of the loss gradient with respect to the convolution output at the same positions,
             BN in training mode (the column space of the weight's gradient lies in the span of these vectors)

and their tau-widths k*(0.95), k*(0.999).  Wrapper over measure_stationarity (loader, checkpoints); nothing in layerspec
or the training scripts is modified.  Resumable: a finished checkpoint is skipped.

    python scripts/measure_gradgram.py --out ../results/gradgram --device cpu
    python scripts/measure_gradgram.py --judge --out ../results/gradgram
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.measure_stationarity import ARMS, SEEDS, labelled_loader, conv_layers, _done   # noqa: E402

POSITIONS = 16
N_IMAGES = 3200          # 3,200 x 16 positions = 51,200 samples: n/C = 100 at C = 512 (gate n/C >= 50); 6,400 would take ~66 min per checkpoint
TAUS = (0.95, 0.999)


def k_star(eig: np.ndarray, tau: float) -> int:
    e = np.clip(np.sort(eig)[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / max(e.sum(), 1e-300), tau) + 1)


class _Second:
    """Running C x C second-moment (and mean) accumulator over sampled position vectors."""

    def __init__(self, C: int):
        self.M = torch.zeros(C, C, dtype=torch.float64); self.s = torch.zeros(C, dtype=torch.float64); self.n = 0

    def add(self, v: torch.Tensor):                     # v: (m, C)
        v = v.double()
        self.M += v.T @ v; self.s += v.sum(0); self.n += v.shape[0]

    def moment(self) -> torch.Tensor:
        return self.M / max(self.n, 1)

    def cov(self) -> torch.Tensor:
        mu = self.s / max(self.n, 1)
        return self.M / max(self.n, 1) - torch.outer(mu, mu)


def _sample(t: torch.Tensor, gen: torch.Generator, k: int = POSITIONS) -> torch.Tensor:
    """(B, C, H, W) -> (B*k, C): k random positions per image, the same positions for the activation and its gradient
    when called with a generator in the same state."""
    B, C, H, W = t.shape
    pos = torch.randint(0, H * W, (B, k), generator=gen)
    flat = t.reshape(B, C, H * W)
    return flat.gather(2, pos.unsqueeze(1).expand(B, C, k)).permute(0, 2, 1).reshape(B * k, C)


def gradient_grams(model, loader, device: str, seed: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Per convolution: (Sigma_out, Gamma) accumulated over the loader, BN in training mode, CE loss."""
    model.train(True)
    layers = conv_layers(model)
    out_acc = {n: None for n, _ in layers}; grad_acc = {n: None for n, _ in layers}
    state = {}

    def fwd_hook(name):
        def f(m, inp, out):
            state[name] = out                            # keep the output tensor to attach a gradient hook
            out.register_hook(lambda g: state.__setitem__(name + "::grad", g.detach()))
        return f
    handles = [m.register_forward_hook(fwd_hook(n)) for n, m in layers]
    gen = torch.Generator().manual_seed(seed)
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        for n, m in layers:
            a = state[n].detach().cpu(); g = state[n + "::grad"].cpu()
            st = gen.get_state()
            va = _sample(a, gen); gen.set_state(st); vg = _sample(g, gen)
            if out_acc[n] is None:
                out_acc[n] = _Second(a.shape[1]); grad_acc[n] = _Second(a.shape[1])
            out_acc[n].add(va); grad_acc[n].add(vg)
        state.clear()
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    return {n: out_acc[n].cov() for n, _ in layers}, {n: grad_acc[n].moment() for n, _ in layers}


def measure_checkpoint(ckpt: str, dataset: str, kw: dict, n_classes: int, data: str, device: str) -> tuple[pd.DataFrame, dict]:
    from scripts.reproduce_garg import build_vgg16_bn_cifar
    model = build_vgg16_bn_cifar(n_classes, "small")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    model.to(device)
    loader, K = labelled_loader(data, dataset, kw.get("classes"), kw.get("per_class"), n_images=N_IMAGES)
    sig, gam = gradient_grams(model, loader, device)
    rows, spectra = [], {}
    for i, (n, m) in enumerate(conv_layers(model)):
        es = np.clip(np.linalg.eigvalsh(sig[n].numpy())[::-1], 0, None); eg = np.clip(np.linalg.eigvalsh(gam[n].numpy())[::-1], 0, None)
        C = int(m.out_channels)
        row = dict(index=i, layer=n, C=C, K=K, n=loader.dataset.__len__() * POSITIONS)
        for tau in TAUS:
            row[f"k{tau}_out"] = k_star(es, tau); row[f"k{tau}_grad"] = k_star(eg, tau)
        rows.append(row); spectra[n + "::out"] = es; spectra[n + "::grad"] = eg
    return pd.DataFrame(rows), spectra


def run(a) -> None:
    out = os.path.join(a.out, "gradgram.csv")
    os.makedirs(a.out, exist_ok=True)
    for arm, (dataset, kw, pattern, n_classes) in ARMS.items():
        for s in SEEDS:
            if _done(out, arm, s):
                print(f"=== {arm} seed {s} done, skipping ===", flush=True); continue
            ckpt = os.path.join(a.results, pattern.format(s=s))
            if not os.path.exists(ckpt):
                print(f"=== {arm} seed {s}: missing {ckpt} ===", flush=True); continue
            t0 = time.time()
            df, spectra = measure_checkpoint(ckpt, dataset, kw, n_classes, a.data, a.device)
            df.insert(0, "seed", s); df.insert(0, "arm", arm)
            np.savez_compressed(os.path.join(a.out, f"{arm}_s{s}_spectra.npz"), **spectra)
            old = pd.read_csv(out) if os.path.exists(out) else None
            (df if old is None else pd.concat([old, df], ignore_index=True)).to_csv(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            late = df.iloc[-3:]
            print(f"=== {arm} seed {s}: late k0.999 out/C {[round(v, 2) for v in late['k0.999_out'] / late.C]} "
                  f"grad/C {[round(v, 2) for v in late['k0.999_grad'] / late.C]} ({time.time() - t0:.0f} s) ===", flush=True)


def judge(a) -> dict:
    """P-b7.2 and P-b7.3 (theory-chain note, B7): the late-layer width follows the width of the output gradients."""
    from scipy.stats import spearmanr
    p = os.path.join(a.out, "gradgram.csv")
    print("==== B7 on the data: output covariance width against output-gradient width ====")
    if not os.path.exists(p):
        print("  nothing measured"); return {}
    d = pd.read_csv(p)
    late = d[d.groupby(["arm", "seed"])["index"].transform(lambda s: s >= s.max() - 2)].copy()
    V = {}
    for tau in TAUS:
        late[f"o{tau}"] = late[f"k{tau}_out"] / late.C; late[f"g{tau}"] = late[f"k{tau}_grad"] / late.C
        rho = spearmanr(late[f"o{tau}"], late[f"g{tau}"]).correlation
        print(f"  tau={tau}: late-layer rows {len(late)}, Spearman(out width, grad width) = {rho:+.2f}; "
              f"grad width >= 3K in {(late[f'k{tau}_grad'] >= 3 * late.K).sum()}/{len(late)}")
        if tau == 0.999:
            V["P-b7.2"] = bool(rho >= 0.7 and (late["k0.999_grad"] >= 3 * late.K).mean() >= 0.9)
    e780 = late[late.arm == "c10_pc640_e780"]; short = late[late.arm == "c10_pc640"]
    if len(e780) and len(short):
        a_, b_ = e780["k0.999_grad"].median(), short["k0.999_grad"].median()
        V["P-b7.3"] = bool(a_ < b_)
        print(f"  P-b7.3: late k0.999(Gamma) median 39,000 steps {a_:.0f} vs 5,000 steps {b_:.0f} -> {'holds' if V['P-b7.3'] else 'fails'}")
    print(f"  P-b7.2 -> {'holds' if V.get('P-b7.2') else 'fails'}")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../results/gradgram")
    p.add_argument("--device", default="cpu")
    p.add_argument("--judge", action="store_true")
    a = p.parse_args(argv)
    if a.judge:
        judge(a); return 0
    run(a); return 0


if __name__ == "__main__":
    sys.exit(main())
