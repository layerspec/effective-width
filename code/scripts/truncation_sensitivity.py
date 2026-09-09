"""A19: the first-order truncation sensitivity of Proposition (truncation), per layer.

For every dense convolution output z_l (all positions), with Sigma_l accumulated
on calibration images and U_k its top-k principal directions at k = k*(tau):

  linear   S_lin(tau)  = E ||J_l r||^2 with r the actual discarded component
                         -(I - U_k U_k^T)(z - mean) at every position of the image,
                         by a finite difference along r (eps = 1e-3): the
                         first-order prediction of the truncation, positions
                         and their correlations included
  actual   S_act(tau)  = E ||f(x; z_l truncated) - f(x)||^2 over eval images:
                         what truncating that one layer actually does to the logits
  margin   m(x) = logit_top1 - logit_top2 on the clean network, per image

Output: <out>/<model>_sensitivity.csv with one row per (layer, tau), and the
margin distribution in <out>/<model>_margins.csv.  Compare S_lin with S_act
(the first-order formula), rank layers by S_act (which truncation hurts), and
set sum_l S_l(tau) beside the margins (where accuracy should start to fall).

    python scripts/truncation_sensitivity.py --model timm:resnet50.a1_in1k \
        --data ../data/imagenet_val_6400 --device cpu --out ../results/sensitivity
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.data import build_loader          # noqa: E402
from layerspec.hooks import SpectrumProbe          # noqa: E402
from layerspec.models import build, file_stem      # noqa: E402


def kstar(lam: np.ndarray, tau: float) -> int:
    lam = np.clip(lam, 0, None)
    c = np.cumsum(lam) / lam.sum()
    return int(np.searchsorted(c, tau) + 1)


class Perturb:
    """One forward hook on one conv module: either truncates its output to the
    top-k principal channel directions, or adds a Gaussian draw from the
    discarded covariance (scaled by eps) at every position."""

    def __init__(self, module: nn.Module, mean, V, k, device):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)       # (C,)
        self.V = torch.as_tensor(V, dtype=torch.float32, device=device)             # (C, C) eigvecs, ascending
        self.k = k
        self.mode = None
        self.eps = 1e-3
        self.sqrt_perp = None                                                        # (C, C-k)
        self.gen = torch.Generator(device="cpu").manual_seed(0)
        self.h = module.register_forward_hook(self._hook)

    def set_perp(self, lam):
        C = self.V.shape[0]
        tail = torch.as_tensor(np.sqrt(np.clip(lam[: C - self.k], 0, None)), dtype=torch.float32, device=self.V.device)
        self.sqrt_perp = self.V[:, : C - self.k] * tail                               # columns: u_i sqrt(lambda_i), i > k

    def _hook(self, _m, _i, out):
        if self.mode is None:
            return None
        N, C, H, W = out.shape
        z = out.permute(0, 2, 3, 1).reshape(-1, C)                                   # (NHW, C)
        if self.mode == "truncate":
            Uk = self.V[:, C - self.k:]                                              # top-k (ascending order)
            zc = z - self.mean
            z2 = self.mean + (zc @ Uk) @ Uk.T
        else:                                                                        # "linear": z + eps * (truncated - z)
            Uk = self.V[:, C - self.k:]
            zc = z - self.mean
            z2 = z + self.eps * ((zc @ Uk) @ Uk.T - zc)
        return z2.reshape(N, H, W, C).permute(0, 3, 1, 2)

    def remove(self):
        self.h.remove()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="timm:resnet50.a1_in1k")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="cpu")
    p.add_argument("--calib", type=int, default=256)
    p.add_argument("--eval", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--taus", type=float, nargs="+", default=[0.95, 0.99, 0.999])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", default="../results/sensitivity")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)

    model, tag = build(a.model, pretrained=True, seed=a.seed)
    model = model.to(a.device).eval()
    loader, _ = build_loader(a.data, a.batch_size, a.workers, image_size=a.image_size,
                             limit=a.calib + a.eval, seed=a.seed)
    batches = [(b[0] if isinstance(b, (list, tuple)) else b) for b in loader]
    xs = torch.cat(batches)
    x_cal, x_eval = xs[: a.calib], xs[a.calib: a.calib + a.eval]

    # covariances of every dense conv output on the calibration images
    t0 = time.time()
    probe = SpectrumProbe(model, positions_per_image=a.positions, include_activations=False,
                          include_linear=False, pooled=False, include_blocks=False, seed=a.seed)
    with torch.no_grad():
        for i in range(0, len(x_cal), a.batch_size):
            model(x_cal[i: i + a.batch_size].to(a.device))
    recs = [r for r in probe.records() if r.kind == "conv" and not r.is_depthwise and r.acc.n >= 2]
    probe.remove()
    mods = dict(model.named_modules())
    print(f"{a.model}: {len(recs)} dense conv layers, covariances in {time.time() - t0:.0f}s", flush=True)

    def logits(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), a.batch_size):
                out.append(model(x[i: i + a.batch_size].to(a.device)).float().cpu())
        return torch.cat(out)

    clean = logits(x_eval)
    top2 = clean.topk(2, dim=1).values
    margins = (top2[:, 0] - top2[:, 1]).numpy()
    pd.DataFrame({"margin": margins}).to_csv(os.path.join(a.out, f"{file_stem(a.model)}_margins.csv"), index=False)

    rows = []
    for rec in recs:
        cov = rec.acc.covariance
        lam, V = np.linalg.eigh(cov)                      # ascending
        lam_desc = lam[::-1]
        for tau in a.taus:
            k = kstar(lam_desc, tau)
            pert = Perturb(mods[rec.name], rec.acc.mean, V, k, a.device)
            pert.set_perp(lam)                            # ascending: the first C-k are the discarded ones
            pert.mode = "truncate"
            d_act = logits(x_eval) - clean
            s_act = float((d_act ** 2).sum(1).mean())
            pert.mode = "linear"
            d = (logits(x_eval) - clean) / pert.eps
            s_lin = [float((d ** 2).sum(1).mean())]
            pert.remove()
            resid = float(lam_desc[k:].sum())
            rows.append(dict(model=a.model, layer=rec.name, depth_index=rec.depth_index, C=rec.C, r_max=rec.r_max,
                             tau=tau, k=k, resid_var=resid, resid_frac=resid / float(lam_desc.sum()),
                             S_lin=float(np.mean(s_lin)), S_lin_sd=float(np.std(s_lin)), S_act=s_act,
                             n_eval=len(x_eval), n_calib=len(x_cal)))
        print(f"  {rec.name:24s} " + "  ".join(f"tau={r['tau']:g}: k={r['k']:4d} S_lin={r['S_lin']:9.3g} S_act={r['S_act']:9.3g}"
                                            for r in rows[-len(a.taus):]), flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(a.out, f"{file_stem(a.model)}_sensitivity.csv")
    df.to_csv(path, index=False)
    for tau in a.taus:
        d = df[df.tau == tau]
        from scipy.stats import spearmanr
        rho = spearmanr(d.S_lin, d.S_act).correlation
        print(f"tau={tau:g}: sum_l S_lin = {d.S_lin.sum():.3g}, sum_l S_act = {d.S_act.sum():.3g}, "
              f"rho(S_lin, S_act) over layers = {rho:+.2f}; median margin^2 = {np.median(margins) ** 2:.3g}", flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
