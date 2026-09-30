"""Th9 / analysis-plan 9.21: the residual stream at the addition.  For every residual block of a timm ResNet, at the
same sampled positions:

    x   the shortcut entering the addition (after the downsample, where there is one)
    f   the branch entering the addition (output of the block's last BN)
    y   their sum, before the ReLU
    z   the block output, after the ReLU

The joint covariance of (x, f) gives Sigma_x, Sigma_f, the cross term and Sigma_y; z has its own.  Per block the script
writes the four widths, the canonical correlations of x and f, and the two floors of Proposition Th9:

    (a)  lambda_i(Sigma_y) >= lambda_{i+r}(Sigma_x),          r = attainable rank of the branch's last convolution
    (b)  lambda_i(Sigma_y) >= (1 - rho_max) lambda_i(Sigma_x), rho_max = largest canonical correlation of x and f

Both hold for any covariance, sample covariances included, so a violation is a bug in the measurement.  What the
data decide is how far the stream is from the floors.  A block is reported when n / (2C) >= 50.

    python scripts/measure_residual.py --out ../results/residual                # the registered list
    python scripts/measure_residual.py --judge --out ../results/residual
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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAINED = ["resnet18.tv_in1k", "resnet34.tv_in1k", "resnet50.tv_in1k", "resnet50.a1_in1k"]
RANDOM = [("resnet18.tv_in1k", 1), ("resnet50.tv_in1k", 1)]          # gamma = 1; with gamma = 0 the branch is zero
GATE = 50.0


def k_star(eig: np.ndarray, tau: float) -> int:
    e = np.clip(np.sort(eig)[::-1], 0, None)
    return int(np.searchsorted(np.cumsum(e) / e.sum(), tau) + 1)


def canonical_correlations(sx: np.ndarray, sf: np.ndarray, sxf: np.ndarray, rel: float = 1e-8) -> np.ndarray:
    """Singular values of Sigma_x^{-1/2} Sigma_xf Sigma_f^{-1/2} on the numerical supports of x and f."""
    def inv_sqrt(s):
        w, v = np.linalg.eigh(s)
        keep = w > rel * w.max()
        return v[:, keep] / np.sqrt(w[keep])
    a, b = inv_sqrt(sx), inv_sqrt(sf)
    return np.clip(np.linalg.svd(a.T @ sxf @ b, compute_uv=False), 0.0, 1.0)


def block_metrics(joint_cov: np.ndarray, z_cov: np.ndarray, r: int) -> dict:
    C = joint_cov.shape[0] // 2
    sx, sf, sxf = joint_cov[:C, :C], joint_cov[C:, C:], joint_cov[:C, C:]
    sy = sx + sf + sxf + sxf.T
    ex, ef, ey, ez = (np.clip(np.sort(np.linalg.eigvalsh(s))[::-1], 0, None) for s in (sx, sf, sy, z_cov))
    cc = canonical_correlations(sx, sf, sxf)
    rho = float(cc.max()) if len(cc) else 0.0
    cs = np.sort(np.linalg.svd(sxf, compute_uv=False))[::-1]          # singular values of the cross-covariance
    es = np.clip(np.sort(np.linalg.eigvalsh(sx + sf))[::-1], 0, None)
    shifted = np.concatenate([ex[r:], np.zeros(min(r, C))])[:C]
    kx = k_star(ex, 0.95)
    ratio = ey[:kx] / np.maximum(ex[:kx], 1e-300)
    return {"C": C, "r": int(r),
            "k95_x": kx, "k95_f": k_star(ef, 0.95), "k95_y": k_star(ey, 0.95), "k95_z": k_star(ez, 0.95),
            "k999_x": k_star(ex, 0.999), "k999_f": k_star(ef, 0.999), "k999_y": k_star(ey, 0.999), "k999_z": k_star(ez, 0.999),
            "rho_max": rho, "rho_median": float(np.median(cc)) if len(cc) else 0.0,
            "energy_f_over_x": float(ef.sum() / ex.sum()),
            "energy_corr": float(np.trace(sxf) / np.sqrt(np.trace(sx) * np.trace(sf))) if ef.sum() > 0 else 0.0,
            "floor_a_min": float(np.min(ey - shifted) / ex[0]),                 # >= 0 up to rounding
            "floor_b_min": float(np.min(ey - (1 - rho) * es) / ex[0]),           # >= 0 up to rounding
            "retention_at_k": float(ratio[-1]), "retention_min": float(ratio.min()),
            "floor_b_at_k": float(1 - rho),
            "b2_lb95": tau_width_lower_bound(ex, ef, cs, ey.sum(), 0.95),
            "b2_lb999": tau_width_lower_bound(ex, ef, cs, ey.sum(), 0.999)}, {"x": ex, "f": ef, "y": ey, "z": ez, "cc": cc, "cs": cs}


def tau_width_lower_bound(ex: np.ndarray, ef: np.ndarray, cs: np.ndarray, tr_y: float, tau: float) -> int:
    """Proposition B2 (cross-term form): k*_tau(Sigma_y) >= the smallest k with
    sum_{i<=k} lambda_i(Sigma_x) + sum_{i<=k} lambda_i(Sigma_f) + 2 sum_{i<=k} sigma_i(Sigma_xf) >= tau * tr(Sigma_y).
    Ky Fan: the k largest eigenvalues of Sigma_x + Sigma_f + C + C^T sum to at most that left-hand side, so every smaller k
    holds less than tau of the variance of y.  Returns a lower bound on k*_tau(Sigma_y) (1 when the bound is vacuous)."""
    C = len(ex)
    k = np.arange(1, C + 1)
    lhs = np.cumsum(ex) + np.cumsum(np.concatenate([ef, np.zeros(C - len(ef))])[:C]) + 2 * np.cumsum(np.concatenate([cs, np.zeros(C)])[:C])
    hit = np.nonzero(lhs >= tau * tr_y)[0]
    return int(k[hit[0]]) if len(hit) else C


def residual_blocks(model):
    """(name, block, last BN, last act, last conv) of every timm BasicBlock / Bottleneck."""
    out = []
    for name, m in model.named_modules():
        if hasattr(m, "bn3") and hasattr(m, "act3") and hasattr(m, "conv3"):
            out.append((name, m, m.bn3, m.act3, m.conv3))
        elif hasattr(m, "bn2") and hasattr(m, "act2") and hasattr(m, "conv2") and not hasattr(m, "conv3"):
            out.append((name, m, m.bn2, m.act2, m.conv2))
    return out


def r_max(conv) -> int:
    g = conv.groups
    return int(g * min(conv.in_channels // g * conv.kernel_size[0] * conv.kernel_size[1], conv.out_channels // g))


class Tap:
    """Samples f, y and z of one block at the same positions of every image."""

    def __init__(self, name, bn, act, conv, positions, gen):
        from layerspec.accumulate import CovarianceAccumulator
        self.name, self.positions, self.gen = name, positions, gen
        C = conv.out_channels
        self.r, self.joint, self.z = r_max(conv), CovarianceAccumulator(2 * C), CovarianceAccumulator(C)
        self.idx = self.f = self.y = None
        self.handles = [bn.register_forward_hook(self._branch), act.register_forward_pre_hook(self._sum),
                        act.register_forward_hook(self._out)]

    def _take(self, t):
        n, c, h, w = t.shape
        return t.detach().reshape(n, c, h * w).permute(0, 2, 1)[torch.arange(n)[:, None], self.idx].reshape(-1, c).double().cpu()

    def _branch(self, module, inputs, output):        # before the in-place addition touches this tensor
        n, _, h, w = output.shape
        p = min(self.positions, h * w)
        self.idx = torch.rand(n, h * w, generator=self.gen).argsort(1)[:, :p]
        self.f = self._take(output)

    def _sum(self, module, inputs):                   # before an in-place ReLU touches it
        self.y = self._take(inputs[0])

    def _out(self, module, inputs, output):
        x = self.y - self.f
        self.joint.update(torch.cat([x, self.f], 1).numpy())
        self.z.update(self._take(output).numpy())

    def cov(self, acc):
        return acc.M2 / (acc.n - 1)


def measure(base: str, seed, a) -> pd.DataFrame:
    import timm
    from layerspec.api import image_loader
    if seed is None:
        model, tag = timm.create_model(base, pretrained=True), f"timm/{base}"
    else:
        torch.manual_seed(seed)
        model, tag = timm.create_model(base, pretrained=False, zero_init_last=False), f"random_init(seed={seed}, zero_init_last=False)"
    model = model.to(a.device).eval()
    gen = torch.Generator().manual_seed(0 if seed is None else seed)
    taps = [Tap(n, bn, act, conv, a.positions, gen) for n, _, bn, act, conv in residual_blocks(model)]
    loader = image_loader(a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit)
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(a.device))
    rows, spectra = [], {}
    for i, t in enumerate(taps):
        m, s = block_metrics(t.cov(t.joint), t.cov(t.z), t.r)
        blk = dict(model.named_modules())[t.name]
        m.update({"block": t.name, "block_index": i, "downsample": blk.downsample is not None, "n": t.joint.n,
                  "n_over_2C": t.joint.n / (2 * m["C"]), "ok": t.joint.n / (2 * m["C"]) >= GATE, "weights": tag})
        rows.append(m)
        spectra.update({f"{t.name}::{k}": v for k, v in s.items()})
    for t in taps:
        for h in t.handles:
            h.remove()
    return pd.DataFrame(rows), spectra


def judge(out: str) -> dict:
    """P9.60-P9.62 (plan 9.21) over the gate-passing blocks of the trained networks."""
    print("==== Th9 on the data: the residual stream at the addition (plan 9.21) ====")
    V, frames = {}, []
    for f in sorted(glob.glob(os.path.join(out, "*_blocks.csv"))):
        d = pd.read_csv(f); d["model"] = os.path.basename(f)[:-11]
        frames.append(d)
    if not frames:
        print("  nothing measured"); return V
    d = pd.concat(frames, ignore_index=True)
    d = d[d.ok.astype(bool)]
    d["trained"] = ~d.model.str.contains("_random")
    print(f"  {'model':28s} {'blocks':>6s} {'k95 y>=x':>9s} {'k95 z/x med':>11s} {'rho_max med':>11s} {'rho_med med':>11s} "
          f"{'retention@k med':>15s} {'min':>6s} {'floor a min':>11s} {'floor b min':>11s}")
    for name, g in d.groupby("model", sort=False):
        print(f"  {name:28s} {len(g):6d} {int((g.k95_y >= g.k95_x).sum()):4d}/{len(g):<4d} {np.median(g.k95_z / g.k95_x):11.2f} "
              f"{g.rho_max.median():11.3f} {g.rho_median.median():11.3f} {g.retention_at_k.median():15.2f} {g.retention_at_k.min():6.2f} "
              f"{g.floor_a_min.min():11.1e} {g.floor_b_min.min():11.1e}")
    t = d[d.trained]
    complete = set(TRAINED) <= {m.replace("timm_", "") for m in t.model.unique()}
    if complete:
        plain = t[~t.downsample.astype(bool)]
        V["P9.60"] = float((plain.k95_y >= plain.k95_x).mean()) >= 0.9
        V["P9.61"] = float((t.rho_max >= 0.9).mean()) > 0.5 and float((t.retention_at_k >= 0.5).mean()) >= 0.9
        V["P9.62"] = bool(t.floor_a_min.min() >= -1e-6 and t.floor_b_min.min() >= -1e-6)
        print(f"  P9.60 (the sum is at least as wide as the stream, k*(0.95), in >= 90% of blocks without a downsample): "
              f"{(plain.k95_y >= plain.k95_x).sum()}/{len(plain)} -> {'holds' if V['P9.60'] else 'fails'}")
        print(f"  P9.61 (floor (b) is loose -- rho_max >= 0.9 in more than half of the blocks -- while the stream keeps at least half "
              f"of its eigenvalue at its own width in >= 90%): rho_max >= 0.9 in {(t.rho_max >= 0.9).sum()}/{len(t)}, "
              f"retention >= 0.5 in {(t.retention_at_k >= 0.5).sum()}/{len(t)} -> {'holds' if V['P9.61'] else 'fails'}")
        print(f"  P9.62 (both floors hold in every block; a violation is a measurement bug): -> {'holds' if V['P9.62'] else 'FAILS'}")
    else:
        print("  trained list incomplete -- not judged yet")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/residual")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    if a.judge:
        judge(a.out); return 0
    os.makedirs(a.out, exist_ok=True)
    for base, seed in [(b, None) for b in TRAINED] + RANDOM:
        name = f"timm_{base}" + ("" if seed is None else f"_random_s{seed}")
        path = os.path.join(a.out, f"{name}_blocks.csv")
        if os.path.exists(path):
            print(f"=== {name} done, skipping ===", flush=True); continue
        print(f"=== {name} start {time.strftime('%F %T')} ===", flush=True)
        t0 = time.time()
        rows, spectra = measure(base, seed, a)
        np.savez_compressed(os.path.join(a.out, f"{name}_spectra.npz"), **spectra)
        rows.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
        print(f"=== {name} end: {len(rows)} blocks, {int(rows.ok.sum())} pass the gate, {time.time() - t0:.0f} s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
