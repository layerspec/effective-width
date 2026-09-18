#!/usr/bin/env python
"""A-priori profile feasibility, part A (notes/apriori-plan-2026-09-16.md):
does the untrained profile depend on the images only through their
second-order statistics?

Gaussian surrogate test.  Estimate the stationary cross-spectral density of
the 6,400 ImageNet-val images (after the pipeline's own preprocessing: resize,
centre crop, ImageNet normalisation) -- one 3x3 Hermitian matrix per spatial
frequency -- and draw Gaussian images with exactly that density.  Measure the
random-init networks (the SAME weights as results/local6400, i.e. the same
seed) on the surrogate images, and compare the k*(0.95)/r_max profile with the
one measured on the real images.  The first convolution is linear, so its
output covariance is identical by construction; everything after the first
ReLU can differ, and the size of that difference is what we want.

    python scripts/apriori_check.py estimate                       # -> results/apriori/rgb_csd.pt
    python scripts/apriori_check.py measure --inputs matched --seed 1 timm:vgg16_bn.tv_in1k_random ...
    python scripts/apriori_check.py compare                        # table + results/apriori/compare.csv

`--inputs` is one of matched (Gaussian, matched cross-spectral density), param (Gaussian,
isotropic power law times one 3x3 channel matrix fitted to it: `fit`),
pink (the pipeline's 1/f smoke-test noise), white (i.i.d. Gaussian).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from layerspec import api  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "data", "imagenet_val_6400")
OUT = os.path.join(ROOT, "results", "apriori")
CSD_PATH = os.path.join(OUT, "rgb_csd.pt")
CSD_PARAM_PATH = os.path.join(OUT, "rgb_csd_param.pt")

FAMILIES = {
    # family -> (random model name, directories holding the real-image random profiles)
    "resnet50": ("resnet50_random", [os.path.join(ROOT, "results", "local6400", f"seed{s}") for s in range(1, 6)]),
    "resnet18": ("timm:resnet18.tv_in1k_random", None),
    "resnet34": ("timm:resnet34.tv_in1k_random", None),
    "vgg16": ("timm:vgg16_bn.tv_in1k_random", None),
    "densenet121": ("timm:densenet121.tv_in1k_random", None),
    "resnet101": ("timm:resnet101.tv_in1k_random", None),
    "wide_resnet50_2": ("timm:wide_resnet50_2.tv_in1k_random", None),
    "resnext50": ("timm:resnext50_32x4d.tv_in1k_random", None),
    "mobilenetv2": ("timm:mobilenetv2_100.ra_in1k_random", None),
    "mobilenetv3": ("timm:mobilenetv3_large_100.ra_in1k_random", None),
    "efficientnet_b0": ("timm:efficientnet_b0.ra_in1k_random", None),
}
ARCH_DIRS = [os.path.join(ROOT, "results", "local6400", "archs", f"seed{s}") for s in range(1, 4)]


def stem(name: str) -> str:
    return name.replace(":", "_")


# ----------------------------------------------------------------- estimate
def estimate(args) -> None:
    """Stationary (circular) cross-spectral density of the preprocessed images."""
    loader = api.image_loader(DATA, batch_size=32, workers=0, image_size=224)
    n = 0
    mean = torch.zeros(3, dtype=torch.float64)
    S = None
    t0 = time.time()
    for x, _ in loader:
        x = x.to(torch.float64)
        mean += x.mean(dim=(0, 2, 3)) * x.shape[0]
        n += x.shape[0]
    mean /= n
    for x, _ in loader:
        x = x.to(torch.float64) - mean[None, :, None, None]
        F = torch.fft.rfft2(x)                       # (B, 3, H, W//2+1)
        acc = torch.einsum("bcij,bdij->cdij", F, F.conj())
        S = acc if S is None else S + acc
    S /= n
    os.makedirs(OUT, exist_ok=True)
    torch.save({"csd": S, "mean": mean, "n": n, "size": 224}, CSD_PATH)
    print(f"estimated csd from {n} images in {time.time() - t0:.0f}s -> {CSD_PATH}")
    print(f"channel means {mean.tolist()}")


def fit_param(args) -> None:
    """Parametric cross-spectral density: isotropic power law times one 3x3 channel matrix,
    S_param(w) = A |w|^-alpha * Sigma_ch (8 numbers), fitted to the measured density."""
    d = torch.load(CSD_PATH)
    S, n = d["csd"], d["size"]
    fy = torch.fft.fftfreq(n).unsqueeze(1)
    fx = torch.fft.rfftfreq(n).unsqueeze(0)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    T = torch.einsum("ccij->ij", S).real
    m = (r > 0) & (r < 0.45)
    bins = np.logspace(np.log10(1 / n), np.log10(0.45), 30)
    rb = np.digitize(r[m].numpy(), bins)
    lr, lt = [], []
    for b in range(1, len(bins)):
        sel = rb == b
        if sel.sum() > 0:
            lr.append(np.log(r[m].numpy()[sel]).mean())
            lt.append(np.log(T[m].numpy()[sel]).mean())
    slope, _ = np.polyfit(lr, lt, 1)
    alpha = -slope
    Sch = S.sum(dim=(2, 3)).real
    Sch = Sch / Sch.trace()
    g = r.clamp_min(1 / n) ** (-alpha)
    g = g * (T[m].sum() / g[m].sum())            # match the total power off DC
    g[0, 0] = T[0, 0]                            # keep the measured DC (image-mean) power
    Sp = g[None, None] * Sch[:, :, None, None]
    rel = float(((S - Sp)[:, :, m]).abs().pow(2).sum().sqrt() / S[:, :, m].abs().pow(2).sum().sqrt())
    torch.save({"csd": Sp.to(S.dtype), "mean": d["mean"], "n": d["n"], "size": n,
                "alpha": float(alpha), "channel": Sch}, CSD_PARAM_PATH)
    print(f"alpha (power) {alpha:.2f}; channel matrix {np.round(Sch.numpy(), 3).tolist()}; "
          f"rel. Frobenius error of the parametric density {rel:.3f} -> {CSD_PARAM_PATH}")


class MatchedGaussian(Dataset):
    """Gaussian images whose circular cross-spectral density equals the estimate."""

    def __init__(self, n: int, seed: int, csd_path: str = CSD_PATH):
        d = torch.load(csd_path)
        S, self.mean, size = d["csd"], d["mean"].to(torch.float32), d["size"]
        self.n, self.seed, self.size = n, seed, size
        # rfft2 of unit white noise has E|W|^2 = H*W per bin, so filter by sqrt(S / (H*W)).
        M = (S / (size * size)).permute(2, 3, 0, 1)          # (H, W', 3, 3) Hermitian PSD
        lam, V = torch.linalg.eigh(M)
        lam = lam.clamp_min(0).sqrt()
        L = V @ torch.diag_embed(lam).to(V.dtype) @ V.conj().transpose(-1, -2)
        self.L = L.to(torch.complex64)                          # (H, W', 3, 3)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        s = self.size
        w = torch.randn(3, s, s, generator=g)
        W = torch.fft.rfft2(w)
        Y = torch.einsum("ijcd,dij->cij", self.L, W)
        y = torch.fft.irfft2(Y, s=(s, s)) + self.mean[:, None, None]
        return y, 0


def make_loader(inputs: str, seed: int, n: int = 6400, batch_size: int = 32):
    if inputs in ("matched", "param"):
        ds = MatchedGaussian(n, seed, CSD_PATH if inputs == "matched" else CSD_PARAM_PATH)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if inputs == "pink":
        return api.image_loader("synthetic", batch_size=batch_size, workers=0, limit=n, seed=seed)
    if inputs == "white":
        return api.image_loader("gaussian", batch_size=batch_size, workers=0, limit=n, seed=seed)
    if inputs == "real":
        return api.image_loader(DATA, batch_size=batch_size, workers=0, image_size=224)
    raise ValueError(inputs)


def check_surrogate(args) -> None:
    """Sanity: the 27x27 (3 channels x 3x3 patch) covariance of the surrogate vs the real images."""
    def patch_cov(loader, max_batches=50):
        acc, n = torch.zeros(27, 27, dtype=torch.float64), 0
        for b, (x, _) in enumerate(loader):
            if b >= max_batches:
                break
            p = torch.nn.functional.unfold(x, 3, stride=8).transpose(1, 2).reshape(-1, 27).to(torch.float64)
            acc += p.T @ p
            n += p.shape[0]
        C = acc / n
        return C
    Cr = patch_cov(make_loader("real", 0))
    Cm = patch_cov(make_loader("matched", 1))
    er = torch.linalg.eigvalsh(Cr).flip(0)
    em = torch.linalg.eigvalsh(Cm).flip(0)
    rel = (Cr - Cm).norm() / Cr.norm()
    print(f"27x27 patch covariance: rel. Frobenius diff {rel:.3f}")
    print("top eigenvalues real   :", np.round(er[:8].numpy(), 3))
    print("top eigenvalues matched:", np.round(em[:8].numpy(), 3))
    for tau in (0.9, 0.95, 0.99):
        kr = int((er.cumsum(0) / er.sum() < tau).sum()) + 1
        km = int((em.cumsum(0) / em.sum() < tau).sum()) + 1
        print(f"k*({tau}) real {kr}  matched {km}")


# ------------------------------------------------------------------ measure
def measure(args) -> None:
    out = os.path.join(OUT, args.inputs, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    manifest = []
    for name in args.models:
        path = os.path.join(out, f"{stem(name)}_layers.csv")
        if os.path.exists(path) and not args.force:
            print(f"skip {name} (exists)")
            continue
        t0 = time.time()
        model = api.load_model(name, seed=args.seed)
        loader = make_loader(args.inputs, args.seed)
        prof = api.profile(model, loader, device=args.device, positions=16, seed=args.seed)
        prof.table["model"] = name
        prof.table["weights"] = f"random_init(seed={args.seed})"
        prof.table["source"] = f"{args.inputs}[6400]"
        prof.to_csv(path)
        np.savez_compressed(os.path.join(out, f"{stem(name)}_spectra.npz"), **prof.spectra)
        dense = prof.dense(0.95)
        meta = {"model": name, "inputs": args.inputs, "seed": args.seed, "n_dense": len(dense),
                "median_k95_rmax": float(dense.iloc[:, -1].median()) if len(dense) else None,
                "elapsed_sec": round(time.time() - t0, 1)}
        manifest.append(meta)
        print(json.dumps(meta))
        del model
    with open(os.path.join(out, "manifest.json"), "a") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")


# ------------------------------------------------------------------ compare
COL = "k_star_rmax_0.95"


def dense_rows(path: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    ok = t["n_over_C_ok"].astype(bool) if "n_over_C_ok" in t else t["ok"].astype(bool)
    keep = (t.kind == "conv") & ok & ~t["is_depthwise"].astype(bool)
    t = t.loc[keep, ["layer", "call_index", "depth_index", COL]].copy()
    t["key"] = t.layer + "#" + t.call_index.astype(str)
    return t.set_index("key").sort_values("depth_index")


def real_profile(family: str, seed: int) -> pd.DataFrame | None:
    name, dirs = FAMILIES[family]
    dirs = dirs or ARCH_DIRS
    for d in dirs:
        if d.endswith(f"seed{seed}"):
            p = os.path.join(d, f"{stem(name)}_layers.csv")
            return dense_rows(p) if os.path.exists(p) else None
    return None


def surrogate_profile(family: str, inputs: str, seed: int) -> pd.DataFrame | None:
    name, _ = FAMILIES[family]
    p = os.path.join(OUT, inputs, f"seed{seed}", f"{stem(name)}_layers.csv")
    return dense_rows(p) if os.path.exists(p) else None


def rho(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, int, float, float]:
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    x, y = j[COL + "_a"], j[COL + "_b"]
    return float(x.corr(y, method="spearman")), len(j), float((x - y).abs().median()), float(y.median())


def compare(args) -> None:
    rows = []
    print(f"{'family':16s} {'inputs':8s} {'L':>4s}  {'rho same-W':>10s} {'rho other-W':>11s} {'R-R ceiling':>11s}  "
          f"{'level real':>10s} {'level surr':>10s} {'med|diff|':>9s}  {'layer-1 real/surr':>18s}")
    for fam in FAMILIES:
        reals = {s: real_profile(fam, s) for s in range(1, 6)}
        reals = {s: r for s, r in reals.items() if r is not None}
        if len(reals) < 2:
            continue
        # R-R ceiling: median Spearman over pairs of real-image random draws
        rr = [rho(reals[a], reals[b])[0] for a in reals for b in reals if a < b]
        ceiling = float(np.median(rr))
        for inputs in args.inputs:
            for s in sorted(reals):
                sur = surrogate_profile(fam, inputs, s)
                if sur is None:
                    continue
                r_same, L, mad, lvl_s = rho(reals[s], sur)
                others = [rho(reals[o], sur)[0] for o in reals if o != s]
                r_other = float(np.median(others)) if others else float("nan")
                lvl_r = float(reals[s][COL].median())
                l1 = f"{reals[s][COL].iloc[0]:.3f}/{sur[COL].iloc[0]:.3f}"
                rows.append(dict(family=fam, inputs=inputs, seed=s, L=L, rho_same_w=r_same, rho_other_w=r_other,
                                 rr_ceiling=ceiling, level_real=lvl_r, level_surrogate=lvl_s, med_abs_diff=mad))
                print(f"{fam:16s} {inputs:8s} {L:4d}  {r_same:+10.2f} {r_other:+11.2f} {ceiling:+11.2f}  "
                      f"{lvl_r:10.3f} {lvl_s:10.3f} {mad:9.3f}  {l1:>18s}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, "compare.csv"), index=False)
        for inputs in args.inputs:
            d = df[df.inputs == inputs]
            if d.empty:
                continue
            frac = (d.rho_same_w / d.rr_ceiling)
            print(f"\n[{inputs}] families {d.family.nunique()}: rho same-W median {d.rho_same_w.median():+.2f}, "
                  f"rho/ceiling median {frac.median():.2f}, >=0.8*ceiling in {(frac >= 0.8).sum()}/{len(d)} runs; "
                  f"level surrogate-real median {(d.level_surrogate - d.level_real).median():+.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    sub.add_parser("check")
    sub.add_parser("fit")
    m = sub.add_parser("measure")
    m.add_argument("models", nargs="+")
    m.add_argument("--inputs", default="matched", choices=["matched", "param", "pink", "white"])
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--device", default="mps")
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--inputs", nargs="+", default=["matched", "param", "pink", "white"])
    args = ap.parse_args(argv)
    {"estimate": estimate, "check": check_surrogate, "fit": fit_param, "measure": measure, "compare": compare}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
