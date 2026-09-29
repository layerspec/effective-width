"""A32 (analysis-plan 9.22): the alignment of a kernel with its input's second-order statistics against the
convolutional neural feature ansatz (Beaglehole et al. 2023), on the same layers of the same networks.

Treating the patches of a layer's input as independent inputs, the gradient of the network output G with respect to
a patch is (dG/dz) W, so the patch AGOP is W^T Gamma W with Gamma the Gram matrix of the gradients at the
convolution's OUTPUT (C_out x C_out), summed over images and positions.  With W = U S V^T,

    W^T W = V S^2 V^T          W^T Gamma W = V S (U^T Gamma U) S V^T

Per dense convolution the script accumulates
    Sigma_patch   centred covariance of the input patches, 48 positions per image   (layerspec PatchProbe)
    Gamma         gradients of v^T G at the convolution output, v a Rademacher vector per image, every position;
                  E[v v^T] = I makes this an unbiased estimate of the full-Jacobian sum over the outputs
and reports, for the retained singular directions, s_i^2, sigma_i^2 = v_i^T Sigma_patch v_i, gamma_i = u_i^T Gamma u_i
and the statistics S1-S4 of the plan.  The entry-wise correlations of the d x d matrices are computed from
C_out x C_out products; nothing of size d x d is formed beyond Sigma_patch itself.

    python scripts/measure_agop.py --out ../results/agop
    python scripts/measure_agop.py --judge --out ../results/agop
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = ["resnet18", "vgg16_bn", "resnet18_random", "vgg16_bn_random"]     # the cheaper network first
SEED_RANDOM, N_PERM, GATE = 1, 10, 50.0


def _ranks(x):
    return pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()


def spearman(a, b) -> float:
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(x, y, z) -> float:
    """Rank correlation of x and y with z held fixed."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt(max(1 - rxz ** 2, 0) * max(1 - ryz ** 2, 0))
    return float((rxy - rxz * ryz) / den) if den > 1e-12 else float("nan")


def gram_correlations(W: np.ndarray, Gamma: np.ndarray) -> tuple[float, float]:
    """Entry-wise Pearson correlation and cosine similarity of A = W^T W and B = W^T Gamma W (both d x d)."""
    d = W.shape[1]
    K = W @ W.T
    w1 = W.sum(1)                                           # W 1_d
    GK = Gamma @ K
    ab, aa, bb = np.sum(K * GK.T), np.sum(K * K), np.sum(GK * GK.T)   # tr(K K Gamma), tr(K K), tr(Gamma K Gamma K)
    sa, sb = float(w1 @ w1), float(w1 @ Gamma @ w1)         # sums of all entries
    n = d * d
    cov = ab - sa * sb / n
    va, vb = aa - sa ** 2 / n, bb - sb ** 2 / n
    pearson = cov / np.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
    cosine = ab / np.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
    return float(pearson), float(cosine)


def layer_stats(W: np.ndarray, Sigma: np.ndarray, Gamma: np.ndarray, rng, rank_tol: float = 1e-6) -> tuple[dict, dict]:
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = int((s > rank_tol * s[0]).sum())
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    gain = s ** 2
    VS = Vt @ Sigma                                          # BLAS; an einsum over (r, d, d) takes minutes at d = 4608
    sigma = np.sum(VS * Vt, axis=1)
    G_u = U.T @ Gamma @ U
    gamma = np.clip(np.diag(G_u), 0, None)
    pearson, cosine = gram_correlations(W, Gamma)
    null = []
    for _ in range(N_PERM):
        p = rng.permutation(Gamma.shape[0])
        null.append(gram_correlations(W, Gamma[np.ix_(p, p)])[0])
    proj = VS @ Vt.T                                         # Sigma_patch in the kernel's row space
    lam, Q = np.linalg.eigh(Sigma)
    order = np.argsort(lam)[::-1]; lam, Q = lam[order], Q[:, order]
    keep = int((lam > rank_tol * lam[0]).sum())
    gain_q = np.sum((W @ Q[:, :keep]) ** 2, axis=0)          # Maennel's f(sigma)^2, up to 1 / C_out
    row = {"r": r, "s1_pearson": pearson, "s1_cosine": cosine, "s1_null": float(np.mean(null)), "s1_null_sd": float(np.std(null)),
           "s2_rho_align": spearman(gain, sigma), "s3_rho_gain_gamma": spearman(gain, gamma),
           "s3_gamma_cv": float(np.std(gamma) / np.mean(gamma)) if np.mean(gamma) > 0 else float("nan"),
           "s3_offdiag": float(1 - np.sum(np.diag(G_u) ** 2) / np.sum(G_u ** 2)) if np.sum(G_u ** 2) > 0 else float("nan"),
           "s4_align_given_gamma": partial_spearman(gain, sigma, gamma),
           "s4_gamma_given_align": partial_spearman(gain, gamma, sigma),
           "rho_sigma_gamma": spearman(sigma, gamma),
           "cosine_sigma_rowspace": float(np.sum(gain * np.diag(proj)) / (np.linalg.norm(gain) * np.linalg.norm(proj))),
           "rho_align_sigma_basis": spearman(gain_q, lam[:keep])}
    return row, {"gain": gain, "sigma": sigma, "gamma": gamma, "lam": lam[:keep], "gain_q": gain_q}


class GradProbe:
    """Keeps the output of every dense convolution of one forward pass and, after backward, adds its gradient Gram."""

    def __init__(self, model: nn.Module):
        self.out, self.gamma, self.n_images, self.n_vectors = {}, {}, 0, {}
        self.handles = [m.register_forward_hook(lambda mod, inp, out, _n=name: self._keep(_n, out))
                        for name, m in model.named_modules() if isinstance(m, nn.Conv2d) and m.groups == 1]

    def _keep(self, name, out):
        if out.requires_grad:
            out.retain_grad()
            self.out[name] = out

    def collect(self, n_images: int) -> None:
        for name, out in self.out.items():
            g = out.grad
            if g is None or g.shape[-1] * g.shape[-2] == 1:
                continue
            g = g.permute(0, 2, 3, 1).reshape(-1, g.shape[1]).double()
            G = (g.T @ g).cpu().numpy()
            self.gamma[name] = self.gamma.get(name, 0) + G
            self.n_vectors[name] = self.n_vectors.get(name, 0) + g.shape[0]
        self.n_images += n_images
        self.out = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def measure(name: str, a) -> tuple[pd.DataFrame, dict]:
    from layerspec import models as _models
    from layerspec.api import image_loader
    from layerspec.decomposition import PatchProbe
    model, tag = _models.build(name, seed=SEED_RANDOM)
    model = model.to(a.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    patches, grads = PatchProbe(model, a.positions, seed=0), GradProbe(model)
    gen = torch.Generator().manual_seed(0)
    loader = image_loader(a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit)
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(a.device).requires_grad_(True)
        logits = model(x)
        v = (torch.randint(0, 2, logits.shape, generator=gen).to(logits) * 2 - 1)
        (logits * v).sum().backward()
        grads.collect(x.shape[0])
    patches.remove(); grads.remove()
    rng = np.random.default_rng(0)
    rows, arrays = [], {}
    for lname, meta in patches.meta.items():
        acc = patches.acc.get(lname)
        if acc is None or lname not in grads.gamma:
            continue
        m = meta["module"]
        W = m.weight.detach().cpu().double().numpy().reshape(m.out_channels, -1)
        Sigma = acc.M2 / (acc.n - 1)
        Gamma = grads.gamma[lname] / grads.n_images
        row, arr = layer_stats(W, Sigma, Gamma, rng)
        row.update({"layer": lname, "depth_index": meta["depth_index"], "C_out": m.out_channels, "d": W.shape[1],
                    "n_patches": acc.n, "n_over_d": acc.n / W.shape[1], "ok": acc.n / W.shape[1] >= GATE,
                    "n_grad_vectors": grads.n_vectors[lname], "weights": tag})
        rows.append(row)
        arrays.update({f"{lname}::{k}": v for k, v in arr.items()})
    return pd.DataFrame(rows).sort_values("depth_index"), arrays


def judge(out: str) -> dict:
    """P9.63-P9.66 (plan 9.22) from the medians over the gate-passing layers of the two trained networks."""
    print("==== A32: alignment with the input covariance and the neural feature ansatz (plan 9.22) ====")
    cols = ["s1_pearson", "s1_null", "s2_rho_align", "s3_rho_gain_gamma", "s3_gamma_cv", "s4_align_given_gamma", "s4_gamma_given_align"]
    med = {}
    print(f"  {'network':18s} {'layers':>6s} " + " ".join(f"{c.replace('s4_', '').replace('s3_', '').replace('s2_', '').replace('s1_', 'S1 '):>18s}" for c in cols))
    for f in sorted(glob.glob(os.path.join(out, "*_agop.csv"))):
        d = pd.read_csv(f); d = d[d.ok.astype(bool)]
        name = os.path.basename(f)[:-9]
        med[name] = {c: float(d[c].median()) for c in cols} | {"n": len(d), "excess": float((d.s1_pearson - d.s1_null).median())}
        print(f"  {name:18s} {len(d):6d} " + " ".join(f"{med[name][c]:+18.2f}" for c in cols))
    V = {}
    if "vgg16_bn" in med and "resnet18" in med:
        v, r = med["vgg16_bn"], med["resnet18"]
        V["P9.63"] = v["s1_pearson"] >= 0.8 and r["s1_pearson"] >= 0.6
        V["P9.64"] = all(m["s1_null"] >= 0.6 and m["excess"] <= 0.3 for m in (v, r))
        V["P9.65"] = all(m["s3_rho_gain_gamma"] <= -0.3 for m in (v, r))
        V["P9.66"] = all(m["s4_align_given_gamma"] >= 0.7 for m in (v, r))
        say = lambda k: "holds" if V[k] else "fails"                                  # noqa: E731
        print(f"  P9.63 (the ansatz is reproduced: S1 median >= 0.8 for vgg16_bn, >= 0.6 for resnet18) -> {say('P9.63')}")
        print(f"  P9.64 (structural floor: S1 with Gamma's channels permuted >= 0.6 and excess <= 0.3; "
              f"excess {v['excess']:+.2f} / {r['excess']:+.2f}) -> {say('P9.64')}")
        print(f"  P9.65 (gradient energy is lower along high-gain directions: Spearman(s^2, gamma) <= -0.3; descriptive) -> {say('P9.65')}")
        print(f"  P9.66 (the alignment survives holding gamma fixed: partial Spearman >= +0.7) -> {say('P9.66')}")
        low = min(v["s4_align_given_gamma"], r["s4_align_given_gamma"])
        if V["P9.66"]:
            print("  Registered consequence: the alignment is not accounted for by the ansatz; two regularities of the same matrix.")
        elif low < 0.3:
            print("  Registered consequence: the alignment is a shadow of the ansatz; rho_align is a gradient-free proxy for it.")
        else:
            print("  Registered consequence: partly accounted for; both numbers are reported.")
    else:
        print("  both trained networks are needed -- not judged yet")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/agop")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    if a.judge:
        judge(a.out); return 0
    os.makedirs(a.out, exist_ok=True)
    for name in a.models:
        path = os.path.join(a.out, f"{name}_agop.csv")
        if os.path.exists(path):
            print(f"=== {name} done, skipping ===", flush=True); continue
        print(f"=== {name} start {time.strftime('%F %T')} ===", flush=True)
        t0 = time.time()
        rows, arrays = measure(name, a)
        np.savez_compressed(os.path.join(a.out, f"{name}_directions.npz"), **arrays)
        rows.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)
        print(f"=== {name} end: {len(rows)} layers, {int(rows.ok.sum())} pass the gate, {time.time() - t0:.0f} s ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
