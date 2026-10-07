"""Plan 9.43 (P9.103-P9.104): the known Hermite (Mehler) closed form of the ReLU step, tested on the residual stream.

For every residual block of one stage it stores the covariance and mean of y = x + f (before the block's ReLU) and the
covariance of z = ReLU(y), then predicts Sigma_z from (mean_y, Sigma_y) under the assumption that y is jointly Gaussian
over the samples (images x positions):  Sigma_hat_z[i,j] = sum_{k>=1} mu_k(i) mu_k(j) rho_ij^k, with
mu_k(i) = E[ReLU(m_i + s_i X) he_k(X)/sqrt(k!)], X ~ N(0,1) (Gauss-Hermite quadrature), and compares k*(tau).

Usage: cd code && python scripts/relu_step_hermite.py --model resnet50.tv_in1k [--pretrained] [--seed 1] [--stage layer3]
       python scripts/relu_step_hermite.py --judge
"""
import argparse, glob, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_residual import Tap, residual_blocks  # noqa: E402

KMAX = 24


def kstar(e, tau):
    e = np.clip(np.sort(e)[::-1], 0, None); c = np.cumsum(e) / e.sum(); return int(np.searchsorted(c, tau) + 1)


def hermite_coeffs(means, stds, kmax=KMAX, npts=120):
    """mu_k(i) for phi(x) = ReLU(m_i + s_i x), normalised probabilists' Hermite basis, k = 0..kmax."""
    from scipy.special import roots_hermitenorm
    x, w = roots_hermitenorm(npts)                     # weight exp(-x^2/2); divide by sqrt(2 pi)
    w = w / np.sqrt(2 * np.pi)
    He = np.zeros((kmax + 1, npts)); He[0] = 1.0; He[1] = x
    for k in range(2, kmax + 1):
        He[k] = x * He[k - 1] - (k - 1) * He[k - 2]
    import math
    fact = np.array([math.factorial(k) for k in range(kmax + 1)], float)
    hn = He / np.sqrt(fact)[:, None]
    phi = np.maximum(means[:, None] + stds[:, None] * x[None, :], 0.0)      # (C, npts)
    return (phi * w[None, :]) @ hn.T                                        # (C, kmax+1)


def predict(mean_y, cov_y, kmax=KMAX):
    s = np.sqrt(np.clip(np.diag(cov_y), 1e-12, None)); R = cov_y / np.outer(s, s); R = np.clip(R, -1, 1)
    mu = hermite_coeffs(mean_y, s, kmax)
    out = np.zeros_like(cov_y); Rk = np.ones_like(R)
    for k in range(1, kmax + 1):
        Rk = Rk * R
        out += np.outer(mu[:, k], mu[:, k]) * Rk
    return out


def measure(a):
    import timm
    from layerspec.api import image_loader
    torch.manual_seed(a.seed)
    model = timm.create_model(a.model, pretrained=a.pretrained, zero_init_last=False) if not a.pretrained else timm.create_model(a.model, pretrained=True)
    model = model.to(a.device).eval()
    gen = torch.Generator().manual_seed(a.seed)
    taps = [Tap(n, bn, act, conv, a.positions, gen) for n, _, bn, act, conv in residual_blocks(model) if n.startswith(a.stage + ".")]
    # y is x + f: the joint accumulator holds (x, f); Sigma_y = Sx + Sf + Sxf + Sfx, mean_y = mean_x + mean_f
    loader = image_loader(a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit)
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(a.device))
    out = {}
    for t in taps:
        J = t.cov(t.joint); C = J.shape[0] // 2
        sy = J[:C, :C] + J[C:, C:] + J[:C, C:] + J[C:, :C]
        my = t.joint.mean[:C] + t.joint.mean[C:]
        out[f"{t.name}::cov_y"] = sy; out[f"{t.name}::mean_y"] = my; out[f"{t.name}::cov_z"] = t.cov(t.z); out[f"{t.name}::mean_z"] = t.z.mean
        for h in t.handles:
            h.remove()
    return out


def judge(out_dir, tau=0.95):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        z = np.load(path); names = sorted({k.split("::")[0] for k in z.files}, key=lambda n: int(n.split(".")[1]))
        tag = os.path.basename(path).replace(".npz", "")
        for n in names:
            cy, my, cz = z[f"{n}::cov_y"], z[f"{n}::mean_y"], z[f"{n}::cov_z"]
            ky, kz = kstar(np.linalg.eigvalsh(cy), tau), kstar(np.linalg.eigvalsh(cz), tau)
            khat = kstar(np.linalg.eigvalsh(predict(my, cy)), tau)
            # Gaussianity of y: excess kurtosis cannot be computed from second moments; report the mean/std ratio instead
            snr = float(np.median(np.abs(my) / np.sqrt(np.clip(np.diag(cy), 1e-12, None))))
            rows.append((tag, n, ky, kz, khat, (khat - kz) / kz, (kz - ky) / ky, (khat - ky) / ky, snr))
            print(f"  {tag:34s} {n}: k(y) {ky:4d}  k(z) {kz:4d}  Hermite k(z_hat) {khat:4d}  err {(khat - kz) / kz:+.2f} | measured ReLU step {(kz - ky) / ky:+.2f}  predicted {(khat - ky) / ky:+.2f} | median |mean|/std of y {snr:.2f}")
    init = [r for r in rows if "random" in r[0]]; tr = [r for r in rows if "random" not in r[0]]
    if init:
        ok = sum(abs(r[5]) <= 0.10 for r in init); r50 = [r for r in init if "resnet50" in r[0]]; r34 = [r for r in init if "resnet34" in r[0]]
        sign50 = sum(abs(r[7]) <= 0.05 for r in r50); sign34 = sum(r[7] <= -0.05 for r in r34)
        print(f"  P9.103: |err| <= 0.10 in {ok}/{len(init)}; ResNet-50 predicted |step| <= 0.05 in {sign50}/{len(r50)}; ResNet-34 predicted step <= -0.05 in {sign34}/{len(r34)} -> "
              f"{'holds' if ok >= 10 and sign50 >= 5 and sign34 >= 5 else 'fails'}")
    if tr:
        bad = sum(abs(r[5]) > 0.20 for r in tr)
        print(f"  P9.104 (trained: Gaussian form misses by > 0.20): {bad}/{len(tr)} -> {'holds' if bad >= 4 else 'fails'}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.tv_in1k"); p.add_argument("--pretrained", action="store_true"); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stage", default="layer3"); p.add_argument("--data", default="../data/imagenet_val_6400"); p.add_argument("--out", default="../results/relu_step")
    p.add_argument("--judge", action="store_true"); p.add_argument("--device", default="cpu"); p.add_argument("--positions", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv); os.makedirs(a.out, exist_ok=True)
    if a.judge:
        judge(a.out); return 0
    tag = a.model.replace(".", "_") + ("_pretrained" if a.pretrained else f"_random_s{a.seed}") + f"_{a.stage}"
    path = os.path.join(a.out, tag + ".npz")
    if os.path.exists(path):
        print(f"{tag} done, skipping"); return 0
    t0 = time.time(); out = measure(a); np.savez_compressed(path, **out); print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
