"""Plan 9.40 (P9.101): overlap of successive branch top subspaces in a residual stage at gamma = 1 initialisation.

Re-runs the residual taps of measure_residual on a random timm ResNet-50 (seed 1, zero_init_last=False) and stores,
for every block of the requested stage, the leading eigenvectors of Sigma_x and Sigma_f; then judges P9.101.

Usage: cd code && python scripts/branch_overlap.py [--judge] [--stage layer3] [--out ../results/branch_overlap]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_residual import Tap, residual_blocks  # noqa: E402

TOP = 200


def kstar(e, tau):
    e = np.clip(np.sort(e)[::-1], 0, None); c = np.cumsum(e) / e.sum(); return int(np.searchsorted(c, tau) + 1)


def measure(a):
    import timm
    from layerspec.api import image_loader
    torch.manual_seed(a.seed)
    model = timm.create_model(a.model, pretrained=False, zero_init_last=False).to(a.device).eval()
    gen = torch.Generator().manual_seed(a.seed)
    taps = [Tap(n, bn, act, conv, a.positions, gen) for n, _, bn, act, conv in residual_blocks(model) if n.startswith(a.stage + ".")]
    loader = image_loader(a.data, batch_size=a.batch_size, workers=a.workers, image_size=224, limit=a.limit)
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x.to(a.device))
    out = {}
    for t in taps:
        J = t.cov(t.joint); C = J.shape[0] // 2
        sx, sf, sxf = J[:C, :C], J[C:, C:], J[:C, C:]
        wx_, Vx_ = np.linalg.eigh(sx); ix = np.argsort(wx_)[::-1][:TOP]; Ux = Vx_[:, ix]
        # cross term along the stream's top directions: diag of Ux^T (Sxf + Sfx) Ux, and the same for Sx and Sf
        out[f"{t.name}::cross_top"] = np.einsum("ij,ji->i", Ux.T @ (sxf + sxf.T), Ux)
        out[f"{t.name}::x_top"] = np.einsum("ij,ji->i", Ux.T @ sx, Ux)
        out[f"{t.name}::f_top"] = np.einsum("ij,ji->i", Ux.T @ sf, Ux)
        out[f"{t.name}::cross_trace"] = np.array([2 * np.trace(sxf)])
        for key, S in (("x", sx), ("f", sf)):
            w, V = np.linalg.eigh(S); idx = np.argsort(w)[::-1][:TOP]
            out[f"{t.name}::{key}_eig"] = w[idx]; out[f"{t.name}::{key}_vec"] = V[:, idx]; out[f"{t.name}::{key}_trace"] = np.array([np.trace(S)])
        out[f"{t.name}::Sx_full_eig"] = np.sort(np.linalg.eigvalsh(sx))[::-1]
        for h in t.handles:
            h.remove()
    return out


def judge(path, tau=0.95):
    z = np.load(path); names = sorted({k.split("::")[0] for k in z.keys()}, key=lambda n: int(n.split(".")[1]))
    kf = {n: kstar(z[f"{n}::f_eig"], tau) for n in names}; kx = {n: kstar(z[f"{n}::Sx_full_eig"], tau) for n in names}
    C = z[f"{names[0]}::x_vec"].shape[0]
    print(f"stage blocks {names}; C={C}; k95(f)={[kf[n] for n in names]}; k95(x)={[kx[n] for n in names]}")
    om = []
    for a_, b_ in zip(names[:-1], names[1:]):
        k = min(kf[a_], kf[b_]); Ua = z[f"{a_}::f_vec"][:, :k]; Ub = z[f"{b_}::f_vec"][:, :k]
        om.append(np.linalg.norm(Ua.T @ Ub, "fro") ** 2 / k)
    print("P9.101(a) overlap omega(b-1,b) =", np.round(om, 3), f"(random expectation ~ {np.mean([kf[n] for n in names])/C:.3f})",
          "->", "holds" if sum(o >= 0.5 for o in om) >= 4 else "fails")
    eta, ratio = [], []
    for bi in range(2, len(names)):
        b = names[bi]; prev = [names[bi - 1], names[bi - 2]]
        U = np.concatenate([z[f"{p}::f_vec"][:, :kf[p]] for p in prev], 1); Q, _ = np.linalg.qr(U); dU = Q.shape[1]
        # Sigma_x of block b reconstructed from its top-200 eigenpairs is partial; use energy inside U via the stored eigvecs/eigs (top-200 carry >= tau of the energy in practice)
        w, V = z[f"{b}::x_eig"], z[f"{b}::x_vec"]; tr = float(z[f"{b}::x_trace"][0])
        inside = float(np.sum(w * np.sum((Q.T @ V) ** 2, 0)))  # energy of the top-200 part inside U
        eta.append(inside / tr); ratio.append(dU / kx[b])
    if f"{names[0]}::cross_top" in z.files:
        for n in names:
            k = kx[n]; cx, xx, ff = z[f"{n}::cross_top"][:k].sum(), z[f"{n}::x_top"][:k].sum(), z[f"{n}::f_top"][:k].sum()
            print(f"  P9.102(beta) {n}: along top-{k} of x: x {xx:.1f}  f {ff:.1f}  cross {cx:+.1f}  cross/x {cx/xx:+.2f}  | trace cross/trace x {float(z[f'{n}::cross_trace'][0])/float(z[f'{n}::x_trace'][0]):+.3f}")
    print("P9.101(b) energy share eta inside union of the two previous branch top subspaces =", np.round(eta, 3), " dim(U)/k95(x) =", np.round(ratio, 2),
          "->", "holds" if sum(e >= 0.9 for e in eta) >= 3 and all(r <= 1.5 for r in ratio) else "fails")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50.tv_in1k"); p.add_argument("--seed", type=int, default=1); p.add_argument("--stage", default="layer3")
    p.add_argument("--data", default="../data/imagenet_val_6400"); p.add_argument("--out", default="../results/branch_overlap")
    p.add_argument("--judge", action="store_true"); p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=32); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.model.replace('.', '_')}_s{a.seed}_{a.stage}.npz")
    if not a.judge:
        t0 = time.time(); out = measure(a); np.savez_compressed(path, **out); print(f"saved {path} ({time.time() - t0:.0f} s)")
    judge(path)


if __name__ == "__main__":
    main()
