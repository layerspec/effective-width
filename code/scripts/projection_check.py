"""Downstream anchor for the effective-width statistic (referee item, 2026-09-07).

Does k*(tau) mean anything for the network's function?  Calibrate on one half
of the 6,400-image subset: accumulate each conv layer's channel covariance
exactly as the measurement does (conv output, sampled positions), take its
eigenvectors and k*(tau).  Then on the other half, at inference and with no
retraining, replace every conv output y by  mean + P_k (y - mean)  where P_k
projects onto the top-k principal channel directions, k = k*(tau) for that
layer, all conv layers at once, and read off top-1 accuracy.  The control
projects onto k random orthonormal directions with the same k per layer.

    python scripts/projection_check.py --model resnet50 --data ../data/imagenet_val_6400 \
        --device mps --out ../results/projection

Not a pruning claim: the network is not rebuilt, only its activations are
projected.  It answers "is the variance beyond k* used?" and nothing else.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.data import FlatImageFolder, IMAGENET_MEAN, IMAGENET_STD      # noqa: E402
from layerspec.hooks import run_probe                                         # noqa: E402
from layerspec.models import build                                            # noqa: E402


def wnid_index() -> dict[str, int]:
    from timm.data import ImageNetInfo
    names = ImageNetInfo().label_names()
    assert len(names) == 1000 and names[0].startswith("n0"), names[:3]
    return {w: i for i, w in enumerate(names)}


class LabelledFolder(FlatImageFolder):
    """FlatImageFolder plus the label parsed from '..._nXXXXXXXX.JPEG'."""
    def __init__(self, root, transform, index):
        super().__init__(root, transform)
        self.labels = []
        for p in self.paths:
            m = re.search(r"_(n\d{8})\.", p.name)
            if not m or m.group(1) not in index:
                raise ValueError(f"no wnid in {p.name}")
            self.labels.append(index[m.group(1)])

    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        return x, self.labels[i]


def kstar(eig: np.ndarray, tau: float) -> int:
    eig = np.clip(eig, 0, None)
    c = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(c, tau) + 1)


@torch.no_grad()
def top1(model, loader, device) -> float:
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    return 100.0 * correct / n


class Projector:
    """Forward hooks that project each conv output onto k directions."""
    def __init__(self, model, layers: dict, device):
        self.handles = []
        for name, (mean, V, k) in layers.items():
            mod = model.get_submodule(name)
            mu = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
            P = torch.tensor(V[:, :k] @ V[:, :k].T, dtype=torch.float32, device=device)  # C x C
            def hook(_m, _i, out, mu=mu, P=P):
                d = out - mu
                d = torch.einsum("nchw,dc->ndhw", d, P)
                return mu + d
            self.handles.append(mod.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="resnet50")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--taus", type=float, nargs="+", default=[0.9, 0.95, 0.99, 0.999])
    p.add_argument("--calib", type=int, default=None, help="calibration images (default: half)")
    p.add_argument("--eval", type=int, default=None, help="evaluation images (default: the other half)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/projection")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    ds = LabelledFolder(a.data, tf, wnid_index())
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_cal = a.calib or len(ds) // 2
    n_eval = a.eval or (len(ds) - n_cal)
    cal = Subset(ds, perm[:n_cal]); ev = Subset(ds, perm[n_cal:n_cal + n_eval])
    mk = lambda d: DataLoader(d, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    cal_loader, ev_loader = mk(cal), mk(ev)

    model, tag = build(a.model, pretrained=True)
    model = model.to(a.device).eval()
    print(f"model {a.model} ({tag}); calibrate on {n_cal}, evaluate on {n_eval} images", flush=True)

    # ---- pass 1: covariances on the calibration half, as in the measurement
    t0 = time.time()
    probe = run_probe(model, cal_loader, device=a.device, positions_per_image=a.positions,
                      include_activations=False, progress=False, seed=a.seed)
    layers = {}
    for rec in probe.records():
        if rec.kind != "conv" or rec.acc is None or rec.acc.n < 2:
            continue
        cov = rec.acc.covariance
        cov = cov() if callable(cov) else cov
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        layers[rec.meta()["layer"]] = (rec.acc.mean, V, w, int(rec.C), float(rec.r_max))
    print(f"  {len(layers)} conv layers calibrated ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    base = top1(model, ev_loader, a.device)
    rows.append(dict(setting="baseline", tau=np.nan, top1=base, mean_k_over_C=1.0, mean_k_over_rmax=np.nan))
    print(f"  baseline top-1 {base:.2f}%", flush=True)
    rng = np.random.default_rng(a.seed)
    for tau in a.taus:
        for setting in ("principal", "random"):
            proj = {}
            kc, kr = [], []
            for name, (mean, V, w, C, rmax) in layers.items():
                k = kstar(w, tau)
                kc.append(k / C); kr.append(k / rmax)
                if setting == "principal":
                    proj[name] = (mean, V, k)
                else:
                    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
                    proj[name] = (mean, Q, k)
            P = Projector(model, proj, a.device)
            acc = top1(model, ev_loader, a.device)
            P.remove()
            rows.append(dict(setting=setting, tau=tau, top1=acc,
                             mean_k_over_C=float(np.mean(kc)), mean_k_over_rmax=float(np.mean(kr))))
            print(f"  tau {tau:<6} {setting:9s} top-1 {acc:6.2f}%   mean k/C {np.mean(kc):.3f}  "
                  f"mean k/r_max {np.mean(kr):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["model"], df["n_calib"], df["n_eval"] = a.model, n_cal, n_eval
    path = os.path.join(a.out, f"{a.model.replace(':', '_')}_projection.csv")
    df.to_csv(path, index=False)
    per = pd.DataFrame([dict(layer=n, C=C, r_max=r, **{f"k_{t}": kstar(w, t) for t in a.taus})
                        for n, (m, V, w, C, r) in layers.items()])
    per.to_csv(os.path.join(a.out, f"{a.model.replace(':', '_')}_projection_layers.csv"), index=False)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
