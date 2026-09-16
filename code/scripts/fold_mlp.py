"""P1 on transformers and ConvNeXt: zero-retraining width reduction of every MLP hidden layer
(fc1 -> GELU -> fc2) by column-subset selection on the GELU output, folded into fc2.

For each block, Sigma_act (hidden x hidden) is accumulated over all tokens/positions of the
calibration half of the ImageNet subset; the greedy (pivoted-Cholesky) subset S of size
m = k*(tau; Sigma_act) is kept; the dropped units are reconstructed linearly from S and the
reconstruction is folded into fc2:

    fc1' = fc1[S, :],   fc2'.W[:, S] = W2[:, S] + W2[:, drop] B,   fc2'.b = b2 + W2[:, drop] c

which is exact (no pooling between act and fc2).  fc2's output width is the residual stream and
is not touched.  Top-1 is measured on the other half with timm's own eval transform.

    python scripts/fold_mlp.py --model vit_base_patch16_224.augreg_in21k_ft_in1k --device mps
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.cssp_check import greedy_cssp, kstar            # noqa: E402
from scripts.projection_check import LabelledFolder, top1, wnid_index  # noqa: E402


class Stats:
    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def update(self, a: torch.Tensor):
        x = a.detach().reshape(-1, a.shape[-1]).to("cpu").double()
        self.n += x.shape[0]
        self.s = x.sum(0) if self.s is None else self.s + x.sum(0)
        self.ss = x.T @ x if self.ss is None else self.ss + x.T @ x

    def finish(self):
        mean = self.s / self.n
        cov = self.ss / self.n - torch.outer(mean, mean)
        return mean, 0.5 * (cov + cov.T)


def set_module(model: nn.Module, name: str, new: nn.Module):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def fold_block(model: nn.Module, prefix: str, mean, cov, S: list[int]):
    """Replace <prefix>.fc1 / <prefix>.fc2 by their folded narrow versions (in place)."""
    mods = dict(model.named_modules())
    fc1, fc2 = mods[prefix + ".fc1"], mods[prefix + ".fc2"]
    H = fc1.out_features
    keep = torch.tensor(sorted(S))
    drop = torch.tensor([j for j in range(H) if j not in set(S)])
    if len(drop) == 0:
        return
    Sss = cov[keep][:, keep]
    B = torch.linalg.pinv(Sss, rtol=1e-8) @ cov[keep][:, drop]          # |S| x |drop|
    c = mean[drop] - B.T @ mean[keep]                                   # |drop|
    W1, b1 = fc1.weight.detach().cpu().double(), fc1.bias.detach().cpu().double()
    W2, b2 = fc2.weight.detach().cpu().double(), fc2.bias.detach().cpu().double()
    n1 = nn.Linear(fc1.in_features, len(keep))
    n2 = nn.Linear(len(keep), fc2.out_features)
    with torch.no_grad():
        n1.weight.copy_(W1[keep].float()); n1.bias.copy_(b1[keep].float())
        n2.weight.copy_((W2[:, keep] + W2[:, drop] @ B.T).float())
        n2.bias.copy_((b2 + W2[:, drop] @ c).float())
    dev, dt = fc1.weight.device, fc1.weight.dtype
    set_module(model, prefix + ".fc1", n1.to(dev, dt))
    set_module(model, prefix + ".fc2", n2.to(dev, dt))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="vit_base_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--taus", type=float, nargs="+", default=[0.999, 0.99, 0.95])
    p.add_argument("--eps", type=float, nargs="*", default=[0.001, 0.01],
                   help="also choose m by the CSSP residual fraction tr(R_S)/tr(Sigma) <= eps (Proposition A's m_S)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/cssp_mlp")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(a.model, pretrained=True).to(a.device).eval()
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg, is_training=False)
    ds = LabelledFolder(a.data, tf, wnid_index())
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(a.seed)).tolist()
    half = len(ds) // 2
    mk = lambda d: DataLoader(d, batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    cal, ev = mk(Subset(ds, perm[:half])), mk(Subset(ds, perm[half:]))
    mods = dict(model.named_modules())
    prefixes = [n[:-4] for n in mods if n.endswith(".mlp.act") or n.endswith("mlp.act")]
    prefixes = [x for x in prefixes if (x + ".fc1") in mods and (x + ".fc2") in mods]
    print(f"{a.model}: {len(prefixes)} MLP blocks; calibrate {half}, evaluate {len(ds) - half}; cfg {cfg['input_size']} crop {cfg['crop_pct']}", flush=True)

    stats = {x: Stats() for x in prefixes}
    hs = [mods[x + ".act"].register_forward_hook(lambda m, i, o, x=x: stats[x].update(o)) for x in prefixes]
    t0 = time.time()
    with torch.no_grad():
        for xb, _ in cal:
            model(xb.to(a.device))
    for h in hs:
        h.remove()
    moments = {x: stats[x].finish() for x in prefixes}
    print(f"  calibration {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    spec = {x: np.clip(np.linalg.eigvalsh(moments[x][1].numpy())[::-1], 0, None) for x in prefixes}
    piv = {x: greedy_cssp(moments[x][1], moments[x][1].shape[0]) for x in prefixes}
    print(f"  spectra and pivots {time.time() - t0:.0f}s", flush=True)

    acc_full = top1(model, ev, a.device)
    n_full = sum(q.numel() for q in model.parameters())
    n_mlp = sum(q.numel() for n, q in model.named_parameters() if ".mlp." in n or "mlp." in n)
    print(f"  full top-1 {acc_full:.2f}%  params {n_full / 1e6:.2f}M (MLP {n_mlp / 1e6:.2f}M)", flush=True)

    rows, per_block = [], []
    schemes = [("tau", t) for t in a.taus] + [("eps", e) for e in a.eps]
    for kind, val in schemes:
        tau = val if kind == "tau" else float("nan")
        narrow = copy.deepcopy(model)
        widths = {}
        for x in prefixes:
            H = spec[x].shape[0]
            if kind == "tau":
                m = min(kstar(spec[x], val), H)
            else:
                tr = np.asarray(piv[x][1]) / piv[x][1][0]
                m = int(np.argmax(tr <= val)) if (tr <= val).any() else H
            widths[x] = m
            order, traces = piv[x]
            per_block.append({"model": a.model, "scheme": f"{kind}_{val}", "tau": tau, "block": x, "H": H, "m": m,
                              "subspace_resid": float(spec[x][m:].sum() / spec[x].sum()) if m < H else 0.0,
                              "cssp_resid": float(traces[m] / traces[0]) if m < len(traces) else 0.0})
            fold_block(narrow, x, *moments[x], sorted(order[:m]))
        t0 = time.time()
        acc = top1(narrow, ev, a.device)
        n_narrow = sum(q.numel() for q in narrow.parameters())
        rows.append({"model": a.model, "scheme": f"{kind}_{val}", "tau": tau, "acc_full": acc_full, "acc_fold": acc, "params_full": n_full,
                     "params_fold": n_narrow, "param_frac": n_narrow / n_full,
                     "mlp_frac": (n_narrow - (n_full - n_mlp)) / n_mlp,
                     "widths": " ".join(str(widths[x]) for x in prefixes)})
        print(f"  {kind} {val:g}: top-1 {acc:.2f} (full {acc_full:.2f})  params {n_narrow / n_full:.3f}  "
              f"MLP hidden kept {rows[-1]['mlp_frac']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
        del narrow
    stem = a.model.replace("/", "_")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{stem}_fold.csv"), index=False)
    pd.DataFrame(per_block).to_csv(os.path.join(a.out, f"{stem}_blocks.csv"), index=False)
    np.savez(os.path.join(a.out, f"{stem}_spectra.npz"), **{x.replace(".", "_"): spec[x] for x in prefixes})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
