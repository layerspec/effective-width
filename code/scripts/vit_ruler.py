"""Read the ruler widths of a trained CIFAR ViT (trajectory_vit.py) at its MLP sites.

For each block's fc1 site the post-GELU covariance is accumulated on 6,400 unlabeled training images (all
tokens) with layerspec.realloc.measure, and the ruler width is k*(tau): the smallest number of eigenvalues
carrying tau of the trace.  The uniform arm gets the same total site width, equal per block (every site's
channel costs the same 2 x dim + 1 parameters, so equal width is equal cost).  Writes widths.json with
{"tau", "full", "ruler", "uniform", "params": {...}} for the queue to read.

    python scripts/vit_ruler.py --checkpoint ../results/vit_c10/full_s0/vit_c10_cifar10_final.pt --out ../results/vit_c10/widths.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trajectory_vit import build_vit   # noqa: E402


def k_star(eig: np.ndarray, tau: float) -> int:
    e = np.sort(np.clip(eig, 0, None))[::-1]
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, tau) + 1)


def ruler_widths(model, calib, device, taus) -> dict:
    from layerspec.realloc import discover_sites, measure
    sites = discover_sites(model)
    mom = measure(model, sites, calib, device, patch=False, max_rows_per_batch=None)
    out = {}
    for tau in taus:
        out[tau] = [k_star(np.linalg.eigvalsh(mom[s.name].act_cov.numpy()), tau) for s in sites]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="../data")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--device", default="cpu")
    p.add_argument("--calib", type=int, default=6400)
    p.add_argument("--tau", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    from scripts.cssp_check import loaders
    from scripts.reproduce_garg import CIFAR_STATS
    model = build_vit(CIFAR_STATS[a.dataset][2])
    sd = torch.load(a.checkpoint, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(a.device).eval()
    calib, _ = loaders(a.data_root, a.calib, 256, 0, a.seed, a.dataset)
    full = [blk.mlp.fc1.out_features for blk in model.blocks]
    taus = sorted({a.tau, 0.95, 0.99, 0.999})
    ks = ruler_widths(model, calib, a.device, taus)
    ruler = ks[a.tau]
    uni = int(round(sum(ruler) / len(ruler)))
    uniform = [uni] * len(ruler)
    dim = model.blocks[0].mlp.fc1.in_features
    per_ch = 2 * dim + 1
    tot = sum(q.numel() for q in model.parameters())
    def params(w):
        return tot - per_ch * sum(full) + per_ch * sum(w)
    res = {"tau": a.tau, "full": full, "ruler": ruler, "uniform": uniform,
           "k_star": {str(t): ks[t] for t in taus},
           "params": {"full": tot, "ruler": params(ruler), "uniform": params(uniform)},
           "fraction": {"ruler": params(ruler) / tot, "uniform": params(uniform) / tot}}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
