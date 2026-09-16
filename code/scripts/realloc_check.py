"""End-to-end check of layerspec.realloc on a real network: the A18 VGG-16 (BN) CIFAR-10 network.

  1. discover_sites -> 13 sites; measure on 6,400 training images (post-activation + patch moments)
  2. shrink every site to the A18 post-activation ruler widths through the package (Proposition A):
     must reproduce fold_cssp.py (93.82% at 67.3% of the parameters, seed 0)
  3. grow every site by 8 channels along the unmet directions (Proposition B): accuracy must be
     unchanged to the last image, and the report says how much input variance the new channels carry
  4. water_fill at the same parameter budget as the ruler widths (Proposition C): the allocation it
     chooses, and its accuracy after shrinking, beside the ruler's

    python scripts/realloc_check.py --device mps
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.realloc import discover_sites, greedy_cssp, grow, measure, shrink, water_fill  # noqa: E402
from scripts.cssp_check import load_model, loaders                                           # noqa: E402
from scripts.reproduce_garg import evaluate                                                    # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default="../results/round3_pt/a5_vgg16_bn_none_s0/vgg16_bn_cifar10_final.pt")
    p.add_argument("--widths-act", default="../results/a18/widths_act.json")
    p.add_argument("--data-root", default="../data")
    p.add_argument("--device", default="mps")
    p.add_argument("--n-calib", type=int, default=6400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grow", type=int, default=8)
    a = p.parse_args(argv)

    model = load_model(a.checkpoint, a.device)
    calib, test = loaders(a.data_root, a.n_calib, a.batch_size, 0, 0)
    sites = discover_sites(model)
    print(f"{len(sites)} sites: {[s.producer for s in sites]}", flush=True)
    t0 = time.time()
    mom = measure(model, sites, calib, a.device, patch=True, max_rows_per_batch=32768)
    print(f"measure: {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        acc_full = evaluate(model, test, a.device)
    print(f"full {acc_full:.2f}%  {n_params(model) / 1e6:.2f}M", flush=True)

    # ---- 2. shrink to the A18 post-activation ruler widths
    widths = json.load(open(a.widths_act))["widths"]["ruler"]
    m2 = copy.deepcopy(model)
    for s, w in zip(sites, widths):
        order, _ = greedy_cssp(mom[s.name].act_cov, w)
        shrink(m2, s, order[:w], mom[s.name])
    with torch.no_grad():
        acc = evaluate(m2, test, a.device)
    print(f"shrink to ruler widths: {acc:.2f}%  {n_params(m2) / 1e6:.2f}M ({n_params(m2) / n_params(model):.3f})  "
          f"[fold_cssp.py gave 93.82 at 0.673]", flush=True)

    # ---- 3. grow every site (deepest first so the measured patch covariances stay valid)
    m3 = copy.deepcopy(model)
    gains = {}
    for s in reversed(sites):
        lam = grow(m3, s, a.grow, mom[s.name])
        tr = float(torch.trace(mom[s.name].patch_cov))
        gains[s.producer] = float(lam.sum() / tr)
    with torch.no_grad():
        acc = evaluate(m3, test, a.device)
    print(f"grow +{a.grow} at every site: {acc:.2f}% (must equal {acc_full:.2f})  {n_params(m3) / 1e6:.2f}M", flush=True)
    print("  unmet variance captured by the new channels (fraction of the site's input patch variance):")
    for s in sites:
        print(f"    {s.producer:12s} {gains[s.producer]:.4f}")

    # ---- 4. water-filling at the ruler's parameter budget
    spectra = [np.clip(np.linalg.eigvalsh(mom[s.name].act_cov.numpy())[::-1], 0, None) for s in sites]
    cost = np.array([s.cost_per_channel(model) for s in sites], dtype=float)
    full_w = np.array([s.width(model) for s in sites])
    fixed = n_params(model) - float(cost @ full_w)               # parameters no site touches (stem input, classifier out)
    budget = float(cost @ np.array(widths))
    wf, mu, obj = water_fill(spectra, cost, budget, min_width=[8] * len(sites))
    m4 = copy.deepcopy(model)
    for s, w in zip(sites, wf):
        order, _ = greedy_cssp(mom[s.name].act_cov, int(w))
        shrink(m4, s, order[:int(w)], mom[s.name])
    with torch.no_grad():
        acc = evaluate(m4, test, a.device)
    print(f"water-fill at the ruler's budget: widths {wf.tolist()}\n  -> {acc:.2f}%  {n_params(m4) / 1e6:.2f}M "
          f"(ruler widths {widths})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
