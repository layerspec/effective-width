"""A29 (analysis-plan 9.18): the layer profile of our own ImageNet-1k ResNet-18 weights (A23 full-width arm,
results/a18_imagenet_pt/resnet18_full_s<seed>/final.pt) and of a random initialisation, with the protocol of the
A17 runs: the 6,400-image validation subset, 16 positions per image, 224 pixels, layerspec's probe.

A23 did not save its initialisation.  The random network is rebuilt from torch.manual_seed(seed) and the same
constructor; it cannot be checked against the network A23 started from, so the judged statistic is every
trained x random pair and "same seed" is a label only (plan 9.18).

Rows go to <out>/resnet18_s<seed>/trajectory_layers.csv in the schema of the A17 runs (epoch 0 = random,
epoch 90 = trained), so checkpoint_analysis.py reads them with the same loader.

    python scripts/a29_measure_r18.py --weights ../results/a18_imagenet_pt --out ../results/a29
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.resnet_widths import RESNET18_WIDTHS, build_resnet18_widths   # noqa: E402

EPOCHS = 90


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--weights", default="../results/a18_imagenet_pt", help="folder with resnet18_full_s<seed>/final.pt")
    p.add_argument("--runs", default="../results/a18_imagenet", help="folder with resnet18_full_s<seed>/result.json (top-1)")
    p.add_argument("--measure-data", default="../data/imagenet_val_6400")
    p.add_argument("--out", default="../results/a29")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--device", default="cpu")
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--measure-size", type=int, default=224)
    p.add_argument("--measure-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args(argv)

    from layerspec.api import image_loader
    from scripts.trajectory_cifar import measure

    loader = image_loader(a.measure_data, batch_size=a.batch_size, workers=a.workers,
                          image_size=a.measure_size, limit=a.measure_limit)
    for seed in a.seeds:
        out = os.path.join(a.out, f"resnet18_s{seed}")
        if os.path.exists(os.path.join(out, "result.json")):
            print(f"=== {out} done, skipping ===", flush=True); continue
        os.makedirs(out, exist_ok=True)
        run = os.path.join(a.runs, f"resnet18_full_s{seed}", "result.json")
        top1 = json.load(open(run))["final_top1"] if os.path.exists(run) else float("nan")
        torch.manual_seed(seed)
        model = build_resnet18_widths(list(RESNET18_WIDTHS), 1000, "imagenet").to(a.device).eval()
        frames = []
        for epoch, acc, state in ((0, float("nan"), None), (EPOCHS, top1, os.path.join(a.weights, f"resnet18_full_s{seed}", "final.pt"))):
            if state is not None:
                model.load_state_dict(torch.load(state, map_location="cpu"))
            t0 = time.time()
            df = measure(model, loader, a.device, a.positions, epoch, acc, seed)
            df.to_csv(os.path.join(out, f"epoch_{epoch:03d}_layers.csv"), index=False)
            frames.append(df)
            conv = df[df.kind == "conv"]
            print(f"  seed {seed} epoch {epoch:3d}  top1 {acc:5.2f}  {len(conv)} conv layers  "
                  f"k*(.95)/C median {conv['k_star_ratio_0.95'].median():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(out, "trajectory_layers.csv"), index=False)
        with open(os.path.join(out, "result.json"), "w") as fh:
            json.dump({"arch": "resnet18", "seed": seed, "epochs": EPOCHS, "final_top1": top1,
                       "weights": os.path.join(a.weights, f"resnet18_full_s{seed}", "final.pt"),
                       "init": "rebuilt from torch.manual_seed(seed); not the saved initialisation of the A23 run",
                       "measure": {"data": a.measure_data, "size": a.measure_size, "positions": a.positions,
                                   "limit": a.measure_limit}}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
