"""A31 (analysis-plan 9.26): the class count alone, on ImageNet images at 224 pixels.  ResNet-50 under the A17
recipe, 1,000 training images per class, 37,000 optimisation steps in every cell (the epoch count follows):

    k10   ten classes        925 epochs
    k20   twenty            468
    k50   fifty             189
    k100  a hundred          95

The training loop is scripts/train_imagenet_a17.py itself, untouched (as in train_imagenet_a28.py): this wrapper
replaces the data (the cell's classes and images, scripts/a31_lists.py) and the measurement sets.

Measurement (k*(0.95)/r_max along the trajectory, the A17 epochs stretched with the run):
    primary     the 6,400-image validation subset of section 6 -- the same images for every cell, for A17 and
                for the image-count cell of A28, so every T-R / T-T is on one image set
    secondary   the cell's own held-out images (256 per class, round-robin; at most 6,400), at the first and
                last epoch (secondary_epoch_XXX_layers.csv); layers failing n/C >= 50 are not reported

    python scripts/train_imagenet_a31.py --cell k10 --data /workspace/imagenet --lists ../results/a31/lists.json \\
        --standard /workspace/imagenet_val_6400 --seed 0 --out ../results/a31/k10_s0

Resumable, as the A17 runs are.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import a31_lists                                        # noqa: E402
from scripts.imagenet_blob import imagenet_split                     # noqa: E402
from scripts.train_imagenet_a28 import schedule                      # noqa: E402
from scripts.train_imagenet_r18 import MEAN, STD, _val_tf            # noqa: E402

STEPS, BATCH = 37000, 256


def epochs_for(k: int, per_class: int = a31_lists.PER_CLASS, batch: int = BATCH, steps: int = STEPS) -> int:
    """Epochs that give `steps` optimiser steps at batch 256 with drop_last (k * per_class images per epoch)."""
    return round(steps / math.ceil(k * per_class / batch))


CELLS = {c: dict(K=k, epochs=epochs_for(k)) for c, k in a31_lists.CELL_K.items()}   # 925 / 468 / 189 / 95


def cell_loaders(k: int, lists: dict, root: str, batch_size: int, workers: int, seed: int, crop: int):
    """The (train, val, n_classes) triple of train_imagenet_r18.loaders for one cell; same transforms, same options."""
    from torchvision import transforms
    from scripts.class_subset import RemapSubset, label_map, subset_indices
    train_tf = transforms.Compose([transforms.RandomResizedCrop(crop), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = imagenet_split(root, "train", train_tf)
    va = imagenet_split(root, "val", _val_tf(crop))
    if tr.classes != va.classes:
        raise SystemExit("train/ and val/ class sets differ")
    a31_lists.check(lists, tr.targets)
    classes = a31_lists.cell_classes(lists, k)
    lmap = label_map(classes)
    tr = RemapSubset(tr, a31_lists.train_indices(lists, k), lmap)
    va = RemapSubset(va, subset_indices(va.targets, classes, None), lmap)
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
              prefetch_factor=4 if workers > 0 else None)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **kw),
            torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, **kw), len(classes))


def holdout_loader(k: int, lists: dict, root: str, crop: int, n: int, workers: int):
    ds = imagenet_split(root, "train", _val_tf(crop))
    a31_lists.check(lists, ds.targets)
    idx = a31_lists.holdout_indices(lists, k)[:n].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=64, shuffle=False,
                                       num_workers=min(workers, 4))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", choices=sorted(CELLS), required=True)
    p.add_argument("--data", required=True, help="train/ and val/ (scripts/fetch_imagenet.py)")
    p.add_argument("--lists", default="../results/a31/lists.json", help="scripts/a31_lists.py output")
    p.add_argument("--standard", default="../data/imagenet_val_6400", help="flat directory of the section-6 subset")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--crop", type=int, default=224)
    p.add_argument("--n-measure", type=int, default=6400)
    p.add_argument("--k", type=int, default=None, help="smoke test: override the cell's class count")
    p.add_argument("--epochs", type=int, default=None, help="smoke test: override the cell's epochs")
    p.add_argument("--lr", type=float, default=None, help="smoke test: override the recipe's learning rate (0.1)")
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke test")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke test")
    a = p.parse_args(argv)

    cell = CELLS[a.cell]
    k = a.k if a.k is not None else cell["K"]
    epochs = a.epochs if a.epochs is not None else cell["epochs"]
    lists = json.load(open(a.lists))
    from layerspec.api import image_loader
    primary = image_loader(a.standard, batch_size=64, workers=min(a.workers, 4), image_size=a.crop, limit=a.n_measure)
    secondary = holdout_loader(k, lists, a.data, a.crop, a.n_measure, a.workers)

    import layerspec.api as api
    import scripts.train_imagenet_a17 as a17
    import scripts.trajectory_cifar as tc
    saved = (a17.loaders, api.image_loader, tc.measure)
    probe = tc.measure

    def measure(model, loader, device, positions, epoch, acc, seed, **kw):
        df = probe(model, loader, device, positions, epoch, acc, seed, **kw)
        if epoch in (0, epochs):
            probe(model, secondary, device, positions, epoch, acc, seed, **kw).to_csv(
                os.path.join(a.out, f"secondary_epoch_{epoch:03d}_layers.csv"), index=False)
        return df

    a17.loaders = lambda root, bs, workers, seed, crop_: cell_loaders(k, lists, root, bs, workers, seed, crop_)
    api.image_loader = lambda *args, **kw: primary
    tc.measure = measure
    argv17 = ["--arch", "resnet50", "--data", a.data, "--measure-data", "a31:standard", "--out", a.out,
              "--epochs", str(epochs), "--batch-size", str(a.batch_size), "--workers", str(a.workers),
              "--seed", str(a.seed), "--device", a.device, "--crop", str(a.crop), "--measure-size", str(a.crop),
              "--positions", str(a.positions), "--schedule", *[str(e) for e in schedule(epochs)]]
    for flag, v in (("--max-train-batches", a.max_train_batches), ("--max-val-batches", a.max_val_batches), ("--lr", a.lr)):
        if v is not None:
            argv17 += [flag, str(v)]
    try:
        rc = a17.main(argv17)
    finally:
        a17.loaders, api.image_loader, tc.measure = saved
    sec = [pd.read_csv(f) for e in (0, epochs)
           if os.path.exists(f := os.path.join(a.out, f"secondary_epoch_{e:03d}_layers.csv"))]
    if sec:
        pd.concat(sec, ignore_index=True).to_csv(os.path.join(a.out, "secondary_layers.csv"), index=False)
    with open(os.path.join(a.out, "a31.json"), "w") as fh:
        json.dump({"cell": a.cell, "K": k, "epochs": epochs, "steps_per_epoch": math.ceil(k * lists["per_class"] / a.batch_size),
                   "crop": a.crop, "n_measure": a.n_measure, "seed": a.seed, "classes": a31_lists.cell_classes(lists, k),
                   "n_secondary_images": len(secondary.dataset), "lists": a.lists, "data": a.data, "standard": a.standard}, fh, indent=2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
