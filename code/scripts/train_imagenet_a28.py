"""A28 (analysis-plan 9.17): what in ImageNet reorders ResNet-50's profile.  The A17 run (ResNet-50, ImageNet-1k,
224 pixels) is the base cell; each cell here changes one thing and trains with the A17 recipe:

    k100_224         a hundred classes, 224 pixels
    k1000_64         a thousand classes, 64 pixels
    k100_64          both
    k100_64_long     k100_64 trained ten times longer (the optimisation-length control)
    k1000_n130_224   a thousand classes, 130 training images each (the image-count control)

The training loop is scripts/train_imagenet_a17.py itself, untouched: this wrapper only replaces the data it is
given (the loaders, the measurement set) and adds a second measurement at the first and last epoch.

Measurement (k*(0.95)/r_max along the trajectory, at the cell's resolution):
    primary     held-out images of the cell's own task -- the first N validation images (thousand classes) or
                the first N of the held-out training images (hundred classes; scripts/a28_lists.py);
                N = 6,400 at 224 pixels and 25,600 at 64, where stage 4 has four positions per image
    secondary   the 6,400-image validation subset of section 6, at the first and last epoch
                (secondary_epoch_XXX_layers.csv), for the correlation with the A17 profile on the same images
The image-count cell is measured on the section-6 subset exactly as A17 was and has no secondary set.

    python scripts/train_imagenet_a28.py --cell k100_224 --data /workspace/imagenet --lists ../results/a28/lists.json \\
        --standard /workspace/imagenet_val_6400 --seed 0 --out ../results/a28/k100_224_s0
    python scripts/train_imagenet_a28.py --cell k1000_64 --data /workspace/imagenet80 ...   (scripts/a28_downsample.py)

Resumable, as the A17 runs are.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import a28_lists                                        # noqa: E402
from scripts.imagenet_blob import imagenet_split                     # noqa: E402
from scripts.train_imagenet_r18 import MEAN, STD, _val_tf            # noqa: E402

CELLS = {
    "k100_224":       dict(classes=True,  per_class=False, crop=224, epochs=90,  measure="holdout",  n_measure=6400),
    "k1000_64":       dict(classes=False, per_class=False, crop=64,  epochs=90,  measure="val",      n_measure=25600),
    "k100_64":        dict(classes=True,  per_class=False, crop=64,  epochs=90,  measure="holdout",  n_measure=25600),
    "k100_64_long":   dict(classes=True,  per_class=False, crop=64,  epochs=900, measure="holdout",  n_measure=25600),
    "k1000_n130_224": dict(classes=False, per_class=True,  crop=224, epochs=90,  measure="standard", n_measure=6400),
}
SCHEDULE_90 = [0, 1, 2, 5, 10, 20, 30, 45, 60, 75, 90]


def schedule(epochs: int) -> list[int]:
    """The A17 measurement epochs, stretched with the run (the long cell measures at ten times each)."""
    k = max(1, epochs // 90)
    s = sorted({e * k for e in SCHEDULE_90 if e * k <= epochs} | {0, epochs})
    return s


def cell_loaders(cell: dict, lists: dict, root: str, batch_size: int, workers: int, seed: int, crop: int):
    """The (train, val, n_classes) triple of train_imagenet_r18.loaders for one cell; same transforms, same options."""
    from torchvision import transforms
    from scripts.class_subset import RemapSubset, label_map, subset_indices
    train_tf = transforms.Compose([transforms.RandomResizedCrop(crop), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = imagenet_split(root, "train", train_tf)
    va = imagenet_split(root, "val", _val_tf(crop))
    if tr.classes != va.classes:
        raise SystemExit("train/ and val/ class sets differ")
    n_classes = len(tr.classes)
    if cell["classes"] or cell["per_class"]:
        a28_lists.check(lists, tr.targets)
        idx = a28_lists.train_indices(lists, tr.targets, cell["classes"], cell["per_class"])
        if cell["classes"]:
            lmap = label_map(lists["classes"])
            tr = RemapSubset(tr, idx, lmap)
            va = RemapSubset(va, subset_indices(va.targets, lists["classes"], None), lmap)
            n_classes = len(lists["classes"])
        else:
            tr = torch.utils.data.Subset(tr, idx.tolist())
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
              prefetch_factor=4 if workers > 0 else None)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **kw),
            torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, **kw), n_classes)


def primary_loader(cell: dict, lists: dict, root: str, standard: str, crop: int, n: int, workers: int):
    from layerspec.api import image_loader
    if cell["measure"] == "standard":
        return image_loader(standard, batch_size=64, workers=min(workers, 4), image_size=crop, limit=n)
    if cell["measure"] == "holdout":
        ds = imagenet_split(root, "train", _val_tf(crop))
        a28_lists.check(lists, ds.targets)
        idx = lists["holdout"][:n]
    else:
        ds = imagenet_split(root, "val", _val_tf(crop))
        idx = list(range(min(n, len(ds))))
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=64, shuffle=False,
                                       num_workers=min(workers, 4))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cell", choices=sorted(CELLS), required=True)
    p.add_argument("--data", required=True, help="train/ and val/ at the cell's resolution (64-pixel cells: the a28_downsample copy)")
    p.add_argument("--lists", default="../results/a28/lists.json", help="scripts/a28_lists.py output")
    p.add_argument("--standard", default="../data/imagenet_val_6400", help="flat directory of the section-6 subset")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--epochs", type=int, default=None, help="smoke test: override the cell's epochs")
    p.add_argument("--lr", type=float, default=None, help="smoke test: override the recipe's learning rate (0.1)")
    p.add_argument("--crop", type=int, default=None, help="smoke test: override the cell's resolution")
    p.add_argument("--n-measure", type=int, default=None, help="smoke test: override the size of the measurement sets")
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke test")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke test")
    a = p.parse_args(argv)

    cell = CELLS[a.cell]
    epochs = a.epochs if a.epochs is not None else cell["epochs"]
    crop = a.crop if a.crop is not None else cell["crop"]
    n_measure = a.n_measure if a.n_measure is not None else cell["n_measure"]
    lists = json.load(open(a.lists)) if (cell["classes"] or cell["per_class"]) else {}
    primary = primary_loader(cell, lists, a.data, a.standard, crop, n_measure, a.workers)
    secondary = None
    if cell["measure"] != "standard":
        from layerspec.api import image_loader
        secondary = image_loader(a.standard, batch_size=64, workers=min(a.workers, 4), image_size=crop,
                                 limit=min(n_measure, 6400) if a.n_measure is not None else None)

    import layerspec.api as api
    import scripts.train_imagenet_a17 as a17
    import scripts.trajectory_cifar as tc
    saved = (a17.loaders, api.image_loader, tc.measure)
    probe = tc.measure

    def measure(model, loader, device, positions, epoch, acc, seed, **kw):
        df = probe(model, loader, device, positions, epoch, acc, seed, **kw)
        if secondary is not None and epoch in (0, epochs):
            probe(model, secondary, device, positions, epoch, acc, seed, **kw).to_csv(
                os.path.join(a.out, f"secondary_epoch_{epoch:03d}_layers.csv"), index=False)
        return df

    a17.loaders = lambda root, bs, workers, seed, crop_: cell_loaders(cell, lists, root, bs, workers, seed, crop_)
    api.image_loader = lambda *args, **kw: primary
    tc.measure = measure
    argv17 = ["--arch", "resnet50", "--data", a.data, "--measure-data", f"a28:{cell['measure']}", "--out", a.out,
              "--epochs", str(epochs), "--batch-size", str(a.batch_size), "--workers", str(a.workers),
              "--seed", str(a.seed), "--device", a.device, "--crop", str(crop), "--measure-size", str(crop),
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
    with open(os.path.join(a.out, "a28.json"), "w") as fh:
        json.dump({"cell": a.cell, **cell, "epochs": epochs, "crop": crop, "n_measure": n_measure, "seed": a.seed,
                   "n_measure_images": len(primary.dataset), "lists": a.lists if lists else None,
                   "classes": lists.get("classes"), "data": a.data, "standard": a.standard}, fh, indent=2)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
