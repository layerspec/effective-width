"""Widths for the width-from-the-ruler experiment (analysis-plan 9.5).

Measures a trained full-width VGG-16_BN on CIFAR-10 *training* images (never
the test split), reads k*(0.999) and k*(0.95) per conv layer, and prints the
13 widths of the three retraining arms:

    b  ruler   width_l = max(8, ceil(k*(0.999)_l))
    c  uniform every layer scaled by one factor so that the parameter count
               matches arm b within 2%
    d  ruler95 width_l = max(8, ceil(k*(0.95)_l))

    python scripts/width_from_ruler.py \
        --checkpoint ../results/round3_pt/a5_vgg16_bn_none_s0/vgg16_bn_cifar10_final.pt \
        --data ../data --device mps --out ../results/a18/widths.json

ResNet-18 (analysis-plan 9.11; basic blocks, so a stage shares one output
width -- rule in scripts/resnet_widths.py) on CIFAR or on an ImageNet
training folder:

    python scripts/width_from_ruler.py --arch resnet18 --tensor act \
        --checkpoint ../results/a18_r18/resnet18_full_s0/resnet18_cifar10_final.pt \
        --data ../data --out ../results/a18_r18/widths_act.json
    python scripts/width_from_ruler.py --arch resnet18 --tensor act --dataset imagenet \
        --checkpoint ../results/a18_imagenet/resnet18_full_s0/final.pt \
        --data /data/imagenet/train --out ../results/a18_imagenet/widths_act.json

The JSON carries the per-layer table (C, r_max, n/C, k* at both tau, gate
flag), the three width lists and their parameter counts, so the arms can be
launched with `trajectory_cifar.py --widths ...` and the numbers audited.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layerspec                                                   # noqa: E402
from scripts.reproduce_garg import CIFAR_STATS, build_vgg16_bn_cifar   # noqa: E402
from scripts.trajectory_cifar import VGG16_WIDTHS, build_vgg16_bn_cifar_widths  # noqa: E402
from scripts import resnet_widths as rw                                          # noqa: E402

MIN_WIDTH = 8
IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000)


def train_image_loader(root: str, n_images: int, batch_size: int, workers: int, seed: int,
                       dataset: str = "cifar10"):
    """`n_images` CIFAR training images, a seeded random subset, test-time
    transform (no augmentation), deterministic order."""
    from torchvision import datasets, transforms
    mean, std, _ = CIFAR_STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    cls = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    ds = cls(root, train=True, download=True, transform=tf)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n_images].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size,
                                       shuffle=False, num_workers=workers)


def imagenet_train_image_loader(root: str, n_images: int, batch_size: int, workers: int, seed: int):
    """`n_images` images from an ImageFolder of ImageNet TRAINING images (the
    class sub-folders written by fetch_imagenet.py), a seeded random subset,
    the evaluation transform (resize 256, centre crop 224, no augmentation)."""
    from torchvision import datasets, transforms
    mean, std, _ = IMAGENET_STATS
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = datasets.ImageFolder(root, transform=tf)
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n_images].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size,
                                       shuffle=False, num_workers=workers)


def n_params(widths, num_classes: int = 10) -> int:
    return sum(p.numel() for p in build_vgg16_bn_cifar_widths(widths, num_classes).parameters())


def uniform_widths_matching(target_params: int, tol: float = 0.02, num_classes: int = 10) -> list[int]:
    """Scale VGG16_WIDTHS by one factor (bisection) until the parameter count is
    within `tol` of `target_params`; widths are rounded and floored at MIN_WIDTH."""
    lo, hi = 0.01, 1.0
    best = None
    for _ in range(40):
        f = (lo + hi) / 2
        w = [max(MIN_WIDTH, int(round(f * c))) for c in VGG16_WIDTHS]
        n = n_params(w, num_classes)
        best = (w, n, f)
        if abs(n - target_params) <= tol * target_params:
            break
        if n > target_params:
            hi = f
        else:
            lo = f
    return best


def write_resnet18(a, prof, num_classes: int) -> int:
    """ResNet-18 arms of plan 9.11: (e) ruler_act by the stage rule and (c) uniform at the same
    parameter count.  The JSON mirrors the VGG one; `sites` keeps every reading the rule used."""
    stem = "imagenet" if a.dataset == "imagenet" else "cifar"
    t = prof.table
    rule = rw.widths_from_profile(t, tau=0.999, min_width=MIN_WIDTH)
    ruler = rule["widths"]
    full_n, ruler_n = rw.n_params(rw.RESNET18_WIDTHS, num_classes, stem), rw.n_params(ruler, num_classes, stem)
    uniform, uniform_n, factor = rw.uniform_widths_matching(ruler_n, num_classes=num_classes, stem=stem)
    act = t[t.kind == "act"].sort_values("depth_index")
    print(f"{'site':22s} {'C':>4s} {'n/C':>6s} {'k*.95':>6s} {'k*.999':>7s}")
    for _, r in act.iterrows():
        print(f"{r.layer + '#' + str(int(r.call_index)):22s} {int(r.C):4d} {r.n_over_C:6.0f} "
              f"{r['k_star_0.95']:6.0f} {r['k_star_0.999']:7.0f}")
    names = ["stem"] + [f"s{s + 1}_{k}" for s in range(4) for k in ("out", "mid0", "mid1")]
    print(f"{'width':10s} {'full':>5s} {'ruler':>6s} {'uniform':>8s}")
    for nm, f_, r_, u_ in zip(names, rw.RESNET18_WIDTHS, ruler, uniform):
        print(f"{nm:10s} {f_:5d} {r_:6d} {u_:8d}")
    print(f"params: full {full_n:,}  ruler {ruler_n:,} ({ruler_n / full_n:.3f})  "
          f"uniform {uniform_n:,} (factor {factor:.3f}, {uniform_n / ruler_n - 1:+.1%} vs ruler)")
    out = {"checkpoint": a.checkpoint, "n_images": a.n_images, "positions": a.positions, "seed": a.seed,
           "split": "train", "min_width": MIN_WIDTH, "arch": "resnet18", "stem": stem,
           "width_order": names, "rule": "plan 9.11: stem/mid after ReLU; stage out = max over blocks after add+ReLU",
           "sites": rule["sites"],
           "layers": act[["layer", "call_index", "C", "n_samples", "n_over_C", "ok", "k_star_0.95", "k_star_0.999"]]
           .to_dict(orient="records"),
           "widths": {"full": list(rw.RESNET18_WIDTHS), "ruler": ruler, "uniform": uniform},
           "params": {"full": full_n, "ruler": ruler_n, "uniform": uniform_n},
           "macs": {k: rw.macs(rw.build_resnet18_widths(v, num_classes, stem), hw)
                    for hw in [(224, 224) if stem == "imagenet" else (32, 32)]
                    for k, v in (("full", rw.RESNET18_WIDTHS), ("ruler", ruler), ("uniform", uniform))},
           "uniform_factor": factor, "dataset": a.dataset, "num_classes": num_classes, "tensor": a.tensor}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print("wrote", a.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="state_dict of the trained full-width VGG-16_BN")
    p.add_argument("--data", default="../data")
    p.add_argument("--n-images", type=int, default=6400)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="JSON file to write")
    p.add_argument("--dataset", choices=("cifar10", "cifar100", "imagenet"), default="cifar10",
                   help="imagenet: --data is an ImageFolder of TRAINING images (plan 9.11)")
    p.add_argument("--arch", choices=("vgg16_bn", "resnet18"), default="vgg16_bn",
                   help="resnet18 applies the stage rule of scripts/resnet_widths.py (plan 9.11)")
    p.add_argument("--tensor", choices=("conv", "act"), default="conv",
                   help="read k* at the conv output (default; the paper's ruler) or after the ReLU, "
                        "the tensor the next layer consumes (analysis-plan 9.8, arm ruler_act)")
    a = p.parse_args(argv)

    if a.dataset == "imagenet":
        num_classes = IMAGENET_STATS[2]
        loader = imagenet_train_image_loader(a.data, a.n_images, a.batch_size, a.workers, a.seed)
    else:
        num_classes = CIFAR_STATS[a.dataset][2]
        loader = train_image_loader(a.data, a.n_images, a.batch_size, a.workers, a.seed, dataset=a.dataset)
    if a.arch == "resnet18":
        if a.tensor != "act":
            raise SystemExit("--arch resnet18 reads widths after the ReLU only (plan 9.11): use --tensor act")
        model = rw.build_resnet18_widths(rw.RESNET18_WIDTHS, num_classes,
                                         "imagenet" if a.dataset == "imagenet" else "cifar")
    else:
        if a.dataset == "imagenet":
            raise SystemExit("--dataset imagenet is only defined for --arch resnet18")
        model = build_vgg16_bn_cifar(num_classes, "small")
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    prof = layerspec.profile(model, loader, device=a.device, positions=a.positions,
                             taus=(0.95, 0.999), include_activations=(a.tensor == "act"), include_blocks=False,
                             seed=a.seed)
    if a.arch == "resnet18":
        return write_resnet18(a, prof, num_classes)
    conv = prof.table[prof.table.kind == a.tensor].sort_values("depth_index").reset_index(drop=True)
    if a.tensor == "act":      # the bound is the conv's; carry it over for the printout
        rm = prof.table[prof.table.kind == "conv"].sort_values("depth_index")["r_max"].to_numpy()
        conv["r_max"] = rm[: len(conv)]
    if len(conv) != 13:
        raise SystemExit(f"expected 13 conv layers, measured {len(conv)}")
    if not conv["ok"].all():
        raise SystemExit("a layer fails the n/C >= 50 gate; use more images:\n" + "\n".join(prof.warnings()))

    ruler = [max(MIN_WIDTH, math.ceil(k)) for k in conv["k_star_0.999"]]
    ruler95 = [max(MIN_WIDTH, math.ceil(k)) for k in conv["k_star_0.95"]]
    full_n, ruler_n, ruler95_n = n_params(VGG16_WIDTHS, num_classes), n_params(ruler, num_classes), n_params(ruler95, num_classes)
    uniform, uniform_n, factor = uniform_widths_matching(ruler_n, num_classes=num_classes)

    print(f"{'layer':12s} {'C':>4s} {'r_max':>5s} {'n/C':>6s} {'k*.95':>6s} {'k*.999':>7s}  ruler  ruler95  uniform")
    for i, r in conv.iterrows():
        print(f"{r.layer:12s} {int(r.C):4d} {int(r.r_max):5d} {r.n_over_C:6.0f} {r['k_star_0.95']:6.0f} "
              f"{r['k_star_0.999']:7.0f}  {ruler[i]:5d}  {ruler95[i]:7d}  {uniform[i]:7d}")
    print(f"params: full {full_n:,}  ruler {ruler_n:,} ({ruler_n / full_n:.3f})  "
          f"uniform {uniform_n:,} (factor {factor:.3f}, {uniform_n / ruler_n - 1:+.1%} vs ruler)  "
          f"ruler95 {ruler95_n:,} ({ruler95_n / full_n:.3f})")
    out = {"checkpoint": a.checkpoint, "n_images": a.n_images, "positions": a.positions, "seed": a.seed,
           "split": "train", "min_width": MIN_WIDTH,
           "layers": conv[["layer", "C", "r_max", "n_samples", "n_over_C", "ok", "k_star_0.95", "k_star_0.999"]]
           .to_dict(orient="records"),
           "widths": {"full": list(VGG16_WIDTHS), "ruler": ruler, "uniform": uniform, "ruler95": ruler95},
           "params": {"full": full_n, "ruler": ruler_n, "uniform": uniform_n, "ruler95": ruler95_n},
           "uniform_factor": factor, "dataset": a.dataset, "num_classes": num_classes, "tensor": a.tensor}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
