"""Validation gate: reproduce Garg, Panda & Roy (IEEE Access 2019, Table 2).

Before extending the measurement to architectures nobody has covered, the
pipeline has to reproduce the one published result it can be checked against.
If our per-layer numbers for VGG-16_BN on CIFAR-10 do not land near theirs, the
problem is our pipeline, not the literature -- and every downstream number is
suspect.

Their reported significant dimensions (99.9% explained variance) for
VGG-16_BN / CIFAR-10, against nominal widths:

    layer :   1    2    3    4    5    6    7    8    9   10   11   12   13
    k*    :  11   42  103  118  238  249  249  424  271  160   36   38   42
    C     :  64   64  128  128  256  256  256  512  512  512  512  512  512
    k*/C  : .17  .66  .80  .92  .93  .97  .97  .83  .53  .31  .07  .07  .08

Caveats to keep in mind when comparing:
  * Their exact training recipe is not fully specified in the paper, and k* is
    sensitive to how well the network converged.  Expect agreement in *shape*
    and rough magnitude, not to the digit.
  * They count every spatial position as a sample; we subsample positions (see
    hooks.py).  With n/C large both estimate the same covariance, but their
    effective sample size was smaller than their nominal one.
  * A tau of 0.999 is loose.  We report several tau, and the comparison against
    their table must use tau = 0.999.

Second purpose: separating two confounded explanations.
------------------------------------------------------

The ImageNet pass reproduced Garg's first eight layers (r = 0.969, MAD = 0.057)
and then diverged: their last five collapse to k*/C = 0.07-0.08 while ours stay
at 0.95-0.97.  The terminal collapse does not happen on ImageNet.  Two
explanations were confounded in that comparison and neither could be ruled out:

  (a) CLASS COUNT.  Neural collapse (Papyan, Han & Donoho 2020) pins the
      penultimate representation's effective rank at C_class - 1, which is 9 on
      CIFAR-10 and 999 on ImageNet.  If that is the mechanism, the collapse
      should follow the class count.

  (b) HEAD DEPTH.  Garg's CIFAR adaptation replaces the ImageNet head with a
      single linear layer, which makes features.40 the penultimate layer.  The
      ImageNet VGG has 4096-4096-1000 after it, so features.40 is three layers
      from the output and neural collapse has somewhere else to happen.  If
      that is the mechanism, the collapse should follow the head, not the
      class count.

The two vary together between the published CIFAR result and our ImageNet one,
so this script runs the 2x2 that separates them: {CIFAR-10, CIFAR-100} x
{small head, deep head}.  Read the result off the interaction --- if k*/C at
features.40 tracks --dataset, (a); if it tracks --head, (b); if both matter,
say so.

Usage:
    # the validation gate on its own
    python scripts/reproduce_garg.py --epochs 100 --out results_garg/

    # the full 2x2 (about four GPU-hours)
    for d in cifar10 cifar100; do for h in small deep; do
      python scripts/reproduce_garg.py --dataset $d --head $h \
             --epochs 100 --out results_garg/
    done; done

Cost: about one GPU-hour on a 3090 per cell, then seconds for the measurement.
Only the (cifar10, small) cell is comparable with Garg's table; the other three
are the contrast.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layerspec import metrics as _metrics          # noqa: E402
from layerspec.hooks import run_probe              # noqa: E402

GARG_TABLE2 = {
    "k_star": [11, 42, 103, 118, 238, 249, 249, 424, 271, 160, 36, 38, 42],
    "C": [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
}


def build_vgg16_bn_cifar(num_classes: int = 10, head: str = "small") -> nn.Module:
    """VGG-16 with BN adapted to 32x32.

    `head="small"` is the standard CIFAR adaptation and the one Garg et al.
    used: a single linear layer, which makes features.40 the penultimate
    representation.  `head="deep"` keeps the ImageNet head's shape
    (4096-4096-num_classes) so that features.40 sits three layers from the
    output, as it does in the ImageNet model.  The convolutional trunk is
    identical in both, which is the point: any difference in the trunk's
    profile is attributable to the head.
    """
    import torchvision.models as tvm

    model = tvm.vgg16_bn(weights=None, num_classes=num_classes)
    if head == "small":
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Linear(512, num_classes)
    elif head == "deep":
        # 2x2 rather than 7x7: at 32x32 input the trunk's output is 1x1, so a
        # 7x7 pool would be mostly padding.  4096 units either way.
        model.avgpool = nn.AdaptiveAvgPool2d(2)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    else:
        raise ValueError(f"unknown head {head!r}: expected 'small' or 'deep'")
    return model


CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}


def cifar_loaders(root: str, batch_size: int, workers: int,
                  dataset: str = "cifar10"):
    from torchvision import datasets, transforms

    mean, std, _ = CIFAR_STATS[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    tr = cls(root, train=True, download=True, transform=train_tf)
    te = cls(root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,
                   num_workers=workers, pin_memory=True, drop_last=True),
        DataLoader(te, batch_size=batch_size, shuffle=False,
                   num_workers=workers, pin_memory=True),
    )


def train(model, train_loader, test_loader, epochs, device, lr=0.1, wd=5e-4):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = evaluate(model, test_loader, device)
            print(f"  epoch {ep+1:3d}/{epochs}  test acc {acc:.2f}%  "
                  f"({time.time()-t0:.0f}s/ep)", flush=True)
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="results_garg")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", default=None,
                   help="skip training and load this state_dict instead")
    p.add_argument("--dataset", default="cifar10",
                   choices=sorted(CIFAR_STATS),
                   help="varies the class count: 10 or 100")
    p.add_argument("--head", default="small", choices=("small", "deep"),
                   help="'small' is Garg's single-linear CIFAR head, which "
                        "makes features.40 penultimate; 'deep' keeps the "
                        "ImageNet 4096-4096-K head, which does not")
    args = p.parse_args(argv)

    n_classes = CIFAR_STATS[args.dataset][2]
    tag = f"{args.dataset}_{args.head}"
    os.makedirs(args.out, exist_ok=True)
    train_loader, test_loader = cifar_loaders(
        args.data_root, args.batch_size, args.workers, args.dataset)

    model = build_vgg16_bn_cifar(n_classes, args.head)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        model = model.to(args.device)
        print(f"loaded {args.checkpoint}")
    else:
        print(f"training VGG-16_BN [{tag}, {n_classes} classes] "
              f"for {args.epochs} epochs")
        model = train(model, train_loader, test_loader, args.epochs, args.device)
        torch.save(model.state_dict(),
                   os.path.join(args.out, f"vgg16_bn_{tag}.pt"))

    acc = evaluate(model, test_loader, args.device)
    print(f"final test accuracy: {acc:.2f}%")

    # Measure on the TEST split, matching the usual convention of probing
    # held-out data.
    print("\nmeasuring per-layer spectra")
    probe = run_probe(model, test_loader, device=args.device,
                      positions_per_image=args.positions,
                      include_activations=False, progress=False)

    rows = []
    for rec in probe.records():
        if rec.acc is None or rec.acc.n < 2:
            continue
        m = _metrics.compute(rec.acc.eigenvalues(), rec.C, rec.acc.n)
        row = rec.meta()
        row.update(m.to_row())
        row["n_over_C"] = rec.acc.n / rec.C
        rows.append(row)

    df = pd.DataFrame(rows)
    df["dataset"] = args.dataset
    df["head"] = args.head
    df["n_classes"] = n_classes
    df["test_acc"] = acc
    df.to_csv(os.path.join(args.out, f"vgg16_{tag}_layers.csv"), index=False)

    # ---- the cell's own contribution to the 2x2 ----------------------------
    # features.40 is the last conv layer.  Under the small head it is the
    # penultimate representation and neural collapse predicts an effective rank
    # near n_classes - 1; under the deep head it is not, and should not
    # collapse.  Printing it per cell means the 2x2 can be read off four log
    # tails without re-loading anything.
    conv = df[df["kind"] == "conv"].sort_values("depth_index")
    if not conv.empty:
        last = conv.iloc[-1]
        print(f"\n=== 2x2 cell [{tag}] ===")
        print(f"  test accuracy            {acc:.2f}%")
        print(f"  last conv layer          {last['layer']} (C={int(last['C'])})")
        print(f"  k*(0.999)                {int(last['k_star_0.999'])}"
              f"   ratio {last['k_star_ratio_0.999']:.3f}")
        print(f"  k*(0.95)                 {int(last['k_star_0.95'])}"
              f"   ratio {last['k_star_ratio_0.95']:.3f}")
        print(f"  effective rank           {last['effective_rank']:.1f}"
              f"   (neural collapse would predict ~{n_classes - 1})")
        print(f"  n/C                      {last['n_over_C']:.0f}"
              f"   {'OK' if last['n_over_C'] >= 50 else 'BELOW GATE -- do not report'}")

    if args.dataset != "cifar10" or args.head != "small":
        print("\nThis cell is a contrast, not the validation gate; "
              "Garg's table applies only to [cifar10_small].")
        return 0

    # ---- comparison against the published table ----------------------------
    ours = conv
    n = min(len(ours), len(GARG_TABLE2["k_star"]))
    theirs_k = np.array(GARG_TABLE2["k_star"][:n])
    theirs_C = np.array(GARG_TABLE2["C"][:n])
    ours_k = ours["k_star_0.999"].to_numpy()[:n]
    ours_C = ours["C"].to_numpy()[:n]

    cmp = pd.DataFrame({
        "layer": ours["layer"].to_numpy()[:n],
        "C_ours": ours_C, "C_garg": theirs_C,
        "k_ours": ours_k, "k_garg": theirs_k,
        "ratio_ours": ours_k / ours_C,
        "ratio_garg": theirs_k / theirs_C,
        "n_over_C": ours["n_over_C"].to_numpy()[:n],
    })
    cmp["abs_diff"] = (cmp.ratio_ours - cmp.ratio_garg).abs()
    cmp.to_csv(os.path.join(args.out, "garg_comparison.csv"), index=False)

    print("\n=== comparison with Garg et al. Table 2 (tau = 0.999) ===")
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if (cmp.C_ours != cmp.C_garg).any():
        print("\n!! channel counts differ -- layer alignment is wrong, "
              "fix that before reading anything else")

    corr = float(np.corrcoef(cmp.ratio_ours, cmp.ratio_garg)[0, 1])
    mad = float(cmp.abs_diff.mean())
    print(f"\nshape agreement (Pearson r on k*/C): {corr:.3f}")
    print(f"mean absolute difference in k*/C:    {mad:.3f}")
    print(
        "\nVerdict: the gate is about SHAPE, not digits -- their training "
        "recipe is not fully specified.\n"
        "  r > 0.9 and MAD < 0.10  -> reproduced; proceed to the ImageNet pass.\n"
        "  r > 0.7                 -> broadly consistent; investigate the "
        "layers that disagree.\n"
        "  r < 0.7                 -> STOP. Something in the pipeline or the "
        "training is wrong."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
