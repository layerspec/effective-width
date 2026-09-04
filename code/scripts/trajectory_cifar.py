"""Per-layer effective-width profile along a training trajectory.

The cross-checkpoint analysis (notes/round2-2026-09-04.md section 6) found
that the trained ResNet-50 profile is uncorrelated with the random-init
profile and has the opposite depth trend.  This script asks WHEN that
happens: it trains VGG-16_BN on CIFAR-10 with the same recipe as the Garg
gate (reproduce_garg.py) and measures the conv-layer spectra on the test
split at a fixed schedule of epochs, epoch 0 being the initialisation.

    python scripts/trajectory_cifar.py --epochs 100 --out ../results/trajectory

Outputs one <out>/epoch_<E>_layers.csv per measured epoch and a combined
<out>/trajectory_layers.csv with an `epoch` column.  Positions are sampled
per image exactly as in the ImageNet runs; at 32x32 input the test split's
10,000 images x 16 positions give n/C >= 312 at C = 512.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec import metrics as _metrics                       # noqa: E402
from layerspec.hooks import run_probe                            # noqa: E402
from scripts.reproduce_garg import (build_vgg16_bn_cifar,        # noqa: E402
                                    cifar_loaders, evaluate)

DEFAULT_SCHEDULE = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def measure(model, loader, device, positions, epoch, acc, seed,
            max_batches=None) -> pd.DataFrame:
    was_training = model.training
    probe = run_probe(model, loader, device=device, positions_per_image=positions,
                      include_activations=False, progress=False, seed=seed,
                      max_batches=max_batches)
    rows = []
    for rec in probe.records():
        if rec.acc is None or rec.acc.n < 2:
            continue
        m = _metrics.compute(rec.acc.eigenvalues(), rec.C, rec.acc.n)
        row = rec.meta()
        row.update(m.to_row())
        row["n_over_C"] = rec.acc.n / rec.C
        if rec.acc_pooled is not None and rec.acc_pooled.n >= 2:
            mp = _metrics.compute(rec.acc_pooled.eigenvalues(), rec.C, rec.acc_pooled.n)
            row.update({"pooled_" + k: v for k, v in mp.to_row().items()})
            row["pooled_n_over_C"] = rec.acc_pooled.n / rec.C
        rows.append(row)
    if was_training:
        model.train()
    df = pd.DataFrame(rows)
    df["epoch"] = epoch
    df["test_acc"] = acc
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", default="../data")
    p.add_argument("--out", default="../results/trajectory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--schedule", type=int, nargs="+", default=None,
                   help="epochs at which to measure (0 = init); default is a "
                        "log-ish grid up to --epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="smoke-test aid: stop each epoch after this many batches")
    p.add_argument("--max-measure-batches", type=int, default=None,
                   help="smoke-test aid: measure on this many test batches only")
    args = p.parse_args(argv)

    schedule = sorted(set(args.schedule or [e for e in DEFAULT_SCHEDULE if e <= args.epochs]))
    if args.epochs not in schedule:
        schedule.append(args.epochs)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    train_loader, test_loader = cifar_loaders(args.data_root, args.batch_size,
                                              args.workers, "cifar10")
    model = build_vgg16_bn_cifar(10, "small").to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = len(train_loader) if args.max_train_batches is None \
        else min(len(train_loader), args.max_train_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    frames = []

    def checkpoint(epoch):
        acc = evaluate(model, test_loader, args.device)
        t0 = time.time()
        df = measure(model, test_loader, args.device, args.positions, epoch, acc,
                     args.seed, max_batches=args.max_measure_batches)
        df.to_csv(os.path.join(args.out, f"epoch_{epoch:03d}_layers.csv"), index=False)
        frames.append(df)
        conv = df[df.kind == "conv"].sort_values("depth_index")
        r = conv["k_star_ratio_0.95"].to_numpy()
        from scipy.stats import spearmanr
        rho = spearmanr(np.arange(len(r)), r).correlation
        print(f"  [measure] epoch {epoch:3d}  test acc {acc:5.2f}%  "
              f"k*(.95)/C median {np.median(r):.3f}  rho(depth) {rho:+.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(args.out, "trajectory_layers.csv"), index=False)

    print(f"schedule: {schedule}", flush=True)
    checkpoint(0)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"  epoch {ep:3d}/{args.epochs}  loss {loss.item():.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ep in schedule:
            checkpoint(ep)

    torch.save(model.state_dict(), os.path.join(args.out, "vgg16_bn_cifar10_final.pt"))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
