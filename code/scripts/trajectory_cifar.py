"""Per-layer effective-width profile along a training trajectory.

The cross-checkpoint analysis (notes/round2-2026-09-04.md section 6) found
that the trained ResNet-50 profile is uncorrelated with the random-init
profile and has the opposite depth trend.  This script asks WHEN that
happens: it trains VGG-16_BN on CIFAR-10 with the same recipe as the Garg
gate (reproduce_garg.py) and measures the conv-layer spectra on the test
split at a fixed schedule of epochs, epoch 0 being the initialisation.

    python scripts/trajectory_cifar.py --epochs 100 --out ../results/trajectory
    python scripts/trajectory_cifar.py --arch resnet50 --epochs 100 \
        --out ../results/trajectory_resnet50          # bottleneck block type

Outputs one <out>/epoch_<E>_layers.csv per measured epoch and a combined
<out>/trajectory_layers.csv with an `epoch` column.  A resume.pt is written
after every epoch and picked up automatically on restart with the same --out,
so a run can be spread over several sessions (removed when the run finishes).  Positions are sampled
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


VGG16_WIDTHS = (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512)
_VGG16_POOL_AFTER = {1, 3, 6, 9, 12}      # torchvision cfg "D": M after these conv indices


def build_vgg16_bn_cifar_widths(widths, num_classes: int = 10):
    """VGG-16_BN (CIFAR head, as build_vgg16_bn_cifar(head="small")) with the 13
    conv widths given explicitly.  With VGG16_WIDTHS it is the standard network,
    parameter for parameter and (under the same seed) weight for weight; with
    the widths read off a trained network's profile it is the arm of the
    width-from-the-ruler experiment (analysis-plan 9.5)."""
    import torch.nn as nn
    from torchvision.models.vgg import VGG, make_layers
    widths = [int(w) for w in widths]
    if len(widths) != 13 or min(widths) < 1:
        raise ValueError(f"--widths needs 13 positive integers, got {widths}")
    cfg = []
    for i, w in enumerate(widths):
        cfg.append(w)
        if i in _VGG16_POOL_AFTER:
            cfg.append("M")
    m = VGG(make_layers(cfg, batch_norm=True), num_classes=num_classes)
    m.avgpool = nn.AdaptiveAvgPool2d(1)
    m.classifier = nn.Linear(widths[-1], num_classes)
    return m


def build_resnet50_cifar(num_classes: int = 10):
    """torchvision ResNet-50 with the usual CIFAR stem: 3x3 stride-1 conv1
    and no max-pool, so a 32x32 input reaches layer4 at 4x4.  Bottleneck
    blocks, expansion 4, r_max/C = 0.25 at every conv3 -- the block type
    whose trained profile (round-2 section 8) was uncorrelated with its
    initialisation.  Random init is torchvision's default (Kaiming, zero
    last-BN gamma disabled), seeded by the caller."""
    import torch.nn as nn
    import torchvision
    m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_resnet_basic52_cifar(num_classes: int = 10):
    """Basic-block ResNet with the SAME conv count as the bottleneck ResNet-50
    (52 vs 53: stem + 2x24 block convs + 3 downsample vs stem + 3x16 + 4),
    the same stage widths at the block input (64/128/256/512), the same CIFAR
    stem, the same depth-per-stage pattern.  Only the block type differs.
    Registered as the controlled block-type experiment (analysis-plan 9.1)."""
    import torch.nn as nn
    import torchvision
    from torchvision.models.resnet import BasicBlock
    m = torchvision.models.ResNet(BasicBlock, [6, 6, 6, 6], num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_mobilenetv2_cifar(num_classes: int = 10):
    """torchvision MobileNetV2 (inverted residual, expansion 6) with the first
    two strides removed so a 32x32 input ends at 4x4, as is usual for CIFAR."""
    import torch.nn as nn
    import torchvision
    m = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    m.features[0][0].stride = (1, 1)          # stem
    m.features[2].conv[1][0].stride = (1, 1)  # first stride-2 inverted residual
    return m


ARCHS = {"vgg16_bn": lambda: build_vgg16_bn_cifar(10, "small"),
         "resnet50": lambda: build_resnet50_cifar(10),
         "resnet_basic52": lambda: build_resnet_basic52_cifar(10),
         "mobilenetv2": lambda: build_mobilenetv2_cifar(10)}


def ortho_penalty(model, kind: str) -> "torch.Tensor":
    """Kernel-orthogonality penalties of Bansal et al. (NeurIPS 2018) on every
    dense conv and linear weight, W reshaped to (C_out, d).
      so   : ||W W^T - I||_F^2 if C_out <= d else ||W^T W - I||_F^2  (Massart 2022: the
             two differ by a constant, so the choice only fixes the constant)
      srip : spectral norm of (W^T W - I), by two power iterations, summed
    Registered as arms of analysis-plan 9.2; ONI (Huang 2020) is a layer, not a
    penalty, and is not implemented here."""
    import torch.nn as nn
    total = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d) and mod.groups == 1:
            W = mod.weight.reshape(mod.out_channels, -1)
        elif isinstance(mod, nn.Linear):
            W = mod.weight
        else:
            continue
        n, d = W.shape
        G = W @ W.T if n <= d else W.T @ W
        I = torch.eye(G.shape[0], device=W.device, dtype=W.dtype)
        M = G - I
        if kind == "so":
            pen = (M * M).sum()
        elif kind == "srip":
            v = torch.randn(M.shape[0], 1, device=W.device, dtype=W.dtype)
            v = v / (v.norm() + 1e-12)
            for _ in range(2):
                v = M @ (M @ v)
                v = v / (v.norm() + 1e-12)
            pen = (M @ v).norm()
        else:
            raise ValueError(kind)
        total = pen if total is None else total + pen
    return total

DEFAULT_SCHEDULE = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class GPUCifarTrain:
    """The CIFAR-10 training split held on the device as uint8, with the same
    augmentation as reproduce_garg.cifar_loaders (RandomCrop(32, padding=4) +
    RandomHorizontalFlip + Normalize) applied on the device per batch.  Same
    distribution, no CPU data loading: on a pod whose CPUs are shared between
    several runs the loader, not the GPU, was the bottleneck (2026-09-08)."""

    def __init__(self, root, batch_size, device, seed):
        from torchvision import datasets
        from scripts.reproduce_garg import CIFAR_STATS
        ds = datasets.CIFAR10(root, train=True, download=True)
        self.x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(device)   # N,3,32,32 uint8
        self.y = torch.tensor(ds.targets, device=device)
        mean, std, _ = CIFAR_STATS["cifar10"]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.bs, self.device = batch_size, device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __len__(self):
        return len(self.x) // self.bs          # drop_last, as the loader does

    def __iter__(self):
        N = len(self.x)
        perm = torch.randperm(N, generator=self.gen, device=self.device)
        ar = torch.arange(32, device=self.device)
        for i in range(len(self)):
            idx = perm[i * self.bs:(i + 1) * self.bs]
            xb = self.x[idx].float().div_(255)
            xp = F.pad(xb, (4, 4, 4, 4))
            n = xb.shape[0]
            oh = torch.randint(0, 9, (n,), generator=self.gen, device=self.device)
            ow = torch.randint(0, 9, (n,), generator=self.gen, device=self.device)
            ih = (oh[:, None] + ar)[:, :, None].expand(n, 32, 32)
            iw = (ow[:, None] + ar)[:, None, :].expand(n, 32, 32)
            nidx = torch.arange(n, device=self.device)[:, None, None]
            xc = xp.permute(0, 2, 3, 1)[nidx, ih, iw].permute(0, 3, 1, 2)      # n,3,32,32
            flip = torch.rand(n, generator=self.gen, device=self.device) < 0.5
            xc = torch.where(flip[:, None, None, None], xc.flip(3), xc)
            yield (xc - self.mean) / self.std, self.y[idx]


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
    p.add_argument("--arch", choices=sorted(ARCHS), default="vgg16_bn")
    p.add_argument("--ortho", choices=["none", "so", "srip"], default="none",
                   help="kernel-orthogonality penalty arm (analysis-plan 9.2)")
    p.add_argument("--ortho-lambda", type=float, default=None,
                   help="penalty weight; default 1e-4 for so, 1e-2 for srip (Bansal et al.)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="smoke-test aid: stop each epoch after this many batches")
    p.add_argument("--max-measure-batches", type=int, default=None,
                   help="smoke-test aid: measure on this many test batches only")
    p.add_argument("--gpu-data", action="store_true",
                   help="hold the training split on the device and augment there (no loader workers)")
    p.add_argument("--widths", type=int, nargs=13, default=None, metavar="W",
                   help="vgg16_bn only: the 13 conv widths (analysis-plan 9.5 arms b/c/d)")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="also write epoch_<E>.pt (state_dict) at every measured epoch (analysis-plan 9.6)")
    args = p.parse_args(argv)
    if args.widths is not None and args.arch != "vgg16_bn":
        p.error("--widths is only defined for --arch vgg16_bn")

    schedule = sorted(set(args.schedule or [e for e in DEFAULT_SCHEDULE if e <= args.epochs]))
    if args.epochs not in schedule:
        schedule.append(args.epochs)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    train_loader, test_loader = cifar_loaders(args.data_root, args.batch_size,
                                              args.workers, "cifar10")
    if args.gpu_data:
        train_loader = GPUCifarTrain(args.data_root, args.batch_size, args.device, args.seed)
    model = (build_vgg16_bn_cifar_widths(args.widths, 10) if args.widths is not None
             else ARCHS[args.arch]()).to(args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    steps = len(train_loader) if args.max_train_batches is None \
        else min(len(train_loader), args.max_train_batches)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)
    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    frames = []
    state_path = os.path.join(args.out, "resume.pt")
    start_epoch = 1
    if os.path.exists(state_path):
        # Resume: the run was interrupted (laptop closed, pod killed).  Model,
        # optimiser, schedule, RNG and the measured frames all come back, so
        # the trajectory is the same one, not a new one.
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["rng_cpu"].cpu())
        start_epoch = st["epoch"] + 1
        traj = os.path.join(args.out, "trajectory_layers.csv")
        if os.path.exists(traj):
            frames.append(pd.read_csv(traj))
        print(f"resumed from {state_path} at epoch {st['epoch']} (next: {start_epoch})", flush=True)

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(), "epoch": epoch,
                    "arch": args.arch, "seed": args.seed, "ortho": args.ortho}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    def checkpoint(epoch):
        acc = evaluate(model, test_loader, args.device)
        t0 = time.time()
        df = measure(model, test_loader, args.device, args.positions, epoch, acc,
                     args.seed, max_batches=args.max_measure_batches)
        df.to_csv(os.path.join(args.out, f"epoch_{epoch:03d}_layers.csv"), index=False)
        if args.save_checkpoints:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(args.out, f"epoch_{epoch:03d}.pt"))
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
    if start_epoch == 1:
        checkpoint(0)
    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            if args.ortho != "none":
                lam = args.ortho_lambda if args.ortho_lambda is not None else (1e-4 if args.ortho == "so" else 1e-2)
                loss = loss + lam * ortho_penalty(model, args.ortho)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"  epoch {ep:3d}/{args.epochs}  loss {loss.item():.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if ep in schedule:
            checkpoint(ep)
        save_state(ep)

    tag = args.arch + ("" if args.ortho == "none" else f"_{args.ortho}")
    torch.save(model.state_dict(), os.path.join(args.out, f"{tag}_cifar10_final.pt"))
    if os.path.exists(state_path):
        os.remove(state_path)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
