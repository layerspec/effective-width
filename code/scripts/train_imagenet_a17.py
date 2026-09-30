"""A17 reduced (submission bar D2; analysis-plan 9.4 and its 2026-09-25 addendum): the controlled
block-type experiment on ImageNet-1k.  One recipe, two architectures with the same stage widths, the
same depth pattern and the same conv count (52 vs 53):

    basic6666  torchvision ResNet(BasicBlock, [6, 6, 6, 6])   -- basic blocks
    resnet50   torchvision resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
    resnet18   torchvision ResNet(BasicBlock, [2, 2, 2, 2])   -- A29b (plan 9.18), the depth axis: 20 conv
    resnet34   torchvision ResNet(BasicBlock, [3, 4, 6, 3])   -- A29b: 36 conv (basic6666 has 52)

Recipe (plan 9.4, fixed before any data): SGD momentum 0.9, batch 256, lr 0.1 cosine to 0 over
90 epochs (per epoch), weight decay 1e-4 on all parameters, RandomResizedCrop(224) + flip, eval
Resize(256) + CenterCrop(224), cross-entropy, no label smoothing / mixup / EMA.  AMP + channels_last.

Along the trajectory (--schedule, epoch 0 = initialisation) the layer profile is measured with the
SAME protocol as the public checkpoints of section 6: the 6,400-image validation subset
(--measure-data, flat directory), 16 positions per image, layerspec's probe; rows go to
epoch_XXX_layers.csv and trajectory_layers.csv in the schema of the CIFAR round-3 runs, so
checkpoint_analysis.py computes R-R / T-T / T-R exactly as for plan 9.1 and can correlate the
final ResNet-50 profile with the six public ResNet-50 recipes (same torchvision layer names).

    python scripts/train_imagenet_a17.py --arch resnet50 --data /workspace/imagenet \\
        --measure-data /workspace/imagenet_val_6400 --seed 0 --out ../results/a17/resnet50_s0

Resumable (resume.pt per epoch); re-running the same command continues the same run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_imagenet_r18 import loaders, evaluate          # noqa: E402

ARCHS = ("basic6666", "resnet50", "resnet18", "resnet34")
DEFAULT_SCHEDULE = [0, 1, 2, 5, 10, 20, 30, 45, 60, 75, 90]


def build(arch: str, num_classes: int = 1000):
    import torchvision
    from torchvision.models.resnet import BasicBlock, Bottleneck
    if arch == "basic6666":
        return torchvision.models.ResNet(BasicBlock, [6, 6, 6, 6], num_classes=num_classes)
    if arch == "resnet50":
        return torchvision.models.ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if arch == "resnet18":
        return torchvision.models.ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if arch == "resnet34":
        return torchvision.models.ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arch", choices=ARCHS, required=True)
    p.add_argument("--data", required=True, help="folder with train/ and val/ (scripts/fetch_imagenet.py)")
    p.add_argument("--measure-data", default="../data/imagenet_val_6400",
                   help="flat directory of the 6,400 validation images the public checkpoints were measured on")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--crop", type=int, default=224, help="training / evaluation crop (smoke tests use less)")
    p.add_argument("--measure-size", type=int, default=224, help="image size of the profile measurement")
    p.add_argument("--positions", type=int, default=16, help="sampled spatial positions per image (section 6 protocol)")
    p.add_argument("--measure-limit", type=int, default=None, help="use at most this many measurement images")
    p.add_argument("--schedule", type=int, nargs="+", default=None, help="epochs at which to measure (0 = init)")
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke test")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke test")
    a = p.parse_args(argv)

    from layerspec.api import image_loader
    from scripts.trajectory_cifar import measure
    from scripts.realloc_controller import model_macs, module_hw

    schedule = sorted(set(a.schedule if a.schedule is not None else [e for e in DEFAULT_SCHEDULE if e <= a.epochs]))
    if a.epochs not in schedule:
        schedule.append(a.epochs)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader, num_classes = loaders(a.data, a.batch_size, a.workers, a.seed, a.crop)
    measure_loader = image_loader(a.measure_data, batch_size=min(a.batch_size, 64), workers=min(a.workers, 4),
                                  image_size=a.measure_size, limit=a.measure_limit)
    model = build(a.arch, num_classes).to(a.device).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    n_macs = model_macs(model, module_hw(build(a.arch, num_classes), torch.zeros(1, 3, 224, 224)))
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=0.0)
    use_amp = a.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state_path, csv_path = os.path.join(a.out, "resume.pt"), os.path.join(a.out, "epochs.csv")
    traj_path = os.path.join(a.out, "trajectory_layers.csv")
    start_epoch, history, frames = 1, [], []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); torch.set_rng_state(st["rng_cpu"].cpu())
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        start_epoch, history = st["epoch"] + 1, st.get("history", [])
        if os.path.exists(traj_path):
            frames.append(pd.read_csv(traj_path))
        print(f"resumed from {state_path} at epoch {st['epoch']} (next: {start_epoch})", flush=True)
    else:
        with open(csv_path, "w") as fh:
            fh.write("epoch,lr,train_loss,top1,top5,seconds,img_per_s\n")

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "epoch": epoch, "history": history, "seed": a.seed, "arch": a.arch}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    def checkpoint(epoch, top1):
        t0 = time.time()
        df = measure(model, measure_loader, a.device, a.positions, epoch, top1, a.seed)
        df.to_csv(os.path.join(a.out, f"epoch_{epoch:03d}_layers.csv"), index=False)
        frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(traj_path, index=False)
        conv = df[df.kind == "conv"]
        print(f"  [measure] epoch {epoch:3d}  top1 {top1:5.2f}  {len(conv)} conv layers  "
              f"k*(.95)/C median {conv['k_star_ratio_0.95'].median():.3f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"arch {a.arch}  params {n_params:,}  MACs@224 {n_macs / 1e9:.3f} G  classes {num_classes}  "
          f"train batches/epoch {len(train_loader)}  device {a.device}  crop {a.crop}  schedule {schedule}", flush=True)
    if start_epoch == 1:
        top1, _ = evaluate(model, val_loader, a.device, a.max_val_batches)
        checkpoint(0, top1)
    for ep in range(start_epoch, a.epochs + 1):
        model.train()
        lr = opt.param_groups[0]["lr"]
        t0, seen, loss_sum, steps = time.time(), 0, 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if a.max_train_batches is not None and i >= a.max_train_batches:
                break
            x = x.to(a.device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(a.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item(); steps += 1; seen += y.numel()
            if i % 500 == 0:
                print(f"    ep {ep} it {i}/{len(train_loader)} loss {loss.item():.3f} "
                      f"{seen / (time.time() - t0):.0f} img/s", flush=True)
        sched.step()
        secs = time.time() - t0
        top1, top5 = evaluate(model, val_loader, a.device, a.max_val_batches)
        row = {"epoch": ep, "lr": lr, "train_loss": loss_sum / max(steps, 1), "top1": top1, "top5": top5,
               "seconds": secs, "img_per_s": seen / secs}
        history.append(row)
        with open(csv_path, "a") as fh:
            fh.write(f"{ep},{lr:.5f},{row['train_loss']:.4f},{top1:.3f},{top5:.3f},{secs:.0f},{row['img_per_s']:.0f}\n")
        print(f"  epoch {ep:3d}/{a.epochs}  lr {lr:.4f}  loss {row['train_loss']:.3f}  "
              f"top1 {top1:5.2f}  top5 {top5:5.2f}  ({secs:.0f}s, {row['img_per_s']:.0f} img/s)", flush=True)
        if ep in schedule:
            checkpoint(ep, top1)
        save_state(ep)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "final.pt"))
    best = max(history, key=lambda r: r["top1"]) if history else {}
    result = {"arch": a.arch, "params": n_params, "macs_224": n_macs, "seed": a.seed, "epochs": a.epochs, "crop": a.crop,
              "recipe": {"optimizer": "SGD", "momentum": 0.9, "lr": a.lr, "schedule": "cosine", "eta_min": 0.0,
                         "wd": a.wd, "batch_size": a.batch_size, "aug": f"RandomResizedCrop{a.crop}+flip",
                         "eval": f"Resize{int(round(a.crop * 256 / 224))}+CenterCrop{a.crop}", "amp": use_amp},
              "measure": {"data": a.measure_data, "size": a.measure_size, "positions": a.positions,
                          "schedule": schedule, "limit": a.measure_limit},
              "final_top1": history[-1]["top1"] if history else None,
              "final_top5": history[-1]["top5"] if history else None,
              "best_top1": best.get("top1"), "best_epoch": best.get("epoch"),
              "train_hours": sum(r["seconds"] for r in history) / 3600,
              "img_per_s_median": sorted(r["img_per_s"] for r in history)[len(history) // 2] if history else None}
    with open(os.path.join(a.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    if os.path.exists(state_path):
        os.remove(state_path)
    print("done", json.dumps({k: result[k] for k in ("arch", "final_top1", "final_top5", "train_hours")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
