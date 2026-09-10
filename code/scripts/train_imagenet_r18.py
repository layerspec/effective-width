"""Train a (width-parameterised) ResNet-18 on ImageNet-1k from scratch with the
torchvision reference recipe (analysis-plan 9.11):

    SGD momentum 0.9, lr 0.1, batch 256, 90 epochs, lr x0.1 at 30 and 60,
    weight decay 1e-4 on all parameters, RandomResizedCrop(224) + horizontal
    flip; eval Resize(256) + CenterCrop(224).  No label smoothing, no mixup,
    no EMA, no distillation.  Mixed precision (fp16 autocast + GradScaler)
    and channels_last for speed; they do not change the recipe.

Arms (plan 9.11):  (a) full width  (e) --widths from width_from_ruler.py
--arch resnet18 --tensor act  (c) --widths uniform.  The 13 widths follow
scripts/resnet_widths.py.

    python scripts/train_imagenet_r18.py --data /workspace/imagenet --seed 0 \\
        --out ../results/a18_imagenet/resnet18_full_s0 [--widths W1 ... W13]

Writes per epoch: epochs.csv (loss, top-1, top-5, seconds, img/s); resume.pt
(atomic; re-running the same command continues the same run); at the end
final.pt (state_dict) and result.json.  The data folder is the output of
scripts/fetch_imagenet.py (train/ and val/ class folders).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.resnet_widths import RESNET18_WIDTHS, build_resnet18_widths, macs  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def loaders(root: str, batch_size: int, workers: int, seed: int):
    from torchvision import datasets, transforms
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(root, "val"), transform=val_tf)
    if tr.classes != va.classes:
        raise SystemExit("train/ and val/ class folders differ")
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
              prefetch_factor=4 if workers > 0 else None)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **kw),
            torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, **kw), len(tr.classes))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    c1 = c5 = n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.startswith("cuda")):
            out = model(x)
            top5 = out.topk(min(5, out.shape[1]), dim=1).indices
        hit = top5 == y[:, None]
        c1 += hit[:, 0].sum().item(); c5 += hit.any(1).sum().item(); n += y.numel()
    return 100.0 * c1 / max(n, 1), 100.0 * c5 / max(n, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="folder with train/ and val/ (scripts/fetch_imagenet.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--widths", type=int, nargs=13, default=None, metavar="W")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[30, 60])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke test")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke test")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader, num_classes = loaders(a.data, a.batch_size, a.workers, a.seed)
    widths = list(a.widths) if a.widths is not None else list(RESNET18_WIDTHS)
    model = build_resnet18_widths(widths, num_classes, "imagenet").to(a.device).to(memory_format=torch.channels_last)
    n_params = sum(q.numel() for q in model.parameters())
    n_macs = macs(model, (224, 224))
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=a.milestones, gamma=0.1)
    use_amp = a.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    state_path, csv_path = os.path.join(a.out, "resume.pt"), os.path.join(a.out, "epochs.csv")
    start_epoch, history = 1, []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); torch.set_rng_state(st["rng_cpu"].cpu())
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        start_epoch, history = st["epoch"] + 1, st.get("history", [])
        print(f"resumed from {state_path} at epoch {st['epoch']} (next: {start_epoch})", flush=True)
    else:
        with open(csv_path, "w") as fh:
            fh.write("epoch,lr,train_loss,top1,top5,seconds,img_per_s\n")

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "epoch": epoch, "history": history, "widths": widths, "seed": a.seed}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    print(f"widths {widths}  params {n_params:,}  MACs@224 {n_macs / 1e9:.3f} G  classes {num_classes}  "
          f"train batches/epoch {len(train_loader)}  device {a.device}", flush=True)
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
        save_state(ep)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "final.pt"))
    best = max(history, key=lambda r: r["top1"]) if history else {}
    result = {"widths": widths, "params": n_params, "macs_224": n_macs, "seed": a.seed, "epochs": a.epochs,
              "recipe": {"optimizer": "SGD", "momentum": 0.9, "lr": a.lr, "wd": a.wd, "batch_size": a.batch_size,
                         "milestones": a.milestones, "gamma": 0.1, "aug": "RandomResizedCrop224+flip",
                         "eval": "Resize256+CenterCrop224", "amp": use_amp},
              "final_top1": history[-1]["top1"] if history else None,
              "final_top5": history[-1]["top5"] if history else None,
              "best_top1": best.get("top1"), "best_epoch": best.get("epoch"),
              "train_hours": sum(r["seconds"] for r in history) / 3600,
              "img_per_s_median": sorted(r["img_per_s"] for r in history)[len(history) // 2] if history else None}
    with open(os.path.join(a.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    if os.path.exists(state_path):
        os.remove(state_path)
    print("done", json.dumps({k: result[k] for k in ("final_top1", "final_top5", "params", "train_hours")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
