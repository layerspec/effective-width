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

Controller arm (P2m, 2026-09-24): --realloc-budget F runs the width controller
of notes/controller-criterion-2026-09-16.md on the 8 block-internal widths
(the residual stream and the stem stay): after each --realloc-epochs epoch it
grows on unmet directions then shrinks by CSSP until the network costs F of
the full network's forward MACs at the training resolution (--realloc-cost
flops, the FLOPs-conserving setting of P2g) or parameters.  Each action is
recorded in realloc.json; the optimiser is rebuilt (momentum buffers reset,
LR schedule continues).  Shared action code: scripts/realloc_controller.py.

    python scripts/train_imagenet_r18.py --data /workspace/imagenet --seed 0 \\
        --out ../results/a18_imagenet/resnet18_full_s0 [--widths W1 ... W13]
    python scripts/train_imagenet_r18.py --data /workspace/imagenet --seed 0 \\
        --out ../results/p2m/r18_f60_s0 --realloc-budget 0.6 --realloc-epochs 10 25

Writes per epoch: epochs.csv (loss, top-1, top-5, seconds, img/s); resume.pt
(atomic; re-running the same command continues the same run, rebuilding the
model at the widths it had); at the end final.pt (state_dict) and result.json.
The data folder is the output of scripts/fetch_imagenet.py (train/ and val/
class folders).
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
from scripts.imagenet_blob import imagenet_split                                # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _val_tf(crop: int):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(int(round(crop * 256 / 224))), transforms.CenterCrop(crop),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def loaders(root: str, batch_size: int, workers: int, seed: int, crop: int = 224):
    from torchvision import transforms
    train_tf = transforms.Compose([transforms.RandomResizedCrop(crop), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = imagenet_split(root, "train", train_tf)          # ImageFolder or blob (scripts/imagenet_blob.py)
    va = imagenet_split(root, "val", _val_tf(crop))
    if tr.classes != va.classes:
        raise SystemExit("train/ and val/ class sets differ")
    g = torch.Generator().manual_seed(seed)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
              prefetch_factor=4 if workers > 0 else None)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **kw),
            torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, **kw), len(tr.classes))


def calib_loader(root: str, n_images: int, batch_size: int, workers: int, seed: int, crop: int = 224):
    """`n_images` TRAINING images, seeded random subset, evaluation transform (the controller's statistics)."""
    import numpy as np
    ds = imagenet_split(root, "train", _val_tf(crop))
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n_images, len(ds)), replace=False)
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=batch_size,
                                       shuffle=False, num_workers=workers)


def widths13_from_model(model) -> list[int]:
    """The flat 13-width vector of scripts/resnet_widths.py read back from a (reallocated) ResNet18W."""
    w = [model.conv1.out_channels]
    for s in range(4):
        stage = getattr(model, f"layer{s + 1}")
        w.append(stage[0].conv2.out_channels)
        w.extend(blk.conv1.out_channels for blk in stage)
    return [int(x) for x in w]


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
    p.add_argument("--crop", type=int, default=224, help="training / evaluation crop (smoke tests use less)")
    p.add_argument("--max-train-batches", type=int, default=None, help="smoke test")
    p.add_argument("--max-val-batches", type=int, default=None, help="smoke test")
    p.add_argument("--realloc-budget", type=float, default=None,
                   help="width controller: after each --realloc-epochs epoch, fill this fraction of the FULL "
                        "network's --realloc-cost over the 8 block-internal widths (realloc.json records each action)")
    p.add_argument("--realloc-cost", choices=("params", "flops"), default="flops")
    p.add_argument("--realloc-epochs", type=int, nargs="+", default=[10, 25],
                   help="epochs after which the controller acts (0 = on the untrained network); grow allowed "
                        "before the last one, shrink only at it")
    p.add_argument("--realloc-calib", type=int, default=6400, help="training images (eval transform) for the statistics")
    p.add_argument("--realloc-grow-cap", type=float, default=0.25, help="max relative growth of a site per action")
    p.add_argument("--realloc-min-width", type=int, default=8)
    p.add_argument("--realloc-grow-random", action="store_true", help="grow along random complement directions")
    a = p.parse_args(argv)
    if a.realloc_budget is not None and a.widths is not None:
        p.error("--realloc-budget starts from the full network; do not combine with --widths")

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader, num_classes = loaders(a.data, a.batch_size, a.workers, a.seed, a.crop)
    state_path, csv_path = os.path.join(a.out, "resume.pt"), os.path.join(a.out, "epochs.csv")
    rl_path = os.path.join(a.out, "realloc.json")
    widths = list(a.widths) if a.widths is not None else list(RESNET18_WIDTHS)
    if os.path.exists(state_path):                        # a controller run resumes at the widths it had reached
        saved = torch.load(state_path, map_location="cpu", weights_only=False).get("widths")
        if saved is not None:
            widths = [int(w) for w in saved]
    model = build_resnet18_widths(widths, num_classes, "imagenet").to(a.device).to(memory_format=torch.channels_last)

    from scripts.realloc_controller import model_macs, module_hw
    _full = build_resnet18_widths(list(RESNET18_WIDTHS), num_classes, "imagenet")
    full_params = sum(q.numel() for q in _full.parameters())
    full_macs = model_macs(_full, module_hw(_full, torch.zeros(1, 3, a.crop, a.crop)))
    del _full

    def make_opt(lr):
        o = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=a.wd)
        return o, torch.optim.lr_scheduler.MultiStepLR(o, milestones=a.milestones, gamma=0.1)
    opt, sched = make_opt(a.lr)
    use_amp = a.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, history, realloc_log = 1, [], []
    if os.path.exists(rl_path):
        realloc_log = json.load(open(rl_path))
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); torch.set_rng_state(st["rng_cpu"].cpu())
        if torch.cuda.is_available() and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        start_epoch, history = st["epoch"] + 1, st.get("history", [])
        print(f"resumed from {state_path} at epoch {st['epoch']} (next: {start_epoch}), widths {widths}", flush=True)
    else:
        with open(csv_path, "w") as fh:
            fh.write("epoch,lr,train_loss,top1,top5,seconds,img_per_s\n")

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "epoch": epoch, "history": history, "widths": widths, "seed": a.seed}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    def reallocate(epoch):
        nonlocal model, opt, sched, widths
        from layerspec.realloc import discover_sites
        from scripts.realloc_controller import reallocate as action
        calib = calib_loader(a.data, a.realloc_calib, min(a.batch_size, 128), min(a.workers, 4), a.seed, a.crop)
        rec = action(model, discover_sites(model), calib, a.device, budget_fraction=a.realloc_budget,
                     cost=a.realloc_cost, full_params=full_params, full_macs=full_macs, input_hw=(a.crop, a.crop),
                     grow_cap=a.realloc_grow_cap, min_width=a.realloc_min_width, grow_random=a.realloc_grow_random,
                     shrink_only=epoch >= max(a.realloc_epochs))
        rec = {"epoch": epoch, **rec}
        model = model.to(a.device).to(memory_format=torch.channels_last)
        widths = widths13_from_model(model)
        lr_now, sched_state = opt.param_groups[0]["lr"], sched.state_dict()
        opt, sched = make_opt(lr_now)
        sched.load_state_dict(sched_state)
        realloc_log.append(rec)
        with open(rl_path, "w") as fh:
            json.dump(realloc_log, fh, indent=1)
        print(f"  [realloc] epoch {epoch}: {rec['before']} -> {rec['after']}  params {rec['params_before'] / 1e6:.2f}M -> "
              f"{rec['params_after'] / 1e6:.2f}M ({rec['actual_fraction']:.3f} of full)  MACs@{a.crop} "
              f"{rec['macs_before'] / 1e9:.3f}G -> {rec['macs_after'] / 1e9:.3f}G ({rec['macs_fraction']:.3f} of full)  "
              f"budget {a.realloc_budget} of full {a.realloc_cost}  "
              f"{'shrink only' if rec['shrink_only'] else f'grew {len(rec['grown'])} sites then shrank'}  "
              f"({rec['seconds']}s)  widths {widths}", flush=True)

    n_params = sum(q.numel() for q in model.parameters())
    print(f"widths {widths}  params {n_params:,}  MACs@224 {macs(model, (224, 224)) / 1e9:.3f} G  classes {num_classes}  "
          f"train batches/epoch {len(train_loader)}  device {a.device}  crop {a.crop}", flush=True)
    if a.realloc_budget is not None and start_epoch == 1 and 0 in a.realloc_epochs:
        reallocate(0)                                     # the a-priori start: act on the untrained network
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
        if a.realloc_budget is not None and ep in a.realloc_epochs:
            reallocate(ep)
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
    n_params = sum(q.numel() for q in model.parameters())
    n_macs = macs(model, (224, 224))
    best = max(history, key=lambda r: r["top1"]) if history else {}
    result = {"widths": widths, "params": n_params, "macs_224": n_macs, "seed": a.seed, "epochs": a.epochs,
              "crop": a.crop, "full_params": full_params, "params_fraction": n_params / full_params,
              "recipe": {"optimizer": "SGD", "momentum": 0.9, "lr": a.lr, "wd": a.wd, "batch_size": a.batch_size,
                         "milestones": a.milestones, "gamma": 0.1, "aug": f"RandomResizedCrop{a.crop}+flip",
                         "eval": f"Resize{int(round(a.crop * 256 / 224))}+CenterCrop{a.crop}", "amp": use_amp},
              "realloc": None if a.realloc_budget is None else
              {"budget": a.realloc_budget, "cost": a.realloc_cost, "epochs": a.realloc_epochs,
               "calib": a.realloc_calib, "grow_cap": a.realloc_grow_cap, "min_width": a.realloc_min_width,
               "grow_random": a.realloc_grow_random, "actions": len(realloc_log),
               "macs_fraction_at_crop": realloc_log[-1]["macs_fraction"] if realloc_log else None},
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
