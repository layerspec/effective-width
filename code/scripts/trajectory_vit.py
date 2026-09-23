"""Transformer arm of the width controller (2026-09-23, author: every quality item that Mac time can buy).

A CIFAR-10 Vision Transformer trained from scratch whose MLP hidden widths (fc1 out = fc2 in, one site per
block) are either fixed (--widths, the full / ruler / uniform arms) or reallocated during training by the same
controller as trajectory_cifar.py (--realloc-budget): measure the post-GELU covariance and the fc1 input
covariance on 6,400 unlabeled training images, water-fill the budget with the unmet-variance eigenvalues
appended, grow along the unmet directions (function-preserving), then shrink by greedy CSSP with the
reconstruction folded into fc2 (Propositions A-C).  The package's `mlp` discoverer finds the sites; nothing
here is transformer-specific except the model, the optimiser and the MACs count.

Model: timm VisionTransformer, image 32, patch 4 (64 tokens + cls), dim 192, depth 6, heads 3, MLP ratio 4
(hidden 768), drop-path 0.1: 2.69M parameters.  Recipe: AdamW lr 1e-3, weight decay 0.05, OneCycle (10%
warm-up, cosine), batch 128, 100 epochs, random crop + flip, label smoothing 0.1, gradient clip 1.0.  Every
Linear sees the same 65 tokens, so the MLP sites' MACs are proportional to their parameters: a parameter
budget on the sites is a FLOPs budget on them (attention and patch-embedding MACs do not depend on the
sites).  MACs are still recorded per action.

    python scripts/trajectory_vit.py --device mps --out ../results/vit_c10/full_s0 --seed 0
    python scripts/trajectory_vit.py --device mps --out ../results/vit_c10/b60_s0 --seed 0 --realloc-budget 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reproduce_garg import cifar_loaders, evaluate, CIFAR_STATS   # noqa: E402
from scripts.trajectory_cifar import measure, DEFAULT_SCHEDULE            # noqa: E402

VIT = dict(img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0, drop_path_rate=0.1)
TOKENS = (VIT["img_size"] // VIT["patch_size"]) ** 2 + 1


def build_vit(num_classes: int = 10, hidden: list[int] | None = None, **over) -> nn.Module:
    """timm VisionTransformer with per-block MLP hidden widths (default: mlp_ratio x dim in every block)."""
    from timm.models.vision_transformer import VisionTransformer
    cfg = dict(VIT, **over)
    m = VisionTransformer(num_classes=num_classes, **cfg)
    if hidden is not None:
        if len(hidden) != cfg["depth"]:
            raise ValueError(f"need {cfg['depth']} hidden widths, got {len(hidden)}")
        for blk, h in zip(m.blocks, hidden):
            d = blk.mlp.fc1.in_features
            if h != blk.mlp.fc1.out_features:
                fc1, fc2 = nn.Linear(d, h), nn.Linear(h, d)
                for fc in (fc1, fc2):
                    nn.init.trunc_normal_(fc.weight, std=0.02)
                    nn.init.zeros_(fc.bias)
                blk.mlp.fc1, blk.mlp.fc2 = fc1, fc2
    return m


def vit_macs(model: nn.Module, tokens: int = TOKENS) -> int:
    """Forward multiply-adds per image: every Linear over `tokens` tokens (the head over one), the patch
    embedding, and the two attention matmuls (q k^T and attn v) per block."""
    total = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            total += m.in_features * m.out_features * (1 if name == "head" else tokens)
        elif isinstance(m, nn.Conv2d):
            total += m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * (tokens - 1)
    for blk in model.blocks:
        total += 2 * tokens * tokens * blk.attn.qkv.in_features
    return int(total)


def site_widths(model: nn.Module) -> list[int]:
    from layerspec.realloc import discover_sites
    return [s.width(model) for s in discover_sites(model)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", default="../data")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--schedule", type=int, nargs="+", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--positions", type=int, default=16, help="tokens per image for the per-epoch measurement")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--widths", type=int, nargs="+", default=None, metavar="H",
                   help=f"MLP hidden width per block ({VIT['depth']} integers); default {int(VIT['mlp_ratio'] * VIT['embed_dim'])}")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-measure-batches", type=int, default=None)
    p.add_argument("--realloc-budget", type=float, default=None,
                   help="controller: after each --realloc-epochs epoch, water-fill this fraction of the FULL model's "
                        "parameters over the MLP sites (grow along unmet directions, shrink by CSSP + fold)")
    p.add_argument("--realloc-epochs", type=int, nargs="+", default=[15, 40])
    p.add_argument("--realloc-calib", type=int, default=6400)
    p.add_argument("--realloc-grow-cap", type=float, default=0.25)
    p.add_argument("--realloc-grow-random", action="store_true")
    p.add_argument("--realloc-min-width", type=int, default=8)
    args = p.parse_args(argv)
    if args.widths is not None and len(args.widths) != VIT["depth"]:
        p.error(f"--widths needs {VIT['depth']} integers, got {len(args.widths)}")

    schedule = sorted(set(args.schedule or [e for e in DEFAULT_SCHEDULE if e <= args.epochs]))
    if args.epochs not in schedule:
        schedule.append(args.epochs)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    num_classes = CIFAR_STATS[args.dataset][2]
    train_loader, test_loader = cifar_loaders(args.data_root, args.batch_size, args.workers, args.dataset)

    state_path = os.path.join(args.out, "resume.pt")
    saved_widths = None
    if os.path.exists(state_path):
        saved_widths = torch.load(state_path, map_location="cpu", weights_only=False).get("widths")
    model = build_vit(num_classes, saved_widths if saved_widths is not None else args.widths).to(args.device)
    _full = build_vit(num_classes)
    full_params = sum(q.numel() for q in _full.parameters())
    full_macs = vit_macs(_full)
    del _full

    def make_opt():
        o = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        s = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
                                                pct_start=0.1)
        return o, s
    steps = len(train_loader) if args.max_train_batches is None else min(len(train_loader), args.max_train_batches)
    opt, sched = make_opt()

    frames, start_epoch, realloc_log = [], 1, []
    rl_path = os.path.join(args.out, "realloc.json")
    if os.path.exists(rl_path):
        realloc_log = json.load(open(rl_path))
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_cpu"].cpu())
        start_epoch = st["epoch"] + 1
        traj = os.path.join(args.out, "trajectory_layers.csv")
        if os.path.exists(traj):
            frames.append(pd.read_csv(traj))
        print(f"resumed from {state_path} at epoch {st['epoch']} (next: {start_epoch})", flush=True)

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng_cpu": torch.get_rng_state(), "epoch": epoch, "arch": "vit_c10", "seed": args.seed,
                    "widths": site_widths(model) if args.realloc_budget is not None else None}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

    def reallocate(epoch):
        """Same two-pass action as trajectory_cifar.py (grow on unmet directions, then iterated CSSP shrink),
        parameter cost, MLP sites only."""
        nonlocal opt, sched
        from layerspec.realloc import discover_sites, greedy_cssp, grow, measure as rmeasure, shrink, unmet_directions, water_fill
        from scripts.cssp_check import loaders as plain_loaders
        t0 = time.time()
        shrink_only = epoch >= max(args.realloc_epochs)
        calib, _ = plain_loaders(args.data_root, args.realloc_calib, 256, 0, args.seed, args.dataset)
        model.eval()
        sites = discover_sites(model)
        cur0 = [s.width(model) for s in sites]
        n0, f0 = sum(q.numel() for q in model.parameters()), vit_macs(model)
        budget_total = args.realloc_budget * full_params

        def fill(mom, allow_grow):
            spectra = [np.clip(np.linalg.eigvalsh(mom[s.name].act_cov.numpy())[::-1], 0, None) for s in sites]
            cost = np.array([s.cost_per_channel(model) for s in sites], dtype=float)
            cur = np.array([s.width(model) for s in sites])
            fixed = sum(q.numel() for q in model.parameters()) - float(cost @ cur)
            hi = cur.copy()
            if allow_grow:
                hi = np.maximum(cur, np.floor(cur * (1 + args.realloc_grow_cap)).astype(int))
                mods = dict(model.named_modules())
                for l, s in enumerate(sites):
                    k = int(hi[l] - cur[l])
                    if k > 0:
                        _, lam = unmet_directions(mods[s.producer], mom[s.name].patch_cov, k)
                        ext = (lam.numpy() / float(torch.trace(mom[s.name].patch_cov))).clip(0, None)
                        spectra[l] = np.concatenate([spectra[l] / spectra[l].sum(), ext])
            target, mu, obj = water_fill(spectra, cost, budget_total - fixed,
                                         min_width=[args.realloc_min_width] * len(sites), max_width=hi.tolist())
            return cur, target, mu, obj

        grown = []
        if not shrink_only:
            mom = rmeasure(model, sites, calib, args.device, patch=True, max_rows_per_batch=2048)
            cur, target, _, _ = fill(mom, True)
            for s, t, c in reversed(list(zip(sites, target, cur))):
                if t > c:
                    grow(model, s, int(t - c), mom[s.name], directions="random" if args.realloc_grow_random else "unmet")
                    grown.append((s.producer, int(c), int(t)))
        for _it in range(6):
            mom = rmeasure(model, sites, calib, args.device, patch=False, max_rows_per_batch=8192)
            cur, target, mu, obj = fill(mom, False)
            if not any(t < c for t, c in zip(target, cur)):
                break
            for s, t, c in zip(sites, target, cur):
                if t < c:
                    order, _ = greedy_cssp(mom[s.name].act_cov, int(t))
                    shrink(model, s, order[:int(t)], mom[s.name])
            if sum(q.numel() for q in model.parameters()) <= budget_total * 1.005:
                break
        model.train()
        sched_state = sched.state_dict()
        opt, sched = make_opt()
        sched.load_state_dict(sched_state)
        after = [s.width(model) for s in sites]
        n1, f1 = sum(q.numel() for q in model.parameters()), vit_macs(model)
        rec = {"epoch": epoch, "shrink_only": shrink_only, "before": cur0, "grown": grown, "after": after,
               "params_before": int(n0), "params_after": int(n1), "budget_fraction": args.realloc_budget,
               "actual_fraction": n1 / full_params, "mu": float(mu), "objective": float(obj), "cost": "params",
               "macs_before": int(f0), "macs_after": int(f1), "macs_fraction": f1 / full_macs,
               "seconds": round(time.time() - t0)}
        realloc_log.append(rec)
        with open(rl_path, "w") as fh:
            json.dump(realloc_log, fh, indent=1)
        print(f"  [realloc] epoch {epoch}: {cur0} -> {after}  params {n0 / 1e6:.2f}M -> {n1 / 1e6:.2f}M "
              f"({n1 / full_params:.3f} of full)  MACs {f0 / 1e6:.0f}M -> {f1 / 1e6:.0f}M ({f1 / full_macs:.3f} of full)  "
              f"{'shrink only' if shrink_only else f'grew {len(grown)} sites then shrank'}  ({rec['seconds']}s)", flush=True)

    def checkpoint(epoch):
        acc = evaluate(model, test_loader, args.device)
        t0 = time.time()
        df = measure(model, test_loader, args.device, args.positions, epoch, acc, args.seed,
                     max_batches=args.max_measure_batches)
        df.to_csv(os.path.join(args.out, f"epoch_{epoch:03d}_layers.csv"), index=False)
        frames.append(df)
        fc1 = df[(df.kind == "linear") & df.layer.str.endswith("mlp.fc1")].sort_values("depth_index")
        r = fc1["k_star_ratio_0.999"].to_numpy() if len(fc1) else np.array([np.nan])
        print(f"  [measure] epoch {epoch:3d}  test acc {acc:5.2f}%  fc1 k*(.999)/H "
              f"{' '.join(f'{v:.2f}' for v in r)}  widths {site_widths(model)}  ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(frames, ignore_index=True).to_csv(os.path.join(args.out, "trajectory_layers.csv"), index=False)

    print(f"schedule: {schedule}  params {sum(q.numel() for q in model.parameters()) / 1e6:.2f}M "
          f"({sum(q.numel() for q in model.parameters()) / full_params:.3f} of full)  widths {site_widths(model)}", flush=True)
    if start_epoch == 1:
        checkpoint(0)
        if args.realloc_budget is not None and 0 in args.realloc_epochs:
            reallocate(0)
    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            if args.max_train_batches is not None and i >= args.max_train_batches:
                break
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        print(f"  epoch {ep:3d}/{args.epochs}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if args.realloc_budget is not None and ep in args.realloc_epochs:
            reallocate(ep)
        if ep in schedule:
            checkpoint(ep)
        save_state(ep)

    torch.save(model.state_dict(), os.path.join(args.out, f"vit_c10_{args.dataset}_final.pt"))
    with open(os.path.join(args.out, "widths.json"), "w") as fh:
        json.dump({"widths": site_widths(model)}, fh)
    if os.path.exists(state_path):
        os.remove(state_path)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
