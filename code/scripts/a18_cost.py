"""Cost columns for the A18 table (submission bar, 2026-09-10): parameters, multiply-adds per
32x32 image, measured inference throughput, and wall-clock training time per arm.

    python scripts/a18_cost.py --results ../results --device cuda
Writes <results>/<sub>/cost.json for a18 (CIFAR-10) and a18_cifar100.
"""
from __future__ import annotations

import argparse, glob, json, os, re, sys, time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.trajectory_cifar import build_vgg16_bn_cifar_widths, VGG16_WIDTHS  # noqa: E402


def macs_and_params(model: nn.Module, device: str) -> tuple[int, int]:
    macs = [0]
    def conv_hook(m, i, o):
        macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features
    hs = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hs += [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=device))
    for h in hs:
        h.remove()
    return macs[0], sum(p.numel() for p in model.parameters())


def throughput(model: nn.Module, device: str, bs: int = 256, iters: int = 30) -> float:
    model.eval()
    x = torch.randn(bs, 3, 32, 32, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return bs * iters / (time.time() - t0)


def train_hours(run_dir: str) -> float | None:
    """Sum of the per-epoch times printed by trajectory_cifar.py (epoch lines only, not measures)."""
    log = run_dir + ".log"
    if not os.path.exists(log):
        return None
    secs = 0.0
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        m = re.search(r"^\s+epoch\s+\d+/\d+\s+loss\s+[\d.]+\s+\((\d+)s\)", line)
        if m:
            secs += float(m.group(1))
    return secs / 3600 if secs else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    for sub, nc in (("a18", 10), ("a18_cifar100", 100)):
        base = os.path.join(a.results, sub)
        if not os.path.isdir(base):
            continue
        W = json.load(open(os.path.join(base, "widths.json")))["widths"]
        wa = os.path.join(base, "widths_act.json")
        if os.path.exists(wa):
            W["ruler_act"] = json.load(open(wa))["widths"]["ruler"]
        W.setdefault("full", list(VGG16_WIDTHS))
        out = {}
        for arm, widths in W.items():
            model = build_vgg16_bn_cifar_widths(widths, nc).to(a.device)
            macs, params = macs_and_params(model, a.device)
            ips = throughput(model, a.device)
            runs = sorted(glob.glob(os.path.join(base, f"vgg16_bn_{arm}_s?")))
            hrs = [h for h in (train_hours(r) for r in runs) if h is not None]
            out[arm] = {"widths": widths, "params": params, "macs": macs,
                        "images_per_s": ips, "train_hours_per_run": (sum(hrs) / len(hrs)) if hrs else None,
                        "n_runs_timed": len(hrs), "device": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device}
            print(f"{sub:13s} {arm:10s} params {params/1e6:6.2f}M  MACs {macs/1e6:7.1f}M  {ips:8.0f} img/s  "
                  f"train {out[arm]['train_hours_per_run'] or float('nan'):.2f} h ({len(hrs)} runs)")
        with open(os.path.join(base, "cost.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print("wrote", os.path.join(base, "cost.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
