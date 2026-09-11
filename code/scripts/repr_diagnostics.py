"""Representation-level diagnostics of the A18 arms (analysis-plan 9.9; bar A20).

Five CIFAR-10 VGG-16_BN arms x 3 seeds (full, ruler, ruler_act, uniform,
ruler95), weights in results/round3_pt and results/a18_pt.  Everything runs
on the Mac; images are 6,400 seeded TEST images, one globally pooled vector
per layer per image (Kornblith et al. 2019 setting; NOT the paper's
position sampling).

  cka      9.9.1  per-layer linear CKA of every arm/seed against full seed 0
                  (+ the seed ceiling full s1, s2 vs s0)     -> repr/cka_layers.csv
  probe    9.9.2  linear probes (multinomial logistic regression, L-BFGS,
                  L2 by 5-fold CV) on the last post-ReLU pooled feature,
                  CIFAR-100 / STL-10 (32x32) / SVHN              -> repr/probe.csv
  arrival  9.9.3  per checkpoint of the 9 A18 runs with checkpoints:
                  layer-median rho_align and layer-median CKA to the final
                  network; arrival epochs                       -> repr/arrival.csv
  corrupt  9.9.4  CIFAR-10-C accuracy per corruption x severity  -> repr/cifar10c.csv

    cd code && python scripts/repr_diagnostics.py cka --results ../results --data ../data --device mps
The judgments (P9.19-P9.25) are printed by checkpoint_analysis.py section 23.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.reproduce_garg import CIFAR_STATS                                    # noqa: E402
from scripts.trajectory_cifar import VGG16_WIDTHS, build_vgg16_bn_cifar_widths, DEFAULT_SCHEDULE  # noqa: E402

ARMS = ("full", "ruler", "ruler_act", "uniform", "ruler95")
SEEDS = (0, 1, 2)
RELU_LAYERS = ["features.2", "features.5", "features.9", "features.12", "features.16", "features.19",
               "features.22", "features.26", "features.29", "features.32", "features.36", "features.39",
               "features.42"]
N_IMAGES = 6400


# ----------------------------------------------------------------------------- networks
def arm_widths(results: str, arm: str) -> list[int]:
    if arm == "full":
        return list(VGG16_WIDTHS)
    fn = "widths_act.json" if arm == "ruler_act" else "widths.json"
    key = "ruler" if arm == "ruler_act" else arm
    return json.load(open(os.path.join(results, "a18", fn)))["widths"][key]


def arm_dir(results: str, arm: str, seed: int) -> str:
    if arm == "full":
        return os.path.join(results, "round3_pt", f"a5_vgg16_bn_none_s{seed}")
    return os.path.join(results, "a18_pt", f"vgg16_bn_{arm}_s{seed}")


def load_net(results: str, arm: str, seed: int, device: str, checkpoint: str | None = None) -> nn.Module:
    m = build_vgg16_bn_cifar_widths(arm_widths(results, arm), 10)
    path = checkpoint or os.path.join(arm_dir(results, arm, seed), "vgg16_bn_cifar10_final.pt")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    return m.to(device).eval()


@torch.no_grad()
def pooled_features(model: nn.Module, loader, device: str, layers=RELU_LAYERS) -> dict[str, np.ndarray]:
    """{layer: (N, C) float32} globally average-pooled post-ReLU outputs."""
    feats = {l: [] for l in layers}
    mods = dict(model.named_modules())
    hooks = [mods[l].register_forward_hook(lambda m, i, o, l=l: feats[l].append(o.mean((2, 3)).float().cpu()))
             for l in layers]
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        model(x.to(device))
    for h in hooks:
        h.remove()
    return {l: torch.cat(v).numpy() for l, v in feats.items()}


# ----------------------------------------------------------------------------- data
def cifar_transform():
    from torchvision import transforms
    mean, std, _ = CIFAR_STATS["cifar10"]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def cifar10_test_subset(data: str, n: int = N_IMAGES, seed: int = 0, batch_size: int = 256, workers: int = 0):
    from torchvision import datasets
    ds = datasets.CIFAR10(data, train=False, download=True, transform=cifar_transform())
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch_size,
                                       shuffle=False, num_workers=workers)


def transfer_datasets(data: str):
    """{name: (train_ds, test_ds)} at 32x32 with the CIFAR-10 normalisation the networks were trained with."""
    from torchvision import datasets, transforms
    mean, std, _ = CIFAR_STATS["cifar10"]
    tf32 = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf = cifar_transform()
    return {
        "cifar100": (datasets.CIFAR100(data, train=True, download=True, transform=tf),
                     datasets.CIFAR100(data, train=False, download=True, transform=tf)),
        "stl10": (datasets.STL10(data, split="train", download=True, transform=tf32),
                  datasets.STL10(data, split="test", download=True, transform=tf32)),
        "svhn": (datasets.SVHN(os.path.join(data, "svhn"), split="train", download=True, transform=tf),
                 datasets.SVHN(os.path.join(data, "svhn"), split="test", download=True, transform=tf)),
    }


# ----------------------------------------------------------------------------- CKA
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64); y = y.astype(np.float64)
    x = x - x.mean(0); y = y - y.mean(0)
    xy = np.linalg.norm(x.T @ y) ** 2
    xx = np.linalg.norm(x.T @ x); yy = np.linalg.norm(y.T @ y)
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


def cmd_cka(a) -> int:
    loader = cifar10_test_subset(a.data, workers=a.workers)
    t0 = time.time()
    ref = pooled_features(load_net(a.results, "full", 0, a.device), loader, a.device)
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            if arm == "full" and s == 0:
                continue
            f = pooled_features(load_net(a.results, arm, s, a.device), loader, a.device)
            for i, l in enumerate(RELU_LAYERS):
                rows.append({"arm": arm, "seed": s, "layer_index": i, "layer": l, "C": f[l].shape[1],
                             "cka_vs_full_s0": linear_cka(f[l], ref[l]), "n_images": N_IMAGES})
            print(f"  {arm} s{s}: median CKA {np.median([r['cka_vs_full_s0'] for r in rows[-13:]]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(a.results, "repr", "cka_layers.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------------- probes
def fit_probe(xtr, ytr, xte, yte, seed: int = 0) -> dict:
    """Multinomial logistic regression, L-BFGS, L2 strength by 5-fold CV (scikit-learn)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5, max_iter=2000, n_jobs=-1,
                               random_state=seed).fit(sc.transform(xtr), ytr)
    return {"test_acc": 100.0 * clf.score(sc.transform(xte), yte), "C": float(clf.C_[0]),
            "train_acc": 100.0 * clf.score(sc.transform(xtr), ytr)}


def cmd_probe(a) -> int:
    dsets = transfer_datasets(a.data)
    layer = RELU_LAYERS[-1]
    out = os.path.join(a.results, "repr", "probe.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = list(pd.read_csv(out).to_dict("records")) if os.path.exists(out) else []
    done = {(r["arm"], r["seed"], r["dataset"]) for r in rows}
    t0 = time.time()
    for arm in ARMS:
        for s in SEEDS:
            net = load_net(a.results, arm, s, a.device)
            for name, (tr, te) in dsets.items():
                if (arm, s, name) in done:
                    continue
                ltr = torch.utils.data.DataLoader(tr, batch_size=512, num_workers=a.workers)
                lte = torch.utils.data.DataLoader(te, batch_size=512, num_workers=a.workers)
                xtr = pooled_features(net, ltr, a.device, [layer])[layer]
                xte = pooled_features(net, lte, a.device, [layer])[layer]
                ytr = np.asarray(_labels(tr))
                yte = np.asarray(_labels(te))
                r = fit_probe(xtr, ytr, xte, yte, seed=s)
                r.update({"arm": arm, "seed": s, "dataset": name, "feature": layer, "dim": xtr.shape[1],
                          "n_train": len(ytr), "n_test": len(yte)})
                rows.append(r)
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"  {arm} s{s} {name}: probe {r['test_acc']:.2f}% (C={r['C']}) ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", out)
    return 0


def _labels(ds) -> list[int]:
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            return [int(v) for v in getattr(ds, attr)]
    return [int(ds[i][1]) for i in range(len(ds))]


# ----------------------------------------------------------------------------- arrival
def cmd_arrival(a) -> int:
    loader = cifar10_test_subset(a.data, workers=a.workers)
    runs = [(arm, s) for arm in ("ruler", "uniform", "ruler95") for s in SEEDS]
    rows, t0 = [], time.time()
    for arm, s in runs:
        d = arm_dir(a.results, arm, s)
        epochs = [e for e in DEFAULT_SCHEDULE if os.path.exists(os.path.join(d, f"epoch_{e:03d}.pt"))]
        final = pooled_features(load_net(a.results, arm, s, a.device, os.path.join(d, f"epoch_{epochs[-1]:03d}.pt")),
                                loader, a.device)
        for e in epochs:
            f = pooled_features(load_net(a.results, arm, s, a.device, os.path.join(d, f"epoch_{e:03d}.pt")),
                                loader, a.device)
            ckas = [linear_cka(f[l], final[l]) for l in RELU_LAYERS]
            dec = os.path.join(a.results, "a18", "decompose", f"vgg16_bn_{arm}_s{s}_epoch_{e:03d}_decompose.csv")
            rho = pd.read_csv(dec)["rho_align"].median() if os.path.exists(dec) else float("nan")
            rho_sigma = pd.read_csv(dec)["rho_align_sigma"].median() if os.path.exists(dec) else float("nan")
            rows.append({"arm": arm, "seed": s, "epoch": e, "cka_to_final_median": float(np.median(ckas)),
                         "cka_to_final_min": float(np.min(ckas)), "rho_align_median": rho,
                         "rho_align_sigma_median": rho_sigma})
        print(f"  {arm} s{s}: {len(epochs)} epochs ({time.time() - t0:.0f}s)", flush=True)
    out = os.path.join(a.results, "repr", "arrival.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)
    return 0


# ----------------------------------------------------------------------------- CIFAR-10-C
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur",
               "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate",
               "jpeg_compression", "speckle_noise", "gaussian_blur", "spatter", "saturate"]


@torch.no_grad()
def _acc(model, x_uint8: np.ndarray, y: np.ndarray, device: str, bs: int = 1000) -> float:
    mean, std, _ = CIFAR_STATS["cifar10"]
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1); std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    correct = 0
    for i in range(0, len(y), bs):
        xb = torch.from_numpy(x_uint8[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255)
        pred = model((xb - mean) / std).argmax(1).cpu().numpy()
        correct += int((pred == y[i:i + bs]).sum())
    return 100.0 * correct / len(y)


def cmd_corrupt(a) -> int:
    root = os.path.join(a.data, "CIFAR-10-C")
    labels = np.load(os.path.join(root, "labels.npy"))
    out = os.path.join(a.results, "repr", "cifar10c.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = list(pd.read_csv(out).to_dict("records")) if os.path.exists(out) else []
    done = {(r["arm"], r["seed"], r["corruption"]) for r in rows}
    nets = {(arm, s): load_net(a.results, arm, s, a.device) for arm in ARMS for s in SEEDS}
    t0 = time.time()
    for c in CORRUPTIONS:
        x = np.load(os.path.join(root, c + ".npy"))              # (50000, 32, 32, 3), severities 1..5
        for (arm, s), net in nets.items():
            if (arm, s, c) in done:
                continue
            for sev in range(1, 6):
                sl = slice((sev - 1) * 10000, sev * 10000)
                rows.append({"arm": arm, "seed": s, "corruption": c, "severity": sev,
                             "acc": _acc(net, x[sl], labels[sl], a.device)})
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {c}: done ({time.time() - t0:.0f}s)", flush=True)
    print("wrote", out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("cka", "probe", "arrival", "corrupt"))
    p.add_argument("--results", default="../results")
    p.add_argument("--data", default="../data")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args(argv)
    return {"cka": cmd_cka, "probe": cmd_probe, "arrival": cmd_arrival, "corrupt": cmd_corrupt}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
