"""A34 (analysis-plan 9.24): two rulers on one network.  The numerical rank of Masarczyk et al. (NeurIPS 2023) -- the
number of eigenvalues of the sample covariance above 1e-3 of the largest -- and our k*(0.999), both read from the
same post-ReLU channel covariance of every convolution of VGG-16_BN, on the same images.

  part a   the end state: the final weights of the CIFAR-10 full network, of the class-count arms of A24
           (K = 13 / 20 / 50 / 100 classes of CIFAR-100; CIFAR-10 at 640 images per class) and of the long arm of
           A25 (640 per class, 780 epochs = 39,000 steps), three seeds each, each on 6,400 of its own training
           images, 16 positions -- the measurement of width_from_ruler.py --tensor act, with the spectrum kept.
               -> <out>/end_state.csv
  part b   the first steps of training, CIFAR-10, seeds 0-2, two recipes: ours (SGD 0.9 Nesterov, weight decay 5e-4,
           OneCycle to 0.1 over 100 epochs, so 0.004 at the start) and theirs (SGD 0.9, constant 0.1, weight decay
           1e-4); batch 128, crop + flip; both rulers at steps 0, 25, 75, 200, 390 and 1950.
               -> <out>/early.csv

Wrapper: layerspec and the training script are imported, not modified.

    python scripts/two_rulers.py --part a --out ../results/two_rulers --device cpu
    python scripts/two_rulers.py --part b --out ../results/two_rulers --device mps
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NR_THRESHOLD = 1e-3          # Masarczyk et al.: singular values of the sample covariance above 1e-3 of the largest
STEPS = (0, 25, 75, 200, 390, 1950)
SEEDS = (0, 1, 2)
ARMS = {   # name: (dataset, loader arguments, checkpoint pattern with {s} for the seed, classes of the task)
    "c10_full":       ("cifar10", {}, "round3_pt/a5_vgg16_bn_none_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c10_pc640":      ("cifar10", {"per_class": 640}, "a24_classes/c10_pc640_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c10_pc640_e780": ("cifar10", {"per_class": 640}, "a25_steps/c10_pc640_e780_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c100_k13":       ("cifar100", {"classes": 13}, "a24_classes/c100_k13_s{s}/vgg16_bn_cifar100_final.pt", 13),
    "c100_k20":       ("cifar100", {"classes": 20}, "a24_classes/c100_k20_s{s}/vgg16_bn_cifar100_final.pt", 20),
    "c100_k50":       ("cifar100", {"classes": 50}, "a24_classes/c100_k50_s{s}/vgg16_bn_cifar100_final.pt", 50),
    "c100_k100":      ("cifar100", {}, "a18_pt/cifar100/vgg16_bn_full_s{s}/vgg16_bn_cifar100_final.pt", 100),
}
RECIPES = {"ours": dict(lr=0.1, wd=5e-4, nesterov=True, onecycle=True),
           "theirs": dict(lr=0.1, wd=1e-4, nesterov=False, onecycle=False)}


def numerical_rank(eigenvalues: np.ndarray, threshold: float = NR_THRESHOLD) -> int:
    """Number of eigenvalues of the covariance above `threshold` times the largest."""
    e = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    if e.size == 0 or e.max() <= 0:
        return 0
    return int((e > threshold * e.max()).sum())


def both_rulers(model, loader, device: str, positions: int = 16, seed: int = 0) -> pd.DataFrame:
    """One row per convolution (post-ReLU tensor, depth order): C, n/C, gate, k*(0.95), k*(0.999), the numerical rank
    at 1e-3 of the largest eigenvalue and, as a sensitivity reading, at 1e-6 (1e-3 on the singular values of the
    representation rather than of the covariance)."""
    import layerspec
    prof = layerspec.profile(model, loader, device=device, positions=positions, taus=(0.95, 0.999),
                             include_activations=True, include_blocks=False, seed=seed)
    t = prof.table[prof.table.kind == "act"].sort_values("depth_index").reset_index(drop=True)
    rows = []
    for i, r in t.iterrows():
        e = prof.spectra[f"{r.layer}::{r.kind}#{int(r.call_index)}"]
        rows.append(dict(index=i, layer=r.layer, C=int(r.C), n_over_C=float(r.n_over_C), ok=bool(r.ok),
                         k95=int(r["k_star_0.95"]), k999=int(r["k_star_0.999"]),
                         nr=numerical_rank(e), nr_1e6=numerical_rank(e, NR_THRESHOLD ** 2),
                         top_share=float(e.max() / e.sum()) if e.sum() > 0 else float("nan")))
    return pd.DataFrame(rows)


def _append(path: str, df: pd.DataFrame) -> None:
    old = pd.read_csv(path) if os.path.exists(path) else None
    out = df if old is None else pd.concat([old, df], ignore_index=True)
    out.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def _done(path: str, **key) -> bool:
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    m = np.ones(len(d), dtype=bool)
    for k, v in key.items():
        m &= (d[k] == v).to_numpy()
    return bool(m.any())


def part_a(a) -> None:
    from scripts.reproduce_garg import CIFAR_STATS, build_vgg16_bn_cifar
    from scripts.width_from_ruler import train_image_loader
    out = os.path.join(a.out, "end_state.csv")
    for arm, (dataset, kw, pattern, n_classes) in ARMS.items():
        for s in SEEDS:
            if _done(out, arm=arm, seed=s):
                print(f"=== {arm} seed {s} done, skipping ===", flush=True); continue
            ckpt = os.path.join(a.results, pattern.format(s=s))
            if not os.path.exists(ckpt):
                print(f"=== {arm} seed {s}: missing {ckpt} ===", flush=True); continue
            t0 = time.time()
            model = build_vgg16_bn_cifar(n_classes, "small")
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
            loader = train_image_loader(a.data, 6400, 64, a.workers, 0, dataset, kw.get("classes"), kw.get("per_class"))
            df = both_rulers(model, loader, a.device)
            if len(df) != 13:
                raise SystemExit(f"{arm} seed {s}: expected 13 convolutions, measured {len(df)}")
            df.insert(0, "seed", s); df.insert(0, "classes", n_classes); df.insert(0, "arm", arm)
            _append(out, df)
            print(f"=== {arm} seed {s}: last three k*(0.999)/C {[round(x, 3) for x in (df.k999 / df.C).tolist()[-3:]]}  "
                  f"({time.time() - t0:.0f} s) ===", flush=True)


def part_b(a) -> None:
    from scripts.reproduce_garg import build_vgg16_bn_cifar
    from scripts.trajectory_cifar import GPUCifarTrain
    from scripts.width_from_ruler import train_image_loader
    out = os.path.join(a.out, "early.csv")
    steps = [s for s in STEPS if s <= a.max_step]
    for recipe, r in RECIPES.items():
        for seed in SEEDS:
            if _done(out, recipe=recipe, seed=seed, step=steps[-1]):
                print(f"=== {recipe} seed {seed} done, skipping ===", flush=True); continue
            t0 = time.time()
            torch.manual_seed(seed)
            model = build_vgg16_bn_cifar(10, "small").to(a.device)
            train = GPUCifarTrain(a.data, 128, a.device, seed)
            opt = torch.optim.SGD(model.parameters(), lr=r["lr"], momentum=0.9, weight_decay=r["wd"], nesterov=r["nesterov"])
            sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=r["lr"], epochs=100, steps_per_epoch=len(train))
                     if r["onecycle"] else None)
            loader = train_image_loader(a.data, 6400, 64, a.workers, 0, "cifar10")
            frames, step, loss_v = [], 0, float("nan")

            def read():
                df = both_rulers(model, loader, a.device)
                df.insert(0, "lr", opt.param_groups[0]["lr"]); df.insert(0, "loss", loss_v)
                df.insert(0, "step", step); df.insert(0, "seed", seed); df.insert(0, "recipe", recipe)
                frames.append(df)
                print(f"    {recipe} seed {seed} step {step:5d}  loss {loss_v:.3f}  last three nr/C "
                      f"{[round(x, 3) for x in (df.nr / df.C).tolist()[-3:]]}  k*(0.999)/C "
                      f"{[round(x, 3) for x in (df.k999 / df.C).tolist()[-3:]]}", flush=True)
            read()
            model.train()
            while step < steps[-1]:
                for x, y in train:
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward(); opt.step()
                    if sched is not None:
                        sched.step()
                    step += 1; loss_v = float(loss)
                    if not np.isfinite(loss_v):
                        raise SystemExit(f"{recipe} seed {seed}: loss is not finite at step {step}")
                    if step in steps:
                        read(); model.train()
                    if step >= steps[-1]:
                        break
            _append(out, pd.concat(frames, ignore_index=True))
            print(f"=== {recipe} seed {seed} end ({time.time() - t0:.0f} s) ===", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--part", choices=("a", "b"), required=True)
    p.add_argument("--results", default="../results")
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../results/two_rulers")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-step", type=int, default=STEPS[-1])
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    (part_a if a.part == "a" else part_b)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
