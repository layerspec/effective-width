"""A35 (analysis-plan 9.25): does the premise of the deep-neural-collapse rank bound hold in a batch-normalised
network?  For every convolution of the 21 CIFAR checkpoints (VGG-16_BN; CIFAR-10 full, the class-count arms of A24,
the long arm of A25), on 6,400 of the task's own training images:

  cos     cosine between the gradient of the unregularised loss L0 and the weight, BN in training mode (batch
          statistics; Proposition B6 predicts 0) and in eval mode (running statistics)
  r       stationarity residual ||lambda W + grad L0|| / ||lambda W||, lambda = 5e-4 (the recipe's weight decay)
  tail_K  energy of the weight (C_out x d) beyond rank K, K = the task's class count
  wcv     within-class variance fraction of the post-ReLU, globally pooled features (K clusters = the labels)

Wrapper: layerspec and the training scripts are imported, not modified.  Resumable: a finished checkpoint is skipped.

    python scripts/measure_stationarity.py --out ../results/stationarity --device cpu
    python scripts/measure_stationarity.py --judge --out ../results/stationarity
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LAMBDA = 5e-4
N_IMAGES, BATCH = 6400, 128
SEEDS = (0, 1, 2)
ARMS = {   # name: (dataset, loader arguments, checkpoint pattern, class count)  -- the arms of A34 (plan 9.24)
    "c10_full":       ("cifar10", {}, "round3_pt/a5_vgg16_bn_none_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c10_pc640":      ("cifar10", {"per_class": 640}, "a24_classes/c10_pc640_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c10_pc640_e780": ("cifar10", {"per_class": 640}, "a25_steps/c10_pc640_e780_s{s}/vgg16_bn_cifar10_final.pt", 10),
    "c100_k13":       ("cifar100", {"classes": 13}, "a24_classes/c100_k13_s{s}/vgg16_bn_cifar100_final.pt", 13),
    "c100_k20":       ("cifar100", {"classes": 20}, "a24_classes/c100_k20_s{s}/vgg16_bn_cifar100_final.pt", 20),
    "c100_k50":       ("cifar100", {"classes": 50}, "a24_classes/c100_k50_s{s}/vgg16_bn_cifar100_final.pt", 50),
    "c100_k100":      ("cifar100", {}, "a18_pt/cifar100/vgg16_bn_full_s{s}/vgg16_bn_cifar100_final.pt", 100),
}


def labelled_loader(root: str, dataset: str, classes, per_class, n_images: int = N_IMAGES, batch: int = BATCH, seed: int = 0):
    """The task's training images with labels renumbered 0..K-1 (plan 9.12 subsets), plain transform, fixed order."""
    from torchvision import datasets, transforms
    from scripts.reproduce_garg import CIFAR_STATS
    from scripts.class_subset import classes_for, label_map, subset_indices
    mean, std, num_classes = CIFAR_STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = (datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10)(root, train=True, download=False, transform=tf)
    cls = classes_for(num_classes, classes)
    lmap = label_map(cls)
    pool = subset_indices(ds.targets, cls, per_class, seed=0)
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed))[:n_images]
    idx = [int(pool[i]) for i in order.tolist()]

    class _Sub(torch.utils.data.Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            x, y = ds[idx[i]]
            return x, lmap[int(y)]
    return torch.utils.data.DataLoader(_Sub(), batch_size=batch, shuffle=False, num_workers=0), len(cls)


def conv_layers(model) -> list[tuple[str, nn.Conv2d]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]


def mean_gradient(model, loader, device: str, train_mode: bool) -> dict[str, torch.Tensor]:
    """Gradient of the mean cross-entropy over the loader with respect to every convolution weight."""
    model.train(train_mode)
    grads = {n: torch.zeros_like(m.weight, dtype=torch.float64) for n, m in conv_layers(model)}
    n_batches = 0
    for x, y in loader:
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        for n, m in conv_layers(model):
            grads[n] += m.weight.grad.detach().double().cpu()
        n_batches += 1
    model.zero_grad(set_to_none=True)
    return {n: g / n_batches for n, g in grads.items()}


def gradient_rows(model, grads: dict[str, torch.Tensor], lam: float = LAMBDA) -> dict[str, tuple[float, float]]:
    """(cos, r) per convolution: cosine of grad and weight, and ||lam W + grad|| / ||lam W||."""
    out = {}
    for n, m in conv_layers(model):
        w = m.weight.detach().double().cpu().flatten(); g = grads[n].flatten()
        cos = float((g @ w) / (g.norm() * w.norm() + 1e-30))
        r = float((lam * w + g).norm() / (lam * w.norm() + 1e-30))
        out[n] = (cos, r)
    return out


def tail_energy(weight: torch.Tensor, k: int) -> float:
    """Energy of the C_out x d matrix beyond rank k, as a fraction of the total."""
    s = torch.linalg.svdvals(weight.detach().double().reshape(weight.shape[0], -1).cpu())
    e = s ** 2
    return float(e[k:].sum() / e.sum()) if k < len(e) else 0.0


@torch.no_grad()
def pooled_features(model, loader, device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Globally pooled post-ReLU features after every convolution (eval mode), and the labels."""
    model.eval()
    convs = conv_layers(model)
    relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    assert len(relus) >= len(convs), "expected a ReLU after every convolution"
    feats = {i: [] for i in range(len(convs))}
    handles = []
    for i, relu in enumerate(relus[:len(convs)]):
        handles.append(relu.register_forward_hook(lambda mod, inp, out, i=i: feats[i].append(out.mean(dim=(2, 3)).double().cpu())))
    ys = []
    for x, y in loader:
        model(x.to(device)); ys.append(y)
    for h in handles:
        h.remove()
    return {convs[i][0]: torch.cat(v) for i, v in feats.items()}, torch.cat(ys)


def within_class_fraction(feat: torch.Tensor, y: torch.Tensor) -> float:
    """Sum over channels of within-class variance divided by total variance (K clusters = the labels)."""
    total = feat.var(dim=0, unbiased=False).sum()
    within = torch.zeros((), dtype=torch.float64)
    for c in y.unique():
        f = feat[y == c]
        within += f.var(dim=0, unbiased=False).sum() * len(f) / len(feat)
    return float(within / (total + 1e-30))


def measure_checkpoint(ckpt: str, dataset: str, kw: dict, n_classes: int, data: str, device: str) -> pd.DataFrame:
    from scripts.reproduce_garg import build_vgg16_bn_cifar
    model = build_vgg16_bn_cifar(n_classes, "small")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
    model.to(device)
    loader, K = labelled_loader(data, dataset, kw.get("classes"), kw.get("per_class"))
    g_train = mean_gradient(model, loader, device, train_mode=True)
    rows_train = gradient_rows(model, g_train)
    g_eval = mean_gradient(model, loader, device, train_mode=False)
    rows_eval = gradient_rows(model, g_eval)
    feats, y = pooled_features(model, loader, device)
    rows = []
    for i, (n, m) in enumerate(conv_layers(model)):
        rows.append(dict(index=i, layer=n, C=int(m.out_channels), d=int(m.weight[0].numel()), K=K,
                         cos_train=rows_train[n][0], r_train=rows_train[n][1], cos_eval=rows_eval[n][0], r_eval=rows_eval[n][1],
                         tail_K=tail_energy(m.weight, K), wcv=within_class_fraction(feats[n], y)))
    return pd.DataFrame(rows)


def _done(path: str, arm: str, seed: int) -> bool:
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    return bool(((d.arm == arm) & (d.seed == seed)).any())


def run(a) -> None:
    out = os.path.join(a.out, "stationarity.csv")
    for arm, (dataset, kw, pattern, n_classes) in ARMS.items():
        for s in SEEDS:
            if _done(out, arm, s):
                print(f"=== {arm} seed {s} done, skipping ===", flush=True); continue
            ckpt = os.path.join(a.results, pattern.format(s=s))
            if not os.path.exists(ckpt):
                print(f"=== {arm} seed {s}: missing {ckpt} ===", flush=True); continue
            t0 = time.time()
            df = measure_checkpoint(ckpt, dataset, kw, n_classes, a.data, a.device)
            df.insert(0, "seed", s); df.insert(0, "arm", arm)
            old = pd.read_csv(out) if os.path.exists(out) else None
            (df if old is None else pd.concat([old, df], ignore_index=True)).to_csv(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            late = df.iloc[-3:]
            print(f"=== {arm} seed {s}: |cos_train| max {df.cos_train.abs().max():.3f}, r_train min {df.r_train.min():.2f}, "
                  f"late tail_K {[round(x, 2) for x in late.tail_K]}, late wcv {[round(x, 2) for x in late.wcv]} ({time.time() - t0:.0f} s) ===", flush=True)


def judge(a) -> dict:
    """P9.76-P9.79 as registered in plan 9.25."""
    path = os.path.join(a.out, "stationarity.csv")
    V = {}
    if not os.path.exists(path):
        print("  stationarity.csv not found"); return V
    d = pd.read_csv(path)
    from scipy.stats import spearmanr
    print("==== A35: the premise of the rank-K bound in a batch-normalised network (plan 9.25) ====")
    for arm, g in d.groupby("arm", sort=False):
        m = g.groupby("index").median(numeric_only=True)
        print(f"  {arm:15s} seeds {g.seed.nunique()}  |cos| train max {g.cos_train.abs().max():.3f} (eval {g.cos_eval.abs().max():.3f})  "
              f"r train min {g.r_train.min():.2f}  tail_K late {' '.join(f'{x:.2f}' for x in m.tail_K.iloc[-3:])}  "
              f"wcv {' '.join(f'{x:.2f}' for x in m.wcv)}")
    complete = all(d[d.arm == arm].seed.nunique() >= 3 for arm in ARMS)
    if complete:
        V["P9.76"] = bool((d.cos_train.abs() <= 0.05).all() and (d.r_train >= 0.9).all())
        print(f"  P9.76 (training-mode BN: |cos| <= 0.05 and r >= 0.9 in every layer of every checkpoint) -> {'holds' if V['P9.76'] else 'fails'}  "
              f"|cos| max {d.cos_train.abs().max():.3f}, r min {d.r_train.min():.2f}; eval-mode |cos| median {d.cos_eval.abs().median():.3f}")
        full = d[d.arm == "c10_full"]; e780 = d[d.arm == "c10_pc640_e780"]
        late_full = full[full["index"] >= full["index"].max() - 2]; late_e = e780[e780["index"] >= e780["index"].max() - 2]
        V["P9.77"] = bool((late_full.tail_K >= 0.3).all() and (late_e.tail_K >= 0.2).all())
        print(f"  P9.77 (late weights far from rank K: tail >= 0.3 CIFAR-10 full, >= 0.2 at 39,000 steps) -> {'holds' if V['P9.77'] else 'fails'}  "
              f"full {late_full.tail_K.min():.2f}..{late_full.tail_K.max():.2f}, e780 {late_e.tail_K.min():.2f}..{late_e.tail_K.max():.2f}")
        rho = [spearmanr(g["index"], g.wcv).correlation for _, g in d.groupby(["arm", "seed"])]
        last_full = full[full["index"] == full["index"].max()].wcv
        last_e = e780[e780["index"] == e780["index"].max()].wcv.median()
        last_p = d[(d.arm == "c10_pc640") & (d["index"] == d["index"].max())].wcv.median()
        V["P9.78"] = bool(max(rho) <= -0.8 and (last_full >= 0.2).all() and last_e < last_p)
        print(f"  P9.78 (wcv falls with depth, Spearman <= -0.8 each; last conv >= 0.2 in CIFAR-10 full; 39,000 steps below 5,000) -> "
              f"{'holds' if V['P9.78'] else 'fails'}  rho max {max(rho):+.2f}; last wcv full {last_full.min():.2f}..{last_full.max():.2f}, e780 {last_e:.2f} vs pc640 {last_p:.2f}")
        es = os.path.join(a.results, "two_rulers", "end_state.csv")
        if os.path.exists(es):
            e = pd.read_csv(es); e = e[e["index"] == e["index"].max()]
            last = d[d["index"] == d["index"].max()].merge(e[["arm", "seed", "k999", "C"]].rename(columns={"C": "C_es"}), on=["arm", "seed"])
            last["k999_C"] = last.k999 / last.C_es
            r1 = spearmanr(last.k999_C, last.wcv).correlation; rk = spearmanr(last.k999_C, last.K).correlation; rkw = spearmanr(last.wcv, last.K).correlation
            partial = (r1 - rk * rkw) / np.sqrt(max(1 - rk ** 2, 1e-12) * max(1 - rkw ** 2, 1e-12))
            print(f"  P9.79 (descriptive, last conv, {len(last)} checkpoints): Spearman(k999/C, wcv) {r1:+.2f}; (k999/C, K) {rk:+.2f}; partial (k999/C, wcv | K) {partial:+.2f}")
    else:
        print("  not all arms have three seeds -- not judged yet")
    return V


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="../results")
    p.add_argument("--data", default="../data")
    p.add_argument("--out", default="../results/stationarity")
    p.add_argument("--device", default="cpu")
    p.add_argument("--judge", action="store_true")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    judge(a) if a.judge else run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
