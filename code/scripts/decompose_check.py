"""Where does a layer's effective width come from: the kernel or the data?

At the tensor the paper measures (conv output, before the nonlinearity and
the shortcut) the channel covariance is exactly

    Sigma_out = W Sigma_patch W^T,

W the (C_out x d) kernel matrix, d = C_in*k_h*k_w, Sigma_patch the d x d
covariance of the receptive-field patch (im2col).  This script accumulates
Sigma_patch per dense conv layer on the 6,400-image subset, takes W from the
checkpoint, and reports for each layer, at each tau:

    out      k*(W Sigma W^T) / r_max          the measured quantity (recomputed)
    kernel   k*(W W^T) / r_max                what the layer would show under white patches
    data     k*(Sigma_patch) / r_max          the patch covariance itself, capped by r_max
    ortho    k*(W_o Sigma W_o^T) / r_max      W_o = polar factor of W (nearest isometry):
                                              what orthogonalising THIS kernel would give
                                              with THIS data, no training
plus the kernel's own spectrum (effective rank of W W^T over r_max, and the
condition number of its top-r_max singular values).  Referee-driven addition
of 2026-09-07; see notes/orthogonality-novelty-2026-09-07.md section 3.

    python scripts/decompose_check.py --model resnet50 --data ../data/imagenet_val_6400 \
        --device mps --out ../results/decompose
    # a CIFAR-10 network from trajectory_cifar.py (round 3, A5):
    python scripts/decompose_check.py --cifar-arch vgg16_bn --checkpoint ../results/round3/a5_vgg16_bn_so_s0/vgg16_bn_so_cifar10_final.pt \
        --data ../data --device mps --out ../results/round3_decompose
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.data import build_loader                  # noqa: E402
from layerspec.models import build                       # noqa: E402
from layerspec.decomposition import PatchProbe, decompose_rows, check_rows, format_check   # noqa: E402

TAUS = (0.9, 0.95, 0.99, 0.999)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="resnet50")
    p.add_argument("--data", default="../data/imagenet_val_6400")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--positions", type=int, default=48)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="../results/decompose")
    p.add_argument("--cifar-arch", default=None,
                   help="a CIFAR-10 architecture from trajectory_cifar.ARCHS; with --checkpoint, "
                        "decompose a network trained by trajectory_cifar.py on the CIFAR-10 test split")
    p.add_argument("--checkpoint", default=None, help="state_dict .pt for --cifar-arch (omit = random init)")
    p.add_argument("--name", default=None, help="output name (default: --model or the checkpoint stem)")
    p.add_argument("--rank-tol", type=float, default=1e-6, help="numerical rank: singular values above this fraction of the largest (ablation A16)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    if a.cifar_arch:
        from scripts.trajectory_cifar import ARCHS
        from scripts.reproduce_garg import cifar_loaders
        torch.manual_seed(a.seed)
        model = ARCHS[a.cifar_arch]()
        if a.checkpoint:
            model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
        tag = a.checkpoint or f"random_init(seed={a.seed})"
        _, loader = cifar_loaders(a.data, a.batch_size, a.workers, "cifar10")
        a.model = a.name or (os.path.splitext(os.path.basename(a.checkpoint))[0] if a.checkpoint
                             else f"{a.cifar_arch}_random{a.seed}")
    else:
        loader, src = build_loader(a.data, a.batch_size, a.workers, limit=a.limit, seed=a.seed)
        model, tag = build(a.model, pretrained=True, seed=a.seed)
        a.model = a.name or a.model
    model = model.to(a.device).eval()
    probe = PatchProbe(model, a.positions, a.seed)
    t0 = time.time()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x.to(a.device))
    probe.remove()
    print(f"{a.model} ({tag}): {len(probe.acc)} dense conv layers, {time.time() - t0:.0f}s", flush=True)

    rows = decompose_rows(probe, TAUS, a.rank_tol)
    for r in rows:
        r["model"] = a.model
    df = pd.DataFrame(rows).sort_values("depth_index")
    df = df[["model"] + [c for c in df.columns if c != "model"]]
    path = os.path.join(a.out, f"{a.model.replace(':', '_')}_decompose.csv")
    df.to_csv(path, index=False)
    print(df[["layer", "d", "C_out", "n_over_d", "out_0.95", "kernel_0.95", "data_0.95", "ortho_0.95",
              "kernel_erank_over_rmax"]].to_string(index=False), flush=True)
    print(format_check(check_rows(df, TAUS)), flush=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
