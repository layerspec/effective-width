"""T3 counterfactuals from one checkpoint and its patch covariances (plan 9.56, P9.128; proofs T3.2).

For every dense convolution of a trained network (the none arm), with W (C x d) and the test-split patch
covariance Sigma_p accumulated exactly as decompose_check.py does, report k*(tau)/r_max of
    out        W Sigma_p W^T                                  measured (recomputed)
    ortho      W_o Sigma_p W_o^T                              isometric kernel (D5's prediction for ONI)
    nd_pred    W D_B Sigma_p D_B W^T                          Network-Deconvolution counterfactual, D_B the
                                                              block-64 (Sigma_bb + eps I)^{-1/2}, eps = 1e-5
    full_pred  W D Sigma_p D W^T                              full whitening, D = (Sigma_p + lambda_max/gain_cap^2 I)^{-1/2}
    both_pred  W_o D Sigma_p D W_o^T                          whitening + isometric kernel (T3.1's maximiser)
    ceiling    ceil(tau C)/r_max                              Lemma T3.0
plus data_0.999 (near-singular Sigma_p flag used by the registration) and the cross-block energy share
1 - ||blockdiag(Sigma_p)||_F^2 / ||Sigma_p||_F^2 (what D_B leaves behind).

    python scripts/t3_counterfactual.py --cifar-arch vgg16_bn --checkpoint ../results/round3/a5_vgg16_bn_none_s0/vgg16_bn_cifar10_final.pt \
        --data ../data --device mps --out ../results/t3/counterfactual_vgg16_bn_none_s0.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layerspec.decomposition import PatchProbe, kstar  # noqa: E402

TAUS = (0.95, 0.999)


def _eigs(M: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]


def isqrt_sym(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    w = np.clip(w, 1e-300, None)
    return (V / np.sqrt(w)) @ V.T


def block_deconv(Sigma: np.ndarray, C_in: int, k2: int, block: int = 64, eps: float = 1e-5) -> np.ndarray:
    """Block-diagonal D_B of Ye et al. 2020: channels in blocks of `block` (features ordered (C_in, kh, kw))."""
    b = min(block, C_in)
    if C_in % b:
        raise ValueError(f"C_in {C_in} not divisible by block {b}")
    f = b * k2
    D = np.zeros_like(Sigma)
    for i in range(C_in // b):
        sl = slice(i * f, (i + 1) * f)
        D[sl, sl] = isqrt_sym(Sigma[sl, sl] + eps * np.eye(f))
    return D


def cross_block_share(Sigma: np.ndarray, C_in: int, k2: int, block: int = 64) -> float:
    b = min(block, C_in)
    f = b * k2
    diag = sum(np.linalg.norm(Sigma[i * f:(i + 1) * f, i * f:(i + 1) * f]) ** 2 for i in range(C_in // b))
    tot = np.linalg.norm(Sigma) ** 2
    return float(max(0.0, 1 - diag / tot)) if tot > 0 else 0.0


def counterfactual_row(name: str, W: np.ndarray, Sigma: np.ndarray, C_in: int, k2: int, depth_index: int,
                       block: int = 64, eps: float = 1e-5, gain_cap: float = 8.0, taus=TAUS) -> dict:
    C, d = W.shape
    r_max = min(C, d)
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    Wo = U @ Vt
    D_B = block_deconv(Sigma, C_in, k2, block, eps)
    lam_data = _eigs(Sigma)
    D = isqrt_sym(Sigma + lam_data[0] / gain_cap ** 2 * np.eye(d))
    lam = dict(out=_eigs(W @ Sigma @ W.T), ortho=_eigs(Wo @ Sigma @ Wo.T),
               nd_pred=_eigs(W @ D_B @ Sigma @ D_B @ W.T), full_pred=_eigs(W @ D @ Sigma @ D @ W.T),
               both_pred=_eigs(Wo @ D @ Sigma @ D @ Wo.T), kernel=_eigs(W @ W.T))
    row = dict(layer=name, depth_index=depth_index, C_in=C_in, C_out=C, d=d, r_max=r_max,
               cross_block_share=cross_block_share(Sigma, C_in, k2, block))
    for tau in taus:
        tk = str(tau)[2:]
        for key, e in lam.items():
            row[f"{key}_{tk}"] = kstar(e, tau) / r_max
        row[f"data_{tk}"] = min(kstar(lam_data, tau), r_max) / r_max
        row[f"ceiling_{tk}"] = min(math.ceil(tau * C), r_max) / r_max
    return row


def counterfactual_table(probe: PatchProbe, block: int = 64, eps: float = 1e-5, gain_cap: float = 8.0) -> pd.DataFrame:
    rows = []
    for name, acc in probe.acc.items():
        meta = probe.meta[name]
        m = meta["module"]
        W = m.weight.detach().cpu().double().reshape(meta["C_out"], meta["d"]).numpy()
        rows.append(counterfactual_row(name, W, acc.covariance, meta["C_in"], meta["k"], meta["depth_index"], block, eps, gain_cap))
    return pd.DataFrame(rows).sort_values("depth_index").reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cifar-arch", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="../data")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--gain-cap", type=float, default=8.0)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from scripts.trajectory_cifar import ARCHS
    from scripts.reproduce_garg import cifar_loaders
    torch.manual_seed(a.seed)
    model = ARCHS[a.cifar_arch]()
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model = model.to(a.device).eval()
    _, loader = cifar_loaders(a.data, a.batch_size, 2, "cifar10")
    probe = PatchProbe(model, a.positions, a.seed)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if a.max_batches is not None and i >= a.max_batches:
                break
            model(x.to(a.device))
    probe.remove()
    df = counterfactual_table(probe, a.block, a.eps, a.gain_cap)
    df.insert(0, "checkpoint", os.path.basename(a.checkpoint))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df[["layer", "C_out", "d", "out_95", "ortho_95", "nd_pred_95", "full_pred_95", "both_pred_95", "ceiling_95", "data_999", "cross_block_share"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
