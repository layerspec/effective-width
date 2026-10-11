#!/usr/bin/env python
"""Plan 9.57 (P9.132-P9.134; proofs Lemma M.2.0 / Theorem M.2): the width at initialisation as a random variable of the
layer's own weights, measured directly on one network -- the patch route for the activation step.

For every dense convolution of a random-initialised network (default: timm vgg16_bn random, seed 1 -- the same weights as
results/freeprob and results/apriori) the layer's input patch matrix X (d x n, n sampled positions) is extracted once, with
all earlier layers fixed.  Then K fresh i.i.d. Gaussian weight matrices G are drawn and for each the convolution step
Z = s G X and the activation step A = relu(Z) are centred over the n samples; their spectra give k*(tau), the tails
R_k = sum_{i>k} lambda_i, T^2 = tr Sigma, and the exact Poincare quantity of Lemma M.2.0(iii),

    G2_k = ||grad_G R_k||_F^2 = 4 s^2 tr(B_k M_p B_k^T),   B_k = F_perp,k o sigma'(Z),   M_p = X X^T / n  (uncentred),

which bounds Var(R_k) <= c_P E G2_k (Theorem M.2(2), first inequality).  The Chebyshev interval of Theorem M.2(3)
(u = 4.5) is estimated from the first half of the draws and its coverage is checked on the second half.  The network's
actual seed-1 weights give one more row ("actual").

    python scripts/m2_activation_seeds.py --device mps --limit 1600 --positions 32 --draws 40
    python scripts/m2_activation_seeds.py --smoke            # 2 layers, 20 images, 6 draws
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layerspec.decomposition import kstar  # noqa: E402
from scripts.th5_interval_stored import chebyshev_interval  # noqa: E402

TAUS = (0.95, 0.999)
U = 4.5
C_P = 1.0
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA = os.path.join(ROOT, "data", "imagenet_val_6400")
OUT = os.path.join(ROOT, "results", "m2_activation")
MODEL = "timm:vgg16_bn.tv_in1k_random"


# ----------------------------------------------------------------------------------------------------------------- patches
def sample_positions(N: int, H: int, W: int, P: int, gen: torch.Generator) -> torch.Tensor:
    """P distinct flat positions (row-major over H*W) per image; (N, P) long."""
    return torch.stack([torch.randperm(H * W, generator=gen)[:P] for _ in range(N)], 0)


def gather_patches(x: torch.Tensor, k: int, pad: int, pos: torch.Tensor) -> torch.Tensor:
    """Patches of x (N, Cin, H, W) at the flat positions pos (N, P), stride 1, zero padding `pad`.
    Returns (N*P, Cin*k*k) in F.unfold's ordering (channel-major, then kh, kw)."""
    N, Cin, H, W = x.shape
    P = pos.shape[1]
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    ys, xs = pos // W, pos % W                                             # (N, P) top-left corner in padded coords
    ky = torch.arange(k, device=x.device)
    rows = ys.to(x.device)[:, :, None, None] + ky[None, None, :, None]     # (N, P, k, 1)
    cols = xs.to(x.device)[:, :, None, None] + ky[None, None, None, :]     # (N, P, 1, k)
    img = torch.arange(N, device=x.device)[:, None, None, None]
    patches = xp.permute(0, 2, 3, 1)[img, rows, cols]                      # (N, P, k, k, Cin)
    return patches.permute(0, 1, 4, 2, 3).reshape(N * P, Cin * k * k)


@torch.no_grad()
def extract_layer_patches(model, loader, idx: int, positions: int, seed: int, device: str, limit: int | None):
    """X (d, n) float32 on the CPU for the convolution model.features[idx]; prefix model.features[:idx] on the images."""
    conv = model.features[idx]
    k, pad = conv.kernel_size[0], conv.padding[0]
    assert conv.stride == (1, 1) and conv.kernel_size[0] == conv.kernel_size[1] and conv.padding[0] == conv.padding[1]
    prefix = model.features[:idx]
    gen = torch.Generator().manual_seed(seed * 1000 + idx)
    chunks, seen = [], 0
    for x, _ in loader:
        if limit is not None and seen >= limit:
            break
        x = x.to(device)
        h = prefix(x)
        N, _, H, W = h.shape
        pos = sample_positions(N, H, W, positions, gen)
        chunks.append(gather_patches(h, k, pad, pos).float().cpu())
        seen += N
    X = torch.cat(chunks, 0).T.contiguous()                                # (d, n)
    W = conv.weight.detach().reshape(conv.out_channels, -1).float().cpu()
    b = conv.bias.detach().float().cpu() if conv.bias is not None else None
    return X, W, b


# ------------------------------------------------------------------------------------------------- tails and the lemma
def rk_and_grad(G: np.ndarray, X: np.ndarray, s: float, k: int, relu: bool = True):
    """Reference (numpy, float64): R_k(G) = ||F - F_k||_F^2 and its gradient (2s/sqrt n)(F_perp o sigma'(Z)) X^T
    (Lemma M.2.0(iii)); used by the tests and to tie tail_statistics to the lemma."""
    n = X.shape[1]
    Z = s * (G @ X)
    Y = np.maximum(Z, 0.0) if relu else Z
    F = (Y - Y.mean(1, keepdims=True)) / math.sqrt(n)
    Uf, sv, Vt = np.linalg.svd(F, full_matrices=False)
    Fk = (Uf[:, :k] * sv[:k]) @ Vt[:k] if k > 0 else np.zeros_like(F)
    Fperp = F - Fk
    Rk = float(np.sum(Fperp ** 2))
    S = (Z > 0).astype(F.dtype) if relu else np.ones_like(F)
    grad = (2.0 * s / math.sqrt(n)) * ((Fperp * S) @ X.T)
    return Rk, grad


def tail_statistics(F: torch.Tensor, S: torch.Tensor | None, X: torch.Tensor, s: float, refresh: int = 64) -> dict:
    """F = centred activations / sqrt(n) (C, n); S = sigma'(Z) mask (C, n) or None for the linear step; X (d, n).
    Returns lam (r,), R_k and G2_k for k = 0..r, T2 and k*(tau)."""
    C, n = F.shape
    d = X.shape[0]
    r = min(C, d)
    Sig = F @ F.T
    Sig = Sig.cpu().double()
    Sig = (Sig + Sig.T) / 2
    lam, Uc = np.linalg.eigh(Sig.numpy())
    order = np.argsort(lam)[::-1]
    lam, Uc = np.clip(lam[order][:r], 0, None), Uc[:, order][:, :r]
    R = np.concatenate([np.cumsum(lam[::-1])[::-1], [0.0]])
    T2 = float(R[0])
    Ud = torch.from_numpy(np.ascontiguousarray(Uc)).to(F.dtype).to(F.device)
    G2 = np.zeros(r + 1)
    scale = 4.0 * s * s / n
    if S is None:                                                          # linear step: F_perp X^T = (I - U_k U_k^T)(F X^T)
        FX = F @ X.T                                                       # (C, d)
        UFX = Ud.T @ FX                                                    # (r, d)
        G2[0] = scale * float((FX ** 2).sum())
        acc = FX.clone()
        for k in range(1, r + 1):
            acc -= Ud[:, k - 1:k] * UFX[k - 1:k]
            if k % refresh == 0:
                acc = FX - Ud[:, :k] @ UFX[:k]
            G2[k] = scale * float((acc ** 2).sum())
    else:
        UF = Ud.T @ F                                                      # (r, n)
        Fperp = F.clone()
        G2[0] = scale * float(((Fperp * S) @ X.T).square().sum())
        for k in range(1, r + 1):
            Fperp -= Ud[:, k - 1:k] * UF[k - 1:k]
            if k % refresh == 0:
                Fperp = F - Ud[:, :k] @ UF[:k]
            G2[k] = scale * float(((Fperp * S) @ X.T).square().sum())
    G2[-1] = 0.0 if r == C or G2[-1] < 1e-12 * max(G2[0], 1e-30) else G2[-1]
    return dict(lam=lam, R=R, G2=G2, T2=T2, kstar={tau: int(kstar(lam, tau)) for tau in TAUS})


def step_statistics(Z: torch.Tensor, X: torch.Tensor, s: float) -> dict:
    n = Z.shape[1]
    out = {}
    Fz = (Z - Z.mean(1, keepdim=True)) / math.sqrt(n)
    out["z"] = tail_statistics(Fz, None, X, s)
    del Fz
    A = torch.relu(Z)
    S = (Z > 0).to(Z.dtype)
    Fa = (A - A.mean(1, keepdim=True)) / math.sqrt(n)
    del A
    out["a"] = tail_statistics(Fa, S, X, s)
    return out


# ---------------------------------------------------------------------------------------------------------- the interval
def estimate_interval(R: np.ndarray, G2: np.ndarray, T2: np.ndarray, lam1_Mp: float, s: float, r: int, u: float = U):
    """R, G2: (K_est, r+1); T2: (K_est,). Theorem M.2(3) with the Poincare first inequality (c_P = 1); the MC-sd and
    lambda_1 versions as controls. Returns per tau the intervals and the curves' values at the MC k*."""
    ER, EG2, ET2 = R.mean(0), G2.mean(0), T2.mean()
    hbar = ER / ET2
    with np.errstate(divide="ignore", invalid="ignore"):
        r_poinc = np.where(ER > 0, np.sqrt(C_P * EG2) / ER, 0.0)
        r_mc = np.where(ER > 0, R.std(0) / ER, 0.0)
        r_lam = np.where(ER > 0, np.sqrt(4.0 * C_P * s * s * lam1_Mp / ER), 0.0)
    q_poinc = float(np.sqrt(C_P * EG2[0]) / ET2)
    q_mc = float(T2.std() / ET2)
    q_lam = float(np.sqrt(4.0 * C_P * s * s * lam1_Mp / ET2))
    out = dict(q_poinc=q_poinc, q_mc=q_mc, q_lam=q_lam, hbar=hbar, r_poinc=r_poinc, r_mc=r_mc)
    for tau in TAUS:
        out[tau] = dict(interval=chebyshev_interval(hbar, r_poinc, q_poinc, u, tau, r),
                        interval_mc=chebyshev_interval(hbar, r_mc, q_mc, u, tau, r),
                        interval_lam=chebyshev_interval(hbar, r_lam, q_lam, u, tau, r))
    return out


# ------------------------------------------------------------------------------------------------------------- the layer
def analyse_layer(X_cpu: torch.Tensor, W_actual: torch.Tensor, C: int, draws: int, seed: int, device: str, layer: str,
                  log=print) -> tuple[list[dict], dict]:
    d, n = X_cpu.shape
    r = min(C, d)
    s = math.sqrt(2.0 / d)
    X = X_cpu.to(device)
    Mp = (X @ X.T / n).cpu().double().numpy()
    lam1_Mp = float(np.linalg.eigvalsh((Mp + Mp.T) / 2).max())
    Xc = X - X.mean(1, keepdim=True)
    Sp = (Xc @ Xc.T / n).cpu().double().numpy()
    del Xc
    lam1_Sp = float(np.linalg.eigvalsh((Sp + Sp.T) / 2).max())
    tr_Sp = float(np.trace(Sp))
    del Mp, Sp
    gen = torch.Generator().manual_seed(seed)
    per = {step: dict(lam=[], R=[], G2=[], T2=[], k={tau: [] for tau in TAUS}) for step in ("z", "a")}
    t0 = time.time()
    # the actual seed-1 weights (scale already in W; k*, R/T^2 and G2/R are scale-free)
    Z = (W_actual.to(device) @ X)
    actual = step_statistics(Z, X, 1.0)
    del Z
    for t in range(draws):
        G = torch.randn(C, d, generator=gen).to(device)
        Z = s * (G @ X)
        st = step_statistics(Z, X, s)
        del Z, G
        for step in ("z", "a"):
            per[step]["lam"].append(st[step]["lam"])
            per[step]["R"].append(st[step]["R"])
            per[step]["G2"].append(st[step]["G2"])
            per[step]["T2"].append(st[step]["T2"])
            for tau in TAUS:
                per[step]["k"][tau].append(st[step]["kstar"][tau])
        if t == 0 or (t + 1) % 10 == 0:
            log(f"  {layer}: draw {t + 1}/{draws}  k*_z(0.95)={st['z']['kstar'][0.95]}  k*_a(0.95)={st['a']['kstar'][0.95]}  "
                f"{(time.time() - t0) / (t + 1):.1f} s/draw", flush=True)
    del X
    rows, arrays = [], dict(n=n, d=d, C=C, r=r, s=s, lam1_Mp=lam1_Mp, lam1_Sp=lam1_Sp, tr_Sp=tr_Sp)
    K_est = draws // 2
    for step in ("z", "a"):
        R = np.stack(per[step]["R"])
        G2 = np.stack(per[step]["G2"])
        T2 = np.asarray(per[step]["T2"])
        arrays[f"R_{step}"], arrays[f"G2_{step}"], arrays[f"T2_{step}"] = R, G2, T2
        arrays[f"lam_{step}"] = np.stack(per[step]["lam"])
        est = estimate_interval(R[:K_est], G2[:K_est], T2[:K_est], lam1_Mp, s, r)
        arrays[f"hbar_{step}"], arrays[f"r_poinc_{step}"], arrays[f"r_mc_{step}"] = est["hbar"], est["r_poinc"], est["r_mc"]
        for tau in TAUS:
            ks = np.asarray(per[step]["k"][tau])
            arrays[f"k_{step}_{str(tau)[2:]}"] = ks
            hold = ks[K_est:]
            (km, kp), (km_mc, kp_mc), (km_l, kp_l) = est[tau]["interval"], est[tau]["interval_mc"], est[tau]["interval_lam"]
            kk = int(np.clip(round(ks.mean()), 1, r))
            k_act = actual[step]["kstar"][tau]
            rows.append(dict(layer=layer, step=step, tau=tau, C=C, d=d, r_max=r, n=n, n_over_C=n / C,
                             k_actual=k_act, mc_mean=float(ks.mean()), mc_sd=float(ks.std()), mc_min=int(ks.min()), mc_max=int(ks.max()),
                             mc_sd_est=float(ks[:K_est].std()), sd_over_rmax=float(ks.std()) / r,
                             k_minus=km, k_plus=kp, k_minus_mc=km_mc, k_plus_mc=kp_mc, k_minus_lam=km_l, k_plus_lam=kp_l,
                             n_holdout=int(len(hold)),
                             cov_holdout=float(np.mean((hold >= km) & (hold <= kp))),
                             cov_holdout_mc=float(np.mean((hold >= km_mc) & (hold <= kp_mc))),
                             cov_holdout_lam=float(np.mean((hold >= km_l) & (hold <= kp_l))),
                             cov_all=float(np.mean((ks >= km) & (ks <= kp))),
                             actual_inside=int(km <= k_act <= kp),
                             halfwidth_over_rmax=(kp - km) / 2.0 / r, halfwidth_lam_over_rmax=(kp_l - km_l) / 2.0 / r,
                             degenerate=int(kp >= r), degenerate_lam=int(kp_l >= r),
                             q_poinc=est["q_poinc"], q_mc=est["q_mc"], q_lam=est["q_lam"],
                             r_poinc_at_k=float(est["r_poinc"][kk]), r_mc_at_k=float(est["r_mc"][kk]), hbar_at_k=float(est["hbar"][kk]),
                             lam1_Mp=lam1_Mp, lam1_Sp=lam1_Sp, tr_Sp=tr_Sp, sr=tr_Sp / lam1_Mp,
                             seconds=time.time() - t0))
    return rows, arrays


# ---------------------------------------------------------------------------------------------------------------- judge
def judge(rows: pd.DataFrame, u: float = U) -> tuple[dict, str]:
    lines = [f"plan 9.57 judgement (u = {u}, guarantee {1 - 4 / u ** 2:.3f}); layers = {rows.layer.nunique()}, "
             f"hold-out draws per layer = {int(rows.n_holdout.iloc[0]) if 'n_holdout' in rows else 'n/a'}"]
    out = {}
    cov_ok = True
    for step in ("z", "a"):
        for tau in TAUS:
            s = rows[(rows.step == step) & (rows.tau == tau)]
            pooled = float((s.cov_holdout * s.n_holdout).sum() / s.n_holdout.sum())
            out[f"coverage_{step}@{tau}"] = pooled
            cov_ok &= pooled >= 0.80
            extra = ""
            if "cov_holdout_mc" in s:
                extra = (f"; MC-sd interval {float((s.cov_holdout_mc * s.n_holdout).sum() / s.n_holdout.sum()):.3f}; "
                         f"lambda_1 interval {float((s.cov_holdout_lam * s.n_holdout).sum() / s.n_holdout.sum()):.3f}; "
                         f"actual weights inside {int(s.actual_inside.sum())}/{len(s)}")
            lines.append(f"step {step} tau={tau}: hold-out coverage {pooled:.3f} ({int((s.cov_holdout * s.n_holdout).sum())}/{int(s.n_holdout.sum())}); "
                         f"median halfwidth/r_max {s.halfwidth_over_rmax.median():.3f}; non-degenerate {1 - s.degenerate.mean():.3f}; "
                         f"median sd(k*)/r_max {s.sd_over_rmax.median():.4f}{extra}")
    a95 = rows[(rows.step == "a") & (rows.tau == 0.95)]
    out["median_halfwidth_over_rmax_a@0.95"] = float(a95.halfwidth_over_rmax.median())
    out["frac_nondegenerate_a@0.95"] = float(1 - a95.degenerate.mean())
    out["max_sd_over_rmax_a@0.95"] = float(a95.sd_over_rmax.max())
    out["layers_sd_le_005"] = int((a95.sd_over_rmax <= 0.05).sum())
    if "mc_sd" in a95:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ((a95.k_plus - a95.k_minus) / 2.0 / a95.mc_sd.replace(0, np.nan)).median()
        out["median_halfwidth_over_mcsd_a@0.95"] = float(ratio)
        lines.append(f"activation step tau=0.95: median halfwidth / sd(k*) {ratio:.2f}")
    for _, rr in a95.iterrows():
        if "mc_sd" in a95:
            lines.append(f"   {rr.layer:12s} C={int(rr.C):4d} d={int(rr.d):5d} actual {int(rr.k_actual):4d}  MC {rr.mc_mean:7.1f} +- {rr.mc_sd:4.1f}  "
                         f"[{int(rr.k_minus)}, {int(rr.k_plus)}]  hold-out {rr.cov_holdout:.2f}  hw/r_max {rr.halfwidth_over_rmax:.3f}  "
                         f"lam1(Mp)/lam1(Sp) {rr.lam1_Mp / rr.lam1_Sp:.2f}")
    p132 = bool(cov_ok)
    p133 = out["median_halfwidth_over_rmax_a@0.95"] <= 0.10 and out["frac_nondegenerate_a@0.95"] >= 0.80
    p134 = bool((a95.sd_over_rmax <= 0.05).all())
    out["P9.132"], out["P9.133"], out["P9.134"] = p132, bool(p133), p134
    lines.append(f"P9.132 (hold-out coverage >= 0.80, both steps, both tau): {'HOLDS' if p132 else 'FAILS'}")
    lines.append(f"P9.133 (activation step tau=0.95: median halfwidth <= 0.10 r_max and k_plus < r_max in >= 80% of layers): {'HOLDS' if p133 else 'FAILS'}")
    lines.append(f"P9.134 (sd(k*_a(0.95)) <= 0.05 r_max in every layer): {'HOLDS' if p134 else 'FAILS'} ({out['layers_sd_le_005']}/{len(a95)})")
    return out, "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--limit", type=int, default=1600)
    ap.add_argument("--positions", type=int, default=32)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1, help="network seed (earlier layers, actual weights, positions)")
    ap.add_argument("--draw-seed", type=int, default=100)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="indices into model.features (default: every conv)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.limit, a.draws, a.positions = 20, 6, 8
        a.layers = a.layers or [0, 3]
        a.out = os.path.join(a.out, "smoke")
    os.makedirs(a.out, exist_ok=True)
    from layerspec import api
    torch.manual_seed(a.seed)
    model = api.load_model(a.model, seed=a.seed).eval().to(a.device)
    loader = api.image_loader(a.data, batch_size=32, workers=2, image_size=224, limit=a.limit, seed=a.seed)
    idxs = a.layers or [i for i, m in enumerate(model.features) if isinstance(m, torch.nn.Conv2d)]
    rows_path = os.path.join(a.out, "rows.csv")
    all_rows = []
    for idx in idxs:
        layer = f"features.{idx}"
        t0 = time.time()
        X, W, b = extract_layer_patches(model, loader, idx, a.positions, a.seed, a.device, a.limit)
        if b is not None and float(b.abs().max()) > 0:
            print(f"[warn] {layer}: non-zero bias (max {float(b.abs().max()):.3g}) ignored for the actual-weights row", file=sys.stderr)
        print(f"{layer}: X {tuple(X.shape)}  C={W.shape[0]}  extracted in {time.time() - t0:.0f} s", flush=True)
        rows, arrays = analyse_layer(X, W, W.shape[0], a.draws, a.draw_seed + idx, a.device, layer)
        del X
        np.savez_compressed(os.path.join(a.out, f"{layer.replace('.', '_')}_draws.npz"), **arrays)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(rows_path, index=False)
        if a.device == "mps":
            torch.mps.empty_cache()
    rows = pd.DataFrame(all_rows)
    verdict, text = judge(rows)
    with open(os.path.join(a.out, "judgement.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    with open(os.path.join(a.out, "judgement.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
