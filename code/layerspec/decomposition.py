"""Where a layer's effective width comes from: the kernel or the data.

At the conv output (before the nonlinearity and the shortcut)
    Sigma_out = W Sigma_patch W^T,
W the (C_out x d) kernel matrix, d = C_in*k_h*k_w, Sigma_patch the d x d
covariance of the receptive-field patch (im2col column).  PatchProbe
accumulates Sigma_patch per dense convolution; decompose_rows reports, per
layer and tau,
    out      k*(W Sigma W^T)/r_max     the measured quantity, recomputed
    kernel   k*(W W^T)/r_max           the layer under white patches
    data     k*(Sigma_patch)/r_max     the patch covariance, capped at r_max
    ortho    k*(W_o Sigma W_o^T)/r_max W_o the polar factor of W: what
                                        orthogonalising this kernel would give with this data
and the Proposition-1 checks (Ostrowski ratios, corollary bound).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .accumulate import CovarianceAccumulator
from .metrics import DEFAULT_TAUS, tau_key


def kstar(eig: np.ndarray, tau: float) -> int:
    eig = np.clip(eig, 0, None)
    if eig.sum() <= 0:
        return 0
    c = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(c, tau) + 1)


def erank(eig: np.ndarray) -> float:
    eig = np.clip(eig, 0, None)
    p = eig / eig.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


class PatchProbe:
    """Forward-pre-hooks on every dense (groups == 1) nn.Conv2d that accumulate
    the covariance of `positions` sampled im2col columns per image."""

    def __init__(self, model: nn.Module, positions: int, seed: int, include_linear: bool = False):
        self.acc: dict[str, CovarianceAccumulator] = {}
        self.meta: dict[str, dict] = {}
        self.handles = []
        self.p = positions
        self.gen = torch.Generator().manual_seed(seed)
        order = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                self.meta[name] = dict(depth_index=order, module=m)
                order += 1
                self.handles.append(m.register_forward_pre_hook(
                    lambda mod, inp, _n=name: self._consume(_n, mod, inp[0])))
            elif include_linear and isinstance(m, nn.Linear):
                self.meta[name] = dict(depth_index=order, module=m)
                order += 1
                self.handles.append(m.register_forward_pre_hook(
                    lambda mod, inp, _n=name: self._consume_linear(_n, mod, inp[0])))

    @torch.no_grad()
    def _consume_linear(self, name: str, m: nn.Linear, x: torch.Tensor) -> None:
        """A linear layer's 'patch' is its input vector; over a token sequence
        (N, T, D) the tokens are the positions.  Unlike the conv path, all
        tokens are taken when positions >= T (exact covariance)."""
        if x.dim() == 3:
            N, T, D = x.shape
            p = min(self.p, T)
            if p < T:
                idx = torch.randint(0, T, (N, p), generator=self.gen).to(x.device)
                x = torch.gather(x, 1, idx.unsqueeze(-1).expand(N, p, D))
            xs = x.reshape(-1, D)
        elif x.dim() == 2:
            xs = x
        else:
            return
        xs = xs.to(torch.float32)
        acc = self.acc.get(name)
        if acc is None:
            acc = self.acc[name] = CovarianceAccumulator(xs.shape[1])
            self.meta[name].update(d=m.in_features, C_in=m.in_features, C_out=m.out_features, k=1)
        bm = xs.mean(0)
        xc = xs - bm
        acc.update_precomputed(xs.shape[0], bm.cpu().double().numpy(), (xc.T @ xc).cpu().double().numpy())

    @torch.no_grad()
    def _consume(self, name: str, m: nn.Conv2d, x: torch.Tensor) -> None:
        N = x.shape[0]
        cols = F.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding, stride=m.stride)  # (N, d, L)
        d, L = cols.shape[1], cols.shape[2]
        if L == 1:
            return                                   # conv on a pooled tensor: not in the profile
        p = min(self.p, L)
        idx = torch.randint(0, L, (N, p), generator=self.gen).to(cols.device)
        cols = torch.gather(cols, 2, idx.unsqueeze(1).expand(N, d, p))   # (N, d, p)
        xs = cols.permute(0, 2, 1).reshape(N * p, d).to(torch.float32)
        acc = self.acc.get(name)
        if acc is None:
            acc = self.acc[name] = CovarianceAccumulator(d)
            self.meta[name].update(d=d, C_in=m.in_channels, C_out=m.out_channels,
                                   k=m.kernel_size[0] * m.kernel_size[1])
        bm = xs.mean(0)
        xc = xs - bm
        acc.update_precomputed(xs.shape[0], bm.cpu().double().numpy(), (xc.T @ xc).cpu().double().numpy())

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def rho_align(sv: np.ndarray, Vt: np.ndarray, Sig: np.ndarray, r: int) -> float:
    """Kernel-data alignment index (analysis-plan 9.6): Spearman correlation,
    over the r retained right-singular directions v_i of W, between the
    kernel gain s_i^2 and the input variance carried by that direction,
    v_i^T Sigma_patch v_i.  Positive: the kernel amplifies the data's
    high-variance directions (widening); negative: it suppresses them
    (compressive); an isometric or isotropic kernel gives 0 (returned as nan
    when either sequence is constant)."""
    if r < 3:
        return float("nan")
    V = Vt[:r]                                         # (r, d)
    v_in = np.einsum("id,de,ie->i", V, Sig, V)         # v_i^T Sig v_i
    gain = sv[:r] ** 2
    # A kernel whose retained singular values are equal up to float error
    # (an isometry) has no ordering to correlate: nan, not noise.
    if np.ptp(gain) <= 1e-6 * gain.max() or np.ptp(v_in) <= 1e-6 * abs(v_in).max():
        return float("nan")
    ra = pd.Series(gain).rank().to_numpy(); rb = pd.Series(v_in).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rho_align_sigma(W: np.ndarray, Sig: np.ndarray, rank_tol: float = 1e-6) -> float:
    """The alignment index in the eigenbasis of Sigma_patch (Proposition 4):
    Spearman correlation, over the principal directions q_j of the patch
    covariance with lambda_j above rank_tol of the largest, between the
    kernel gain ||W q_j||^2 and the input variance lambda_j.  For the
    least-squares layer the gain is c_j / lambda_j^2, so a flat task relevance
    c_j gives -1 (whitening) and an isotropic kernel gives 0."""
    lam, Q = np.linalg.eigh(Sig)
    order = np.argsort(lam)[::-1]; lam, Q = lam[order], Q[:, order]
    r = int((lam > lam[0] * rank_tol).sum())
    if r < 3:
        return float("nan")
    gain = np.sum((W @ Q[:, :r]) ** 2, axis=0)
    if np.ptp(gain) <= 1e-6 * gain.max():
        return float("nan")
    ra = pd.Series(gain).rank().to_numpy(); rb = pd.Series(lam[:r]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _sym_eigs(S: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]


def decompose_rows(probe: PatchProbe, taus: tuple[float, ...] = DEFAULT_TAUS,
                   rank_tol: float = 1e-6) -> list[dict]:
    """One row per dense convolution with an accumulated patch covariance."""
    rows = []
    for name, acc in probe.acc.items():
        meta = probe.meta[name]
        m = meta["module"]
        d, C_out = meta["d"], meta["C_out"]
        r_max = min(d, C_out)
        W = m.weight.detach().cpu().double().reshape(C_out, d).numpy()
        Sig = acc.covariance
        lam_data = np.sort(np.linalg.eigvalsh(Sig))[::-1]
        lam_out = _sym_eigs(W @ Sig @ W.T)
        lam_kernel = _sym_eigs(W @ W.T)
        U, sv, Vt = np.linalg.svd(W, full_matrices=False)
        Wo = U @ Vt                                     # polar factor, nearest (semi-)isometry
        lam_ortho = _sym_eigs(Wo @ Sig @ Wo.T)
        # Proposition 1: with W = P W_o, Sigma_out = P Sigma_o P and (Ostrowski)
        # lambda_i(out) = theta_i lambda_i(ortho), sigma_min^2 <= theta_i <= sigma_max^2
        # over the r = rank(W) nonzero directions; hence
        #   k*_out(tau) <= k*_ortho(tau'),  tau' = kappa^2 tau / (1 - tau + kappa^2 tau).
        r = int((sv > sv[0] * rank_tol).sum())
        kappa = float(sv[0] / sv[r - 1])
        n_null = int(len(sv) - r)
        ratio = lam_out[:r] / np.clip(lam_ortho[:r], 1e-300, None)
        ostrowski_ok = bool(np.all(ratio >= sv[r - 1] ** 2 * (1 - 1e-6))
                            and np.all(ratio <= sv[0] ** 2 * (1 + 1e-6)))
        row = dict(layer=name, depth_index=meta["depth_index"], C_in=meta["C_in"], C_out=C_out,
                   k=meta["k"], d=d, r_max=r_max, n_samples=acc.n, n_over_d=acc.n / d,
                   kernel_erank_over_rmax=erank(sv ** 2) / r_max,
                   kernel_cond_top_rmax=float(sv[0] / sv[min(r_max, len(sv)) - 1]),
                   data_erank_over_rmax=min(erank(lam_data), r_max) / r_max,
                   rank_W=r, n_null=n_null, kappa=kappa, ostrowski_ok=ostrowski_ok)
        for tau in taus:
            tk = tau_key(tau)
            tau_k = kappa ** 2 * tau / (1 - tau + kappa ** 2 * tau)
            row[f"bound_{tk}"] = kstar(lam_ortho, tau_k) / r_max if tau_k < 1 else 1.0
            row[f"bound_ok_{tk}"] = bool(kstar(lam_out, tau) <= (kstar(lam_ortho, tau_k) if tau_k < 1 else r_max))
            row[f"out_{tk}"] = kstar(lam_out, tau) / r_max
            row[f"kernel_{tk}"] = kstar(lam_kernel, tau) / r_max
            row[f"data_{tk}"] = min(kstar(lam_data, tau), r_max) / r_max
            row[f"ortho_{tk}"] = kstar(lam_ortho, tau) / r_max
        row["rho_align"] = rho_align(sv, Vt, Sig, r)
        row["rho_align_sigma"] = rho_align_sigma(W, Sig, rank_tol)
        rows.append(row)
    return rows


def check_rows(df: pd.DataFrame, taus: tuple[float, ...] = DEFAULT_TAUS) -> dict:
    """Proposition-1 counts over a decomposition table, plus the sample-size
    gate: `n_over_d` < 50 means the patch covariance (and therefore all four
    of out/kernel/data/ortho) is a sampling artefact for that row, same
    reasoning as `n_over_C_ok` in `record_row` -- not added as a CSV column,
    since the decomposition table's own header must not change."""
    n = len(df)
    return {"ostrowski": (int(df.ostrowski_ok.sum()), n),
            "corollary": {tau: (int(df[f"bound_ok_{tau_key(tau)}"].sum()), n) for tau in taus},
            "kappa_median": float(df.kappa.median()) if n else float("nan"),
            "kappa_max": float(df.kappa.max()) if n else float("nan"),
            "n_null_layers": int((df.n_null > 0).sum()),
            "n_over_d_min": float(df.n_over_d.min()) if n else float("nan"),
            "n_below_50": int((df.n_over_d < 50).sum()) if n else 0,
            "rho_align_median": float(df.rho_align.median()) if n and "rho_align" in df else float("nan")}


def format_check(c: dict) -> str:
    """The one-line summary decompose_check.py prints (kept byte-identical)."""
    return (f"Proposition 1 check: Ostrowski ratios within [s_min^2, s_max^2] on {c['ostrowski'][0]}/{c['ostrowski'][1]} layers; "
            f"k*_out(tau) <= k*_ortho(tau') on " + ", ".join(f"tau={t}: {v[0]}/{v[1]}" for t, v in c["corollary"].items())
            + f";  kappa median {c['kappa_median']:.1f}, max {c['kappa_max']:.1f}")
