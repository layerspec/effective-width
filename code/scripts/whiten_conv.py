"""Input-patch whitening folded into a convolution: the two factors of Sigma_out = W Sigma_patch W^T
(proofs T3; plan 9.56; prior art notes/t3-prior-art-2026-10-11.md).

Two modes, both exposed as an nn.Conv2d subclass so every probe that hooks nn.Conv2d (SpectrumProbe,
PatchProbe) sees the whitened convolution as an ordinary layer:

  mode="nd"   Network Deconvolution of Ye et al. (ICLR 2020), the published operator with its defaults:
              im2col patches of the layer input, channels in blocks of `block` (64), per-batch centred
              covariance of each block with sampling stride 3, D = (Cov + eps I)^{-1/2} by the coupled
              Newton-Schulz iteration (n_iter = 5) WITH autograd, mean subtracted, D folded into the kernel
              (y = X (D w) + b - mu D w); running mean / running D by EMA (momentum 0.1) for eval.  Ye et al.
              remove batch norm: apply_whitening(..., remove_bn=True) replaces the paired BN by Identity.
  mode="full" the maximiser of Proposition T3.1: the FULL d x d patch covariance (d = C_in k^2) tracked by
              EMA, D = (Cov + eps I)^{-1/2} with eps = lambda_max(Cov) / gain_cap^2 (gain_cap = 8: no direction is
              amplified more than 8x relative to the top one -- the implicit cap of ND's five Newton-Schulz
              steps, (3/2)^5 = 7.6; with eps -> 0 the 4x4 maps, where samples < d, diverged in the smoke test),
              recomputed every `update_every` training steps after a warm-up of `warmup` steps with D = I
              (eigh, float64, detached), no mean subtraction (a batch norm follows and centres), and with
              orthogonal=True the stored kernel is replaced by its polar factor (scripts/oni.py, T = 20) so
              the effective kernel is W_o D and Sigma_out = W_o D Sigma_patch D W_o^T -> I_C.

bake_to_reference(whitened, reference) writes the effective kernels into a plain copy of the architecture so
decompose_check.py / checkpoint_analysis.py read the arm like any other: the conv weight becomes W_eff; the
bias term goes into the conv bias when the reference conv has one, otherwise into the beta of the paired
batch norm (set to the identity map: gamma = 1, running_mean = 0, running_var = 1 - bn.eps); a BN that was
removed (nd) is likewise reset to the identity map in the reference.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.oni import ONI  # noqa: E402


def isqrt_newton_schulz(A: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Coupled Newton-Schulz for A^{-1/2} (A SPD, batched over leading dims): Y_0 = A/||A||_F, Z_0 = I,
    T = (3I - Z Y)/2, Y <- Y T, Z <- T Z; Z_k / sqrt(||A||_F) -> A^{-1/2}.  Ye et al. 2020, Sec. 4.5.3."""
    n = A.shape[-1]
    norm = A.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
    Y = A / norm
    I = torch.eye(n, dtype=A.dtype, device=A.device).expand_as(A)
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z / norm.sqrt()


def sym_eigh(A: torch.Tensor):
    """Eigendecomposition of a symmetric matrix in float64 on the CPU (MPS has no eigh), symmetrised first.
    macOS LAPACK occasionally returns error 26 on well-conditioned inputs; fall back to numpy, then to a
    jittered retry, so one bad call does not kill a training run."""
    A64 = A.detach().cpu().double()          # move THEN cast: .to("cpu", float64) from MPS returns garbage (MPS has no float64)
    A64 = 0.5 * (A64 + A64.transpose(-1, -2))
    try:
        return torch.linalg.eigh(A64)
    except Exception:
        pass
    try:
        import numpy as np
        w, V = np.linalg.eigh(A64.numpy())
        return torch.from_numpy(w), torch.from_numpy(V)
    except Exception:
        n = A64.shape[-1]
        jit = 1e-7 * A64.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / n
        return torch.linalg.eigh(A64 + jit * torch.eye(n, dtype=torch.float64))


def isqrt_eigh(A: torch.Tensor, reg: float = 0.0) -> torch.Tensor:
    """Exact symmetric inverse square root (A + reg I)^{-1/2} by eigendecomposition in float64."""
    w, V = sym_eigh(A)
    w = (w + reg).clamp_min(1e-12 * float(w.max()))
    return ((V * w.rsqrt()) @ V.transpose(-1, -2)).to(A.dtype).to(A.device)


class WhitenedConv2d(nn.Conv2d):
    def __init__(self, conv: nn.Conv2d, mode: str = "nd", block: int = 64, eps: float = 1e-5, n_iter: int = 5,
                 momentum: float = 0.1, sampling_stride: int = 3, update_every: int = 50, gain_cap: float = 8.0,
                 warmup: int | None = None, orthogonal: bool = False, oni_iters: int = 20):
        if conv.groups != 1:
            raise ValueError("whitening is defined for dense convolutions only")
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                         conv.dilation, conv.groups, bias=conv.bias is not None, padding_mode=conv.padding_mode)
        with torch.no_grad():
            self.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.bias.copy_(conv.bias)
        if mode not in ("nd", "full"):
            raise ValueError(mode)
        self.mode, self.eps, self.n_iter, self.momentum = mode, float(eps), int(n_iter), float(momentum)
        self.sampling_stride, self.update_every, self.gain_cap = int(sampling_stride), int(update_every), float(gain_cap)
        self.warmup = int(update_every if warmup is None else warmup)
        k2 = self.kernel_size[0] * self.kernel_size[1]
        d = self.in_channels * k2
        self.d = d
        if mode == "nd":
            self.block = min(int(block), self.in_channels)
            if self.in_channels % self.block:
                raise ValueError(f"in_channels {self.in_channels} not divisible by block {self.block}")
            self.nb = self.in_channels // self.block
            f = self.block * k2
            self.register_buffer("running_mean", torch.zeros(d))
            self.register_buffer("running_deconv", torch.eye(f).repeat(self.nb, 1, 1))
        else:
            self.register_buffer("running_cov", torch.eye(d))
            self.register_buffer("deconv", torch.eye(d))
            self.register_buffer("num_steps", torch.zeros((), dtype=torch.long))
        self.oni = ONI(oni_iters) if orthogonal else None

    # ---- patches -------------------------------------------------------------------------------
    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """im2col columns of the input at sampling stride, as (M, d) in the (C_in, kh, kw) feature order
        of the flattened kernel."""
        s = (self.stride[0] * self.sampling_stride, self.stride[1] * self.sampling_stride)
        cols = F.unfold(x, self.kernel_size, self.dilation, self.padding, s)      # (N, d, L)
        return cols.transpose(1, 2).reshape(-1, self.d)

    # ---- deconvolution matrix ------------------------------------------------------------------
    def _nd_deconv(self, X: torch.Tensor):
        """Per-batch block covariance -> D (nb, f, f) with autograd, and the batch mean (d)."""
        mu = X.mean(0)
        Xc = (X - mu).reshape(X.shape[0], self.nb, -1)                              # (M, nb, f)
        cov = torch.einsum("mbf,mbg->bfg", Xc, Xc) / X.shape[0]
        f = cov.shape[-1]
        cov = cov + self.eps * torch.eye(f, dtype=cov.dtype, device=cov.device)
        return isqrt_newton_schulz(cov, self.n_iter), mu

    @torch.no_grad()
    def _full_update(self, X: torch.Tensor) -> None:
        mu = X.mean(0)
        Xc = X - mu
        cov = Xc.T @ Xc / X.shape[0]
        if int(self.num_steps) == 0:
            self.running_cov.copy_(cov)
        else:
            self.running_cov.mul_(1 - self.momentum).add_(self.momentum * cov)
        self.num_steps += 1
        n = int(self.num_steps)
        if n >= max(self.warmup, 1) and (n == max(self.warmup, 1) or n % self.update_every == 0):
            self.refresh_full_deconv()

    @torch.no_grad()
    def refresh_full_deconv(self) -> None:
        w, V = sym_eigh(self.running_cov)
        reg = float(w[-1]) / self.gain_cap ** 2                       # eps = lambda_max / gain_cap^2
        w = (w + reg).clamp_min(1e-12 * float(w[-1]))
        self.deconv.copy_(((V * w.rsqrt()) @ V.T).to(self.deconv.dtype).to(self.deconv.device))

    # ---- effective kernel ----------------------------------------------------------------------
    def _kernel(self) -> torch.Tensor:
        W = self.weight
        if self.oni is not None:
            W = self.oni(W)
        return W.reshape(self.out_channels, self.d)

    def _effective(self, D: torch.Tensor, mu: torch.Tensor | None):
        Wm = self._kernel()
        if self.mode == "nd":
            W_eff = torch.einsum("obf,bfg->obg", Wm.reshape(self.out_channels, self.nb, -1), D).reshape(self.out_channels, self.d)
        else:
            W_eff = Wm @ D
        b = self.bias if self.bias is not None else torch.zeros(self.out_channels, dtype=W_eff.dtype, device=W_eff.device)
        if mu is not None:
            b = b - W_eff @ mu
        return W_eff.view_as(self.weight), b

    def effective_kernel(self):
        """(W_eff, b_eff) under the running statistics (what eval uses and what bake writes)."""
        with torch.no_grad():
            if self.mode == "nd":
                return self._effective(self.running_deconv, self.running_mean)
            return self._effective(self.deconv, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device.type
        with torch.autocast(dev if dev in ("cuda", "cpu", "mps") else "cpu", enabled=False):
            xf = x.float()
            if self.training:
                X = self._patches(xf)
                if self.mode == "nd":
                    D, mu = self._nd_deconv(X)
                    with torch.no_grad():
                        self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
                        self.running_deconv.mul_(1 - self.momentum).add_(self.momentum * D.detach())
                    W_eff, b_eff = self._effective(D, mu)
                else:
                    self._full_update(X)
                    W_eff, b_eff = self._effective(self.deconv, None)
            else:
                W_eff, b_eff = self.effective_kernel()
            y = F.conv2d(xf, W_eff, b_eff, self.stride, self.padding, self.dilation, self.groups)
        return y.to(x.dtype)


# ---- model surgery -------------------------------------------------------------------------------
def _parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@torch.no_grad()
def pair_conv_bn(model: nn.Module, sample: torch.Tensor) -> dict[str, str]:
    """{conv_name: bn_name} for every dense conv whose output tensor feeds a BatchNorm2d directly,
    found by one traced forward pass (robust to Sequential vs attribute layouts)."""
    out_ids, bn_in, hooks, keep = {}, {}, [], []   # `keep` holds the tensors so ids are not reused mid-pass

    def _conv_out(mod, i, o, n):
        keep.append(o)
        out_ids[id(o)] = n

    def _bn_in(mod, i, n):
        keep.append(i[0])
        bn_in[n] = id(i[0])

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1:
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: _conv_out(mod, i, o, n)))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_pre_hook(lambda mod, i, n=name: _bn_in(mod, i, n)))
    was_training = model.training
    model.eval()
    model(sample)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return {out_ids[t]: bn for bn, t in bn_in.items() if t in out_ids}


def apply_whitening(model: nn.Module, mode: str, remove_bn: bool, sample: torch.Tensor | None = None, **kw) -> nn.Module:
    """Replace every dense nn.Conv2d by WhitenedConv2d(mode); with remove_bn, the BN fed by each of them
    becomes nn.Identity (Ye et al. 2020).  Records model.whiten_pairs = {conv: bn} for baking."""
    sample = torch.zeros(1, 3, 32, 32) if sample is None else sample
    pairs = pair_conv_bn(model, sample)
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Conv2d) and not isinstance(m, WhitenedConv2d) and m.groups == 1:
            parent, attr = _parent(model, name)
            setattr(parent, attr, WhitenedConv2d(m, mode=mode, **kw))
            if remove_bn and name in pairs:
                bparent, battr = _parent(model, pairs[name])
                setattr(bparent, battr, nn.Identity())
    model.whiten_pairs = pairs
    model.whiten_mode, model.whiten_remove_bn = mode, bool(remove_bn)
    return model


def whitened_modules(model: nn.Module) -> dict[str, WhitenedConv2d]:
    return {n: m for n, m in model.named_modules() if isinstance(m, WhitenedConv2d)}


@torch.no_grad()
def bake_to_reference(whitened: nn.Module, reference: nn.Module) -> nn.Module:
    """Write the effective kernels of `whitened` into `reference` (the plain architecture, same names).
    Non-whitened parameters and buffers (trained BNs, classifier) are copied; see module docstring."""
    pairs = getattr(whitened, "whiten_pairs", {})
    wmods = whitened_modules(whitened)
    ref_mods = dict(reference.named_modules())
    src_sd = whitened.state_dict()
    dst_sd = reference.state_dict()
    for k in dst_sd:
        if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
            dst_sd[k].copy_(src_sd[k])
    reference.load_state_dict(dst_sd)
    for name, m in wmods.items():
        W_eff, b_eff = m.effective_kernel()
        conv = ref_mods[name]
        conv.weight.copy_(W_eff.to(conv.weight.dtype))
        bn_name = pairs.get(name)
        bn = ref_mods.get(bn_name) if bn_name else None
        removed = getattr(whitened, "whiten_remove_bn", False)
        if conv.bias is not None:
            conv.bias.copy_(b_eff.to(conv.bias.dtype))
            if removed and isinstance(bn, nn.BatchNorm2d):
                _identity_bn(bn, None)
        else:
            if not isinstance(bn, nn.BatchNorm2d):
                if float(b_eff.abs().max()) > 1e-6:
                    raise RuntimeError(f"{name}: bias term with no bias slot and no paired BN")
            elif removed:
                _identity_bn(bn, b_eff)
            elif float(b_eff.abs().max()) > 1e-6:
                raise RuntimeError(f"{name}: mode full should not produce a bias shift")
    return reference


@torch.no_grad()
def _identity_bn(bn: nn.BatchNorm2d, beta: torch.Tensor | None) -> None:
    bn.weight.fill_(1.0)
    bn.bias.zero_() if beta is None else bn.bias.copy_(beta.to(bn.bias.dtype))
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0 - bn.eps)   # sqrt(var + eps) = 1 exactly
    if bn.num_batches_tracked is not None:
        bn.num_batches_tracked.zero_()
