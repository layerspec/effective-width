"""ONI, the orthogonalisation-by-Newton-iteration layer of Huang et al. (CVPR 2020), as a weight
parametrisation (submission bar D5, 2026-09-24): the kernel the convolution USES is the polar factor
of the stored kernel, so every dense layer is an isometry by construction rather than by a penalty
(the SO / SRIP arms of plan 9.2 are penalties and reached only half the predicted effective-width gain).

For a kernel W of shape (n, d) (conv: n = C_out, d = C_in k_h k_w):
    rows orthonormal (W W^T = I) when n <= d, columns orthonormal (W^T W = I) when n > d
    (Huang 2020 note that for n > d the rows cannot be orthogonal).
Newton-Schulz:  V = A / sigma_max(A) (spectral bounding, sigma_max by power iteration on a
persistent vector, 15 iterations on the first call then 2, x1.05 margin, detached), then
X_0 = V, X_{t+1} = 1.5 X_t - 0.5 (X_t X_t^T) X_t, W_orth = X_T.  This is Huang's B_T V with
B_{t+1} = 1.5 B_t - 0.5 B_t^3 (V V^T) written in its polar form: the same limit (V V^T)^{-1/2} V,
but the one-sided B iteration loses commutativity in float32 and diverges after it has converged
(verified 2026-09-24: error 1e-4 at t = 5, nan at t = 20), the polar form does not.  Five iterations (the paper's T) bring a random kaiming kernel to erank/r_max > 0.99;
thirty give machine precision.  Computed in float32 outside autocast.

    apply_oni(model)      # every nn.Conv2d (groups == 1) and nn.Linear gets the parametrisation
    bake_oni(model)       # replace each stored kernel by its orthogonalised value and drop the
                          # parametrisation: same function, plain state_dict keys
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class ONI(nn.Module):
    def __init__(self, iters: int = 20, power_iters: int = 2):
        super().__init__()
        self.iters, self.power_iters = int(iters), int(power_iters)
        self.register_buffer("u", torch.empty(0), persistent=False)

    def _sigma_max(self, A: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n = A.shape[0]
            iters = self.power_iters
            if self.u.numel() != n:                       # first call: converge from a random start
                self.u = torch.randn(n, generator=torch.Generator().manual_seed(0)).to(A)
                iters = max(iters, 15)
            u = self.u
            for _ in range(iters):
                v = A.T @ u; v = v / (v.norm() + 1e-12)
                u = A @ v;   u = u / (u.norm() + 1e-12)
            self.u = u
            return 1.05 * (u @ (A @ v)).clamp_min(1e-12)   # margin: the Rayleigh quotient is a lower bound

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        shape = W.shape
        dev = W.device.type
        with torch.autocast(dev if dev in ("cuda", "cpu", "mps") else "cpu", enabled=False):
            M = W.reshape(shape[0], -1).float()
            transpose = M.shape[0] > M.shape[1]
            A = M.T if transpose else M                       # rows to orthonormalise, A is m x k, m <= k
            V = A / self._sigma_max(A)
            X = V                                             # Newton-Schulz for the polar factor:
            for _ in range(self.iters):                       # X <- 1.5 X - 0.5 (X X^T) X, the same limit
                X = 1.5 * X - 0.5 * (X @ X.T) @ X             # (V V^T)^{-1/2} V as Huang's B_T V, but stable
            Wo = X
            if transpose:
                Wo = Wo.T
        return Wo.reshape(shape).to(W.dtype)


def _targets(model: nn.Module):
    for m in model.modules():
        if (isinstance(m, nn.Conv2d) and m.groups == 1) or isinstance(m, nn.Linear):
            yield m


def apply_oni(model: nn.Module, iters: int = 20) -> nn.Module:
    """Register the ONI parametrisation on every dense conv and linear kernel (in place)."""
    for m in _targets(model):
        if not parametrize.is_parametrized(m, "weight"):
            parametrize.register_parametrization(m, "weight", ONI(iters))
    return model


def oni_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in _targets(model) if parametrize.is_parametrized(m, "weight")]


def bake_oni(model: nn.Module) -> nn.Module:
    """Store the orthogonalised kernels as the plain weights and remove the parametrisation."""
    for m in oni_modules(model):
        parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
    return model
