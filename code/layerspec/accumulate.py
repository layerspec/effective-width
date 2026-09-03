"""Streaming, numerically-stable covariance accumulation over channel space.

The quantity we want, for each conv layer, is the C x C covariance of that
layer's channel responses, where each *spatial position of each image* is one
sample.  Two things make the naive implementation wrong:

1.  Precision.  Summing millions of outer products in float32 loses several
    digits, and the small eigenvalues -- exactly the ones that decide k* -- are
    the first casualties.  We therefore use Chan et al.'s pairwise/parallel
    update, which combines two independently-centred blocks without ever
    forming a large uncentred sum, and we keep the running accumulator in
    float64 on the CPU.

2.  Effective sample size.  Spatially adjacent activations are strongly
    correlated, so N*H*W is *not* the number of independent samples.  Counting
    them all inflates the apparent sample size and makes the finite-sampling
    control look better than it is.  We therefore subsample a fixed number of
    random spatial positions per image (see `positions_per_image` in the
    hook layer) and report that count as n.

Reference for the update rule:
    Chan, Golub & LeVeque (1983), "Algorithms for computing the sample
    variance: analysis and recommendations."
"""

from __future__ import annotations

import numpy as np


class CovarianceAccumulator:
    """Accumulate mean and scatter of C-dimensional samples arriving in blocks.

    Maintains
        n     : number of samples seen
        mean  : (C,)   running mean
        M2    : (C, C) running scatter, i.e. sum of (x - mean)(x - mean)^T

    so that the unbiased covariance is M2 / (n - 1).
    """

    __slots__ = ("dim", "n", "mean", "M2")

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros((self.dim, self.dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Fold in a block of samples.

        Parameters
        ----------
        x : (m, C) array.  Cast to float64 internally.  Centring is done
            per-block before the outer product, which is what keeps the
            magnitudes small enough that the block scatter is accurate even
            when the caller computed it from float32 activations.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (m, {self.dim}) block, got {x.shape}")

        m = x.shape[0]
        if m == 0:
            return

        block_mean = x.mean(axis=0)
        xc = x - block_mean
        block_M2 = xc.T @ xc

        self._merge(m, block_mean, block_M2)

    def update_precomputed(
        self, count: int, block_mean: np.ndarray, block_M2: np.ndarray
    ) -> None:
        """Fold in a block whose mean and centred scatter were computed elsewhere.

        This is the fast path: the caller computes `block_mean` and
        `block_M2 = (x - block_mean)^T (x - block_mean)` on the GPU, then hands
        over two small arrays.  Only C and C x C values cross the bus per block,
        not the samples themselves.
        """
        if count == 0:
            return
        self._merge(
            int(count),
            np.asarray(block_mean, dtype=np.float64),
            np.asarray(block_M2, dtype=np.float64),
        )

    def _merge(self, m: int, block_mean: np.ndarray, block_M2: np.ndarray) -> None:
        if self.n == 0:
            self.n = m
            self.mean = block_mean.copy()
            self.M2 = block_M2.copy()
            return

        n_a, n_b = self.n, m
        n_ab = n_a + n_b
        delta = block_mean - self.mean

        # Chan parallel update.  The delta term is what a naive
        # "sum of per-block scatters" would omit.
        self.M2 += block_M2 + np.outer(delta, delta) * (n_a * n_b / n_ab)
        self.mean += delta * (n_b / n_ab)
        self.n = n_ab

    @property
    def covariance(self) -> np.ndarray:
        """Unbiased sample covariance, (C, C)."""
        if self.n < 2:
            raise ValueError(f"need >= 2 samples, have {self.n}")
        cov = self.M2 / (self.n - 1)
        # Symmetrise: the update is symmetric in exact arithmetic, but rounding
        # can leave a ~1e-16 asymmetry that upsets some eigensolvers.
        return (cov + cov.T) * 0.5

    def eigenvalues(self, clip_negative: bool = True) -> np.ndarray:
        """Eigenvalues of the covariance, descending.

        A sample covariance is PSD in exact arithmetic; rounding can push the
        smallest eigenvalues slightly negative.  Those are numerically zero and
        are clipped, not discarded -- dropping them would silently change C and
        therefore every ratio we report.
        """
        vals = np.linalg.eigvalsh(self.covariance)[::-1]
        if clip_negative:
            vals = np.maximum(vals, 0.0)
        return vals

    def state(self) -> dict:
        return {"dim": self.dim, "n": self.n, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_state(cls, state: dict) -> "CovarianceAccumulator":
        acc = cls(int(state["dim"]))
        acc.n = int(state["n"])
        acc.mean = np.asarray(state["mean"], dtype=np.float64)
        acc.M2 = np.asarray(state["M2"], dtype=np.float64)
        return acc


def merge(a: CovarianceAccumulator, b: CovarianceAccumulator) -> CovarianceAccumulator:
    """Combine two accumulators over the same channel space.

    Used to fold together shards produced by separate workers, so a run split
    across several rented machines gives bit-comparable results to one machine.
    """
    if a.dim != b.dim:
        raise ValueError(f"dimension mismatch: {a.dim} vs {b.dim}")
    out = CovarianceAccumulator(a.dim)
    out.n, out.mean, out.M2 = a.n, a.mean.copy(), a.M2.copy()
    out._merge(b.n, b.mean, b.M2)
    return out
