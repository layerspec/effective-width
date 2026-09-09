"""Dimensionality metrics computed from an eigenvalue spectrum.

Four different quantities get called "the dimensionality of a representation"
in this literature and they are *not* interchangeable -- Ansuini et al. (2019)
report PC-ID ~ 200 and TwoNN ID ~ 18 for the same VGG-16 layer, an order of
magnitude apart.  Everything here is a *linear* measure computed from the
covariance spectrum; none of it estimates manifold dimension.  Keep the names
straight in the paper.

Implemented:
    k_star(tau)     number of PCs reaching tau of total variance.  This is the
                    Garg/Panda/Roy quantity (they used tau = 0.999).
    participation_ratio   (sum L)^2 / sum L^2.  The Elmoznino & Bonner
                    quantity.  Smooth, no threshold, but weights the head of
                    the spectrum heavily.
    effective_rank  exp(entropy of the normalised spectrum), Roy & Vetterli
                    (2007).  Threshold-free like PR but responds to the tail.
    stable_rank     sum L / L_1.  Cheapest, most head-dominated.
    powerlaw_alpha  slope of log L vs log i over a stated index range.  The
                    range MUST be reported: Kong et al. (2022) fit PCs 10-999,
                    and the value moves a lot with the window.

All of these are reported both raw and divided by C (the nominal channel
count), because the paper's claim is about the *ratio*.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

DEFAULT_TAUS = (0.90, 0.95, 0.99, 0.999)


def tau_key(tau: float) -> str:
    """Column-name suffix for a tau value (e.g. 0.990 -> "0.99").  The one
    place this formatting is decided; `api.py` and `decompose.py` both import
    it rather than each picking their own `f"{tau:g}"` / `f"{tau}"`."""
    return f"{tau:g}"


@dataclass
class SpectrumMetrics:
    C: int
    n_samples: int
    total_variance: float
    k_star: dict = field(default_factory=dict)          # tau -> int
    k_star_ratio: dict = field(default_factory=dict)    # tau -> k*/C
    participation_ratio: float = float("nan")
    participation_ratio_norm: float = float("nan")
    effective_rank: float = float("nan")
    effective_rank_norm: float = float("nan")
    stable_rank: float = float("nan")
    stable_rank_norm: float = float("nan")
    powerlaw_alpha: float = float("nan")
    powerlaw_fit_lo: int = 0
    powerlaw_fit_hi: int = 0
    powerlaw_r2: float = float("nan")
    nonzero_eigs: int = 0

    def to_row(self) -> dict:
        row = asdict(self)
        for tau, v in row.pop("k_star").items():
            row[f"k_star_{tau}"] = v
        for tau, v in row.pop("k_star_ratio").items():
            row[f"k_star_ratio_{tau}"] = v
        return row


def k_star(eigenvalues: np.ndarray, tau: float) -> int:
    """Smallest k whose leading eigenvalues carry at least `tau` of the variance."""
    total = eigenvalues.sum()
    if total <= 0:
        return 0
    cumulative = np.cumsum(eigenvalues) / total
    # searchsorted on a nondecreasing array; +1 converts index to a count.
    return int(np.searchsorted(cumulative, tau) + 1)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    s1 = eigenvalues.sum()
    s2 = np.square(eigenvalues).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def effective_rank(eigenvalues: np.ndarray) -> float:
    """exp(Shannon entropy of the normalised spectrum). Roy & Vetterli (2007)."""
    total = eigenvalues.sum()
    if total <= 0:
        return float("nan")
    p = eigenvalues / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(eigenvalues: np.ndarray) -> float:
    return float(eigenvalues.sum() / eigenvalues[0]) if eigenvalues[0] > 0 else float("nan")


def powerlaw_alpha(
    eigenvalues: np.ndarray, lo: int = 10, hi: int | None = None
) -> tuple[float, int, int, float]:
    """Fit L_i ~ i^(-alpha) by least squares on log-log, over ranks [lo, hi).

    Returns (alpha, lo_used, hi_used, r2).  `lo` is 1-based and defaults to 10
    to skip the head, following the convention in the V1-eigenspectrum
    literature.  Note Pospisil & Pillow (2025) argue the spectrum is a *broken*
    power law, so a single alpha is a summary, not a model -- report the window.
    """
    n = len(eigenvalues)
    hi = n if hi is None else min(hi, n)
    lo = max(1, lo)
    if hi - lo < 8:
        return float("nan"), lo, hi, float("nan")

    idx = np.arange(lo, hi + 1, dtype=np.float64)
    vals = eigenvalues[lo - 1 : hi]
    mask = vals > 0
    if mask.sum() < 8:
        return float("nan"), lo, hi, float("nan")

    x = np.log(idx[mask])
    y = np.log(vals[mask])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), lo, hi, r2


def compute(
    eigenvalues: np.ndarray,
    C: int,
    n_samples: int,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    powerlaw_lo: int = 10,
    powerlaw_hi: int | None = None,
) -> SpectrumMetrics:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    total = float(eigenvalues.sum())
    alpha, lo, hi, r2 = powerlaw_alpha(eigenvalues, powerlaw_lo, powerlaw_hi)

    m = SpectrumMetrics(
        C=int(C),
        n_samples=int(n_samples),
        total_variance=total,
        participation_ratio=participation_ratio(eigenvalues),
        effective_rank=effective_rank(eigenvalues),
        stable_rank=stable_rank(eigenvalues),
        powerlaw_alpha=alpha,
        powerlaw_fit_lo=lo,
        powerlaw_fit_hi=hi,
        powerlaw_r2=r2,
        nonzero_eigs=int((eigenvalues > total * 1e-12).sum()) if total > 0 else 0,
    )
    for tau in taus:
        key = tau_key(tau)
        k = k_star(eigenvalues, tau)
        m.k_star[key] = k
        m.k_star_ratio[key] = k / C if C else float("nan")

    m.participation_ratio_norm = m.participation_ratio / C if C else float("nan")
    m.effective_rank_norm = m.effective_rank / C if C else float("nan")
    m.stable_rank_norm = m.stable_rank / C if C else float("nan")
    return m
