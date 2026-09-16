"""Reverse water-filling over the sites' spectra (Proposition C).

Each site l has a normalised, decreasing spectrum lam_l (eigenvalues of its post-activation
covariance divided by their sum) and a cost per channel p_l.  Discarding channels beyond C_l
costs D_l(C_l) = sum_{i > C_l} lam_l[i]; the allocation minimises sum_l w_l D_l(C_l) subject to
sum_l p_l C_l = P.  The continuous solution keeps channel i of site l iff
w_l lam_l[i] / p_l >= mu, one threshold mu for every site (equal marginal discarded variance
per unit cost); the discrete greedy pass then moves single channels while that lowers the
objective.  No tau anywhere.
"""
from __future__ import annotations

import numpy as np


def _widths_at(mu: float, spectra: list[np.ndarray], cost: np.ndarray, w: np.ndarray,
               lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    out = np.empty(len(spectra), dtype=int)
    for l, lam in enumerate(spectra):
        keep = int(np.sum(w[l] * lam / cost[l] >= mu))
        out[l] = min(max(keep, lo[l]), hi[l])
    return out


def water_fill(spectra: list[np.ndarray], cost, budget: float, *, weights=None, min_width=None,
               max_width=None, discrete: bool = True):
    """Widths (one per site) that spend `budget` (in cost units) on the spectra.

    spectra:   per site, eigenvalues of the post-activation covariance, decreasing (any scale)
    cost:      per site, cost of one channel (parameters or FLOPs)
    budget:    total cost to allocate, sum_l cost_l * width_l
    weights:   per-site sensitivity weights w_l (default 1)
    min_width / max_width: per-site bounds (default 1 and the spectrum length)
    Returns (widths, mu, objective).
    """
    L = len(spectra)
    spectra = [np.clip(np.asarray(s, dtype=float), 0, None) for s in spectra]
    spectra = [s / s.sum() if s.sum() > 0 else s for s in spectra]
    cost = np.asarray(cost, dtype=float)
    w = np.ones(L) if weights is None else np.asarray(weights, dtype=float)
    lo = np.ones(L, dtype=int) if min_width is None else np.asarray(min_width, dtype=int)
    hi = np.array([len(s) for s in spectra]) if max_width is None else np.asarray(max_width, dtype=int)
    hi = np.minimum(hi, np.array([len(s) for s in spectra]))      # no value is known beyond the spectrum
    lo = np.minimum(lo, hi)
    # bisection on mu: widths decrease with mu
    ratios = np.concatenate([w[l] * s / cost[l] for l, s in enumerate(spectra)])
    a, b = 0.0, float(ratios.max()) * 1.0001 + 1e-30
    for _ in range(200):
        mid = 0.5 * (a + b)
        if float(cost @ _widths_at(mid, spectra, cost, w, lo, hi)) > budget:
            a = mid
        else:
            b = mid
    widths = _widths_at(b, spectra, cost, w, lo, hi)
    mu = b

    def objective(ws):
        return float(sum(w[l] * spectra[l][ws[l]:].sum() for l in range(L)))

    if discrete:
        # fill any slack greedily, then swap single channels while the objective drops
        spent = float(cost @ widths)
        improved = True
        while improved:
            improved = False
            gains = np.array([w[l] * spectra[l][widths[l]] / cost[l] if widths[l] < hi[l] else -np.inf
                              for l in range(L)])
            losses = np.array([w[l] * spectra[l][widths[l] - 1] / cost[l] if widths[l] > lo[l] else np.inf
                               for l in range(L)])
            g = int(np.argmax(gains))
            if gains[g] > -np.inf and spent + cost[g] <= budget:
                widths[g] += 1; spent += cost[g]; improved = True
                continue
            s = int(np.argmin(losses))
            if gains[g] > losses[s] and g != s and spent - cost[s] + cost[g] <= budget:
                widths[g] += 1; widths[s] -= 1; spent += cost[g] - cost[s]; improved = True
    return widths, mu, objective(widths)
