"""
Passive comparisons. A static mix holds w in the asset and 1 - w in T-bills,
rebalanced every month. The question is always whether timing beats just
holding less of the asset.
"""

from __future__ import annotations

import numpy as np

from . import metrics as mt


def mix(asset, rf, w: float):
    return w * np.asarray(asset, dtype=float) + (1.0 - w) * np.asarray(rf, dtype=float)


def vol_matched_w(target, asset, rf) -> float:
    """w so the mix has the same monthly volatility as `target`. Solved exactly."""
    a = np.asarray(asset, dtype=float)
    c = np.asarray(rf, dtype=float)
    t_sd = np.std(np.asarray(target, dtype=float), ddof=1)
    # var(w a + (1-w) c) = w^2 var(a-c) + 2 w cov(a-c, c) + var(c), solve for w >= 0
    x = a - c
    A = np.var(x, ddof=1)
    B = 2 * np.cov(x, c, ddof=1)[0, 1]
    C = np.var(c, ddof=1) - t_sd ** 2
    disc = B * B - 4 * A * C
    if A <= 0 or disc < 0:
        return float("nan")
    return float((-B + np.sqrt(disc)) / (2 * A))


def return_matched_w(target, asset, rf, w_max: float = 3.0) -> tuple[float, str]:
    """
    w so the mix has the same CAGR as `target`. Growth of a mix rises with w
    and then falls once you're levered enough, so search on the rising side.
    Returns (w, flag), flag is "", "levered", "below_tbills" or "unreachable".
    """
    a = np.asarray(asset, dtype=float)
    c = np.asarray(rf, dtype=float)
    goal = mt.cagr(target)
    g = lambda w: mt.cagr(mix(a, c, w))
    grid = np.linspace(0.0, w_max, 301)
    vals = np.array([g(w) for w in grid])
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    top = int(np.argmax(vals))
    if goal <= vals[0]:
        return 0.0, "below_tbills"
    if goal > vals[top]:
        return float("nan"), "unreachable"
    lo, hi = 0.0, grid[top]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if g(mid) < goal:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return float(w), ("levered" if w > 1 else "")


def best_growth_w(asset, rf, w_max: float = 3.0) -> float:
    """The mix weight with the highest CAGR on a grid from 0 to w_max."""
    grid = np.linspace(0.0, w_max, 301)
    vals = [mt.cagr(mix(asset, rf, w)) for w in grid]
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    return float(grid[int(np.argmax(vals))])


def sixty_forty(equity, bonds, w_eq: float = 0.6):
    return w_eq * np.asarray(equity, dtype=float) + (1 - w_eq) * np.asarray(bonds, dtype=float)
