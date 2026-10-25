"""Performance and risk metrics. Most of the study works on monthly returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def to_monthly(daily: pd.Series | pd.DataFrame):
    """Compound daily returns into calendar months, indexed by Period('M')."""
    g = (1.0 + daily).groupby(daily.index.to_period("M")).prod() - 1.0
    return g


def to_quarterly(monthly: pd.Series | pd.DataFrame):
    return (1.0 + monthly).groupby(monthly.index.asfreq("Q")).prod() - 1.0


def cagr(r: pd.Series, periods: int = 12) -> float:
    r = np.asarray(r, dtype=float)
    total = np.prod(1.0 + r)
    years = len(r) / periods
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def vol(r, periods: int = 12) -> float:
    return float(np.std(np.asarray(r, dtype=float), ddof=1) * np.sqrt(periods))


def sharpe(r, rf=0.0, periods: int = 12) -> float:
    x = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def drawdown(r) -> np.ndarray:
    w = np.cumprod(1.0 + np.asarray(r, dtype=float))
    w = np.r_[1.0, w]  # start from the initial dollar so a first month loss counts
    return (w / np.maximum.accumulate(w) - 1.0)[1:]


def max_dd_depth(r) -> float:
    """Worst drop from a peak as a positive number, 0.40 means down 40%."""
    return float(-drawdown(r).min())


def ulcer(r) -> float:
    dd = drawdown(r)
    return float(np.sqrt(np.mean(dd ** 2)))


def worst_window(r, n: int = 12) -> float:
    x = np.log1p(np.asarray(r, dtype=float))
    if len(x) < n:
        return float("nan")
    c = np.r_[0.0, np.cumsum(x)]
    return float(np.expm1((c[n:] - c[:-n]).min()))


def cvar(r, q: float = 0.05) -> float:
    x = np.sort(np.asarray(r, dtype=float))
    k = max(1, int(np.floor(q * len(x))))
    return float(x[:k].mean())


def longest_underwater(r) -> int:
    dd = drawdown(r)
    best = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def summary(r: pd.Series, rf: pd.Series, periods: int = 12) -> dict:
    r = pd.Series(r).astype(float)
    return {
        "cagr": cagr(r, periods),
        "vol": vol(r, periods),
        "sharpe": sharpe(r, rf, periods),
        "max_dd": max_dd_depth(r),
        "ulcer": ulcer(r),
        "worst_12m": worst_window(r, 12 if periods == 12 else 252),
        "cvar_5": cvar(r),
        "skew": float(stats.skew(r)),
        "longest_underwater_months": longest_underwater(r) if periods == 12 else None,
        "months": len(r),
    }
