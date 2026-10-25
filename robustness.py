"""
robustness.py

Checks on whether the backtest result is real or just luck.

The main notebook picks one rule (50/200) and tests it once. That is a fair
test, but one test on one stretch of history can still mislead. These
functions look at the same question from a few other angles:

  sensitivity_grid      every fast/slow pair, so 50/200 isn't cherry picked
  walk_forward          re-pick the windows every year using only past data
  bootstrap_sharpe_diff resample the test period to get a confidence interval
  deflated_sharpe       how much the in sample Sharpe shrinks after counting
                        how many rules were tried (Bailey and Lopez de Prado 2014)
  pbo_cscv              probability the in sample winner is below median
                        out of sample (Bailey, Borwein, Lopez de Prado, Zhu 2017)
  cost_sensitivity      how fast the strategy dies as trading gets pricier

Signals are always built on the full price history and then sliced, so the
moving averages are already warmed up on the first test day. Nothing uses
data from after the day it trades.

Every function takes an optional rf (daily T-bill returns). With it, cash
earns interest and all Sharpe ratios are measured over T-bills. The
bootstrap, deflated Sharpe and PBO code now lives in trendlab/stats.py and
is imported here so old code keeps working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_core as bc
from trendlab.stats import (stationary_bootstrap_idx, expected_max_sharpe,
                            deflated_sharpe, pbo_cscv)

__all__ = ["strategy_returns", "returns_matrix", "make_pairs", "sensitivity_grid", "walk_forward",
           "stationary_bootstrap_idx", "bootstrap_sharpe_diff", "expected_max_sharpe", "deflated_sharpe",
           "pbo_cscv", "cost_sensitivity"]

TRADING_DAYS = bc.TRADING_DAYS


def strategy_returns(prices: pd.Series, fast: int, slow: int, cost_bps: float = 5.0,
                     rf=None, delay: int = 1) -> pd.Series:
    """Daily net returns of one crossover rule over the whole price history."""
    pos = bc.ma_crossover_positions(prices, fast, slow, delay)
    return bc.run_backtest(prices, pos, cost_bps, rf)["strategy"]


def returns_matrix(prices: pd.Series, pairs, cost_bps: float = 5.0, rf=None, delay: int = 1) -> pd.DataFrame:
    """One column of daily returns per (fast, slow) pair."""
    return pd.DataFrame(
        {f"{f}/{s}": strategy_returns(prices, f, s, cost_bps, rf, delay) for f, s in pairs}
    )


def _sr(r, rf):
    return bc.sharpe(r, rf=None if rf is None else rf.reindex(r.index))


def make_pairs(fast_list, slow_list):
    return [(f, s) for f in fast_list for s in slow_list if f < s]


# ---------------------------------------------------------------------------
# 1. Parameter sensitivity
# ---------------------------------------------------------------------------
def sensitivity_grid(prices, split, fast_list, slow_list, cost_bps=5.0, rf=None, delay=1) -> pd.DataFrame:
    """In sample and out of sample Sharpe for every pair, next to buy and hold."""
    bh = prices.pct_change().fillna(0.0)
    bh_in, bh_out = _sr(bh[bh.index < split], rf), _sr(bh[bh.index >= split], rf)
    rows = []
    for f, s in make_pairs(fast_list, slow_list):
        r = strategy_returns(prices, f, s, cost_bps, rf, delay)
        rows.append({
            "fast": f, "slow": s,
            "sharpe_in": _sr(r[r.index < split], rf),
            "sharpe_out": _sr(r[r.index >= split], rf),
        })
    out = pd.DataFrame(rows)
    out["bh_in"], out["bh_out"] = bh_in, bh_out
    out["beats_bh_out"] = out["sharpe_out"] > bh_out
    return out


# ---------------------------------------------------------------------------
# 2. Walk forward
# ---------------------------------------------------------------------------
def walk_forward(prices, fast_list, slow_list, first_year=2008, min_years=5, cost_bps=5.0,
                 rf=None, delay=1):
    """
    Every January, pick the pair with the best Sharpe on all data before that
    year, then hold it for the year. Stitch the years together.

    This is closer to what someone would actually do than a single split,
    because the choice keeps getting updated as new data comes in.
    """
    pairs = make_pairs(fast_list, slow_list)
    R = returns_matrix(prices, pairs, cost_bps, rf, delay)
    bh = prices.pct_change().fillna(0.0)
    years = sorted(set(prices.index.year))
    picks, chunks = [], []
    for y in years:
        if y < first_year:
            continue
        past = R[R.index.year < y]
        if past.index.year.nunique() < min_years:
            continue
        sr = past.apply(lambda c: _sr(c, rf))
        best = sr.idxmax()
        this = R.loc[R.index.year == y, best]
        picks.append({"year": y, "pick": best, "sharpe_past": sr[best],
                      "ret_year": (1 + this).prod() - 1,
                      "bh_year": (1 + bh[bh.index.year == y]).prod() - 1})
        chunks.append(this)
    wf = pd.concat(chunks)
    return pd.DataFrame(picks), wf, bh.loc[wf.index]


# ---------------------------------------------------------------------------
# 3. Bootstrap confidence interval for the Sharpe gap
# ---------------------------------------------------------------------------
def bootstrap_sharpe_diff(strat, bench, n_boot=2000, mean_block=20, seed=0, rf=None):
    """
    Resample whole days (both series together, so their link is kept) in
    random blocks that average `mean_block` days, which keeps volatility
    clusters intact. Returns the bootstrap distribution of
    Sharpe(strategy) minus Sharpe(benchmark).
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(strat, dtype=float), np.asarray(bench, dtype=float)
    if rf is not None:  # work with returns over T-bills
        c = np.asarray(rf, dtype=float)
        a, b = a - c, b - c
    n = len(a)
    k = np.sqrt(TRADING_DAYS)
    out = np.empty(n_boot)
    for j in range(n_boot):
        ix = stationary_bootstrap_idx(n, mean_block, rng)
        x, y = a[ix], b[ix]
        out[j] = k * (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1))
    return out


# 4 and 5, the deflated Sharpe ratio and PBO, are imported from trendlab.stats at the top.


# ---------------------------------------------------------------------------
# 6. Costs
# ---------------------------------------------------------------------------
def cost_sensitivity(prices, split, fast, slow, costs=(0, 5, 10, 20, 50), rf=None, delay=1):
    bh = prices.pct_change().fillna(0.0)
    bh_out = _sr(bh[bh.index >= split], rf)
    rows = []
    for c in costs:
        r = strategy_returns(prices, fast, slow, c, rf, delay)
        rows.append({"cost_bps": c, "sharpe_out": _sr(r[r.index >= split], rf), "bh_out": bh_out})
    return pd.DataFrame(rows)
