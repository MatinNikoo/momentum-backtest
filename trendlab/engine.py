"""
The backtest engine. Signals in, daily returns out.

Timing convention, since this is where backtests usually cheat:
  signal[t]   uses information up to and including the close of day t
  delay       trading days between that close and the close you trade at
  position[t] the exposure that earns day t's return (close t-1 to close t)

So with delay=1 you see the close on day t, trade at the close of t+1, and
the first return you earn is day t+2's. That's a shift of delay + 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def positions_from_signal(signal: pd.Series, delay: int = 1) -> pd.Series:
    if delay < 0:
        raise ValueError("delay can't be negative, that would be lookahead")
    return signal.shift(delay + 1).fillna(0.0)


def run(returns: pd.Series, positions: pd.Series, rf: pd.Series | float | None = None,
        cost_bps: float = 5.0) -> pd.DataFrame:
    """
    Daily net returns for one asset.

    Whatever isn't in the asset sits in T-bills and earns rf that day. A
    position above 1 borrows the difference at rf, below 0 is short. Each
    unit of position change pays cost_bps.
    """
    r = returns.astype(float).fillna(0.0)
    pos = positions.reindex(r.index).fillna(0.0)
    if rf is None:
        cash = pd.Series(0.0, index=r.index)
    elif np.isscalar(rf):
        cash = pd.Series(float(rf), index=r.index)
    else:
        cash = rf.reindex(r.index).fillna(0.0)

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    cost = turnover * cost_bps / 10_000.0
    gross = pos * r + (1.0 - pos) * cash
    return pd.DataFrame({
        "strategy": gross - cost,
        "asset": r,
        "rf": cash,
        "position": pos,
        "turnover": turnover,
        "cost": cost,
    })


def run_portfolio(sleeves: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """
    Combine daily sleeve returns with fixed weights, rebalanced at each
    month end. Inside a month the weights drift with the sleeves, which is
    what you'd actually hold.
    """
    df = pd.DataFrame(sleeves).fillna(0.0)
    w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    month = df.index.to_period("M")
    out = []
    for _, chunk in df.groupby(month):
        growth = (1.0 + chunk).cumprod()
        value = (growth * w).sum(axis=1)  # value of the book, starting at sum(w)
        prev = value.shift(1).fillna(w.sum())
        out.append(value / prev - 1.0)
    return pd.concat(out).rename("portfolio")


def total_return_index(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0.0)).cumprod()
