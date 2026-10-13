"""
backtest_core.py

Core logic for the moving average crossover backtest. The notebook imports
everything from here so it can be tested on its own.

Three choices matter most:
  1. Next day execution. A signal from today's close is traded at tomorrow's
     close. Using today's close for both would be lookahead bias.
  2. Trading costs. Every position change pays a cost, 5 bps (0.05%) per
     side by default.
  3. Train and test split. The windows are picked on the in sample years
     only and then judged on years they never saw.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # trading days in a year, used to annualize


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_prices(
    ticker: str = "SPY",
    start: str = "2000-01-01",
    cache_path: str | None = None,
) -> pd.Series:
    """
    Load daily ADJUSTED closing prices for `ticker`.

    Adjusted closes include dividends and splits. Without them buy and hold
    loses about 2% a year of dividends and looks worse than it really was.

    Saves to a CSV so repeat runs don't download again and the results stay
    the same even if the data source changes.
    """
    if cache_path and os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s.astype(float)

    import yfinance as yf  # imported here so the tests run without internet

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check your connection or ticker."
        )
    close = df["Close"]
    # yfinance sometimes returns a one column DataFrame, so flatten it
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker

    if cache_path:
        close.to_csv(cache_path)
    return close.astype(float)


# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------
def ma_crossover_positions(
    prices: pd.Series, fast: int, slow: int
) -> pd.Series:
    """
    Long only moving average crossover.

    Position is 1 (fully invested) when the fast average is above the slow
    one, otherwise 0 (cash). 50/200 is the classic "golden cross".

    The position is shifted by one day. The averages use today's close, so
    the earliest you could trade on them is tomorrow.
    """
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    raw_signal = (fast_ma > slow_ma).astype(float)
    return raw_signal.shift(1).fillna(0.0)  # trade tomorrow, not today


def run_backtest(
    prices: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Turn prices and daily positions into daily strategy returns.

    cost_bps is the one way cost in basis points (5 = 0.05%). It's charged on
    every change in position, so a full round trip (0 to 1 to 0) costs
    2 * cost_bps.

    Returns daily returns for the strategy and for buy and hold.
    """
    daily_ret = prices.pct_change().fillna(0.0)

    gross = positions * daily_ret                      # return while invested
    turnover = positions.diff().abs().fillna(0.0)      # position changes
    costs = turnover * (cost_bps / 10_000.0)           # bps to decimal
    net = gross - costs

    return pd.DataFrame(
        {"strategy": net, "buy_hold": daily_ret}, index=prices.index
    )


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def cagr(returns: pd.Series) -> float:
    """Yearly growth rate from daily returns."""
    total = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series) -> float:
    """Yearly volatility, the daily standard deviation times sqrt(252)."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of volatility.
    rf_annual is the yearly risk free rate. It defaults to 0, which keeps
    things simple but means every Sharpe here is a bit higher than it would
    be with cash earning interest.
    """
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0:
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """
    Worst drop from a peak, as a negative number (-0.55 means down 55%
    from the previous high).
    """
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def summarize(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Table of metrics, one column per strategy."""
    out = {}
    for col in returns_df.columns:
        r = returns_df[col]
        out[col] = {
            "CAGR": cagr(r),
            "Annual Vol": annual_vol(r),
            "Sharpe (rf=0)": sharpe(r),
            "Max Drawdown": max_drawdown(r),
        }
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Parameter search, in sample only
# ----------------------------------------------------------------------------
def grid_search(
    prices: pd.Series,
    fast_grid: list[int],
    slow_grid: list[int],
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Run every (fast, slow) pair on `prices` and rank them by Sharpe.
    Only pass in sample prices here. The winner gets frozen and tested once
    on the out of sample years.
    """
    rows = []
    for f in fast_grid:
        for s in slow_grid:
            if f >= s:
                continue
            pos = ma_crossover_positions(prices, f, s)
            res = run_backtest(prices, pos, cost_bps)
            rows.append(
                {
                    "fast": f,
                    "slow": s,
                    "sharpe": sharpe(res["strategy"]),
                    "cagr": cagr(res["strategy"]),
                    "max_dd": max_drawdown(res["strategy"]),
                    "trades": int(pos.diff().abs().sum()),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
