"""
backtest_core.py
================
Core logic for a moving-average crossover (trend-following / momentum) backtest.

Everything the notebook uses lives here so it can be imported, tested,
and reused. Read every function — interviewers will ask about these details.

Key design decisions (and why):
  1. NEXT-DAY EXECUTION: the signal computed from today's close is traded
     at TOMORROW's close. Using today's close for both signal and execution
     is lookahead bias — the #1 rookie backtesting error.
  2. TRANSACTION COSTS: every position change pays a cost (default 5 bps
     = 0.05% per side). Strategies that trade often die here.
  3. TRAIN/TEST SPLIT: parameters are chosen on the in-sample window only,
     then judged on out-of-sample data they never saw. Reporting in-sample
     results as "performance" is overfitting.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # convention: trading days per year, used to annualize


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

    Adjusted close matters: it folds dividends and splits back into the
    price series. Backtesting SPY on unadjusted prices silently throws away
    ~2% per year of dividend return and understates buy-and-hold.

    Caches to CSV so repeated runs don't hammer the data source and so
    results are reproducible even if the API changes.
    """
    if cache_path and os.path.exists(cache_path):
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s.astype(float)

    import yfinance as yf  # imported here so tests can run without network

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker}. Check your connection or ticker."
        )
    close = df["Close"]
    # yfinance sometimes returns a 1-column DataFrame; flatten to Series.
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
    Long-only moving-average crossover.

    Position = 1 (fully invested) when the fast MA is above the slow MA,
    else 0 (in cash). The classic 50/200 version is the "golden cross".

    CRITICAL: the returned position is shifted by one day. The MA values
    use today's close, so the earliest we could actually trade on that
    information is tomorrow. `.shift(1)` enforces that.
    """
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    raw_signal = (fast_ma > slow_ma).astype(float)
    return raw_signal.shift(1).fillna(0.0)  # <-- no lookahead


def run_backtest(
    prices: pd.Series,
    positions: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Turn a price series + daily positions into daily strategy returns.

    cost_bps: one-way transaction cost in basis points (5 = 0.05%).
    Charged on every unit of position change (|Δposition|), so a full
    round trip (0 -> 1 -> 0) costs 2 * cost_bps.

    Returns a DataFrame with daily returns for the strategy and buy-&-hold.
    """
    daily_ret = prices.pct_change().fillna(0.0)

    gross = positions * daily_ret                      # return while invested
    turnover = positions.diff().abs().fillna(0.0)      # position changes
    costs = turnover * (cost_bps / 10_000.0)           # bps -> decimal
    net = gross - costs

    return pd.DataFrame(
        {"strategy": net, "buy_hold": daily_ret}, index=prices.index
    )


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from daily returns."""
    total = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annual_vol(returns: pd.Series) -> float:
    """Annualized volatility (std of daily returns scaled by sqrt(252))."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """
    Annualized Sharpe ratio: excess return per unit of volatility.
    rf_annual: annual risk-free rate (0 keeps it simple; mention this
    simplification in your writeup — with rf > 0 every Sharpe drops).
    """
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0:
        return float("nan")
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """
    Worst peak-to-trough loss of the equity curve. Returned as a negative
    number (e.g. -0.55 means a 55% loss from the prior peak).
    """
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()


def summarize(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Metrics table (rows = metrics, columns = strategies)."""
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
# Parameter search (in-sample ONLY — that's the whole point)
# ----------------------------------------------------------------------------
def grid_search(
    prices: pd.Series,
    fast_grid: list[int],
    slow_grid: list[int],
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Evaluate every (fast, slow) pair on `prices` and rank by Sharpe.
    Call this with IN-SAMPLE prices only. The winner is then frozen and
    judged once on out-of-sample data.
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
