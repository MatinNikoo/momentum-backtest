"""
What each rule says right now.

    python -m trendlab today            # from the snapshot files, no internet
    python -m trendlab today --live     # pull fresh ETF prices from Yahoo and T-bill yields from FRED

This is a research tool. It is not investment advice and it doesn't know
anything about your situation.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import data as dt
from . import rules as rl

SNAPSHOT_ASSETS = ["us", "dev_ex_us", "ust10", "gld", "dbc"]
LIVE_TICKERS = {"SPY": "US stocks", "EFA": "Developed ex US stocks", "IEF": "7 to 10 year Treasuries",
                "GLD": "Gold", "DBC": "Commodities"}
BANNER = "Research only. Not investment advice. Signals use closes up to the date shown."


def describe(px: pd.Series, rf: pd.Series) -> dict:
    """Current state of the three protocol rules for one total return index."""
    px = px.dropna()
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    sig = rl.ma_crossover(px, 50, 200)
    if len(px) < 200 or np.isnan(ma200.iloc[-1]):
        raise ValueError("need at least 200 days of prices")
    changes = np.flatnonzero(np.diff(sig.values) != 0)
    last_change = px.index[changes[-1] + 1] if len(changes) else px.index[0]
    sma = rl.sma_month(px, 10)
    mom = rl.tsmom(px, rf.reindex(px.index).fillna(0.0), 12)
    return {
        "date": str(px.index[-1].date()),
        "ma50_over_ma200_pct": float(ma50.iloc[-1] / ma200.iloc[-1] - 1),
        "ma_50_200": "in" if sig.iloc[-1] == 1 else "out",
        "ma_50_200_since": str(last_change.date()),
        "ma_50_200_changed_today": bool(len(changes) and changes[-1] + 1 == len(px) - 1),
        "sma_10m": "in" if sma.iloc[-1] == 1 else "out",
        "tsmom_12m": "in" if mom.iloc[-1] > 0 else "out",
    }


def from_snapshot() -> dict:
    d = dt.daily_table()
    rf = d["rf"].dropna()
    out = {}
    for a in SNAPSHOT_ASSETS:
        r = d[a].dropna()
        px = (1 + r).cumprod()
        # T-bill data stops earlier than the ETF prices, so hold the last known daily rate
        rfa = rf.reindex(px.index).ffill().fillna(0.0)
        out[dt.ASSETS[a]] = describe(px, rfa)
    return out


def from_live() -> dict:
    import requests
    import yfinance as yf

    raw = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3", timeout=60).text
    y = pd.read_csv(io.StringIO(raw), na_values=["."]).dropna()
    y = pd.Series(y.iloc[:, 1].astype(float).values / 100, index=pd.to_datetime(y.iloc[:, 0]))
    out = {}
    for t, name in LIVE_TICKERS.items():
        df = yf.download(t, period="3y", auto_adjust=True, progress=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        rf = ((1 + y.reindex(close.index).ffill()) ** (1 / 252) - 1).fillna(0.0)
        out[f"{name} ({t})"] = describe(close.astype(float), rf)
    return out


def table(states: dict) -> str:
    df = pd.DataFrame(states).T
    df["ma50_over_ma200_pct"] = df["ma50_over_ma200_pct"].map(lambda v: f"{v:+.1%}")
    df = df.rename(columns={"ma50_over_ma200_pct": "50d vs 200d", "ma_50_200": "50/200",
                            "ma_50_200_since": "same state since", "ma_50_200_changed_today": "flipped today",
                            "sma_10m": "10 month", "tsmom_12m": "12m momentum"})
    return BANNER + "\n\n" + df.to_string()
