"""
Loads the snapshot files in data/snapshots (built by data_sources.py) and
hands out one asset at a time as daily returns plus the T-bill return for
the same days. Everything here is offline.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "data", "snapshots")

ASSETS = {
    "us": "US stock market (Ken French)",
    "dev_ex_us": "Developed ex US stocks (Ken French, USD)",
    "europe": "Europe stocks (Ken French, USD)",
    "japan": "Japan stocks (Ken French, USD)",
    "asia_pac_ex_japan": "Asia Pacific ex Japan stocks (Ken French, USD)",
    "ust10": "US 10 year Treasury (model from FRED DGS10)",
    "gld": "Gold (GLD)",
    "dbc": "Commodities (DBC)",
    "ief": "7 to 10 year Treasuries (IEF)",
    "spy": "SPY (cached prices from the first study)",
}


@lru_cache(maxsize=1)
def daily_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(SNAP, "daily_returns.csv.gz"), index_col=0, parse_dates=True)


@lru_cache(maxsize=1)
def monthly_table() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(SNAP, "monthly_yields_and_bonds.csv"), index_col=0)
    m.index = pd.PeriodIndex(m.index, freq="M")
    return m


def rf_daily() -> pd.Series:
    rf = daily_table()["rf"].dropna()
    return rf


def last_rf_date() -> pd.Timestamp:
    return rf_daily().index[-1]


def asset(name: str, start=None, end=None) -> tuple[pd.Series, pd.Series]:
    """
    (daily returns, daily T-bill returns) on the asset's own trading days,
    cut at the last day the T-bill series exists. T-bill interest from days
    the asset didn't trade gets added to its next trading day so none is lost.
    """
    if name == "spy":
        px = pd.read_csv(os.path.join(ROOT, "spy_prices.csv"), index_col=0, parse_dates=True).iloc[:, 0]
        r = px.astype(float).pct_change().iloc[1:]
    else:
        r = daily_table()[name].dropna()
    r = r[r.index <= last_rf_date()]
    rf = cash_returns(r.index, "tbill")
    return _window(r.rename(name), rf, start, end)


def _window(r, rf, start, end):
    if start is not None:
        keep = r.index >= pd.Timestamp(start)
        r, rf = r[keep], rf[keep]
    if end is not None:
        keep = r.index <= pd.Timestamp(end)
        r, rf = r[keep], rf[keep]
    return r, rf.rename("rf")


def cash_returns(index: pd.DatetimeIndex, cash="tbill") -> pd.Series:
    """
    What cash earns on each day of `index`. "tbill" is the daily T-bill series
    (the last known rate carries on past its end), a number is a flat annual
    rate, and 0 or "none" means cash earns nothing.
    """
    if isinstance(cash, str) and cash not in ("tbill", "none"):
        raise ValueError(f"cash must be \"tbill\", \"none\" or an annual rate like 0.03, not {cash!r}")
    if isinstance(cash, bool) or (not isinstance(cash, str) and not np.isscalar(cash)):
        raise ValueError(f"cash must be \"tbill\", \"none\" or an annual rate like 0.03, not {cash!r}")
    if cash == "none":
        cash = 0.0
    if not isinstance(cash, str):
        gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
        days = np.r_[1.0, gaps]
        return pd.Series((1.0 + float(cash)) ** (days / 365.0) - 1.0, index=index, name="rf")
    rf_all = rf_daily()
    rf_all = rf_all[rf_all.index >= index[0]]
    # T-bill returns between two trading days are tiny, so summing them is fine
    bucket = pd.Series(index, index=index).reindex(rf_all.index.union(index)).bfill()
    rf = rf_all.reindex(bucket.index).fillna(0.0).groupby(bucket.values).sum().reindex(index).fillna(0.0)
    late = index > last_rf_date()
    if late.any():
        # past the end of the T-bill data, keep the last daily rate for each weekday
        days = index.values.astype("datetime64[D]")
        weekdays = np.r_[1, np.busday_count(days[:-1], days[1:])].clip(min=1)
        rf[late] = rf_daily().iloc[-1] * weekdays[late]
    return rf.rename("rf")


PRICE_COLUMNS = ("adj close", "adj_close", "adjclose", "close", "price", "last", "value")


def from_csv(path: str, column=None, kind: str = "price", date_column=None,
             name=None, cash="tbill", start=None, end=None) -> tuple[pd.Series, pd.Series]:
    """
    Daily returns for any asset from your own csv, plus the cash returns for
    the same days. The file needs a date column and one price (or return)
    column. Prices should be adjusted for dividends and splits if you want
    total returns. kind = "returns" means the column already holds decimal
    returns (0.01 is 1%).
    """
    if kind not in ("price", "returns"):
        raise ValueError(f"kind must be \"price\" or \"returns\", not {kind!r}")
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except FileNotFoundError:
        raise ValueError(f"price file not found: {path}") from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ValueError(f"{path} doesn't look like a csv ({e})") from None
    df.columns = [str(c).strip() for c in df.columns]
    dcol = date_column or df.columns[0]
    if dcol not in df.columns:
        raise ValueError(f"{path} has no date column {dcol!r}, its columns are {list(df.columns)}")
    if column is None:
        lower = {c.lower(): c for c in df.columns if c != dcol}
        hits = [lower[c] for c in PRICE_COLUMNS if c in lower]
        rest = [c for c in df.columns if c != dcol]
        if hits:
            column = hits[0]
        elif len(rest) == 1:
            column = rest[0]
        else:
            raise ValueError(f"{path} has several columns {rest}, say which one with column = \"...\"")
    if column not in df.columns:
        raise ValueError(f"{path} has no column {column!r}, its columns are {list(df.columns)}")
    try:
        idx = pd.to_datetime(df[dcol])
    except (ValueError, TypeError) as e:
        raise ValueError(f"couldn't read the dates in column {dcol!r} of {path} ({e})") from None
    # values like "1,234.50" or "$99" show up in hand made files
    x = pd.to_numeric(df[column].astype(str).str.replace(r"[,$%\s]", "", regex=True), errors="coerce")
    x.index = idx.dt.tz_localize(None) if getattr(idx.dt, "tz", None) is not None else idx
    x = x[~x.index.isna()].sort_index()
    if x.index.has_duplicates:
        raise ValueError(f"{path} has {int(x.index.duplicated().sum())} repeated dates")
    x = x.dropna()
    if len(x) < 30:
        raise ValueError(f"{path} has only {len(x)} usable rows in column {column!r}")
    if kind == "price":
        if (x <= 0).any():
            raise ValueError(f"column {column!r} in {path} has zero or negative prices, "
                             f"use kind = \"returns\" if it holds returns")
        r = x.pct_change().iloc[1:]
    else:
        if (x <= -1).any():
            raise ValueError(f"column {column!r} in {path} has returns of -100% or worse, they should be decimals like 0.01")
        r = x
    r = r.rename(name or os.path.splitext(os.path.basename(path))[0])
    rf = cash_returns(r.index, cash)
    return _window(r, rf, start, end)


def long_yield() -> pd.Series:
    return monthly_table()["long_yield"].dropna()


def bond_monthly() -> pd.Series:
    """US 10 year Treasury monthly returns: monthly yield model before 1962, daily DGS10 model after."""
    m = monthly_table()["ust10_monthly_model"].dropna()
    d = daily_table()["ust10"].dropna()
    dm = (1 + d).groupby(d.index.to_period("M")).prod() - 1
    dm = dm[dm.index > pd.Period("1962-01", "M")]  # Jan 1962 is a partial month in the daily file
    return pd.concat([m[m.index <= pd.Period("1962-01", "M")], dm]).sort_index()


def em_monthly() -> tuple[pd.Series, pd.Series]:
    m = monthly_table()
    return m["em"].dropna(), m["em_rf"].dropna()
