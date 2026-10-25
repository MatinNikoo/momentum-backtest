"""
Trading rules. Each one takes a total return index (and T-bill returns when
it needs them) and gives back a daily signal: the exposure you'd want after
seeing that day's close. The engine handles the delay, so nothing in here
shifts anything forward. Every rolling window only looks backwards.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

ANN = 261  # MOP (2012) annualize daily variance with 261


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last trading day of each month in the index."""
    m = index.to_period("M")
    return np.r_[m[1:] != m[:-1], True]


def month_is_complete(last: pd.Timestamp) -> bool:
    """True if no weekday is left in `last`'s month. Old data has Saturday sessions, which count as done."""
    return (last + pd.offsets.BDay(1)).month != last.month


def _complete_month_end(index: pd.DatetimeIndex) -> np.ndarray:
    # the last row of the data might be mid month. Only treat it as a month end
    # if no weekday is left in its month.
    mask = month_end_mask(index)
    if not month_is_complete(index[-1]):
        mask[-1] = False
    return mask


def ma_crossover(px: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """1 when the fast average is above the slow one, else 0."""
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    f = px.rolling(fast).mean()
    s = px.rolling(slow).mean()
    return (f > s).astype(float)


def price_above_ma(px: pd.Series, window: int = 200) -> pd.Series:
    """1 when the close is above its own moving average, else 0."""
    return (px > px.rolling(window).mean()).astype(float)


def ema_crossover(px: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Like ma_crossover with exponential averages. Spans are in days and the first `slow` days stay out."""
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    f = px.ewm(span=fast, adjust=False).mean()
    s = px.ewm(span=slow, adjust=False).mean()
    sig = (f > s).astype(float)
    sig.iloc[:slow] = 0.0
    return sig


def sma_month(px: pd.Series, months: int = 10) -> pd.Series:
    """Faber (2007). At each month end, 1 if the index is above its 10 month end average."""
    me = px[_complete_month_end(px.index)]
    sig_me = (me > me.rolling(months).mean()).astype(float)
    sig_me[me.rolling(months).mean().isna()] = np.nan
    return sig_me.reindex(px.index).ffill().fillna(0.0)


def ewm_vol(returns: pd.Series, com: int = 60) -> pd.Series:
    """Annualized exponentially weighted vol, MOP (2012) style, using data up to each day."""
    var = (returns ** 2).ewm(com=com, min_periods=com).mean()
    return np.sqrt(var * ANN)


def tsmom(px: pd.Series, rf: pd.Series, lookback: int = 12, vol_target: float | None = None,
          cap: float = 1.0, long_only: bool = True, com: int = 60) -> pd.Series:
    """
    Time series momentum (Moskowitz, Ooi and Pedersen 2012). At each month
    end look at the past `lookback` months: asset total return minus T-bill
    return. Positive means long. Negative means cash (long_only) or short.

    With vol_target the size is vol_target / ewm vol, capped at `cap`.
    """
    mask = _complete_month_end(px.index)
    cash_idx = (1.0 + rf.reindex(px.index).fillna(0.0)).cumprod()
    me_px, me_cash = px[mask], cash_idx[mask]
    excess = me_px / me_px.shift(lookback) - me_cash / me_cash.shift(lookback)
    direction = np.sign(excess)
    if long_only:
        direction = direction.clip(lower=0.0)
    size = pd.Series(1.0, index=me_px.index)
    if vol_target is not None:
        vol = ewm_vol(px.pct_change(), com=com)[mask]
        size = (vol_target / vol).clip(upper=cap)
    sig_me = direction * size
    sig_me[excess.isna() | size.isna()] = np.nan
    return sig_me.reindex(px.index).ffill().fillna(0.0)


def breakout(px: pd.Series, n_entry: int = 100, n_exit: int | None = None) -> pd.Series:
    """
    Donchian style. Go in when the close beats the highest close of the past
    n_entry days, get out when it falls below the lowest close of the past
    n_exit days (n_entry // 2 by default).
    """
    n_exit = n_exit or n_entry // 2
    hi = px.rolling(n_entry).max().shift(1)
    lo = px.rolling(n_exit).min().shift(1)
    state = pd.Series(np.nan, index=px.index)
    state[px > hi] = 1.0
    state[px < lo] = 0.0
    return state.ffill().fillna(0.0)


def vol_target(signal: pd.Series, returns: pd.Series, target: float = 0.10,
               cap: float = 1.0, com: int = 60) -> pd.Series:
    """Scale any 0/1 signal by target / ewm vol, capped. Uses vol known at the signal close."""
    size = (target / ewm_vol(returns, com=com)).clip(upper=cap)
    return (signal * size).fillna(0.0)


def warmup_days(spec: dict) -> int:
    """Trading days of history a rule needs before its signal means anything (a bit generous for monthly rules)."""
    name = spec["rule"]
    if name in ("ma_crossover", "ema_crossover"):
        n = spec.get("slow", 200)
    elif name == "price_above_ma":
        n = spec.get("window", 200)
    elif name == "breakout":
        n = spec.get("n_entry", 100)
    elif name == "sma_month":
        n = 23 * (spec.get("months", 10) + 1)
    elif name == "tsmom":
        n = 23 * (spec.get("lookback", 12) + 1)
    else:
        n = 252
    if spec.get("vol_target") is not None:
        n = max(n, int(spec.get("com", 60)))
    return int(n)


RULES = {
    "ma_crossover": ma_crossover,
    "ema_crossover": ema_crossover,
    "price_above_ma": price_above_ma,
    "sma_month": sma_month,
    "tsmom": tsmom,
    "breakout": breakout,
}


def params(name: str) -> dict:
    """Parameter names and defaults of a rule, with the vol targeting options every rule accepts."""
    if name not in RULES:
        raise ValueError(f"unknown rule {name!r}, pick one of {', '.join(sorted(RULES))}")
    sig = inspect.signature(RULES[name])
    out = {k: v.default for k, v in list(sig.parameters.items())[1:] if v.default is not inspect.Parameter.empty}
    if name != "tsmom":
        out.update(vol_target=None, cap=1.0, com=60)
    return out


def check_spec(spec: dict) -> None:
    """Raise a readable error if the rule name or one of its parameters is wrong."""
    name = spec.get("rule")
    known = params(name)
    extra = [k for k in spec if k != "rule" and k not in known]
    if extra:
        raise ValueError(f"rule {name} has no parameter {extra[0]!r}, its parameters are {', '.join(known)}")
    if spec.get("cap") is not None and spec.get("vol_target") is None and name != "tsmom":
        raise ValueError("cap only means something together with vol_target")


def make_signal(spec: dict, px: pd.Series, rf: pd.Series) -> pd.Series:
    """Build a signal from a config dict like {"rule": "ma_crossover", "fast": 50, "slow": 200}."""
    check_spec(spec)
    spec = dict(spec)
    name = spec.pop("rule")
    vt = spec.pop("vol_target", None)
    if name == "tsmom":
        return tsmom(px, rf, vol_target=vt, **spec)
    cap, com = spec.pop("cap", 1.0), spec.pop("com", 60)
    sig = RULES[name](px, **spec)
    if vt is not None:
        sig = vol_target(sig, px.pct_change(), target=vt, cap=cap, com=com)
    return sig
