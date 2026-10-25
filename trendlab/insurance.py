"""
Insurance accounting. Trend following is treated like an insurance policy
on the asset. The premium is what it gives up in normal months, the payout
is what it saves in bear markets. Also regime labels for the monthly data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import stationary_bootstrap_matrix


def episodes(r, threshold: float = 0.20) -> list[dict]:
    """
    Peak to trough drawdowns of the return series r (monthly) that reach
    `threshold`. Positions are return indices: the bear months are
    peak..trough-1 and the recovery months are trough..recovered-1.
    """
    r = np.asarray(r, dtype=float)
    w = np.r_[1.0, np.cumprod(1.0 + r)]
    out = []
    peak = 0
    t = 1
    n = len(w)
    while t < n:
        if w[t] >= w[peak]:
            peak = t
            t += 1
            continue
        # in a drawdown from `peak`, find where it ends
        end = t
        while end < n and w[end] < w[peak]:
            end += 1
        seg = w[peak:end]
        trough = peak + int(np.argmin(seg))
        depth = 1.0 - w[trough] / w[peak]
        if depth >= threshold:
            out.append({"peak": peak, "trough": trough,
                        "recovered": end if end < n else None, "depth": float(depth)})
        peak = end if end < n else peak
        t = end + 1 if end < n else n
    return out


def swing_episodes(r, threshold: float = 0.20) -> list[dict]:
    """
    Bear markets the way people usually date them: a fall of `threshold` from
    a local peak, ending at the lowest point before a rise of `threshold` from
    it. Unlike `episodes`, the old all time high doesn't have to be regained
    first, so a second bear market inside a long slump still counts (1937 to
    1938 in the US, for example). Same index convention as `episodes`.
    """
    r = np.asarray(r, dtype=float)
    w = np.r_[1.0, np.cumprod(1.0 + r)]
    out = []
    bear = False
    peak = trough = 0
    for t in range(1, len(w)):
        if not bear:
            if w[t] > w[peak]:
                peak = t
            elif w[t] <= (1 - threshold) * w[peak]:
                bear, trough = True, t
        else:
            if w[t] < w[trough]:
                trough = t
            elif w[t] >= (1 + threshold) * w[trough]:
                out.append({"peak": peak, "trough": trough, "recovered": None,
                            "depth": float(1 - w[trough] / w[peak])})
                bear, peak = False, t
    if bear:
        out.append({"peak": peak, "trough": trough, "recovered": None, "depth": float(1 - w[trough] / w[peak])})
    return out


def labels_from_episodes(n: int, eps: list[dict]) -> np.ndarray:
    """0 normal, 1 bear (peak to trough), 2 recovery (trough back to the old peak)."""
    lab = np.zeros(n, dtype=int)
    for e in eps:
        lab[e["peak"]:e["trough"]] = 1
        rec_end = e["recovered"] if e["recovered"] is not None else n
        lab[e["trough"]:rec_end] = 2
    return lab


def account(strat, bench, eps, n_boot=2000, mean_block=12, seed=0, notional=100_000.0):
    """
    Premium and payout of strat relative to bench, in log return and in
    dollars per `notional`. Bootstrap intervals are 90%.
    """
    s = np.asarray(strat, dtype=float)
    b = np.asarray(bench, dtype=float)
    g = np.log1p(s) - np.log1p(b)
    n = len(g)
    bear = np.zeros(n, dtype=bool)
    rows = []
    for e in eps:
        sl = slice(e["peak"], e["trough"])
        bear[sl] = True
        ws = notional * np.prod(1 + s[sl])
        wb = notional * np.prod(1 + b[sl])
        rows.append({"peak": e["peak"], "trough": e["trough"], "months": e["trough"] - e["peak"],
                     "depth": e["depth"], "payout_log": float(g[sl].sum()),
                     "dollars_saved": float(ws - wb)})

    def stat(gg, bb):
        normal = ~bb
        prem = 12 * gg[normal].mean() if normal.any() else np.nan
        bear_rate = 12 * gg[bb].mean() if bb.any() else np.nan
        return prem, bear_rate, 12 * gg.mean()

    point = stat(g, bear)
    idx = stationary_bootstrap_matrix(n, mean_block, n_boot, seed)
    draws = np.array([stat(g[ix], bear[ix]) for ix in idx])
    lo, hi = np.nanquantile(draws, [0.05, 0.95], axis=0)

    pay = np.array([r["payout_log"] for r in rows])
    dollars = np.array([r["dollars_saved"] for r in rows])
    wins = int((pay > 0).sum())
    sign_p = float(sps.binomtest(wins, len(pay), 0.5, alternative="greater").pvalue) if len(pay) else np.nan
    # spread of the average episode payout in dollars, resampling whole episodes
    rng = np.random.default_rng(seed)
    if len(dollars) > 1:
        ep_draws = np.array([dollars[rng.integers(len(dollars), size=len(dollars))].mean()
                             for _ in range(n_boot)])
        ep_lo, ep_hi = np.quantile(ep_draws, [0.05, 0.95])
    else:
        ep_lo = ep_hi = np.nan

    to_dollars = lambda x: notional * np.expm1(x)
    return {
        "premium_log_per_year": point[0], "premium_ci": [lo[0], hi[0]],
        "bear_gap_log_per_year": point[1], "bear_gap_ci": [lo[1], hi[1]],
        "net_log_per_year": point[2], "net_ci": [lo[2], hi[2]],
        "premium_dollars_per_year": to_dollars(point[0]),
        "premium_dollars_ci": [to_dollars(lo[0]), to_dollars(hi[0])],
        "episodes": rows,
        "episodes_won": wins, "n_episodes": len(rows), "sign_test_p": sign_p,
        "mean_dollars_saved_per_episode": float(dollars.mean()) if len(dollars) else np.nan,
        "mean_dollars_saved_ci": [float(ep_lo), float(ep_hi)],
        "bear_months": int(bear.sum()), "normal_months": int((~bear).sum()),
    }


def regime_gaps(gap, labels: dict, n_boot=2000, mean_block=12, seed=0):
    """Average monthly gap inside each regime (boolean masks), with a 90% block bootstrap interval."""
    gap = np.asarray(gap, dtype=float)
    n = len(gap)
    idx = stationary_bootstrap_matrix(n, mean_block, n_boot, seed)
    out = {}
    for name, mask in labels.items():
        m = np.asarray(mask, dtype=bool)
        if m.sum() == 0:
            continue
        draws = np.array([gap[ix][m[ix]].mean() if m[ix].any() else np.nan for ix in idx])
        lo, hi = np.nanquantile(draws, [0.05, 0.95])
        out[name] = {"months": int(m.sum()), "mean_gap": float(gap[m].mean()),
                     "ci": [float(lo), float(hi)]}
    return out


def month_labels(index: pd.PeriodIndex, bh_monthly, daily_asset: pd.Series | None,
                 tbill_monthly, long_yield: pd.Series, threshold: float) -> dict:
    """The regime masks from the protocol, all lined up with `index`."""
    bh = np.asarray(bh_monthly, dtype=float)
    eps = episodes(bh, threshold)
    lab = labels_from_episodes(len(bh), eps)
    crash_cut = np.quantile(bh, 0.05)
    out = {
        "bear": lab == 1,
        "recovery": lab == 2,
        "crash_months": bh <= crash_cut,
    }
    if daily_asset is not None:
        rv = daily_asset.groupby(daily_asset.index.to_period("M")).std().reindex(index)
        cut = np.nanquantile(rv.values, 1 / 3)
        out["calm"] = (rv.values <= cut)
    tb = np.asarray(tbill_monthly, dtype=float) * 12
    med = np.median(tb)
    out["high_rates"] = tb > med
    out["low_rates"] = tb <= med
    ly = long_yield.reindex(index)
    chg = (ly - long_yield.shift(12).reindex(index)).values
    out["rising_long_rates"] = chg > 0
    out["falling_long_rates"] = chg <= 0
    return out
