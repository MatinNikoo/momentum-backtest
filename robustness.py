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
                        out of sample (Bailey, Borwein, Lopez de Prado, Zhu 2014)
  cost_sensitivity      how fast the strategy dies as trading gets pricier

Signals are always built on the full price history and then sliced, so the
moving averages are already warmed up on the first test day. Nothing uses
data from after the day it trades.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

import backtest_core as bc

TRADING_DAYS = bc.TRADING_DAYS


def strategy_returns(prices: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
    """Daily net returns of one crossover rule over the whole price history."""
    pos = bc.ma_crossover_positions(prices, fast, slow)
    return bc.run_backtest(prices, pos, cost_bps)["strategy"]


def returns_matrix(prices: pd.Series, pairs, cost_bps: float = 5.0) -> pd.DataFrame:
    """One column of daily returns per (fast, slow) pair."""
    return pd.DataFrame(
        {f"{f}/{s}": strategy_returns(prices, f, s, cost_bps) for f, s in pairs}
    )


def make_pairs(fast_list, slow_list):
    return [(f, s) for f in fast_list for s in slow_list if f < s]


# ---------------------------------------------------------------------------
# 1. Parameter sensitivity
# ---------------------------------------------------------------------------
def sensitivity_grid(prices, split, fast_list, slow_list, cost_bps=5.0) -> pd.DataFrame:
    """In sample and out of sample Sharpe for every pair, next to buy and hold."""
    bh = prices.pct_change().fillna(0.0)
    bh_in, bh_out = bc.sharpe(bh[bh.index < split]), bc.sharpe(bh[bh.index >= split])
    rows = []
    for f, s in make_pairs(fast_list, slow_list):
        r = strategy_returns(prices, f, s, cost_bps)
        rows.append({
            "fast": f, "slow": s,
            "sharpe_in": bc.sharpe(r[r.index < split]),
            "sharpe_out": bc.sharpe(r[r.index >= split]),
        })
    out = pd.DataFrame(rows)
    out["bh_in"], out["bh_out"] = bh_in, bh_out
    out["beats_bh_out"] = out["sharpe_out"] > bh_out
    return out


# ---------------------------------------------------------------------------
# 2. Walk forward
# ---------------------------------------------------------------------------
def walk_forward(prices, fast_list, slow_list, first_year=2008, min_years=5, cost_bps=5.0):
    """
    Every January, pick the pair with the best Sharpe on all data before that
    year, then hold it for the year. Stitch the years together.

    This is closer to what someone would actually do than a single split,
    because the choice keeps getting updated as new data comes in.
    """
    pairs = make_pairs(fast_list, slow_list)
    R = returns_matrix(prices, pairs, cost_bps)
    bh = prices.pct_change().fillna(0.0)
    years = sorted(set(prices.index.year))
    picks, chunks = [], []
    for y in years:
        if y < first_year:
            continue
        past = R[R.index.year < y]
        if past.index.year.nunique() < min_years:
            continue
        sr = past.apply(bc.sharpe)
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
def stationary_bootstrap_idx(n, mean_block, rng):
    """Politis and Romano (1994) stationary bootstrap indices."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def bootstrap_sharpe_diff(strat, bench, n_boot=2000, mean_block=20, seed=0):
    """
    Resample whole days (both series together, so their link is kept) in
    random blocks that average `mean_block` days, which keeps volatility
    clusters intact. Returns the bootstrap distribution of
    Sharpe(strategy) minus Sharpe(benchmark).
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(strat), np.asarray(bench)
    n = len(a)
    k = np.sqrt(TRADING_DAYS)
    out = np.empty(n_boot)
    for j in range(n_boot):
        ix = stationary_bootstrap_idx(n, mean_block, rng)
        x, y = a[ix], b[ix]
        out[j] = k * (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1))
    return out


# ---------------------------------------------------------------------------
# 4. Deflated Sharpe ratio
# ---------------------------------------------------------------------------
def expected_max_sharpe(sr_var, n_trials):
    """Expected best Sharpe out of n_trials rules with no real skill."""
    g = 0.5772156649  # Euler-Mascheroni constant
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return np.sqrt(sr_var) * ((1 - g) * z1 + g * z2)


def deflated_sharpe(returns, trial_sharpes):
    """
    Probability that the chosen rule's true Sharpe is above zero, after
    accounting for how many rules were tried and for fat tails.
    Sharpes here are per day, not annualized, which is what the formula expects.
    """
    r = np.asarray(returns)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(np.var(trial_sharpes, ddof=1), len(trial_sharpes))
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return {"sr_daily": sr, "sr0_daily": sr0, "dsr": stats.norm.cdf(z),
            "psr_vs_zero": stats.norm.cdf(sr * np.sqrt(T - 1) / denom)}


# ---------------------------------------------------------------------------
# 5. Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------
def pbo_cscv(R: pd.DataFrame, n_blocks=16):
    """
    Cut the period into n_blocks pieces. For every way of choosing half of
    them as a pretend "in sample", pick the best rule there and see where it
    ranks on the other half. PBO is the share of splits where the winner
    lands in the bottom half. Around 0.5 means picking the best rule is no
    better than picking at random.
    """
    X = R.values
    T, N = X.shape
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
    # per block sums so each split is fast
    s1 = np.array([X[b].sum(0) for b in blocks])
    s2 = np.array([(X[b] ** 2).sum(0) for b in blocks])
    cnt = np.array([len(b) for b in blocks])[:, None]

    def sharpe_of(sel):
        n = cnt[sel].sum()
        m = s1[sel].sum(0) / n
        v = s2[sel].sum(0) / n - m ** 2
        return m / np.sqrt(np.maximum(v, 1e-18))

    logits = []
    all_b = set(range(n_blocks))
    for train in combinations(range(n_blocks), n_blocks // 2):
        test = sorted(all_b - set(train))
        sr_tr, sr_te = sharpe_of(list(train)), sharpe_of(test)
        best = int(np.argmax(sr_tr))
        rank = stats.rankdata(sr_te)[best] / (N + 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits,
            "n_splits": comb(n_blocks, n_blocks // 2)}


# ---------------------------------------------------------------------------
# 6. Costs
# ---------------------------------------------------------------------------
def cost_sensitivity(prices, split, fast, slow, costs=(0, 5, 10, 20, 50)):
    bh = prices.pct_change().fillna(0.0)
    bh_out = bc.sharpe(bh[bh.index >= split])
    rows = []
    for c in costs:
        r = strategy_returns(prices, fast, slow, c)
        rows.append({"cost_bps": c, "sharpe_out": bc.sharpe(r[r.index >= split]), "bh_out": bh_out})
    return pd.DataFrame(rows)
