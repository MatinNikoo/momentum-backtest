"""
Statistics used to decide how much to trust a result: block bootstrap,
deflated Sharpe, PBO, White's Reality Check, Hansen's SPA, and the
Treynor-Mazuy convexity fit.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# stationary bootstrap (Politis and Romano 1994)
# ---------------------------------------------------------------------------
def stationary_bootstrap_idx(n, mean_block, rng):
    """One path of stationary bootstrap indices."""
    idx = np.empty(n, dtype=np.int64)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def stationary_bootstrap_matrix(n, mean_block, n_boot, seed=0, length=None):
    """n_boot paths at once, shape (n_boot, length). Same idea, looped over time instead of paths."""
    rng = np.random.default_rng(seed)
    length = length or n
    p = 1.0 / mean_block
    out = np.empty((n_boot, length), dtype=np.int64)
    cur = rng.integers(n, size=n_boot)
    for t in range(length):
        out[:, t] = cur
        jump = rng.random(n_boot) < p
        cur = np.where(jump, rng.integers(n, size=n_boot), (cur + 1) % n)
    return out


# ---------------------------------------------------------------------------
# deflated Sharpe ratio (Bailey and Lopez de Prado 2014)
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
    counting how many rules were tried and allowing for fat tails. Sharpes
    are per period (per day for daily returns, per month for monthly), not annualized.
    """
    r = np.asarray(returns)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(np.var(trial_sharpes, ddof=1), len(trial_sharpes))
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return {"sr_per_period": sr, "sr0_per_period": sr0, "dsr": stats.norm.cdf(z),
            "psr_vs_zero": stats.norm.cdf(sr * np.sqrt(T - 1) / denom)}


# ---------------------------------------------------------------------------
# probability of backtest overfitting (Bailey, Borwein, Lopez de Prado, Zhu 2017)
# ---------------------------------------------------------------------------
def pbo_cscv(R, n_blocks=16):
    """
    Cut the sample into n_blocks pieces. For every way to pick half of them
    as a pretend in sample, take the best rule there and see where it ranks
    on the rest. PBO is the share of splits where it lands in the bottom half.
    """
    X = np.asarray(R, dtype=float)
    T, N = X.shape
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]
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
# White (2000) Reality Check and Hansen (2005) SPA
# ---------------------------------------------------------------------------
def reality_check_spa(D, n_boot=2000, mean_block=12, seed=0):
    """
    D is (n periods, K rules) of performance differentials vs a benchmark,
    so positive means the rule did better. The null is that no rule has a
    positive mean. Returns both p-values.
    """
    D = np.asarray(D, dtype=float)
    n, K = D.shape
    dbar = D.mean(0)
    idx = stationary_bootstrap_matrix(n, mean_block, n_boot, seed)
    boot_means = np.stack([D[ix].mean(0) for ix in idx])  # (n_boot, K)

    # Reality Check, not studentized
    v = np.sqrt(n) * dbar.max()
    v_star = np.sqrt(n) * (boot_means - dbar).max(axis=1)
    p_rc = float((v_star >= v).mean())

    # SPA, studentized and with the consistent recentering
    omega = np.sqrt(n) * boot_means.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    t_spa = max(float((np.sqrt(n) * dbar / omega).max()), 0.0)
    keep = np.sqrt(n) * dbar / omega >= -np.sqrt(2 * np.log(np.log(n)))
    # rules that are clearly bad keep their negative mean so they can't drive the max
    mu_c = np.where(keep, 0.0, dbar)
    z = np.sqrt(n) * (boot_means - dbar + mu_c) / omega
    t_star = np.maximum(z.max(axis=1), 0.0)
    p_spa = float((t_star >= t_spa).mean()) if t_spa > 0 else 1.0
    best = int(np.argmax(dbar))
    return {"p_reality_check": p_rc, "p_spa": p_spa, "best_rule": best,
            "best_mean_diff": float(dbar[best]), "n_rules": K, "n_periods": n}


# ---------------------------------------------------------------------------
# convexity
# ---------------------------------------------------------------------------
def treynor_mazuy(y, x):
    """Fit y = a + b x + g x^2. Returns (a, b, g)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x, x * x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return tuple(float(c) for c in coef)


def bootstrap_ci(stat_fn, arrays, n_boot=2000, mean_block=12, seed=0, level=0.90):
    """
    Stationary block bootstrap of a statistic of several aligned arrays,
    which are resampled together. Returns (point, lo, hi, draws).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    point = stat_fn(*arrays)
    idx = stationary_bootstrap_matrix(n, mean_block, n_boot, seed)
    draws = np.array([stat_fn(*[a[ix] for a in arrays]) for ix in idx], dtype=float)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(draws, [a, 1 - a])
    return float(point), float(lo), float(hi), draws
