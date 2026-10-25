"""Metrics, matching weights and the bond pricer against hand computed answers."""
import numpy as np
import pandas as pd

from trendlab import benchmarks as bm
from trendlab import metrics as mt
from data_sources import bond_returns_from_yields, par_bond_price


def test_cagr_and_drawdown_by_hand():
    assert np.isclose(mt.cagr([0.10, -0.10], periods=1), 0.99 ** 0.5 - 1)
    assert np.isclose(mt.max_dd_depth([-0.5, 0.2]), 0.5)  # a loss in the very first month counts
    assert np.isclose(mt.max_dd_depth([0.1, -0.2, 0.05, -0.1]), 1 - (0.8 * 1.05 * 0.9))
    assert mt.max_dd_depth([0.01, 0.02]) == 0.0


def test_worst_window_cvar_sharpe():
    r = [0.1, -0.2, -0.1, 0.3]
    assert np.isclose(mt.worst_window(r, 2), 0.8 * 0.9 - 1)
    assert np.isclose(mt.cvar(np.arange(100) / 100.0, 0.05), np.mean(np.arange(5) / 100.0))
    x = np.array([0.01, 0.03, 0.02, 0.00])
    assert np.isclose(mt.sharpe(x, 0.01, periods=12), (x - 0.01).mean() / (x - 0.01).std(ddof=1) * np.sqrt(12))


def test_to_monthly_compounds():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01"])
    m = mt.to_monthly(pd.Series([0.1, 0.1, -0.5], index=idx))
    assert np.allclose(m.values, [0.21, -0.5])
    q = mt.to_quarterly(pd.Series([0.1, 0.1, 0.1, 0.1], index=pd.period_range("2024-01", periods=4, freq="M")))
    assert np.allclose(q.values, [1.1 ** 3 - 1, 0.1])


def test_vol_and_return_matching_hit_their_targets():
    rng = np.random.default_rng(1)
    a = rng.normal(0.008, 0.045, 600)
    c = np.full(600, 0.003) + rng.normal(0, 0.0005, 600)
    target = 0.6 * a + 0.4 * c + rng.normal(0, 0.01, 600)
    w = bm.vol_matched_w(target, a, c)
    assert np.isclose(np.std(bm.mix(a, c, w), ddof=1), np.std(target, ddof=1), rtol=1e-9)
    w2, flag = bm.return_matched_w(target, a, c)
    assert flag in ("", "levered")
    assert np.isclose(mt.cagr(bm.mix(a, c, w2)), mt.cagr(target), atol=1e-9)


def test_return_matching_flags():
    a = np.full(120, -0.01)  # an asset that only loses
    c = np.full(120, 0.002)
    w, flag = bm.return_matched_w(np.full(120, 0.004), a, c)
    assert flag == "unreachable" and np.isnan(w)
    assert bm.best_growth_w(a, c) == 0.0
    w, flag = bm.return_matched_w(np.full(120, 0.0), np.full(120, 0.01), c)
    assert flag == "below_tbills" and w == 0.0


def test_par_bond_prices():
    assert np.isclose(par_bond_price(0.05, 0.05, 10), 1.0)
    assert par_bond_price(0.06, 0.05, 10) < 1.0 < par_bond_price(0.04, 0.05, 10)
    assert np.isclose(par_bond_price(0.0, 0.02, 10), 1.2)  # zero yield: face plus all coupons
    # modified duration of a par bond, y bond equivalent, semiannual
    y, M, h = 0.05, 10.0, 1e-5
    dur = -(par_bond_price(y + h, y, M) - par_bond_price(y - h, y, M)) / (2 * h)
    assert np.isclose(dur, (1 - (1 + y / 2) ** (-2 * M)) / y, rtol=1e-6)


def test_bond_returns_constant_yield_is_just_carry():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05"])
    r = bond_returns_from_yields(pd.Series(0.04, index=idx))
    # repricing at the same yield with a shorter maturity is still par, so only the coupon accrues
    assert np.allclose(r.values, [0.04 * 1 / 365, 0.04 * 3 / 365], atol=1e-12)


def test_bond_return_sign_follows_yield_move():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = bond_returns_from_yields(pd.Series([0.04, 0.05, 0.04], index=idx))
    assert r.iloc[0] < -0.07 and r.iloc[1] > 0.07  # about 8 years of duration times 1 point


def test_daily_rf_rescaled_to_the_monthly_rate():
    from data_sources import rescale_daily_rf
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    daily = pd.Series(0.0002, index=idx)  # rounded, so it compounds to the wrong monthly number
    monthly = pd.Series([0.004], index=pd.PeriodIndex(["2024-01"], freq="M"))
    out = rescale_daily_rf(daily, monthly)
    jan = out[out.index.month == 1]
    assert np.isclose((1 + jan).prod() - 1, 0.004)
    assert np.allclose(out[out.index.month == 2], 0.0002)  # no monthly number yet, left alone
