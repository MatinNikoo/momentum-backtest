"""The engine: timing, costs, cash and portfolios against hand worked examples."""
import numpy as np
import pandas as pd
import pytest

from trendlab import engine as en

IDX = pd.bdate_range("2024-01-01", periods=4)


def test_delay_is_a_shift_of_delay_plus_one():
    sig = pd.Series([1.0, 1.0, 0.0, 0.0], index=IDX)
    assert en.positions_from_signal(sig, 0).tolist() == [0, 1, 1, 0]
    assert en.positions_from_signal(sig, 1).tolist() == [0, 0, 1, 1]
    assert en.positions_from_signal(sig, 2).tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError):
        en.positions_from_signal(sig, -1)


def test_hand_worked_returns_costs_and_cash():
    r = pd.Series([0.0, 0.10, -0.05, 0.02], index=IDX)
    rf = pd.Series(0.001, index=IDX)
    pos = pd.Series([0.0, 1.0, 1.0, 0.0], index=IDX)
    out = en.run(r, pos, rf, cost_bps=10)
    # day 0 in cash, day 1 buys (pays 10 bps), day 2 holds, day 3 sells (pays 10 bps) and earns cash
    assert np.allclose(out["strategy"].values, [0.001, 0.099, -0.05, 0.0])
    assert np.allclose(out["turnover"].values, [0, 1, 0, 1])
    assert np.isclose(out["cost"].sum(), 0.002)


def test_partial_and_levered_positions():
    r = pd.Series([0.02, 0.02], index=IDX[:2])
    rf = pd.Series(0.001, index=IDX[:2])
    half = en.run(r, pd.Series([0.5, 0.5], index=IDX[:2]), rf, cost_bps=0)
    assert np.allclose(half["strategy"], 0.5 * 0.02 + 0.5 * 0.001)
    lev = en.run(r, pd.Series([2.0, 2.0], index=IDX[:2]), rf, cost_bps=0)
    assert np.allclose(lev["strategy"], 2 * 0.02 - 1 * 0.001)  # borrow the extra 1 at the T-bill rate
    short = en.run(r, pd.Series([-1.0, -1.0], index=IDX[:2]), rf, cost_bps=0)
    assert np.allclose(short["strategy"], -0.02 + 2 * 0.001)


def test_first_day_entry_pays_cost():
    r = pd.Series([0.01, 0.01], index=IDX[:2])
    out = en.run(r, pd.Series([1.0, 1.0], index=IDX[:2]), None, cost_bps=5)
    assert np.isclose(out["strategy"].iloc[0], 0.01 - 0.0005)


def test_always_in_with_no_cost_is_buy_and_hold(fake):
    r, rf = fake
    out = en.run(r, pd.Series(1.0, index=r.index), rf, cost_bps=0)
    assert np.allclose(out["strategy"], r)


def test_portfolio_rebalances_at_month_end():
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01"])
    a = pd.Series([0.10, 0.0, 0.10], index=idx)
    b = pd.Series([0.0, 0.10, 0.0], index=idx)
    p = en.run_portfolio({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    # day 1: 0.5*1.1 + 0.5 = 1.05. day 2: 0.55 + 0.55 = 1.10. February starts back at 50/50
    assert np.allclose(p.values, [0.05, 1.10 / 1.05 - 1, 0.05])


def test_total_return_index_round_trip(fake):
    r, _ = fake
    px = en.total_return_index(r)
    assert np.allclose(px.pct_change().iloc[1:], r.iloc[1:])
