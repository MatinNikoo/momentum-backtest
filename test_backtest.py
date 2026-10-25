"""
Tests for backtest_core and robustness on made up data, so no internet is needed.

    python test_backtest.py     # prints each check
    pytest                      # runs these plus everything in tests/
"""
import numpy as np
import pandas as pd

import backtest_core as bc
import robustness as rb

rng = np.random.default_rng(42)

# --- Fake prices: a random walk with drift, about 16 years ---
n = 4000
dates = pd.bdate_range("2010-01-01", periods=n)
daily = rng.normal(0.0004, 0.011, n)  # about 10% a year drift, 17% a year vol
prices = pd.Series(100 * np.cumprod(1 + daily), index=dates, name="SYN")
rf = pd.Series(0.0001, index=dates)  # about 2.5% a year on cash


def test_positions_trade_the_day_after_the_signal():
    pos = bc.ma_crossover_positions(prices, 50, 200)
    assert set(pos.unique()) <= {0.0, 1.0}, "positions must be 0 or 1"
    raw = (prices.rolling(50).mean() > prices.rolling(200).mean()).astype(float)
    # signal at day t's close, trade at day t+1's close, first return earned is day t+2's
    assert (pos.values == raw.shift(2).fillna(0).values).all()
    # delay=0 is what the first version did: trade at the signal's own close
    assert (bc.ma_crossover_positions(prices, 50, 200, delay=0).values == raw.shift(1).fillna(0).values).all()


def test_costs_reduce_returns_and_zero_cost_is_gross():
    pos = bc.ma_crossover_positions(prices, 50, 200)
    res_cost = bc.run_backtest(prices, pos, cost_bps=5.0)
    res_free = bc.run_backtest(prices, pos, cost_bps=0.0)
    assert (1 + res_cost["strategy"]).prod() < (1 + res_free["strategy"]).prod()
    gross_manual = (pos * prices.pct_change().fillna(0)).sum()
    assert abs(res_free["strategy"].sum() - gross_manual) < 1e-12
    n_switch = pos.diff().abs().sum()
    assert np.isclose(res_free["strategy"].sum() - res_cost["strategy"].sum(), n_switch * 5e-4)


def test_cash_earns_the_tbill_rate():
    pos = bc.ma_crossover_positions(prices, 50, 200)
    with_rf = bc.run_backtest(prices, pos, cost_bps=0.0, rf=rf)["strategy"]
    no_rf = bc.run_backtest(prices, pos, cost_bps=0.0)["strategy"]
    assert np.allclose(with_rf - no_rf, (1 - pos) * 0.0001)
    flat = pd.Series(0.0, index=dates)
    assert np.allclose(bc.run_backtest(prices, flat, rf=rf)["strategy"], 0.0001)


def test_buy_and_hold_column_and_known_answer():
    res = bc.run_backtest(prices, pd.Series(1.0, index=dates), cost_bps=0.0, rf=rf)
    assert np.allclose(res["buy_hold"], prices.pct_change().fillna(0))
    assert np.allclose(res["strategy"], res["buy_hold"])  # always in, no cost, is buy and hold


def test_metrics_finite_and_sharpe_over_tbills():
    res = bc.run_backtest(prices, bc.ma_crossover_positions(prices, 50, 200), rf=rf)
    summary = bc.summarize(res, rf=rf)
    assert "Sharpe" in summary.index and summary.loc["Max Drawdown"].max() <= 0
    assert np.isfinite(summary.values.astype(float)).all()
    r = res["buy_hold"]
    assert np.isclose(bc.sharpe(r, rf=rf), bc.sharpe(r, rf_annual=0.0001 * 252))


def test_grid_search_ranks_by_sharpe():
    grid = bc.grid_search(prices, [20, 50, 100], [100, 150, 200], cost_bps=5.0, rf=rf)
    assert len(grid) > 0 and grid["sharpe"].is_monotonic_decreasing


def test_robustness_checks_on_noise():
    pbos = [rb.pbo_cscv(pd.DataFrame(rng.normal(0, 0.01, (1500, 20))), n_blocks=10)["pbo"] for _ in range(20)]
    assert 0.4 < np.mean(pbos) < 0.6, "with pure noise, picking the best rule should be a coin flip"
    skill = rng.normal(0, 0.01, (1500, 20))
    skill[:, 0] += 0.002
    assert rb.pbo_cscv(pd.DataFrame(skill), n_blocks=10)["pbo"] < 0.1
    assert rb.expected_max_sharpe(0.01, 100) > rb.expected_max_sharpe(0.01, 10) > 0
    a = rng.normal(0.0005, 0.01, 1500)
    assert np.allclose(rb.bootstrap_sharpe_diff(a, a, n_boot=200, seed=1), 0)
    c = rng.normal(0, 0.01, 1500)
    assert np.array_equal(rb.bootstrap_sharpe_diff(a, c, n_boot=200, seed=1),
                          rb.bootstrap_sharpe_diff(a, c, n_boot=200, seed=1))


def test_warm_up():
    warm = rb.strategy_returns(prices, 50, 200, rf=rf)
    # no position until the 200 day average exists plus the one day delay, only cash interest
    assert len(warm) == len(prices) and np.allclose(warm.iloc[:201], 0.0001)


def test_load_rf_matches_snapshot():
    idx = pd.bdate_range("1990-01-02", "1990-12-31")
    got = bc.load_rf(idx)
    assert 0.07 < got.sum() < 0.09  # T-bills paid about 8% in 1990


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("\nALL TESTS PASSED")
