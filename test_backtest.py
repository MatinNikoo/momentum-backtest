"""Tests for backtest_core and robustness on made up data, so no internet is needed."""
import numpy as np
import pandas as pd
import backtest_core as bc

rng = np.random.default_rng(42)

# --- Fake prices: a random walk with drift, about 16 years ---
n = 4000
dates = pd.bdate_range("2010-01-01", periods=n)
daily = rng.normal(0.0004, 0.011, n)  # about 10% a year drift, 17% a year vol
prices = pd.Series(100 * np.cumprod(1 + daily), index=dates, name="SYN")

# --- 1. Positions: only 0/1, properly shifted (no lookahead) ---
pos = bc.ma_crossover_positions(prices, 50, 200)
assert set(pos.unique()) <= {0.0, 1.0}, "positions must be 0 or 1"
fast_ma = prices.rolling(50).mean()
slow_ma = prices.rolling(200).mean()
raw = (fast_ma > slow_ma).astype(float)
assert (pos.iloc[1:].values == raw.shift(1).fillna(0).iloc[1:].values).all(), \
    "positions must equal yesterday's signal (shift enforced)"
print("1. position logic + no-lookahead shift: OK")

# --- 2. Costs reduce returns, and zero cost equals gross ---
res_cost = bc.run_backtest(prices, pos, cost_bps=5.0)
res_free = bc.run_backtest(prices, pos, cost_bps=0.0)
total_cost = (1 + res_cost["strategy"]).prod()
total_free = (1 + res_free["strategy"]).prod()
assert total_cost < total_free, "costs must reduce compounded return"
gross_manual = (pos * prices.pct_change().fillna(0)).sum()
assert abs(res_free["strategy"].sum() - gross_manual) < 1e-12
print(f"2. costs applied correctly: OK (cost drag = {total_free - total_cost:.4f})")

# --- 3. Buy & hold column matches raw returns ---
assert np.allclose(res_cost["buy_hold"], prices.pct_change().fillna(0))
print("3. buy-and-hold benchmark: OK")

# --- 4. Metrics sane ---
summary = bc.summarize(res_cost)
assert summary.loc["Max Drawdown"].max() <= 0, "drawdown must be <= 0"
assert np.isfinite(summary.values.astype(float)).all(), "metrics must be finite"
print("4. metrics finite and sane: OK")
print(summary.round(3).to_string())

# --- 5. Grid search runs and ranks by Sharpe ---
grid = bc.grid_search(prices, [20, 50, 100], [100, 150, 200], cost_bps=5.0)
assert len(grid) > 0 and grid["sharpe"].is_monotonic_decreasing
print(f"5. grid search: OK ({len(grid)} combos, best = "
      f"{int(grid.iloc[0]['fast'])}/{int(grid.iloc[0]['slow'])})")

# --- 6. Known answer: always invested with no cost equals buy and hold ---
always = pd.Series(1.0, index=prices.index)
res_all = bc.run_backtest(prices, always, cost_bps=0.0)
assert np.allclose(res_all["strategy"], res_all["buy_hold"])
print("6. known answer (always in = buy and hold): OK")

# --- 7. Robustness checks behave on data with no skill in it ---
import robustness as rb

# One noise sample can land anywhere, so average over many of them
pbos = [rb.pbo_cscv(pd.DataFrame(rng.normal(0, 0.01, (1500, 20))), n_blocks=10)["pbo"] for _ in range(20)]
assert 0.4 < np.mean(pbos) < 0.6, "with pure noise, picking the best rule should be a coin flip"
skill = rng.normal(0, 0.01, (1500, 20)); skill[:, 0] += 0.002   # one rule really is better
pbo_skill = rb.pbo_cscv(pd.DataFrame(skill), n_blocks=10)["pbo"]
assert pbo_skill < 0.1, "when one rule is truly better, PBO should be near 0"
noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 30)))
print(f"7. PBO: about 0.5 on noise ({np.mean(pbos):.2f}), near 0 with real skill ({pbo_skill:.2f}): OK")

assert rb.expected_max_sharpe(0.01, 100) > rb.expected_max_sharpe(0.01, 10) > 0
d = rb.deflated_sharpe(noise.iloc[:, 0], (noise.mean() / noise.std()).values)
assert 0 <= d["dsr"] <= 1
print("8. deflated Sharpe: more trials means a higher bar: OK")

a = rng.normal(0.0005, 0.01, 1500)
b1 = rb.bootstrap_sharpe_diff(a, a, n_boot=200, seed=1)
assert np.allclose(b1, 0), "same series vs itself should give a zero gap"
c = rng.normal(0, 0.01, 1500)
assert np.array_equal(rb.bootstrap_sharpe_diff(a, c, n_boot=200, seed=1),
                      rb.bootstrap_sharpe_diff(a, c, n_boot=200, seed=1)), "same seed, same result"
print("9. bootstrap: zero gap for identical series, and reproducible: OK")

warm = rb.strategy_returns(prices, 50, 200)
assert len(warm) == len(prices) and warm.iloc[:200].abs().sum() == 0, "no position before the slow average exists"
print("10. warm up: flat until the 200 day average exists: OK")

print("\nALL TESTS PASSED")
