"""Sanity tests for backtest_core using synthetic data (no network needed)."""
import numpy as np
import pandas as pd
import backtest_core as bc

rng = np.random.default_rng(42)

# --- Synthetic price series: geometric random walk with drift, ~16 years ---
n = 4000
dates = pd.bdate_range("2010-01-01", periods=n)
daily = rng.normal(0.0004, 0.011, n)  # ~10%/yr drift, ~17%/yr vol
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

# --- 2. Costs reduce returns; zero-cost equals gross ---
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

# --- 6. Known-answer test: always-invested with 0 cost == buy & hold ---
always = pd.Series(1.0, index=prices.index)
res_all = bc.run_backtest(prices, always, cost_bps=0.0)
assert np.allclose(res_all["strategy"], res_all["buy_hold"])
print("6. known-answer (always-in = buy&hold): OK")

print("\nALL TESTS PASSED")
