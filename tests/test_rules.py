"""Rules on series where the right answer is obvious."""
import numpy as np
import pandas as pd

from trendlab import rules as rl

IDX = pd.bdate_range("2010-01-01", periods=800)


def test_rules_on_a_steady_rise_and_fall():
    up = pd.Series(np.linspace(100, 300, 800), index=IDX)
    down = pd.Series(np.linspace(300, 100, 800), index=IDX)
    rf = pd.Series(0.0001, index=IDX)
    assert rl.ma_crossover(up, 50, 200).iloc[-1] == 1 and rl.ma_crossover(down, 50, 200).iloc[-1] == 0
    assert rl.ma_crossover(up, 50, 200).iloc[:199].sum() == 0  # nothing before the slow average exists
    assert rl.sma_month(up, 10).iloc[-1] == 1 and rl.sma_month(down, 10).iloc[-1] == 0
    assert rl.tsmom(up, rf).iloc[-1] == 1 and rl.tsmom(down, rf).iloc[-1] == 0
    assert rl.tsmom(down, rf, long_only=False).iloc[-1] == -1


def test_tsmom_needs_to_beat_tbills():
    px = pd.Series(np.linspace(100, 101, 800), index=IDX)  # about 0.3% a year
    rf = pd.Series(0.0004, index=IDX)  # about 10% a year
    assert rl.tsmom(px, rf).iloc[-1] == 0


def test_monthly_rules_only_change_on_month_ends():
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 800)), index=IDX)
    sig = rl.sma_month(px, 10)
    changes = sig.index[np.r_[False, np.diff(sig.values) != 0]]
    month_ends = set(IDX[rl.month_end_mask(IDX)])
    assert len(changes) > 0 and all(d in month_ends for d in changes)


def test_partial_last_month_is_not_a_month_end():
    idx = pd.bdate_range("2024-01-01", "2024-03-14")
    assert not rl._complete_month_end(idx)[-1]
    idx2 = pd.bdate_range("2024-01-01", "2024-03-29")
    assert rl._complete_month_end(idx2)[-1]


def test_breakout_state_machine():
    vals = [10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8]
    px = pd.Series(vals, index=pd.bdate_range("2024-01-01", periods=len(vals)), dtype=float)
    sig = rl.breakout(px, n_entry=4, n_exit=2)
    # in once it beats the last 4 closes, out once it drops below the last 2
    assert sig.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_vol_target_scales_down_volatile_assets():
    rng = np.random.default_rng(1)
    calm = pd.Series(rng.normal(0, 0.003, 800), index=IDX)
    wild = pd.Series(rng.normal(0, 0.03, 800), index=IDX)
    one = pd.Series(1.0, index=IDX)
    assert rl.vol_target(one, calm, 0.10).iloc[-1] == 1.0  # capped at 1
    assert 0.1 < rl.vol_target(one, wild, 0.10).iloc[-1] < 0.3


def test_warmup_days():
    assert rl.warmup_days({"rule": "ma_crossover", "fast": 50, "slow": 200}) == 200
    assert rl.warmup_days({"rule": "tsmom", "lookback": 12}) >= 252
