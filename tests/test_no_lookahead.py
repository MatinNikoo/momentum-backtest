"""
No rule may use data from after the day it trades. Change every price after
day k and the positions up to day k + delay + 1 must not move.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hs

from trendlab import engine as en
from trendlab import rules as rl
from conftest import fake_returns

SPECS = [
    {"rule": "ma_crossover", "fast": 50, "slow": 200},
    {"rule": "ma_crossover", "fast": 10, "slow": 30, "vol_target": 0.10},
    {"rule": "sma_month", "months": 10},
    {"rule": "tsmom", "lookback": 12},
    {"rule": "tsmom", "lookback": 12, "vol_target": 0.10},
    {"rule": "tsmom", "lookback": 6, "vol_target": 0.40, "cap": 5.0, "long_only": False},
    {"rule": "breakout", "n_entry": 100},
    {"rule": "ema_crossover", "fast": 20, "slow": 100},
    {"rule": "price_above_ma", "window": 150, "vol_target": 0.10},
]


def positions(spec, r, rf, delay):
    px = en.total_return_index(r)
    return en.positions_from_signal(rl.make_signal(spec, px, rf), delay)


def check(spec, k, delay, seed):
    r, rf = fake_returns(n=1200, seed=seed)
    base = positions(spec, r, rf, delay)
    rng = np.random.default_rng(seed + 1)
    r2 = r.copy()
    r2.iloc[k + 1:] = rng.normal(0, 0.05, len(r) - k - 1)  # rewrite the whole future
    rf2 = rf.copy()
    rf2.iloc[k + 1:] = 0.003
    moved = positions(spec, r2, rf2, delay)
    upto = k + delay + 2
    assert np.array_equal(base.values[:upto], moved.values[:upto]), spec


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("delay", [0, 1, 3])
def test_future_changes_do_not_move_past_positions(spec, delay):
    for k in [300, 555, 800, 1100]:
        check(spec, k, delay, seed=k)


@settings(max_examples=30, deadline=None)
@given(k=hs.integers(260, 1150), delay=hs.integers(0, 5), which=hs.integers(0, len(SPECS) - 1),
       seed=hs.integers(0, 10_000))
def test_no_lookahead_property(k, delay, which, seed):
    check(SPECS[which], k, delay, seed)


@pytest.mark.parametrize("spec", SPECS)
def test_truncating_the_data_gives_the_same_signal(spec):
    # a rule run on data up to day k must say what the full run said on day k
    r, rf = fake_returns(n=1000, seed=7)
    px = en.total_return_index(r)
    full = rl.make_signal(spec, px, rf)
    for k in [400, 650, 999]:
        part = rl.make_signal(spec, px.iloc[:k + 1], rf.iloc[:k + 1])
        assert np.allclose(part.values, full.values[:k + 1]), (spec, k)


def test_the_check_catches_a_peeking_rule():
    # make sure the check can fail: a position that uses the same day's signal sees that day's return
    r, rf = fake_returns(n=600, seed=3)
    caught = 0
    for k in range(300, 580, 7):
        px = en.total_return_index(r)
        cheat = rl.ma_crossover(px, 5, 20)  # no shift at all
        r2 = r.copy()
        r2.iloc[k + 1:] = -0.08
        cheat2 = rl.ma_crossover(en.total_return_index(r2), 5, 20)
        caught += not np.array_equal(cheat.values[:k + 2], cheat2.values[:k + 2])
    assert caught > 0
