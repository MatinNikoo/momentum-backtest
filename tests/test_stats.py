"""Bootstrap, Reality Check / SPA, Treynor-Mazuy, PBO and deflated Sharpe."""
import numpy as np

from trendlab import insurance as ins
from trendlab import stats as st


def test_bootstrap_indices_are_valid_and_reproducible():
    a = st.stationary_bootstrap_matrix(100, 10, 50, seed=3)
    b = st.stationary_bootstrap_matrix(100, 10, 50, seed=3)
    assert a.shape == (50, 100) and a.min() >= 0 and a.max() < 100
    assert np.array_equal(a, b)
    # with blocks of about 10, about 90% of steps continue the block
    cont = (np.diff(a, axis=1) % 100 == 1).mean()
    assert 0.85 < cont < 0.95


def test_reality_check_and_spa_calibration():
    rng = np.random.default_rng(0)
    p_null = [st.reality_check_spa(rng.normal(0, 1, (240, 10)), n_boot=200, seed=i)["p_spa"] for i in range(40)]
    assert np.mean(np.array(p_null) < 0.10) <= 0.25  # rejects pure noise rarely
    D = rng.normal(0, 1, (240, 10))
    D[:, 3] += 0.4  # one rule really is better
    out = st.reality_check_spa(D, n_boot=500, seed=1)
    assert out["p_spa"] < 0.01 and out["p_reality_check"] < 0.01 and out["best_rule"] == 3


def test_spa_ignores_terrible_rules_more_than_rc():
    rng = np.random.default_rng(5)
    D = rng.normal(0, 1, (300, 30))
    D[:, 0] += 0.12
    D[:, 1:] -= 1.0  # very bad rules make the plain Reality Check conservative
    out = st.reality_check_spa(D, n_boot=500, seed=2)
    assert out["p_spa"] <= out["p_reality_check"]


def test_treynor_mazuy_recovers_coefficients():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.08, 400)
    y = 0.01 + 0.5 * x + 2.0 * x ** 2 + rng.normal(0, 0.001, 400)
    a, b, g = st.treynor_mazuy(y, x)
    assert abs(a - 0.01) < 1e-3 and abs(b - 0.5) < 0.01 and abs(g - 2.0) < 0.05


def test_pbo_and_deflated_sharpe():
    rng = np.random.default_rng(42)
    pbos = [st.pbo_cscv(rng.normal(0, 0.01, (1500, 20)), n_blocks=10)["pbo"] for _ in range(20)]
    assert 0.4 < np.mean(pbos) < 0.6  # picking the best of pure noise is a coin flip
    skill = rng.normal(0, 0.01, (1500, 20))
    skill[:, 0] += 0.002
    assert st.pbo_cscv(skill, n_blocks=10)["pbo"] < 0.1
    assert st.expected_max_sharpe(0.01, 100) > st.expected_max_sharpe(0.01, 10) > 0
    noise = rng.normal(0, 0.01, (2000, 30))
    d = st.deflated_sharpe(noise[:, 0], noise.mean(0) / noise.std(0))
    assert 0 <= d["dsr"] <= 1


def test_episodes_labels_and_dollars_by_hand():
    r = np.array([-0.1, -0.2, 0.5, 0.0])  # wealth 1, .9, .72, 1.08, 1.08
    eps = ins.episodes(r, 0.20)
    assert len(eps) == 1
    e = eps[0]
    assert (e["peak"], e["trough"], e["recovered"]) == (0, 2, 3) and np.isclose(e["depth"], 0.28)
    assert ins.labels_from_episodes(4, eps).tolist() == [1, 1, 2, 0]
    acc = ins.account(np.zeros(4), r, eps, n_boot=50)
    assert np.isclose(acc["episodes"][0]["dollars_saved"], 100_000 * (1 - 0.72))
    assert acc["episodes_won"] == 1
    assert ins.episodes(r, 0.30) == []


def test_episode_that_never_recovers():
    eps = ins.episodes(np.array([0.1, -0.3, 0.05]), 0.2)
    assert eps[0]["recovered"] is None and eps[0]["peak"] == 1
    assert ins.labels_from_episodes(3, eps).tolist() == [0, 1, 2]


def test_regime_gaps_mean():
    gap = np.r_[np.full(50, 0.01), np.full(50, -0.02)]
    out = ins.regime_gaps(gap, {"a": np.r_[np.ones(50, bool), np.zeros(50, bool)]}, n_boot=100)
    assert np.isclose(out["a"]["mean_gap"], 0.01) and out["a"]["months"] == 50


def test_swing_episodes_find_a_second_bear_inside_a_slump():
    # down 50%, up 30% (old high not regained), down 30% again, then up 30%
    r = np.array([-0.5, 0.3, -0.3, 0.3])
    assert len(ins.episodes(r, 0.20)) == 1  # the all time high rule only sees one long fall
    sw = ins.swing_episodes(r, 0.20)
    assert [(e["peak"], e["trough"]) for e in sw] == [(0, 1), (2, 3)]
    assert np.isclose(sw[1]["depth"], 0.3)
