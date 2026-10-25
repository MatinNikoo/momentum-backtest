"""Config files, the CLI, the tearsheet, Monte Carlo and the today script, on synthetic data where possible."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from trendlab import config as cf
from trendlab import montecarlo as mc
from trendlab import today as td
from conftest import fake_returns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loader(name):
    seeds = {"us": 1, "dev_ex_us": 2, "ust10": 3, "gld": 4}
    r, rf = fake_returns(n=2500, seed=seeds[name])
    return r.rename(name), rf


def test_config_validation():
    with pytest.raises(ValueError):
        cf.normalize({"rule": {"rule": "ma_crossover"}})
    with pytest.raises(ValueError):
        cf.normalize({"data": {"assets": ["nope"]}, "rule": {"rule": "ma_crossover"}})
    with pytest.raises(ValueError):
        cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover"}, "execution": {"delay": -1}})
    c = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover"}})
    assert c["execution"]["delay"] == 1 and c["execution"]["cost_bps"] == 5.0


def test_every_example_config_loads():
    for f in os.listdir(os.path.join(ROOT, "configs")):
        cfg = cf.load(os.path.join(ROOT, "configs", f))
        assert cfg["name"] == f[:-5]


def test_single_and_multi_asset_runs():
    cfg = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100}})
    res = cf.run(cfg, loader)
    assert res["strategy"].index[0] >= res["eval_start"]
    tab = cf.headline(res)
    assert np.isfinite(tab.loc["sharpe"]).all()
    multi = cf.normalize({"data": {"assets": ["us", "ust10"]},
                          "rule": {"rule": "tsmom", "lookback": 12, "vol_target": 0.1},
                          "portfolio": {"weights": {"us": 0.6, "ust10": 0.4}}})
    res = cf.run(multi, loader)
    # the passive book is a 60/40 of the two fake assets, rebalanced monthly, so on the first day of
    # every month its return is exactly 0.6 a + 0.4 b
    b = res["benchmark"]
    first = b.groupby(b.index.to_period("M")).head(1)
    want = (0.6 * loader("us")[0] + 0.4 * loader("ust10")[0]).reindex(first.index)
    assert np.allclose(first.values, want.values)
    assert res["weights"] == {"us": 0.6, "ust10": 0.4}


def test_monte_carlo_shapes_and_sanity():
    m = pd.DataFrame({"a": np.full(240, 0.01), "b": np.full(240, -0.01)})
    w = mc.simulate(m, horizon=12, n_paths=100)
    assert w.shape == (100, 12, 2)
    s = mc.summarize(w, ["a", "b"])
    assert np.isclose(s.loc["p50_end_wealth", "a"], 1.01 ** 12)
    assert s.loc["prob_loss", "a"] == 0 and s.loc["prob_loss", "b"] == 1
    assert np.isclose(s.loc["median_max_dd", "b"], 1 - 0.99 ** 12)


def test_tearsheet_end_to_end(tmp_path):
    from trendlab import tearsheet as ts
    cfg = cf.normalize({"name": "fake", "data": {"assets": ["us"]},
                        "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100},
                        "report": {"n_boot": 40, "sweep": {"fast": [10, 20], "slow": [50, 100]}}})
    out, outdir = ts.make(cfg, str(tmp_path), loader)
    assert os.path.getsize(os.path.join(outdir, "tearsheet.pdf")) > 10_000
    summary = json.load(open(os.path.join(outdir, "summary.json")))
    for key in ["metrics", "smile", "regimes_vs_volmatched", "deflated_sharpe", "pbo", "monte_carlo_10y"]:
        assert key in summary
    assert len(summary["sweep_sharpes_annual"]) == 4


def test_today_describe():
    r, rf = fake_returns(n=600, seed=9, drift=0.002)  # a strong uptrend
    s = td.describe((1 + r).cumprod(), rf)
    assert s["ma_50_200"] == "in" and s["tsmom_12m"] == "in"
    assert "Not investment advice" in td.table({"fake": s})
    with pytest.raises(ValueError):
        td.describe((1 + r.iloc[:100]).cumprod(), rf)


def test_cli_runs_on_the_snapshot(tmp_path, capsys):
    from trendlab.cli import main
    cfg = tmp_path / "c.toml"
    cfg.write_text('name = "t"\n[data]\nassets = ["us"]\nend = "1950-12-31"\n[rule]\nrule = "sma_month"\nmonths = 10\n')
    main(["backtest", str(cfg), "--out", str(tmp_path / "d.csv")])
    out = capsys.readouterr().out
    assert "Sharpe over T-bills" in out and os.path.exists(tmp_path / "d.csv")
    main(["today"])
    assert "Not investment advice" in capsys.readouterr().out


def test_cli_gives_a_clean_error_for_a_missing_config(capsys):
    from trendlab.cli import main
    with pytest.raises(SystemExit) as e:
        main(["backtest", "configs/does_not_exist.toml"])
    assert e.value.code == 2 and "config file not found" in capsys.readouterr().err
