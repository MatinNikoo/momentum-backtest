"""Custom csv prices, overrides, sizing, sweeps, walk forward, html reports and the newer commands."""
import json

import numpy as np
import pandas as pd
import pytest

from trendlab import config as cf
from trendlab import data as dt
from trendlab import rules as rl
from trendlab import sweep as sw
from trendlab.cli import main
from conftest import fake_returns


def loader(name):
    r, rf = fake_returns(n=4500, seed=11, drift=0.0005)
    return r.rename(name), rf


def write_prices(path, n=800, seed=3, name="Close", extra=None, sep=","):
    r, _ = fake_returns(n=n, seed=seed)
    px = 100 * (1 + r).cumprod()
    df = pd.DataFrame({"Date": px.index.strftime("%Y-%m-%d"), name: px.values})
    if extra:
        df[extra] = 1
    df.to_csv(path, index=False, sep=sep)
    return px


# ---- csv input and cash ----------------------------------------------------
def test_csv_prices_become_returns(tmp_path):
    px = write_prices(tmp_path / "a.csv")
    r, rf = dt.from_csv(str(tmp_path / "a.csv"), cash=0.0)
    assert np.allclose(r.values, px.pct_change().dropna().values)
    assert (rf == 0).all() and r.name == "a"


def test_csv_column_detection_and_returns_kind(tmp_path):
    write_prices(tmp_path / "b.csv", name="Adj Close", extra="Volume")
    r, _ = dt.from_csv(str(tmp_path / "b.csv"), cash=0.0)
    assert len(r) == 799
    write_prices(tmp_path / "c.csv", name="x", extra="y")
    with pytest.raises(ValueError, match="several columns"):
        dt.from_csv(str(tmp_path / "c.csv"))
    rr = pd.DataFrame({"d": pd.bdate_range("2020-01-01", periods=60), "ret": np.full(60, 0.001)})
    rr.to_csv(tmp_path / "d.csv", index=False)
    r, _ = dt.from_csv(str(tmp_path / "d.csv"), kind="returns", cash=0.0)
    assert np.allclose(r.values, 0.001)


def test_csv_semicolon_separator_and_unsorted_rows(tmp_path):
    px = write_prices(tmp_path / "e.csv", sep=";")
    df = pd.read_csv(tmp_path / "e.csv", sep=";").iloc[::-1]
    df.to_csv(tmp_path / "e.csv", index=False, sep=";")
    r, _ = dt.from_csv(str(tmp_path / "e.csv"), cash=0.0)
    assert np.allclose(r.values, px.pct_change().dropna().values)


@pytest.mark.parametrize("bad,msg", [("dup", "repeated dates"), ("neg", "zero or negative"), ("short", "only")])
def test_csv_problems_have_readable_errors(tmp_path, bad, msg):
    write_prices(tmp_path / "f.csv")
    df = pd.read_csv(tmp_path / "f.csv")
    if bad == "dup":
        df = pd.concat([df, df.iloc[[5]]])
    elif bad == "neg":
        df.loc[10, "Close"] = -1
    else:
        df = df.iloc[:10]
    df.to_csv(tmp_path / "f.csv", index=False)
    with pytest.raises(ValueError, match=msg):
        dt.from_csv(str(tmp_path / "f.csv"))
    with pytest.raises(ValueError, match="not found"):
        dt.from_csv(str(tmp_path / "missing.csv"))


def test_flat_cash_rate_compounds_to_the_annual_rate():
    idx = pd.date_range("2021-01-01", periods=365, freq="D")
    rf = dt.cash_returns(idx, 0.05)
    assert np.isclose((1 + rf).prod(), 1.05 ** (364 / 365) * (1.05 ** (1 / 365)), rtol=1e-9)
    assert (dt.cash_returns(idx, "none") == 0).all()
    with pytest.raises(ValueError):
        dt.cash_returns(idx, "libor")


def test_tbill_cash_keeps_the_last_rate_past_the_end_of_the_data():
    end = dt.last_rf_date()
    idx = pd.bdate_range(end - pd.Timedelta(days=10), end + pd.Timedelta(days=14))
    rf = dt.cash_returns(idx, "tbill")
    after = rf[idx > end]
    assert (after > 0).all() and np.allclose(after.values, dt.rf_daily().iloc[-1])


# ---- config: files, overrides, sizing -----------------------------------------
def test_config_reads_csv_files_relative_to_the_config(tmp_path):
    write_prices(tmp_path / "px.csv", n=900)
    (tmp_path / "c.toml").write_text('[data]\nassets = ["mine"]\ncash = 0.02\n[data.files]\nmine = "px.csv"\n'
                                     '[rule]\nrule = "price_above_ma"\nwindow = 50\n')
    cfg = cf.load(str(tmp_path / "c.toml"))
    res = cf.run(cfg)
    assert len(res["strategy"]) > 500 and res["rf"].iloc[1] > 0


def test_overrides_change_values_and_reject_nonsense():
    cfg = {"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover"}}
    cf.apply_overrides(cfg, ["execution.cost_bps=25", "rule.fast=20", "data.start=\"1990-01-01\""])
    c = cf.normalize(cfg)
    assert c["execution"]["cost_bps"] == 25 and c["rule"]["fast"] == 20 and c["data"]["start"] == "1990-01-01"
    with pytest.raises(ValueError, match="section.key=value"):
        cf.apply_overrides({}, ["nodot=1"])


def test_sizing_folds_into_the_rule_and_fraction_scales_positions():
    base = {"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100}}
    half = {**base, "sizing": {"fraction": 0.5}}
    a, b = cf.run(cf.normalize(base), loader), cf.run(cf.normalize(half), loader)
    assert np.allclose(b["per_asset"]["us"]["position"], 0.5 * a["per_asset"]["us"]["position"])
    vt = cf.normalize({**base, "sizing": {"vol_target": 0.1, "max_leverage": 2.0, "vol_window": 30}})
    assert vt["rule"]["vol_target"] == 0.1 and vt["rule"]["cap"] == 2.0 and vt["rule"]["com"] == 30
    pos = cf.run(vt, loader)["per_asset"]["us"]["position"]
    assert pos.max() <= 2.0 + 1e-12
    with pytest.raises(ValueError, match="both"):
        cf.normalize({**base, "rule": {**base["rule"], "vol_target": 0.2}, "sizing": {"vol_target": 0.1}})
    with pytest.raises(ValueError, match="unknown"):
        cf.normalize({**base, "sizing": {"kelly": 1}})


def test_bad_rule_parameters_are_named():
    with pytest.raises(ValueError, match="no parameter 'slw'"):
        cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover", "slw": 200}})
    with pytest.raises(ValueError, match="unknown rule"):
        cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "magic"}})
    with pytest.raises(ValueError, match="vol_target"):
        rl.check_spec({"rule": "breakout", "cap": 2.0})


def test_new_rules_on_obvious_series():
    idx = pd.bdate_range("2010-01-01", periods=600)
    up = pd.Series(np.linspace(100, 300, 600), index=idx)
    down = pd.Series(np.linspace(300, 100, 600), index=idx)
    assert rl.price_above_ma(up, 100).iloc[-1] == 1 and rl.price_above_ma(down, 100).iloc[-1] == 0
    assert rl.ema_crossover(up, 20, 100).iloc[-1] == 1 and rl.ema_crossover(down, 20, 100).iloc[-1] == 0
    assert rl.ema_crossover(up, 20, 100).iloc[:100].sum() == 0


def test_short_windows_say_what_is_wrong():
    cfg = cf.normalize({"data": {"assets": ["us"], "end": "1926-09-30"}, "rule": {"rule": "ma_crossover"}})
    with pytest.raises(ValueError, match="widen"):
        cf.run(cfg)


# ---- sweeps ---------------------------------------------------------------------
def test_parse_grid():
    g = sw.parse_grid(["fast=20,50", "slow=100:200:50", "sizing.vol_target=0.05:0.15:0.05", "execution.delay=0,1"])
    assert g["fast"] == [20, 50] and g["slow"] == [100, 150, 200]
    assert g["sizing.vol_target"] == [0.05, 0.1, 0.15] and g["execution.delay"] == [0, 1]
    for bad in ["fast", "fast=", "slow=5:1:1", "x=1:2:0"]:
        with pytest.raises(ValueError):
            sw.parse_grid([bad])


def test_grid_skips_fast_not_below_slow_and_validates_names():
    cfg = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover"}})
    cells = list(sw.expand(cfg, {"fast": [20, 100, 200], "slow": [100, 200]}))
    assert [p for p, _ in cells] == [{"fast": 20, "slow": 100}, {"fast": 20, "slow": 200}, {"fast": 100, "slow": 200}]
    with pytest.raises(ValueError, match="no parameter"):
        list(sw.expand(cfg, {"nope": [1]}))
    with pytest.raises(ValueError, match="no option"):
        list(sw.expand(cfg, {"execution.fraction": [1]}))


def test_every_cell_is_scored_on_the_same_dates_and_matches_a_single_run():
    cfg = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100}})
    runs = sw.run_grid(cfg, {"fast": [10, 20], "slow": [60, 250]}, loader)
    firsts = {r["strategy"].index[0] for _, r in runs["cells"]}
    assert len(firsts) == 1 and runs["base"]["strategy"].index[0] == firsts.pop()
    p, r = runs["cells"][1]  # fast 10, slow 250
    single = cf.run(sw.with_params(cfg, p), loader)["strategy"]
    assert np.allclose(r["strategy"].values, single[single.index >= runs["start"]].values)
    tab = sw.table(runs)
    assert list(tab.columns[:2]) == ["fast", "slow"] and len(tab) == 4 and tab["max_dd"].between(0, 1).all()


def test_overfit_check_counts_the_cells():
    cfg = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover"}})
    runs = sw.run_grid(cfg, {"fast": [10, 20, 40], "slow": [100, 200]}, loader)
    chk = sw.overfit_check(runs)
    assert chk["cells"] == 6 and 0 <= chk["dsr"] <= 1 and 0 <= chk["pbo"] <= 1


def test_heatmap_files(tmp_path):
    cfg = cf.normalize({"data": {"assets": ["us"]}, "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100}})
    for grid, keys in [({"fast": [10, 20], "slow": [60, 100]}, ["slow", "fast"]), ({"slow": [60, 100, 150]}, ["slow"])]:
        runs = sw.run_grid(cfg, grid, loader)
        out = tmp_path / f"{len(keys)}.png"
        sw.heatmap(sw.table(runs), keys, "sharpe", str(out), base_params=cfg["rule"], benchmark=0.3)
        assert out.stat().st_size > 5_000


# ---- walk forward -------------------------------------------------------------------
def make_runs(n_months=180):
    """Two fake cells where which one is better flips every 5 years, plus a base cell and a benchmark."""
    daily = pd.bdate_range("2000-01-03", periods=n_months * 21)
    rng = np.random.default_rng(0)

    def cell(sign):
        mu = np.where((np.arange(len(daily)) // (21 * 60)) % 2 == 0, sign, -sign) * 0.0006
        s = pd.Series(mu + rng.normal(0, 0.004, len(daily)), index=daily)
        return {"strategy": s, "benchmark": s * 0 + 0.0002, "rf": s * 0 + 0.0001, "eval_start": daily[0]}

    a, b = cell(1), cell(-1)
    return {"cells": [({"n": 1}, a), ({"n": 2}, b)], "base": a, "start": daily[0]}


def test_walk_forward_picks_on_the_past_only_and_stitches_test_windows():
    runs = make_runs()
    wf = sw.walk_forward(runs, "cagr", train_years=2, test_years=1)
    f = wf["folds"]
    assert len(f) * 12 >= len(wf["returns"]) > (len(f) - 1) * 12
    assert f["test_from"].is_monotonic_increasing and f["train_score"].notna().all()
    m = [cf.monthly(r)["strategy"] for _, r in runs["cells"]]
    for _, row in f.iterrows():
        start = pd.Period(row["test_from"], "M")
        got = wf["returns"]["walk_forward"].loc[start:start]
        assert np.isclose(got.iloc[0], m[int(row["n"]) - 1].loc[start])


def test_walk_forward_ignores_the_future():
    runs = make_runs()
    wf = sw.walk_forward(runs, "sharpe", train_years=3, test_years=1)
    cut = 120 * 21
    for _, r in runs["cells"]:
        r["strategy"] = r["strategy"].copy()
        r["strategy"].iloc[cut:] = 0.05  # rewrite everything after month 120
    wf2 = sw.walk_forward(runs, "sharpe", train_years=3, test_years=1)
    keep = wf["folds"]["test_from"] <= "2010-01"  # training ends before the rewritten months
    assert keep.sum() >= 5
    assert (wf["folds"]["n"][keep].values == wf2["folds"]["n"][keep].values).all()


def test_walk_forward_rejects_short_samples_and_bad_metrics():
    runs = make_runs(60)
    with pytest.raises(ValueError, match="needs"):
        sw.walk_forward(runs, "sharpe", train_years=10)
    with pytest.raises(ValueError, match="metric"):
        sw.walk_forward(runs, "luck", train_years=1)


# ---- reports and command line ---------------------------------------------------------
@pytest.mark.filterwarnings("ignore:All-NaN slice")  # the fake series has no 20% bear market to account for
def test_html_tearsheet(tmp_path):
    from trendlab import tearsheet as ts
    cfg = cf.normalize({"name": "fakehtml", "data": {"assets": ["us"]},
                        "rule": {"rule": "ma_crossover", "fast": 20, "slow": 100}, "report": {"n_boot": 60}})
    short = lambda name: (fake_returns(n=2500, seed=5)[0].rename(name), fake_returns(n=2500)[1])
    ts.make(cfg, str(tmp_path), short, fmt="html")
    text = (tmp_path / "tearsheet.html").read_text()
    assert "data:image/png;base64" in text and "fakehtml" in text and "<table>" in text
    assert not (tmp_path / "tearsheet.pdf").exists()
    with pytest.raises(ValueError):
        ts.make(cfg, str(tmp_path), loader, fmt="docx")


@pytest.fixture
def prices(tmp_path):
    write_prices(tmp_path / "asset.csv", n=4000)
    return str(tmp_path / "asset.csv")


def test_cli_backtest_on_a_csv_with_overrides(prices, tmp_path, capsys):
    main(["backtest", prices, "--rule", "ema_crossover", "--set", "rule.fast=20", "--set", "rule.slow=100",
          "--set", "data.cash=0.02", "--json", str(tmp_path / "s.json"), "--plot", str(tmp_path / "p.png")])
    out = capsys.readouterr().out
    assert "Sharpe over T-bills" in out and "ema_crossover fast=20 slow=100" in out
    js = json.load(open(tmp_path / "s.json"))
    assert js["rule"]["fast"] == 20 and (tmp_path / "p.png").stat().st_size > 5_000


def test_cli_sweep_and_walkforward(prices, tmp_path, capsys):
    main(["sweep", prices, "--grid", "fast=10,20", "--grid", "slow=60,100", "--metric", "max_dd",
          "--heatmap", str(tmp_path / "h.png"), "--out", str(tmp_path / "t.csv")])
    out = capsys.readouterr().out
    assert "best cell" in out and "deflated Sharpe" in out
    assert len(pd.read_csv(tmp_path / "t.csv")) == 4 and (tmp_path / "h.png").exists()
    main(["walkforward", prices, "--grid", "fast=10,20", "--grid", "slow=60,100", "--train-years", "5",
          "--folds", "--out", str(tmp_path / "w.csv"), "--plot", str(tmp_path / "w.png")])
    out = capsys.readouterr().out
    assert "Walk forward" in out and "test_from" in out and (tmp_path / "w.png").exists()
    assert "train_score" in pd.read_csv(tmp_path / "w.csv").columns


def test_cli_init_writes_a_config_that_runs(prices, tmp_path, capsys):
    path = str(tmp_path / "mine.toml")
    main(["init", path, "--csv", prices])
    with pytest.raises(SystemExit):
        main(["init", path])  # won't overwrite
    assert "already exists" in capsys.readouterr().err
    cfg = cf.load(path)
    assert cfg["data"]["assets"] == ["asset"] and len(cf.run(cfg)["strategy"]) > 1000


def test_cli_lists_and_today_on_a_csv(prices, capsys):
    main(["rules"])
    assert "ema_crossover" in capsys.readouterr().out
    main(["assets"])
    out = capsys.readouterr().out
    assert "1926-07-01" in out and "spy" in out
    main(["today", "--csv", prices])
    assert "asset (csv)" in capsys.readouterr().out


@pytest.mark.parametrize("argv,msg", [
    (["backtest", "nope.csv"], "price file not found"),
    (["sweep", "configs/spy_50_200.toml", "--grid", "slw=1,2"], "no parameter"),
    (["walkforward", "configs/spy_50_200.toml", "--grid", "fast=10"], "at least two"),
    (["backtest", "configs/spy_50_200.toml", "--set", "execution.cost_bps"], "section.key=value"),
    (["backtest", "configs/spy_50_200.toml", "--set", "data.assets=[\"zzz\"]"], "unknown assets"),
])
def test_cli_errors_are_one_readable_line(argv, msg, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    err = capsys.readouterr().err
    assert e.value.code == 2 and msg in err and "Traceback" not in err


def test_a_month_is_finished_when_no_weekday_is_left_in_it():
    for day, done in [("2026-08-31", True), ("2026-08-28", False), ("2026-08-14", False),
                      ("1950-12-30", True), ("2026-02-27", True), ("2026-05-29", True)]:
        assert rl.month_is_complete(pd.Timestamp(day)) is done, day
