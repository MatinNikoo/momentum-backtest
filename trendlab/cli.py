"""
Command line, run as `python -m trendlab <command>` or `trendlab <command>` after `pip install -e .`.

    backtest     headline numbers for a config or a csv of prices
    tearsheet    full report as pdf and/or html
    sweep        grid of parameters with a table and a heatmap
    walkforward  pick parameters on the past, trade them on the next window
    montecarlo   range of 10 year outcomes from block bootstrapped history
    today        what each rule says on the latest data
    assets       what's in the data snapshots
    rules        the trading rules and their parameters
    init         write a starter config
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd

from . import __version__
from . import config as cf
from . import data as dt
from . import metrics as mt
from . import rules as rl
from . import sweep as sw

EPILOG = """\
examples
  trendlab backtest configs/us_century_50_200.toml
  trendlab backtest prices.csv --rule ema_crossover --set rule.fast=20 --set rule.slow=100
  trendlab sweep configs/us_century_50_200.toml --grid fast=20:100:20 --grid slow=100:300:50
  trendlab walkforward configs/us_century_50_200.toml --grid fast=20,50,100 --grid slow=150,200,250
  trendlab tearsheet configs/spy_50_200.toml --format both

A config is a TOML file with [data], [rule] and optionally [execution], [sizing], [portfolio] and [report].
Every option is listed in docs/usage.md. --set changes any value for one run without touching the file,
for example --set execution.cost_bps=25 or --set data.cash=0.03.

Research tool, not investment advice.
"""

CONFIG_HELP = ("a .toml config (see configs/), or a .csv of prices to try a rule on quickly "
               "(a date column and one price column)")


def _config_args(p):
    p.add_argument("config", help=CONFIG_HELP)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value for this run, repeatable (execution.delay=2, rule.fast=20)")
    p.add_argument("--rule", default="ma_crossover", choices=sorted(rl.RULES),
                   help="rule to use when the input is a csv (default ma_crossover)")
    p.add_argument("--column", help="price column to use when the input is a csv with several")
    p.add_argument("--kind", choices=["price", "returns"], default="price",
                   help="whether the csv column holds prices or decimal returns (default price)")


def _load(args):
    if args.config.lower().endswith(".csv"):
        return cf.quick(args.config, args.rule, args.column, args.kind, args.overrides)
    if not os.path.exists(args.config):
        raise ValueError(f"config file not found: {args.config} (the examples are in configs/)")
    return cf.load(args.config, args.overrides)


def _pct(v):
    return f"{v:.1%}"


def format_summary(cols: dict) -> str:
    """cols maps a column title to a metrics.summary dict. Returns an aligned text table."""
    rows = [("CAGR", "cagr", _pct), ("Volatility", "vol", _pct), ("Sharpe over T-bills", "sharpe", lambda v: f"{v:.2f}"),
            ("Max drawdown", "max_dd", _pct), ("Ulcer index", "ulcer", _pct), ("Worst 12 months", "worst_12m", _pct),
            ("CVaR 5% (monthly)", "cvar_5", _pct), ("Skew", "skew", lambda v: f"{v:.2f}"),
            ("Longest drawdown, months", "longest_underwater_months", lambda v: f"{int(v)}"),
            ("Months", "months", lambda v: f"{int(v)}")]
    tab = pd.DataFrame({t: [f(s[k]) for _, k, f in rows] for t, s in cols.items()}, index=[r[0] for r in rows])
    return tab.to_string(justify="right")


def _describe(cfg, res) -> str:
    ex = cfg["execution"]
    rule = " ".join(f"{k}={v}" for k, v in cfg["rule"].items() if k != "rule")
    size = f", {cfg['sizing']['fraction']:g}x size" if cfg["sizing"].get("fraction") not in (None, 1) else ""
    years = (res["strategy"].index[-1] - res["strategy"].index[0]).days / 365.25
    assets = ", ".join(cfg["data"]["assets"])
    head = assets if cfg["name"] == assets else f"{cfg['name']}: {assets}"
    return (f"{head}, {cfg['rule']['rule']} {rule}\n"
            f"{res['strategy'].index[0].date()} to {res['strategy'].index[-1].date()} ({years:.1f} years), "
            f"trade {ex['delay']} day{'s' if ex['delay'] != 1 else ''} after the signal, "
            f"{ex['cost_bps']:g} bps a trade{size}")


def _backtest(args):
    cfg = _load(args)
    res = cf.run(cfg)
    m = cf.monthly(res)
    cols = {"Strategy": mt.summary(m["strategy"].values, m["rf"].values),
            "Buy and hold": mt.summary(m["benchmark"].values, m["rf"].values)}
    print(_describe(cfg, res) + "\n")
    print(format_summary(cols))
    tv = cf.turnover(res)
    grow = {k: float(np.prod(1 + m[k].values)) for k in ("strategy", "benchmark")}
    print(f"\nGrowth of $1 {grow['strategy']:.2f} against {grow['benchmark']:.2f}. Average exposure "
          f"{tv['avg_exposure']:.0%}, position changed on {tv['switches_per_year_per_asset']:.1f} days a year.")
    if args.out:
        df = pd.DataFrame({"strategy": res["strategy"], "benchmark": res["benchmark"], "rf": res["rf"]})
        df.to_csv(args.out)
        print("daily returns written to", args.out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"name": cfg["name"], "start": str(res["strategy"].index[0].date()),
                       "end": str(res["strategy"].index[-1].date()), "metrics": cols, "turnover": tv,
                       "rule": cfg["rule"], "execution": cfg["execution"]}, f, indent=1, default=float)
        print("summary written to", args.json)
    if args.plot:
        from . import tearsheet as ts
        ts.quick_plot(res, cfg["name"], args.plot)
        print("chart written to", args.plot)


def _tearsheet(args):
    from . import tearsheet as ts
    cfg = _load(args)
    if args.n_boot:
        cfg["report"]["n_boot"] = args.n_boot
    out, outdir = ts.make(cfg, args.outdir, fmt=args.format)
    files = {"pdf": ["tearsheet.pdf"], "html": ["tearsheet.html"], "both": ["tearsheet.pdf", "tearsheet.html"]}
    print(f"wrote {', '.join(os.path.join(outdir, f) for f in files[args.format])} and summary.json\n")
    m = out["metrics"]
    print(format_summary({"Strategy": m["strategy"], "Buy and hold": m["benchmark"], "Same vol mix": m["vol_matched"]}))


def _grid_for(cfg, args) -> dict:
    grid = sw.parse_grid(args.grid) if args.grid else cfg["report"].get("sweep")
    if not grid:
        raise ValueError("no grid. Pass --grid name=v1,v2,v3 (or name=start:stop:step), once per parameter, "
                         "or put a sweep table under [report] in the config")
    return grid


def _metric_text(metric, v):
    return f"{v:.1%}" if metric in ("cagr", "max_dd", "ulcer", "vol") else f"{v:.2f}"


def _sweep(args):
    cfg = _load(args)
    grid = _grid_for(cfg, args)
    runs = sw.run_grid(cfg, grid)
    tab = sw.table(runs)
    keys = list(grid)
    metric = args.metric
    print(f"{cfg['name']}: {len(tab)} cells, all scored from {runs['start'].date()} "
          f"to {runs['base']['strategy'].index[-1].date()}\n")
    show = tab.sort_values(metric, ascending=not sw.METRICS[metric][1]) if (len(keys) != 2) else tab
    if len(keys) == 2:
        pv = tab.pivot(index=keys[1], columns=keys[0], values=metric).sort_index().sort_index(axis=1)
        print(f"{sw.METRICS[metric][0]}, rows are {keys[1]}, columns are {keys[0]}")
        print(pv.map(lambda v: _metric_text(metric, v)).to_string())
    else:
        fmt = {c: (lambda v, c=c: _metric_text(c, v)) for c in ["cagr", "vol", "sharpe", "max_dd", "ulcer", "calmar"]}
        fmt["switches_a_year"] = lambda v: f"{v:.1f}"
        print(show.to_string(index=False, formatters=fmt))
    at = sw.best_index(tab, metric)
    best = {k: tab.loc[at, k] for k in keys}
    bm_ = cf.monthly(runs["base"])
    bench = mt.summary(bm_["benchmark"].values, bm_["rf"].values)
    ref = bench[metric] if metric != "calmar" else bench["cagr"] / bench["max_dd"]
    print(f"\nbest cell {sw.label(best)} with {metric} {_metric_text(metric, tab.loc[at, metric])}, "
          f"buy and hold {_metric_text(metric, ref)}")
    chk = sw.overfit_check(runs) if any(not k.startswith("execution.") for k in keys) else None
    if chk:
        pbo = f" and the probability of backtest overfitting is {chk['pbo']:.2f}" if "pbo" in chk else ""
        print(f"Choosing the best of {chk['cells']} cells overstates it. After counting the tries, the deflated Sharpe "
              f"probability of the top cell by Sharpe is {chk['dsr']:.2f}{pbo}.")
    if args.out:
        tab.to_csv(args.out, index=False)
        print("table written to", args.out)
    if not args.no_plot and len(keys) <= 2:
        path = args.heatmap or os.path.join("reports", cfg["name"], f"sweep_{metric}.png")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        base = {**cfg["rule"], **{f"{s}.{k}": v for s in ("execution", "sizing") for k, v in cfg[s].items()}}
        sw.heatmap(tab, keys if len(keys) == 1 else [keys[1], keys[0]], metric, path, base_params=base,
                   benchmark=ref, title=cfg["name"])
        print("figure written to", path)


def _walkforward(args):
    cfg = _load(args)
    grid = _grid_for(cfg, args)
    runs = sw.run_grid(cfg, grid)
    wf = sw.walk_forward(runs, args.metric, args.train_years, args.test_years, args.expanding)
    f = wf["folds"]
    print(f"{cfg['name']}: {len(f)} test windows of {args.test_years:g} year, {len(runs['cells'])} parameter sets, "
          f"{'expanding' if args.expanding else 'rolling'} {args.train_years:g} year training on {args.metric}\n")
    print(f"out of sample from {f['test_from'].iloc[0]} to {f['test_to'].iloc[-1]}\n")
    cols = {"Walk forward": wf["summary"]["walk_forward"], "Fixed config": wf["summary"]["fixed_config"],
            "Average cell": wf["summary"]["mean_of_cells"], "Buy and hold": wf["summary"]["buy_and_hold"]}
    print(format_summary({k: v.to_dict() for k, v in cols.items()}))
    inn, out = f["train_score"].mean(), f["test_score"].mean()
    print(f"\nMean {args.metric} of the chosen set, {_metric_text(args.metric, inn)} in training and "
          f"{_metric_text(args.metric, out)} in the test windows. A test score far below the training score "
          f"means the grid was fitted to noise.")
    if args.folds:
        print("\n" + f.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.out:
        f.to_csv(args.out, index=False)
        print("windows written to", args.out)
    if args.plot:
        sw.plot_walk_forward(wf, args.plot, cfg["name"])
        print("figure written to", args.plot)


def _montecarlo(args):
    from . import benchmarks as bm
    from . import montecarlo as mc
    cfg = _load(args)
    res = cf.run(cfg)
    m = cf.monthly(res)
    w = bm.vol_matched_w(m["strategy"], m["benchmark"], m["rf"])
    m["vol_matched"] = bm.mix(m["benchmark"], m["rf"], w)
    wealth = mc.simulate(m[["strategy", "benchmark", "vol_matched"]], horizon=12 * args.years,
                         n_paths=args.paths, seed=args.seed)
    print(f"{args.paths:,} block bootstrap paths of {args.years} years from {m.index[0]} to {m.index[-1]}")
    print("This assumes the future looks like a reshuffled past. Treat it as a range, not a forecast.\n")
    tab = mc.summarize(wealth, ["strategy", "benchmark", "vol_matched"])
    print(tab.round(3).to_string())
    if args.out:
        tab.to_csv(args.out)
        print("table written to", args.out)


def _today(args):
    from . import today as td
    states = td.from_live() if args.live else td.from_snapshot()
    for path in args.csv or []:
        r, rf = dt.from_csv(path)
        states[r.name + " (csv)"] = td.describe((1 + r).cumprod(), rf)
    print(td.table(states))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(states, f, indent=1)


def _assets(args):
    print(f"{'name':20s} {'from':10s}  {'to':10s}  description")
    for k, v in dt.ASSETS.items():
        r, _ = dt.asset(k)
        print(f"{k:20s} {r.index[0].date()}  {r.index[-1].date()}  {v}")
    print("\nFor your own prices add a csv under [data.files] in a config, see docs/usage.md.")


def _rules(args):
    for name in sorted(rl.RULES):
        doc = " ".join((rl.RULES[name].__doc__ or "").strip().split("\n\n")[0].split())
        print(f"{name}\n" + textwrap.fill(doc, 96, initial_indent="  ", subsequent_indent="  "))
        shown = ", ".join(f"{k}={v}" for k, v in rl.params(name).items() if k not in ("vol_target", "cap", "com"))
        print(f"  parameters: {shown}")
    print("\nEvery rule also takes vol_target (annual volatility, like 0.10), cap (largest position, default 1) "
          "and com (volatility window in days, default 60).")


STARTER = """\
[rule]
rule = "ma_crossover"     # trendlab rules lists the others
fast = 50
slow = 200

[execution]
delay = 1                 # trade this many days after the signal close
cost_bps = 5              # paid on every unit of position change

# [sizing]
# vol_target = 0.10       # scale the position to 10% annual volatility
# max_leverage = 1.0
# fraction = 0.5          # hold half the position

# [report]
# sweep = { fast = [20, 50, 100], slow = [150, 200, 250] }
"""


def _init(args):
    if os.path.exists(args.path) and not args.force:
        raise ValueError(f"{args.path} already exists, pass --force to overwrite it")
    name = os.path.splitext(os.path.basename(args.path))[0]
    if args.csv:
        asset = os.path.splitext(os.path.basename(args.csv))[0]
        data = (f'[data]\nassets = ["{asset}"]\ncash = "tbill"      # or an annual rate like 0.03\n\n'
                f'[data.files.{asset}]\npath = "{args.csv}"\n'
                f'# column = "Adj Close"   # if the file has several price columns\n')
    else:
        data = '[data]\nassets = ["us"]              # trendlab assets lists them\n'
    with open(args.path, "w") as f:
        f.write(f'# starter config, every option is in docs/usage.md\nname = "{name}"\n\n{data}\n{STARTER}')
    print("wrote", args.path)


def build_parser():
    ap = argparse.ArgumentParser(prog="trendlab", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Trend following backtests with honest defaults: no lookahead, "
                                             "trades one day after the signal, cash earns the T-bill rate.",
                                 epilog=EPILOG)
    ap.add_argument("--version", action="version", version=f"trendlab {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    def add(name, fn, help_, **kw):
        p = sub.add_parser(name, help=help_, description=help_, formatter_class=argparse.RawDescriptionHelpFormatter, **kw)
        p.set_defaults(fn=fn)
        return p

    p = add("backtest", _backtest, "Run a config or a csv of prices and print the headline numbers.")
    _config_args(p)
    p.add_argument("--out", help="write daily returns to this csv")
    p.add_argument("--json", help="write the numbers to this json file")
    p.add_argument("--plot", help="save a growth and drawdown chart to this png")

    p = add("tearsheet", _tearsheet, "Full report with bootstrap intervals, regimes, the smile and Monte Carlo.")
    _config_args(p)
    p.add_argument("--outdir", help="folder for the report (default reports/<name>)")
    p.add_argument("--format", choices=["pdf", "html", "both"], default="pdf",
                   help="pdf, a single self contained html page, or both (default pdf)")
    p.add_argument("--n-boot", type=int, help="bootstrap resamples (default 1000, fewer is faster)")

    p = add("sweep", _sweep, "Backtest a grid of parameters and show a table and a heatmap.",
            epilog="--grid takes a rule parameter and its values: fast=20,50,100 or slow=100:300:50 "
                   "(start:stop:step). Use a dotted name for other settings, like execution.delay=0,1,2 "
                   "or sizing.vol_target=0.08,0.10,0.12. With one or two parameters you get a heatmap.")
    _config_args(p)
    p.add_argument("--grid", action="append", metavar="NAME=VALUES",
                   help="parameter values to try, repeat for more parameters")
    p.add_argument("--metric", choices=list(sw.METRICS), default="sharpe", help="what to show and rank by (default sharpe)")
    p.add_argument("--heatmap", help="where to save the figure (default reports/<name>/sweep_<metric>.png)")
    p.add_argument("--no-plot", action="store_true", help="skip the figure")
    p.add_argument("--out", help="write the full table to this csv")

    p = add("walkforward", _walkforward, "Pick parameters on a training window, trade them on the next one, repeat.",
            epilog="Each test window uses the grid cell that scored best on the training window before it, "
                   "so the stitched result is out of sample. Switching between cells is assumed free.")
    _config_args(p)
    p.add_argument("--grid", action="append", metavar="NAME=VALUES", help="same as sweep, falls back to [report] sweep")
    p.add_argument("--metric", choices=list(sw.METRICS), default="sharpe", help="how the training window ranks cells")
    p.add_argument("--train-years", type=float, default=10, help="training window length (default 10)")
    p.add_argument("--test-years", type=float, default=1, help="test window length (default 1)")
    p.add_argument("--expanding", action="store_true", help="train on everything before the test window, not a fixed length")
    p.add_argument("--folds", action="store_true", help="print every window and what it chose")
    p.add_argument("--out", help="write the windows to this csv")
    p.add_argument("--plot", help="save a figure to this png")

    p = add("montecarlo", _montecarlo, "Range of future outcomes from block bootstrapped history.")
    _config_args(p)
    p.add_argument("--years", type=int, default=10, help="path length (default 10)")
    p.add_argument("--paths", type=int, default=10_000, help="number of paths (default 10000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the table to this csv")

    p = add("today", _today, "What the 50/200, 10 month and 12 month momentum rules say on the latest data.")
    p.add_argument("--live", action="store_true", help="download fresh prices (needs internet and yfinance)")
    p.add_argument("--csv", action="append", metavar="FILE", help="also check a csv of your own prices, repeatable")
    p.add_argument("--json", help="write the states to this json file")

    add("assets", _assets, "List the assets in the data snapshots and the dates they cover.")
    add("rules", _rules, "List the trading rules and their parameters.")

    p = add("init", _init, "Write a starter config you can edit.")
    p.add_argument("path", nargs="?", default="my_strategy.toml")
    p.add_argument("--csv", help="start from your own csv of prices instead of a built in asset")
    p.add_argument("--force", action="store_true", help="overwrite the file if it exists")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except BrokenPipeError:  # output piped into head and the like
        sys.stderr.close()
    except (ValueError, OSError) as e:
        ap.exit(2, f"trendlab: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
