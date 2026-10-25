# Using trendlab

This is the reference for the command line and the config files. The README has the research. This page is for running your own tests.

## Install

```bash
pip install -r requirements.txt     # pinned versions the study was run with
pip install -e .                    # optional, gives you a `trendlab` command
```

Without the second line everything works as `python -m trendlab <command>` from the repo folder. The data snapshots live in `data/snapshots/`, so run it from a clone and use an editable install, not a plain one.

## Commands

| Command | What it does |
|---|---|
| `backtest` | Headline numbers for a config, or for a csv of prices with `--rule` |
| `tearsheet` | Full report with bootstrap intervals, regimes, the smile and Monte Carlo, as pdf, html or both |
| `sweep` | Backtest a grid of parameters, print a table and save a heatmap |
| `walkforward` | Pick parameters on a training window, trade them on the next window, repeat |
| `montecarlo` | Range of 10 year outcomes from block bootstrapped history |
| `today` | What the 50/200, 10 month and 12 month momentum rules say on the latest data |
| `assets` | The assets in the snapshots and the dates they cover |
| `rules` | The trading rules and their parameters |
| `init` | Write a starter config |

`trendlab <command> --help` lists every option for that command.

These options work on every command that takes a config.

* `--set section.key=value` changes one value for this run, for example `--set execution.cost_bps=25` or `--set rule.fast=20`. Repeat it as needed.
* `--rule`, `--column` and `--kind` only matter when you pass a csv of prices instead of a config (see below). `--kind returns` says the column holds decimal returns.

### Backtest

```bash
python -m trendlab backtest configs/us_century_50_200.toml
python -m trendlab backtest configs/us_century_50_200.toml --out daily.csv --json numbers.json --plot chart.png
```

It prints the strategy against buy and hold on the same assets. The strategy trades one day after its signal, pays the cost on every position change and earns the T-bill rate on whatever isn't invested. `--out` saves the daily returns, `--json` the numbers and `--plot` a growth and drawdown chart.

### Your own prices

Any csv with a date column and a price column works. Yahoo, Stooq, Tiingo and most broker exports are fine. Use adjusted prices if you want total returns.

```bash
python -m trendlab backtest btc.csv --rule ma_crossover --set rule.fast=20 --set rule.slow=100
python -m trendlab init btc.toml --csv btc.csv      # same thing as a config you can edit
```

The first column is the date unless you say `date_column`. If the file has one price column it is used. With several, the first of `Adj Close`, `Close`, `Price`, `Last` or `Value` is used, or you name it with `column`. Commas, semicolons and tabs are all read. Rows can be in any order. Repeated dates, zero or negative prices and unreadable dates stop the run with a message that names the file and the problem.

In a config the file goes under `[data.files]`. Paths are relative to the config file.

```toml
[data]
assets = ["btc", "us"]          # your own names and built in ones mix freely
cash = "tbill"                  # or a flat annual rate such as 0.03, or "none"

[data.files.btc]
path = "btc.csv"
column = "Adj Close"            # optional
kind = "price"                  # or "returns" if the column holds decimal returns, 0.01 is 1%
date_column = "Date"            # optional
```

`configs/custom_prices.toml` is a working example. A few things to know about your own data.

* Windows are counted in rows, not in calendar time. A 200 row average on an asset that trades seven days a week covers 29 weeks, not 40.
* Cash earns the daily T-bill rate by default. Past the end of the T-bill data (August 2026 in the snapshots) the last known rate carries on. For another currency use a flat annual rate.
* A last month that isn't finished is left out of the monthly numbers.

### Sweeps

```bash
python -m trendlab sweep configs/us_century_50_200.toml --grid fast=20,50,100 --grid slow=150,200,250
python -m trendlab sweep configs/us_century_50_200.toml --grid fast=10:100:10 --metric calmar
python -m trendlab sweep configs/us_century_50_200.toml --grid execution.delay=0,1,2,3 --grid execution.cost_bps=0,5,25,50 --metric max_dd
```

A grid value is a list (`20,50,100`) or a range (`start:stop:step`, both ends included). A bare name is a rule parameter. A dotted name reaches the other tables, so `execution.delay`, `execution.cost_bps` and `sizing.vol_target` work too. Without `--grid` it uses the `sweep` table under `[report]` in the config. Combinations where `fast` is not below `slow` are skipped.

Every cell is scored on the same dates, starting once the slowest cell has enough history, so a long window isn't compared with a short one on a different period. The metric is one of `sharpe`, `cagr`, `calmar`, `max_dd`, `ulcer` and `vol`. With one or two parameters you also get a figure (`--heatmap path.png`, or `--no-plot`) with the config's own cell outlined in red and buy and hold in the title. `--out` saves the full table.

When the grid changes the rule itself, the output ends with the deflated Sharpe probability of the best cell and the probability of backtest overfitting across the cells, so you see how much picking the best of N flatters it. Execution grids (delay and cost) are sensitivity checks, not a search, so they skip that line.

### Walk forward

```bash
python -m trendlab walkforward configs/us_century_50_200.toml --grid fast=20,50,100 --grid slow=150,200,250
python -m trendlab walkforward prices.csv --grid fast=10,20,50 --grid slow=100,200 --train-years 5 --expanding --folds
```

At the end of each training window it picks the cell with the best metric, then holds that cell through the next test window. The test windows are stitched together into one out of sample series and compared with the config's own fixed rule, the average cell and buy and hold. The last line shows the mean training score of the picks next to their mean test score. A big gap between the two means the grid was fitted to noise.

Defaults are 10 years of training and 1 year of testing, rolling. `--expanding` trains on everything before the test window. `--folds` prints every window and what it chose, `--out` saves them and `--plot` draws the equity curves and the chosen parameters over time. Signals only use the past, so slicing each cell's full history gives the same numbers as running it live. Switching from one cell to another is assumed free, which flatters the result a little.

### Tearsheet

```bash
python -m trendlab tearsheet configs/spy_50_200.toml --format both
```

`--format pdf` is the default. `html` writes one self contained page with the figures inside it, which you can email or open anywhere. `both` writes both. The summary numbers always go to `summary.json` next to them, in `reports/<name>/` unless you pass `--outdir`. `--n-boot` sets the number of bootstrap resamples (1000 by default, 200 is plenty for a first look). If the config has a `[report] sweep` table the report adds the deflated Sharpe and the probability of backtest overfitting across it.

`reports/example/` has the report for the main study rule in both formats.

### Today

```bash
python -m trendlab today
python -m trendlab today --csv mine.csv --live
```

States of the 50/200, 10 month and 12 month rules on the last date of each series. `--live` pulls fresh ETF prices from Yahoo (needs `yfinance` and internet). `--csv` adds your own file. This is research output, not a signal service and not advice.

## Config reference

A config is a TOML file. `[data]` and `[rule]` are required, the rest has defaults. `trendlab init` writes a starter.

### `[data]`

| Key | Default | Meaning |
|---|---|---|
| `assets` | required | Names from `trendlab assets`, or names you define under `files` |
| `files` | none | Your own csv files, see above |
| `cash` | `"tbill"` | What cash earns. `"tbill"`, `"none"` or a flat annual rate like `0.03` |
| `start`, `end` | all data | Cut the raw data to these dates before the rule sees it |
| `eval_start` | automatic | First day that counts in the results. The default is the first month start after every rule has enough history |

`start` changes what the rule sees. `eval_start` only changes what is scored, so the rule still has its history. For fair comparisons across sweeps and walk forward the later of the cells' start dates is used.

### `[rule]`

`rule` picks one of the following. `trendlab rules` prints the same list with defaults.

| Rule | Parameters | What it does |
|---|---|---|
| `ma_crossover` | `fast = 50`, `slow = 200` | In when the fast average is above the slow one |
| `ema_crossover` | `fast = 50`, `slow = 200` | Same with exponential averages |
| `price_above_ma` | `window = 200` | In when the close is above its own average |
| `sma_month` | `months = 10` | Faber's rule, checked at each month end |
| `tsmom` | `lookback = 12`, `long_only = true` | Time series momentum, in when the past return beats T-bills. `long_only = false` shorts the losers |
| `breakout` | `n_entry = 100`, `n_exit = 50` | Donchian channel, in on a new high, out on a new low |

Every rule also takes `vol_target` (annual volatility such as `0.10`), `cap` (largest position, default 1) and `com` (volatility window in days, default 60). A misspelled parameter stops the run and lists the ones that exist.

### `[execution]`

| Key | Default | Meaning |
|---|---|---|
| `delay` | `1` | Trading days between the signal close and the close you trade at. 0 trades at the close the signal was computed from and is only there to measure how much that matters |
| `cost_bps` | `5` | Paid on every unit of position change, in basis points |

With `delay = 1` the signal on day t is traded at the close of t + 1 and the first return earned is day t + 2's.

### `[sizing]`

| Key | Default | Meaning |
|---|---|---|
| `vol_target` | none | Scale the position to this annual volatility, using an exponentially weighted estimate known at the signal close |
| `max_leverage` | `1.0` | Largest position when vol targeting. Above 1 borrows at the T-bill rate |
| `vol_window` | `60` | Center of mass of the volatility estimate, in days |
| `fraction` | `1.0` | Multiply every position by this, for example 0.5 to run half size |

These are the same as the rule's own `vol_target`, `cap` and `com`, and can't be set in both places. `configs/vol_targeted_trend.toml` shows it.

### `[portfolio]`

`weights = "equal"` or a table like `{ us = 0.6, ust10 = 0.4 }`. Each asset runs its own rule and the sleeves are combined with these weights, rebalanced at every month end. The benchmark is the same weights held without the rule.

### `[report]`

| Key | Default | Meaning |
|---|---|---|
| `n_boot` | `1000` | Bootstrap resamples in the tearsheet |
| `mean_block` | `12` | Mean block length in months for the stationary bootstrap |
| `seed` | `0` | Seed for every random draw, so a report reruns to the same numbers |
| `sweep` | none | Table such as `{ fast = [20, 50, 100], slow = [150, 200, 250] }`, used by `sweep` and `walkforward` when there is no `--grid`, and by the tearsheet for the deflated Sharpe and PBO |

## From Python

```python
from trendlab import config as cf, sweep

cfg = cf.load("configs/us_century_50_200.toml", overrides=["execution.cost_bps=25"])
res = cf.run(cfg)                       # daily strategy, benchmark and T-bill returns
print(cf.headline(res))

runs = sweep.run_grid(cfg, {"fast": [20, 50, 100], "slow": [150, 200, 250]})
print(sweep.table(runs))
wf = sweep.walk_forward(runs, "sharpe", train_years=10)
print(wf["summary"])
```

`cf.run` also takes a `loader(name)` that returns daily returns and daily cash returns for an asset, which is how the tests run the whole pipeline on synthetic data.

## When something stops

Every error that comes from the inputs is one line starting with `trendlab:` and a nonzero exit code. The usual ones.

* `config file not found` or `price file not found`. Check the path. Paths inside a config are relative to the config.
* `unknown assets`. Names come from `trendlab assets` or from `[data.files]`.
* `rule ma_crossover has no parameter 'slw'`. The message lists the real ones.
* `has N days in the chosen date range and the rule needs about M`. The data, after `start` and `end`, is too short for the rule. Widen the range or shorten the windows.
* `only N months of data, a 10 year training window plus 1 year test needs M`. Shorten `--train-years`.
