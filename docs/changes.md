# What changed from the first version

## Two bugs in the first version

**Trading at the signal's own close.** The first version shifted the signal by one day and earned close to close returns. That sounds right, but it means the position that earns day t+1's return was bought at day t's close, which is the same close the moving averages were computed from. You can't know a closing price and trade at it. The README said the trade happened at the next day's close, and it didn't. Now `ma_crossover_positions` has a `delay` argument. `delay=1` (the default) trades at the next close and shifts the signal by two days. `delay=0` reproduces the old behavior so the difference can be measured.

**Cash earned nothing.** When the rule was out of the market it earned 0%. It should earn the T-bill rate. Now `run_backtest` takes `rf`, the daily T-bill return from the Ken French data library (`bc.load_rf`), and Sharpe ratios are measured over T-bills. The daily T-bill numbers are rescaled so each month matches Ken French's monthly rate exactly (see `docs/data.md`).

## What the fixes did to the old headline

The 50/200 rule on SPY, out of sample from 2018-01-02 to 2026-06-10, 5 bps a trade. From `results/spy_study.json`, made by the notebook.

| Version | Strategy CAGR | Sharpe | Growth of $1 | SPY CAGR | SPY Sharpe | SPY growth of $1 |
|---|---|---|---|---|---|---|
| Old, trade at the signal close, cash at 0% | 9.2% | 0.63 (over 0%) | $2.10 | 14.3% | 0.79 (over 0%) | $3.09 |
| Cash earns T-bills | 9.6% | 0.49 | $2.17 | 14.3% | 0.66 | $3.09 |
| And trade at the next close (now) | 9.3% | 0.47 | $2.12 | 14.3% | 0.66 | $3.09 |

The worst drawdown is 33.7% in every version, the same as SPY's, because the rule was still invested through the March 2020 crash.

The Sharpe ratios drop mostly because they are now measured over T-bills instead of over zero. That is the right way to do it, and it hits buy and hold the same way. On returns the two fixes almost cancel. The conclusion of the first study (the rule loses to buy and hold out of sample) didn't change.

With the fixed engine the in sample search in the notebook picks 20/200 instead of 50/200 (50/200 is a close second), so the notebook's out of sample section now tests 20/200. The PBO across the 89 pairs went from 0.57 to 0.63. All of this is in the notebook. The protection study uses 50/200 as its primary rule because that is the rule I froze in the first study, and I wrote that down in the protocol before running anything.

## Everything else that's new

* A precommitted protection study (`docs/protocol.md`, `run_study.py`, `paper/`) on data the first study never touched.
* `trendlab/`, a small library with a command line, config files, a tearsheet, Monte Carlo and a "what does the rule say today" script.
* Tests for no lookahead, cost and cash accounting, the bond pricer, matching weights and the statistics, plus CI.
* Pinned requirements, a Makefile and data snapshots, so `make all` reruns everything offline.

`index.html` (the live report) is still the old notebook export with the old numbers. It will be regenerated separately.

## Command line additions (trendlab 0.3)

None of these touch the study. The numbers in `results/study.json` and the paper come from the same code paths as before, and I checked that the daily returns of every example config and the signal of every rule on every asset are identical to the earlier version.

* Your own prices. `[data.files]` in a config, or `trendlab backtest prices.csv`, reads any csv with a date and a price column. Cash can be the T-bill series, a flat annual rate or nothing.
* `sweep` runs a grid of parameters with a table, a heatmap and the deflated Sharpe and PBO of the best cell. `walkforward` picks parameters on a training window and trades them on the next one.
* `[sizing]` with vol targeting, a leverage cap and a fixed fraction. Two new rules, `ema_crossover` and `price_above_ma`, both covered by the no lookahead tests.
* `--set section.key=value` on every command, `init`, `rules`, a clearer `assets`, and `--help` text that explains the options. Config mistakes stop with one readable line.
* Tearsheets as html as well as pdf. The month that is not finished yet is now judged by "no weekday left in the month", so data that ends on a Saturday before 1953 isn't cut a month short. It doesn't change any number in the study because the data ends on a Monday.
* `docs/usage.md` has the config reference.

