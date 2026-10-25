# Deviations from the protocol

Anything I did differently from `docs/protocol.md` or `docs/protocol_addendum.md`, with the date and why. None of these change the primary test, its samples or its verdict rule. Everything listed here was decided after the protocol was frozen, so it's exploratory unless it's a pure bug fix.

## 2026-10-20. Return matching when trend beats every static mix

The protocol matches trend with the static mix that has the same CAGR. On commodities (DBC) and Japan the trend strategy grew faster than every mix of the asset and T-bills, so no such mix exists. The protocol didn't say what to do here. In that case (and in any bootstrap resample where it happens) I compare against the mix with the highest CAGR instead, and the result is flagged `unreachable` in `results/study.json`. Trend already wins on growth there, so this only decides whether it also wins on drawdown. It doesn't touch C1 or C2.

## 2026-10-22. Extras added after I had seen the quick run

1. Secondary tail measures for the primary comparison. Max drawdown is decided by one episode, so next to it I also report the Ulcer index (the root mean square of the drawdown over the whole path), CVaR 5% and the worst 12 month return, each as trend minus the same growth mix, with the same bootstrap. They are in `results/study.json` under `primary.secondary`.
2. A gold data check. I ruled out the World Bank monthly gold series in the protocol because it is made of monthly averages. To show the problem is real, `gold_averaging_check` compares the 10 month rule on the averaged series with the same rule on GLD month end prices over the same months, and measures what averaging does to the autocorrelation of returns.
3. The smile without its biggest quarter. The quarterly smile fit for US stocks is pulled around by one quarter, 1933 Q2, when the market rose about 91% and the 50/200 rule, which was out for most of it, made about 25%. So I also report the curvature with that single most extreme quarter dropped (`smile.gamma_without_biggest_quarter`).

## 2026-10-22. Things I ran that the addendum didn't list

The addendum said what to run. A few things I reported go beyond it, so here they are in one place.

* Data snooping on C2 as well as C1. A5 only names C1. I ran the same family, tests and settings on developed markets outside the US.
* The Reality Check and SPA against buy and hold, next to the protocol's comparison against each rule's same volatility mix.
* In the Monte Carlo (A7), besides ending wealth, worst drawdown and the chance of losing money, I report the chance of a drawdown over 30%, the chance that trend ends ahead of buy and hold and of the same volatility mix, and the chance that trend's worst drawdown is smaller than the mix's. That last one came out at about 50%, a coin flip, and it's in the README and the paper.
* The diversified time series momentum portfolio (A3) starts once at least two markets have a signal, which means 1963 when the Treasury series joins US stocks. The addendum didn't say when to start it.

## 2026-10-22. Fixes after an outside review

An independent review of the whole repo found the problems below. Each is fixed and everything was rerun from a clean state.

1. T-bill returns. The daily RF column in the Ken French file is the monthly rate spread over the days and rounded to 0.01% a day. In some years that rounding is off by about a point (1949, 1950, 2002, 2003, 2025), and before 2000 it understated cash by about 0.14 points a year on average. I now spread each month's RF from the Ken French monthly file evenly over that month's trading days without rounding (`data_sources.rescale_daily_rf`), so the days compound to exactly the monthly number. This is a data fix, not a design change, but it moves every number a little.
2. Bear markets dated from local peaks. The protocol dates a bear market from the all time high, so a second fall inside a long slump doesn't count until the old high is regained. In the US that hides 1937 to 1938 (buy and hold down about half) and 1940 to 1942, because the 1929 peak wasn't regained until 1945. Those months count as "recovery" under the protocol. I kept the protocol definition for every headline number and added a sensitivity check that dates bear markets the usual way, a 20% fall from a local peak that ends at the low before a 20% rise (`insurance.swing_episodes`). It's under `insurance_swing_*` in the results and in its own table in the paper.
3. The 12 months after a crash. The events table had "window plus 12 months", which starts at the start of the window. I added the 12 months after the window ends, which is what "after the low" means, and the README now uses that.
4. The time series momentum portfolio double counted non US stocks. It held developed ex US and also Europe, Japan and Asia Pacific, which make up developed ex US. I dropped developed ex US, so it is seven markets in four asset classes.
5. Smile quarters. The fit now uses full calendar quarters only. Before, a sample that started in September added a one month quarter.
6. Warm up for the data snooping family. The slowest rules need 300 days of history, not 200, so the family's evaluation now starts once all of them have a position.
7. Turnover. Switches per year are now divided by calendar years. Before 1953 the US file has Saturday sessions, so dividing by 252 trading days understated turnover by about 15% in those years.
8. Paper tables. `paper_numbers.py` now always reads the saved `results/study.json`, so running it on its own gives exactly the tables the full run gives.
