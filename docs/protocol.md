# Protocol for the protection study

Written 2026-10-20, before I ran any trend rule on any of the new data below. Its SHA-256 was saved to `results/protocol_sha256.txt` at the same time, and `run_study.py` checks it before it runs. If I change anything after seeing results, it goes in `docs/deviations.md` with the date and the reason, and this file stays as it is.

## Question

Is trend following economically useful as portfolio protection even when it underperforms buy-and-hold in absolute returns?

My first study already showed the 50/200 crossover loses to holding SPY on return. That doesn't settle much. A rule that sits in cash part of the time should grow less money. Holding 70% stocks and 30% T-bills also grows less money and also has smaller drawdowns. So the real question is whether trend following buys protection more cheaply than just owning less stock.

## What counts as "useful"

For one market, take the trend strategy and find the static mix of the asset and T-bills that grew at exactly the same long run rate (same CAGR). Then compare their worst drawdowns.

```
delta = maxDD_depth(trend) - maxDD_depth(return matched mix)
```

Depth is a positive number (0.40 means down 40% from the peak). If delta is below zero, trend following gave a shallower worst drawdown for the same growth, so the protection was cheaper than de-risking. If delta is zero or above, you could have had the same protection by holding less of the asset.

Uncertainty comes from a stationary block bootstrap of the monthly returns (Politis and Romano 1994), mean block length 12 months, 2,000 resamples, seed 0. Trend, asset and T-bill returns are resampled together, and the matching weight is solved again inside every resample.

### Verdict rule, fixed now

The primary verdict uses the 50/200 rule on the two untouched equity samples, C1 and C2 below.

| Verdict | Condition |
|---|---|
| Supported | delta < 0 in both C1 and C2, and delta < 0 in at least 90% of bootstrap resamples in each |
| Mixed | delta < 0 in both but the 90% bar fails in one or both, or delta < 0 in only one |
| Not supported | delta >= 0 in both |

Everything else in the study is reported but is not part of the verdict.

## What I have already seen

SPY from 2000-01-03 to 2026-06-10 was used to pick 50/200 and then looked at many times in the first study. It is not clean confirmation anymore. Any number from it is labeled "development" and is context only. The US market from Ken French over 2000 to now overlaps it, so it gets the same label.

## Untouched evaluation data

| ID | Asset | Source | Data starts | Role |
|---|---|---|---|---|
| C1 | US stock market, 1926 to 1999 | Ken French daily `Mkt-RF + RF` (CRSP value weighted, dividends included) | 1926-07-01 | primary |
| C2 | Developed markets ex US, in USD | Ken French daily `Developed_ex_US_3_Factors` (`Mkt-RF + RF`) | 1990-07-02 | primary |
| C3 | US 10 year Treasury | built from FRED `DGS10` daily yields (method below) | 1962-01-02 | other asset class |
| C4 | Gold | GLD adjusted close from Yahoo | 2004-11-18 | other asset class |
| C5 | Broad commodities | DBC adjusted close from Yahoo | 2006-02-06 | other asset class |
| X1 to X3 | Europe, Japan, Asia Pacific ex Japan | Ken French daily regional files, USD | 1990-07-02 | regional detail |

Every sample ends on the last day of the Ken French daily file, because that is where the T-bill series ends. C1 ends on 1999-12-31 on purpose.

I checked that these files download and looked at their date ranges and column names. I have not computed any strategy result on them.

Before running anything I also ruled out two sources. Monthly gold prices from the World Bank pink sheet are monthly averages, and averaging makes a price series look trendier than it is, which would flatter exactly the thing I'm testing. Spot oil from EIA is not something you can invest in, since futures roll costs are a big part of commodity returns. So gold and commodities only use ETFs, which means they only start in the 2000s.

## Bond returns from yields

FRED gives yields, not returns. Each day I assume you hold a 10 year par bond bought at yesterday's yield `y0` (so its coupon is `y0`), and reprice it at today's yield `y1` with maturity shortened by the time that passed. Return equals the price change plus coupon accrued over the actual calendar days. Semiannual coupons, like the Treasury convention.

Before the bond series is used I check it against IEF (iShares 7 to 10 year Treasury ETF, adjusted close from Yahoo, 2002 on) and against Damodaran's annual 10 year Treasury returns. I fix the pass bar now. Monthly correlation with IEF of at least 0.95, and a CAGR gap within 1 percentage point a year. If it fails, bond results are reported as unvalidated.

The 60/40 benchmark needs bonds back to 1927. Before 1953-04 I use Shiller's monthly long term government yield, and from 1953-04 to 1961-12 FRED `GS10`, both run through the same model monthly. From 1962 on I use the daily `DGS10` model. These monthly yields are averages over the month, which smooths returns a bit. That only touches the 60/40 benchmark, never a trend signal.

## Rules (no tuning, nothing picked from these samples)

All signals are built on the total return index of the asset. The primary rule is the one I froze in the first study.

| Rule | Hold the asset when | Otherwise |
|---|---|---|
| 50/200 (primary) | 50 day average is above the 200 day average | T-bills |
| 10 month average | at a month end, the index is above the average of its last 10 month end values | T-bills |
| 12 month momentum | at a month end, the past 12 month return beats the past 12 month T-bill return | T-bills |

The second and third are standard rules from Faber (2007) and Moskowitz, Ooi and Pedersen (2012). They are reported next to 50/200 as a check, never used to pick a winner.

## Execution and cash

* The signal uses day t's close. The trade happens at day t+1's close. The position earns returns from there on. This is a delay of 1 day.
* Cash earns the daily T-bill return from the Ken French file whenever the position is out.
* Each unit of position change pays 5 bps.
* Sensitivity: delay of 0, 1, 2, 3, 5 and 10 days. Costs of 0, 5, 10, 25 and 50 bps. Turnover is reported as switches per year and share of days invested.

## Benchmarks

| Name | What it is |
|---|---|
| Buy and hold | 100% the asset |
| Vol matched | w in the asset, 1 - w in T-bills, rebalanced monthly, w chosen so monthly volatility equals the trend strategy's |
| Return matched | same mix but w chosen so CAGR equals the trend strategy's. w above 1 means borrowing at the T-bill rate, and I flag it |
| Exposure matched | w equals the share of days the trend strategy was invested |
| 60/40 | 60% equities, 40% US 10 year Treasury, rebalanced monthly (equity samples only) |
| 60/40 with trend | same, but the equity 60% runs the 50/200 rule (equity samples only) |

## Metrics

Daily returns are compounded into calendar months and every metric below uses monthly returns, so daily and monthly data are treated the same way. The evaluation window starts at the first full calendar month in which all three rules already have a position. Data before that is only used to warm up the signals.

CAGR, annual volatility, Sharpe ratio over T-bills, max drawdown, worst 12 month return, CVaR 5% (average of the worst 5% of months), and skewness of monthly returns.

## Insurance accounting

A bear market is a peak to trough window of buy and hold where the drop reaches 20% (10% for the Treasury series, since its volatility is about half). For each of these I record the payout, which is the log return of trend minus the log return of the comparison from peak to trough. All other months are normal time, and the premium is the average monthly log return gap there, times 12. This is done against buy and hold and against the vol matched mix.

Uncertainty is a sign test across episodes and a 90% stationary block bootstrap interval (12 month blocks, 2,000 resamples) for the premium, the payout per bear year, and the net gap.

## Regimes

All of these are labels on the evaluation months, decided after the fact. They are never used for trading.

| Regime | Definition |
|---|---|
| Bear | inside a peak to trough window as above |
| Recovery | from the trough back to the old peak |
| Crash months | worst 5% of buy and hold monthly returns |
| Calm | months where buy and hold's realized daily volatility within the month is in the lowest third |
| High rates / low rates | T-bill yield above or below the sample median |
| Rising / falling long rates | 12 month change in the US long term yield above zero, or not |

For each regime I report the average monthly return gap (trend minus buy and hold, and trend minus vol matched) with a 90% block bootstrap interval.

## What this can't show

Even with a clean result, these are a handful of markets that share a lot of history. C2 overlaps the period I already studied, and big crashes like 2008 hit every equity market at once. With a few bear markets per sample, everything about bear markets is a small sample. I'll say so next to the numbers.
