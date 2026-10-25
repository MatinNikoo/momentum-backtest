# Protocol addendum

Written 2026-10-20, after `docs/protocol.md` was frozen and the data was downloaded, but still before any trend rule was run on the new data. Its SHA-256 is saved in `results/protocol_sha256.txt` next to the first one. Nothing here changes the primary test or its verdict rule. Everything below is secondary and reported whatever it shows.

## A1. One framework for the whole century

The US file covers 1926-07 to now. Besides the C1 window (to 1999), I run the same rules, costs and delays on the full US history and report named episodes in one table. The 1929 crash, World War 2, the 1970s, 1987, 2000 to 2002, 2008, 2020 and 2022. Anything after 1999 stays labeled development because it overlaps SPY.

## A2. Emerging markets

Ken French has emerging markets only at monthly frequency (from 1989-07). The 50/200 rule needs daily data, so emerging markets only get the two month end rules. With monthly data the signal at a month end is traded at that same close, so this is a delay of 0 days and I'll flag it.

## A3. Time series momentum with volatility scaling

Following Moskowitz, Ooi and Pedersen (2012).

* Volatility estimate. Exponentially weighted variance of daily returns with center of mass 60 days, annualized with 261, using data up to the signal day only.
* Long or cash version for protection. At each month end, if the 12 month excess return is positive, hold `min(1, 10% / vol)` of the asset and the rest in T-bills, else all T-bills. No leverage.
* Diversified long/short version, as in the paper. At each month end each available asset gets `sign(12 month excess return) * 40% / vol`, and the portfolio is the average over available assets, financed at the T-bill rate. It uses the daily assets (US, developed ex US, Europe, Japan, Asia Pacific ex Japan, 10 year Treasury, gold, commodities) as they become available.

## A4. Convexity, the "trend following smile"

Quarterly excess returns of the strategy against quarterly excess returns of the asset, fit with the Treynor and Mazuy (1966) regression

```
r_strategy = a + b * r_asset + g * r_asset^2 + e
```

g > 0 is the smile. Two checks on whether it is real.

* Stationary block bootstrap of quarters (mean block 4 quarters, 2,000 resamples) for an interval on g.
* Placebo timing. Shift the whole position series in time by a random offset of at least 252 days (wrapping around), rerun the strategy, and refit g, 1,000 times. That keeps the same time in the market and the same switching pattern but breaks any link to what the market does next. The p-value is the share of placebo g at or above the real g.

## A5. Data snooping across every rule I tried

On C1 (US 1927 to 1999, monthly), the family is every rule this repo has touched. The 89 moving average pairs from the first study, the 10 month average, 12 month momentum, the vol scaled long or cash momentum, and breakout rules (in on an N day high, out on an N/2 day low, N in 50, 100, 200, 250). For rule k the loss differential is

```
d_k = r_rule_k - r_volmatched_k
```

so the null is that no rule beats the passive mix with the same volatility on average. I report White's (2000) Reality Check p-value and Hansen's (2005) SPA p-value (consistent version), using the stationary bootstrap, mean block 12 months, 2,000 resamples, seed 0. I also report the deflated Sharpe ratio of the best rule in the family and the PBO of picking the best rule by in sample Sharpe of `d_k` (16 blocks).

## A6. Insurance in dollars

Same bear markets as the protocol. Per $100,000 held at the start of each bear market, how many dollars the trend strategy ended ahead of (a) buy and hold and (b) the vol matched mix at the trough. And per $100,000, the dollars a year it gave up in normal time. Each with a 90% bootstrap interval.

## A7. Range of future outcomes

A Monte Carlo that resamples the US 1927 to 2026 monthly history in 12 month blocks (keeping trend, market and T-bill returns from the same months together) into 10,000 ten year paths. It shows the spread of ending wealth, worst drawdown and chance of losing money for buy and hold, the 50/200 rule and the vol matched mix. This assumes the future looks like a reshuffled past, which is a strong assumption.
