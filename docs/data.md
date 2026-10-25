# Data

Everything the study uses is in `data/snapshots/`, built by `python data_sources.py`. The raw downloads go to `data/raw/`, which is not in the repo. If you rebuild the snapshots you'll get newer data and the numbers will move a little, so the committed snapshots are the ones the results come from. They end on 2026-08-31 for the T-bill series (the last day in the Ken French file when I downloaded it on 2026-10-20). The ETF prices go to 2026-10-19 but the study cuts everything at the T-bill end.

## Files

| File | What's in it |
|---|---|
| `daily_returns.csv.gz` | Daily total returns (decimal) for every daily asset, one column each, plus the daily T-bill return `rf` and the 10 year yield `dgs10`. Outer joined on date, so a blank means that market was closed |
| `monthly_yields_and_bonds.csv` | Monthly long term Treasury yield (1871 on), the monthly bond return model built from it, emerging markets monthly return and its T-bill, and World Bank monthly average gold prices (only used to show why I didn't use them) |
| `damodaran_tbond_annual.csv` | Damodaran's annual 10 year Treasury returns, 1928 on. Only used to check my bond model |
| `../../spy_prices.csv` | The cached SPY adjusted closes from the first study, 2000-01-03 to 2026-06-10, unchanged |

## Sources

| Column | Source | Notes |
|---|---|---|
| `us` | Ken French data library, `F-F_Research_Data_Factors_daily` | `Mkt-RF + RF`. CRSP value weighted US market, all NYSE, AMEX and NASDAQ stocks, dividends included. From 1926-07-01 |
| `rf` | same file, `RF`, fixed with the monthly `RF` from `F-F_Research_Data_Factors` | One month T-bill return. The daily file spreads the monthly rate over the trading days and rounds it to 0.01% a day, which is off by up to about a point in some years. I spread each month's rate from the monthly file evenly over that month's trading days without rounding, so the days compound to exactly the monthly number. A few months in the 1930s had slightly negative T-bill returns and those are kept. This is what cash earns everywhere in the study |
| `dev_ex_us`, `europe`, `japan`, `asia_pac_ex_japan` | Ken French international daily 3 factor files | `Mkt-RF + RF`, in US dollars, from 1990-07-02. The RF in these files is the same US T-bill |
| `em` | Ken French `Emerging_5_Factors` (monthly) | There is no daily emerging markets file, so only month end rules run on it |
| `ust10` | FRED `DGS10` (10 year constant maturity Treasury yield, daily) | Turned into returns with the bond model below. From 1962-01-02 |
| `long_yield` | Shiller's long term government bond yield (via datahub.io's copy of his data) before 1953-04, then FRED `GS10` | Monthly averages. Only used for the 60/40 benchmark before 1962 and the "long rates rising or falling" regime |
| `gld`, `dbc`, `ief` | Yahoo Finance adjusted close through `yfinance` | GLD (gold, from 2004-11), DBC (diversified commodity futures, from 2006-02) and IEF (7 to 10 year Treasuries, from 2002-07, only used to check the bond model). Adjusted closes include distributions and are net of each fund's fees |
| `gold_avg_price` | World Bank pink sheet via github.com/datasets/gold-prices | Monthly AVERAGE prices, which is why they aren't used for any result. See below |
| Damodaran | pages.stern.nyu.edu/~adamodar, `histretSP` | Annual returns, only for the bond check |

The Ken French data is free for research use with credit to Kenneth R. French. FRED data is public. Yahoo data is for personal research use. I only commit small derived snapshots, not the raw files.

## Bond returns from yields

FRED gives a yield, not a return. Each day I assume you hold a 10 year bond bought at par at yesterday's yield $y_0$, so its coupon is $y_0$. Today it is repriced at today's yield $y_1$ with its maturity shortened by the days that passed, and it earns the coupon for those days.

```
r = P(y1; coupon = y0, maturity = 10 - dt) - 1 + y0 * dt,    dt = calendar days / 365
P(y; c, M) = c/2 * (1 - (1 + y/2)^(-2M)) / (y/2) + (1 + y/2)^(-2M)
```

Monthly yields use the same formula with `dt = 1/12`. The check I fixed in the protocol before using it was a monthly correlation with IEF of at least 0.95 and a CAGR gap within 1 point a year. It passed. The real numbers are in `results/study.json` under `bond_validation` and in `figures/bond_model_check.png`. The model is a bit more volatile than IEF, which makes sense since a 10 year par bond has a longer duration than a 7 to 10 year fund.

## Things that are wrong with this data, on purpose or not

* Before 1953 the US file has Saturday trading sessions, so a year has about 300 rows instead of 252. A 200 day average covers less calendar time then, which I left as is because that's how the market actually traded.
* Ken French returns are for an index nobody could buy before index funds. Trading it at 5 bps in the 1930s is not realistic. That's why the cost sensitivity goes up to 50 bps.
* The international files are in US dollars, so they include currency moves. That is what a US investor would get, unhedged.
* The bond series is a model, not a fund. It ignores bid ask spreads and the small differences between a constant maturity yield and a real bond.
* Before 1962 the bond series uses monthly average yields, which smooths returns. It only feeds the 60/40 benchmark.
* GLD and DBC start in the 2000s, so gold and commodities have only a few bear markets each (three for gold, one for commodities).
* Monthly average prices make a series look trendier than it is (Working 1960). On the World Bank gold series the lag one autocorrelation of monthly returns is about 0.2, while GLD month end returns have about none, and the 10 month rule looks much better on the averaged series than on the real one. The numbers are in `results/study.json` under `gold_averaging_check`.

## Your own data

The snapshots above are what the study uses. For anything else, `trendlab` reads a csv of prices or returns, see `docs/usage.md`. It takes the cash rate from the same daily T-bill series unless you give a flat rate.

