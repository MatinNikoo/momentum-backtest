# Does a Simple Trend Following Strategy Beat Buy and Hold?

A backtest of the 50/200 day moving average crossover (the "golden cross") on SPY. I wanted to answer one question: after trading costs, on years the strategy was never tuned on, does it beat just holding the index?

**Live report:** https://matinnikoo.github.io/momentum-backtest/

**Short answer:** no. From 2018 to 2026 the strategy returned 8.4% a year with a 0.60 Sharpe. Buy and hold returned 14.2% with a 0.79 Sharpe. Both had the same worst drawdown, 33.7%. It looked a lot better on the years I tuned it on, so a good chunk of the edge was overfitting.

## How I tried to keep it honest

A lot of backtests online make the same three mistakes, so I built this one to avoid them.

1. **No lookahead.** A signal from today's close gets traded at tomorrow's close (the `.shift(1)` in `backtest_core.py`).
2. **Trading costs.** Every time the position changes it pays 5 bps.
3. **Out of sample testing.** I picked the parameters using 2000 to 2017 only, then tested once on 2018 to now. Comparing the two periods shows how much of the edge was overfitting.

## What's in the repo

| File | What it is |
|---|---|
| `backtest.ipynb` | The full analysis, from the data to the results and conclusions |
| `backtest_core.py` | The strategy, cost and metrics code, with comments |
| `test_backtest.py` | Unit tests for the lookahead shift, the cost math and a known answer check |
| `spy_prices.csv` | Cached SPY prices so runs are reproducible |
| `index.html` | The notebook exported as a web page, which is the live report |
| `requirements.txt` | Dependencies |

## Run it

```bash
pip install -r requirements.txt
python test_backtest.py        # checks the engine, runs offline
jupyter notebook backtest.ipynb
```

The first run downloads about 25 years of SPY data with `yfinance` and saves it to `spy_prices.csv`, so later runs are instant.

To rebuild the live report from the notebook:

```bash
jupyter nbconvert --to html backtest.ipynb --output index.html
```

## Limitations

* Sharpe assumes a 0% risk free rate. Counting interest on cash would help the strategy a little.
* It's one asset over one stretch of history, so the result might not hold elsewhere.
* Trades fill at the close with a flat 5 bps cost. No taxes or slippage.

## License

MIT
