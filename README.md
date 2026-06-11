# Does a Simple Trend-Following Strategy Beat Buy-and-Hold?

A backtest of the classic moving-average crossover ("golden cross") strategy on
SPY, built to answer one question honestly: **after transaction costs, on data
the strategy was never tuned on, does it beat just holding the index?**

> **TL;DR:** Out-of-sample (2018–2026), the 50/200 crossover returned 8.4% CAGR with a 0.60 Sharpe vs 14.2% and 0.79 for buy-and-hold, with identical max drawdowns (-33.7%). After costs, on unseen data, the strategy did not beat the index — and the in-sample vs out-of-sample gap directly measures how much of its apparent edge was overfitting.

## Why this project

Most retail backtests are wrong in the same three ways. This one is built to
avoid them, and the code shows how:

1. **No lookahead bias** — signals computed from today's close are traded at
   tomorrow's close (`.shift(1)` in `backtest_core.py`).
2. **Transaction costs** — every position change pays 5 bps; nothing trades free.
3. **Out-of-sample testing** — parameters are chosen on 2000–2017 only, then
   judged once on 2018–present. The in-sample-vs-out-of-sample table directly
   measures how much of the "edge" was overfitting.

## Repo contents

| File | What it is |
|---|---|
| `backtest.ipynb` | The full analysis: data, methodology, results, honest conclusions |
| `backtest_core.py` | All strategy / cost / metrics logic, heavily commented |
| `test_backtest.py` | Sanity tests (lookahead shift, cost math, known-answer check) |
| `requirements.txt` | Dependencies |

## Run it

```bash
pip install -r requirements.txt
python test_backtest.py        # verify the engine (runs offline)
jupyter notebook backtest.ipynb
```

The first run downloads ~25 years of SPY data via `yfinance` and caches it to
`spy_prices.csv` so later runs are instant and reproducible.

## View it as a website

The notebook exports to a standalone HTML page:

```bash
jupyter nbconvert --to html backtest.ipynb --output index.html
```

To publish free on **GitHub Pages**: commit `index.html` to the repo,
then Settings → Pages → deploy from branch → root. Your report will be live at
`https://<username>.github.io/<repo>/`.

## Honest limitations

- Sharpe assumes a 0% risk-free rate; crediting cash yields would help the strategy.
- One asset, one historical window — results may not generalize.
- Fills at the close with a flat 5 bps cost; no taxes or slippage modeling.

## License

MIT
