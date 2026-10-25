"""
One command tearsheet for any config:

    python -m trendlab tearsheet configs/us_century_50_200.toml [--format pdf|html|both]

Writes reports/<name>/tearsheet.pdf and/or tearsheet.html plus summary.json.
The html file is a single page with the figures inside it, so it can be
emailed or opened anywhere.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import benchmarks as bm
from . import config as cf
from . import insurance as ins
from . import metrics as mt
from . import montecarlo as mc
from . import stats as st
from . import sweep as sw

TREND, BH, MIX = "#C44E52", "#5B6B8C", "#8C8C8C"


_monthly = cf.monthly


def compute(result, cfg, sweep_results=None):
    rep = cfg["report"]
    nb, blk, seed = int(rep["n_boot"]), int(rep["mean_block"]), int(rep["seed"])
    m = _monthly(result)
    s, b, c = m["strategy"].values, m["benchmark"].values, m["rf"].values
    w_vol = bm.vol_matched_w(s, b, c)
    mix = bm.mix(b, c, w_vol)
    out = {"name": result["name"], "start": str(m.index[0]), "end": str(m.index[-1]), "months": len(m),
           "rule": cfg["rule"], "assets": cfg["data"]["assets"], "weights": result["weights"],
           "execution": cfg["execution"], "w_vol_matched": w_vol,
           "metrics": {"strategy": mt.summary(s, c), "benchmark": mt.summary(b, c),
                       "vol_matched": mt.summary(mix, c)}}
    out["turnover"] = cf.turnover(result)

    # bootstrap bands
    shp = lambda x, y, z: mt.sharpe(x, z) - mt.sharpe(y, z)
    p, lo, hi, _ = st.bootstrap_ci(shp, [s, b, c], n_boot=nb, mean_block=blk, seed=seed)
    out["sharpe_diff_vs_benchmark"] = {"point": p, "ci90": [lo, hi]}

    def dd_delta(x, y, z):
        w, flag = bm.return_matched_w(x, y, z)
        if flag == "unreachable":
            w = bm.best_growth_w(y, z)
        return mt.max_dd_depth(x) - mt.max_dd_depth(bm.mix(y, z, w))
    p, lo, hi, draws = st.bootstrap_ci(dd_delta, [s, b, c], n_boot=nb, mean_block=blk, seed=seed)
    out["maxdd_vs_return_matched"] = {"point": p, "ci90": [lo, hi], "share_below_zero": float(np.nanmean(draws < 0))}

    # regimes against the benchmark and the same volatility mix
    eps = ins.episodes(b, 0.20)
    lab = ins.labels_from_episodes(len(b), eps)
    daily_b = result["benchmark"]
    rv = daily_b.groupby(daily_b.index.to_period("M")).std().reindex(m.index).values
    masks = {"bear": lab == 1, "recovery": lab == 2, "crash_months": b <= np.quantile(b, 0.05),
             "calm": rv <= np.nanquantile(rv, 1 / 3), "all": np.ones(len(b), bool)}
    out["regimes_vs_benchmark"] = ins.regime_gaps(s - b, masks, n_boot=nb, mean_block=blk, seed=seed)
    out["regimes_vs_volmatched"] = ins.regime_gaps(s - mix, masks, n_boot=nb, mean_block=blk, seed=seed)
    out["insurance_vs_volmatched"] = ins.account(s, mix, eps, n_boot=nb, mean_block=blk, seed=seed)

    # smile
    q = mt.to_quarterly(m)[(m.groupby(m.index.asfreq("Q")).size() == 3).values]  # full quarters only
    ys, xs = (q["strategy"] - q["rf"]).values, (q["benchmark"] - q["rf"]).values
    a0, b0, g0 = st.treynor_mazuy(ys, xs)
    _, glo, ghi, _ = st.bootstrap_ci(lambda y, x: st.treynor_mazuy(y, x)[2], [ys, xs],
                                     n_boot=nb, mean_block=4, seed=seed)
    out["smile"] = {"beta": b0, "gamma": g0, "gamma_ci90": [glo, ghi], "quarters": len(q)}

    # Monte Carlo
    wealth = mc.simulate(pd.DataFrame({"strategy": s, "benchmark": b, "vol_matched": mix}),
                         horizon=120, n_paths=min(10_000, 5 * nb), seed=seed)
    out["monte_carlo_10y"] = mc.summarize(wealth, ["strategy", "benchmark", "vol_matched"]).to_dict()

    # deflated Sharpe and PBO across the sweep, if there is one
    if sweep_results:
        R = pd.DataFrame({k: _monthly(r)["strategy"] - _monthly(r)["rf"] for k, r in sweep_results.items()}).dropna()
        srs = R.mean() / R.std()
        mine = R[_key(cfg["rule"])] if _key(cfg["rule"]) in R else R[srs.idxmax()]
        out["deflated_sharpe"] = st.deflated_sharpe(mine.values, srs.values)
        out["pbo"] = {k: v for k, v in st.pbo_cscv(R.values, n_blocks=16 if len(R) >= 160 else 8).items()
                      if k != "logits"}
        out["sweep_sharpes_annual"] = (srs * np.sqrt(12)).round(4).to_dict()
    return out, m, (xs, ys), wealth, mix


_key = sw.spec_key


def sweep(cfg, loader=None):
    grid = cfg["report"].get("sweep")
    if not grid:
        return None
    res = {}
    for params, c2 in sw.expand(cfg, grid):
        res[_key(c2["rule"])] = cf.run(c2, loader)
    return res


def _table(ax, df, title):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11)
    tb = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="upper left", cellLoc="right")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.25)


def _fmt_metrics(mets):
    names = {"strategy": "Strategy", "benchmark": "Buy and hold", "vol_matched": "Same volatility mix"}
    rows = {}
    for k, v in mets.items():
        rows[names.get(k, k)] = [f"{v['cagr']:.1%}", f"{v['vol']:.1%}", f"{v['sharpe']:.2f}", f"{v['max_dd']:.1%}",
                   f"{v['worst_12m']:.1%}", f"{v['cvar_5']:.1%}", f"{v['skew']:.2f}"]
    return pd.DataFrame(rows, index=["CAGR", "Volatility", "Sharpe", "Max drawdown", "Worst 12 months",
                                     "CVaR 5% (monthly)", "Skew"]).T


def figures(out, m, smile_xy, wealth, mix, table=True):
    s, b = m["strategy"].values, m["benchmark"].values
    t = m.index.to_timestamp()
    figs = []
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.6 if table else 0.01, 1.9, 0.9])
    ax0 = fig.add_subplot(gs[0])
    if table:
        _table(ax0, _fmt_metrics(out["metrics"]),
               f"{out['name']}   {out['start']} to {out['end']}   "
               f"{' '.join(f'{k}={v}' for k, v in out['rule'].items()).replace('rule=', '')}   "
               f"delay {out['execution']['delay']}d, cost {out['execution']['cost_bps']} bps")
    else:
        ax0.axis("off")
    ax1 = fig.add_subplot(gs[1])
    for x, lab, col, ls in [(s, "strategy", TREND, "-"), (b, "benchmark (same assets, held)", BH, "-"),
                            (mix, f"same volatility mix ({out['w_vol_matched']:.0%})", MIX, "--")]:
        ax1.plot(t, np.cumprod(1 + x), color=col, ls=ls, label=lab)
    sw.log_axis(ax1)
    ax1.legend(frameon=False)
    ax1.set_title("Growth of $1, after costs, cash at T-bills")
    ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    for x, col, ls in [(s, TREND, "-"), (b, BH, "-"), (mix, MIX, "--")]:
        ax2.plot(t, mt.drawdown(x), color=col, ls=ls, lw=1)
    ax2.set_title("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)
    
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    ex = pd.DataFrame({"s": s - m["rf"].values, "b": b - m["rf"].values}, index=t)
    roll = ex.rolling(36).mean() / ex.rolling(36).std() * np.sqrt(12)
    ax[0, 0].plot(roll.index, roll["s"], color=TREND, label="strategy")
    ax[0, 0].plot(roll.index, roll["b"], color=BH, label="benchmark")
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_title("Rolling 36 month Sharpe")
    ax[0, 0].legend(frameon=False)
    yr = (1 + pd.DataFrame({"strategy": s, "benchmark": b}, index=m.index)).groupby(m.index.year).prod() - 1
    xx = np.arange(len(yr))
    ax[0, 1].bar(xx - 0.2, yr["strategy"], 0.4, color=TREND, label="strategy")
    ax[0, 1].bar(xx + 0.2, yr["benchmark"], 0.4, color=BH, label="benchmark")
    step = max(1, len(yr) // 12)
    ax[0, 1].set_xticks(xx[::step], yr.index[::step], fontsize=8)
    ax[0, 1].set_title("Calendar year returns")
    ax[0, 1].legend(frameon=False)
    rows = {}
    for k in out["regimes_vs_benchmark"]:
        a1, a2 = out["regimes_vs_benchmark"][k], out["regimes_vs_volmatched"][k]
        rows[k] = [a1["months"], f"{12 * a1['mean_gap']:.1%} [{12 * a1['ci'][0]:.1%}, {12 * a1['ci'][1]:.1%}]",
                   f"{12 * a2['mean_gap']:.1%} [{12 * a2['ci'][0]:.1%}, {12 * a2['ci'][1]:.1%}]"]
    _table(ax[1, 0], pd.DataFrame(rows, index=["months", "vs benchmark, a year", "vs same vol mix, a year"]).T,
           "Regimes (90% block bootstrap intervals)")
    xs, ys = smile_xy
    ax[1, 1].scatter(xs, ys, s=10, color=TREND, alpha=0.5)
    X = np.column_stack([np.ones_like(xs), xs, xs * xs])
    coef = np.linalg.lstsq(X, ys, rcond=None)[0]
    g = np.linspace(xs.min(), xs.max(), 100)
    ax[1, 1].plot(g, coef[0] + coef[1] * g + coef[2] * g * g, color="k")
    sm = out["smile"]
    ax[1, 1].set_title(f"Quarterly smile, curvature {sm['gamma']:.2f} "
                       f"[{sm['gamma_ci90'][0]:.2f}, {sm['gamma_ci90'][1]:.2f}]", fontsize=10)
    ax[1, 1].set_xlabel("benchmark over T-bills")
    ax[1, 1].set_ylabel("strategy over T-bills")
    for a in [ax[0, 0], ax[0, 1], ax[1, 1]]:
        a.grid(alpha=0.3)
    fig.tight_layout()
    figs.append(fig)
    
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    tt = np.arange(1, wealth.shape[1] + 1) / 12
    names = [("strategy", TREND), ("benchmark", BH), ("same vol mix", MIX)]
    for j, (lab, col) in enumerate(names):
        p5, p50, p95 = np.percentile(wealth[:, :, j], [5, 50, 95], axis=0)
        ax[0, 0].plot(tt, p50, color=col, label=lab)
        ax[0, 0].fill_between(tt, p5, p95, color=col, alpha=0.12)
    ax[0, 0].set_yscale("log")
    ax[0, 0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]))
    ax[0, 0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:g}"))
    ax[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0, 0].set_title("Growth of 1 over 10 years, block bootstrap, median and 5 to 95%", fontsize=10)
    ax[0, 0].set_xlabel("years")
    ax[0, 0].legend(frameon=False)
    start = np.ones((wealth.shape[0], 1, wealth.shape[2]))
    peak = np.maximum.accumulate(np.concatenate([start, wealth], axis=1), axis=1)[:, 1:, :]
    mdd = -(wealth / peak - 1.0).min(axis=1)
    bins = np.linspace(0, max(0.05, mdd.max()), 40)
    for j, (lab, col) in enumerate(names):
        ax[0, 1].hist(mdd[:, j], bins=bins, histtype="step", color=col, lw=1.5, density=True, label=lab)
    ax[0, 1].set_title("Worst drawdown within 10 years, across paths", fontsize=10)
    ax[0, 1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[0, 1].legend(frameon=False)
    for a in [ax[0, 0], ax[0, 1]]:
        a.grid(alpha=0.3)
    if "sweep_sharpes_annual" in out:
        ranked = pd.Series(out["sweep_sharpes_annual"]).sort_values()
        lab = [", ".join(p for p in k.split(",") if not p.startswith("rule=")) for k in ranked.index]
        ax[1, 0].barh(np.arange(len(ranked)), ranked.values, color=BH)
        ax[1, 0].set_yticks(np.arange(len(ranked)), lab, fontsize=7)
        ax[1, 0].set_title("Sharpe over T-bills for every rule in the sweep", fontsize=10)
        ax[1, 0].grid(alpha=0.3)
    else:
        ax[1, 0].axis("off")
    ax[1, 1].axis("off")
    lines = _key_lines(out)
    ax[1, 1].text(0, 1, "\n".join(lines).replace("$", r"\$"), va="top", fontsize=9.5, family="monospace")
    fig.tight_layout()
    figs.append(fig)
    return figs


def _key_lines(out):
    d = out["sharpe_diff_vs_benchmark"]
    dd = out["maxdd_vs_return_matched"]
    ins_ = out["insurance_vs_volmatched"]
    lines = [
        f"Sharpe minus benchmark      {d['point']:+.2f}  [{d['ci90'][0]:+.2f}, {d['ci90'][1]:+.2f}]",
        f"Max DD minus same growth mix {dd['point']:+.1%}  [{dd['ci90'][0]:+.1%}, {dd['ci90'][1]:+.1%}]",
        f"   below zero in {dd['share_below_zero']:.0%} of resamples",
        "Cost vs same vol mix in normal times",
        f"   {ins_['premium_dollars_per_year']:+,.0f} USD a year per 100k",
        f"Bear markets where it beat the mix  {ins_['episodes_won']} of {ins_['n_episodes']}",
        f"Switches a year per asset   {out['turnover']['switches_per_year_per_asset']:.1f}",
        f"Average exposure            {out['turnover']['avg_exposure']:.0%}",
    ]
    if "pbo" in out:
        lines += [f"Deflated Sharpe probability {out['deflated_sharpe']['dsr']:.2f}",
                  f"PBO over {len(out['sweep_sharpes_annual'])} sweep rules     {out['pbo']['pbo']:.2f}"]
    lines += ["", "Intervals are 90% stationary block bootstrap.", "Research output, not investment advice."]
    return lines


def render(out, m, smile_xy, wealth, mix, path):
    figs = figures(out, m, smile_xy, wealth, mix)
    with PdfPages(path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    for fig in figs:
        plt.close(fig)


CSS = """
body { font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #222; margin: 0; background: #fafafa; }
main { max-width: 980px; margin: 0 auto; padding: 28px 20px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.sub { color: #666; margin: 0 0 20px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: #fff; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
th { background: #f0f0f0; font-weight: 600; }
img { width: 100%; height: auto; background: #fff; border: 1px solid #ddd; margin: 8px 0 20px; }
pre { background: #fff; border: 1px solid #ddd; padding: 12px 14px; overflow-x: auto; }
footer { color: #777; font-size: 13px; margin-top: 28px; }
"""


def render_html(out, figs, path):
    rule = " ".join(f"{k}={v}" for k, v in out["rule"].items() if k != "rule")
    ex = out["execution"]
    met = _fmt_metrics(out["metrics"])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["", *met.columns])
    body = "".join("<tr><td>" + html.escape(str(i)) + "</td>" + "".join(f"<td>{html.escape(v)}</td>" for v in row)
                   + "</tr>" for i, row in zip(met.index, met.values))
    imgs = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        imgs.append('<img alt="tearsheet figure" src="data:image/png;base64,'
                    + base64.b64encode(buf.getvalue()).decode() + '">')
    text = html.escape("\n".join(_key_lines(out)))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(out["name"])} tearsheet</title><style>{CSS}</style></head><body><main>'
            f'<h1>{html.escape(out["name"])}</h1>'
            f'<p class="sub">{out["start"]} to {out["end"]}, {html.escape(out["rule"]["rule"])} {html.escape(rule)}, '
            f'trade {ex["delay"]} day{"s" if ex["delay"] != 1 else ""} after the signal, {ex["cost_bps"]:g} bps a trade. '
            f'Assets {html.escape(", ".join(out["assets"]))}.</p>'
            f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            + "".join(imgs) + f"<pre>{text}</pre>"
            + "<footer>Cash earns the T-bill rate and every position change pays the cost shown. "
              "Research output, not investment advice.</footer></main></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def quick_plot(result, name, path):
    """Growth of $1 and drawdown for a backtest, strategy against buy and hold."""
    m = cf.monthly(result)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    t = m.index.to_timestamp()
    for col, lab, color in [("strategy", "strategy", TREND), ("benchmark", "buy and hold", BH)]:
        ax[0].plot(t, np.cumprod(1 + m[col].values), color=color, label=lab)
        ax[1].plot(t, mt.drawdown(m[col].values), color=color, lw=1)
    sw.log_axis(ax[0])
    ax[0].set_title(f"{name}, growth of $1 after costs", fontsize=10, loc="left")
    ax[0].legend(frameon=False)
    ax[1].set_title("Drawdown", fontsize=10, loc="left")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make(cfg, outdir=None, loader=None, fmt="pdf"):
    if fmt not in ("pdf", "html", "both"):
        raise ValueError("format must be pdf, html or both")
    result = cf.run(cfg, loader)
    sweeps = sweep(cfg, loader)
    out, m, xy, wealth, mix = compute(result, cfg, sweeps)
    outdir = outdir or os.path.join("reports", cfg["name"])
    os.makedirs(outdir, exist_ok=True)
    if fmt in ("pdf", "both"):
        render(out, m, xy, wealth, mix, os.path.join(outdir, "tearsheet.pdf"))
    if fmt in ("html", "both"):
        figs = figures(out, m, xy, wealth, mix, table=False)
        render_html(out, figs, os.path.join(outdir, "tearsheet.html"))
        for fig in figs:
            plt.close(fig)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(_clean(out), f, indent=1)
    return out, outdir


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x
