"""Figures for the protection study. Called by run_study.py, reads what it just computed."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

TREND = "#C44E52"
BH = "#5B6B8C"
MIX = "#8C8C8C"
OTHER = "#55A868"

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.35, "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False})


def _money(x, pos=None):
    # a real minus sign in front of the dollar sign, not "$-8,000"
    return ("\u2212" if x < 0 else "") + f"${abs(x):,.0f}"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def century(keep, R):
    us = keep["US"]
    t = us["months"].to_timestamp()
    s = us["series"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for k, lab, col, ls in [("buy_and_hold", "Buy and hold", BH, "-"),
                            ("trend_50_200", "50/200 trend, T-bills when out", TREND, "-"),
                            ("vol_matched", f"Same volatility mix ({R['samples']['US']['weights']['vol_matched']:.0%} stocks)", MIX, "--")]:
        w = np.cumprod(1 + s[k])
        ax[0].plot(t, w, color=col, ls=ls, lw=1.4, label=f"{lab}, ends at ${w[-1]:,.0f}")
        dd = w / np.maximum.accumulate(np.r_[1.0, w])[1:] - 1
        ax[1].plot(t, dd, color=col, ls=ls, lw=1.1)
    for e in us["eps"]:
        a, b = t[e["peak"]], t[e["trough"] - 1]
        for x in ax:
            x.axvspan(a, b, color="k", alpha=0.10, lw=0)
    # falls of 20% from a local peak that the protocol's definition misses, because the
    # old high hadn't been regained yet (1937 to 1938 and 1940 to 1942)
    proto = [(e["peak"], e["trough"]) for e in us["eps"]]
    for e in us["eps_swing"]:
        if any(p <= e["peak"] < q for p, q in proto):
            continue
        a, b = t[e["peak"]], t[e["trough"] - 1]
        for x in ax:
            x.axvspan(a, b, facecolor="none", edgecolor="#999", hatch="///", lw=0)
    for x in ax:
        x.axvline(pd.Timestamp("2000-01-01"), color="k", lw=1, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Growth of $1 (log scale)")
    ax[0].set_title("US stocks 1927 to 2026, everything after costs, cash earns T-bills")
    ax[0].text(pd.Timestamp("1999-01-01"), 0.6, "untouched before 2000  ", ha="right", fontsize=9)
    ax[0].text(pd.Timestamp("2001-01-01"), 0.6, "  seen before", ha="left", fontsize=9)
    ax[0].legend(loc="upper left", frameon=False)
    ax[1].set_ylabel("Drop from peak")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].set_title("Drawdowns. Gray: falls of 20% or more from the all time high. Hatched: 20% falls from a local peak")
    _save(fig, "century.png")


def primary_forest(R):
    order = ["C1", "C2", "C3", "C4", "C5", "X1", "X2", "X3", "D1", "D2", "US"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ys = np.arange(len(order))[::-1]
    for y, sid in zip(ys, order):
        s = R["samples"][sid]
        p = s["primary"]
        col = TREND if sid in ("C1", "C2") else (MIX if s["role"] in ("development", "century view") else BH)
        ax.plot(p["ci90"], [y, y], color=col, lw=2.2)
        ax.plot(p["delta"], y, "o", color=col, ms=7)
        ax.text(max(p["ci90"][1], p["delta"]) + 0.02, y, f"{p['share_below_zero']:.0%} of resamples below 0",
                va="center", fontsize=8.5, color="#444")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(ys, [f"{sid}  {R['samples'][sid]['label']}" for sid in order])
    ax.set_xlabel("Worst drawdown of trend minus worst drawdown of the same growth static mix\n(left of 0 means trend protected better for the same growth)")
    ax.set_title(f"Primary test, 90% block bootstrap intervals. Verdict by the protocol: {R['verdict']}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(-0.75, 1.0)
    _save(fig, "primary_test.png")


def smile(R, keep, tsmom):
    port, us_m, rfm = tsmom
    panels = [("C1", "US stocks 1927 to 1999, 50/200"), ("C2", "Developed ex US, 50/200"),
              ("C3", "US 10y Treasury, 50/200")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (sid, title) in zip(axes, panels):
        sm = R["samples"][sid]["smile"]
        x, y = np.array(sm["q_asset"]), np.array(sm["q_strat"])
        _smile_panel(ax, x, y, sm["gamma"], sm["gamma_ci90"], title)
        ax.text(0.03, 0.80, f"placebo p = {sm['placebo_p']:.2f}", transform=ax.transAxes, fontsize=8.5)
    pm = pd.DataFrame({"p": port.values, "u": us_m.values, "c": rfm.values}, index=port.index)
    q = (1 + pm).groupby(pm.index.asfreq("Q")).prod() - 1
    q = q[(pm.groupby(pm.index.asfreq("Q")).size() == 3).values]  # full quarters only, same as run_study
    tp = R["tsmom_portfolio"]["smile"]
    _smile_panel(axes[3], (q.u - q.c).values, (q.p - q.c).values, tp["gamma"], tp["gamma_ci90"],
                 "Diversified TSMOM vs US stocks, 1963 on")
    axes[0].set_ylabel("Strategy return over T-bills, per quarter")
    for ax in axes:
        ax.set_xlabel("Asset return over T-bills, per quarter")
    _save(fig, "smile.png")


def _smile_panel(ax, x, y, g, ci, title):
    ax.scatter(x, y, s=10, alpha=0.5, color=TREND, lw=0)
    X = np.column_stack([np.ones_like(x), x, x * x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, coef[0] + coef[1] * xx + coef[2] * xx ** 2, color="k", lw=1.5)
    ax.axhline(0, color="#999", lw=0.8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.text(0.03, 0.90, f"curvature {g:.2f}, 90% CI {ci[0]:.2f} to {ci[1]:.2f}", transform=ax.transAxes, fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def insurance_fig(R):
    us = R["samples"]["US"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    eps_bh = us["insurance_vs_bh"]["episodes"]
    eps_mx = us["insurance_vs_volmatched"]["episodes"]
    x = np.arange(len(eps_bh))
    ax[0].bar(x - 0.2, [e["dollars_saved"] for e in eps_bh], 0.4, color=BH, label="vs buy and hold")
    ax[0].bar(x + 0.2, [e["dollars_saved"] for e in eps_mx], 0.4, color=MIX, label="vs same volatility mix")
    ax[0].set_xticks(x, [f"{e['from_month'][:4]}\n{e['depth']:.0%}" for e in eps_bh], fontsize=8.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("Dollars ahead at the bottom, per $100,000")
    ax[0].set_title("Payout. US bear markets of 20% or more (year started, depth)")
    ax[0].legend(frameon=False)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_money))

    sids = ["C1", "C2", "C3", "US"]
    ys = np.arange(len(sids))[::-1]
    for y, sid in zip(ys, sids):
        for off, key, col in [(0.15, "insurance_vs_bh", BH), (-0.15, "insurance_vs_volmatched", MIX)]:
            i = R["samples"][sid][key]
            ax[1].plot(i["premium_dollars_ci"], [y + off] * 2, color=col, lw=2.2)
            ax[1].plot(i["premium_dollars_per_year"], y + off, "o", color=col,
                       label=("vs buy and hold" if key.endswith("bh") else "vs same volatility mix") if sid == "C1" else None)
    ax[1].legend(frameon=False, loc="center left", fontsize=9)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(ys, [R["samples"][s]["label"] for s in sids])
    ax[1].set_xlabel("Dollars a year per $100,000, outside bear markets")
    ax[1].set_title("Premium. What it costs in normal times (90% CI)")
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_money))
    _save(fig, "insurance.png")


def execution(R):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for a, sid in zip(ax, ["C1", "C2"]):
        g = pd.DataFrame(R["samples"][sid]["execution"])
        for cst, col in zip([0, 5, 10, 25, 50], plt.cm.viridis(np.linspace(0, 0.9, 5))):
            h = g[g.cost_bps == cst]
            a.plot(h["delay"], h["sharpe"], "o-", color=col, label=f"{cst} bps")
        bh = R["samples"][sid]["metrics"]["buy_and_hold"]["sharpe"]
        # a stock and T-bill mix has almost the same Sharpe as the stock, so one line covers both
        a.axhline(bh, color=BH, ls="--", lw=1.2, label="buy and hold, or any\nstock and T-bill mix")
        a.set_title(R["samples"][sid]["label"])
        a.set_xlabel("Days between the signal close and the trade")
        a.set_ylabel("Sharpe ratio over T-bills")
    ax[0].legend(frameon=False, fontsize=8.5, title="cost per trade", title_fontsize=8.5)
    _save(fig, "execution.png")


def regimes(R):
    names = ["bear", "recovery", "crash_months", "calm", "high_rates", "low_rates",
             "rising_long_rates", "falling_long_rates"]
    pretty = ["Bear market", "Recovery", "Crash months", "Calm", "High T-bill rate", "Low T-bill rate",
              "Long rates rising", "Long rates falling"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for a, sid in zip(ax, ["C1", "C2"]):
        rg = R["samples"][sid]["regimes_vs_volmatched"]
        ys = np.arange(len(names))[::-1]
        for y, n in zip(ys, names):
            if n not in rg:
                continue
            v = rg[n]
            a.plot([v["ci"][0] * 12, v["ci"][1] * 12], [y, y], color=TREND, lw=2)
            a.plot(v["mean_gap"] * 12, y, "o", color=TREND)
            a.text(1.02, y, f"{v['months']} mo", transform=a.get_yaxis_transform(), va="center", fontsize=8)
        a.axvline(0, color="k", lw=0.8)
        a.set_yticks(ys, pretty)
        a.set_title(R["samples"][sid]["label"])
        a.set_xlabel("Trend minus same volatility mix, % a year (90% CI)")
        a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "regimes.png")


def monte_carlo(R, wealth):
    mc = R["monte_carlo"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    t = np.arange(1, wealth.shape[1] + 1) / 12
    for j, (k, lab, col) in enumerate([("buy_and_hold", "Buy and hold", BH),
                                       ("trend_50_200", "50/200 trend", TREND),
                                       ("vol_matched", "Same volatility mix", MIX)]):
        p5, p50, p95 = np.percentile(wealth[:, :, j], [5, 50, 95], axis=0)
        ax[0].plot(t, p50, color=col, lw=1.6, label=f"{lab}, median")
        ax[0].fill_between(t, p5, p95, color=col, alpha=0.12)
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:g}"))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("Years")
    ax[0].set_ylabel("Growth of $1, 5th to 95th percentile")
    ax[0].set_title(f"{mc['n_paths']:,} resampled 10 year paths from US 1927 to 2026")
    ax[0].legend(frameon=False, loc="upper left")
    peak = np.maximum.accumulate(np.concatenate([np.ones((wealth.shape[0], 1, 3)), wealth], axis=1), axis=1)[:, 1:, :]
    mdd = -(wealth / peak - 1).min(axis=1)
    bins = np.linspace(0, 0.9, 46)
    for j, (lab, col) in enumerate([("Buy and hold", BH), ("50/200 trend", TREND), ("Same volatility mix", MIX)]):
        ax[1].hist(mdd[:, j], bins=bins, histtype="step", lw=1.6, color=col, label=lab, density=True)
    ax[1].set_xlabel("Worst drawdown within the 10 years")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Worst drawdown across paths")
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].legend(frameon=False)
    _save(fig, "monte_carlo.png")


def snooping(R, logits):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    sn = R["snooping"]["C1"]
    d = pd.Series(sn["mean_diff_by_rule_annual"]).sort_values()
    cols = [TREND if k == "ma_50_200" else (OTHER if not k.startswith("ma_") else BH) for k in d.index]
    ax[0].bar(np.arange(len(d)), d.values, color=cols, width=0.9)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks([])
    ax[0].set_xlabel(f"All {sn['n_rules']} rules, sorted (red is 50/200, green are the non moving average rules)")
    ax[0].set_ylabel("Return over the same volatility mix, a year")
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    v = sn["vs_volmatched"]
    ax[0].set_title(f"US 1927 to 1999. Reality Check p = {v['p_reality_check']:.2f}, SPA p = {v['p_spa']:.2f}")
    ax[1].hist(logits["C1"], bins=40, color=BH)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_title(f"PBO = {sn['pbo_vs_volmatched']['pbo']:.2f}. Where the in sample best rule ranks out of sample")
    ax[1].set_xlabel("Logit of out of sample rank (left of 0 = bottom half)")
    _save(fig, "snooping.png")


def bond_check(R, j, k):
    bv = R["bond_validation"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    t = j.index.to_timestamp()
    ax[0].plot(t, np.cumprod(1 + j["model"]), color=TREND, label="my 10 year model from FRED yields")
    ax[0].plot(t, np.cumprod(1 + j["ief"]), color=BH, label="IEF, a real Treasury ETF")
    ax[0].set_title(f"Monthly correlation {bv['corr_monthly_vs_ief']:.3f}")
    ax[0].set_ylabel("Growth of $1")
    ax[0].legend(frameon=False)
    ax[1].scatter(k["damodaran"], k["model"], s=14, color=TREND)
    lim = [min(k.min()) - 0.02, max(k.max()) + 0.02]
    ax[1].plot(lim, lim, color="k", lw=0.8)
    ax[1].set_xlabel("Damodaran's 10 year Treasury return, per year")
    ax[1].set_ylabel("My model, per year")
    ax[1].set_title(f"1928 to 2025, correlation {bv['corr_annual_vs_damodaran']:.3f}")
    for a in [ax[1]]:
        a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "bond_model_check.png")


def make_all(R, keep, logits, tsmom, mc_wealth, bv_m, bv_a):
    os.makedirs(FIG, exist_ok=True)
    century(keep, R)
    primary_forest(R)
    smile(R, keep, tsmom)
    insurance_fig(R)
    execution(R)
    regimes(R)
    monte_carlo(R, mc_wealth)
    snooping(R, logits)
    bond_check(R, bv_m, bv_a)
