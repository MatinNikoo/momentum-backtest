"""
run_study.py

Runs the protection study exactly as written in docs/protocol.md and
docs/protocol_addendum.md, from the snapshots in data/snapshots.

    python run_study.py            # full run, about 10 minutes on 2 cores
    python run_study.py --quick    # 200 resamples instead of 2,000, for checking the plumbing

Writes results/study.json, results/tables/*.csv, figures/*.png and the
numbers the paper pulls in (paper/tables/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from trendlab import benchmarks as bm
from trendlab import data as dt
from trendlab import engine as en
from trendlab import insurance as ins
from trendlab import metrics as mt
from trendlab import rules as rl
from trendlab import stats as st

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(ROOT, "figures")

DELAY = 1
COST = 5.0
SEED = 0

# id: (data name, label, role, eval end, bear threshold, in the primary verdict)
SAMPLES = {
    "C1": ("us", "US stocks 1927 to 1999", "confirmation", "1999-12-31", 0.20, True),
    "C2": ("dev_ex_us", "Developed ex US stocks", "confirmation", None, 0.20, True),
    "C3": ("ust10", "US 10y Treasury", "other asset", None, 0.10, False),
    "C4": ("gld", "Gold (GLD)", "other asset", None, 0.20, False),
    "C5": ("dbc", "Commodities (DBC)", "other asset", None, 0.20, False),
    "X1": ("europe", "Europe stocks", "regional", None, 0.20, False),
    "X2": ("japan", "Japan stocks", "regional", None, 0.20, False),
    "X3": ("asia_pac_ex_japan", "Asia Pacific ex Japan", "regional", None, 0.20, False),
    "D1": ("spy", "SPY 2000 to 2026", "development", None, 0.20, False),
    "D2": ("us", "US stocks 2000 to 2026", "development", None, 0.20, False),
    "US": ("us", "US stocks 1927 to 2026", "century view", None, 0.20, False),
}
EQUITY = {"C1", "C2", "X1", "X2", "X3", "D1", "D2", "US"}

RULES = {
    "ma_50_200": {"rule": "ma_crossover", "fast": 50, "slow": 200},
    "sma_10m": {"rule": "sma_month", "months": 10},
    "tsmom_12m": {"rule": "tsmom", "lookback": 12},
    "tsmom_12m_vol": {"rule": "tsmom", "lookback": 12, "vol_target": 0.10},
}
PROTOCOL_RULES = ["ma_50_200", "sma_10m", "tsmom_12m"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def check_protocol():
    rec = open(os.path.join(RES, "protocol_sha256.txt")).read().split()
    want = {rec[0]: rec[1], rec[3]: rec[4]}
    got = {
        "protocol": hashlib.sha256(open(os.path.join(ROOT, "docs", "protocol.md"), "rb").read()).hexdigest(),
        "addendum": hashlib.sha256(open(os.path.join(ROOT, "docs", "protocol_addendum.md"), "rb").read()).hexdigest(),
    }
    for k in got:
        if got[k] != want[k]:
            raise SystemExit(f"docs/{k} changed after it was frozen. Log changes in docs/deviations.md instead.")
    return got


def clean(x):
    """Make numpy things json friendly."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Period):
        return str(x)
    return x


def eval_start(index: pd.DatetimeIndex, delay: int = DELAY, need_days: int = 200) -> pd.Timestamp:
    """
    First day of the first full month where every rule has a position. The
    protocol rules need 200 days (50/200) and 13 month ends (12 month
    momentum). The data snooping family needs up to 300 days.
    """
    me = index[rl.month_end_mask(index)]
    last_needed = max(index[need_days - 1], me[12])
    pos_day = index[index.get_loc(last_needed) + delay + 1]
    return (pos_day + pd.offsets.MonthBegin(1)).normalize()


def run_rule(spec, r, rf, delay=DELAY, cost=COST):
    px = en.total_return_index(r)
    sig = rl.make_signal(spec, px, rf)
    pos = en.positions_from_signal(sig, delay)
    return en.run(r, pos, rf, cost)


def window(df, start, end):
    m = df.index >= start
    if end is not None:
        m &= df.index <= pd.Timestamp(end)
    return df[m]


def matched_mix(strat, asset, rf):
    w, flag = bm.return_matched_w(strat, asset, rf)
    if flag == "unreachable":
        # trend grew faster than any static mix, so compare with the fastest growing mix
        # (a case the protocol didn't cover, logged in docs/deviations.md)
        w = bm.best_growth_w(asset, rf)
    return bm.mix(asset, rf, w), w, flag


def rm_delta(strat, asset, rf):
    """maxDD depth of the trend strategy minus that of the return matched mix."""
    mix, w, flag = matched_mix(strat, asset, rf)
    return mt.max_dd_depth(strat) - mt.max_dd_depth(mix), w, flag


# extra tail measures against the same mix (not in the protocol, see docs/deviations.md).
# each is written as a loss, so below zero still means trend did better
SECONDARY = {
    "ulcer": mt.ulcer,
    "cvar_5": lambda x: -mt.cvar(x),
    "worst_12m": lambda x: -mt.worst_window(x, 12),
}


def all_deltas(strat, asset, rf):
    mix, w, flag = matched_mix(strat, asset, rf)
    out = [mt.max_dd_depth(strat) - mt.max_dd_depth(mix)]
    out += [f(strat) - f(mix) for f in SECONDARY.values()]
    return np.array(out), w, flag


def primary_test(s, a, c, n_boot, seed=SEED):
    d0, w0, flag = all_deltas(s, a, c)
    idx = st.stationary_bootstrap_matrix(len(s), 12, n_boot, seed)
    draws = np.array([all_deltas(s[ix], a[ix], c[ix])[0] for ix in idx])
    out = None
    sec = {}
    for j, name in enumerate(["max_dd"] + list(SECONDARY)):
        ok = draws[:, j][np.isfinite(draws[:, j])]
        lo, hi = np.quantile(ok, [0.05, 0.95])
        row = {"delta": d0[j], "ci90": [lo, hi], "share_below_zero": float((ok < 0).mean())}
        if j == 0:
            out = {**row, "w_return_matched": w0, "flag": flag, "n_valid_resamples": int(len(ok))}
        else:
            sec[name] = row
    out["secondary"] = sec
    return out


def gold_averaging_check():
    """
    Extra, not in the protocol. Monthly average prices (World Bank) look trendier
    than month end prices. Compare the lag one autocorrelation and the 10 month
    rule on the averaged series with the same rule on GLD month ends.
    """
    m = dt.monthly_table()["gold_avg_price"].dropna()
    ra = m.pct_change().dropna()
    g = dt.daily_table()["gld"].dropna()
    rg = mt.to_monthly(g)
    rg = rg[(rg.index > rg.index[0]) & (rg.index <= pd.Period(dt.last_rf_date(), "M"))]
    rfm = mt.to_monthly(dt.rf_daily())
    common = ra.index.intersection(rg.index)
    out = {"start": str(common[0]), "end": str(common[-1]), "months": len(common)}
    for name, r in [("monthly_average", ra.loc[common]), ("gld_month_end", rg.loc[common])]:
        c = rfm.reindex(r.index).fillna(0.0)
        px = (1 + r).cumprod()
        sig = (px > px.rolling(10).mean()).astype(float)
        pos = sig.shift(1).fillna(0.0)
        strat = (pos * r + (1 - pos) * c).iloc[10:]
        out[name] = {"autocorr_lag1": float(r.autocorr(1)), "trend_sharpe": mt.sharpe(strat, c.iloc[10:]),
                     "buy_hold_sharpe": mt.sharpe(r.iloc[10:], c.iloc[10:])}
    rl_ = ra[ra.index >= pd.Period("1975-01", "M")]
    out["autocorr_lag1_average_1975_on"] = float(rl_.autocorr(1))
    # what averaging does to a pure random walk (Working 1960 says about 0.25)
    rng = np.random.default_rng(SEED)
    d = rng.normal(0, 0.01, 21 * 6000)
    p = np.exp(np.cumsum(d))
    avg = p.reshape(-1, 21).mean(1)
    out["random_walk_average_autocorr"] = float(pd.Series(np.diff(np.log(avg))).autocorr(1))
    return out


# ----------------------------------------------------------------------------
# one sample
# ----------------------------------------------------------------------------
def study_sample(sid, n_boot, bonds_m, long_y, quick=False):
    name, label, role, end, thr, primary = SAMPLES[sid]
    r, rf = dt.asset(name)
    if end is not None:
        r, rf = r[r.index <= end], rf[rf.index <= end]
    start = eval_start(r.index)
    if sid in ("D2",):
        start = max(start, pd.Timestamp("2000-01-01"))
    runs = {k: window(run_rule(v, r, rf), start, None) for k, v in RULES.items()}
    main = runs["ma_50_200"]
    s = mt.to_monthly(main["strategy"]).values
    a = mt.to_monthly(main["asset"]).values
    c = mt.to_monthly(main["rf"]).values
    idx_m = mt.to_monthly(main["strategy"]).index

    out = {"id": sid, "label": label, "role": role, "asset": name,
           "eval_start": str(idx_m[0]), "eval_end": str(idx_m[-1]), "months": len(s)}

    # benchmarks
    w_vol = bm.vol_matched_w(s, a, c)
    w_ret, ret_flag = bm.return_matched_w(s, a, c)
    w_exp = float(main["position"].mean())
    series = {"buy_and_hold": a, "trend_50_200": s,
              "vol_matched": bm.mix(a, c, w_vol), "exposure_matched": bm.mix(a, c, w_exp)}
    if np.isfinite(w_ret):
        series["return_matched"] = bm.mix(a, c, w_ret)
    for k in ["sma_10m", "tsmom_12m", "tsmom_12m_vol"]:
        series[k] = mt.to_monthly(runs[k]["strategy"]).values
    if sid in EQUITY:
        b = bonds_m.reindex(idx_m).values
        if np.isfinite(b).all():
            series["60_40"] = bm.sixty_forty(a, b)
            series["60_40_trend"] = bm.sixty_forty(s, b)
    out["weights"] = {"vol_matched": w_vol, "return_matched": w_ret, "return_matched_flag": ret_flag,
                      "exposure_matched": w_exp}
    out["metrics"] = {k: mt.summary(v, c) for k, v in series.items()}

    # turnover
    pos = main["position"]
    switches = float((pos.diff().abs() > 0).sum())
    # calendar years, since the data before 1953 has Saturday sessions and more than 252 rows a year
    years = (pos.index[-1] - pos.index[0]).days / 365.25
    out["turnover"] = {"switches_per_year": switches / years, "share_days_invested": float(pos.mean()),
                       "avg_days_per_holding": float(len(pos) / max(switches, 1)),
                       "avg_calendar_days_per_holding": float(365.25 * years / max(switches, 1))}

    # primary statistic
    out["primary"] = primary_test(s, a, c, n_boot)

    # insurance accounting against buy and hold and against the vol matched mix
    eps = ins.episodes(a, thr)
    out["bear_threshold"] = thr
    out["insurance_vs_bh"] = ins.account(s, a, eps, n_boot=n_boot, seed=SEED)
    out["insurance_vs_volmatched"] = ins.account(s, series["vol_matched"], eps, n_boot=n_boot, seed=SEED)
    # sensitivity, not in the protocol: bear markets dated from local peaks (see docs/deviations.md)
    eps_sw = ins.swing_episodes(a, thr)
    out["insurance_swing_vs_bh"] = ins.account(s, a, eps_sw, n_boot=n_boot, seed=SEED)
    out["insurance_swing_vs_volmatched"] = ins.account(s, series["vol_matched"], eps_sw, n_boot=n_boot, seed=SEED)
    for key in ["insurance_vs_bh", "insurance_vs_volmatched", "insurance_swing_vs_bh", "insurance_swing_vs_volmatched"]:
        for e in out[key]["episodes"]:
            e["from_month"] = str(idx_m[e["peak"]])  # first month of the fall
            e["to_month"] = str(idx_m[e["trough"] - 1])  # month the trough was hit

    # regimes
    labels = ins.month_labels(idx_m, a, main["asset"], c, long_y, thr)
    out["regimes_vs_bh"] = ins.regime_gaps(s - a, labels, n_boot=n_boot, seed=SEED)
    out["regimes_vs_volmatched"] = ins.regime_gaps(s - series["vol_matched"], labels, n_boot=n_boot, seed=SEED)

    # execution: delays and costs for the primary rule
    grid = []
    for d in [0, 1, 2, 3, 5, 10]:
        for cst in [0, 5, 10, 25, 50]:
            x = window(run_rule(RULES["ma_50_200"], r, rf, delay=d, cost=cst), start, None)
            xs = mt.to_monthly(x["strategy"]).values
            dd, _, _ = rm_delta(xs, a, c)
            grid.append({"delay": d, "cost_bps": cst, "cagr": mt.cagr(xs), "sharpe": mt.sharpe(xs, c),
                         "max_dd": mt.max_dd_depth(xs), "delta_vs_return_matched": dd})
    out["execution"] = grid

    # convexity: quarterly Treynor-Mazuy
    out["smile"] = smile_test(main, r, rf, start, n_boot, n_placebo=200 if quick else 1000)
    return out, {"months": idx_m, "series": series, "rf": c, "daily": main, "eps": eps, "eps_swing": eps_sw,
                 "labels": labels}


def smile_test(main, r, rf, start, n_boot, n_placebo):
    m = mt.to_monthly(main[["strategy", "asset", "rf"]])
    q = mt.to_quarterly(m)
    # only full quarters, so a sample that starts or ends mid quarter doesn't add a short one
    full = (m.groupby(m.index.asfreq("Q")).size() == 3).values
    q = q[full]
    ys, xs = (q["strategy"] - q["rf"]).values, (q["asset"] - q["rf"]).values
    a0, b0, g0 = st.treynor_mazuy(ys, xs)
    # how much rides on the single most extreme quarter (1933 Q2 in the US)
    keep_q = np.abs(xs) < np.abs(xs).max()
    g_drop = st.treynor_mazuy(ys[keep_q], xs[keep_q])[2]
    pt, lo, hi, _ = st.bootstrap_ci(lambda y, x: st.treynor_mazuy(y, x)[2], [ys, xs],
                                    n_boot=n_boot, mean_block=4, seed=SEED)
    # placebo: same positions, shifted in time so they can't know anything about the market
    pos = main["position"].values
    rr, cc = main["asset"].values, main["rf"].values
    qid = pd.factorize(main.index.to_period("Q"))[0]
    nq = qid.max() + 1
    lr_a = np.bincount(qid, np.log1p(rr), nq)[full]
    lr_c = np.bincount(qid, np.log1p(cc), nq)[full]
    xq = np.expm1(lr_a) - np.expm1(lr_c)
    rng = np.random.default_rng(SEED)
    n = len(pos)
    gs = []
    for _ in range(n_placebo):
        k = int(rng.integers(252, n - 252))
        p = np.roll(pos, k)
        turn = np.abs(np.diff(p, prepend=p[0]))
        strat = p * rr + (1 - p) * cc - turn * COST / 1e4
        sq = np.expm1(np.bincount(qid, np.log1p(strat), nq)[full]) - np.expm1(lr_c)
        gs.append(st.treynor_mazuy(sq, xq)[2])
    gs = np.array(gs)
    return {"alpha_q": a0, "beta": b0, "gamma": g0, "gamma_ci90": [lo, hi], "gamma_without_biggest_quarter": g_drop,
            "placebo_p": float((gs >= g0).mean()), "placebo_gamma_median": float(np.median(gs)),
            "placebo_gamma_95": float(np.quantile(gs, 0.95)), "quarters": int(len(ys)),
            "q_strat": ys.tolist(), "q_asset": xs.tolist()}


# ----------------------------------------------------------------------------
# emerging markets, monthly rules only (addendum A2)
# ----------------------------------------------------------------------------
def study_em(n_boot):
    r_m, rf_m = dt.em_monthly()
    rf_m = rf_m.reindex(r_m.index).fillna(0.0)
    ts = pd.DatetimeIndex([pd.offsets.BMonthEnd().rollback(p.to_timestamp(how="end").normalize()) for p in r_m.index])
    r = pd.Series(r_m.values, index=ts)
    rf = pd.Series(rf_m.values, index=ts)
    px = en.total_return_index(r)
    out = {"id": "EM", "label": "Emerging markets (monthly)", "role": "monthly only", "rules": {}}
    for k in ["sma_10m", "tsmom_12m"]:
        sig = rl.make_signal(RULES[k], px, rf)
        pos = sig.shift(1).fillna(0.0)  # monthly data: trade at the signal month end, a delay of 0 days
        res = en.run(r, pos, rf, COST)
        res = res.iloc[13:]  # 12 month momentum needs 13 month ends before its first position
        s, a, c = res["strategy"].values, res["asset"].values, res["rf"].values
        w_vol = bm.vol_matched_w(s, a, c)
        out["rules"][k] = {
            "eval_start": str(r_m.index[13]), "eval_end": str(r_m.index[-1]), "months": len(s),
            "metrics": {"buy_and_hold": mt.summary(a, c), "trend": mt.summary(s, c),
                        "vol_matched": mt.summary(bm.mix(a, c, w_vol), c)},
            "w_vol_matched": w_vol,
            "primary_style_delta": primary_test(s, a, c, n_boot),
        }
    return out


# ----------------------------------------------------------------------------
# data snooping across the whole family on C1 and C2 (addendum A5)
# ----------------------------------------------------------------------------
def family_specs():
    specs = {}
    for f in range(10, 101, 10):
        for s in range(100, 301, 25):
            if f < s:
                specs[f"ma_{f}_{s}"] = {"rule": "ma_crossover", "fast": f, "slow": s}
    specs["sma_10m"] = RULES["sma_10m"]
    specs["tsmom_12m"] = RULES["tsmom_12m"]
    specs["tsmom_12m_vol"] = RULES["tsmom_12m_vol"]
    for n in [50, 100, 200, 250]:
        specs[f"breakout_{n}"] = {"rule": "breakout", "n_entry": n}
    return specs


def snooping(sid, n_boot):
    name, _, _, end, _, _ = SAMPLES[sid]
    r, rf = dt.asset(name)
    if end is not None:
        r, rf = r[r.index <= end], rf[rf.index <= end]
    start = eval_start(r.index, need_days=300)  # the slowest rules in the family need 300 days
    specs = family_specs()
    D_mix, D_bh, R = {}, {}, {}
    for k, spec in specs.items():
        x = window(run_rule(spec, r, rf), start, None)
        m = mt.to_monthly(x[["strategy", "asset", "rf"]])
        s, a, c = m["strategy"].values, m["asset"].values, m["rf"].values
        w = bm.vol_matched_w(s, a, c)
        D_mix[k] = s - bm.mix(a, c, w)
        D_bh[k] = s - a
        R[k] = s - c
    D_mix, D_bh, R = pd.DataFrame(D_mix), pd.DataFrame(D_bh), pd.DataFrame(R)
    rc_mix = st.reality_check_spa(D_mix.values, n_boot=n_boot, mean_block=12, seed=SEED)
    rc_bh = st.reality_check_spa(D_bh.values, n_boot=n_boot, mean_block=12, seed=SEED)
    rc_mix["best_rule"] = D_mix.columns[rc_mix["best_rule"]]
    rc_bh["best_rule"] = D_bh.columns[rc_bh["best_rule"]]
    sr = D_mix.mean() / D_mix.std()
    best = sr.idxmax()
    dsr = st.deflated_sharpe(D_mix[best].values, sr.values)
    pbo = st.pbo_cscv(D_mix.values, n_blocks=16)
    sr_ex = R.mean() / R.std()
    dsr_ex = st.deflated_sharpe(R[sr_ex.idxmax()].values, sr_ex.values)
    ann = lambda v: float(v) * 12
    return {
        "sample": sid, "n_rules": len(specs),
        "vs_volmatched": {**rc_mix, "best_mean_diff_annual": ann(rc_mix["best_mean_diff"]),
                          "share_rules_positive": float((D_mix.mean() > 0).mean())},
        "vs_buy_and_hold": {**rc_bh, "best_mean_diff_annual": ann(rc_bh["best_mean_diff"]),
                            "share_rules_positive": float((D_bh.mean() > 0).mean())},
        "deflated_sharpe_vs_volmatched": {"best_rule": best, **dsr},
        "deflated_sharpe_over_tbills": {"best_rule": sr_ex.idxmax(), **dsr_ex},
        "pbo_vs_volmatched": {"pbo": pbo["pbo"], "n_splits": pbo["n_splits"]},
        "mean_diff_by_rule_annual": (D_mix.mean() * 12).round(6).to_dict(),
    }, pbo["logits"]


# ----------------------------------------------------------------------------
# diversified time series momentum, long/short, MOP style (addendum A3)
# ----------------------------------------------------------------------------
def tsmom_portfolio():
    # developed ex US is left out because Europe, Japan and Asia Pacific already make it up
    names = ["us", "europe", "japan", "asia_pac_ex_japan", "ust10", "gld", "dbc"]
    sleeves, avail = {}, {}
    for nm in names:
        r, rf = dt.asset(nm)
        px = en.total_return_index(r)
        sig = rl.tsmom(px, rf, lookback=12, vol_target=0.40, cap=np.inf, long_only=False)
        pos = en.positions_from_signal(sig, DELAY)
        res = en.run(r, pos, rf, COST)
        m = mt.to_monthly(res[["strategy", "rf", "position"]])
        sleeves[nm] = m["strategy"] - m["rf"]  # excess return of the sleeve
        avail[nm] = (res["position"] != 0).groupby(res.index.to_period("M")).any()
    ex = pd.DataFrame(sleeves)
    av = pd.DataFrame(avail).reindex(ex.index).fillna(False).astype(bool)
    ex = ex.where(av)
    port_ex = ex.mean(axis=1, skipna=True).dropna()
    n_assets = av.sum(axis=1).reindex(port_ex.index)
    _, rf_us = dt.asset("us")
    rfm = mt.to_monthly(rf_us).reindex(port_ex.index)
    usr, _ = dt.asset("us")
    us_m = mt.to_monthly(usr).reindex(port_ex.index)
    # start once at least two asset classes are in (US stocks and Treasuries, 1963)
    first = n_assets[n_assets >= 2].index[0]
    port_ex, rfm, us_m, n_assets = port_ex[first:], rfm[first:], us_m[first:], n_assets[first:]
    port = port_ex + rfm
    pm = pd.DataFrame({"p": port.values, "u": us_m.values, "c": rfm.values}, index=port.index)
    q = mt.to_quarterly(pm)[(pm.groupby(pm.index.asfreq("Q")).size() == 3).values]  # full quarters only
    a0, b0, g0 = st.treynor_mazuy((q.p - q.c).values, (q.u - q.c).values)
    _, lo, hi, _ = st.bootstrap_ci(lambda y, x: st.treynor_mazuy(y, x)[2], [(q.p - q.c).values, (q.u - q.c).values],
                                   n_boot=2000, mean_block=4, seed=SEED)
    worst = us_m.nsmallest(12)
    return {
        "start": str(port.index[0]), "end": str(port.index[-1]), "months": len(port),
        "metrics": mt.summary(port.values, rfm.values),
        "us_metrics_same_months": mt.summary(us_m.values, rfm.values),
        "corr_with_us": float(np.corrcoef(port.values, us_m.values)[0, 1]),
        "smile": {"beta": b0, "gamma": g0, "gamma_ci90": [lo, hi], "quarters": len(q)},
        "worst_us_months": [{"month": str(k), "us": float(v), "tsmom": float(port[k])} for k, v in worst.items()],
        "avg_assets": float(n_assets.mean()),
    }, port, us_m, rfm


# ----------------------------------------------------------------------------
# century view (addendum A1) and Monte Carlo (A7)
# ----------------------------------------------------------------------------
EVENTS = [
    ("1929 crash and Depression", "1929-09", "1932-06"),
    ("World War 2", "1939-09", "1945-08"),
    ("1970s", "1970-01", "1979-12"),
    ("1973 to 1974 bear market", "1973-01", "1974-09"),
    ("1987 crash", "1987-09", "1987-11"),
    ("Dot com bust", "2000-09", "2002-09"),
    ("2008 financial crisis", "2007-11", "2009-02"),
    ("2020 COVID crash", "2020-02", "2020-03"),
    ("2022 rate shock", "2022-01", "2022-09"),
]


def century_events(months, series):
    rows = []
    for name, a, b in EVENTS:
        pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
        m = (months >= pa) & (months <= pb)
        m12 = (months >= pa) & (months <= pb + 12)
        nxt = (months > pb) & (months <= pb + 12)  # the 12 months after the window
        row = {"event": name, "start": a, "end": b, "after_2000": pa.year >= 2000}
        for k in ["buy_and_hold", "trend_50_200", "vol_matched", "tsmom_12m"]:
            row[k] = float(np.prod(1 + series[k][m]) - 1)
            row[k + "_plus12m"] = float(np.prod(1 + series[k][m12]) - 1)
            row[k + "_next12m"] = float(np.prod(1 + series[k][nxt]) - 1)
        rows.append(row)
    return rows


def monte_carlo(series, rf, n_paths=10_000, horizon=120, seed=SEED):
    names = ["buy_and_hold", "trend_50_200", "vol_matched"]
    X = np.column_stack([series[k] for k in names])
    idx = st.stationary_bootstrap_matrix(len(X), 12, n_paths, seed, length=horizon)
    paths = X[idx]  # (paths, horizon, 3)
    wealth = np.cumprod(1 + paths, axis=1)
    end = wealth[:, -1, :]
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_paths, 1, 3)), wealth], axis=1), axis=1)[:, 1:, :]
    mdd = -(wealth / peak - 1).min(axis=1)
    out = {}
    for j, k in enumerate(names):
        out[k] = {
            "end_wealth_pct": dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                       np.percentile(end[:, j], [5, 25, 50, 75, 95]).tolist())),
            "max_dd_pct": dict(zip(["p5", "p50", "p95"], np.percentile(mdd[:, j], [5, 50, 95]).tolist())),
            "prob_loss": float((end[:, j] < 1).mean()),
            "prob_dd_over_30": float((mdd[:, j] > 0.30).mean()),
        }
    out["prob_trend_beats_bh"] = float((end[:, 1] > end[:, 0]).mean())
    out["prob_trend_beats_volmatched"] = float((end[:, 1] > end[:, 2]).mean())
    out["prob_trend_smaller_dd_than_volmatched"] = float((mdd[:, 1] < mdd[:, 2]).mean())
    out["n_paths"], out["horizon_months"] = n_paths, horizon
    return out, wealth


# ----------------------------------------------------------------------------
# bond model check (protocol, before anything else)
# ----------------------------------------------------------------------------
def bond_validation():
    D = dt.daily_table()
    um = mt.to_monthly(D["ust10"].dropna())
    im = mt.to_monthly(D["ief"].dropna())
    j = pd.concat([um, im], axis=1, keys=["model", "ief"]).dropna()
    j = j[(j.index > j.index[0]) & (j.index < j.index[-1])]  # drop partial first and last month
    dam = pd.read_csv(os.path.join(dt.SNAP, "damodaran_tbond_annual.csv"), index_col=0).iloc[:, 0]
    bmo = dt.bond_monthly()
    ba = (1 + bmo).groupby(bmo.index.year).prod() - 1
    k = pd.concat([ba, dam], axis=1, keys=["model", "damodaran"]).dropna()
    k = k[(k.index >= 1928) & (k.index <= 2025)]
    corr = float(j.corr().iloc[0, 1])
    gap = mt.cagr(j["model"]) - mt.cagr(j["ief"])
    return {
        "ief_months": len(j), "ief_start": str(j.index[0]), "ief_end": str(j.index[-1]),
        "corr_monthly_vs_ief": corr, "cagr_model": mt.cagr(j["model"]), "cagr_ief": mt.cagr(j["ief"]),
        "cagr_gap": gap, "vol_model": mt.vol(j["model"]), "vol_ief": mt.vol(j["ief"]),
        "passes": bool(corr >= 0.95 and abs(gap) <= 0.01),
        "damodaran_years": len(k), "corr_annual_vs_damodaran": float(k.corr().iloc[0, 1]),
        "mean_annual_gap_vs_damodaran": float((k["model"] - k["damodaran"]).mean()),
    }, j, k


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="run everything but save nothing (used by CI)")
    args = ap.parse_args()
    n_boot = 200 if args.quick else 2000
    t0 = time.time()
    hashes = check_protocol()
    if not args.no_write:
        os.makedirs(TAB, exist_ok=True)
        os.makedirs(FIG, exist_ok=True)

    bonds_m = dt.bond_monthly()
    long_y = dt.long_yield()
    results = {"protocol_sha256": hashes, "settings": {"delay_days": DELAY, "cost_bps": COST,
               "n_boot": n_boot, "mean_block_months": 12, "seed": SEED},
               "data_end": str(dt.last_rf_date().date())}

    print("bond model check")
    results["bond_validation"], bv_m, bv_a = bond_validation()

    keep = {}
    results["samples"] = {}
    for sid in SAMPLES:
        print(f"sample {sid} ({time.time() - t0:.0f}s)")
        res, k = study_sample(sid, n_boot, bonds_m, long_y, quick=args.quick)
        results["samples"][sid] = res
        keep[sid] = k

    # verdict, exactly as the protocol says
    p1, p2 = results["samples"]["C1"]["primary"], results["samples"]["C2"]["primary"]
    neg = [p["delta"] < 0 for p in (p1, p2)]
    strong = [p["share_below_zero"] >= 0.90 for p in (p1, p2)]
    if all(neg) and all(strong):
        verdict = "Supported"
    elif any(neg):
        verdict = "Mixed"
    else:
        verdict = "Not supported"
    results["verdict"] = verdict
    print("verdict:", verdict)

    print(f"emerging markets ({time.time() - t0:.0f}s)")
    results["emerging"] = study_em(n_boot)

    print(f"data snooping ({time.time() - t0:.0f}s)")
    results["snooping"] = {}
    logits = {}
    for sid in ["C1", "C2"]:
        results["snooping"][sid], logits[sid] = snooping(sid, n_boot)

    print(f"diversified time series momentum ({time.time() - t0:.0f}s)")
    results["tsmom_portfolio"], tp, tp_us, tp_rf = tsmom_portfolio()

    print(f"century view and Monte Carlo ({time.time() - t0:.0f}s)")
    us = keep["US"]
    results["century_events"] = century_events(us["months"], us["series"])
    results["monte_carlo"], mc_wealth = monte_carlo(us["series"], us["rf"],
                                                    n_paths=2000 if args.quick else 10_000)

    print(f"gold averaging check ({time.time() - t0:.0f}s)")
    results["gold_averaging_check"] = gold_averaging_check()

    results["runtime_seconds"] = round(time.time() - t0, 1)
    if args.no_write:
        print(f"done in {time.time() - t0:.0f}s, nothing written")
        return
    with open(os.path.join(RES, "study.json"), "w") as f:
        json.dump(clean(results), f, indent=1)
    write_tables(results)

    if not args.no_figures:
        import study_figures
        study_figures.make_all(results, keep, logits, (tp, tp_us, tp_rf), mc_wealth, bv_m, bv_a)
    import paper_numbers
    paper_numbers.write()  # reads the saved json, so a rerun of paper_numbers.py alone gives the same tables
    print(f"done in {time.time() - t0:.0f}s")


def write_tables(R):
    rows = []
    for sid, s in R["samples"].items():
        for k, m in s["metrics"].items():
            rows.append({"sample": sid, "label": s["label"], "series": k, **m})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "metrics.csv"), index=False)
    rows = []
    for sid, s in R["samples"].items():
        p = s["primary"]
        rows.append({"sample": sid, "label": s["label"], "role": s["role"], "start": s["eval_start"],
                     "end": s["eval_end"], "delta": p["delta"], "ci_lo": p["ci90"][0], "ci_hi": p["ci90"][1],
                     "share_below_zero": p["share_below_zero"], "w_return_matched": p["w_return_matched"],
                     "flag": p["flag"]})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "primary.csv"), index=False)
    rows = []
    for sid, s in R["samples"].items():
        for vs in ["vs_bh", "vs_volmatched"]:
            for e in s[f"insurance_{vs}"]["episodes"]:
                rows.append({"sample": sid, "vs": vs, **e})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "bear_episodes.csv"), index=False)
    rows = []
    for sid, s in R["samples"].items():
        for g in s["execution"]:
            rows.append({"sample": sid, **g})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "execution.csv"), index=False)
    rows = []
    for sid, s in R["samples"].items():
        for vs in ["regimes_vs_bh", "regimes_vs_volmatched"]:
            for k, v in s[vs].items():
                rows.append({"sample": sid, "vs": vs.replace("regimes_", ""), "regime": k, "months": v["months"],
                             "mean_gap_monthly": v["mean_gap"], "ci_lo": v["ci"][0], "ci_hi": v["ci"][1]})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "regimes.csv"), index=False)
    pd.DataFrame(R["century_events"]).to_csv(os.path.join(TAB, "century_events.csv"), index=False)


if __name__ == "__main__":
    main()
