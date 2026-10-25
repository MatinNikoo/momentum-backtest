"""
Turns results/*.json into paper/tables/numbers.tex (one macro per number)
and paper/tables/*.tex, so no number in the paper is typed by hand.

    python paper_numbers.py

In the paper, \\n{C1.primary} prints a number and stops the build if the
key doesn't exist.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper", "tables")

LABELS = {
    "buy_and_hold": "Buy and hold", "trend_50_200": "50/200 trend", "vol_matched": "Same volatility mix",
    "return_matched": "Same growth mix", "exposure_matched": "Same exposure mix", "60_40": "60/40",
    "60_40_trend": "60/40, stocks on 50/200", "sma_10m": "10 month average", "tsmom_12m": "12 month momentum",
    "tsmom_12m_vol": "12 month momentum, vol scaled",
}


def pct(x, d=1, sign=False):
    if x is None:
        return "n/a"
    if round(100 * x, d) == 0:
        x = 0.0  # no "-0.0"
    s = f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"
    return s.replace("-", "$-$") + "\\%"


def num(x, d=2, sign=False):
    if x is None:
        return "n/a"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def usd(x):
    if x is None:
        return "n/a"
    s = f"{abs(x):,.0f}"
    return ("$-$" if x < 0 else "") + "\\$" + s


class Numbers:
    def __init__(self):
        self.d = {}

    def put(self, key, val):
        if key in self.d:
            raise KeyError(f"duplicate number {key}")
        self.d[key] = val

    def tex(self):
        lines = ["% made by paper_numbers.py from results/*.json, do not edit by hand",
                 "\\newcommand{\\n}[1]{\\ifcsname num@#1\\endcsname\\csname num@#1\\endcsname"
                 "\\else\\errmessage{missing number #1}\\fi}"]
        for k, v in sorted(self.d.items()):
            lines.append(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}")
        return "\n".join(lines) + "\n"


def table(path, header, rows, align, caption=None):
    body = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "mid":
            body.append("\\midrule")
        else:
            body.append(" & ".join(r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, path), "w") as f:
        f.write("\n".join(body) + "\n")


def write(R=None):
    if R is None:
        R = json.load(open(os.path.join(ROOT, "results", "study.json")))
    os.makedirs(OUT, exist_ok=True)
    N = Numbers()
    S = R["samples"]
    N.put("verdict", R["verdict"])
    N.put("nboot", f"{R['settings']['n_boot']:,}")
    N.put("dataend", R["data_end"])
    N.put("protocolhash", R["protocol_sha256"]["protocol"][:12])
    N.put("addendumhash", R["protocol_sha256"]["addendum"][:12])

    for sid, s in S.items():
        N.put(f"{sid}.start", s["eval_start"])
        N.put(f"{sid}.end", s["eval_end"])
        N.put(f"{sid}.months", str(s["months"]))
        for k, m in s["metrics"].items():
            for f, fn in [("cagr", pct), ("vol", pct), ("max_dd", pct), ("worst_12m", pct), ("cvar_5", pct),
                          ("ulcer", pct)]:
                N.put(f"{sid}.{k}.{f}", fn(m[f]))
            N.put(f"{sid}.{k}.sharpe", num(m["sharpe"]))
            N.put(f"{sid}.{k}.skew", num(m["skew"]))
        w = s["weights"]
        N.put(f"{sid}.wvol", pct(w["vol_matched"], 0))
        N.put(f"{sid}.wret", pct(w["return_matched"], 0) if w["return_matched"] is not None else "n/a")
        N.put(f"{sid}.wexp", pct(w["exposure_matched"], 0))
        p = s["primary"]
        N.put(f"{sid}.delta", pct(p["delta"], 1, True))
        N.put(f"{sid}.deltapts", num(abs(100 * p["delta"]), 1))  # size in drawdown points, no sign, no %
        N.put(f"{sid}.deltalo", pct(p["ci90"][0], 1, True))
        N.put(f"{sid}.deltahi", pct(p["ci90"][1], 1, True))
        N.put(f"{sid}.share", pct(p["share_below_zero"], 0))
        # rough size an effect would need to clear the 90% bar, reading the bootstrap spread off the 90% interval
        spread = (p["ci90"][1] - p["ci90"][0]) / (2 * 1.645)
        N.put(f"{sid}.needed", pct(-1.2816 * spread, 0))
        N.put(f"{sid}.neededpts", num(1.2816 * spread * 100, 0))
        for k, v in p["secondary"].items():
            key = k.replace("_", "")
            N.put(f"{sid}.sec.{key}", pct(v["delta"], 1, True))
            N.put(f"{sid}.sec.{key}.share", pct(v["share_below_zero"], 0))
            N.put(f"{sid}.sec.{key}.pts", num(abs(100 * v["delta"]), 1))
        t = s["turnover"]
        N.put(f"{sid}.switches", num(t["switches_per_year"], 1))
        N.put(f"{sid}.invested", pct(t["share_days_invested"], 0))
        N.put(f"{sid}.holddays", f"{t['avg_days_per_holding']:.0f}")
        sm = s["smile"]
        N.put(f"{sid}.gamma", num(sm["gamma"]))
        N.put(f"{sid}.gammalo", num(sm["gamma_ci90"][0]))
        N.put(f"{sid}.gammahi", num(sm["gamma_ci90"][1]))
        N.put(f"{sid}.placebop", num(sm["placebo_p"]))
        N.put(f"{sid}.beta", num(sm["beta"]))
        N.put(f"{sid}.gammadrop", num(sm["gamma_without_biggest_quarter"]))
        for vs, tag in [("insurance_vs_bh", "bh"), ("insurance_vs_volmatched", "mix"),
                        ("insurance_swing_vs_bh", "swbh"), ("insurance_swing_vs_volmatched", "swmix")]:
            i = s[vs]
            N.put(f"{sid}.{tag}.prem", usd(i["premium_dollars_per_year"]))
            N.put(f"{sid}.{tag}.premlo", usd(i["premium_dollars_ci"][0]))
            N.put(f"{sid}.{tag}.premhi", usd(i["premium_dollars_ci"][1]))
            # the same premium written as a positive cost, for sentences
            N.put(f"{sid}.{tag}.cost", usd(-i["premium_dollars_per_year"]))
            N.put(f"{sid}.{tag}.costlo", usd(-i["premium_dollars_ci"][1]))
            N.put(f"{sid}.{tag}.costhi", usd(-i["premium_dollars_ci"][0]))
            N.put(f"{sid}.{tag}.premlog", pct(i["premium_log_per_year"], 1, True))
            N.put(f"{sid}.{tag}.bear", pct(i["bear_gap_log_per_year"], 1, True))
            N.put(f"{sid}.{tag}.bearlo", pct(i["bear_gap_ci"][0], 1, True))
            N.put(f"{sid}.{tag}.bearhi", pct(i["bear_gap_ci"][1], 1, True))
            N.put(f"{sid}.{tag}.net", pct(i["net_log_per_year"], 1, True))
            N.put(f"{sid}.{tag}.netlo", pct(i["net_ci"][0], 1, True))
            N.put(f"{sid}.{tag}.nethi", pct(i["net_ci"][1], 1, True))
            N.put(f"{sid}.{tag}.won", str(i["episodes_won"]))
            N.put(f"{sid}.{tag}.neps", str(i["n_episodes"]))
            N.put(f"{sid}.{tag}.signp", num(i["sign_test_p"], 3) if i["sign_test_p"] is not None else "n/a")
            N.put(f"{sid}.{tag}.saved", usd(i["mean_dollars_saved_per_episode"]))
            N.put(f"{sid}.{tag}.savedlo", usd(i["mean_dollars_saved_ci"][0]))
            N.put(f"{sid}.{tag}.savedhi", usd(i["mean_dollars_saved_ci"][1]))
        for vs, tag in [("regimes_vs_bh", "bh"), ("regimes_vs_volmatched", "mix")]:
            for k, v in s[vs].items():
                key = k.replace("_", "")
                N.put(f"{sid}.reg.{tag}.{key}", pct(12 * v["mean_gap"], 1, True))
                N.put(f"{sid}.reg.{tag}.{key}.abs", pct(abs(12 * v["mean_gap"]), 1))
                N.put(f"{sid}.reg.{tag}.{key}.lo", pct(12 * v["ci"][0], 1, True))
                N.put(f"{sid}.reg.{tag}.{key}.hi", pct(12 * v["ci"][1], 1, True))
                N.put(f"{sid}.reg.{tag}.{key}.months", str(v["months"]))
        for g in s["execution"]:
            N.put(f"{sid}.exec.d{g['delay']}.c{g['cost_bps']}.sharpe", num(g["sharpe"]))
            N.put(f"{sid}.exec.d{g['delay']}.c{g['cost_bps']}.cagr", pct(g["cagr"]))
            N.put(f"{sid}.exec.d{g['delay']}.c{g['cost_bps']}.delta", pct(g["delta_vs_return_matched"], 1, True))

    bv = R["bond_validation"]
    N.put("bond.corr", num(bv["corr_monthly_vs_ief"], 3))
    N.put("bond.cagrmodel", pct(bv["cagr_model"], 2))
    N.put("bond.cagrief", pct(bv["cagr_ief"], 2))
    N.put("bond.gap", pct(bv["cagr_gap"], 2, True))
    N.put("bond.volmodel", pct(bv["vol_model"]))
    N.put("bond.volief", pct(bv["vol_ief"]))
    N.put("bond.damcorr", num(bv["corr_annual_vs_damodaran"], 3))
    N.put("bond.damgap", pct(bv["mean_annual_gap_vs_damodaran"], 2, True))
    N.put("bond.damyears", str(bv["damodaran_years"]))
    N.put("bond.iefmonths", str(bv["ief_months"]))
    N.put("bond.iefstart", bv["ief_start"])

    for sid, sn in R["snooping"].items():
        v, b = sn["vs_volmatched"], sn["vs_buy_and_hold"]
        N.put(f"snoop.{sid}.nrules", str(sn["n_rules"]))
        N.put(f"snoop.{sid}.rc", num(v["p_reality_check"]))
        N.put(f"snoop.{sid}.spa", num(v["p_spa"]))
        N.put(f"snoop.{sid}.best", v["best_rule"].replace("_", "\\_"))
        N.put(f"snoop.{sid}.bestdiff", pct(v["best_mean_diff_annual"], 1, True))
        N.put(f"snoop.{sid}.sharepos", pct(v["share_rules_positive"], 0))
        N.put(f"snoop.{sid}.rcbh", num(b["p_reality_check"]))
        N.put(f"snoop.{sid}.spabh", num(b["p_spa"]))
        N.put(f"snoop.{sid}.shareposbh", pct(b["share_rules_positive"], 0))
        N.put(f"snoop.{sid}.bestdiffbh", pct(b["best_mean_diff_annual"], 1, True))
        N.put(f"snoop.{sid}.dsr", num(sn["deflated_sharpe_vs_volmatched"]["dsr"]))
        N.put(f"snoop.{sid}.dsrex", num(sn["deflated_sharpe_over_tbills"]["dsr"], 3))
        N.put(f"snoop.{sid}.pbo", num(sn["pbo_vs_volmatched"]["pbo"]))
        N.put(f"snoop.{sid}.ma50200", pct(sn["mean_diff_by_rule_annual"]["ma_50_200"], 2, True))

    tp = R["tsmom_portfolio"]
    N.put("tsmom.start", tp["start"])
    for f in ["cagr", "vol", "max_dd"]:
        N.put(f"tsmom.{f}", pct(tp["metrics"][f]))
        N.put(f"tsmom.us.{f}", pct(tp["us_metrics_same_months"][f]))
    N.put("tsmom.sharpe", num(tp["metrics"]["sharpe"]))
    N.put("tsmom.us.sharpe", num(tp["us_metrics_same_months"]["sharpe"]))
    N.put("tsmom.corr", num(tp["corr_with_us"]))
    N.put("tsmom.gamma", num(tp["smile"]["gamma"]))
    N.put("tsmom.gammalo", num(tp["smile"]["gamma_ci90"][0]))
    N.put("tsmom.gammahi", num(tp["smile"]["gamma_ci90"][1]))
    N.put("tsmom.beta", num(tp["smile"]["beta"]))
    N.put("tsmom.avgassets", num(tp["avg_assets"], 1))
    worst = tp["worst_us_months"]
    N.put("tsmom.worstwins", str(sum(1 for w in worst if w["tsmom"] > 0)))
    N.put("tsmom.worstn", str(len(worst)))

    mc = R["monte_carlo"]
    N.put("mc.paths", f"{mc['n_paths']:,}")
    for k, tag in [("buy_and_hold", "bh"), ("trend_50_200", "trend"), ("vol_matched", "mix")]:
        N.put(f"mc.{tag}.p50", num(mc[k]["end_wealth_pct"]["p50"]))
        N.put(f"mc.{tag}.p5", num(mc[k]["end_wealth_pct"]["p5"]))
        N.put(f"mc.{tag}.p95", num(mc[k]["end_wealth_pct"]["p95"]))
        N.put(f"mc.{tag}.ddp50", pct(mc[k]["max_dd_pct"]["p50"], 0))
        N.put(f"mc.{tag}.ddp95", pct(mc[k]["max_dd_pct"]["p95"], 0))
        N.put(f"mc.{tag}.ploss", pct(mc[k]["prob_loss"], 0))
        N.put(f"mc.{tag}.pdd30", pct(mc[k]["prob_dd_over_30"], 0))
    N.put("mc.trendbeatsbh", pct(mc["prob_trend_beats_bh"], 0))
    N.put("mc.trendbeatsmix", pct(mc["prob_trend_beats_volmatched"], 0))
    N.put("mc.trendsmallerdd", pct(mc["prob_trend_smaller_dd_than_volmatched"], 0))

    em = R["emerging"]["rules"]
    for k, v in em.items():
        key = k.replace("_", "")
        N.put(f"em.{key}.start", v["eval_start"])
        for s_ in ["buy_and_hold", "trend", "vol_matched"]:
            N.put(f"em.{key}.{s_.replace('_', '')}.cagr", pct(v["metrics"][s_]["cagr"]))
            N.put(f"em.{key}.{s_.replace('_', '')}.sharpe", num(v["metrics"][s_]["sharpe"]))
            N.put(f"em.{key}.{s_.replace('_', '')}.maxdd", pct(v["metrics"][s_]["max_dd"]))
        N.put(f"em.{key}.delta", pct(v["primary_style_delta"]["delta"], 1, True))
        N.put(f"em.{key}.share", pct(v["primary_style_delta"]["share_below_zero"], 0))

    g = R["gold_averaging_check"]
    N.put("gold.start", g["start"])
    N.put("gold.end", g["end"])
    N.put("gold.acavg", num(g["monthly_average"]["autocorr_lag1"]))
    N.put("gold.acgld", num(g["gld_month_end"]["autocorr_lag1"]))
    N.put("gold.acrw", num(g["random_walk_average_autocorr"]))
    N.put("gold.ac75", num(g["autocorr_lag1_average_1975_on"]))
    for k, tag in [("monthly_average", "avg"), ("gld_month_end", "gld")]:
        N.put(f"gold.{tag}.trend", num(g[k]["trend_sharpe"]))
        N.put(f"gold.{tag}.bh", num(g[k]["buy_hold_sharpe"]))

    for e in R["century_events"]:
        slug = "".join(ch for ch in e["event"].lower() if ch.isalnum())
        for k in ["buy_and_hold", "trend_50_200", "vol_matched", "tsmom_12m"]:
            N.put(f"ev.{slug}.{k}", pct(e[k], 1))
            N.put(f"ev.{slug}.{k}.plus12", pct(e[k + "_plus12m"], 1))
            N.put(f"ev.{slug}.{k}.next12", pct(e[k + "_next12m"], 1))

    spy_path = os.path.join(ROOT, "results", "spy_study.json")
    if os.path.exists(spy_path):
        spy = json.load(open(spy_path))
        steps = list(spy["fix_steps"].values())
        for i, st_ in enumerate(steps):
            for f in ["cagr", "max_dd", "bh_cagr", "bh_max_dd"]:
                N.put(f"spy.s{i}.{f.replace('_', '')}", pct(st_[f]))
            for f in ["sharpe", "bh_sharpe", "growth_of_1", "bh_growth_of_1"]:
                N.put(f"spy.s{i}.{f.replace('_', '')}", num(st_[f]))
        N.put("spy.pbo", num(spy["pbo"]))
        N.put("spy.picked", spy["picked"])

    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write(N.tex())
    write_tables(R)


def write_tables(R):
    S = R["samples"]
    # primary test
    rows = []
    for sid in ["C1", "C2", "C3", "C4", "C5", "X1", "X2", "X3", "D1", "D2", "US"]:
        s = S[sid]
        p = s["primary"]
        u = p["secondary"]["ulcer"]
        flag = {"unreachable": "$^\\dagger$", "levered": "$^\\ddagger$"}.get(p["flag"], "")
        if sid in ("C3", "X1", "D1"):
            rows.append("mid")
        rows.append([sid, s["label"], f"{s['eval_start']} to {s['eval_end']}", pct(p["delta"], 1, True) + flag,
                     f"[{pct(p['ci90'][0], 0, True)}, {pct(p['ci90'][1], 0, True)}]", pct(p["share_below_zero"], 0),
                     pct(u["delta"], 1, True), pct(u["share_below_zero"], 0)])
    table("primary.tex", ["", "Sample", "Period", "$\\Delta$ max DD", "90\\% CI", "$P(\\Delta<0)$",
                          "$\\Delta$ Ulcer", "$P(\\Delta<0)$"], rows, "llllrrrr")

    # metrics for the two confirmation samples
    for sid in ["C1", "C2", "US"]:
        rows = []
        for k in ["buy_and_hold", "trend_50_200", "vol_matched", "return_matched", "exposure_matched", "60_40",
                  "60_40_trend", "sma_10m", "tsmom_12m", "tsmom_12m_vol"]:
            m = S[sid]["metrics"].get(k)
            if m is None:
                continue
            rows.append([LABELS[k], pct(m["cagr"]), pct(m["vol"]), num(m["sharpe"]), pct(m["max_dd"]),
                         pct(m["worst_12m"]), pct(m["cvar_5"]), num(m["skew"])])
        table(f"metrics_{sid}.tex", ["", "CAGR", "Vol", "Sharpe", "Max DD", "Worst 12m", "CVaR 5\\%", "Skew"],
              rows, "lrrrrrrr")

    # every sample, short
    rows = []
    for sid in ["C1", "C2", "C3", "C4", "C5", "X1", "X2", "X3", "D1", "D2", "US"]:
        s = S[sid]
        b, t, v = s["metrics"]["buy_and_hold"], s["metrics"]["trend_50_200"], s["metrics"]["vol_matched"]
        if sid in ("C3", "X1", "D1"):
            rows.append("mid")
        rows.append([sid, s["label"], pct(b["cagr"]), pct(t["cagr"]), pct(v["cagr"]), num(b["sharpe"]),
                     num(t["sharpe"]), pct(b["max_dd"]), pct(t["max_dd"]), pct(v["max_dd"])])
    table("overview.tex", ["", "", "\\multicolumn{3}{c}{CAGR}", "\\multicolumn{2}{c}{Sharpe}",
                           "\\multicolumn{3}{c}{Max drawdown}"], [["", "", "B\\&H", "Trend", "Mix", "B\\&H", "Trend",
                                                                   "B\\&H", "Trend", "Mix"], "mid"] + rows,
          "llrrrrrrrr")

    # insurance
    rows = []
    for sid in ["C1", "C2", "C3", "C4", "C5", "US"]:
        s = S[sid]
        for vs, lab in [("insurance_vs_bh", "B\\&H"), ("insurance_vs_volmatched", "mix")]:
            i = s[vs]
            rows.append([f"{sid} vs {lab}", usd(i["premium_dollars_per_year"]),
                         f"[{usd(i['premium_dollars_ci'][0])}, {usd(i['premium_dollars_ci'][1])}]",
                         pct(i["bear_gap_log_per_year"], 1, True), f"{i['episodes_won']} of {i['n_episodes']}",
                         usd(i["mean_dollars_saved_per_episode"]), pct(i["net_log_per_year"], 1, True),
                         f"[{pct(i['net_ci'][0], 1, True)}, {pct(i['net_ci'][1], 1, True)}]"])
    table("insurance.tex", ["", "Premium/yr", "90\\% CI", "Bear gap/yr", "Bears won", "Saved/bear", "Net/yr",
                            "90\\% CI"], rows, "lrrrrrrr")

    # US bear markets
    rows = []
    a, b = S["US"]["insurance_vs_bh"]["episodes"], S["US"]["insurance_vs_volmatched"]["episodes"]
    for e1, e2 in zip(a, b):
        rows.append([f"{e1['from_month']} to {e1['to_month']}", str(e1["months"]), pct(e1["depth"], 0),
                     usd(e1["dollars_saved"]), usd(e2["dollars_saved"])])
    table("episodes_us.tex", ["Fall", "Months", "Depth", "Saved vs B\\&H", "Saved vs mix"], rows, "lrrrr")

    # named events
    rows = []
    for e in R["century_events"]:
        rows.append([e["event"] + (" (seen)" if e["after_2000"] else ""), f"{e['start']} to {e['end']}",
                     pct(e["buy_and_hold"], 0), pct(e["trend_50_200"], 0), pct(e["vol_matched"], 0),
                     pct(e["tsmom_12m"], 0), pct(e["buy_and_hold_plus12m"], 0), pct(e["trend_50_200_plus12m"], 0)])
    table("events.tex", ["Event", "Window", "B\\&H", "50/200", "Mix", "12m mom.", "B\\&H +12m", "50/200 +12m"],
          rows, "llrrrrrr")

    # regimes
    names = [("bear", "Bear market"), ("recovery", "Recovery"), ("crash_months", "Crash months"), ("calm", "Calm"),
             ("high_rates", "High T-bill rate"), ("low_rates", "Low T-bill rate"),
             ("rising_long_rates", "Long rates rising"), ("falling_long_rates", "Long rates falling")]
    rows = []
    for k, lab in names:
        row = [lab]
        for sid in ["C1", "C2"]:
            v = S[sid]["regimes_vs_volmatched"].get(k)
            if v is None:
                row += ["", ""]
                continue
            row += [str(v["months"]), f"{pct(12 * v['mean_gap'], 1, True)} [{pct(12 * v['ci'][0], 0, True)}, "
                                      f"{pct(12 * v['ci'][1], 0, True)}]"]
        rows.append(row)
    table("regimes.tex", ["", "C1 months", "C1 trend $-$ mix, per year", "C2 months", "C2 trend $-$ mix, per year"],
          rows, "lrlrl")

    # execution, Sharpe and the primary statistic side by side
    rows = []
    for sid in ["C1", "C2"]:
        g = {(x["delay"], x["cost_bps"]): x for x in S[sid]["execution"]}
        for c in [0, 5, 25, 50]:
            rows.append([sid if c == 0 else "", f"{c} bps"] + [num(g[(d, c)]["sharpe"]) for d in [0, 1, 2, 3, 5, 10]]
                        + [pct(g[(d, c)]["delta_vs_return_matched"], 1, True) for d in [0, 1, 2, 3, 5, 10]])
        if sid == "C1":
            rows.append("mid")
    table("execution.tex", ["", "Cost", "\\multicolumn{6}{c}{Sharpe over T-bills}",
                            "\\multicolumn{6}{c}{$\\Delta$ max DD vs same growth mix}"],
          [["", "", "0d", "1d", "2d", "3d", "5d", "10d", "0d", "1d", "2d", "3d", "5d", "10d"], "mid"] + rows,
          "llrrrrrrrrrrrr")

    # bear markets dated from local peaks, the sensitivity check
    rows = []
    for sid in ["C1", "C2", "US"]:
        for key, lab in [("insurance_swing_vs_bh", "B\\&H"), ("insurance_swing_vs_volmatched", "mix")]:
            i = S[sid][key]
            rows.append([f"{sid} vs {lab}", f"{i['episodes_won']} of {i['n_episodes']}",
                         usd(i["mean_dollars_saved_per_episode"]), usd(i["premium_dollars_per_year"]),
                         pct(i["net_log_per_year"], 1, True)])
    table("insurance_swing.tex", ["", "Bears won", "Saved/bear", "Premium/yr", "Net/yr"], rows, "lrrrr")
    rows = []
    for e in S["US"]["insurance_swing_vs_bh"]["episodes"]:
        rows.append([f"{e['from_month']} to {e['to_month']}", pct(e["depth"], 0), usd(e["dollars_saved"])])
    table("episodes_swing_us.tex", ["Fall", "Depth", "Saved vs B\\&H"], rows, "lrr")


if __name__ == "__main__":
    write()
