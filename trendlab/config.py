"""
Config files and running a strategy from one.

A config is a small TOML file, see configs/ for examples and docs/usage.md for
every option. The only required parts are [data] assets and [rule].
"""

from __future__ import annotations

import os
import tomllib

import numpy as np
import pandas as pd

from . import data as dt
from . import engine as en
from . import metrics as mt
from . import rules as rl

DEFAULTS = {
    "execution": {"delay": 1, "cost_bps": 5.0},
    "portfolio": {"weights": "equal"},
    "sizing": {},
    "report": {"n_boot": 1000, "mean_block": 12, "seed": 0},
}
SIZING_KEYS = {"fraction", "vol_target", "max_leverage", "vol_window"}
FILE_KEYS = {"path", "column", "kind", "date_column"}


def load(path: str, overrides=()) -> dict:
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} isn't valid TOML ({e})") from None
    apply_overrides(cfg, overrides)
    return normalize(cfg, name=os.path.splitext(os.path.basename(path))[0],
                     base_dir=os.path.dirname(os.path.abspath(path)))


def apply_overrides(cfg: dict, items) -> dict:
    """Set values from "section.key=value" strings, like execution.cost_bps=10 or rule.fast=20."""
    for item in items:
        key, eq, raw = item.partition("=")
        parts = key.strip().split(".")
        if not eq or len(parts) < 2 or not all(parts):
            raise ValueError(f"can't read --set {item!r}, use section.key=value like execution.cost_bps=10")
        try:
            value = tomllib.loads(f"v = {raw.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw.strip()
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {item!r}: {p} isn't a table")
        node[parts[-1]] = value
    return cfg


def normalize(cfg: dict, name: str = "strategy", base_dir: str | None = None) -> dict:
    out = {k: dict(v) for k, v in DEFAULTS.items()}
    for k, v in cfg.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v
    out.setdefault("name", name)
    out["_dir"] = base_dir or os.getcwd()
    if "data" not in out or not out["data"].get("assets"):
        raise ValueError("config needs [data] with a list of assets")
    if "rule" not in out or "rule" not in out["rule"]:
        raise ValueError(f"config needs a [rule] table with a rule name, pick one of {', '.join(sorted(rl.RULES))}")
    out["data"]["files"] = _files(out["data"].get("files", {}), out["_dir"])
    unknown = [a for a in out["data"]["assets"] if a not in dt.ASSETS and a not in out["data"]["files"]]
    if unknown:
        raise ValueError(f"unknown assets {unknown}, pick from {sorted(dt.ASSETS)} "
                         f"or point to your own csv under [data.files]")
    ex = out["execution"]
    if not isinstance(ex["delay"], int) or ex["delay"] < 0:
        raise ValueError("delay is a whole number of days, 0 or more")
    if not isinstance(ex["cost_bps"], (int, float)) or ex["cost_bps"] < 0:
        raise ValueError("cost_bps is a number of basis points, 0 or more")
    out["rule"] = _sized_rule(out["rule"], out["sizing"])
    rl.check_spec(out["rule"])
    return out


def _files(files: dict, base_dir: str) -> dict:
    out = {}
    for name, f in files.items():
        f = {"path": f} if isinstance(f, str) else dict(f)
        bad = set(f) - FILE_KEYS
        if "path" not in f or bad:
            raise ValueError(f"[data.files.{name}] needs a path and may have column, kind and date_column"
                             + (f", not {sorted(bad)}" if bad else ""))
        f["path"] = os.path.normpath(os.path.join(base_dir, os.path.expanduser(f["path"])))
        out[name] = f
    return out


def _sized_rule(rule: dict, sizing: dict) -> dict:
    """Fold the [sizing] options that belong to the rule (vol targeting) into the rule spec."""
    bad = set(sizing) - SIZING_KEYS
    if bad:
        raise ValueError(f"unknown [sizing] option {sorted(bad)[0]!r}, options are {', '.join(sorted(SIZING_KEYS))}")
    rule = dict(rule)
    if sizing.get("fraction") is not None and sizing["fraction"] <= 0:
        raise ValueError("sizing fraction must be above 0")
    for src, dst in [("vol_target", "vol_target"), ("max_leverage", "cap"), ("vol_window", "com")]:
        if sizing.get(src) is not None:
            if dst in rule:
                raise ValueError(f"{dst} is set in both [rule] and [sizing], keep one")
            rule[dst] = sizing[src]
    if rule.get("vol_target") is not None and rule["vol_target"] <= 0:
        raise ValueError("vol_target is an annual volatility like 0.10, it has to be above 0")
    return rule


def weights_for(cfg: dict) -> dict[str, float]:
    assets = cfg["data"]["assets"]
    w = cfg["portfolio"]["weights"]
    if w == "equal":
        return {a: 1.0 / len(assets) for a in assets}
    if isinstance(w, dict):
        missing = [a for a in assets if a not in w]
        if missing:
            raise ValueError(f"weights missing for {missing}")
        out = {a: float(w[a]) for a in assets}
        if min(out.values()) < 0 or sum(out.values()) <= 0:
            raise ValueError("portfolio weights can't be negative and need a positive total")
        return out
    raise ValueError("portfolio weights must be 'equal' or a table like {us = 0.6, ust10 = 0.4}")


def default_loader(cfg: dict):
    """Snapshot assets by name, your own csv files for the names under [data.files]."""
    files, data = cfg["data"]["files"], cfg["data"]

    def load_asset(name):
        if name in files:
            f = files[name]
            return dt.from_csv(f["path"], column=f.get("column"), kind=f.get("kind", "price"),
                               date_column=f.get("date_column"), name=name, cash=data.get("cash", "tbill"))
        r, rf = dt.asset(name)
        if data.get("cash", "tbill") != "tbill":
            rf = dt.cash_returns(r.index, data["cash"]).rename("rf")
        return r, rf
    return load_asset


def run(cfg: dict, loader=None) -> dict:
    """
    Run the config. `loader(name)` returns (daily returns, daily T-bill
    returns) and defaults to the snapshots plus any csv under [data.files].
    Returns a dict with the daily strategy, the passive book with the same
    weights, T-bills, and each asset's own backtest.
    """
    loader = loader or default_loader(cfg)
    ex, fraction = cfg["execution"], cfg["sizing"].get("fraction", 1.0)
    sleeves, passive, rfs, per_asset = {}, {}, {}, {}
    for a in cfg["data"]["assets"]:
        r, rf = loader(a)
        first, last = cfg["data"].get("start"), cfg["data"].get("end")
        keep = np.ones(len(r), bool)
        if first:
            keep &= r.index >= pd.Timestamp(first)
        if last:
            keep &= r.index <= pd.Timestamp(last)
        r, rf = r[keep], rf[keep]
        need = rl.warmup_days(cfg["rule"]) + ex["delay"] + 2
        if len(r) < need + 21:
            raise ValueError(f"{a} has {len(r)} days in the chosen date range and the rule needs about "
                             f"{need} just to start, widen [data] start and end")
        px = en.total_return_index(r)
        sig = rl.make_signal(cfg["rule"], px, rf) * fraction
        pos = en.positions_from_signal(sig, ex["delay"])
        res = en.run(r, pos, rf, ex["cost_bps"])
        per_asset[a] = res
        sleeves[a], passive[a], rfs[a] = res["strategy"], res["asset"], res["rf"]

    w = weights_for(cfg)
    if len(sleeves) == 1:
        a = next(iter(sleeves))
        strat, book, rf = sleeves[a], passive[a], rfs[a]
    else:
        strat = en.run_portfolio(sleeves, w)
        book = en.run_portfolio(passive, w)
        # each sleeve books T-bill interest on its own trading days, a weighted average keeps the total right
        tot = sum(w.values())
        rf = (pd.DataFrame(rfs).fillna(0.0) * pd.Series(w) / tot).sum(axis=1).reindex(strat.index).fillna(0.0)

    start = cfg["data"].get("eval_start")
    start = pd.Timestamp(start) if start else auto_start(per_asset, cfg["rule"], ex["delay"])
    keep = strat.index >= start
    if keep.sum() < 24:
        raise ValueError(f"only {int(keep.sum())} days are left after the evaluation start {start.date()}, "
                         f"the data ends {strat.index[-1].date()}")
    return {"name": cfg["name"], "strategy": strat[keep], "benchmark": book[keep],
            "rf": rf[keep], "per_asset": per_asset, "weights": w, "eval_start": start}


def auto_start(per_asset: dict, spec: dict, delay: int) -> pd.Timestamp:
    """First month start after every asset's signal has had enough history to exist."""
    need = rl.warmup_days(spec) + delay + 1
    starts = [res.index[min(need, len(res) - 1)] for res in per_asset.values()]
    return (max(starts) + pd.offsets.MonthBegin(1)).normalize()


def turnover(result: dict) -> dict:
    """How often the rule switches and how much of the time it is invested, averaged over assets."""
    pos = pd.concat([r["position"] for r in result["per_asset"].values()], axis=1)
    pos = pos[pos.index >= result["eval_start"]]
    years = max((pos.index[-1] - pos.index[0]).days / 365.25, 1e-9)  # calendar years, old data has Saturdays
    switches = (pos.diff().abs() > 0).sum().sum() / years / pos.shape[1]
    return {"switches_per_year_per_asset": float(switches), "avg_exposure": float(pos.mean().mean())}


def monthly(result: dict) -> pd.DataFrame:
    """Monthly strategy, benchmark and T-bill returns. A month that isn't finished yet is left out."""
    df = pd.DataFrame({"strategy": result["strategy"], "benchmark": result["benchmark"], "rf": result["rf"]})
    m = mt.to_monthly(df)
    return m if rl.month_is_complete(df.index[-1]) else m.iloc[:-1]


def headline(result: dict) -> pd.DataFrame:
    m = monthly(result)
    rows = {k: mt.summary(m[k].values, m["rf"].values) for k in ["strategy", "benchmark"]}
    return pd.DataFrame(rows)


def quick(path: str, rule: str = "ma_crossover", column=None, kind: str = "price", overrides=()) -> dict:
    """A config for one csv of prices, so a rule can be tried without writing a toml file."""
    if not os.path.exists(path):
        raise ValueError(f"price file not found: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    file = {"path": path, "kind": kind}
    if column:
        file["column"] = column
    cfg = {"name": name, "data": {"assets": [name], "files": {name: file}}, "rule": {"rule": rule}}
    apply_overrides(cfg, overrides)
    return normalize(cfg, name=name, base_dir=os.getcwd())
