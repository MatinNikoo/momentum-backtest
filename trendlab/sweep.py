"""
Parameter sweeps and walk forward tests. Every cell of a sweep is a full
backtest of the config with some parameters swapped, all scored on the same
dates so a slow rule isn't compared on a shorter window than a fast one.
"""

from __future__ import annotations

import copy
import itertools
import tomllib

import numpy as np
import pandas as pd

from . import config as cf
from . import metrics as mt
from . import rules as rl
from . import stats as st

MAX_CELLS = 2000
# metric name -> (label, higher is better)
METRICS = {
    "sharpe": ("Sharpe over T-bills", True),
    "cagr": ("CAGR", True),
    "calmar": ("CAGR over max drawdown", True),
    "max_dd": ("Max drawdown", False),
    "ulcer": ("Ulcer index", False),
    "vol": ("Volatility", False),
}


def _number(text: str):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def parse_grid(items) -> dict:
    """
    ["fast=20,50,100", "slow=100:300:50"] -> {"fast": [20, 50, 100], "slow": [100, 150, 200, 250, 300]}.
    start:stop:step includes both ends. A name with a dot like execution.delay
    reaches outside the rule, a bare name is a rule parameter.
    """
    grid = {}
    for item in items:
        key, eq, vals = item.partition("=")
        key = key.strip()
        if not eq or not key or not vals.strip():
            raise ValueError(f"can't read grid {item!r}, use name=v1,v2,v3 or name=start:stop:step")
        if ":" in vals and "," not in vals:
            try:
                a, b, step = (float(x) for x in vals.split(":"))
            except ValueError:
                raise ValueError(f"can't read the range in {item!r}, use start:stop:step") from None
            if step <= 0 or b < a:
                raise ValueError(f"range in {item!r} needs step above 0 and stop at least start")
            n = int(round((b - a) / step))
            out = [a + i * step for i in range(n + 1)]
            out = [int(v) if all(float(x).is_integer() for x in (a, b, step)) else round(v, 10) for v in out]
        else:
            out = [_number(v.strip()) for v in vals.split(",")]
        grid[key] = out
    return grid


def label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def spec_key(spec: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(spec.items()))


SIZING_TO_RULE = {"vol_target": "vol_target", "max_leverage": "cap", "vol_window": "com"}


def with_params(cfg: dict, params: dict) -> dict:
    """A copy of the config with values swapped. Bare names are rule parameters, dotted
    ones like execution.delay reach the other tables."""
    c2 = copy.deepcopy(cfg)
    for k, v in params.items():
        section, _, name = k.rpartition(".")
        if section == "sizing" and name in SIZING_TO_RULE:
            c2["rule"][SIZING_TO_RULE[name]] = v
            continue
        section = section or "rule"
        if not isinstance(c2.get(section), dict):
            raise ValueError(f"grid name {k!r} points at [{section}], which this config doesn't have")
        allowed = {"execution": set(cf.DEFAULTS["execution"]), "sizing": cf.SIZING_KEYS}
        if section in allowed and name not in allowed[section]:
            raise ValueError(f"grid name {k!r}: [{section}] has no option {name!r}")
        c2[section][name] = v
    rl.check_spec(c2["rule"])
    return c2


def expand(cfg: dict, grid: dict):
    """(params, config) for every combination in the grid, leaving out fast >= slow."""
    if not grid:
        raise ValueError("the sweep needs a grid, like --grid fast=20,50,100 --grid slow=100,200")
    size = int(np.prod([len(v) for v in grid.values()]))
    if size > MAX_CELLS:
        raise ValueError(f"that grid has {size} cells, the limit is {MAX_CELLS}")
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        c2 = with_params(cfg, params)
        sp = c2["rule"]
        if "fast" in sp and "slow" in sp and sp["fast"] >= sp["slow"]:
            continue
        yield params, c2


def _trim(res: dict, start) -> dict:
    keep = res["strategy"].index >= start
    out = dict(res)
    for k in ("strategy", "benchmark", "rf"):
        out[k] = res[k][keep]
    out["eval_start"] = start
    return out


def run_grid(cfg: dict, grid: dict, loader=None) -> dict:
    """Run every cell and the config as it stands, all trimmed to the latest start among them."""
    cells = [(p, cf.run(c, loader)) for p, c in expand(cfg, grid)]
    if not cells:
        raise ValueError("every combination in the grid was skipped (fast must be smaller than slow)")
    base = cf.run(cfg, loader)
    start = max([r["eval_start"] for _, r in cells] + [base["eval_start"]])
    return {"cells": [(p, _trim(r, start)) for p, r in cells], "base": _trim(base, start), "start": start}


def metric_row(res: dict) -> dict:
    m = cf.monthly(res)
    s = mt.summary(m["strategy"].values, m["rf"].values)
    row = {k: s[k] for k in ["cagr", "vol", "sharpe", "max_dd", "ulcer"]}
    row["calmar"] = row["cagr"] / row["max_dd"] if row["max_dd"] > 0 else float("nan")
    row["switches_a_year"] = cf.turnover(res)["switches_per_year_per_asset"]
    return row


def table(runs: dict) -> pd.DataFrame:
    rows = [{**p, **metric_row(r)} for p, r in runs["cells"]]
    return pd.DataFrame(rows)


def best_index(tab: pd.DataFrame, metric: str):
    col = tab[metric]
    return col.idxmax() if METRICS[metric][1] else col.idxmin()


def overfit_check(runs: dict) -> dict | None:
    """Deflated Sharpe of the best cell and the probability of backtest overfitting across the cells."""
    cols = {}
    for p, r in runs["cells"]:
        m = cf.monthly(r)
        cols[label(p)] = m["strategy"] - m["rf"]
    R = pd.DataFrame(cols).dropna()
    if R.shape[1] < 2 or len(R) < 48:
        return None
    srs = R.mean() / R.std()
    best = srs.idxmax()
    out = {"cells": int(R.shape[1]), "best_cell": best, "dsr": st.deflated_sharpe(R[best].values, srs.values)["dsr"]}
    if len(R) >= 96:
        out["pbo"] = st.pbo_cscv(R.values, n_blocks=16 if len(R) >= 160 else 8)["pbo"]
    return out


# ---------------------------------------------------------------------------
# walk forward
# ---------------------------------------------------------------------------
def _score(metric: str, s: np.ndarray, rf: np.ndarray) -> float:
    if metric == "sharpe":
        return mt.sharpe(s, rf)
    if metric == "cagr":
        return mt.cagr(s)
    if metric == "vol":
        return mt.vol(s)
    if metric == "max_dd":
        return mt.max_dd_depth(s)
    if metric == "ulcer":
        return mt.ulcer(s)
    dd = mt.max_dd_depth(s)
    return mt.cagr(s) / dd if dd > 0 else float("nan")


def walk_forward(runs: dict, metric: str = "sharpe", train_years: float = 10, test_years: float = 1,
                 expanding: bool = False) -> dict:
    """
    At the end of each training window pick the cell with the best metric,
    then hold that cell's own returns through the next test window. Cells'
    signals only use the past, so slicing their full history is the same as
    running them live. Switching from one cell to another is assumed free.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    higher = METRICS[metric][1]
    if len(runs["cells"]) < 2:
        raise ValueError("walk forward needs at least two parameter sets to choose between, add a --grid")
    names = [label(p) for p, _ in runs["cells"]]
    monthly = [cf.monthly(r) for _, r in runs["cells"]]
    strat = pd.DataFrame({n: m["strategy"] for n, m in zip(names, monthly)}).dropna()
    rf = monthly[0]["rf"].reindex(strat.index)
    bench = monthly[0]["benchmark"].reindex(strat.index)
    base = cf.monthly(runs["base"])["strategy"].reindex(strat.index)
    n, tr, te = len(strat), int(round(12 * train_years)), int(round(12 * test_years))
    if tr < 12 or te < 1:
        raise ValueError("training needs at least 1 year and testing at least 1 month")
    if n < tr + te:
        raise ValueError(f"only {n} months of data, a {train_years} year training window plus "
                         f"{test_years} year test needs {tr + te}")
    folds, pieces = [], []
    for i in range(tr, n, te):
        j = min(i + te, n)
        lo = 0 if expanding else i - tr
        scores = np.array([_score(metric, strat.iloc[lo:i][c].values, rf.iloc[lo:i].values) for c in names])
        scores = np.where(np.isfinite(scores), scores, -np.inf if higher else np.inf)
        k = int(np.argmax(scores) if higher else np.argmin(scores))
        chosen = runs["cells"][k][0]
        pieces.append(strat.iloc[i:j][names[k]])
        test_score = _score(metric, strat.iloc[i:j][names[k]].values, rf.iloc[i:j].values)
        folds.append({"test_from": str(strat.index[i]), "test_to": str(strat.index[j - 1]), **chosen,
                      "train_score": float(scores[k]), "test_score": float(test_score)})
    oos = pd.concat(pieces)
    span = oos.index
    frame = pd.DataFrame({"walk_forward": oos, "fixed_config": base.reindex(span),
                          "mean_of_cells": strat.reindex(span).mean(axis=1),
                          "buy_and_hold": bench.reindex(span), "rf": rf.reindex(span)})
    rows = {k: mt.summary(frame[k].values, frame["rf"].values)
            for k in ["walk_forward", "fixed_config", "mean_of_cells", "buy_and_hold"]}
    return {"folds": pd.DataFrame(folds), "returns": frame, "summary": pd.DataFrame(rows),
            "metric": metric, "train_years": train_years, "test_years": test_years, "expanding": expanding}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def log_axis(ax):
    """Log scale with plain number ticks at 1, 2 and 5 per decade instead of 10^x."""
    import matplotlib.ticker as tk
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(tk.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_minor_locator(tk.NullLocator())


def heatmap(tab: pd.DataFrame, params: list, metric: str, path: str, base_params=None, benchmark=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_, higher = METRICS[metric]
    pct = metric in ("cagr", "max_dd", "ulcer", "vol")
    fig, ax = plt.subplots(figsize=(7.5, 5.2) if len(params) == 2 else (7.5, 4.2))
    if len(params) == 1:
        p = params[0]
        ax.plot(tab[p].astype(str), tab[metric], marker="o", color="#C44E52")
        if benchmark is not None:
            ax.axhline(benchmark, color="#5B6B8C", ls="--", label="buy and hold")
            ax.legend(frameon=False)
        ax.set_xlabel(p)
        ax.set_ylabel(label_)
        ax.grid(alpha=0.3)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    else:
        py, px = params
        pv = tab.pivot(index=py, columns=px, values=metric).sort_index().sort_index(axis=1)
        im = ax.imshow(pv.values, cmap="Blues" if higher else "Blues_r", aspect="auto", origin="upper")
        ax.set_xticks(range(pv.shape[1]), [str(c) for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]), [str(i) for i in pv.index])
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        lo, hi = np.nanmin(pv.values), np.nanmax(pv.values)
        for (i, j), v in np.ndenumerate(pv.values):
            if np.isfinite(v):
                dark = (v - lo) / (hi - lo + 1e-12) > 0.55 if higher else (v - lo) / (hi - lo + 1e-12) < 0.45
                ax.text(j, i, f"{v:.1%}" if pct else f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if dark else "#222222")
        if base_params and all(k in base_params for k in (py, px)):
            try:
                i, j = list(pv.index).index(base_params[py]), list(pv.columns).index(base_params[px])
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#C44E52", lw=2))
            except ValueError:
                pass
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label_)
        if pct:
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ref = ""
    if benchmark is not None:
        ref = f", buy and hold {benchmark:.1%}" if pct else f", buy and hold {benchmark:.2f}"
    ax.set_title(f"{title}, {label_.lower()}{ref}".strip(", "), fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_walk_forward(wf: dict, path: str, name: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, folds = wf["returns"], wf["folds"]
    pcols = [c for c in folds.columns if c not in ("test_from", "test_to", "train_score", "test_score")]
    n = 1 + min(len(pcols), 2)
    fig, ax = plt.subplots(n, 1, figsize=(8, 3.6 + 1.3 * (n - 1)), sharex=True,
                           gridspec_kw={"height_ratios": [3] + [1] * (n - 1)})
    ax = np.atleast_1d(ax)
    t = f.index.to_timestamp()
    for col, lab, color, ls in [("walk_forward", "walk forward", "#C44E52", "-"),
                                ("fixed_config", "fixed config", "#8C8C8C", "--"),
                                ("buy_and_hold", "buy and hold", "#5B6B8C", "-")]:
        ax[0].plot(t, np.cumprod(1 + f[col].values), color=color, ls=ls, label=lab)
    log_axis(ax[0])
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"{name} out of sample, {wf['train_years']:g} year "
                    f"{'expanding' if wf['expanding'] else 'rolling'} training, picked on {wf['metric']}".strip(),
                    fontsize=10, loc="left")
    starts = pd.PeriodIndex(folds["test_from"], freq="M").to_timestamp()
    for a, c in zip(ax[1:], pcols[:2]):
        a.step(starts, folds[c], where="post", color="#5B6B8C")
        a.set_ylabel(c, fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
