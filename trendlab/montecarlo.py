"""
Monte Carlo of future paths by block bootstrapping history. It answers
"if the next ten years look like a reshuffled version of the past, what range
of outcomes should I expect", which is a big if.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import stationary_bootstrap_matrix


def simulate(monthly: pd.DataFrame, horizon: int = 120, n_paths: int = 10_000,
             mean_block: int = 12, seed: int = 0) -> np.ndarray:
    """Wealth paths, shape (n_paths, horizon, n_columns). Columns are resampled together."""
    X = monthly.to_numpy(dtype=float)
    idx = stationary_bootstrap_matrix(len(X), mean_block, n_paths, seed, length=horizon)
    return np.cumprod(1.0 + X[idx], axis=1)


def summarize(wealth: np.ndarray, names) -> pd.DataFrame:
    end = wealth[:, -1, :]
    start = np.ones((wealth.shape[0], 1, wealth.shape[2]))
    peak = np.maximum.accumulate(np.concatenate([start, wealth], axis=1), axis=1)[:, 1:, :]
    mdd = -(wealth / peak - 1.0).min(axis=1)
    years = wealth.shape[1] / 12
    rows = {}
    for j, n in enumerate(names):
        e = end[:, j]
        rows[n] = {
            "median_cagr": float(np.median(e) ** (1 / years) - 1),
            "p5_end_wealth": float(np.percentile(e, 5)),
            "p50_end_wealth": float(np.percentile(e, 50)),
            "p95_end_wealth": float(np.percentile(e, 95)),
            "prob_loss": float((e < 1).mean()),
            "median_max_dd": float(np.median(mdd[:, j])),
            "p95_max_dd": float(np.percentile(mdd[:, j], 95)),
        }
    return pd.DataFrame(rows)
