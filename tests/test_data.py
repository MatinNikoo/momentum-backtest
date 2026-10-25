"""The snapshot files the study runs on. Cheap checks that they are what docs/data.md says."""
import numpy as np
import pandas as pd

from trendlab import data as dt


def test_snapshot_coverage():
    d = dt.daily_table()
    assert d["us"].dropna().index[0] == pd.Timestamp("1926-07-01")
    assert d["dev_ex_us"].dropna().index[0] == pd.Timestamp("1990-07-02")
    assert d["ust10"].dropna().index[0] == pd.Timestamp("1962-01-03")
    assert d["gld"].dropna().index[0] == pd.Timestamp("2004-11-19")
    assert d["dbc"].dropna().index[0] == pd.Timestamp("2006-02-07")
    assert d["rf"].dropna().between(-0.0001, 0.001).all()  # a few 1930s months had slightly negative T-bill returns
    for c in ["us", "dev_ex_us", "europe", "japan", "asia_pac_ex_japan", "ust10", "gld", "dbc", "ief"]:
        assert d[c].dropna().abs().max() < 0.25, c


def test_asset_keeps_every_tbill_day():
    # interest from days the asset was closed is moved to its next open day, never dropped
    r, rf = dt.asset("dev_ex_us")
    full = dt.rf_daily()
    full = full[(full.index >= r.index[0]) & (full.index <= r.index[-1])]
    assert np.isclose(rf.sum(), full.sum(), atol=1e-9)


def test_monthly_bonds_are_continuous():
    b = dt.bond_monthly()
    assert b.index[0] <= pd.Period("1926-07", "M")
    full = pd.period_range(pd.Period("1926-07", "M"), b.index[-1], freq="M")
    assert b.reindex(full).loc[pd.Period("1926-07", "M"):].notna().all()
