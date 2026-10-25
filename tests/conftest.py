import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fake_returns(n=3000, seed=0, drift=0.0004, vol=0.011, start="2000-01-03"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = pd.Series(rng.normal(drift, vol, n), index=idx, name="fake")
    rf = pd.Series(0.0001, index=idx, name="rf")
    return r, rf


@pytest.fixture
def fake():
    return fake_returns()
