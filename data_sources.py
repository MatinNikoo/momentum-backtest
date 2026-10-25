"""
data_sources.py

Downloads every series the protection study uses and turns them into the
small snapshot files in data/snapshots/. Raw downloads go to data/raw/,
which is gitignored. The study itself only reads the snapshots, so it runs
offline and gives the same numbers every time.

    python data_sources.py          # download (or reuse data/raw) and rebuild snapshots

Where each series comes from is written up in docs/data.md.
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
SNAP = os.path.join(HERE, "data", "snapshots")

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "us": "F-F_Research_Data_Factors_daily_CSV.zip",
    "dev_ex_us": "Developed_ex_US_3_Factors_Daily_CSV.zip",
    "europe": "Europe_3_Factors_Daily_CSV.zip",
    "japan": "Japan_3_Factors_Daily_CSV.zip",
    "asia_pac_ex_japan": "Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip",
}
FRENCH_US_MONTHLY = "F-F_Research_Data_Factors_CSV.zip"  # monthly RF, used to fix the rounded daily RF
FRENCH_EM_MONTHLY = "Emerging_5_Factors_CSV.zip"  # no daily file exists for emerging markets
GOLD_MONTHLY = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
DAMODARAN = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html"
YAHOO = ["GLD", "DBC", "IEF"]

BOND_MATURITY = 10.0


# ----------------------------------------------------------------------------
# downloading, with everything cached in data/raw
# ----------------------------------------------------------------------------
def _get(url: str, fname: str, tries: int = 4) -> bytes:
    path = os.path.join(RAW, fname)
    if os.path.exists(path):
        return open(path, "rb").read()
    import requests

    os.makedirs(RAW, exist_ok=True)
    status = None
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60)
            status = r.status_code
            if r.status_code == 200:
                open(path, "wb").write(r.content)
                return r.content
        except requests.exceptions.RequestException as e:  # the proxy drops connections now and then
            status = str(e)[:80]
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"could not download {url} ({status})")


def french_daily(key: str) -> pd.DataFrame:
    """Ken French daily file -> DataFrame of decimal returns, columns mkt and rf."""
    raw = _get(FRENCH + FRENCH_FILES[key], FRENCH_FILES[key])
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        # data rows start with an 8 digit date, everything else is header or notes
        if len(parts) >= 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["date", "mkt_rf", "smb", "hml", "rf"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    df = df[~df.index.duplicated()]
    # missing values are coded as -99.99 or -999 in these files
    df = df.where(df > -0.99)
    return pd.DataFrame({"mkt": df["mkt_rf"] + df["rf"], "rf": df["rf"]}).dropna()


def french_monthly_rf() -> pd.Series:
    """Monthly one month T-bill return from the Ken French US factors file."""
    raw = _get(FRENCH + FRENCH_US_MONTHLY, FRENCH_US_MONTHLY)
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and len(parts[0]) == 4 and parts[0].isdigit() and rows:
            break  # the annual table starts here
        if len(parts) >= 5 and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["month", "mkt_rf", "smb", "hml", "rf"]
    idx = pd.PeriodIndex(pd.to_datetime(df["month"], format="%Y%m"), freq="M")
    return pd.Series(df["rf"].astype(float).values / 100.0, index=idx, name="rf_monthly")


def rescale_daily_rf(daily: pd.Series, monthly: pd.Series) -> pd.Series:
    """
    The daily RF in the Ken French file is the monthly rate spread over the
    trading days and rounded to 0.01%, so in some years it is off by about a
    point. Spread the monthly rate evenly again without the rounding, so the
    days of each month compound to exactly the monthly RF. Months the monthly
    file doesn't have yet keep the daily numbers.
    """
    month = daily.index.to_period("M")
    n = daily.groupby(month).transform("size")
    target = monthly.reindex(month).values
    fixed = (1.0 + target) ** (1.0 / n.values) - 1.0
    return pd.Series(np.where(np.isnan(target), daily.values, fixed), index=daily.index, name=daily.name)


def french_monthly_em() -> pd.DataFrame:
    """Ken French emerging markets, monthly only. Returns columns mkt and rf."""
    raw = _get(FRENCH + FRENCH_EM_MONTHLY, FRENCH_EM_MONTHLY)
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and len(parts[0]) == 4 and parts[0].isdigit() and rows:
            break  # the annual table starts here
        if len(parts) >= 7 and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts)
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["month", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["month"], format="%Y%m"), freq="M")
    df = df.set_index("month").astype(float) / 100.0
    df = df.where(df > -0.99)
    return pd.DataFrame({"mkt": df["mkt_rf"] + df["rf"], "rf": df["rf"]}).dropna()


def gold_monthly_average() -> pd.Series:
    """World Bank pink sheet gold, monthly AVERAGE prices (via datahub). Only used to show the averaging trap."""
    raw = _get(GOLD_MONTHLY, "gold_monthly.csv").decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["month", "price"]
    s = df.set_index(pd.PeriodIndex(pd.to_datetime(df["month"]), freq="M"))["price"].astype(float)
    return s[~s.index.duplicated(keep="last")]


def fred(series: str) -> pd.Series:
    raw = _get(FRED + series, f"fred_{series}.csv").decode()
    df = pd.read_csv(io.StringIO(raw), na_values=["."])
    df.columns = ["date", series]
    s = df.set_index(pd.to_datetime(df["date"]))[series].astype(float).dropna()
    return s / 100.0  # percent to decimal


def shiller_long_rate() -> pd.Series:
    raw = _get(SHILLER, "shiller_sp500.csv").decode()
    df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"]).set_index("Date")
    s = df["Long Interest Rate"]
    return s[s > 0] / 100.0  # rows after 2023-06 are filled with 0


def damodaran_tbond() -> pd.Series:
    """Damodaran's annual 10 year Treasury total returns, used only to check my bond model."""
    raw = _get(DAMODARAN, "damodaran_histretSP.html").decode("latin-1")
    t = pd.read_html(io.StringIO(raw))[0]
    out = {}
    for _, row in t.iterrows():
        y = str(row.iloc[0]).strip()
        if y.isdigit() and len(y) == 4:
            v = str(row.iloc[4]).replace("%", "").strip()
            try:
                out[int(y)] = float(v) / 100.0
            except ValueError:
                pass
    return pd.Series(out, name="damodaran_tbond").sort_index()


def yahoo(ticker: str, tries: int = 5) -> pd.Series:
    """Adjusted close from Yahoo. It rate limits a lot, so go slow and cache."""
    path = os.path.join(RAW, f"yahoo_{ticker}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s.astype(float).rename(ticker)
    import yfinance as yf

    for k in range(tries):
        try:
            df = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
            if len(df):
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                os.makedirs(RAW, exist_ok=True)
                close.rename(ticker).to_csv(path)
                return close.astype(float).rename(ticker)
        except Exception as e:  # yahoo throws all sorts of things when it rate limits
            print(f"  yahoo {ticker} try {k + 1} failed: {e}")
        time.sleep(20 * (k + 1))
    raise RuntimeError(f"no data from Yahoo for {ticker}")


# ----------------------------------------------------------------------------
# bond returns from yields
# ----------------------------------------------------------------------------
def par_bond_price(y, coupon, maturity):
    """
    Price per 1 of face for a bond with semiannual coupons, annual coupon
    rate `coupon`, yield `y` (bond equivalent) and `maturity` in years.
    Fractional maturities just use a fractional number of periods.
    """
    y = np.asarray(y, dtype=float)
    n = 2.0 * np.asarray(maturity, dtype=float)
    h = y / 2.0
    disc = (1.0 + h) ** (-n)
    # as y goes to 0 the annuity term goes to n, handle it so tiny yields don't blow up
    annuity = np.where(np.abs(h) > 1e-10, (1.0 - disc) / np.where(np.abs(h) > 1e-10, h, 1.0), n)
    return coupon / 2.0 * annuity + disc


def bond_returns_from_yields(yields: pd.Series, maturity: float = BOND_MATURITY,
                             daily: bool = True) -> pd.Series:
    """
    Total return of a constant maturity bond rolled every period.

    Buy a par bond at last period's yield y0 (so the coupon is y0), then
    reprice it at this period's yield y1 with maturity shortened by the time
    that passed. Add the coupon accrued over that time. With daily data the
    time step is actual calendar days / 365, with monthly data it's 1/12.
    """
    y = yields.dropna().astype(float)
    y0, y1 = y.shift(1), y
    if daily:
        dt = y.index.to_series().diff().dt.days / 365.0
    else:
        dt = pd.Series(1.0 / 12.0, index=y.index)
    price = par_bond_price(y1.values, y0.values, maturity - dt.values)
    ret = price - 1.0 + y0.values * dt.values
    return pd.Series(ret, index=y.index).iloc[1:]


# ----------------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------------
def build_snapshots() -> None:
    os.makedirs(SNAP, exist_ok=True)

    print("Ken French daily files")
    fr = {k: french_daily(k) for k in FRENCH_FILES}
    rf = rescale_daily_rf(fr["us"]["rf"], french_monthly_rf()).rename("rf")

    print("FRED and Shiller yields")
    dgs10 = fred("DGS10")
    gs10 = fred("GS10")
    shiller = shiller_long_rate()

    print("Yahoo ETFs (slow on purpose)")
    etf = {}
    for t in YAHOO:
        etf[t] = yahoo(t)
        time.sleep(5)

    print("Emerging markets (monthly) and gold monthly averages")
    em = french_monthly_em()
    gold_avg = gold_monthly_average()

    print("Damodaran annual bond returns")
    dam = damodaran_tbond()

    # ---- daily table ----
    # outer join so no series loses a day just because another market was closed
    cols = {k: v["mkt"] for k, v in fr.items()}
    cols["rf"] = rf
    cols["ust10"] = bond_returns_from_yields(dgs10, daily=True)
    cols["dgs10"] = dgs10
    for t, s in etf.items():
        cols[t.lower()] = s.pct_change().iloc[1:]
    daily = pd.concat(cols, axis=1, sort=True)
    daily.index.name = "date"
    daily.round(10).to_csv(os.path.join(SNAP, "daily_returns.csv.gz"), compression="gzip")

    # ---- monthly long yield, Shiller before 1953-04 and FRED GS10 after ----
    sh = shiller.copy()
    sh.index = sh.index.to_period("M")
    g = gs10.copy()
    g.index = g.index.to_period("M")
    long_yield = pd.concat([sh[sh.index < pd.Period("1953-04", "M")], g]).sort_index()
    long_yield = long_yield[~long_yield.index.duplicated(keep="last")]
    ly = long_yield.copy()
    ly.index = ly.index.to_timestamp()
    bond_m_from_monthly = bond_returns_from_yields(ly, daily=False)
    bond_m_from_monthly.index = bond_m_from_monthly.index.to_period("M")

    monthly = pd.DataFrame({
        "long_yield": long_yield,
        "ust10_monthly_model": bond_m_from_monthly,
    })
    monthly["em"] = em["mkt"]
    monthly["em_rf"] = em["rf"]
    monthly["gold_avg_price"] = gold_avg[gold_avg.index >= pd.Period("1968-01", "M")]
    monthly.index.name = "month"
    monthly.round(10).to_csv(os.path.join(SNAP, "monthly_yields_and_bonds.csv"))

    dam.index.name = "year"
    dam.round(6).to_csv(os.path.join(SNAP, "damodaran_tbond_annual.csv"))
    print("snapshots written to", SNAP)


if __name__ == "__main__":
    build_snapshots()
