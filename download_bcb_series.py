"""
download_bcb_series.py
======================
Downloads daily series from the Banco Central do Brasil SGS API:

  - CDI daily rate   (SGS 12)  -> data/rates_cdi.csv
  - SELIC daily rate (SGS 11)  -> merged as fallback column
  - IBOV index level (SGS 7)   -> data/ibov_daily.csv  (BCB stopped SGS-7 in
                                  Sep-2019; extended with Yahoo ^BVSP)

Incremental by default: refetches only from a few days before the newest row
already on disk and merges. `--full` rebuilds from 2000 (the SGS API caps
daily-frequency requests at 10 years, so full pulls are chunked).

Fail-soft: if BCB or Yahoo is down and the files already exist, we warn and
keep the existing files so the rest of the nightly pipeline still runs on the
data we have (the 2026-08-11 run died on a BCB 502 and lost the night's B3
ingest). Without existing files the error propagates.

Output columns
--------------
rates_cdi.csv : date, cdi_daily_pct, selic_daily_pct, r_annual, r_cc
                r_annual = (1 + cdi_daily_pct/100)**252 - 1
                r_cc     = 252*ln(1 + cdi_daily_pct/100)   (continuous)
ibov_daily.csv: date, ibov_close
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT_DIR = BASE / "data"
OUT_DIR.mkdir(exist_ok=True)

RATES_PATH = OUT_DIR / "rates_cdi.csv"
IBOV_PATH = OUT_DIR / "ibov_daily.csv"

API = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
       "?formato=json&dataInicial={d0}&dataFinal={d1}")
YAHOO = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"
         "?period1={p1}&period2={p2}&interval=1d")

FULL_START = pd.Timestamp("2000-01-01")
OVERLAP_DAYS = 10            # refetch window behind the newest cached row
CHUNK_YEARS = 9              # SGS caps daily series at 10y per request


def _get_json(url, label, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:            # SGS: no observations in the range
                return []
            if attempt == tries - 1:
                raise
            print(f"   retry {attempt + 1} for {label}: HTTP {e.code}")
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"   retry {attempt + 1} for {label}: {e}")
        time.sleep(5)


def fetch_series(code, start, end, label=""):
    """SGS series between two dates, chunked to respect the 10-year cap."""
    frames = []
    d0 = start
    while d0 <= end:
        d1 = min(pd.Timestamp(year=d0.year + CHUNK_YEARS, month=12, day=31), end)
        url = API.format(code=code, d0=d0.strftime("%d/%m/%Y"),
                         d1=d1.strftime("%d/%m/%Y"))
        data = _get_json(url, f"{label} {d0.date()}..{d1.date()}")
        df = pd.DataFrame(data)
        print(f"   {label} SGS-{code} {d0.date()}..{d1.date()}: {len(df):,} obs")
        if len(df):
            frames.append(df)
        d0 = d1 + pd.Timedelta(days=1)
        time.sleep(1)
    if not frames:
        return pd.DataFrame(columns=["date", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["data"], format="%d/%m/%Y")
    out["value"] = pd.to_numeric(out["valor"], errors="coerce")
    out = out[["date", "value"]].dropna().drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def fetch_yahoo_ibov(start, end):
    url = YAHOO.format(p1=int(start.timestamp()),
                       p2=int((end + pd.Timedelta(days=1)).timestamp()))
    ydata = _get_json(url, "Yahoo ^BVSP")
    yres = ydata["chart"]["result"][0]
    yh = pd.DataFrame({
        "date": pd.to_datetime(yres["timestamp"], unit="s").normalize(),
        "ibov_close": yres["indicators"]["quote"][0]["close"],
    }).dropna().drop_duplicates("date")
    return yh


def _merge(old, new):
    """Union on date. Existing values win, holes are filled from the fresh pull
    (CDI for day D is published on D+1, so D's row first lands SELIC-only).
    BCB does not revise prints, and rewriting them only churns the nightly
    commit (float last-digit drift between numpy versions was a 38-line diff)."""
    if old is None or old.empty:
        return new.sort_values("date").reset_index(drop=True)
    merged = (old.set_index("date")
              .combine_first(new.set_index("date"))
              .sort_index().reset_index())
    return merged


def _read_existing(path, cols):
    if not path.exists():
        return None
    # round_trip: the default parser can land on the neighbouring double, which
    # then re-serialises with a different last digit and rewrites every row
    df = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    return df[[c for c in cols if c in df.columns]]


def refresh_rates(start, end, full):
    old = None if full else _read_existing(
        RATES_PATH, ["date", "cdi_daily_pct", "selic_daily_pct", "r_annual", "r_cc"])

    print("\n[1/3] CDI daily rate ...")
    cdi = fetch_series(12, start, end, "CDI").rename(columns={"value": "cdi_daily_pct"})
    print("\n[2/3] SELIC daily rate ...")
    selic = fetch_series(11, start, end, "SELIC").rename(columns={"value": "selic_daily_pct"})

    new = cdi.merge(selic, on="date", how="outer")
    rates = _merge(old, new)
    for c in ("r_annual", "r_cc"):
        if c not in rates.columns:
            rates[c] = np.nan
    # derived columns only where the inputs are new: rows we had no CDI for
    # (SELIC-only placeholder, or brand new). Everything else is left as-is.
    had_cdi = (set(old.loc[old["cdi_daily_pct"].notna(), "date"])
               if old is not None else set())
    todo = rates["r_annual"].isna() | ~rates["date"].isin(had_cdi)
    daily = (rates.loc[todo, "cdi_daily_pct"]
             .fillna(rates.loc[todo, "selic_daily_pct"]) / 100.0)
    rates.loc[todo, "r_annual"] = (1.0 + daily) ** 252 - 1.0
    rates.loc[todo, "r_cc"] = 252.0 * np.log1p(daily)
    rates = rates[["date", "cdi_daily_pct", "selic_daily_pct", "r_annual", "r_cc"]]
    rates.to_csv(RATES_PATH, index=False)
    print(f"\n  Saved {RATES_PATH}  ({len(rates):,} rows, "
          f"{rates['date'].min().date()} - {rates['date'].max().date()})")


def refresh_ibov(start, end, full):
    old = None if full else _read_existing(IBOV_PATH, ["date", "ibov_close"])

    frames = []
    if full:
        # SGS-7 covers 2000-2019 and is dead after; only worth pulling on a rebuild
        print("\n[3/3] IBOV index level (BCB SGS-7, historical) ...")
        frames.append(fetch_series(7, start, end, "IBOV")
                      .rename(columns={"value": "ibov_close"}))
    print("\n[4/4] IBOV from Yahoo ^BVSP ...")
    frames.append(fetch_yahoo_ibov(start, end))

    new = pd.concat(frames, ignore_index=True).dropna()
    # BCB print wins where both exist (Yahoo history is rounded) — BCB is first
    new = new.drop_duplicates("date", keep="first")
    ibov = _merge(old, new)
    ibov.to_csv(IBOV_PATH, index=False)
    print(f"  Saved {IBOV_PATH}  ({len(ibov):,} rows, "
          f"{ibov['date'].min().date()} - {ibov['date'].max().date()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="rebuild from 2000 instead of extending the cached files")
    args = ap.parse_args()

    end = pd.Timestamp.today().normalize()
    if args.full or not RATES_PATH.exists():
        start_rates, full_rates = FULL_START, True
    else:
        last = pd.read_csv(RATES_PATH, usecols=["date"], parse_dates=["date"])["date"].max()
        start_rates, full_rates = last - pd.Timedelta(days=OVERLAP_DAYS), False
    if args.full or not IBOV_PATH.exists():
        start_ibov, full_ibov = FULL_START, True
    else:
        last = pd.read_csv(IBOV_PATH, usecols=["date"], parse_dates=["date"])["date"].max()
        start_ibov, full_ibov = last - pd.Timedelta(days=OVERLAP_DAYS), False

    print("=" * 60)
    print(f" BCB SGS / Yahoo refresh — rates from {start_rates.date()}"
          f"{' (full)' if full_rates else ''}, IBOV from {start_ibov.date()}"
          f"{' (full)' if full_ibov else ''}")
    print("=" * 60)

    failures = []
    for name, fn, args_ in (("rates", refresh_rates, (start_rates, end, full_rates)),
                            ("ibov", refresh_ibov, (start_ibov, end, full_ibov))):
        try:
            fn(*args_)
        except Exception as e:
            path = RATES_PATH if name == "rates" else IBOV_PATH
            if not path.exists():
                raise
            print(f"::warning::{name} refresh failed ({e}); keeping existing "
                  f"{path.name}")
            failures.append(name)
    if failures:
        print(f"\nfinished with stale: {', '.join(failures)} (fail-soft by design)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
