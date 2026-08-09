"""
download_bcb_series.py
======================
Downloads daily series from the Banco Central do Brasil SGS API:

  - CDI daily rate   (SGS 12)  -> data/rates_cdi.csv
  - SELIC daily rate (SGS 11)  -> merged as fallback column
  - IBOV index level (SGS 7)   -> data/ibov_daily.csv

The SGS API caps daily-frequency requests at 10 years, so we chunk.

Output columns
--------------
rates_cdi.csv : date, cdi_daily_pct, selic_daily_pct, r_annual
                r_annual = (1 + cdi_daily_pct/100)**252 - 1   (continuous-equivalent
                also provided as r_cc = 252*ln(1+cdi_daily_pct/100))
ibov_daily.csv: date, ibov_close
"""

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT_DIR = BASE / "data"
OUT_DIR.mkdir(exist_ok=True)

API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={d0}&dataFinal={d1}"


def fetch_series(code, start_year=2000, end_year=2026, label=""):
    """Fetch an SGS series in <=9-year chunks (API caps daily series at 10y)."""
    frames = []
    for y0 in range(start_year, end_year + 1, 9):
        y1 = min(y0 + 8, end_year)
        url = API.format(code=code, d0=f"01/01/{y0}", d1=f"31/12/{y1}")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"   retry {attempt+1} for {label} {y0}-{y1}: {e}")
                time.sleep(5)
        df = pd.DataFrame(data)
        if len(df):
            frames.append(df)
        print(f"   {label} SGS-{code} {y0}-{y1}: {len(df):,} obs")
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["data"], format="%d/%m/%Y")
    out["value"] = pd.to_numeric(out["valor"], errors="coerce")
    out = out[["date", "value"]].dropna().drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def main():
    print("=" * 60)
    print(" BCB SGS download: CDI (12), SELIC (11), IBOV (7)")
    print("=" * 60)

    print("\n[1/3] CDI daily rate ...")
    cdi = fetch_series(12, label="CDI").rename(columns={"value": "cdi_daily_pct"})

    print("\n[2/3] SELIC daily rate ...")
    selic = fetch_series(11, label="SELIC").rename(columns={"value": "selic_daily_pct"})

    print("\n[3/3] IBOV index level ...")
    ibov = fetch_series(7, label="IBOV").rename(columns={"value": "ibov_close"})

    rates = cdi.merge(selic, on="date", how="outer").sort_values("date")
    # effective daily rate: CDI, falling back to SELIC where CDI missing
    daily = rates["cdi_daily_pct"].fillna(rates["selic_daily_pct"]) / 100.0
    rates["r_annual"] = (1.0 + daily) ** 252 - 1.0          # effective annual
    rates["r_cc"] = 252.0 * np.log1p(daily)                  # continuously compounded

    rates_path = OUT_DIR / "rates_cdi.csv"
    rates.to_csv(rates_path, index=False)
    print(f"\n  Saved {rates_path}  ({len(rates):,} rows, "
          f"{rates['date'].min().date()} - {rates['date'].max().date()})")
    print(f"  r_annual range: {rates['r_annual'].min():.2%} - {rates['r_annual'].max():.2%}")

    # BCB discontinued SGS-7 in Sep-2019 — extend with Yahoo ^BVSP
    print("\n[4/4] Extending IBOV with Yahoo ^BVSP ...")
    yurl = ("https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"
            "?period1=946684800&period2=2145916800&interval=1d")
    req = urllib.request.Request(yurl, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        ydata = json.loads(resp.read())
    yres = ydata["chart"]["result"][0]
    yh = pd.DataFrame({
        "date": pd.to_datetime(yres["timestamp"], unit="s").normalize(),
        "ibov_yahoo": yres["indicators"]["quote"][0]["close"],
    }).dropna().drop_duplicates("date")
    ibov = ibov.merge(yh, on="date", how="outer").sort_values("date")
    ibov["ibov_close"] = ibov["ibov_close"].fillna(ibov["ibov_yahoo"])
    ibov = ibov[["date", "ibov_close"]].dropna().reset_index(drop=True)

    ibov_path = OUT_DIR / "ibov_daily.csv"
    ibov.to_csv(ibov_path, index=False)
    print(f"  Saved {ibov_path}  ({len(ibov):,} rows, "
          f"{ibov['date'].min().date()} - {ibov['date'].max().date()})")


if __name__ == "__main__":
    main()
