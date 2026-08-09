"""
vxbr_daily_rates.py
===================
VXBR replication v2 — same S&P/B3 Ibovespa VIX methodology as backtest_vxbr.py
with two fixes:

  1. Per-date risk-free rate from the daily CDI curve (data/rates_cdi.csv)
     instead of a constant 12%.
  2. A second index variant computed from bid/ask midpoints (vxbr_mid) where
     both sides are quoted, alongside the close-price variant (vxbr) for
     continuity with v1. The official VIX methodology uses mid-quotes.

Output: results/vxbr_replication_v2.csv
  date, vxbr, vxbr_mid, sigma1, sigma2, T1, T2, F1, F2, K0_1, K0_2,
  n_strikes1, n_strikes2, r_used, ibov_spot_est, bracketed
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2000, 2027))
MIN_BIZ_DAYS_TO_EXPIRY = 6
T30 = 30 / 365
MIN_STRIKES_PER_EXPIRY = 3

NEEDED_COLS = ["refdate", "bdi_code", "specification_code",
               "strike_price", "close", "best_bid", "best_ask",
               "maturity_date", "volume", "traded_contracts", "symbol"]


# ── Rates ─────────────────────────────────────────────────────────────────────
def load_rate_lookup():
    """Return a function date -> continuously-compounded annual CDI rate."""
    rates = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    rates = rates.dropna(subset=["r_cc"]).sort_values("date")
    s = rates.set_index("date")["r_cc"]
    # forward-fill onto a full calendar so weekend/holiday lookups resolve
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()

    def lookup(d):
        d = pd.Timestamp(d)
        if d in full.index:
            return float(full.loc[d])
        return float(full.iloc[-1]) if d > full.index[-1] else float(full.iloc[0])

    return lookup


# ── VIX machinery (per-day r) ────────────────────────────────────────────────
def estimate_forward(calls, puts, T, r, price_col):
    pairs = pd.merge(
        calls[["strike_price", price_col]].rename(columns={price_col: "C"}),
        puts[["strike_price", price_col]].rename(columns={price_col: "P"}),
        on="strike_price",
    ).dropna()
    if len(pairs) == 0:
        return np.nan
    pairs["F"] = pairs["strike_price"] + np.exp(r * T) * (pairs["C"] - pairs["P"])
    best = pairs.loc[(pairs["C"] - pairs["P"]).abs().idxmin()]
    return float(best["F"])


def vix_variance(strip_df, F, K0, T, r):
    df = strip_df.dropna(subset=["Q"]).sort_values("strike_price")
    if len(df) < 2:
        return np.nan
    K = df["strike_price"].values.astype(float)
    Q = df["Q"].values.astype(float)
    n = len(K)
    dK = np.empty(n)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    if n > 2:
        dK[1:-1] = (K[2:] - K[:-2]) / 2
    contrib = (dK / K**2) * np.exp(r * T) * Q
    sigma2 = (2 / T) * contrib.sum() - (1 / T) * (F / K0 - 1) ** 2
    return max(sigma2, 0.0)


def build_otm_strip(exp_df, K0, price_col):
    calls = exp_df[exp_df["bdi_code"] == 74].set_index("strike_price")[price_col]
    puts = exp_df[exp_df["bdi_code"] == 75].set_index("strike_price")[price_col]

    all_strikes = sorted(set(calls.index) | set(puts.index))
    rows = []
    for K in all_strikes:
        if K < K0:
            q = puts.get(K, np.nan)
        elif K > K0:
            q = calls.get(K, np.nan)
        else:
            c, p = calls.get(K, np.nan), puts.get(K, np.nan)
            q = np.nanmean([c, p]) if not (np.isnan(c) and np.isnan(p)) else np.nan
        if pd.notna(q) and q > 0:
            rows.append({"strike_price": K, "Q": q})

    if not rows:
        return pd.DataFrame(columns=["strike_price", "Q"])

    df = pd.DataFrame(rows).sort_values("strike_price").reset_index(drop=True)
    k0_idx = df.index[df["strike_price"] == K0].tolist()
    if not k0_idx:
        k0_idx = [(df["strike_price"] - K0).abs().idxmin()]
    k0_idx = k0_idx[0]

    keep = [k0_idx]
    for i in range(k0_idx - 1, -1, -1):
        if df.loc[i, "Q"] > 0:
            keep.append(i)
        else:
            break
    for i in range(k0_idx + 1, len(df)):
        if df.loc[i, "Q"] > 0:
            keep.append(i)
        else:
            break
    return df.loc[sorted(set(keep))].reset_index(drop=True)


def sigma30_from_strip(day_data, r, price_col):
    """Run the two-expiry VIX interpolation using the given price column.
    Returns (vxbr, diag_dict) or (nan, None)."""
    expiries = sorted(day_data["maturity_date"].unique())
    if len(expiries) < 2:
        return np.nan, None

    for i in range(len(expiries) - 1):
        e1, e2 = expiries[i], expiries[i + 1]
        d1 = day_data[day_data["maturity_date"] == e1]
        d2 = day_data[day_data["maturity_date"] == e2]
        T1, T2 = d1["T"].iloc[0], d2["T"].iloc[0]

        F1 = estimate_forward(d1[d1["bdi_code"] == 74], d1[d1["bdi_code"] == 75], T1, r, price_col)
        F2 = estimate_forward(d2[d2["bdi_code"] == 74], d2[d2["bdi_code"] == 75], T2, r, price_col)
        if np.isnan(F1) or np.isnan(F2) or F1 <= 0 or F2 <= 0:
            continue

        K0_1 = max((k for k in sorted(d1["strike_price"].unique()) if k <= F1), default=None)
        K0_2 = max((k for k in sorted(d2["strike_price"].unique()) if k <= F2), default=None)
        if K0_1 is None or K0_2 is None:
            continue

        strip1 = build_otm_strip(d1, K0_1, price_col)
        strip2 = build_otm_strip(d2, K0_2, price_col)
        if len(strip1) < MIN_STRIKES_PER_EXPIRY or len(strip2) < MIN_STRIKES_PER_EXPIRY:
            continue

        s1 = vix_variance(strip1, F1, K0_1, T1, r)
        s2 = vix_variance(strip2, F2, K0_2, T2, r)
        if np.isnan(s1) or np.isnan(s2) or abs(T2 - T1) < 1e-8:
            continue

        w1 = (T2 - T30) / (T2 - T1)
        w2 = (T30 - T1) / (T2 - T1)
        vxbr2 = (s1 * T1 * w1 + s2 * T2 * w2) * (365.0 / 30.0)
        vxbr = 100.0 * np.sqrt(max(vxbr2, 0.0))

        diag = {
            "sigma1": 100 * np.sqrt(s1), "sigma2": 100 * np.sqrt(s2),
            "T1": T1, "T2": T2, "F1": F1, "F2": F2,
            "K0_1": K0_1, "K0_2": K0_2,
            "n_strikes1": len(strip1), "n_strikes2": len(strip2),
            "bracketed": T1 <= T30 <= T2,
        }
        return vxbr, diag

    return np.nan, None


def main():
    print("=" * 65)
    print(" VXBR v2: daily CDI rates + mid-quote variant")
    print("=" * 65)

    rate_of = load_rate_lookup()
    records = []

    for year in YEARS:
        path = DATA_DIR / f"year={year}" / "part-0.parquet"
        if not path.exists():
            print(f"   {year}: file not found, skipping")
            continue

        df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()
        ibov = df[
            (df["specification_code"].str.strip().str.startswith("IBO"))
            & (df["bdi_code"].isin([74, 75]))
            & (df["strike_price"] > 0)
        ].copy()
        del df
        if len(ibov) == 0:
            print(f"   {year}: no IBOV options")
            continue

        ibov["refdate"] = pd.to_datetime(ibov["refdate"])
        ibov["maturity_date"] = pd.to_datetime(ibov["maturity_date"])
        # mid where both sides quoted, else NaN
        bid = pd.to_numeric(ibov["best_bid"], errors="coerce")
        ask = pd.to_numeric(ibov["best_ask"], errors="coerce")
        ibov["mid"] = np.where((bid > 0) & (ask > 0) & (ask >= bid), (bid + ask) / 2, np.nan)

        year_hits = 0
        for day, day_data in ibov.groupby("refdate"):
            r = rate_of(day)
            dd = day_data.copy()
            dd["T"] = (dd["maturity_date"] - day).dt.days / 365.0
            dd = dd[dd["T"] > MIN_BIZ_DAYS_TO_EXPIRY / 252]

            close_dd = dd[dd["close"] > 0]
            vx_close, diag = sigma30_from_strip(close_dd, r, "close")
            if diag is None:
                continue
            mid_dd = dd[dd["mid"] > 0]
            vx_mid, _ = sigma30_from_strip(mid_dd, r, "mid") if len(mid_dd) else (np.nan, None)

            rec = {"date": pd.Timestamp(day), "vxbr": vx_close, "vxbr_mid": vx_mid,
                   "r_used": r, "ibov_spot_est": diag["F1"] * np.exp(-r * diag["T1"])}
            rec.update(diag)
            records.append(rec)
            year_hits += 1

        print(f"   {year}: {len(ibov):6,} records -> {year_hits} days")
        del ibov

    results = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    n0 = len(results)
    results = results[(results["vxbr"] >= 5) & (results["vxbr"] <= 200)]
    if n0 - len(results):
        print(f"   dropped {n0 - len(results)} implausible days (vxbr outside 5-200)")

    out = OUT_DIR / "vxbr_replication_v2.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(results):,} days, "
          f"{results['date'].min().date()} - {results['date'].max().date()})")
    print(f"  vxbr (close) mean {results['vxbr'].mean():.2f}  "
          f"median {results['vxbr'].median():.2f}")
    print(f"  vxbr_mid coverage: {results['vxbr_mid'].notna().mean():.1%} of days, "
          f"mean {results['vxbr_mid'].mean():.2f}")
    print(f"  r_used range: {results['r_used'].min():.2%} - {results['r_used'].max():.2%}")


if __name__ == "__main__":
    main()
