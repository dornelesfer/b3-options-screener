"""
backtest_vxbr.py
================
Replicates the S&P/B3 Ibovespa VIX (VXBR) methodology from B3 COTAHIST data
and produces a daily implied-volatility time series for backtesting.

Methodology (follows Cboe VIX white-paper, adapted for IBOV):
  1. Use only IBOV index options  (bdi_code ∈ {74,75}, spec_code starts with 'IBO')
     NOTE: B3 changed spec_code from 'IBO' to 'IBO/' in March 2025 — filter uses startswith.
  2. For each day, identify the two nearest expiries with > 6 biz-days to go
     (near-term T1, next-term T2) — rolling to T2/T3 when T1 < 6 biz-days
  3. Estimate forward F via put-call parity per expiry
  4. Drop options with zero close price (proxy for zero bid in EOD data)
  5. Build OTM strip: puts for K < F, calls for K > F, average at K0
  6. Compute model-free variance σ²(T) = (2/T)Σ(ΔK/K²)e^(rT)Q(K) − (1/T)(F/K0−1)²
  7. Interpolate to 30-day constant maturity:
       VXBR² = [σ₁²·T1·(T2−T30)/(T2−T1) + σ₂²·T2·(T30−T1)/(T2−T1)] · (365/30)
       VXBR  = 100 · √VXBR²

Outputs (saved to results/):
  - vxbr_replication.csv        daily VXBR replication values
  - vxbr_backtest_plot.png      time-series + term-structure diagnostics
  - vxbr_surface_heatmap.png    IV surface heatmap (k vs T vs date)
  - vxbr_backtest_report.txt    text summary statistics
"""

import os, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR   = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# Backtest window: 2023 (pre-VXBR) through available 2025 data
YEARS      = list(range(2000, 2027))  # full history through 2026
R_BRAZIL   = 0.12          # annualised risk-free rate (conservative; ideally use daily SELIC)
MIN_BIZ_DAYS_TO_EXPIRY = 6 # VIX rule: skip near-term when < 6 business days remain
T30        = 30 / 365      # target 30-day maturity (years)
MIN_STRIKES_PER_EXPIRY = 3 # minimum OTM strikes to include an expiry

# ── Helpers ───────────────────────────────────────────────────────────────────

def calendar_to_biz(n_calendar_days, approx_biz_per_year=252):
    """Approximate calendar → business day conversion."""
    return n_calendar_days * approx_biz_per_year / 365

def estimate_forward(calls, puts, T, r):
    """Put-call parity forward: F = K + e^(rT)*(C - P), pick min |C-P|."""
    pairs = pd.merge(
        calls[["strike_price","close"]].rename(columns={"close":"C"}),
        puts[["strike_price","close"]].rename(columns={"close":"P"}),
        on="strike_price"
    )
    if len(pairs) == 0:
        return np.nan
    pairs["F"] = pairs["strike_price"] + np.exp(r * T) * (pairs["C"] - pairs["P"])
    best = pairs.loc[(pairs["C"] - pairs["P"]).abs().idxmin()]
    return float(best["F"])

def vix_variance(strip_df, F, K0, T, r):
    """
    VIX model-free variance formula.
    strip_df must have columns: strike_price, Q (OTM option price).
    Returns σ² ≥ 0, or np.nan if insufficient data.
    """
    df = strip_df.dropna(subset=["Q"]).sort_values("strike_price")
    if len(df) < 2:
        return np.nan
    K  = df["strike_price"].values.astype(float)
    Q  = df["Q"].values.astype(float)
    n  = len(K)
    dK = np.empty(n)
    dK[0]    = K[1] - K[0]
    dK[-1]   = K[-1] - K[-2]
    if n > 2:
        dK[1:-1] = (K[2:] - K[:-2]) / 2
    contrib  = (dK / K**2) * np.exp(r * T) * Q
    sigma2   = (2 / T) * contrib.sum() - (1 / T) * (F / K0 - 1) ** 2
    return max(sigma2, 0.0)

def build_otm_strip(exp_df, F, K0):
    """
    Build OTM option strip following VIX rules:
      - OTM puts  (K < K0): use put prices
      - OTM calls (K > K0): use call prices
      - At K0: average of call and put prices (if both exist)
      - Exclude any option with close == 0 (proxy for zero bid)
    """
    calls = exp_df[exp_df["bdi_code"] == 74].set_index("strike_price")["close"]
    puts  = exp_df[exp_df["bdi_code"] == 75].set_index("strike_price")["close"]

    all_strikes = sorted(set(calls.index) | set(puts.index))
    rows = []
    for K in all_strikes:
        if K < K0:
            q = puts.get(K, np.nan)
        elif K > K0:
            q = calls.get(K, np.nan)
        else:   # K == K0
            c = calls.get(K, np.nan)
            p = puts.get(K, np.nan)
            q = np.nanmean([c, p]) if not (np.isnan(c) and np.isnan(p)) else np.nan
        if pd.notna(q) and q > 0:
            rows.append({"strike_price": K, "Q": q})

    if len(rows) == 0:
        return pd.DataFrame(columns=["strike_price","Q"])

    # Exclude consecutive zero-bid boundary rule (VIX: stop at first K with Q=0 going outward)
    # Here Q>0 is already enforced; enforce contiguity from K0 outward
    df = pd.DataFrame(rows).sort_values("strike_price").reset_index(drop=True)
    k0_idx = df.index[df["strike_price"] == K0].tolist()
    if not k0_idx:
        # find nearest
        k0_idx = [(df["strike_price"] - K0).abs().idxmin()]
    k0_idx = k0_idx[0]

    # Walk outward left from K0
    left_ok = [k0_idx]
    for i in range(k0_idx - 1, -1, -1):
        if df.loc[i, "Q"] > 0:
            left_ok.append(i)
        else:
            break
    # Walk outward right from K0
    right_ok = []
    for i in range(k0_idx + 1, len(df)):
        if df.loc[i, "Q"] > 0:
            right_ok.append(i)
        else:
            break

    keep = sorted(set(left_ok + right_ok))
    return df.loc[keep].reset_index(drop=True)

def compute_day_vxbr(day_data, r=R_BRAZIL):
    """
    Compute VXBR for a single trading day.
    Returns dict with keys: vxbr, sigma1, sigma2, T1, T2, F1, F2, n_strikes1, n_strikes2
    """
    day_data = day_data.copy()
    day_data["T"] = (pd.to_datetime(day_data["maturity_date"]) -
                     pd.to_datetime(day_data["refdate"].iloc[0])).dt.days / 365.0

    # Filter: positive close, positive T, > 6 business-day equivalent
    day_data = day_data[
        (day_data["close"] > 0) &
        (day_data["T"] > MIN_BIZ_DAYS_TO_EXPIRY / 252)
    ].copy()

    expiries = sorted(day_data["maturity_date"].unique())
    if len(expiries) < 2:
        return None

    # Try expiry pairs until we find one with enough strikes on both legs
    result = None
    for i in range(len(expiries) - 1):
        e1, e2 = expiries[i], expiries[i+1]
        d1 = day_data[day_data["maturity_date"] == e1]
        d2 = day_data[day_data["maturity_date"] == e2]
        T1 = d1["T"].iloc[0]
        T2 = d2["T"].iloc[0]

        # Near-term must bracket 30 days from below, next-term from above
        # (standard VIX requirement: T1 < T30 < T2)
        # If T1 > T30, both expiries are beyond 30d — still compute but flag
        calls1, puts1 = d1[d1["bdi_code"]==74], d1[d1["bdi_code"]==75]
        calls2, puts2 = d2[d2["bdi_code"]==74], d2[d2["bdi_code"]==75]

        F1 = estimate_forward(calls1, puts1, T1, r)
        F2 = estimate_forward(calls2, puts2, T2, r)
        if np.isnan(F1) or np.isnan(F2) or F1 <= 0 or F2 <= 0:
            continue

        K0_1 = max((k for k in sorted(d1["strike_price"].unique()) if k <= F1), default=None)
        K0_2 = max((k for k in sorted(d2["strike_price"].unique()) if k <= F2), default=None)
        if K0_1 is None or K0_2 is None:
            continue

        strip1 = build_otm_strip(d1, F1, K0_1)
        strip2 = build_otm_strip(d2, F2, K0_2)
        if len(strip1) < MIN_STRIKES_PER_EXPIRY or len(strip2) < MIN_STRIKES_PER_EXPIRY:
            continue

        s1 = vix_variance(strip1, F1, K0_1, T1, r)
        s2 = vix_variance(strip2, F2, K0_2, T2, r)
        if np.isnan(s1) or np.isnan(s2):
            continue

        # 30-day interpolation (VIX formula, annualised)
        # σ²_30 = [σ₁²·T1·(T2-T30)/(T2-T1) + σ₂²·T2·(T30-T1)/(T2-T1)] × (365/30)
        if abs(T2 - T1) < 1e-8:
            continue
        if T1 <= T30 <= T2:
            w1 = (T2 - T30) / (T2 - T1)
            w2 = (T30 - T1) / (T2 - T1)
        else:
            # Both legs beyond 30d or both below — interpolate linearly anyway
            w1 = (T2 - T30) / (T2 - T1)
            w2 = (T30 - T1) / (T2 - T1)

        vxbr2  = (s1 * T1 * w1 + s2 * T2 * w2) * (365.0 / 30.0)
        vxbr   = 100.0 * np.sqrt(max(vxbr2, 0.0))

        result = {
            "vxbr"     : vxbr,
            "sigma1"   : 100 * np.sqrt(s1),
            "sigma2"   : 100 * np.sqrt(s2),
            "T1"       : T1,
            "T2"       : T2,
            "F1"       : F1,
            "F2"       : F2,
            "K0_1"     : K0_1,
            "K0_2"     : K0_2,
            "n_strikes1": len(strip1),
            "n_strikes2": len(strip2),
            "expiry1"  : e1,
            "expiry2"  : e2,
            "bracketed": T1 <= T30 <= T2,
        }
        break   # found a valid pair

    return result

# ── Process year-by-year to keep memory footprint small ──────────────────────
print("=" * 65)
print(" VXBR Backtest: S&P/B3 Ibovespa VIX Replication")
print(f" Window: {YEARS[0]}–{YEARS[-1]}  |  Risk-free rate: {R_BRAZIL*100:.0f}% p.a.")
print("=" * 65)
print()

# Only load the columns needed — full parquet files are >1 GB each for recent years
NEEDED_COLS = ["refdate", "bdi_code", "specification_code",
               "strike_price", "close", "best_bid", "best_ask",
               "maturity_date", "volume", "traded_contracts", "symbol"]

records = []
total_days_attempted = 0
total_raw = 0

for year in YEARS:
    path = DATA_DIR / f"year={year}" / "part-0.parquet"
    if not path.exists():
        print(f"   ⚠  {year}: file not found, skipping")
        continue

    # Read only needed columns — keeps per-year memory ~10x smaller
    df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()

    # B3 changed spec_code from 'IBO' (pre-Mar 2025) to 'IBO/' (Mar 2025 onwards)
    ibov = df[
        (df["specification_code"].str.strip().str.startswith("IBO")) &
        (df["bdi_code"].isin([74, 75])) &
        (df["strike_price"] > 0) &
        (df["close"] > 0)
    ].copy()
    del df  # free full frame immediately

    if len(ibov) == 0:
        print(f"   {year}: — no IBOV options")
        continue

    ibov["refdate"]       = pd.to_datetime(ibov["refdate"])
    ibov["maturity_date"] = pd.to_datetime(ibov["maturity_date"])
    total_raw += len(ibov)

    trading_days_yr = sorted(ibov["refdate"].unique())
    total_days_attempted += len(trading_days_yr)
    year_hits = 0

    for day in trading_days_yr:
        day_data = ibov[ibov["refdate"] == day].copy()
        res = compute_day_vxbr(day_data)
        if res is None:
            continue
        res["date"] = pd.Timestamp(day)
        res["ibov_spot_est"] = res["F1"] * np.exp(-R_BRAZIL * res["T1"])
        records.append(res)
        year_hits += 1

    print(f"   {year}: {len(ibov):6,} records → {year_hits}/{len(trading_days_yr)} days computed")
    del ibov

print(f"\n   Total raw records : {total_raw:,}")
print(f"   ✅ VXBR computed for {len(records)} of {total_days_attempted} trading days")

# ── Build results dataframe ───────────────────────────────────────────────────
results = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

# Mark VXBR inception date (March 19, 2024)
vxbr_inception = pd.Timestamp("2024-03-19")
results["post_inception"] = results["date"] >= vxbr_inception

# Sanity filter: drop obvious outliers (VXBR > 200 or < 5 usually indicate data issues)
results_clean = results[(results["vxbr"] >= 5) & (results["vxbr"] <= 200)].copy()
n_dropped = len(results) - len(results_clean)
if n_dropped > 0:
    print(f"   ⚠  Dropped {n_dropped} days with implausible VXBR values (outside 5–200)")

results = results_clean.copy()

# Save CSV
csv_path = OUT_DIR / "vxbr_replication.csv"
results.to_csv(csv_path, index=False)
print(f"\n💾 Saved: {csv_path}")

# ── Summary statistics ────────────────────────────────────────────────────────
def summary_stats(s, label):
    print(f"\n  {label}")
    print(f"    N days   : {len(s)}")
    print(f"    Mean     : {s.mean():.2f}")
    print(f"    Median   : {s.median():.2f}")
    print(f"    Std      : {s.std():.2f}")
    print(f"    Min      : {s.min():.2f}  ({results.loc[s.idxmin(),'date'].date()})")
    print(f"    Max      : {s.max():.2f}  ({results.loc[s.idxmax(),'date'].date()})")
    print(f"    >30 (high vol) : {(s>30).mean()*100:.1f}% of days")
    return s.describe()

print("\n" + "=" * 65)
print(" BACKTEST STATISTICS")
print("=" * 65)
all_stats   = summary_stats(results["vxbr"], "Full window (2023–2025)")
pre_stats   = summary_stats(results.loc[~results["post_inception"], "vxbr"],
                             f"Pre-inception  (before {vxbr_inception.date()})")
post_stats  = summary_stats(results.loc[results["post_inception"], "vxbr"],
                             f"Post-inception ({vxbr_inception.date()} onward)")

# Save text report
report_path = OUT_DIR / "vxbr_backtest_report.txt"
with open(report_path, "w") as f:
    f.write("VXBR Replication Backtest Report\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Methodology : S&P/B3 Ibovespa VIX (Cboe VIX methodology)\n")
    f.write(f"Data source : B3 COTAHIST yearly files (IBOV index options only)\n")
    f.write(f"Window      : {results['date'].min().date()} — {results['date'].max().date()}\n")
    f.write(f"Risk-free r : {R_BRAZIL*100:.0f}% p.a. (constant; ideally use daily SELIC)\n")
    f.write(f"VXBR launch : {vxbr_inception.date()}\n\n")
    f.write("Full-window statistics:\n")
    f.write(str(all_stats) + "\n\n")
    f.write("Pre-inception statistics:\n")
    f.write(str(pre_stats) + "\n\n")
    f.write("Post-inception statistics:\n")
    f.write(str(post_stats) + "\n\n")
    f.write("Key dates:\n")
    idx_max = results["vxbr"].idxmax()
    idx_min = results["vxbr"].idxmin()
    f.write(f"  Peak VXBR  : {results.loc[idx_max,'vxbr']:.2f} on {results.loc[idx_max,'date'].date()}\n")
    f.write(f"  Trough VXBR: {results.loc[idx_min,'vxbr']:.2f} on {results.loc[idx_min,'date'].date()}\n")
    f.write(f"\nNote: VXBR values are the 30-day constant-maturity implied volatility index\n")
    f.write(f"      expressed as a percentage (e.g., 20.0 = 20% annualised implied vol).\n")

print(f"\n💾 Saved report: {report_path}")

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n📈 Generating plots ...")

BRAZIL_BLUE  = "#009c3b"
BRAZIL_GOLD  = "#FFDF00"
BRAZIL_RED   = "#e63946"
LIGHT_GRAY   = "#f5f5f5"
MID_GRAY     = "#aaaaaa"
DARK         = "#1a1a2e"

# ── Plot 1: Main backtest dashboard ──────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor="white")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35,
                         top=0.92, bottom=0.07, left=0.08, right=0.97)

ax1 = fig.add_subplot(gs[0, :])   # full-width: VXBR time series
ax2 = fig.add_subplot(gs[1, 0])   # near-term σ1 vs next-term σ2
ax3 = fig.add_subplot(gs[1, 1])   # distribution histogram
ax4 = fig.add_subplot(gs[2, 0])   # IBOV spot (estimated)
ax5 = fig.add_subplot(gs[2, 1])   # term structure: n_strikes over time

# ── ax1: VXBR time series ─────────────────────────────────────────────────
pre  = results[~results["post_inception"]]
post = results[results["post_inception"]]

ax1.fill_between(results["date"], results["vxbr"], alpha=0.12, color=BRAZIL_BLUE)
ax1.plot(pre["date"],  pre["vxbr"],  color=MID_GRAY,   lw=1.4, label="Pre-inception (backfilled)")
ax1.plot(post["date"], post["vxbr"], color=BRAZIL_BLUE, lw=1.8, label="Post-inception (live period)")
ax1.axvline(vxbr_inception, color=BRAZIL_RED, lw=1.5, ls="--", alpha=0.8,
            label=f"VXBR launch ({vxbr_inception.date()})")
ax1.axhline(results["vxbr"].mean(), color=BRAZIL_GOLD, lw=1.2, ls=":",
            label=f"Mean = {results['vxbr'].mean():.1f}")
ax1.axhline(30, color=BRAZIL_RED, lw=0.8, ls=":", alpha=0.5, label="30 threshold")

# shade high-vol periods
ax1.fill_between(results["date"], 0, results["vxbr"],
                 where=results["vxbr"] > 30,
                 alpha=0.25, color=BRAZIL_RED, label="High-vol (>30)")

ax1.set_title("VXBR Replication — 30-Day Ibovespa Implied Volatility Index",
              fontsize=14, fontweight="bold", pad=10)
ax1.set_ylabel("VXBR (implied vol, %)", fontsize=11)
ax1.legend(fontsize=8.5, ncol=3, loc="upper left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.tick_params(axis="x", rotation=30, labelsize=8)
ax1.set_ylim(0, max(results["vxbr"].max() * 1.1, 50))
ax1.grid(True, alpha=0.3, ls="--")
ax1.set_facecolor(LIGHT_GRAY)

# ── ax2: near vs next-term IV ─────────────────────────────────────────────
ax2.scatter(results["date"], results["sigma1"], s=6, alpha=0.5,
            color=BRAZIL_BLUE, label="Near-term σ₁")
ax2.scatter(results["date"], results["sigma2"], s=6, alpha=0.5,
            color=BRAZIL_RED,  label="Next-term σ₂")
ax2.plot(results["date"], results["vxbr"], color="black", lw=1.0,
         alpha=0.6, label="VXBR (30d interp)")
ax2.set_title("Near-term vs Next-term Implied Vol", fontsize=11, fontweight="bold")
ax2.set_ylabel("Implied vol (%)", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, ls="--")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.tick_params(axis="x", rotation=30, labelsize=8)
ax2.set_facecolor(LIGHT_GRAY)

# ── ax3: histogram ────────────────────────────────────────────────────────
bins = np.linspace(results["vxbr"].min(), results["vxbr"].max(), 35)
ax3.hist(pre["vxbr"],  bins=bins, alpha=0.55, color=MID_GRAY,   label="Pre-inception")
ax3.hist(post["vxbr"], bins=bins, alpha=0.7,  color=BRAZIL_BLUE, label="Post-inception")
ax3.axvline(results["vxbr"].mean(),   color="black",      ls="--", lw=1.2, label=f"Mean={results['vxbr'].mean():.1f}")
ax3.axvline(results["vxbr"].median(), color=BRAZIL_GOLD,  ls=":",  lw=1.2, label=f"Median={results['vxbr'].median():.1f}")
ax3.set_title("VXBR Distribution", fontsize=11, fontweight="bold")
ax3.set_xlabel("VXBR (%)"); ax3.set_ylabel("Days")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, ls="--")
ax3.set_facecolor(LIGHT_GRAY)

# ── ax4: estimated IBOV spot ──────────────────────────────────────────────
ax4.plot(results["date"], results["ibov_spot_est"] / 1000,
         color=BRAZIL_BLUE, lw=1.3)
ax4.axvline(vxbr_inception, color=BRAZIL_RED, lw=1.2, ls="--", alpha=0.7)
ax4.set_title("Estimated IBOV Spot Level", fontsize=11, fontweight="bold")
ax4.set_ylabel("IBOV (× 1,000 pts)", fontsize=10)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax4.tick_params(axis="x", rotation=30, labelsize=8)
ax4.grid(True, alpha=0.3, ls="--")
ax4.set_facecolor(LIGHT_GRAY)

# ── ax5: option strip width (liquidity indicator) ─────────────────────────
ax5.plot(results["date"], results["n_strikes1"], lw=1.0, color=BRAZIL_BLUE,
         alpha=0.7, label="Near-term strikes")
ax5.plot(results["date"], results["n_strikes2"], lw=1.0, color=BRAZIL_RED,
         alpha=0.7, label="Next-term strikes")
ax5.axvline(vxbr_inception, color="black", lw=1.2, ls="--", alpha=0.5)
ax5.set_title("Option Strip Width (Liquidity)", fontsize=11, fontweight="bold")
ax5.set_ylabel("# OTM strikes in strip", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3, ls="--")
ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax5.tick_params(axis="x", rotation=30, labelsize=8)
ax5.set_facecolor(LIGHT_GRAY)

# title
fig.suptitle(
    "VXBR Replication Backtest  |  IBOV Index Options (COTAHIST)\n"
    "S&P/B3 Ibovespa VIX Methodology  |  30-Day Constant-Maturity",
    fontsize=13, fontweight="bold", y=0.97
)

plot1_path = OUT_DIR / "vxbr_backtest_plot.png"
fig.savefig(str(plot1_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"💾 Saved: {plot1_path}")

# ── Plot 2: Historical term-structure & vol-regime chart ─────────────────────
# (Raw option data is no longer in memory — build from results DataFrame only)
print("   Building term-structure diagnostics plot ...")
try:
    fig2, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    axes = axes.flatten()

    # Panel 1: VXBR full history heatmap by year × month
    results["year"]  = results["date"].dt.year
    results["month"] = results["date"].dt.month
    pivot_ym = results.groupby(["year","month"])["vxbr"].mean().unstack()
    cmap2 = LinearSegmentedColormap.from_list("vxbr2", ["#e8f5e9","#ffd600","#e53935","#6a0dad"])
    im2 = axes[0].imshow(pivot_ym.values, aspect="auto", cmap=cmap2, origin="lower",
                          vmin=5, vmax=min(60, results["vxbr"].quantile(0.99)))
    axes[0].set_yticks(range(len(pivot_ym.index)))
    axes[0].set_yticklabels(pivot_ym.index.astype(str), fontsize=8)
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    axes[0].set_title("Monthly Average VXBR — Year × Month Heatmap", fontsize=11, fontweight="bold")
    fig2.colorbar(im2, ax=axes[0], label="VXBR (%)")
    axes[0].set_facecolor(LIGHT_GRAY)

    # Panel 2: Near-term vs next-term T1/T2 over time
    axes[1].scatter(results["date"], results["T1"]*365, s=3, alpha=0.4,
                    color=BRAZIL_BLUE, label="Near-term T1 (days)")
    axes[1].scatter(results["date"], results["T2"]*365, s=3, alpha=0.4,
                    color=BRAZIL_RED,  label="Next-term T2 (days)")
    axes[1].axhline(30, color="black", lw=0.8, ls="--", alpha=0.5, label="30d")
    axes[1].set_title("Near/Next-term Days to Expiry", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Calendar days to expiry")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, ls="--")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].set_facecolor(LIGHT_GRAY)

    # Panel 3: σ1 – σ2 spread (term-structure slope)
    results["ts_slope"] = results["sigma1"] - results["sigma2"]
    axes[2].bar(results["date"], results["ts_slope"],
                color=np.where(results["ts_slope"] > 0, BRAZIL_RED, BRAZIL_BLUE),
                width=1.5, alpha=0.7)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Term-Structure Slope (σ₁ − σ₂)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Near − Next implied vol (pp)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].grid(True, alpha=0.3, ls="--")
    axes[2].set_facecolor(LIGHT_GRAY)

    # Panel 4: Forward price (IBOV spot estimate) and VXBR overlay
    ax4b = axes[3].twinx()
    axes[3].plot(results["date"], results["ibov_spot_est"]/1000,
                 color=BRAZIL_BLUE, lw=1.2, label="IBOV est. (left)")
    ax4b.plot(results["date"], results["vxbr"],
              color=BRAZIL_RED, lw=1.0, alpha=0.7, label="VXBR (right)")
    axes[3].set_ylabel("IBOV × 1,000 pts", color=BRAZIL_BLUE, fontsize=9)
    ax4b.set_ylabel("VXBR (%)", color=BRAZIL_RED, fontsize=9)
    axes[3].set_title("IBOV Level vs VXBR", fontsize=11, fontweight="bold")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].grid(True, alpha=0.3, ls="--")
    axes[3].set_facecolor(LIGHT_GRAY)
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    axes[3].legend(lines1+lines2, labels1+labels2, fontsize=8)

    fig2.suptitle(f"VXBR Replication Diagnostics  |  {YEARS[0]}–{YEARS[-1]}  |  IBOV Index Options",
                   fontsize=13, fontweight="bold")
    fig2.tight_layout()
    plot2_path = OUT_DIR / "vxbr_surface_heatmap.png"
    fig2.savefig(str(plot2_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"💾 Saved: {plot2_path}")
except Exception as e:
    print(f"   ⚠  Diagnostics plot skipped: {e}")

# ── Final summary ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print(" FINAL BACKTEST SUMMARY")
print("=" * 65)
print(f"  Period          : {results['date'].min().date()} – {results['date'].max().date()}")
print(f"  Trading days    : {len(results)}")
print(f"  VXBR mean       : {results['vxbr'].mean():.2f}%")
print(f"  VXBR median     : {results['vxbr'].median():.2f}%")
print(f"  VXBR std-dev    : {results['vxbr'].std():.2f}%")
idx_max = results["vxbr"].idxmax()
idx_min = results["vxbr"].idxmin()
print(f"  Peak VXBR       : {results.loc[idx_max,'vxbr']:.2f}% on {results.loc[idx_max,'date'].date()}")
print(f"  Trough VXBR     : {results.loc[idx_min,'vxbr']:.2f}% on {results.loc[idx_min,'date'].date()}")
print(f"  High-vol days   : {(results['vxbr']>30).sum()} ({(results['vxbr']>30).mean()*100:.1f}%)")
print(f"  Bracketed days  : {results['bracketed'].sum()} ({results['bracketed'].mean()*100:.1f}%)")
print(f"  Avg near strikes: {results['n_strikes1'].mean():.1f}")
print(f"  Avg next strikes: {results['n_strikes2'].mean():.1f}")
print()
print("  Output files:")
print(f"    {csv_path}")
print(f"    {plot1_path}")
print(f"    {report_path}")
print()
print("  NOTE: Actual VXBR official values were not available for comparison")
print("  (external data feeds blocked in this environment). To validate,")
print("  overlay the CSV with official VXBR data from B3 / Investing.com.")
print()
print("✅ Backtest complete.")
