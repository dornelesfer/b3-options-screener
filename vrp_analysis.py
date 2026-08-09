"""
vrp_analysis.py
===============
Quantifies the Brazilian volatility risk premium (VRP):

    VRP_t = VXBR_t (30d implied vol)  -  RV_{t -> t+21d} (subsequent realized vol)

This is the model-free edge measurement for a short-vol harvesting strategy:
a short 30-day variance swap replicated by the VIX strip earns ~(IV^2 - RV^2)
in variance points; the sign and persistence of IV - RV is the alpha.

Inputs : results/vxbr_replication_v2.csv  (from vxbr_daily_rates.py)
         data/ibov_daily.csv              (from download_bcb_series.py)
Outputs: results/vrp_daily.csv
         results/vrp_analysis.png
         results/vrp_report.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
RV_WINDOW = 21          # trading days ~ 30 calendar days
ANN = 252

BLUE, RED, GOLD, GRAY = "#009c3b", "#e63946", "#c9a200", "#888888"


def newey_west_tstat(x, lags=None):
    """t-stat of the mean with Newey-West (HAC) standard errors.
    Overlapping 21-day windows induce ~MA(20) autocorrelation, so default
    lags = RV_WINDOW."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    if lags is None:
        lags = RV_WINDOW
    e = x - x.mean()
    g0 = (e @ e) / n
    s = g0
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[:-L] @ e[L:]) / n
    se = np.sqrt(s / n)
    return x.mean() / se


def main():
    vx = pd.read_csv(RES / "vxbr_replication_v2.csv", parse_dates=["date"])
    ibov = pd.read_csv(BASE / "data" / "ibov_daily.csv", parse_dates=["date"])
    ibov = ibov.sort_values("date").reset_index(drop=True)

    # ── forward realized vol over next RV_WINDOW trading days ────────────────
    ibov["logret"] = np.log(ibov["ibov_close"]).diff()
    # RV at t = vol of returns from t+1 .. t+RV_WINDOW (annualized, zero-mean)
    sq = ibov["logret"] ** 2
    fwd_var = sq.shift(-1).rolling(RV_WINDOW).sum().shift(-(RV_WINDOW - 1))
    ibov["rv_fwd"] = np.sqrt(ANN / RV_WINDOW * fwd_var) * 100
    # trailing RV (information available at t, for conditioning)
    trail_var = sq.rolling(RV_WINDOW).sum()
    ibov["rv_trail"] = np.sqrt(ANN / RV_WINDOW * trail_var) * 100

    df = vx.merge(ibov[["date", "ibov_close", "rv_fwd", "rv_trail"]], on="date", how="inner")

    # choose IV series: mid where available, else close-based
    df["iv"] = df["vxbr_mid"].fillna(df["vxbr"])
    df["iv_source"] = np.where(df["vxbr_mid"].notna(), "mid", "close")

    df["vrp"] = df["iv"] - df["rv_fwd"]                        # vol points
    df["vrp_var"] = (df["iv"] / 100) ** 2 - (df["rv_fwd"] / 100) ** 2   # variance points
    df["ts_slope"] = df["sigma1"] - df["sigma2"]               # backwardation > 0
    df["iv_minus_trail"] = df["iv"] - df["rv_trail"]           # ex-ante spread (tradeable signal)

    d = df.dropna(subset=["vrp"]).reset_index(drop=True)
    d.to_csv(RES / "vrp_daily.csv", index=False)

    # ── report ────────────────────────────────────────────────────────────────
    lines = []
    P = lines.append
    P("Brazilian Volatility Risk Premium — VXBR vs subsequent 21d realized vol")
    P("=" * 74)
    P(f"Window        : {d['date'].min().date()} - {d['date'].max().date()}  ({len(d):,} days)")
    P(f"IV source     : {(d['iv_source']=='mid').mean():.0%} mid-quote days, rest close-based")
    P(f"Rates         : daily CDI (r range {d['r_used'].min():.1%}-{d['r_used'].max():.1%})")
    P("")
    P(f"Mean IV (VXBR)           : {d['iv'].mean():7.2f}")
    P(f"Mean subsequent RV       : {d['rv_fwd'].mean():7.2f}")
    P(f"Mean VRP (IV - fwd RV)   : {d['vrp'].mean():7.2f} vol pts   "
      f"(NW t-stat {newey_west_tstat(d['vrp']):.2f})")
    P(f"Median VRP               : {d['vrp'].median():7.2f}")
    P(f"Hit rate (VRP > 0)       : {(d['vrp'] > 0).mean():7.1%}")
    P(f"Mean VRP (variance pts)  : {d['vrp_var'].mean():7.4f}   "
      f"(short 30d var-swap gross return per unit vega notional)")
    P("")

    P("By year")
    P("-" * 74)
    P(f"  {'Year':<6}{'N':>5}{'IV':>8}{'fwdRV':>8}{'VRP':>8}{'Hit%':>7}{'p5 VRP':>9}")
    for y, g in d.groupby(d["date"].dt.year):
        P(f"  {y:<6}{len(g):>5}{g['iv'].mean():>8.1f}{g['rv_fwd'].mean():>8.1f}"
          f"{g['vrp'].mean():>8.1f}{(g['vrp']>0).mean()*100:>6.0f}%"
          f"{g['vrp'].quantile(0.05):>9.1f}")
    P("")

    # conditioning: is the premium richer when ex-ante spread is wide?
    P("Conditional on ex-ante spread (IV - trailing RV), quintiles")
    P("-" * 74)
    d["q_spread"] = pd.qcut(d["iv_minus_trail"], 5, labels=False, duplicates="drop") + 1
    P(f"  {'Q':<4}{'N':>6}{'IV-trailRV':>12}{'realized VRP':>14}{'Hit%':>7}{'NW t':>7}")
    for q, g in d.groupby("q_spread"):
        P(f"  {q:<4.0f}{len(g):>6}{g['iv_minus_trail'].mean():>12.1f}"
          f"{g['vrp'].mean():>14.1f}{(g['vrp']>0).mean()*100:>6.0f}%"
          f"{newey_west_tstat(g['vrp']):>7.2f}")
    P("")

    P("Conditional on term-structure slope (sigma1 - sigma2)")
    P("-" * 74)
    contango = d[d["ts_slope"] <= 0]
    backwd = d[d["ts_slope"] > 0]
    P(f"  Contango  (s1<=s2, n={len(contango):,}): VRP {contango['vrp'].mean():6.2f}  "
      f"hit {(contango['vrp']>0).mean():.0%}  NW t {newey_west_tstat(contango['vrp']):.2f}")
    P(f"  Backwardation (s1>s2, n={len(backwd):,}): VRP {backwd['vrp'].mean():6.2f}  "
      f"hit {(backwd['vrp']>0).mean():.0%}  NW t {newey_west_tstat(backwd['vrp']):.2f}")
    P("")

    P("Conditional on IV level")
    P("-" * 74)
    d["q_iv"] = pd.qcut(d["iv"], 4, labels=False, duplicates="drop") + 1
    for q, g in d.groupby("q_iv"):
        P(f"  IV Q{q:.0f} (mean {g['iv'].mean():5.1f}): VRP {g['vrp'].mean():6.2f}  "
          f"hit {(g['vrp']>0).mean():.0%}  worst {g['vrp'].min():7.1f}")
    P("")
    P("Notes: VRP>0 means implied exceeded subsequent realized (short vol paid).")
    P("Overlapping windows -> Newey-West t-stats with 21 lags.")

    report = "\n".join(lines)
    (RES / "vrp_report.txt").write_text(report)
    print(report)

    # ── plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), facecolor="white",
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})

    ax = axes[0]
    ax.plot(d["date"], d["iv"], lw=0.9, color=BLUE, label="VXBR (30d implied)")
    ax.plot(d["date"], d["rv_fwd"], lw=0.9, color=RED, alpha=0.75,
            label="Subsequent 21d realized")
    ax.set_title("Implied vs Subsequent Realized Volatility — IBOV",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Vol (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1]
    colr = np.where(d["vrp"] > 0, BLUE, RED)
    ax.bar(d["date"], d["vrp"], width=2.0, color=colr, alpha=0.6)
    roll = d.set_index("date")["vrp"].rolling(63).mean()
    ax.plot(roll.index, roll.values, color="black", lw=1.3, label="63d rolling mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Daily VRP (IV − fwd RV) — mean {d['vrp'].mean():.1f} pts, "
                 f"hit rate {(d['vrp']>0).mean():.0%}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Vol pts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")

    ax = axes[2]
    ax.scatter(d["iv_minus_trail"], d["vrp"], s=5, alpha=0.35, color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    # binned means
    bins = pd.qcut(d["iv_minus_trail"], 20, duplicates="drop")
    bm = d.groupby(bins).agg(x=("iv_minus_trail", "mean"), y=("vrp", "mean"))
    ax.plot(bm["x"], bm["y"], color=RED, lw=2, marker="o", ms=4,
            label="binned mean")
    ax.set_title("Ex-ante spread (IV − trailing RV) vs realized VRP",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("IV − trailing 21d RV (known at entry)")
    ax.set_ylabel("Realized VRP")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")

    for a in axes[:2]:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(RES / "vrp_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RES/'vrp_daily.csv'}, {RES/'vrp_analysis.png'}, {RES/'vrp_report.txt'}")


if __name__ == "__main__":
    main()
