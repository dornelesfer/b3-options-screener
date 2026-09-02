"""
implied_carry.py
================
Options-implied carry on PETR4 / VALE3 (VALE5 pre-2017), with the dividend
protection test.

For every (date, expiry) with traded call+put pairs at the same strike:

    F_impl = K + e^{rT} (C - P)          per pair (median across pairs)
    q_impl = r - ln(F_impl / S) / T      annualized implied carry residual

If B3's strike adjustment fully protects options from dividends, q_impl
should NOT respond to upcoming dividends (coef ~ 0 in the regression below);
whatever remains is borrow (aluguel) + early-exercise premium + mispricing.
If protection leaks, q_impl loads on the upcoming dividend yield (coef ~ 1)
and dividend forecasting is directly monetizable via conversions/reversals.

Filters: 10-90 DTE, strikes within +-10% of spot (limits American-call
early-exercise contamination), both legs traded_contracts > 0, close > 0.

Outputs: results/implied_carry_daily.csv, results/implied_carry_report.txt,
         results/implied_carry.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
DTE_LO, DTE_HI = 10, 90
MONEYNESS = 0.10
MIN_PAIRS = 2

BLUE, RED, GOLD, GRAY = "#009c3b", "#e63946", "#c9a200", "#888888"


def load():
    from data_cache import load_options
    opts = load_options("equity_options")
    opts = opts[(opts["close"] > 0) & (opts["traded_contracts"] > 0)]

    spot = pq.read_table(str(BASE / "data" / "equity_spot.parquet")).to_pandas()
    spot = (spot[spot["close"] > 0]
            .drop_duplicates(["refdate", "symbol"])
            .pivot(index="refdate", columns="symbol", values="close"))

    rates = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    r_cc = rates.dropna(subset=["r_cc"]).set_index("date")["r_cc"]
    r_cc = r_cc.reindex(pd.date_range(r_cc.index.min(), r_cc.index.max())).ffill()

    divs = {}
    for name in ("PETR4", "VALE3"):
        d = pd.read_csv(BASE / f"data/dividends_{name}.csv", parse_dates=["exdate"])
        divs[name] = d
    divs["VALE5"] = divs["VALE3"]        # same company, pre-unification proxy
    return opts, spot, r_cc, divs


def upcoming_div_yield(divs, S, d0, d1, T):
    """Annualized yield of dividends going ex in (d0, d1]."""
    dd = divs[(divs["exdate"] > d0) & (divs["exdate"] <= d1)]
    if dd.empty or T <= 0:
        return 0.0
    return float(dd["amount"].sum()) / S / T


def main():
    print("=" * 74)
    print(" Options-implied carry & the dividend-protection test")
    print("=" * 74)
    opts, spot, r_cc, divs = load()

    rows = []
    for und, g_und in opts.groupby("underlying"):
        if und not in spot.columns:
            continue
        s_series = spot[und].dropna()
        print(f"  {und}: {len(g_und):,} option rows ...")

        for (day, expiry), g in g_und.groupby(["refdate", "maturity_date"]):
            dte = (expiry - day).days
            if not (DTE_LO <= dte <= DTE_HI) or day not in s_series.index:
                continue
            S = float(s_series.loc[day])
            T = dte / 365.0
            r = float(r_cc.loc[day]) if day in r_cc.index else np.nan
            if np.isnan(r):
                continue

            near = g[np.abs(g["strike_price"] - S) / S <= MONEYNESS]
            calls = near[near["bdi_code"] == 78].set_index("strike_price")["close"]
            puts = near[near["bdi_code"] == 82].set_index("strike_price")["close"]
            common = sorted(set(calls.index) & set(puts.index))
            if len(common) < MIN_PAIRS:
                continue

            f_impls = [K + np.exp(r * T) * (float(calls.loc[K]) - float(puts.loc[K]))
                       for K in common]
            F = float(np.median(f_impls))
            if F <= 0:
                continue
            q_impl = r - np.log(F / S) / T
            rows.append({
                "date": day, "expiry": expiry, "underlying": und,
                "dte": dte, "S": S, "F_impl": F, "n_pairs": len(common),
                "r": r, "q_impl": q_impl,
                "div_yld_ahead": upcoming_div_yield(divs[und], S, day, expiry, T),
            })

    df = pd.DataFrame(rows).sort_values(["underlying", "date", "dte"])
    # one obs per (underlying, date): expiry nearest 30 DTE
    df["dist30"] = (df["dte"] - 30).abs()
    daily = (df.sort_values("dist30").groupby(["underlying", "date"]).first()
             .reset_index().sort_values(["underlying", "date"]))
    daily.to_csv(RES / "implied_carry_daily.csv", index=False)

    lines = []
    P = lines.append
    P("Options-implied carry (q_impl) — PETR4 / VALE3 / VALE5")
    P("=" * 74)
    P("q_impl = r_CDI - ln(F_impl/S)/T, annualized. Under full dividend")
    P("protection q_impl ~ borrow fee + early-exercise + mispricing (small, >0).")
    P("")

    for und, g in daily.groupby("underlying"):
        P(f"{und}:  {len(g):,} days  ({g['date'].min().date()} - {g['date'].max().date()})")
        P(f"   q_impl  median {g['q_impl'].median():7.2%}   "
          f"IQR [{g['q_impl'].quantile(.25):6.2%}, {g['q_impl'].quantile(.75):6.2%}]")

        # dividend protection regression: q_impl = a + b * div_yld_ahead
        gg = g.dropna(subset=["q_impl", "div_yld_ahead"])
        x, y = gg["div_yld_ahead"].values, gg["q_impl"].values
        if len(gg) > 100 and x.std() > 0:
            b = np.cov(x, y)[0, 1] / np.var(x)
            a = y.mean() - b * x.mean()
            resid = y - (a + b * x)
            se_b = np.sqrt(resid.var() / (len(x) * np.var(x)))
            P(f"   protection test:  q_impl = {a:.4f} + {b:.3f} x div_yld_ahead"
              f"   (t = {b/se_b:.1f})")
            P(f"   -> coef ~1 means dividends flow into forwards (unprotected);")
            P(f"      coef ~0 means strike protection holds.")
            days_with_div = (gg["div_yld_ahead"] > 0).mean()
            P(f"   days with a dividend before expiry: {days_with_div:.0%}")
        P("")

    P("Yearly median q_impl (30d), by underlying")
    P("-" * 74)
    piv = (daily.groupby([daily["date"].dt.year, "underlying"])["q_impl"]
           .median().unstack().round(4))
    P(piv.to_string())
    P("")
    P("Trading read: persistent q_impl > CDI-consistent borrow (~0.5-2% for these")
    P("names) = synthetic forward trades cheap vs spot -> reversal (buy synthetic,")
    P("short stock) collects the residual; spikes around big dividends = the")
    P("protection mechanism lagging announced-but-not-ex dividends.")

    report = "\n".join(lines)
    (RES / "implied_carry_report.txt").write_text(report)
    print("\n" + report)

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="white", sharex=True)
    for und, c in (("PETR4", BLUE), ("VALE3", RED), ("VALE5", GOLD)):
        g = daily[daily["underlying"] == und]
        if g.empty:
            continue
        roll = g.set_index("date")["q_impl"].rolling(21, min_periods=10).median()
        axes[0].plot(roll.index, roll.values * 100, lw=1.2, color=c, label=und)
        axes[1].plot(g["date"], g["div_yld_ahead"] * 100, lw=0.9, color=c,
                     alpha=0.7, label=und)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Implied carry residual q_impl — 21d rolling median (%)",
                      fontweight="bold")
    axes[0].set_ylabel("% p.a."); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, ls="--")
    axes[0].set_ylim(-25, 40)
    axes[1].set_title("Annualized yield of dividends going ex before expiry (%)",
                      fontweight="bold")
    axes[1].set_ylabel("% p.a."); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, ls="--")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(RES / "implied_carry.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RES/'implied_carry_daily.csv'}, {RES/'implied_carry_report.txt'}, "
          f"{RES/'implied_carry.png'}")


if __name__ == "__main__":
    main()
