"""
covered_call_flow.py
====================
Tests the structural retail covered-call flow on PETR4: persistent call
selling should cheapen calls relative to puts at the same strike.

Measurement (daily, nearest cycle 20-45 DTE)
    gap = IV(ATM call) - IV(ATM put)   at the same strike
    gap < 0  =>  calls cheap (covered-call supply pressure)

Strategies (monthly cycles, held to expiry, delta-hedged daily, PETR4 stock
as the hedge; B3 strike protection handled: on each ex-date the strike is
reduced by the dividend and the hedge position receives it)

    A. long ATM call, always            (baseline: pays the VRP, expect <= 0)
    B. long ATM call when gap in bottom expanding quintile (calls very cheap)
    C. short ATM put, always            (equity VRP harvest baseline)
    D. short ATM put when gap in bottom quintile (puts relatively rich)

Costs: B3+brokerage 0.314% of premium, 5% spread haircut (paid on entry for
longs, received-less for shorts), 0.05% stock hedge cost per rebalance
(equities cost more than index futures).

Outputs: results/covered_call_daily.csv, results/covered_call_report.txt,
         results/covered_call_flow.png
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

from backtest_short_vol import bs_delta, implied_vol

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"

UND = "PETR4"
DTE_LO, DTE_HI = 20, 45
FEE_PCT = 0.00114 + 0.0020
HAIRCUT = 0.05
HEDGE_COST = 0.0005          # stock, per rebalance on traded notional
MIN_HISTORY = 250
Q_CHEAP = 0.20

BLUE, RED, GOLD, GRAY = "#009c3b", "#e63946", "#c9a200", "#888888"


def load():
    opts = pq.read_table(str(BASE / "data" / "equity_options.parquet")).to_pandas()
    opts = opts[(opts["underlying"] == UND) & (opts["close"] > 0)
                & (opts["traded_contracts"] > 0)]
    spot = pq.read_table(str(BASE / "data" / "equity_spot.parquet")).to_pandas()
    spot = (spot[(spot["symbol"] == UND) & (spot["close"] > 0)]
            .drop_duplicates("refdate").set_index("refdate")["close"].sort_index())
    rates = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    r_cc = rates.dropna(subset=["r_cc"]).set_index("date")["r_cc"]
    r_cc = r_cc.reindex(pd.date_range(r_cc.index.min(), r_cc.index.max())).ffill()
    divs = pd.read_csv(BASE / f"data/dividends_{UND}.csv", parse_dates=["exdate"])
    return opts, spot, r_cc, divs


# ── daily gap measurement + cycle entries ─────────────────────────────────────
def build_daily_gap(opts, spot, r_cc):
    rows = []
    for (day, expiry), g in opts.groupby(["refdate", "maturity_date"]):
        dte = (expiry - day).days
        if not (DTE_LO <= dte <= DTE_HI) or day not in spot.index:
            continue
        S = float(spot.loc[day])
        T = dte / 365.0
        r = float(r_cc.loc[day]) if day in r_cc.index else np.nan
        if np.isnan(r):
            continue
        F = S * np.exp(r * T)
        calls = g[g["bdi_code"] == 78].set_index("strike_price")["close"]
        puts = g[g["bdi_code"] == 82].set_index("strike_price")["close"]
        common = sorted(set(calls.index) & set(puts.index))
        if not common:
            continue
        K = min(common, key=lambda k: abs(k - F))
        if abs(K - F) / F > 0.05:
            continue
        iv_c = implied_vol(float(calls.loc[K]), S, K, T, r, "C")
        iv_p = implied_vol(float(puts.loc[K]), S, K, T, r, "P")
        if np.isnan(iv_c) or np.isnan(iv_p):
            continue
        rows.append({"date": day, "expiry": expiry, "dte": dte, "S": S, "K": K,
                     "call_px": float(calls.loc[K]), "put_px": float(puts.loc[K]),
                     "iv_call": iv_c * 100, "iv_put": iv_p * 100,
                     "gap": (iv_c - iv_p) * 100})
    df = pd.DataFrame(rows).sort_values("date")
    # one obs per day: nearest to 30 dte
    df["dist30"] = (df["dte"] - 30).abs()
    return (df.sort_values("dist30").groupby("date").first()
            .reset_index().sort_values("date").reset_index(drop=True))


# ── single-leg delta-hedged trade with dividend-protected strike ─────────────
def simulate_leg(entry, expiry, K0, px, cp, side, spot, r_cc, divs):
    """side=+1 long option, -1 short. Returns dict or None."""
    if entry not in spot.index:
        return None
    S0 = float(spot.loc[entry])
    T0 = (expiry - entry).days / 365.0
    r0 = float(r_cc.loc[entry]) if entry in r_cc.index else float(r_cc.iloc[-1])
    iv = implied_vol(px, S0, K0, T0, r0, cp)
    if np.isnan(iv):
        return None

    path = spot.loc[entry:expiry]
    if len(path) < 5 or (expiry - path.index[-1]).days > 5:
        return None
    dv = divs[(divs["exdate"] > entry) & (divs["exdate"] <= expiry)]

    # entry cash flow: long pays px*(1+HAIRCUT), short receives px*(1-HAIRCUT)
    prem = px * (1 - side * HAIRCUT)     # signed below via side

    hedge_pnl, hedge_cost, h_prev = 0.0, 0.0, 0.0
    K = K0
    for i in range(len(path) - 1):
        t, t1 = path.index[i], path.index[i + 1]
        S_t = float(path.iloc[i])
        T_rem = (expiry - t).days / 365.0
        r_t = float(r_cc.loc[t]) if t in r_cc.index else r0
        # hedge offsets the option delta: position = -side * delta
        h = -side * bs_delta(S_t, K, T_rem, r_t, iv, cp)
        hedge_cost += abs(h - h_prev) * S_t * HEDGE_COST
        # dividends ex between t (exclusive) and t1 (inclusive):
        dd = dv[(dv["exdate"] > t) & (dv["exdate"] <= t1)]
        div_amt = float(dd["amount"].sum())
        # stock hedge receives the dividend; strike is protected (reduced)
        hedge_pnl += h * (float(path.iloc[i + 1]) - S_t + div_amt)
        K -= div_amt
        h_prev = h
    S_T = float(path.iloc[-1])
    hedge_cost += abs(h_prev) * S_T * HEDGE_COST

    payoff = max(S_T - K, 0.0) if cp == "C" else max(K - S_T, 0.0)
    fees = FEE_PCT * px
    # option leg P&L: side*(payoff - prem_fv); premium financed/earning CDI
    pnl = side * (payoff - prem * np.exp(r0 * T0)) + hedge_pnl - hedge_cost - fees
    return {"entry": entry, "expiry": expiry, "K": K0, "cp": cp, "side": side,
            "S0": S0, "S_T": S_T, "iv_entry": iv * 100, "prem": px,
            "pnl": pnl, "pnl_pct": pnl / S0 * 100}


def perf(t, label):
    t = t.dropna(subset=["pnl_pct"])
    if len(t) < 5:
        return f"  {label:<38} insufficient trades ({len(t)})"
    per_yr = len(t) / max((t["entry"].max() - t["entry"].min()).days / 365.25, 1)
    mu, sd = t["pnl_pct"].mean(), t["pnl_pct"].std()
    sharpe = mu / sd * np.sqrt(per_yr) if sd > 0 else np.nan
    return (f"  {label:<38} N={len(t):>4}  mean={mu:6.3f}%  "
            f"hit={(t['pnl_pct']>0).mean():5.1%}  Sharpe={sharpe:5.2f}  "
            f"worst={t['pnl_pct'].min():7.2f}%")


def main():
    print("=" * 76)
    print(f" Covered-call flow test — {UND} ATM call vs put IV, delta-hedged trades")
    print("=" * 76)
    opts, spot, r_cc, divs = load()

    print("\n[1/3] Building daily call/put IV gap ...")
    daily = build_daily_gap(opts, spot, r_cc)
    daily.to_csv(RES / "covered_call_daily.csv", index=False)
    print(f"  {len(daily):,} days ({daily['date'].min().date()} - "
          f"{daily['date'].max().date()})")

    # expanding quintile threshold of gap (no lookahead)
    gap = daily.set_index("date")["gap"]
    thr = gap.expanding(MIN_HISTORY).quantile(Q_CHEAP).shift(1)
    daily["cheap"] = daily["gap"].values < thr.values

    print("[2/3] Monthly cycles: first eligible day per expiry ...")
    cycles = daily.groupby("expiry").first().reset_index()

    print("[3/3] Simulating ...")
    rows = []
    for _, c in cycles.iterrows():
        for cp, px, side_lbls in (("C", c["call_px"], ("long_call",)),
                                  ("P", c["put_px"], ("short_put",))):
            side = +1 if cp == "C" else -1
            res = simulate_leg(c["date"], c["expiry"], c["K"], px, cp, side,
                               spot, r_cc, divs)
            if res:
                res["cheap"] = bool(c["cheap"]) if pd.notna(c["cheap"]) else False
                res["gap"] = c["gap"]
                rows.append(res)
    tr = pd.DataFrame(rows)
    lc = tr[tr["cp"] == "C"]
    sp = tr[tr["cp"] == "P"]

    lines = []
    P = lines.append
    P(f"Covered-call flow — {UND}")
    P("=" * 76)
    P(f"Daily gap = IV(ATM call) - IV(ATM put), same strike, ~30 DTE")
    P(f"  days          : {len(daily):,}")
    P(f"  mean gap      : {daily['gap'].mean():6.2f} vol pts")
    P(f"  median gap    : {daily['gap'].median():6.2f}")
    P(f"  gap < 0 days  : {(daily['gap'] < 0).mean():.0%}   "
      f"(calls cheaper than puts)")
    P("")
    P("Yearly median gap (vol pts):")
    ym = daily.groupby(daily["date"].dt.year)["gap"].median().round(2)
    P("  " + "  ".join(f"{y}:{v:+.1f}" for y, v in ym.items()))
    P("")
    P("Delta-hedged monthly trades, held to expiry (pnl % of stock notional)")
    P("-" * 76)
    P(perf(lc, "A. long ATM call, always"))
    P(perf(lc[lc["cheap"]], "B. long call when gap in bottom q20"))
    P(perf(sp, "C. short ATM put, always"))
    P(perf(sp[sp["cheap"]], "D. short put when gap in bottom q20"))
    P("")
    P("Read: A vs B isolates whether call cheapness (covered-call supply) is")
    P("deep enough to overcome the vol risk premium a long option pays.")
    P("C is the single-stock VRP baseline; D adds the relative-richness filter.")

    report = "\n".join(lines)
    (RES / "covered_call_report.txt").write_text(report)
    print("\n" + report)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="white")
    roll = daily.set_index("date")["gap"].rolling(42, min_periods=15).median()
    axes[0].plot(roll.index, roll.values, lw=1.3, color=BLUE)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title(f"{UND} ATM call IV - put IV, 2-month rolling median (vol pts)",
                      fontweight="bold")
    axes[0].grid(alpha=0.3, ls="--")
    axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for t, lbl, c in ((lc, "long call always", GRAY),
                      (lc[lc["cheap"]], "long call, cheap filter", BLUE),
                      (sp, "short put always", GOLD),
                      (sp[sp["cheap"]], "short put, cheap filter", RED)):
        t = t.dropna(subset=["pnl_pct"])
        if len(t) > 4:
            axes[1].plot(t["expiry"], t["pnl_pct"].cumsum(), lw=1.4,
                         color=c, label=lbl)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Cumulative P&L (% of notional)", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, ls="--")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(RES / "covered_call_flow.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RES/'covered_call_daily.csv'}, {RES/'covered_call_report.txt'}, "
          f"{RES/'covered_call_flow.png'}")


if __name__ == "__main__":
    main()
