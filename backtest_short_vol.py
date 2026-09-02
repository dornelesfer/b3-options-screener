"""
backtest_short_vol.py
=====================
Delta-hedged short-vol harvest on IBOV index options — honest holding-period
backtest (no same-day mark-to-market).

Trade template
--------------
  * One position per monthly expiry cycle: on the first day an expiry has
    25-35 calendar DTE and both ATM legs actually traded (contracts > 0),
    sell 1 ATM straddle (strike nearest the CDI-implied forward) at close
    prices minus a spread haircut.
  * Delta-hedge daily to expiry with the index (WIN futures proxy), deltas
    from Black-Scholes at each leg's entry implied vol, r = daily CDI.
  * Settle at expiry on the IBOV close (cash-settled), pay all costs.

Signals compared
----------------
  A. Unconditional      — sell every cycle.
  B. Spread filter      — sell only when VXBR - trailing 21d RV > 0.
  C. Expanding quintile — sell only when the spread is above its expanding
                          40th percentile (no look-ahead; needs 250 obs).

Costs (configurable)
--------------------
  * Option fees: B3 emolumentos+liquidação 0.114% + brokerage 0.20% of premium.
  * Spread haircut on entry premium: default 5% (close is a mid proxy;
    sensitivity table at 3/5/10% reported).
  * Hedge: 0.02% of traded hedge notional per rebalance (WIN futures are
    fixed-fee, ~R$0.52/contract RT; 0.02% is deliberately conservative).

Inputs : data/ibov_options_all.parquet, data/ibov_daily.csv,
         data/rates_cdi.csv, results/vrp_daily.csv
Outputs: results/short_vol_trades.csv, results/short_vol_report.txt,
         results/short_vol_backtest.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"

DTE_MIN, DTE_MAX = 25, 35
FEE_PCT = 0.00114 + 0.0020        # B3 + brokerage, on premium notional
HAIRCUT = 0.05                    # entry spread haircut on premium
HEDGE_COST = 0.0002               # per rebalance, on traded hedge notional
IV_BOUNDS = (0.05, 2.0)
MIN_HISTORY_FOR_QUANTILE = 250

BLUE, RED, GOLD, GRAY = "#009c3b", "#e63946", "#c9a200", "#888888"


# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sig, cp):
    if T <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sig, cp):
    if T <= 0:
        if cp == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, cp):
    intrinsic = bs_price(S, K, 0, r, 0.2, cp)
    if price <= intrinsic + 1e-9:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, cp) - price,
                      IV_BOUNDS[0], IV_BOUNDS[1], xtol=1e-6)
    except ValueError:
        return np.nan


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data():
    from data_cache import load_options
    opts = load_options("ibov_options_all")
    opts = opts[(opts["close"] > 0)]

    ibov = pd.read_csv(BASE / "data" / "ibov_daily.csv", parse_dates=["date"])
    ibov = ibov.sort_values("date").set_index("date")["ibov_close"]

    rates = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    r_cc = rates.dropna(subset=["r_cc"]).set_index("date")["r_cc"]
    r_cc = r_cc.reindex(pd.date_range(r_cc.index.min(), r_cc.index.max())).ffill()

    vrp = pd.read_csv(RES / "vrp_daily.csv", parse_dates=["date"])
    signal = vrp.set_index("date")["iv_minus_trail"]
    return opts, ibov, r_cc, signal


def spot_on_or_before(ibov, d, max_back=5):
    """IBOV close on d, or nearest previous trading day."""
    d = pd.Timestamp(d)
    for k in range(max_back + 1):
        dd = d - pd.Timedelta(days=k)
        if dd in ibov.index:
            return float(ibov.loc[dd]), dd
    return np.nan, None


# ── Trade simulation ─────────────────────────────────────────────────────────
def simulate_trade(entry, expiry, K, c_px, p_px, ibov, r_cc):
    """Short 1 ATM straddle, daily delta hedge, hold to expiry.
    Returns dict or None."""
    S0, _ = spot_on_or_before(ibov, entry, 0)
    if np.isnan(S0):
        return None
    T0 = (expiry - entry).days / 365.0
    r0 = float(r_cc.loc[entry]) if entry in r_cc.index else float(r_cc.iloc[-1])

    iv_c = implied_vol(c_px, S0, K, T0, r0, "C")
    iv_p = implied_vol(p_px, S0, K, T0, r0, "P")
    if np.isnan(iv_c) or np.isnan(iv_p):
        return None

    # premium received at entry (after spread haircut)
    prem_gross = c_px + p_px
    prem_net = prem_gross * (1 - HAIRCUT)

    # path of trading days from entry to expiry (inclusive of settlement day)
    path = ibov.loc[entry:expiry]
    if len(path) < 5 or (expiry - path.index[-1]).days > 5:
        return None
    days = path.index
    S_T = float(path.iloc[-1])

    # daily hedge: hold h_t = +delta(straddle) to offset the short position
    hedge_pnl, hedge_cost, h_prev = 0.0, 0.0, 0.0
    for i in range(len(days) - 1):
        t = days[i]
        S_t = float(path.iloc[i])
        T_rem = (expiry - t).days / 365.0
        r_t = float(r_cc.loc[t]) if t in r_cc.index else r0
        h = (bs_delta(S_t, K, T_rem, r_t, iv_c, "C")
             + bs_delta(S_t, K, T_rem, r_t, iv_p, "P"))
        hedge_cost += abs(h - h_prev) * S_t * HEDGE_COST
        hedge_pnl += h * (float(path.iloc[i + 1]) - S_t)
        h_prev = h
    hedge_cost += abs(h_prev) * S_T * HEDGE_COST     # final unwind

    payoff = max(S_T - K, 0.0) + max(K - S_T, 0.0)
    prem_fv = prem_net * np.exp(r0 * T0)             # premium earns CDI
    fees = FEE_PCT * prem_gross

    pnl = prem_fv - payoff + hedge_pnl - hedge_cost - fees
    return {
        "entry": entry, "expiry": expiry, "K": K, "S0": S0, "S_T": S_T,
        "dte": (expiry - entry).days, "call_px": c_px, "put_px": p_px,
        "iv_entry": (iv_c + iv_p) / 2 * 100, "prem_gross": prem_gross,
        "payoff": payoff, "hedge_pnl": hedge_pnl,
        "costs": hedge_cost + fees + prem_gross * HAIRCUT,
        "pnl_pts": pnl, "pnl_pct": pnl / S0 * 100,
    }


def find_cycle_entries(opts, ibov, r_cc):
    """For each expiry, first eligible entry day with a traded ATM pair."""
    entries = []
    for expiry, g in opts.groupby("maturity_date"):
        g = g.copy()
        g["dte"] = (expiry - g["refdate"]).dt.days
        g = g[(g["dte"] >= DTE_MIN) & (g["dte"] <= DTE_MAX)
              & (g["traded_contracts"] > 0)]
        if g.empty:
            continue
        for day in sorted(g["refdate"].unique()):
            dd = g[g["refdate"] == day]
            calls = dd[dd["bdi_code"] == 74].set_index("strike_price")["close"]
            puts = dd[dd["bdi_code"] == 75].set_index("strike_price")["close"]
            common = sorted(set(calls.index) & set(puts.index))
            if not common:
                continue
            S, _ = spot_on_or_before(ibov, day, 0)
            if np.isnan(S):
                continue
            r = float(r_cc.loc[day]) if day in r_cc.index else 0.10
            T = (expiry - day).days / 365.0
            F = S * np.exp(r * T)
            K = min(common, key=lambda k: abs(k - F))
            if abs(K - F) / F > 0.05:      # no strike within 5% of forward
                continue
            entries.append({"entry": pd.Timestamp(day), "expiry": pd.Timestamp(expiry),
                            "K": float(K), "call_px": float(calls.loc[K]),
                            "put_px": float(puts.loc[K])})
            break                           # first eligible day only
    return pd.DataFrame(entries).sort_values("entry").reset_index(drop=True)


# ── Strategy stats ────────────────────────────────────────────────────────────
def perf(trades, label):
    t = trades.dropna(subset=["pnl_pct"])
    if len(t) == 0:
        return f"  {label:<26} no trades"
    per_yr = len(t) / max((t["entry"].max() - t["entry"].min()).days / 365.25, 1)
    mu, sd = t["pnl_pct"].mean(), t["pnl_pct"].std()
    sharpe = mu / sd * np.sqrt(per_yr) if sd > 0 else np.nan
    cum = t["pnl_pct"].cumsum()
    dd = (cum - cum.cummax()).min()
    return (f"  {label:<26} N={len(t):>4}  mean={mu:6.3f}%  hit={ (t['pnl_pct']>0).mean():5.1%}"
            f"  Sharpe={sharpe:5.2f}  worst={t['pnl_pct'].min():7.2f}%  maxDD={dd:7.2f}%")


def main():
    print("=" * 74)
    print(" Delta-hedged short ATM straddle — IBOV index options, held to expiry")
    print("=" * 74)
    opts, ibov, r_cc, signal = load_data()

    print("\n[1/3] Finding monthly cycle entries ...")
    entries = find_cycle_entries(opts, ibov, r_cc)
    print(f"  {len(entries)} cycles with a tradeable ATM pair "
          f"({entries['entry'].min().date()} - {entries['entry'].max().date()})")

    print("[2/3] Simulating trades ...")
    rows = []
    for _, e in entries.iterrows():
        res = simulate_trade(e["entry"], e["expiry"], e["K"],
                             e["call_px"], e["put_px"], ibov, r_cc)
        if res:
            rows.append(res)
    trades = pd.DataFrame(rows)
    print(f"  {len(trades)} simulated (rest dropped: bad IV inversion or no path)")

    # attach signal (as-of entry date; signal uses only trailing info).
    # VXBR isn't computable every day — carry the last value up to 5 calendar
    # days forward (stale-but-known, no look-ahead).
    sig_daily = signal.dropna().reindex(
        pd.date_range(signal.index.min(), signal.index.max())).ffill(limit=5)
    trades["signal"] = sig_daily.reindex(trades["entry"]).values
    # expanding 40th percentile threshold, shifted so entry-day obs excluded
    s = signal.dropna()
    thresh = s.expanding(MIN_HISTORY_FOR_QUANTILE).quantile(0.40).shift(1)
    trades["sig_thresh"] = thresh.reindex(trades["entry"], method="ffill").values

    trades["take_B"] = trades["signal"] > 0
    trades["take_C"] = trades["signal"] > trades["sig_thresh"]
    trades.to_csv(RES / "short_vol_trades.csv", index=False)

    print("\n[3/3] Results (pnl in % of index notional per trade)")
    lines = [
        "Delta-hedged short ATM IBOV straddle — holding-period backtest",
        "=" * 74,
        f"Window   : {trades['entry'].min().date()} - {trades['expiry'].max().date()}",
        f"Costs    : fees {FEE_PCT:.3%} of premium, haircut {HAIRCUT:.0%} of premium, "
        f"hedge {HEDGE_COST:.2%}/rebalance",
        f"Cycles   : {len(trades)} trades, avg DTE {trades['dte'].mean():.0f}, "
        f"avg entry IV {trades['iv_entry'].mean():.1f}%",
        "",
        perf(trades, "A. Unconditional"),
        perf(trades[trades["take_B"]], "B. Spread > 0"),
        perf(trades[trades["take_C"].fillna(False)], "C. Spread > exp. q40"),
        "",
        "Cost sensitivity (strategy B), haircut on entry premium:",
    ]
    base_costless = trades.copy()
    for hc in (0.03, 0.05, 0.10):
        adj = trades.copy()
        # re-derive pnl under different haircut: shift premium by (HAIRCUT-hc)
        dprem = adj["prem_gross"] * (HAIRCUT - hc)
        adj["pnl_pct"] = adj["pnl_pct"] + dprem / adj["S0"] * 100
        lines.append(perf(adj[adj["take_B"]], f"   haircut {hc:.0%}"))
    lines += [
        "",
        "Worst 5 trades (B):",
    ]
    worst = trades[trades["take_B"]].nsmallest(5, "pnl_pct")
    for _, w in worst.iterrows():
        lines.append(f"   {w['entry'].date()} -> {w['expiry'].date()}  K={w['K']:,.0f}"
                     f"  IV={w['iv_entry']:.0f}%  pnl={w['pnl_pct']:.2f}%")
    lines += [
        "",
        "P&L decomposition per avg trade (pts): "
        f"premium {trades['prem_gross'].mean():,.0f}, payoff {trades['payoff'].mean():,.0f}, "
        f"hedge {trades['hedge_pnl'].mean():,.0f}, costs {trades['costs'].mean():,.0f}",
        "",
        "Notes: entry at close-price mid proxy minus haircut; both legs must have",
        "traded that day; settlement on IBOV close at expiry; premium earns CDI;",
        "delta hedged daily at entry IVs; no intra-cycle re-marking of options.",
    ]
    report = "\n".join(lines)
    (RES / "short_vol_report.txt").write_text(report)
    print(report)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")

    ax = axes[0][0]
    for take, lbl, c in [(trades["take_B"], "B: spread>0", BLUE),
                         (pd.Series(True, index=trades.index), "A: always", GRAY)]:
        t = trades[take.fillna(False)]
        ax.plot(t["expiry"], t["pnl_pct"].cumsum(), lw=1.6, color=c, label=lbl)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Cumulative P&L (% of notional, 1x per cycle)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, ls="--")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0][1]
    ax.hist(trades["pnl_pct"], bins=40, color=BLUE, alpha=0.75)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Per-trade P&L distribution (%)", fontweight="bold")
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1][0]
    ax.scatter(trades["signal"], trades["pnl_pct"], s=18, alpha=0.6, color=BLUE)
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Entry signal (IV − trail RV) vs trade P&L", fontweight="bold")
    ax.set_xlabel("signal at entry"); ax.set_ylabel("pnl %")
    ax.grid(alpha=0.3, ls="--")

    ax = axes[1][1]
    yr = trades[trades["take_B"]].groupby(trades["entry"].dt.year)["pnl_pct"].sum()
    ax.bar(yr.index.astype(str), yr.values,
           color=[BLUE if v > 0 else RED for v in yr.values], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Strategy B: yearly P&L (%)", fontweight="bold")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(alpha=0.3, ls="--")

    fig.tight_layout()
    fig.savefig(RES / "short_vol_backtest.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RES/'short_vol_trades.csv'}, {RES/'short_vol_report.txt'}, "
          f"{RES/'short_vol_backtest.png'}")


if __name__ == "__main__":
    main()
