"""
backtest_short_vol_v2.py
========================
Extension of backtest_short_vol.py:

  Structures
    - ATM straddle (baseline, as v1)
    - 25-delta strangle: short OTM call at strike nearest +0.25 delta and
      OTM put nearest -0.25 delta (deltas from ATM IV proxy at entry).
      Both legs must have traded that day.

  Sizing (per monthly cycle)
    - fixed   : 1 unit per cycle (v1 behaviour)
    - vega    : units scaled to constant vega across trades
                (target = median ATM straddle vega, so results are
                comparable with the fixed-size baseline)
    - vega+sig: vega-normalized, then scaled by the expanding percentile
                rank of the entry signal (IV - trailing RV):
                rank < 0.40 -> skip; else units *= 2*(rank-0.40)/0.60
                (0 at the threshold, 2x at the strongest signal ever seen;
                 no look-ahead: rank vs history strictly before entry)

  Everything else identical to v1: entry at close minus haircut, daily BS
  delta hedge at entry IVs with daily CDI, cash settlement on IBOV close,
  B3 fees + hedge costs, hold to expiry.

Outputs: results/short_vol_v2_trades.csv, results/short_vol_v2_report.txt,
         results/short_vol_v2_backtest.png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

from backtest_short_vol import (
    bs_price, bs_delta, implied_vol, load_data, spot_on_or_before,
    DTE_MIN, DTE_MAX, FEE_PCT, HAIRCUT, HEDGE_COST,
)

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
TARGET_DELTA = 0.25
MIN_HISTORY = 250
BLUE, RED, GOLD, GRAY = "#009c3b", "#e63946", "#c9a200", "#888888"


def bs_vega(S, K, T, r, sig):
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ── generic multi-leg short structure, delta-hedged, held to expiry ───────────
def simulate_structure(entry, expiry, legs, ibov, r_cc):
    """legs: list of (K, entry_price, 'C'|'P'). Short 1x each leg.
    Returns dict or None."""
    S0, _ = spot_on_or_before(ibov, entry, 0)
    if np.isnan(S0):
        return None
    T0 = (expiry - entry).days / 365.0
    r0 = float(r_cc.loc[entry]) if entry in r_cc.index else float(r_cc.iloc[-1])

    ivs = []
    for K, px, cp in legs:
        iv = implied_vol(px, S0, K, T0, r0, cp)
        if np.isnan(iv):
            return None
        ivs.append(iv)

    prem_gross = sum(px for _, px, _ in legs)
    prem_net = prem_gross * (1 - HAIRCUT)

    path = ibov.loc[entry:expiry]
    if len(path) < 5 or (expiry - path.index[-1]).days > 5:
        return None
    S_T = float(path.iloc[-1])

    hedge_pnl, hedge_cost, h_prev = 0.0, 0.0, 0.0
    for i in range(len(path) - 1):
        t = path.index[i]
        S_t = float(path.iloc[i])
        T_rem = (expiry - t).days / 365.0
        r_t = float(r_cc.loc[t]) if t in r_cc.index else r0
        h = sum(bs_delta(S_t, K, T_rem, r_t, iv, cp)
                for (K, _, cp), iv in zip(legs, ivs))
        hedge_cost += abs(h - h_prev) * S_t * HEDGE_COST
        hedge_pnl += h * (float(path.iloc[i + 1]) - S_t)
        h_prev = h
    hedge_cost += abs(h_prev) * S_T * HEDGE_COST

    payoff = sum(max(S_T - K, 0.0) if cp == "C" else max(K - S_T, 0.0)
                 for K, _, cp in legs)
    fees = FEE_PCT * prem_gross
    pnl = prem_net * np.exp(r0 * T0) - payoff + hedge_pnl - hedge_cost - fees

    vega = sum(bs_vega(S0, K, T0, r0, iv) for (K, _, cp), iv in zip(legs, ivs))
    return {"entry": entry, "expiry": expiry, "S0": S0, "S_T": S_T,
            "dte": (expiry - entry).days,
            "iv_entry": float(np.mean(ivs)) * 100,
            "prem_gross": prem_gross, "payoff": payoff,
            "vega": vega, "pnl_pts": pnl, "pnl_pct": pnl / S0 * 100}


# ── strike selection per cycle ────────────────────────────────────────────────
def cycle_candidates(opts, ibov, r_cc):
    """For each expiry: first day at 25-35 DTE with traded legs.
    Returns list of dicts with straddle and (if available) strangle legs."""
    out = []
    for expiry, g in opts.groupby("maturity_date"):
        g = g.copy()
        g["dte"] = (expiry - g["refdate"]).dt.days
        g = g[(g["dte"] >= DTE_MIN) & (g["dte"] <= DTE_MAX)
              & (g["traded_contracts"] > 0) & (g["close"] > 0)]
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
            day = pd.Timestamp(day)
            r = float(r_cc.loc[day]) if day in r_cc.index else 0.10
            T = (expiry - day).days / 365.0
            F = S * np.exp(r * T)
            K_atm = min(common, key=lambda k: abs(k - F))
            if abs(K_atm - F) / F > 0.05:
                continue

            rec = {"entry": day, "expiry": pd.Timestamp(expiry),
                   "straddle": [(float(K_atm), float(calls.loc[K_atm]), "C"),
                                (float(K_atm), float(puts.loc[K_atm]), "P")]}

            # ATM IV proxy for delta targeting
            iv_atm = implied_vol(float(calls.loc[K_atm]), S, K_atm, T, r, "C")
            if not np.isnan(iv_atm):
                kc = [k for k in calls.index if k > F]
                kp = [k for k in puts.index if k < F]
                if kc and kp:
                    Kc = min(kc, key=lambda k: abs(
                        bs_delta(S, k, T, r, iv_atm, "C") - TARGET_DELTA))
                    Kp = min(kp, key=lambda k: abs(
                        bs_delta(S, k, T, r, iv_atm, "P") + TARGET_DELTA))
                    dc = bs_delta(S, Kc, T, r, iv_atm, "C")
                    dp = bs_delta(S, Kp, T, r, iv_atm, "P")
                    # accept if reasonably OTM (delta within 0.10-0.40)
                    if 0.10 <= dc <= 0.40 and -0.40 <= dp <= -0.10:
                        rec["strangle"] = [(float(Kc), float(calls.loc[Kc]), "C"),
                                           (float(Kp), float(puts.loc[Kp]), "P")]
            out.append(rec)
            break
    return out


def perf(t, label):
    t = t.dropna(subset=["pnl_scaled"])
    t = t[t["units"] > 0]
    if len(t) < 5:
        return f"  {label:<30} insufficient trades ({len(t)})"
    per_yr = len(t) / max((t["entry"].max() - t["entry"].min()).days / 365.25, 1)
    mu, sd = t["pnl_scaled"].mean(), t["pnl_scaled"].std()
    sharpe = mu / sd * np.sqrt(per_yr) if sd > 0 else np.nan
    cum = t["pnl_scaled"].cumsum()
    dd = (cum - cum.cummax()).min()
    return (f"  {label:<30} N={len(t):>4}  mean={mu:6.3f}%  hit={(t['pnl_scaled']>0).mean():5.1%}"
            f"  Sharpe={sharpe:5.2f}  worst={t['pnl_scaled'].min():7.2f}%  maxDD={dd:7.2f}%")


def main():
    print("=" * 78)
    print(" Short-vol v2 — straddle vs 25d strangle, fixed vs vega/signal sizing")
    print("=" * 78)
    opts, ibov, r_cc, signal = load_data()

    print("\n[1/3] Selecting cycles & strikes ...")
    cands = cycle_candidates(opts, ibov, r_cc)
    n_strangle = sum("strangle" in c for c in cands)
    print(f"  {len(cands)} cycles, strangle legs available on {n_strangle}")

    print("[2/3] Simulating ...")
    rows = []
    for c in cands:
        for struct in ("straddle", "strangle"):
            if struct not in c:
                continue
            res = simulate_structure(c["entry"], c["expiry"], c[struct], ibov, r_cc)
            if res:
                res["structure"] = struct
                rows.append(res)
    tr = pd.DataFrame(rows)

    # signal with 5-day ffill (as v1)
    sig_daily = signal.dropna().reindex(
        pd.date_range(signal.index.min(), signal.index.max())).ffill(limit=5)
    tr["signal"] = sig_daily.reindex(tr["entry"]).values

    # expanding percentile rank of signal at entry (strictly prior history)
    s_hist = signal.dropna()
    ranks = []
    for d, sv in zip(tr["entry"], tr["signal"]):
        prior = s_hist.loc[:d - pd.Timedelta(days=1)]
        ranks.append((prior < sv).mean() if len(prior) >= MIN_HISTORY
                     and not np.isnan(sv) else np.nan)
    tr["rank"] = ranks

    print("[3/3] Results\n")
    lines = ["Short-vol v2 — structures & sizing  (pnl % of index notional)",
             "=" * 78,
             f"Cycles: {len(cands)}  window {tr['entry'].min().date()} - "
             f"{tr['expiry'].max().date()}  costs as v1 (5% haircut)",
             ""]

    for struct in ("straddle", "strangle"):
        t = tr[tr["structure"] == struct].copy().sort_values("entry").reset_index(drop=True)
        vega_ref = t["vega"].median()

        # fixed 1x
        t["units"] = 1.0
        t["pnl_scaled"] = t["pnl_pct"]
        lines.append(perf(t, f"{struct}  fixed 1x (all)"))

        tb = t[t["signal"] > 0].copy()
        tb["units"] = 1.0
        tb["pnl_scaled"] = tb["pnl_pct"]
        lines.append(perf(tb, f"{struct}  fixed 1x, spread>0"))

        # vega-normalized, spread>0
        tv = t[t["signal"] > 0].copy()
        tv["units"] = vega_ref / tv["vega"]
        tv["pnl_scaled"] = tv["pnl_pct"] * tv["units"]
        lines.append(perf(tv, f"{struct}  vega-norm, spread>0"))

        # vega-normalized + signal-scaled
        ts_ = t.copy()
        w = 2 * (ts_["rank"] - 0.40) / 0.60
        ts_["units"] = np.where(ts_["rank"] >= 0.40, w, 0.0) * vega_ref / ts_["vega"]
        ts_["pnl_scaled"] = ts_["pnl_pct"] * ts_["units"]
        lines.append(perf(ts_, f"{struct}  vega+signal scale"))
        lines.append("")

        if struct == "straddle":
            tr_strad = ts_
        else:
            tr_strang = ts_

    tr.to_csv(RES / "short_vol_v2_trades.csv", index=False)
    report = "\n".join(lines)
    (RES / "short_vol_v2_report.txt").write_text(report)
    print(report)

    # plot: cumulative comparison
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")
    for ts_, lbl, c in ((tr_strad, "straddle vega+signal", BLUE),
                        (tr_strang, "strangle vega+signal", RED)):
        tt = ts_[ts_["units"] > 0].dropna(subset=["pnl_scaled"])
        ax.plot(tt["expiry"], tt["pnl_scaled"].cumsum(), lw=1.7, color=c, label=lbl)
    base = tr[(tr["structure"] == "straddle") & (tr["signal"] > 0)]
    ax.plot(base["expiry"], base["pnl_pct"].cumsum(), lw=1.2, color=GRAY,
            ls="--", label="v1 baseline (straddle 1x, spread>0)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.set_title("Short-vol v2 — cumulative P&L (% notional)", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(RES / "short_vol_v2_backtest.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RES/'short_vol_v2_trades.csv'}, {RES/'short_vol_v2_report.txt'}, "
          f"{RES/'short_vol_v2_backtest.png'}")


if __name__ == "__main__":
    main()
