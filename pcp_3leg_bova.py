"""
pcp_3leg_bova.py
================
Explicit 3-leg put-call parity arbitrage for BOVA11 (IBOV ETF) options.

Legs
----
  1. Buy BOVA call  (or sell — depending on signal)
  2. Sell BOVA put  (or buy)
  3. Sell (or buy) BOVA11 ETF spot as delta hedge

This extends the existing OLS-signal backtest (pcp_backtest.py) by
explicitly modelling the cost of the ETF hedge leg that was previously
ignored. For a truly hedged arb you MUST execute all three legs.

P&L structure (hold-to-expiry)
-------------------------------
  gross_pnl          = exec_edge  × n_contracts          [locked at entry via bid/ask]
  options_comm_rt    = 2 × B3_option_fee × (C_notional + P_notional)
                                                          [options round-trip B3 + brokerage]
  etf_comm_rt        = 2 × B3_etf_fee  × (BOVA11 × n)   [ETF round-trip B3 + brokerage]
  etf_spread_cost    = half_spread_BOVA11 × n             [one crossing at entry]
  net_pnl_3leg       = gross_pnl - options_comm_rt - etf_comm_rt - etf_spread_cost

Sizing constraints
------------------
  n = min(
      option_participation:  0.10 × min(c_vol, p_vol), cap 50 000,
      etf_participation:     0.05 × BOVA11_daily_vol,  cap 200 000
  )

B3 fee rates
------------
  Options (Bovespa, opções): emolumento 0.0325% + liquidação 0.0245% = 0.057%/leg
  ETF equity trades:         emolumento 0.0275% + liquidação 0.0275% = 0.055%/leg
  Brokerage (configurable):  0.10% per leg both instruments
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

BT_DETAIL_IN   = OUT_DIR / "pcp_backtest_detail.csv"   # from pcp_backtest.py
OUT_DETAIL     = OUT_DIR / "pcp_3leg_bova_detail.csv"
OUT_SUMMARY    = OUT_DIR / "pcp_3leg_bova_summary.csv"
OUT_REPORT     = OUT_DIR / "pcp_3leg_bova_report.txt"
OUT_CHART      = OUT_DIR / "pcp_3leg_bova_chart.png"

# ── Cost parameters ───────────────────────────────────────────────────────────
# Options legs (Bovespa segment)
B3_OPT_LEG      = 0.000570   # 0.057% per leg (emolumento + liquidação)
BROK_OPT_LEG    = 0.001000   # 0.10% per leg brokerage
OPT_FEE_LEG     = B3_OPT_LEG + BROK_OPT_LEG   # 0.157% per leg

# ETF leg (BOVA11 spot, equity segment)
B3_ETF_LEG      = 0.000550   # 0.055% per leg (emolumento 0.0275% + liquidação 0.0275%)
BROK_ETF_LEG    = 0.001000   # 0.10% per leg brokerage
ETF_FEE_LEG     = B3_ETF_LEG + BROK_ETF_LEG   # 0.155% per leg
ETF_FEE_RT      = ETF_FEE_LEG * 2             # 0.310% round-trip

# Sizing
OPT_PARTICIPATION  = 0.10     # max 10% of min(c_vol, p_vol)
OPT_MAX            = 50_000   # absolute cap per leg
ETF_PARTICIPATION  = 0.05     # max 5% of BOVA11 daily traded_contracts
ETF_MAX            = 200_000  # hard cap on ETF hedge

MIN_EXEC_EDGE      = 0.05     # minimum signal edge (BRL)
MAX_SPREAD_PCT     = 0.02     # filter out BOVA11 rows with spread > 2% (data artifacts)


def load_bova11_spot() -> pd.DataFrame:
    """Load BOVA11 ETF daily prices (close, bid, ask, volume) for all years."""
    frames = []
    for yr in range(2010, 2027):
        parq = DATA_DIR / f"year={yr}" / "part-0.parquet"
        if not parq.exists():
            continue
        try:
            df = pq.read_table(
                parq,
                columns=["refdate", "bdi_code", "symbol", "close",
                         "best_bid", "best_ask", "traded_contracts"]
            ).to_pandas()
            b11 = df[(df["bdi_code"] == 14) & (df["symbol"] == "BOVA11")].copy()
            b11["refdate"] = pd.to_datetime(b11["refdate"])
            frames.append(b11)
        except Exception:
            continue

    if not frames:
        raise RuntimeError("No BOVA11 data found in parquet files")

    out = pd.concat(frames, ignore_index=True)

    # Clean bid/ask: filter data artifacts (spread > 2% of close or negative)
    out["spread"] = out["best_ask"] - out["best_bid"]
    bad_spread = (out["spread"] < 0) | (out["spread"] > out["close"] * MAX_SPREAD_PCT)
    out.loc[bad_spread, ["best_bid", "best_ask", "spread"]] = np.nan
    # Fallback half-spread: use 0.05 BRL if bid/ask stale (typical end-of-day)
    out["half_spread"] = np.where(
        out["spread"].notna(),
        out["spread"] / 2,
        0.05  # conservative fallback: 5 cents on ~R$100+ ETF
    )
    return out[["refdate", "close", "best_bid", "best_ask", "traded_contracts",
                "half_spread"]].rename(columns={
                    "close": "etf_close",
                    "best_bid": "etf_bid",
                    "best_ask": "etf_ask",
                    "traded_contracts": "etf_vol",
                    "half_spread": "etf_half_spread",
                })


def apply_3leg_costs(sig: pd.DataFrame, bova11: pd.DataFrame) -> pd.DataFrame:
    """
    Given BOVA signals (already sized from options-only backtest),
    merge BOVA11 spot data and recompute full 3-leg net P&L.
    """
    sig = sig.merge(bova11, on="refdate", how="left")

    missing_etf = sig["etf_close"].isna()
    if missing_etf.any():
        print(f"  Warning: {missing_etf.sum()} signals have no BOVA11 spot data — dropped")
        sig = sig[~missing_etf].copy()

    n = sig["n_contracts"].values
    ee = sig["exec_edge"].values

    # ── 3-leg sizing: further constrain by ETF liquidity ──────────────────────
    etf_cap = np.minimum(
        sig["etf_vol"].fillna(0).values * ETF_PARTICIPATION,
        ETF_MAX
    ).astype(int)
    n_3leg = np.minimum(n, etf_cap)
    n_3leg = np.where(n_3leg < 1, 0, n_3leg)
    sig["n_3leg"] = n_3leg

    # Drop where sizing goes to 0
    sig = sig[sig["n_3leg"] >= 1].copy()
    n = sig["n_3leg"].values
    ee = sig["exec_edge"].values

    # ── P&L ──────────────────────────────────────────────────────────────────
    sig["gross_pnl_3l"] = ee * n

    # Options commission (round-trip): 2 × fee/leg × (C_notional + P_notional)
    c_prem = sig["c_mid"].fillna(sig["c_close"]).clip(lower=0.01).values
    p_prem = sig["p_mid"].fillna(sig["p_close"]).clip(lower=0.01).values
    opt_notional = (c_prem + p_prem) * n
    sig["opt_comm_rt"] = 2 * OPT_FEE_LEG * opt_notional  # 2×(entry+exit)

    # ETF commission (round-trip): entry + exit (at expiry)
    etf_notional = sig["etf_close"].values * n
    sig["etf_comm_rt"] = ETF_FEE_RT * etf_notional

    # ETF half-spread cost (one crossing at entry)
    sig["etf_spread_cost"] = sig["etf_half_spread"].values * n

    # Total costs
    sig["total_cost_3l"] = sig["opt_comm_rt"] + sig["etf_comm_rt"] + sig["etf_spread_cost"]

    # Net P&L
    sig["net_pnl_3l"] = sig["gross_pnl_3l"] - sig["total_cost_3l"]

    # Edge after all costs (per contract) for comparison
    sig["net_edge_3l"] = sig["net_pnl_3l"] / n

    return sig


def print_and_write(txt: str, fh):
    print(txt)
    fh.write(txt + "\n")


def run():
    print("Loading existing BOVA signals from backtest detail CSV...")
    all_bt = pd.read_csv(BT_DETAIL_IN, low_memory=False)
    all_bt["refdate"] = pd.to_datetime(all_bt["refdate"])

    bova = all_bt[all_bt["underlying_key"].str.contains("BOVA", na=False)].copy()
    print(f"  Total backtest signals:  {len(all_bt):,}")
    print(f"  BOVA signals:            {len(bova):,}")
    print()

    print("Loading BOVA11 spot data (all years)...")
    bova11 = load_bova11_spot()
    print(f"  BOVA11 trading days loaded: {len(bova11):,}")
    print(f"  Date range: {bova11['refdate'].min().date()} → {bova11['refdate'].max().date()}")
    print(f"  Mean price: R${bova11['etf_close'].mean():.2f}")
    print(f"  Mean daily volume: {bova11['etf_vol'].mean()/1e6:.1f}M contracts/day")
    print(f"  Mean half-spread: {bova11['etf_half_spread'].mean():.4f} BRL")
    print()

    print("Applying 3-leg costs...")
    res = apply_3leg_costs(bova, bova11)
    print(f"  Signals with full 3-leg data: {len(res):,}")
    print()

    # ── Save detail ───────────────────────────────────────────────────────────
    res.to_csv(OUT_DETAIL, index=False)
    print(f"Detail saved → {OUT_DETAIL}")

    # ── Build yearly summary ──────────────────────────────────────────────────
    res["year"] = pd.to_datetime(res["refdate"]).dt.year
    by_yr = res.groupby("year").agg(
        n_signals     = ("net_pnl_3l", "count"),
        gross_pnl     = ("gross_pnl_3l", "sum"),
        opt_comm      = ("opt_comm_rt", "sum"),
        etf_comm      = ("etf_comm_rt", "sum"),
        etf_spread    = ("etf_spread_cost", "sum"),
        total_cost    = ("total_cost_3l", "sum"),
        net_pnl       = ("net_pnl_3l", "sum"),
        pct_pos       = ("net_pnl_3l", lambda x: (x > 0).mean() * 100),
        mean_edge     = ("exec_edge", "mean"),
        mean_net_edge = ("net_edge_3l", "mean"),
        total_contracts = ("n_3leg", "sum"),
    )
    by_yr.to_csv(OUT_SUMMARY)
    print(f"Summary saved → {OUT_SUMMARY}")

    # ── Write report ──────────────────────────────────────────────────────────
    with open(OUT_REPORT, "w") as fh:
        W = lambda s: print_and_write(s, fh)

        W("=" * 70)
        W("3-LEG BOVA11 PCP ARBITRAGE BACKTEST")
        W("Signals: BOVA options (BDI 78/82)  |  Hedge: BOVA11 ETF (BDI 14)")
        W("=" * 70)
        W("")
        W("STRATEGY OVERVIEW")
        W("-" * 40)
        W("  Leg 1+2: Buy call / Sell put (or inverse) using OLS-detected PCP signal")
        W("  Leg 3:   Sell (or buy) BOVA11 ETF spot as delta-neutral hedge")
        W("  Horizon: Hold to expiry (options settle in cash vs BOVA11 price)")
        W("  Signal:  exec_edge = max(buy_edge, sell_edge) from bid/ask quotes")
        W("")
        W("FEE STRUCTURE (per round-trip trade)")
        W("-" * 40)
        W(f"  Options B3 fee:    {B3_OPT_LEG*100:.4f}% per leg  (emolumento + liquidação)")
        W(f"  Options brokerage: {BROK_OPT_LEG*100:.4f}% per leg")
        W(f"  Options total:     {OPT_FEE_LEG*200:.4f}% round-trip on premium notional")
        W(f"  ETF B3 fee:        {B3_ETF_LEG*100:.4f}% per leg  (emolumento + liquidação)")
        W(f"  ETF brokerage:     {BROK_ETF_LEG*100:.4f}% per leg")
        W(f"  ETF total:         {ETF_FEE_RT*100:.4f}% round-trip on spot notional")
        W(f"  ETF half-spread:   actual bid/ask (median ~0.03–0.05 BRL)")
        W("")
        W("SIZING")
        W("-" * 40)
        W(f"  Options: {OPT_PARTICIPATION*100:.0f}% of min(c_vol, p_vol), cap {OPT_MAX:,} contracts")
        W(f"  ETF:     {ETF_PARTICIPATION*100:.0f}% of BOVA11 daily vol, cap {ETF_MAX:,} contracts")
        W(f"  Final n: min(options_cap, ETF_cap)")
        W("")
        W("=" * 70)
        W("AGGREGATE RESULTS (all years)")
        W("=" * 70)
        W("")

        tot = by_yr.sum()
        W(f"  Total signals (with BOVA11 data):  {int(tot['n_signals']):,}")
        W(f"  Total contracts traded:            {int(tot['total_contracts']):,}")
        W(f"  Gross P&L:                         R$ {tot['gross_pnl']:>14,.2f}")
        W(f"  Options commissions (RT):          R$ {tot['opt_comm']:>14,.2f}")
        W(f"  ETF commissions (RT):              R$ {tot['etf_comm']:>14,.2f}")
        W(f"  ETF spread cost (entry):           R$ {tot['etf_spread']:>14,.2f}")
        W(f"  Total costs:                       R$ {tot['total_cost']:>14,.2f}")
        W(f"  Net P&L (3-leg):                   R$ {tot['net_pnl']:>14,.2f}")
        W(f"  Cost/Gross ratio:                  {tot['total_cost']/max(tot['gross_pnl'],1)*100:.1f}%")
        W(f"  Net/Gross ratio:                   {tot['net_pnl']/max(tot['gross_pnl'],1)*100:.1f}%")
        W("")
        W(f"  % signals profitable (net):        {(res['net_pnl_3l'] > 0).mean()*100:.1f}%")
        W(f"  Mean exec_edge (gross):            {res['exec_edge'].mean():.4f} BRL/share")
        W(f"  Mean net_edge (after all costs):   {res['net_edge_3l'].mean():.4f} BRL/share")
        W("")

        W("=" * 70)
        W("YEAR-BY-YEAR BREAKDOWN")
        W("=" * 70)
        W("")
        W(f"  {'Year':4}  {'N':>5}  {'Gross':>12}  {'OptComm':>10}  {'ETFComm':>10}  "
          f"{'ETFSprd':>8}  {'Net':>12}  {'%Pos':>5}  {'MeanEdge':>8}  {'NetEdge':>8}")
        W(f"  {'-'*4}  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}")
        for yr, r in by_yr.iterrows():
            W(f"  {yr:4d}  {int(r['n_signals']):>5,}  "
              f"R${r['gross_pnl']:>11,.2f}  "
              f"R${r['opt_comm']:>9,.2f}  "
              f"R${r['etf_comm']:>9,.2f}  "
              f"R${r['etf_spread']:>7,.2f}  "
              f"R${r['net_pnl']:>11,.2f}  "
              f"{r['pct_pos']:>5.1f}%  "
              f"{r['mean_edge']:>8.4f}  "
              f"{r['mean_net_edge']:>8.4f}")
        W("")

        # ── Comparison vs options-only backtest ───────────────────────────────
        W("=" * 70)
        W("COMPARISON: Options-only backtest vs 3-leg (with ETF hedge cost)")
        W("=" * 70)
        W("")

        # Pull from the original backtest for BOVA signals
        bova_orig = all_bt[all_bt["underlying_key"].str.contains("BOVA", na=False)].copy()
        bova_orig["year"] = pd.to_datetime(bova_orig["refdate"]).dt.year
        orig_by_yr = bova_orig.groupby("year").agg(
            n_orig   = ("net_pnl_rt", "count"),
            net_orig = ("net_pnl_rt", "sum"),
        )
        cmp = by_yr[["n_signals", "net_pnl"]].join(orig_by_yr, how="outer").fillna(0)

        W(f"  {'Year':4}  {'N_3leg':>7}  {'Net_3leg':>14}  {'N_orig':>7}  {'Net_orig':>14}  {'Diff':>14}  {'Impact%':>7}")
        W(f"  {'-'*4}  {'-'*7}  {'-'*14}  {'-'*7}  {'-'*14}  {'-'*14}  {'-'*7}")
        for yr, r in cmp.iterrows():
            diff = r["net_pnl"] - r["net_orig"]
            pct  = diff / max(abs(r["net_orig"]), 1) * 100
            W(f"  {yr:4d}  {int(r['n_signals']):>7,}  R${r['net_pnl']:>13,.2f}  "
              f"{int(r['n_orig']):>7,}  R${r['net_orig']:>13,.2f}  "
              f"R${diff:>13,.2f}  {pct:>+7.1f}%")

        W("")
        net_3leg_tot = by_yr["net_pnl"].sum()
        net_orig_tot = bova_orig["net_pnl_rt"].sum()
        W(f"  TOTAL  {int(tot['n_signals']):>7,}  R${net_3leg_tot:>13,.2f}  "
          f"{len(bova_orig):>7,}  R${net_orig_tot:>13,.2f}  "
          f"R${net_3leg_tot - net_orig_tot:>13,.2f}  "
          f"{(net_3leg_tot - net_orig_tot)/max(abs(net_orig_tot),1)*100:>+7.1f}%")
        W("")

        W("INTERPRETATION")
        W("-" * 40)
        etf_cost_total = tot["etf_comm"] + tot["etf_spread"]
        W(f"  Adding explicit ETF hedge costs R${etf_cost_total:,.0f} total")
        W(f"  = ETF commission:  R${tot['etf_comm']:,.0f}")
        W(f"  = ETF bid/ask:     R${tot['etf_spread']:,.0f}")
        etf_per_sig = etf_cost_total / max(tot["n_signals"], 1)
        W(f"  Average {etf_per_sig:.2f} BRL extra per signal (ETF hedge friction)")
        W("")
        W("  The 3-leg approach ADDS the hedge cost not captured in options-only P&L.")
        W("  In practice this friction is real — you must cross the BOVA11 spread")
        W("  and pay ETF commissions to neutralise your delta at entry.")
        W("")
        W("  Practical note: for buy(C-P) signals (long options spread, short BOVA11),")
        W("  BOVA11 short-selling requires the lending market. However, BOVA11 is one")
        W("  of the most borrowed ETFs in Brazil — typical borrow rate ~0.5-1.5% p.a.")
        W("  (not modelled here; minimal for short DTE arb). Alternatively, the")
        W("  sell(C-P) direction (short spread, long BOVA11 spot) avoids borrow entirely.")
        W("")

        # ── Signal direction breakdown ─────────────────────────────────────────
        W("=" * 70)
        W("SIGNAL DIRECTION BREAKDOWN")
        W("=" * 70)
        W("")
        for side in ["buy(C-P)", "sell(C-P)"]:
            sub = res[res["signal_side"] == side]
            W(f"  {side}:")
            W(f"    Count: {len(sub):,}  |  Net P&L: R${sub['net_pnl_3l'].sum():,.2f}"
              f"  |  Mean exec_edge: {sub['exec_edge'].mean():.4f}"
              f"  |  Mean net_edge: {sub['net_edge_3l'].mean():.4f}")
        W("")

        # ── Top 20 signals ─────────────────────────────────────────────────────
        W("TOP 20 SIGNALS BY NET P&L")
        W("-" * 40)
        top = res.nlargest(20, "net_pnl_3l")[
            ["refdate", "c_sym", "p_sym", "strike_price", "dte",
             "exec_edge", "n_3leg", "gross_pnl_3l", "total_cost_3l", "net_pnl_3l",
             "signal_side", "etf_close"]
        ]
        for _, row in top.iterrows():
            W(f"  {str(row['refdate'])[:10]}  {row['c_sym']}/{row['p_sym']}"
              f"  K={row['strike_price']:.1f}  DTE={int(row['dte']):3d}"
              f"  edge={row['exec_edge']:.4f}  n={int(row['n_3leg']):,}"
              f"  gross=R${row['gross_pnl_3l']:,.2f}  cost=R${row['total_cost_3l']:,.2f}"
              f"  net=R${row['net_pnl_3l']:,.2f}  [{row['signal_side']}]"
              f"  BOVA11=R${row['etf_close']:.2f}")

    print()
    print(f"Report saved → {OUT_REPORT}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("3-Leg BOVA11 PCP Arbitrage — Complete Cost Model", fontsize=14, fontweight="bold")

    # Panel 1: Annual net P&L (3-leg vs options-only)
    ax = axes[0, 0]
    bova_orig_by_yr = bova_orig.groupby("year")["net_pnl_rt"].sum()
    x = np.arange(len(by_yr))
    w = 0.35
    bars1 = ax.bar(x - w/2, by_yr["net_pnl"], w, label="3-Leg (with ETF hedge)", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x + w/2, [bova_orig_by_yr.get(yr, 0) for yr in by_yr.index], w,
                   label="Options-only (original BT)", color="#94a3b8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(by_yr.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Net P&L (BRL)")
    ax.set_title("Annual Net P&L: 3-Leg vs Options-Only")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    # Panel 2: Cost breakdown stacked bar (per year)
    ax = axes[0, 1]
    ax.bar(by_yr.index, by_yr["opt_comm"], label="Options comm RT", color="#dc2626", alpha=0.8)
    ax.bar(by_yr.index, by_yr["etf_comm"], bottom=by_yr["opt_comm"],
           label="ETF comm RT", color="#f97316", alpha=0.8)
    ax.bar(by_yr.index, by_yr["etf_spread"],
           bottom=by_yr["opt_comm"] + by_yr["etf_comm"],
           label="ETF spread cost", color="#fbbf24", alpha=0.8)
    ax.plot(by_yr.index, by_yr["gross_pnl"], "b-o", label="Gross P&L", linewidth=2, markersize=4)
    ax.set_title("Gross P&L vs Cost Breakdown by Year")
    ax.set_ylabel("BRL")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

    # Panel 3: Exec edge vs net edge distribution
    ax = axes[1, 0]
    ax.hist(res["exec_edge"].clip(upper=5), bins=60, alpha=0.6, color="#2563eb", label="Gross exec_edge", density=True)
    ax.hist(res["net_edge_3l"].clip(lower=-1, upper=5), bins=60, alpha=0.6, color="#16a34a", label="Net edge (3-leg)", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.axvline(res["exec_edge"].median(), color="#2563eb", linestyle=":", linewidth=1,
               label=f"Gross median {res['exec_edge'].median():.3f}")
    ax.axvline(res["net_edge_3l"].median(), color="#16a34a", linestyle=":", linewidth=1,
               label=f"Net median {res['net_edge_3l'].median():.3f}")
    ax.set_xlabel("Edge per contract (BRL)")
    ax.set_ylabel("Density")
    ax.set_title("Gross vs Net Edge Distribution")
    ax.legend(fontsize=8)

    # Panel 4: Cumulative net P&L over time
    ax = axes[1, 1]
    res_sorted = res.sort_values("refdate")
    cum_3l   = res_sorted["net_pnl_3l"].cumsum()
    orig_sorted = bova_orig.sort_values("refdate")
    cum_orig = orig_sorted["net_pnl_rt"].cumsum()
    ax.plot(res_sorted["refdate"], cum_3l, color="#2563eb", linewidth=1.5, label="3-Leg cumulative net P&L")
    ax.plot(orig_sorted["refdate"], cum_orig, color="#94a3b8", linewidth=1, linestyle="--",
            label="Options-only cumulative net P&L", alpha=0.7)
    ax.set_title("Cumulative Net P&L Over Time")
    ax.set_ylabel("Cumulative BRL")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout()
    fig.savefig(OUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved → {OUT_CHART}")
    print()
    print("Done.")


if __name__ == "__main__":
    run()
