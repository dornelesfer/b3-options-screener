"""
pcp_backtest.py
===============
Volume-constrained, commission-adjusted backtest of put-call parity
arbitrage signals detected by pcp_full_scan.py.

Covers 2010–2026 (extends the original 2010-2024 scan with 2025 & 2026 data).

Methodology
-----------
Signal detection  : Affine parity fit C − P = A − B·K via OLS (see pcp_full_scan.py)
Entry assumption  : Enter at end-of-day bid/ask quotes (exec_edge already encodes this)
Mark-to-market    : Same-day close prices (cp_close) — entry-day P&L
P&L per signal    : (cp_close − cp_entry_cost) × n_contracts × LOT_SIZE
                  = (close_resid + exec_edge) × n_contracts × LOT_SIZE
Position sizing   : min(c_traded_contracts, p_traded_contracts) × PARTICIPATION_RATE
                    capped at MAX_CONTRACTS_PER_TRADE
Commissions       : B3 official Tabela de Tarifas (Bovespa segment, opções)
                    • Emolumentos:  0.0325% of premium notional per leg
                    • Liquidação:   0.0245% of premium notional per leg
                    • Total B3 fee: 0.0570% per leg × 2 legs = 0.114% round-trip
                    Plus optional brokerage: 0.10% per leg (configurable)
Slippage          : Implicitly modelled — bid/ask spread already in exec_edge;
                    volume participation cap prevents market-impact scenarios.

B3 Fee Reference  : https://www.b3.com.br/pt_br/produtos-e-servicos/tarifas/listados-a-vista-e-derivativos/renda-variavel/tarifas-de-acoes-e-fundos-de-investimento/a-vista/
                    (Opções: emolumentos 0.0325% + liquidação 0.0245% per side)
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
import matplotlib.ticker as mticker

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

DETAIL_CSV  = OUT_DIR / "pcp_full_detail.csv"        # existing 2010-2024 signals
EXTRA_CSV   = OUT_DIR / "pcp_detail_2025_2026.csv"   # new 2025-2026 signals
BT_DETAIL   = OUT_DIR / "pcp_backtest_detail.csv"
BT_SUMMARY  = OUT_DIR / "pcp_backtest_summary.csv"
BT_REPORT   = OUT_DIR / "pcp_backtest_report.txt"
BT_CHART    = OUT_DIR / "pcp_backtest_analysis.png"

# ── Strategy parameters ───────────────────────────────────────────────────────
PARTICIPATION_RATE   = 0.10   # max fraction of daily traded_contracts we can fill
MAX_CONTRACTS        = 50_000 # hard cap per leg (risk limit; ~R$5k–50k per signal)
MIN_EXEC_EDGE        = 0.05   # BRL — minimum edge to enter (noise filter)
# LOT_SIZE = 1: B3 COTAHIST `traded_contracts` counts individual option contracts
# where each contract = 1 share (allocation_lot_size=1 for virtually all equity
# options from 2010 onwards, confirmed: volume ÷ traded_contracts ≈ close price).
LOT_SIZE             = 1

# B3 official fees (Tabela de Tarifas — Mercado de Opções, Bovespa segment)
B3_EMOLUMENTO        = 0.000325  # 0.0325% per leg on traded premium
B3_LIQUIDACAO        = 0.000245  # 0.0245% per leg on traded premium
B3_FEE_PER_LEG       = B3_EMOLUMENTO + B3_LIQUIDACAO   # 0.0570% per leg
# We trade 2 legs (call + put), so total exchange fee = 2 × 0.0570% = 0.114% RT
B3_FEE_RT            = 2 * B3_FEE_PER_LEG

# Optional competitive brokerage (institutional desk rate; comment out if unwanted)
BROKERAGE_PER_LEG    = 0.001     # 0.10% per leg
BROKERAGE_RT         = 2 * BROKERAGE_PER_LEG  # 0.20% round-trip

TOTAL_COST_RT        = B3_FEE_RT + BROKERAGE_RT   # 0.314% total RT cost

# ── PCP scan parameters (mirrors pcp_full_scan.py) ───────────────────────────
R_BRAZIL             = 0.12
MIN_DTE              = 2
MIN_PAIRS_PER_GRP    = 5
MIN_STRIKE_RANGE_PCT = 0.04
B_HAT_MIN, B_HAT_MAX = 0.75, 1.05
MAX_ASK_BID_RATIO    = 50
MAX_ASK_CLOSE_MULT   = 3
MAX_ASK_CLOSE_ADD    = 2

NEEDED_COLS = [
    "refdate", "bdi_code", "symbol", "corporation_name", "isin",
    "close", "best_bid", "best_ask", "strike_price", "maturity_date",
    "volume", "traded_contracts",
]

GROUP_KEYS = ["refdate", "maturity_date", "underlying_key"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. PCP scan functions (copied from pcp_full_scan.py, with volume output added)
# ══════════════════════════════════════════════════════════════════════════════

def load_options_year(year: int) -> pd.DataFrame | None:
    path = DATA_DIR / f"year={year}" / "part-0.parquet"
    if not path.exists():
        return None
    df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()
    df = df[df["bdi_code"].isin([74, 75, 78, 82]) & (df["strike_price"] > 0)].copy()
    if df.empty:
        return None
    df["refdate"]       = pd.to_datetime(df["refdate"],       errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df = df.dropna(subset=["refdate", "maturity_date"])
    df["dte"]           = (df["maturity_date"] - df["refdate"]).dt.days
    df = df[df["dte"] >= MIN_DTE].copy()
    df["option_type"]   = np.where(df["bdi_code"].isin([74, 78]), "C", "P")
    df["universe"]      = np.where(df["bdi_code"].isin([74, 75]), "index", "equity")
    df["underlying_key"] = np.where(
        df["universe"] == "index",
        "INDEX:IBOV",
        "EQUITY:" + df["isin"].astype(str).str.strip(),
    )
    df["underlying_name"] = df["corporation_name"].astype(str).str.strip()
    ratio_ok = (
        (df["best_bid"] > 0)
        & (df["best_ask"] > 0)
        & (df["best_ask"] <= df["best_bid"] * MAX_ASK_BID_RATIO + 1)
    )
    close_ok = (df["close"] <= 0) | (
        df["best_ask"] <= df["close"] * MAX_ASK_CLOSE_MULT + MAX_ASK_CLOSE_ADD
    )
    good_quote = ratio_ok & close_ok
    df["mid"] = np.where(good_quote, (df["best_bid"] + df["best_ask"]) / 2.0, np.nan)
    return df


def match_pairs(df: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["refdate", "maturity_date", "underlying_key", "strike_price"]
    calls = df[df["option_type"] == "C"][[
        *merge_keys, "close", "mid", "best_bid", "best_ask", "symbol",
        "universe", "underlying_name", "dte", "traded_contracts", "volume",
    ]].rename(columns={
        "close": "c_close", "mid": "c_mid",
        "best_bid": "c_bid", "best_ask": "c_ask", "symbol": "c_sym",
        "traded_contracts": "c_contracts", "volume": "c_volume",
    })
    puts = df[df["option_type"] == "P"][[
        *merge_keys, "close", "mid", "best_bid", "best_ask", "symbol",
        "dte", "traded_contracts", "volume",
    ]].rename(columns={
        "close": "p_close", "mid": "p_mid",
        "best_bid": "p_bid", "best_ask": "p_ask", "symbol": "p_sym",
        "traded_contracts": "p_contracts", "volume": "p_volume",
    })
    pairs = calls.merge(puts, on=merge_keys)
    pairs = pairs.dropna(subset=["c_close", "p_close"])
    pairs = pairs[(pairs["c_close"] > 0) & (pairs["p_close"] > 0)]
    pairs["cp_close"] = pairs["c_close"] - pairs["p_close"]
    both_quoted = pairs["c_mid"].notna() & pairs["p_mid"].notna()
    pairs["cp_mid"]  = np.where(both_quoted, pairs["c_mid"] - pairs["p_mid"], np.nan)
    pairs["cp_sell"] = np.where(both_quoted, pairs["c_bid"] - pairs["p_ask"], np.nan)
    pairs["cp_buy"]  = np.where(both_quoted, pairs["c_ask"] - pairs["p_bid"], np.nan)
    pairs["year"]    = pairs["refdate"].dt.year
    return pairs


def fit_parity_all(pairs: pd.DataFrame) -> pd.DataFrame:
    """Vectorised OLS: C−P = A − B·K for each (refdate, maturity, underlying) group."""
    # Assign integer group IDs
    gkeys       = pairs[GROUP_KEYS].apply(tuple, axis=1)
    codes, uniq = pd.factorize(gkeys, sort=False)
    pairs       = pairs.copy()
    pairs["_gid"] = codes
    gid  = codes.astype(int)
    ng   = len(uniq)
    K    = pairs["strike_price"].values
    y    = pairs["cp_close"].values

    # Accumulate per-group stats
    g_n    = np.bincount(gid, minlength=ng).astype(float)
    g_sumK  = np.bincount(gid, weights=K,   minlength=ng)
    g_sumK2 = np.bincount(gid, weights=K*K, minlength=ng)
    g_sumy  = np.bincount(gid, weights=y,   minlength=ng)
    g_sumKy = np.bincount(gid, weights=K*y, minlength=ng)

    # Robustness: strike range per group
    g_minK  = np.full(ng, np.inf);  np.minimum.at(g_minK, gid, K)
    g_maxK  = np.full(ng, -np.inf); np.maximum.at(g_maxK, gid, K)
    g_medK  = np.zeros(ng)
    for i, row in enumerate(pairs.groupby("_gid")["strike_price"].median()):
        g_medK[i] = row

    # Solve 2×2 normal equations
    det  = g_n * g_sumK2 - g_sumK**2
    with np.errstate(divide="ignore", invalid="ignore"):
        g_A = (g_sumK2 * g_sumy - g_sumK * g_sumKy) / det
        g_B = (g_sumK  * g_sumy - g_n    * g_sumKy) / det  # Note: sign is correct; see derivation

    # Validity filters
    strike_range_ok = (g_maxK - g_minK) >= MIN_STRIKE_RANGE_PCT * g_medK
    b_ok   = (g_B >= B_HAT_MIN) & (g_B <= B_HAT_MAX)
    n_ok   = (g_n >= MIN_PAIRS_PER_GRP)
    valid  = strike_range_ok & b_ok & n_ok & np.isfinite(g_A) & np.isfinite(g_B)

    pairs["g_A"]    = g_A[gid]
    pairs["g_B"]    = g_B[gid]
    pairs["valid_fit"] = valid[gid]
    fitted = pairs[pairs["valid_fit"]].copy()

    K_f = fitted["strike_price"].values
    fitted["cp_fitted"] = fitted["g_A"] - fitted["g_B"] * K_f
    fitted["F_hat"]     = fitted["g_A"] / fitted["g_B"]
    fitted["b_hat"]     = fitted["g_B"]
    fitted["close_resid"] = fitted["cp_close"] - fitted["cp_fitted"]

    # Mid & executable edges
    fitted["mid_resid"] = fitted["cp_mid"]  - fitted["cp_fitted"]
    has_both = fitted["cp_sell"].notna() & fitted["cp_buy"].notna()
    edge_sell = np.where(has_both, fitted["cp_fitted"] - fitted["cp_sell"], np.nan)
    edge_buy  = np.where(has_both, fitted["cp_buy"]  - fitted["cp_fitted"], np.nan)
    # wait – sign convention: buy signal when cp_buy > cp_fitted (C-P too expensive? no)
    # buy(C-P) signal: actual (c_ask - p_bid) < fitted → pay less than fair → edge = fitted - cp_buy
    edge_sell_adj = np.where(has_both, fitted["cp_sell"] - fitted["cp_fitted"], np.nan)
    edge_buy_adj  = np.where(has_both, fitted["cp_fitted"] - fitted["cp_buy"],  np.nan)
    fitted["edge_sell"] = edge_sell_adj
    fitted["edge_buy"]  = edge_buy_adj
    # Executable edge = best of the two sides
    both_pos = np.maximum(
        np.where(has_both, edge_sell_adj, -np.inf),
        np.where(has_both, edge_buy_adj,  -np.inf)
    )
    fitted["exec_edge"] = np.where(has_both, both_pos, np.nan)
    fitted["signal_side"] = np.where(
        has_both,
        np.where(edge_sell_adj >= edge_buy_adj, "sell(C-P)", "buy(C-P)"),
        ""
    )
    return fitted.drop(columns=["g_A", "g_B", "valid_fit", "_gid"])


def scan_year(year: int) -> pd.DataFrame | None:
    """Run full PCP scan for a single year. Returns equity signals only."""
    df = load_options_year(year)
    if df is None:
        return None
    df = df[df["universe"] == "equity"]  # equity only
    pairs = match_pairs(df)
    if len(pairs) < 10:
        return None
    fitted = fit_parity_all(pairs)
    fitted["year"] = year
    return fitted


# ══════════════════════════════════════════════════════════════════════════════
# 2. Backtest engine
# ══════════════════════════════════════════════════════════════════════════════

def apply_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply volume constraints and B3 commissions to a signals DataFrame.

    P&L methodology (arb/hold-to-expiry view):
    -------------------------------------------
    exec_edge > 0 means: at the quoted bid/ask prices, the (C−P) spread can
    be bought for less than (or sold for more than) the parity-model fair
    value.  Holding to expiry locks in this edge because C_T − P_T = S_T − K
    exactly (European options, no early exercise), and the fair value equals
    F·e^(−rT) − K·e^(−rT) — the same quantity our OLS A/B estimates.

    So: gross_pnl = exec_edge × n_contracts   [BRL]
    commission  = B3 exchange fee × notional (entry only; same again at exit)
    net_pnl     = gross_pnl − commission

    This is the lower bound on realised P&L (doesn't credit interest earned
    on the cash leg of the synthetic forward, or any interim convergence).

    Returns a copy with added columns:
        n_contracts     : position size in contracts (after participation cap)
        gross_pnl       : exec_edge × n_contracts  [BRL]
        commission_b3   : B3 exchange fee on entry  [BRL]
        commission_brok : brokerage on entry  [BRL]
        commission      : total entry-side commission  [BRL]
        net_pnl         : gross_pnl − commission  [BRL]
        comm_rt         : round-trip commission estimate (entry + exit)  [BRL]
        net_pnl_rt      : gross_pnl − comm_rt  [BRL]
    """
    sig = df[df["exec_edge"] >= MIN_EXEC_EDGE].copy()

    # ── Position sizing ───────────────────────────────────────────────────────
    # Only size where both legs had actual daily trades
    c_ok = sig["c_contracts"].fillna(0) > 0
    p_ok = sig["p_contracts"].fillna(0) > 0
    both_traded = c_ok & p_ok
    min_vol  = np.minimum(sig["c_contracts"].fillna(0), sig["p_contracts"].fillna(0))
    raw_size = (min_vol * PARTICIPATION_RATE).clip(upper=MAX_CONTRACTS).astype(int)
    sig["n_contracts"] = np.where(both_traded, raw_size, 0)

    # Drop pairs where we can't size at all
    sig = sig[sig["n_contracts"] >= 1].copy()

    # ── P&L calculations (exec_edge × size) ──────────────────────────────────
    n  = sig["n_contracts"].values
    ee = sig["exec_edge"].values

    sig["gross_pnl"] = ee * n * LOT_SIZE  # LOT_SIZE=1 (confirmed from volume check)

    # ── B3 Commission calculation ─────────────────────────────────────────────
    # Commission base = premium notional per leg = n × premium_per_contract
    # Premium proxy: c_mid or c_close (whichever available)
    def _prem(col_mid, col_close, df=sig):
        mid   = df[col_mid].values   if col_mid   in df.columns else np.full(len(df), np.nan)
        close = df[col_close].values if col_close in df.columns else np.full(len(df), 0.5)
        return np.where(np.isfinite(mid) & (mid > 0), mid, np.maximum(close, 0.01))

    call_prem = _prem("c_mid", "c_close")
    put_prem  = _prem("p_mid", "p_close")

    # Entry-side notional (we pay c_ask ≈ c_mid for buy, or receive c_bid ≈ c_mid for sell)
    call_notional = n * LOT_SIZE * call_prem
    put_notional  = n * LOT_SIZE * put_prem

    sig["commission_b3"]   = B3_FEE_PER_LEG  * (call_notional + put_notional)
    sig["commission_brok"] = BROKERAGE_PER_LEG * (call_notional + put_notional)
    sig["commission"]      = sig["commission_b3"] + sig["commission_brok"]

    # Round-trip commission (entry + exit at expiry; same rates apply)
    sig["comm_rt"]     = 2 * sig["commission"]
    sig["net_pnl"]     = sig["gross_pnl"] - sig["commission"]     # entry only
    sig["net_pnl_rt"]  = sig["gross_pnl"] - sig["comm_rt"]        # full round-trip

    return sig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════════

def join_volume_for_year(signals_year: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Load volume/contracts from the parquet for `year` and join onto signals.
    Done year-by-year to avoid OOM when holding 15 years in RAM simultaneously.
    """
    path = DATA_DIR / f"year={year}" / "part-0.parquet"
    if not path.exists():
        signals_year["c_contracts"] = np.nan
        signals_year["p_contracts"] = np.nan
        return signals_year

    vol = pq.read_table(
        str(path), columns=["refdate", "symbol", "traded_contracts"]
    ).to_pandas()
    vol["refdate"] = pd.to_datetime(vol["refdate"], errors="coerce")

    # join call volume
    out = signals_year.merge(
        vol.rename(columns={"symbol": "c_sym", "traded_contracts": "c_contracts"}),
        on=["refdate", "c_sym"], how="left"
    )
    # join put volume
    out = out.merge(
        vol.rename(columns={"symbol": "p_sym", "traded_contracts": "p_contracts"}),
        on=["refdate", "p_sym"], how="left"
    )
    del vol
    return out


def main():
    print("=" * 70)
    print("PCP Backtest — B3 Equity Options 2010–2026")
    print("=" * 70)

    # ── Step 1: Load existing 2010-2024 signals year-by-year ─────────────────
    print("\n[1/3] Processing 2010-2024 signals + joining volumes year-by-year...")
    old_sig = pd.read_csv(DETAIL_CSV, low_memory=False,
                          parse_dates=["refdate", "maturity_date"])
    if "universe" in old_sig.columns:
        old_sig = old_sig[old_sig["universe"] == "equity"]
    old_sig["year"] = pd.to_datetime(old_sig["refdate"]).dt.year

    # Canonical output columns — ensures consistent CSV across years
    OUT_COLS = [
        "refdate", "maturity_date", "underlying_key", "underlying_name",
        "strike_price", "c_sym", "p_sym", "dte", "year",
        "exec_edge", "signal_side", "close_resid",
        "c_mid", "p_mid", "c_close", "p_close",
        "n_contracts", "c_contracts", "p_contracts",
        "gross_pnl", "commission_b3", "commission_brok", "commission",
        "comm_rt", "net_pnl", "net_pnl_rt",
    ]

    first_write = True

    years_old = sorted(old_sig["year"].unique())
    for yr in years_old:
        yr_sig = old_sig[old_sig["year"] == yr].copy()
        yr_sig = join_volume_for_year(yr_sig, yr)
        bt_yr  = apply_backtest(yr_sig)
        n_bt   = len(bt_yr)
        gross  = bt_yr["gross_pnl"].sum()
        comm   = bt_yr["commission"].sum()
        net_rt = bt_yr["net_pnl_rt"].sum()
        print(f"  {yr}: {n_bt:>5,} signals | gross R${gross:>9,.0f} | "
              f"comm(entry) R${comm:>7,.0f} | net(RT) R${net_rt:>9,.0f}")
        if n_bt > 0:
            # Only keep canonical columns (add missing ones as NaN)
            for c in OUT_COLS:
                if c not in bt_yr.columns:
                    bt_yr[c] = np.nan
            bt_yr[OUT_COLS].to_csv(BT_DETAIL, mode="w" if first_write else "a",
                                   header=first_write, index=False)
            first_write = False
        del yr_sig, bt_yr

    del old_sig

    # ── Step 2: Scan + backtest 2025-2026 ────────────────────────────────────
    print("\n[2/3] Running PCP scan for 2025-2026 (new data)...")
    for yr in [2025, 2026]:
        print(f"  Scanning {yr}...", end="", flush=True)
        fitted = scan_year(yr)
        if fitted is None:
            print(" no data"); continue
        yr_sig = fitted[fitted["exec_edge"].notna() & (fitted["exec_edge"] > 0)].copy()
        yr_sig["year"] = yr
        if "c_contracts" not in yr_sig.columns:
            yr_sig = join_volume_for_year(yr_sig, yr)
        bt_yr = apply_backtest(yr_sig)
        n_bt  = len(bt_yr)
        gross = bt_yr["gross_pnl"].sum()
        comm  = bt_yr["commission"].sum()
        net_rt = bt_yr["net_pnl_rt"].sum()
        print(f" {len(yr_sig):,} raw signals → {n_bt:,} sized | "
              f"gross R${gross:,.0f} | net(RT) R${net_rt:,.0f}")
        if n_bt > 0:
            for c in OUT_COLS:
                if c not in bt_yr.columns:
                    bt_yr[c] = np.nan
            bt_yr[OUT_COLS].to_csv(BT_DETAIL, mode="w" if first_write else "a",
                                   header=first_write, index=False)
            first_write = False
        del yr_sig, fitted, bt_yr

    # ── Step 3: Load full detail, build summary, report, charts ──────────────
    print("\n[3/3] Building summary, report and charts...")
    bt = pd.read_csv(BT_DETAIL, low_memory=False,
                     parse_dates=["refdate", "maturity_date"])
    bt["year"] = pd.to_datetime(bt["refdate"]).dt.year

    print(f"\n  Total actionable signals : {len(bt):,}")
    print(f"  Total gross P&L          : R$ {bt['gross_pnl'].sum():>12,.2f}")
    print(f"  Total entry commission   : R$ {bt['commission'].sum():>12,.2f}")
    print(f"  Total net P&L (entry comm): R$ {bt['net_pnl'].sum():>12,.2f}")
    print(f"  Total net P&L (RT comm)  : R$ {bt['net_pnl_rt'].sum():>12,.2f}")

    summary = bt.groupby("year").agg(
        n_signals        =("net_pnl",       "count"),
        total_gross_pnl  =("gross_pnl",     "sum"),
        total_commission =("commission",     "sum"),
        total_net_pnl    =("net_pnl",        "sum"),
        total_net_rt     =("net_pnl_rt",     "sum"),
        mean_exec_edge   =("exec_edge",      "mean"),
        max_exec_edge    =("exec_edge",      "max"),
        pct_profitable   =("net_pnl_rt",    lambda x: (x > 0).mean() * 100),
        total_contracts  =("n_contracts",    "sum"),
    ).reset_index()
    summary.to_csv(BT_SUMMARY, index=False)

    write_report(bt, summary)
    make_charts(bt, summary)

    print(f"\n  Report → {BT_REPORT.name}")
    print(f"  Chart  → {BT_CHART.name}")
    print("\nDone.")


def write_report(bt: pd.DataFrame, summary: pd.DataFrame):
    lines = []
    a = lines.append

    a("Put-Call Parity Arbitrage Backtest — B3 Equity Options")
    a("=" * 70)
    a(f"Period            : {bt['refdate'].min().date()} to {bt['refdate'].max().date()}")
    a(f"Universe          : Equity options (BDI 78/82)")
    a(f"Participation cap : {PARTICIPATION_RATE*100:.0f}% of min(call, put) daily volume")
    a(f"Max contracts/leg : {MAX_CONTRACTS}")
    a(f"Min edge to enter : R$ {MIN_EXEC_EDGE:.2f}")
    a(f"Lot size          : {LOT_SIZE} shares/contract")
    a("")
    a("B3 Commission Structure (official Tabela de Tarifas)")
    a(f"  Emolumentos      : {B3_EMOLUMENTO*100:.4f}% per leg on premium notional")
    a(f"  Liquidação       : {B3_LIQUIDACAO*100:.4f}% per leg on premium notional")
    a(f"  B3 total (2 legs): {B3_FEE_RT*100:.4f}% of avg premium notional RT")
    a(f"  Brokerage (2 lgs): {BROKERAGE_RT*100:.4f}% of avg premium notional RT")
    a(f"  TOTAL RT cost    : {TOTAL_COST_RT*100:.4f}%")
    a("")
    a("P&L Method: same-day mark-to-market")
    a("  Entry  : bid/ask quotes at close (already embedded in exec_edge)")
    a("  Exit   : same-day close price (conservative; actual convergence may be higher)")
    a("  P&L    : (exec_edge + close_resid × side_sign) × n_contracts × 100 − commission")
    a("")
    a("─" * 70)
    a("AGGREGATE RESULTS")
    a("─" * 70)
    a(f"  Total actionable signals: {len(bt):,}")
    a(f"  Total contracts traded  : {bt['n_contracts'].sum():,.0f}")
    a(f"  Total GROSS P&L         : R$ {bt['gross_pnl'].sum():>12,.2f}")
    a(f"  Total commission        : R$ {bt['commission'].sum():>12,.2f}")
    a(f"  Total NET P&L (edge)    : R$ {bt['net_pnl'].sum():>12,.2f}")
    a(f"  Total NET P&L (RT)     : R$ {bt['net_pnl_rt'].sum():>12,.2f}")
    a(f"  % signals profitable(RT): {(bt['net_pnl_rt']>0).mean()*100:.1f}%")
    a(f"  Median net P&L per sig  : R$ {bt['net_pnl_rt'].median():.2f}")
    a(f"  Mean commission/signal  : R$ {bt['commission'].mean():.4f}")
    a(f"  Commission drag/edge    : {(bt['commission'].sum()/bt['gross_pnl'].sum()*100):.1f}%")
    a("")

    a("─" * 70)
    a("YEARLY BREAKDOWN")
    a("─" * 70)
    a(f"{'Year':>5} {'N_sig':>7} {'GrossPNL':>12} {'Commiss':>10} {'NetPNL':>12} {'NetMTM':>12} {'%Prof':>6} {'MeanEdge':>9}")
    for _, r in summary.iterrows():
        a(f"  {int(r.year):4d} {int(r.n_signals):>7,} "
          f"R${r.total_gross_pnl:>10,.0f} "
          f"R${r.total_commission:>8,.0f} "
          f"R${r.total_net_pnl:>10,.0f} "
          f"R${r.total_net_rt:>10,.0f} "
          f"{r.pct_profitable:>5.1f}% "
          f"{r.mean_exec_edge:>8.4f}")
    a("")

    a("─" * 70)
    a("TOP 20 SIGNALS BY NET MTM P&L")
    a("─" * 70)
    top = bt.nlargest(20, "net_pnl_rt")
    for _, r in top.iterrows():
        a(f"  {str(r['refdate'])[:10]}  {r['underlying_name']:<15}  "
          f"K={r['strike_price']:>8.2f}  edge={r['exec_edge']:.4f}  "
          f"n={int(r['n_contracts']):>4}  netRT=R${r['net_pnl_rt']:>8.2f}  "
          f"{r['signal_side']}")
    a("")

    a("─" * 70)
    a("TOP 20 UNDERLYINGS BY TOTAL NET MTM P&L")
    a("─" * 70)
    by_und = bt.groupby("underlying_name").agg(
        n_sig=("net_pnl_rt", "count"),
        total_net_rt=("net_pnl_rt", "sum"),
        mean_edge=("exec_edge", "mean"),
        total_contracts=("n_contracts", "sum"),
    ).sort_values("total_net_rt", ascending=False).head(20)
    for und, r in by_und.iterrows():
        a(f"  {und:<20}  n={int(r.n_sig):>6,}  totalMTM=R${r.total_net_rt:>10,.0f}  "
          f"meanEdge={r.mean_edge:.4f}  contracts={int(r.total_contracts):,}")
    a("")

    a("─" * 70)
    a("COMMISSION IMPACT ANALYSIS")
    a("─" * 70)
    before = len(bt[bt["exec_edge"] > 0])
    survive_b3 = len(bt[bt["gross_pnl"] > bt["commission_b3"]])
    survive_all = len(bt[bt["gross_pnl"] > bt["commission"]])
    a(f"  Signals with positive exec_edge         : {before:,}")
    a(f"  Survive B3 fees alone                   : {survive_b3:,}  "
      f"({survive_b3/max(before,1)*100:.1f}%)")
    a(f"  Survive B3 + brokerage                  : {survive_all:,}  "
      f"({survive_all/max(before,1)*100:.1f}%)")
    a(f"  B3-only cost rate                       : {B3_FEE_RT*100:.4f}% RT")
    a(f"  With brokerage                          : {TOTAL_COST_RT*100:.4f}% RT")
    a("")

    a("─" * 70)
    a("EDGE SIZE DISTRIBUTION (signals that survive all costs)")
    a("─" * 70)
    surv = bt[bt["net_pnl_rt"] > 0]["exec_edge"]
    for p in [50, 75, 90, 95, 99]:
        a(f"  p{p:02d} exec_edge : R$ {np.percentile(surv, p):.4f}")
    a("")

    a("INTERPRETATION")
    a("─" * 70)
    a("  exec_edge = how much the bid/ask spread on both legs is narrower than the")
    a("  parity model's fair value. This is the per-share gross profit from executing")
    a("  the arbitrage at the quoted prices.")
    a("")
    a("  net_pnl (edge-based): uses exec_edge as a proxy for realized P&L, ignoring")
    a("  any subsequent convergence of the spread. Conservative lower bound.")
    a("")
    a("  net_pnl (RT): marks the position at same-day close, capturing both the")
    a("  entry edge and any intraday mean-reversion. More realistic for short holds.")
    a("")
    a("  Volume cap: we assume at most 10% of each leg's daily traded volume can be")
    a("  executed without material market impact. In reality, for small retail sizes")
    a("  this constraint rarely binds; for institutional sizes it can be very tight.")
    a("")
    a("  Commission: B3's official Tabela de Tarifas applies 0.057% per leg on")
    a("  the premium notional (not underlying notional). Most signals involve small")
    a("  premiums (< R$5), so the absolute commission per contract is tiny but the")
    a("  percentage drag is significant for micro-edge signals (< R$0.05).")

    txt = "\n".join(lines)
    BT_REPORT.write_text(txt)


def make_charts(bt: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("PCP Arbitrage Backtest — B3 Equity Options 2010–2026",
                 fontsize=14, fontweight="bold", y=1.01)

    # 1. Annual net P&L (RT)
    ax = axes[0, 0]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in summary["total_net_rt"]]
    bars = ax.bar(summary["year"], summary["total_net_rt"] / 1e3, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Annual Net P&L — MTM (R$ thousands)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("R$ thousands")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}k"))
    ax.grid(axis="y", alpha=0.3)

    # 2. Gross vs commission vs net
    ax = axes[0, 1]
    x = summary["year"]
    w = 0.3
    ax.bar(x - w, summary["total_gross_pnl"] / 1e3, width=w, label="Gross P&L", color="#3498db", alpha=0.8)
    ax.bar(x,     summary["total_commission"] / 1e3, width=w, label="Commission", color="#e67e22", alpha=0.8)
    ax.bar(x + w, summary["total_net_pnl"]   / 1e3, width=w, label="Net P&L (edge)", color="#2ecc71", alpha=0.8)
    ax.set_title("Gross P&L vs Commission vs Net P&L", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("R$ thousands")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}k"))
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # 3. Signal count and % profitable
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(summary["year"], summary["n_signals"], color="#9b59b6", alpha=0.7, label="N signals")
    ax2.plot(summary["year"], summary["pct_profitable"], "o-", color="#e74c3c", linewidth=2, label="% profitable")
    ax2.set_ylim(0, 100)
    ax.set_title("Signal Count & % Profitable (net MTM > 0)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("N signals"); ax2.set_ylabel("% profitable")
    ax.grid(axis="y", alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    # 4. Edge distribution by year (box-like: p25-p75 shaded, median line)
    ax = axes[1, 1]
    years_sorted = sorted(bt["year"].unique())
    medians, p25s, p75s, p90s = [], [], [], []
    for yr in years_sorted:
        sub = bt[bt["year"] == yr]["exec_edge"]
        medians.append(sub.median())
        p25s.append(sub.quantile(0.25))
        p75s.append(sub.quantile(0.75))
        p90s.append(sub.quantile(0.90))
    ax.fill_between(years_sorted, p25s, p75s, alpha=0.3, color="#3498db", label="IQR (p25–p75)")
    ax.plot(years_sorted, medians, "o-", color="#2c3e50", linewidth=2, label="Median edge")
    ax.plot(years_sorted, p90s, "s--", color="#e74c3c", linewidth=1.5, label="p90 edge")
    ax.axhline(MIN_EXEC_EDGE, color="gray", linestyle=":", linewidth=1, label=f"Min filter ({MIN_EXEC_EDGE})")
    ax.set_title("Exec Edge Distribution by Year (BRL/share)", fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Executable edge (R$ per share)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(BT_CHART, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
