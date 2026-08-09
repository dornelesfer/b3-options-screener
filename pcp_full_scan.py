"""
pcp_full_scan.py
================
Comprehensive put-call parity arbitrage scan across all B3 options.

Fixes vs previous version:
  - Filters out sentinel ask values (e.g., 10000) via ask/bid ratio check
  - Fully vectorised: merge → fit → edge in pandas, no Python row loops
  - Annual + aggregate outputs with visualisations

Method
------
For each (refdate, maturity, underlying) group with >= 3 valid call/put pairs:
  1. Fit affine parity line via OLS:  C - P = A - B*K  (≥3 pairs needed)
     A ≈ F*exp(-rT),  B ≈ exp(-rT)  →  F_hat = A/B
  2. Compute residual per pair:  mid_resid = (C-P)_mid - fitted
  3. Compute executable edge:
       edge_sell = (C_bid - P_ask) - fitted   (sell C-P)
       edge_buy  = fitted - (C_ask - P_bid)   (buy  C-P)
       exec_edge = max(edge_sell, edge_buy)

Signals with exec_edge > 0 are theoretically executable (bid/ask spread is
wider than the fitted parity deviation).  In practice, transaction costs
(~0.1-0.3% round-trip on B3) further reduce profitability.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR  = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

R_BRAZIL           = 0.12   # approximate CDI / Selic (for discounting K)
MIN_DTE            = 2      # minimum days-to-expiry
MIN_PAIRS_PER_GRP    = 5      # minimum call-put pairs to fit parity line
MIN_STRIKE_RANGE_PCT = 0.04   # require (max_K - min_K) / median_K >= 4%
B_HAT_MIN            = 0.75   # plausible lower bound for exp(-rT) estimate
B_HAT_MAX            = 1.05   # plausible upper bound (small cushion for noise)
MAX_ASK_BID_RATIO    = 50     # filter out best_ask > bid * MAX_ASK_BID_RATIO
MAX_ASK_CLOSE_MULT   = 3      # filter out best_ask > close * mult + 2
MAX_ASK_CLOSE_ADD    = 2      # (catches sentinels like 10000, 192, 11 etc.)

NEEDED_COLS = [
    "refdate", "bdi_code", "symbol", "corporation_name", "isin",
    "close", "best_bid", "best_ask", "strike_price", "maturity_date",
    "volume", "traded_contracts",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_options_year(year: int) -> pd.DataFrame | None:
    path = DATA_DIR / f"year={year}" / "part-0.parquet"
    if not path.exists():
        return None

    df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()

    # Keep only option rows with valid strikes
    df = df[df["bdi_code"].isin([74, 75, 78, 82]) & (df["strike_price"] > 0)].copy()
    if df.empty:
        return None

    df["refdate"]      = pd.to_datetime(df["refdate"],      errors="coerce")
    df["maturity_date"]= pd.to_datetime(df["maturity_date"],errors="coerce")
    df = df.dropna(subset=["refdate", "maturity_date"])

    df["dte"]          = (df["maturity_date"] - df["refdate"]).dt.days
    df               = df[df["dte"] >= MIN_DTE].copy()

    df["option_type"]  = np.where(df["bdi_code"].isin([74, 78]), "C", "P")
    df["universe"]     = np.where(df["bdi_code"].isin([74, 75]), "index", "equity")
    df["underlying_key"] = np.where(
        df["universe"] == "index",
        "INDEX:IBOV",
        "EQUITY:" + df["isin"].astype(str).str.strip(),
    )
    df["underlying_name"] = df["corporation_name"].astype(str).str.strip()

    # Quote quality: require both sides > 0 AND no sentinel values
    ratio_ok = (
        (df["best_bid"] > 0)
        & (df["best_ask"] > 0)
        & (df["best_ask"] <= df["best_bid"] * MAX_ASK_BID_RATIO + 1)
    )
    # Secondary filter: ask shouldn't exceed MAX_ASK_CLOSE_MULT × close + add
    # (only applied where close > 0 to avoid filtering legit 0-close options)
    close_ok = (df["close"] <= 0) | (
        df["best_ask"] <= df["close"] * MAX_ASK_CLOSE_MULT + MAX_ASK_CLOSE_ADD
    )
    good_quote = ratio_ok & close_ok
    df["mid"] = np.where(good_quote, (df["best_bid"] + df["best_ask"]) / 2.0, np.nan)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Vectorised matching & parity fitting
# ──────────────────────────────────────────────────────────────────────────────

def match_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join calls with puts on (refdate, maturity, underlying, strike).

    Key design choice
    -----------------
    The parity LINE is fitted with CLOSE prices (last-traded), which are
    more reliable than bid/ask mids when market-maker asks are stale or lazy.
    Executable edges are still computed from bid/ask quotes.
    """
    merge_keys = ["refdate", "maturity_date", "underlying_key", "strike_price"]

    calls = df[df["option_type"] == "C"][[
        *merge_keys, "close", "mid", "best_bid", "best_ask", "symbol",
        "universe", "underlying_name", "dte"
    ]].rename(columns={
        "close": "c_close", "mid": "c_mid",
        "best_bid": "c_bid", "best_ask": "c_ask", "symbol": "c_sym",
    })

    puts = df[df["option_type"] == "P"][[
        *merge_keys, "close", "mid", "best_bid", "best_ask", "symbol"
    ]].rename(columns={
        "close": "p_close", "mid": "p_mid",
        "best_bid": "p_bid", "best_ask": "p_ask", "symbol": "p_sym",
    })

    pairs = calls.merge(puts, on=merge_keys)

    # For fitting: require both close prices > 0
    pairs = pairs[(pairs["c_close"] > 0) & (pairs["p_close"] > 0)].copy()

    # cp_close: used for OLS parity line fitting (close prices, actual trades)
    pairs["cp_close"] = pairs["c_close"] - pairs["p_close"]

    # cp_mid: for display / comparison (may be NaN if quotes are bad)
    pairs["cp_mid"] = np.where(
        pairs["c_mid"].notna() & pairs["p_mid"].notna(),
        pairs["c_mid"] - pairs["p_mid"],
        np.nan,
    )

    # Executable spreads (both bids/asks required)
    both_quoted = (
        (pairs["c_bid"] > 0) & (pairs["c_ask"] > 0)
        & (pairs["p_bid"] > 0) & (pairs["p_ask"] > 0)
    )
    pairs["cp_sell"] = np.where(both_quoted, pairs["c_bid"] - pairs["p_ask"], np.nan)
    pairs["cp_buy"]  = np.where(both_quoted, pairs["c_ask"] - pairs["p_bid"], np.nan)

    return pairs


GROUP_KEYS = ["refdate", "maturity_date", "underlying_key"]


def fit_parity_all(pairs: pd.DataFrame) -> pd.DataFrame:
    """Fully vectorised OLS of C-P = A - B*K per (date, maturity, underlying).

    Uses CLOSE prices for fitting (robust vs stale bid/ask).
    Computes group membership, drops groups < MIN_PAIRS_PER_GRP, then solves
    all 2x2 normal equations simultaneously via grouped sums.
    """
    df = pairs.copy()
    K  = df["strike_price"].astype(float)
    y  = df["cp_close"].astype(float)   # use close for parity fitting

    # Encode groups as integer IDs for efficient aggregation
    group_codes, group_keys_df = pd.factorize(
        df[GROUP_KEYS].apply(tuple, axis=1)
    )
    df["_gid"] = group_codes

    # Count per group; keep only groups with enough pairs
    cnt = df.groupby("_gid")["_gid"].transform("count")
    df  = df[cnt >= MIN_PAIRS_PER_GRP].copy()
    K   = df["strike_price"].astype(float).values
    y   = df["cp_close"].astype(float).values
    gid = df["_gid"].values

    # Per-group sums for OLS normal equations:  min ||[1, -K]*[A;B] - y||²
    # X = [1, -K]  →  X'X = [[n, -ΣK], [-ΣK, ΣK²]]
    #                 X'y = [Σy, -ΣKy]
    ng = gid.max() + 1
    g_n    = np.bincount(gid, minlength=ng).astype(float)
    g_sumK  = np.bincount(gid, weights=K,    minlength=ng)
    g_sumK2 = np.bincount(gid, weights=K*K,  minlength=ng)
    g_sumy  = np.bincount(gid, weights=y,    minlength=ng)
    g_sumKy = np.bincount(gid, weights=K*y,  minlength=ng)
    g_minK  = np.zeros(ng);  np.minimum.at(g_minK,  gid, K)
    g_maxK  = np.zeros(ng);  np.maximum.at(g_maxK,  gid, K)
    g_medK  = g_sumK / np.maximum(g_n, 1)          # mean K (proxy for median)

    # Solve 2×2 system per group in one vectorised pass
    # Model: y = A - B*K  →  X = [1, -K]
    # X'X = [[n, -ΣK], [-ΣK, ΣK²]]
    # X'y = [Σy, -Σ(Ky)]
    # Solution:  [A, B] = (X'X)^{-1} X'y
    # A = (ΣK²·Σy - ΣK·Σ(Ky)) / det
    # B = (ΣK·Σy  - n·Σ(Ky))  / det
    # where det = n·ΣK² - (ΣK)²
    det = g_n * g_sumK2 - g_sumK ** 2
    det = np.where(np.abs(det) < 1e-10, np.nan, det)

    g_A = (g_sumK2 * g_sumy - g_sumK * g_sumKy) / det
    g_B = (g_sumK  * g_sumy - g_n    * g_sumKy) / det

    # Validity checks:
    #  1. Strike range must be >= MIN_STRIKE_RANGE_PCT of mean-K
    #  2. B_hat must be in plausible discount-factor range [B_HAT_MIN, B_HAT_MAX]
    strike_range_ok = (g_maxK - g_minK) >= MIN_STRIKE_RANGE_PCT * g_medK
    b_ok = (g_B >= B_HAT_MIN) & (g_B <= B_HAT_MAX)
    valid_fit = strike_range_ok & b_ok & np.isfinite(g_A) & np.isfinite(g_B)

    # For invalid groups, mark results NaN so they are excluded later
    g_A = np.where(valid_fit, g_A, np.nan)
    g_B = np.where(valid_fit, np.clip(g_B, 1e-6, 1.0), np.nan)
    g_F = np.where(valid_fit, g_A / np.where(g_B > 1e-6, g_B, np.nan), np.nan)

    # Map back to row level
    df["a_hat"] = g_A[gid]
    df["b_hat"] = g_B[gid]
    df["F_hat"] = g_F[gid]
    # Drop rows where fit was invalid
    df = df[df["F_hat"].notna()].copy()
    if df.empty:
        return df

    K_valid = df["strike_price"].astype(float).values
    df["cp_fitted"]   = df["a_hat"] - df["b_hat"] * K_valid
    df["close_resid"] = df["cp_close"] - df["cp_fitted"]
    df["mid_resid"]   = df["cp_mid"]   - df["cp_fitted"]
    df["edge_sell"]   = df["cp_sell"]  - df["cp_fitted"]
    df["edge_buy"]    = df["cp_fitted"]- df["cp_buy"]
    df["exec_edge"]   = df[["edge_sell", "edge_buy"]].max(axis=1)
    df["signal_side"] = np.where(
        df["exec_edge"].fillna(0) <= 0, "none",
        np.where(
            df["edge_sell"].fillna(-np.inf) >= df["edge_buy"].fillna(-np.inf),
            "sell(C-P)", "buy(C-P)",
        ),
    )
    return df.drop(columns=["_gid"])


# ──────────────────────────────────────────────────────────────────────────────
# Yearly summary
# ──────────────────────────────────────────────────────────────────────────────

def yearly_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stats per year."""
    d = detail.copy()
    d["year"] = d["refdate"].dt.year

    grp = d.groupby("year").agg(
        n_pairs          = ("exec_edge",    "count"),
        n_groups         = ("refdate",      lambda x: x.nunique()),
        n_signals        = ("exec_edge",    lambda x: (x > 0).sum()),
        pct_signals      = ("exec_edge",    lambda x: 100 * (x > 0).mean()),
        max_exec_edge    = ("exec_edge",    "max"),
        median_exec_edge = ("exec_edge",    "median"),
        median_abs_resid = ("close_resid",  lambda x: x.abs().median()),
        mean_abs_resid   = ("close_resid",  lambda x: x.abs().mean()),
    ).reset_index()
    return grp


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def build_text_report(detail: pd.DataFrame, summary_yr: pd.DataFrame,
                      expiry_summary: pd.DataFrame, top_n: int = 20) -> str:
    lines: list[str] = []
    lines += [
        "Put-Call Parity Arbitrage Scan — B3 All Options",
        "=" * 70, "",
        f"Window  : {detail['refdate'].min().date()} to {detail['refdate'].max().date()}",
        f"Universe: equity options (BDI 78/82) + index options (BDI 74/75)",
        f"Total matched pairs analysed : {len(detail):,}",
        f"Pairs with positive exec edge: {int((detail['exec_edge'] > 0).sum()):,}  "
        f"({100*(detail['exec_edge']>0).mean():.1f}%)",
        f"Max executable edge          : {detail['exec_edge'].max():.4f}",
        f"Median |close residual|      : {detail['close_resid'].abs().median():.4f}",
        "",
    ]

    lines += ["Yearly Overview", "-" * 70]
    hdr = f"{'Year':>6}  {'Pairs':>8}  {'Signals':>8}  {'Sig%':>6}  {'MaxEdge':>9}  {'MedResid':>9}"
    lines.append(hdr)
    for _, row in summary_yr.iterrows():
        lines.append(
            f"{int(row.year):>6}  {int(row.n_pairs):>8,}  {int(row.n_signals):>8,}  "
            f"{row.pct_signals:>6.1f}%  {row.max_exec_edge:>9.4f}  {row.median_abs_resid:>9.4f}"
        )
    lines.append("")

    lines += [f"Top {top_n} Executable Signals (largest edge)", "-" * 70]
    top = detail[detail["exec_edge"] > 0].nlargest(top_n, "exec_edge")
    for _, r in top.iterrows():
        lines.append(
            f"{r['refdate'].date()}  exp={r['maturity_date'].date()}  "
            f"{r['underlying_name']:<18}  K={r['strike_price']:>8.2f}  "
            f"edge={r['exec_edge']:>8.4f}  {r['signal_side']:<12}  "
            f"C={r['c_sym']:<12}  P={r['p_sym']}"
        )
    lines.append("")

    # Top underlyings by signal frequency
    lines += [f"Top 20 Underlyings by Signal Count", "-" * 70]
    by_und = (
        detail[detail["exec_edge"] > 0]
        .groupby(["underlying_name", "universe"])
        .agg(
            n_signals     = ("exec_edge", "count"),
            max_edge      = ("exec_edge", "max"),
            mean_edge     = ("exec_edge", "mean"),
        )
        .sort_values("n_signals", ascending=False)
        .head(20)
        .reset_index()
    )
    for _, r in by_und.iterrows():
        lines.append(
            f"  {r['underlying_name']:<22}  {r['universe']:<8}  "
            f"signals={int(r.n_signals):>6,}  max_edge={r.max_edge:>8.4f}  mean_edge={r.mean_edge:>7.4f}"
        )
    lines.append("")

    lines += ["Interpretation", "-" * 70]
    lines += [
        "exec_edge > 0 means the bid/ask spread creates a net profit before",
        "transaction costs using the fitted parity line as the fair value.",
        "In practice, B3 equity options have bid-ask spreads of 5-20% of mid,",
        "so most small signals are sub-transaction-cost.  Signals > 0.5 BRL",
        "(for equity) or > 100 pts (for index) are more likely exploitable.",
        "",
        "Note: Large edges on illiquid options with wide spreads can be",
        "artefacts — always verify traded contracts > 0 on both legs.",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def make_charts(detail: pd.DataFrame, summary_yr: pd.DataFrame) -> None:
    """Generate and save summary charts."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("B3 Put-Call Parity Analysis — All Options", fontsize=14, fontweight="bold")

    # 1) % pairs with positive exec edge, by year
    ax = axes[0, 0]
    ax.bar(summary_yr["year"].astype(int), summary_yr["pct_signals"], color="steelblue", alpha=0.8)
    ax.set_title("% Pairs with Executable Arbitrage Signal (by year)")
    ax.set_xlabel("Year");  ax.set_ylabel("% of Matched Pairs")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(axis="y", alpha=0.4)

    # 2) Median absolute close-residual by year
    ax = axes[0, 1]
    ax.plot(summary_yr["year"].astype(int), summary_yr["median_abs_resid"],
            marker="o", color="darkorange", linewidth=2)
    ax.set_title("Median |Close Residual| (C-P close deviation from fitted parity)")
    ax.set_xlabel("Year");  ax.set_ylabel("BRL / index pts")
    ax.grid(alpha=0.4)

    # 3) Top-20 underlyings by total signals
    signals = (
        detail[detail["exec_edge"] > 0]
        .groupby("underlying_name")["exec_edge"]
        .count()
        .sort_values(ascending=False)
        .head(20)
    )
    ax = axes[1, 0]
    ax.barh(signals.index[::-1], signals.values[::-1], color="forestgreen", alpha=0.8)
    ax.set_title("Top 20 Underlyings by Number of Arbitrage Signals")
    ax.set_xlabel("Number of Executable Signals")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.4)

    # 4) Distribution of exec_edge (positive signals only, clipped)
    ax = axes[1, 1]
    pos = detail.loc[detail["exec_edge"] > 0, "exec_edge"]
    clip_hi = pos.quantile(0.95)
    pos_c = pos[pos <= clip_hi]
    ax.hist(pos_c, bins=60, color="crimson", alpha=0.75)
    ax.set_title(f"Distribution of Positive Executable Edges (≤ p95 = {clip_hi:.2f})")
    ax.set_xlabel("Executable Edge (BRL / index pts)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    out = OUT_DIR / "pcp_arbitrage_analysis.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="B3 PCP arbitrage full scan.")
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year",   type=int, default=2024)
    ap.add_argument("--top-n",      type=int, default=20)
    ap.add_argument(
        "--universe",
        choices=["equity", "index", "all"],
        default="equity",
        help="Option universe to scan (default: equity). Index options excluded by default "
             "because IBOV is not directly tradeable so PCP deviations there are structural.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))

    all_yr_summaries: list[dict] = []
    all_expiry_tops:  list[pd.DataFrame] = []

    # Incremental CSV path: write header on first year, append thereafter
    detail_path  = OUT_DIR / "pcp_full_detail.csv"
    yr_path      = OUT_DIR / "pcp_yearly_summary.csv"
    expiry_path  = OUT_DIR / "pcp_expiry_summary.csv"
    report_path  = OUT_DIR / "pcp_full_report.txt"

    COLS_TO_SAVE = [
        "refdate", "maturity_date", "underlying_key", "underlying_name",
        "universe", "strike_price", "c_sym", "p_sym",
        "c_close", "p_close", "cp_close",
        "c_mid", "p_mid", "cp_mid", "cp_sell", "cp_buy",
        "dte", "F_hat", "b_hat", "cp_fitted",
        "close_resid", "mid_resid", "edge_sell", "edge_buy", "exec_edge", "signal_side",
    ]

    first_write = True  # track whether detail CSV header has been written

    print("=" * 70)
    print(" B3 Put-Call Parity Arbitrage Scan  (vectorised)")
    print(f" Years: {years[0]}–{years[-1]}  |  Universe: {args.universe}")
    print("=" * 70)

    for year in years:
        df = load_options_year(year)
        if df is None:
            print(f"  {year}: no data")
            continue

        # Filter by universe
        if args.universe == "equity":
            df = df[df["universe"] == "equity"]
        elif args.universe == "index":
            df = df[df["universe"] == "index"]

        pairs = match_pairs(df)
        if pairs.empty:
            print(f"  {year}: no valid pairs after filtering")
            continue

        fitted = fit_parity_all(pairs)
        if fitted.empty:
            print(f"  {year}: no groups with ≥{MIN_PAIRS_PER_GRP} pairs")
            continue

        n_sig = int((fitted["exec_edge"] > 0).sum())
        pct   = 100 * n_sig / len(fitted)
        print(
            f"  {year}: {len(df):>7,} option rows  →  {len(pairs):>6,} pairs  →  "
            f"{len(fitted):>6,} fitted  →  {n_sig:>5,} signals ({pct:.1f}%)  "
            f"max_edge={fitted['exec_edge'].max():.3f}"
        )

        # ── Incremental save to detail CSV ────────────────────────────────
        fitted["year"] = year
        cols = [c for c in COLS_TO_SAVE + ["year"] if c in fitted.columns]
        fitted[cols].to_csv(
            detail_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

        # ── Year summary ──────────────────────────────────────────────────
        yr_row = yearly_summary(fitted).iloc[0].to_dict()
        all_yr_summaries.append(yr_row)

        # ── Keep top-100 expiry groups per year (memory-efficient) ────────
        exp_yr = (
            fitted
            .groupby(GROUP_KEYS + ["underlying_name", "universe"])
            .agg(
                n_pairs        = ("exec_edge", "count"),
                max_exec_edge  = ("exec_edge", "max"),
                n_signals      = ("exec_edge", lambda x: (x > 0).sum()),
                max_abs_resid  = ("close_resid", lambda x: x.abs().max()),
                F_hat          = ("F_hat", "first"),
            )
            .reset_index()
        )
        all_expiry_tops.append(exp_yr.nlargest(100, "max_exec_edge"))
        del fitted, pairs, df   # free memory

    if not all_yr_summaries:
        print("No results produced.")
        return

    # ── Final aggregation (only summaries, no full detail concat) ─────────
    yr_summary = pd.DataFrame(all_yr_summaries)
    expiry_summary = (
        pd.concat(all_expiry_tops, ignore_index=True)
        .sort_values("max_exec_edge", ascending=False)
    )

    yr_summary.to_csv(yr_path, index=False)
    expiry_summary.to_csv(expiry_path, index=False)

    # For the report/chart we load from the saved detail CSV (avoids re-accumulating)
    print("\nLoading saved detail for report/chart…")
    detail = pd.read_csv(detail_path, parse_dates=["refdate", "maturity_date"], low_memory=False)

    report_text = build_text_report(detail, yr_summary, expiry_summary, args.top_n)
    report_path.write_text(report_text, encoding="utf-8")

    # Charts
    make_charts(detail, yr_summary)

    print()
    print(f"Saved: {detail_path}  ({len(detail):,} rows)")
    print(f"Saved: {yr_path}")
    print(f"Saved: {expiry_path}")
    print(f"Saved: {report_path}")
    print()
    print(report_text[:3000])   # preview


if __name__ == "__main__":
    main()
