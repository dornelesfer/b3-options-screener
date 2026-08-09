"""
analyze_put_call_parity.py
==========================

Standalone scan of B3 option put-call-parity dislocations from the same parquet
snapshots used by the VXBR backtest.

The analysis is intentionally separate from `backtest_vxbr.py` and focuses on
matched call/put pairs with the same date, expiry, strike, and underlying.

By default it scans both:
  - IBOV index options  (BDI 74/75)
  - equity options      (BDI 78/82)

This version is quote-driven: it requires positive bid/ask quotes on both the
call and the put, and does not fall back to close prices.

Outputs (saved to results/):
  - put_call_parity_detailed.csv
  - put_call_parity_expiry_summary.csv
  - put_call_parity_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE = Path(__file__).parent
DATA_DIR = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_YEARS = list(range(2000, 2027))
R_BRAZIL = 0.12
MIN_TIME_TO_EXPIRY_DAYS = 2
MIN_PAIRS_PER_EXPIRY = 3
TOP_N = 25

NEEDED_COLS = [
    "refdate",
    "bdi_code",
    "symbol",
    "corporation_name",
    "specification_code",
    "isin",
    "close",
    "best_bid",
    "best_ask",
    "strike_price",
    "maturity_date",
    "volume",
    "traded_contracts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan B3 option put-call parity dislocations."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_YEARS[0],
        help="First year to include.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_YEARS[-1],
        help="Last year to include.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of largest opportunities to show in the text report.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional cap on analyzed date/expiry groups for quick validation.",
    )
    parser.add_argument(
        "--universe",
        choices=["index", "equity", "all"],
        default="all",
        help="Which option universe to analyze.",
    )
    return parser.parse_args()


def universe_mask(df: pd.DataFrame, universe: str) -> pd.Series:
    if universe == "index":
        return (
            df["bdi_code"].isin([74, 75])
            & df["specification_code"].astype(str).str.strip().str.startswith("IBO")
        )
    if universe == "equity":
        return df["bdi_code"].isin([78, 82])
    return (
        (
            df["bdi_code"].isin([74, 75])
            & df["specification_code"].astype(str).str.strip().str.startswith("IBO")
        )
        | df["bdi_code"].isin([78, 82])
    )


def load_year(year: int, universe: str) -> pd.DataFrame:
    path = DATA_DIR / f"year={year}" / "part-0.parquet"
    if not path.exists():
        return pd.DataFrame(columns=NEEDED_COLS)

    df = pq.read_table(str(path), columns=NEEDED_COLS).to_pandas()
    df = df[
        universe_mask(df, universe)
        & (df["strike_price"] > 0)
    ].copy()

    if df.empty:
        return df

    df["refdate"] = pd.to_datetime(df["refdate"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"])
    df["underlying_name"] = df["corporation_name"].astype(str).str.strip()
    df["underlying_isin"] = df["isin"].astype(str).str.strip()
    df["universe"] = np.where(df["bdi_code"].isin([74, 75]), "index", "equity")
    df["underlying_key"] = np.where(
        df["universe"] == "index",
        "INDEX:IBOV",
        "EQUITY:" + df["underlying_isin"],
    )
    df["option_type"] = np.select(
        [
            df["bdi_code"].isin([74, 78]),
            df["bdi_code"].isin([75, 82]),
        ],
        [
            "C",
            "P",
        ],
        default=None,
    )
    df["mid"] = np.where(
        (df["best_bid"] > 0) & (df["best_ask"] > 0),
        (df["best_bid"] + df["best_ask"]) / 2.0,
        np.nan,
    )
    df["days_to_expiry"] = (df["maturity_date"] - df["refdate"]).dt.days
    df = df[
        (df["days_to_expiry"] >= MIN_TIME_TO_EXPIRY_DAYS)
        & df["option_type"].isin(["C", "P"])
    ].copy()
    return df


def estimate_parity_line(cp_pairs: pd.DataFrame) -> tuple[float, float, float]:
    """
    Fit the affine parity relation:
      C - P = A - B*K

    A ~= discounted spot/carry term
    B ~= exp(-rT)

    Returns (A, B, F_hat) where F_hat = A / B.
    """
    work = cp_pairs[["strike_price", "cp_mid"]].dropna().copy()
    if len(work) < MIN_PAIRS_PER_EXPIRY:
        return np.nan, np.nan, np.nan

    x = np.vstack([np.ones(len(work)), -work["strike_price"].to_numpy(dtype=float)]).T
    y = work["cp_mid"].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    a_hat = float(beta[0])
    b_hat = float(np.clip(beta[1], 1e-6, 1.0))
    f_hat = a_hat / b_hat if b_hat > 0 else np.nan
    return a_hat, b_hat, f_hat


def build_pairs(expiry_df: pd.DataFrame) -> pd.DataFrame:
    calls = expiry_df[expiry_df["option_type"] == "C"].copy()
    puts = expiry_df[expiry_df["option_type"] == "P"].copy()

    call_cols = {
        "symbol": "call_symbol",
        "close": "call_close",
        "best_bid": "call_bid",
        "best_ask": "call_ask",
        "mid": "call_mid",
        "volume": "call_volume",
        "traded_contracts": "call_contracts",
    }
    put_cols = {
        "symbol": "put_symbol",
        "close": "put_close",
        "best_bid": "put_bid",
        "best_ask": "put_ask",
        "mid": "put_mid",
        "volume": "put_volume",
        "traded_contracts": "put_contracts",
    }

    calls = calls[["strike_price", *call_cols.keys()]].rename(columns=call_cols)
    puts = puts[["strike_price", *put_cols.keys()]].rename(columns=put_cols)

    pairs = calls.merge(puts, on="strike_price", how="inner")
    if pairs.empty:
        return pairs

    pairs["cp_mid"] = pairs["call_mid"] - pairs["put_mid"]
    pairs["cp_sell"] = pairs["call_bid"].fillna(0.0) - pairs["put_ask"].fillna(np.inf)
    pairs["cp_buy"] = pairs["call_ask"].fillna(np.inf) - pairs["put_bid"].fillna(0.0)
    pairs["total_contracts"] = (
        pairs["call_contracts"].fillna(0.0) + pairs["put_contracts"].fillna(0.0)
    )
    pairs["total_volume"] = (
        pairs["call_volume"].fillna(0.0) + pairs["put_volume"].fillna(0.0)
    )
    valid_quote_pair = (
        pairs["call_bid"].gt(0)
        & pairs["call_ask"].gt(0)
        & pairs["put_bid"].gt(0)
        & pairs["put_ask"].gt(0)
    )
    pairs = pairs[valid_quote_pair].copy()
    return pairs


def analyze_expiry(expiry_df: pd.DataFrame) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    base = expiry_df[
        [
            "refdate",
            "maturity_date",
            "days_to_expiry",
            "underlying_key",
            "underlying_name",
            "underlying_isin",
            "universe",
        ]
    ].iloc[0]
    pairs = build_pairs(expiry_df)
    if len(pairs) < MIN_PAIRS_PER_EXPIRY:
        return None, None

    a_hat, b_hat, f_hat = estimate_parity_line(pairs)
    if np.isnan(a_hat) or np.isnan(b_hat) or np.isnan(f_hat):
        return None, None

    t = float(base["days_to_expiry"]) / 365.0
    parity_theoretical_discount = np.exp(-R_BRAZIL * t)

    pairs = pairs.copy()
    pairs["refdate"] = base["refdate"]
    pairs["maturity_date"] = base["maturity_date"]
    pairs["days_to_expiry"] = int(base["days_to_expiry"])
    pairs["underlying_key"] = base["underlying_key"]
    pairs["underlying_name"] = base["underlying_name"]
    pairs["underlying_isin"] = base["underlying_isin"]
    pairs["universe"] = base["universe"]
    pairs["T"] = t
    pairs["parity_a_hat"] = a_hat
    pairs["parity_b_hat"] = b_hat
    pairs["forward_hat"] = f_hat
    pairs["discount_theoretical"] = parity_theoretical_discount
    pairs["cp_fitted"] = a_hat - b_hat * pairs["strike_price"]
    pairs["mid_residual"] = pairs["cp_mid"] - pairs["cp_fitted"]

    sell_available = pairs["call_bid"].gt(0) & pairs["put_ask"].gt(0)
    buy_available = pairs["call_ask"].gt(0) & pairs["put_bid"].gt(0)

    pairs["edge_sell_cp"] = np.where(
        sell_available,
        pairs["cp_sell"] - pairs["cp_fitted"],
        np.nan,
    )
    pairs["edge_buy_cp"] = np.where(
        buy_available,
        pairs["cp_fitted"] - pairs["cp_buy"],
        np.nan,
    )
    pairs["executable_edge"] = pairs[["edge_sell_cp", "edge_buy_cp"]].max(axis=1, skipna=True)
    pairs["has_executable_signal"] = pairs["executable_edge"].fillna(0.0) > 0
    pairs["signal_side"] = np.select(
        [
            pairs["has_executable_signal"]
            & (pairs["edge_sell_cp"].fillna(-np.inf) >= pairs["edge_buy_cp"].fillna(-np.inf)),
            pairs["has_executable_signal"]
            & (pairs["edge_buy_cp"].fillna(-np.inf) > pairs["edge_sell_cp"].fillna(-np.inf)),
        ],
        [
            "sell(C-P)",
            "buy(C-P)",
        ],
        default="none",
    )

    summary = {
        "refdate": base["refdate"],
        "maturity_date": base["maturity_date"],
        "days_to_expiry": int(base["days_to_expiry"]),
        "underlying_key": base["underlying_key"],
        "underlying_name": base["underlying_name"],
        "underlying_isin": base["underlying_isin"],
        "universe": base["universe"],
        "n_pairs": int(len(pairs)),
        "forward_hat": float(f_hat),
        "discount_hat": float(b_hat),
        "discount_theoretical": float(parity_theoretical_discount),
        "max_abs_mid_residual": float(pairs["mid_residual"].abs().max()),
        "mean_abs_mid_residual": float(pairs["mid_residual"].abs().mean()),
        "max_executable_edge": float(pairs["executable_edge"].max(skipna=True)),
        "n_executable_signals": int(pairs["has_executable_signal"].sum()),
    }
    return pairs, summary


def build_report(detail: pd.DataFrame, summary: pd.DataFrame, top_n: int) -> str:
    lines: list[str] = []
    lines.append("Put-Call Parity Analysis Report")
    lines.append("=" * 65)
    lines.append("")

    if detail.empty:
        lines.append("No matched call/put pairs passed the analysis filters.")
        return "\n".join(lines)

    lines.append(f"Window             : {detail['refdate'].min().date()} — {detail['refdate'].max().date()}")
    lines.append(f"Universe           : {', '.join(sorted(detail['universe'].dropna().unique()))}")
    lines.append(f"Matched pairs       : {len(detail):,}")
    lines.append(f"Date/expiry groups  : {len(summary):,}")
    lines.append(f"Pairs with signal   : {int(detail['has_executable_signal'].sum()):,}")
    lines.append(
        f"Max executable edge : {detail['executable_edge'].max(skipna=True):.4f}"
    )
    lines.append(
        f"Median |mid residual| : {detail['mid_residual'].abs().median():.4f}"
    )
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("`mid_residual` compares the matched pair's C-P mid to the parity line fitted")
    lines.append("from all strikes of the same expiry. `executable_edge` uses bid/ask quotes:")
    lines.append("positive values mean the pair sits outside the fitted parity band.")
    lines.append("")

    top = detail.sort_values("executable_edge", ascending=False).head(top_n)
    lines.append(f"Top {len(top)} executable signals")
    lines.append("-" * 65)
    for _, row in top.iterrows():
        lines.append(
            f"{row['refdate'].date()} | exp {row['maturity_date'].date()} | "
            f"{row['underlying_name']} | K={row['strike_price']:.2f} | edge={row['executable_edge']:.4f} | "
            f"side={row['signal_side']} | C={row['call_symbol']} | P={row['put_symbol']}"
        )

    grp = summary.sort_values("max_executable_edge", ascending=False).head(top_n)
    lines.append("")
    lines.append(f"Top {len(grp)} date/expiry groups")
    lines.append("-" * 65)
    for _, row in grp.iterrows():
        lines.append(
            f"{row['refdate'].date()} | exp {row['maturity_date'].date()} | "
            f"{row['underlying_name']} | pairs={int(row['n_pairs'])} | max_edge={row['max_executable_edge']:.4f} | "
            f"max_|mid_residual|={row['max_abs_mid_residual']:.4f} | "
            f"F_hat={row['forward_hat']:.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))

    all_details: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    print("=" * 65)
    print(" B3 Put-Call Parity Analysis")
    print(f" Window: {years[0]}–{years[-1]} | Universe: {args.universe}")
    print("=" * 65)
    analyzed_groups = 0

    for year in years:
        df = load_year(year, args.universe)
        if df.empty:
            print(f"   {year}: no qualifying option rows")
            continue

        year_groups = 0
        year_signals = 0
        for _, expiry_df in df.groupby(
            ["refdate", "maturity_date", "underlying_key"],
            sort=True,
        ):
            if args.max_groups is not None and analyzed_groups >= args.max_groups:
                break
            detail, summary = analyze_expiry(expiry_df)
            if detail is None:
                continue
            all_details.append(detail)
            all_summaries.append(summary)
            year_groups += 1
            analyzed_groups += 1
            year_signals += int(summary["n_executable_signals"] > 0)

        print(
            f"   {year}: {len(df):6,} rows -> {year_groups:4d} expiry groups, "
            f"{year_signals:4d} groups with executable signal"
        )
        if args.max_groups is not None and analyzed_groups >= args.max_groups:
            break

    detail_df = (
        pd.concat(all_details, ignore_index=True)
        if all_details
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(all_summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["max_executable_edge", "max_abs_mid_residual"],
            ascending=[False, False],
        )

    detail_path = OUT_DIR / "put_call_parity_detailed.csv"
    summary_path = OUT_DIR / "put_call_parity_expiry_summary.csv"
    report_path = OUT_DIR / "put_call_parity_report.txt"

    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    report_text = build_report(detail_df, summary_df, args.top_n)
    report_path.write_text(report_text, encoding="utf-8")

    print()
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    if not detail_df.empty:
        print(
            f"Max executable edge: {detail_df['executable_edge'].max(skipna=True):.4f}"
        )
        print(
            f"Pairs with executable signal: {int(detail_df['has_executable_signal'].sum()):,}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
