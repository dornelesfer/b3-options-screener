"""
election_risk_petr4.py
======================
Empirical base rates for PETR4 around Brazilian presidential elections.

PETR4 is the most politically-levered large cap on B3 (state-controlled,
government sets fuel-pricing policy), so election cycles are its dominant
risk event. This measures what ACTUALLY happened in the six cycles since
2002, to size a hedge against realized history rather than intuition.

For each cycle we measure, relative to the first-round date:
  - run-up   : return over the 53 trading-ish days before (matching today's
               distance to 2026 R1)
  - R1 gap   : return on the first trading day after round 1
  - R1->R2   : return between the rounds
  - R2 gap   : return on the first trading day after round 2
  - peak-to-trough drawdown inside the Aug-Dec window
  - realized vol in the window vs the rest of the year

Outputs: results/election_risk_petr4.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
RES.mkdir(exist_ok=True)
ANN = 252

# (first round, second round or None)
ELECTIONS = [
    ("2002-10-06", "2002-10-27"),
    ("2006-10-01", "2006-10-29"),
    ("2010-10-03", "2010-10-31"),
    ("2014-10-05", "2014-10-26"),
    ("2018-10-07", "2018-10-28"),
    ("2022-10-02", "2022-10-30"),
]
NEXT = ("2026-10-04", "2026-10-25")


def load_petr4():
    """Prefer Yahoo (adjusted, long history); fall back to B3 spot cache."""
    p = BASE / "data" / "spot_yahoo_PETR4.csv"
    s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["close"].sort_index()
    return s.dropna()


def nearest_after(idx, day):
    later = idx[idx > pd.Timestamp(day)]
    return later[0] if len(later) else None


def nearest_before(idx, day):
    prior = idx[idx <= pd.Timestamp(day)]
    return prior[-1] if len(prior) else None


def main():
    px = load_petr4()
    lr = np.log(px).diff()

    lines = []
    P = lines.append
    P("PETR4 around Brazilian presidential elections — realized history")
    P("=" * 78)
    P(f"Price history: {px.index.min().date()} - {px.index.max().date()}")
    P(f"2026 cycle: R1 {NEXT[0]}, R2 {NEXT[1]}")
    P("")
    P("Per-cycle moves (spot returns, not adjusted for dividends paid in window)")
    P("-" * 78)
    P(f"  {'cycle':<8}{'run-up':>9}{'R1 gap':>9}{'R1->R2':>9}{'R2 gap':>9}"
      f"{'Aug-Dec DD':>12}{'win RV':>9}{'yr RV':>8}")

    rows = []
    for r1, r2 in ELECTIONS:
        y = pd.Timestamp(r1).year
        d_r1 = nearest_before(px.index, r1)
        d_r1a = nearest_after(px.index, r1)
        d_r2 = nearest_before(px.index, r2) if r2 else None
        d_r2a = nearest_after(px.index, r2) if r2 else None
        if d_r1 is None or d_r1a is None:
            continue

        # run-up window: 53 calendar days before R1 (matches today's distance)
        d_start = nearest_before(px.index, pd.Timestamp(r1) - pd.Timedelta(days=53))
        runup = px.loc[d_r1] / px.loc[d_start] - 1

        r1_gap = px.loc[d_r1a] / px.loc[d_r1] - 1
        between = (px.loc[d_r2] / px.loc[d_r1a] - 1) if d_r2 is not None else np.nan
        r2_gap = (px.loc[d_r2a] / px.loc[d_r2] - 1) if d_r2a is not None else np.nan

        # Aug 1 - Dec 31 peak-to-trough
        win = px.loc[f"{y}-08-01":f"{y}-12-31"]
        dd = (win / win.cummax() - 1).min()

        win_rv = lr.loc[f"{y}-08-01":f"{y}-12-31"].std() * np.sqrt(ANN)
        yr_rv = lr.loc[f"{y}-01-01":f"{y}-12-31"].std() * np.sqrt(ANN)

        rows.append({"year": y, "runup": runup, "r1_gap": r1_gap,
                     "between": between, "r2_gap": r2_gap, "dd": dd,
                     "win_rv": win_rv, "yr_rv": yr_rv})
        P(f"  {y:<8}{runup:>8.1%}{r1_gap:>9.1%}{between:>9.1%}{r2_gap:>9.1%}"
          f"{dd:>12.1%}{win_rv:>9.0%}{yr_rv:>8.0%}")

    df = pd.DataFrame(rows)
    P("")
    P(f"  {'median':<8}{df['runup'].median():>8.1%}{df['r1_gap'].median():>9.1%}"
      f"{df['between'].median():>9.1%}{df['r2_gap'].median():>9.1%}"
      f"{df['dd'].median():>12.1%}{df['win_rv'].median():>9.0%}"
      f"{df['yr_rv'].median():>8.0%}")
    P(f"  {'worst':<8}{df['runup'].min():>8.1%}{df['r1_gap'].min():>9.1%}"
      f"{df['between'].min():>9.1%}{df['r2_gap'].min():>9.1%}"
      f"{df['dd'].min():>12.1%}")
    P("")

    P("What this implies for hedge sizing")
    P("-" * 78)
    P(f"  Median Aug-Dec peak-to-trough drawdown : {df['dd'].median():.1%}")
    P(f"  Worst  Aug-Dec peak-to-trough drawdown : {df['dd'].min():.1%}  "
      f"({int(df.loc[df['dd'].idxmin(),'year'])})")
    P(f"  Cycles with DD worse than -20%         : "
      f"{(df['dd'] < -0.20).sum()} of {len(df)}")
    P(f"  Cycles with DD worse than -30%         : "
      f"{(df['dd'] < -0.30).sum()} of {len(df)}")
    P(f"  Median realized vol Aug-Dec            : {df['win_rv'].median():.0%}"
      f"  (vs {df['yr_rv'].median():.0%} full-year)")
    P("")
    P("  Single-day gap risk around the rounds:")
    P(f"    R1 next-day move: median {df['r1_gap'].median():+.1%}, "
      f"range [{df['r1_gap'].min():+.1%}, {df['r1_gap'].max():+.1%}]")
    P(f"    R2 next-day move: median {df['r2_gap'].median():+.1%}, "
      f"range [{df['r2_gap'].min():+.1%}, {df['r2_gap'].max():+.1%}]")
    P("")
    P("  NOTE: gaps are two-sided — PETR4 rallied hard in some cycles and")
    P("  crashed in others. A hedge that only removes downside costs premium;")
    P("  a hedge that also caps upside (collar / rolled calls) is cheaper but")
    P("  forfeits the rally case, which is ~half of the historical outcomes.")

    # distribution of 74-day forward returns (today -> post R2) unconditionally
    fwd = px.pct_change(52).shift(-52).dropna()   # ~74 calendar days
    P("")
    P("Unconditional 74-calendar-day PETR4 return distribution (all history)")
    P("-" * 78)
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        P(f"    p{int(q*100):<3} {fwd.quantile(q):+7.1%}")

    report = "\n".join(lines)
    (RES / "election_risk_petr4.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
