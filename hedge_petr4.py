"""
hedge_petr4.py
==============
Election hedges for a covered-call PETR4 position, priced on B3 CLOSING
TRADE prices with hard liquidity gates.

WHY CLOSE, NOT MID: an audit of PETR4's EOD book (2026-08-24) found bids
pinned at R$0.04 against R$2.01 asks — 190%+ "spreads" and many zero asks.
Those quotes are stale placeholders, not executable markets; a mid built
from them understates put prices by ~50% and manufactures phantom skew
edges. `close` is an actual print, corroborated by traded_contracts.
Same lesson as the discarded PCP backtest in this repo.

Position is set in POSITION below. Candidate structures are compared on
net cost, the vol bought vs sold, executability at the user's size, and
book P&L across a spot grid at the hedge's expiry.

Run:  python3 hedge_petr4.py
Out:  results/hedge_petr4_report.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from data_cache import load_options
from screener_metrics import implied_vol, bs_delta

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
RES = BASE / "results"
RES.mkdir(exist_ok=True)

# ── POSITION ────────────────────────────────────────────────────────────────
N_SHARES = 18_500
N_CALLS = 18_300
# Covered calls actually held — set these once known.
CALL_STRIKE = None            # e.g. 45.17
CALL_EXPIRY = None            # e.g. pd.Timestamp("2026-09-18")

FEE_PCT = 0.00114 + 0.0020    # B3 emolumentos+liquidação + brokerage, RT
SLIPPAGE = 0.05               # 5% of premium, adverse, per leg
MIN_VOL_MULT = 2.0            # need traded_contracts >= 2x our size to call it liquid

R1, R2 = pd.Timestamp("2026-10-04"), pd.Timestamp("2026-10-25")
E_OCT, E_NOV, E_DEC = (pd.Timestamp("2026-10-16"), pd.Timestamp("2026-11-19"),
                       pd.Timestamp("2026-12-18"))


def load():
    sp = pq.read_table(str(BASE / "data" / "equity_spot.parquet")).to_pandas()
    sp["refdate"] = pd.to_datetime(sp["refdate"])
    p4 = (sp[(sp.symbol == "PETR4") & (sp.close > 0)]
          .drop_duplicates("refdate").set_index("refdate")["close"].sort_index())
    day = p4.index.max()

    r = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    r = r.dropna(subset=["r_cc"]).set_index("date")["r_cc"]
    r = float(r.reindex([day], method="ffill").iloc[0])

    o = load_options("equity_options")
    o["refdate"] = pd.to_datetime(o["refdate"])
    o["maturity_date"] = pd.to_datetime(o["maturity_date"])
    ch = o[(o.underlying == "PETR4") & (o.refdate == day) & (o.close > 0)].copy()
    ch["type"] = np.where(ch.bdi_code == 78, "C", "P")
    return p4, float(p4.loc[day]), day, r, ch


def pick(ch, expiry, K_target, cp, S, day, r, min_traded=0):
    """Closest strike with enough traded volume. Priced at close."""
    g = ch[(ch.maturity_date == expiry) & (ch["type"] == cp)
           & (ch.traded_contracts >= min_traded)]
    if g.empty:
        return None
    g = g.iloc[(g.strike_price - K_target).abs().argsort()]
    row = g.iloc[0]
    T = (expiry - day).days / 365
    px = float(row.close)
    iv = implied_vol(px, S, float(row.strike_price), T, r, cp)
    return {"K": float(row.strike_price), "px": px, "symbol": row.symbol,
            "traded": int(row.traded_contracts), "expiry": expiry, "cp": cp,
            "iv": iv, "T": T,
            "delta": bs_delta(S, float(row.strike_price), T, r, iv, cp)
            if iv == iv else np.nan}


def main():
    p4, S, day, r, ch = load()
    lr = np.log(p4).diff()
    rv21 = lr.tail(21).std() * np.sqrt(252) * 100
    rv63 = lr.tail(63).std() * np.sqrt(252) * 100

    L = []
    P = L.append
    P("PETR4 ELECTION HEDGE — priced on traded closes (bid/ask unusable)")
    P("=" * 78)
    P(f"Spot R$ {S:.2f} on {day.date()}   CDI {r:.2%}   "
      f"RV21 {rv21:.0f}%   RV63 {rv63:.0f}%")
    P(f"Long {N_SHARES:,} shares = R$ {N_SHARES*S:,.0f}   |   "
      f"short {N_CALLS:,} covered calls")
    P(f"R1 {R1.date()} (+{(R1-day).days}d)    R2 {R2.date()} (+{(R2-day).days}d)")
    P("")

    # ── term structure, close-based ─────────────────────────────────────────
    P("ATM vol term structure — the election premium, from traded prices")
    P("-" * 78)
    P(f"  {'expiry':<12}{'dte':>5}{'covers':>14}{'callIV':>9}{'putIV':>8}"
      f"{'put-call':>10}{'call vs RV63':>14}")
    for e in sorted({pd.Timestamp(x) for x in ch.maturity_date.unique()}):
        dte = (e - day).days
        if not (3 < dte <= 130):
            continue
        T = dte / 365
        F = S * np.exp(r * T)
        c = pick(ch, e, F, "C", S, day, r, min_traded=1000)
        p = pick(ch, e, F, "P", S, day, r, min_traded=1000)
        if not c or not p:
            continue
        if abs(c["K"] / F - 1) > 0.03 or abs(p["K"] / F - 1) > 0.03:
            continue
        if not (c["iv"] == c["iv"] and p["iv"] == p["iv"]):
            continue
        cov = "R1+R2" if e > R2 else "R1 only" if e > R1 else "pre-election"
        P(f"  {str(e.date()):<12}{dte:>5}{cov:>14}{c['iv']*100:>9.1f}"
          f"{p['iv']*100:>8.1f}{(c['iv']-p['iv'])*100:>+10.1f}"
          f"{c['iv']*100-rv63:>+14.1f}")
    P("")
    P("  Two things to read here:")
    P("  1. Oct/Nov calls carry ~6-7 vol pts of event premium over September.")
    P("  2. put-call gap is ~-3.5 pts vs the -0.9 pt PETR4 norm measured in")
    P("     covered_call_flow.py — downside protection is ~4x its usual")
    P("     richness. You are buying into that, so structure matters.")
    P("")

    # ── liquidity reality check ─────────────────────────────────────────────
    P(f"Liquidity at your size ({N_SHARES:,} contracts) — puts, last session")
    P("-" * 78)
    P(f"  {'expiry':<12}{'strikes w/ vol>=2x size':>26}{'best strikes (vol)':>34}")
    for e in (E_OCT, E_NOV, E_DEC):
        ge = ch[(ch.maturity_date == e) & (ch["type"] == "P")
                & (ch.strike_price.between(S * 0.70, S * 1.05))]
        ok = ge[ge.traded_contracts >= MIN_VOL_MULT * N_SHARES]
        top = ok.nlargest(3, "traded_contracts")
        s = "  ".join(f"{t.strike_price:.2f}({t.traded_contracts/1000:.0f}k)"
                      for t in top.itertuples())
        P(f"  {str(e.date()):<12}{len(ok):>26}{s:>34}")
    P("")
    P("  October is where the volume is. November — the only listed expiry")
    P("  that survives round 2 — is materially thinner, which is the central")
    P("  trade-off in this whole decision.")
    P("")

    # ── structures ──────────────────────────────────────────────────────────
    def build(name, expiry, spec, note=""):
        legs = []
        for K_t, cp, q, minv in spec:
            lg = pick(ch, expiry, K_t, cp, S, day, r, min_traded=minv)
            if lg is None or lg["iv"] != lg["iv"]:
                return None
            lg["qty"] = q * N_SHARES
            legs.append(lg)
        gross = sum(l["px"] * l["qty"] for l in legs)
        slip = sum(abs(l["px"] * l["qty"]) for l in legs) * SLIPPAGE
        fees = sum(abs(l["px"] * l["qty"]) for l in legs) * FEE_PCT
        return {"name": name, "expiry": expiry, "legs": legs, "note": note,
                "cost": gross + slip + fees, "gross": gross}

    MINV = int(MIN_VOL_MULT * N_SHARES)
    cands = [
        build("A. Trava baixa OCT 42/36", E_OCT,
              [(S, "P", +1, MINV), (S * 0.855, "P", -1, MINV)],
              "your idea — but expires 9d BEFORE round 2"),
        build("B. Trava baixa NOV 43/38", E_NOV,
              [(S * 1.025, "P", +1, MINV), (S * 0.905, "P", -1, MINV)],
              "the two Nov strikes that actually trade; covers R1+R2"),
        build("C. Put seca NOV 43", E_NOV,
              [(S * 1.025, "P", +1, MINV)],
              "no cap on protection, full premium paid"),
        build("D. Put seca OCT 42", E_OCT,
              [(S, "P", +1, MINV)], "R1 only"),
        build("E. Trava baixa OCT 42/33 (wide)", E_OCT,
              [(S, "P", +1, MINV), (S * 0.785, "P", -1, MINV)],
              "wider protection band, R1 only"),
    ]
    cands = [c for c in cands if c]

    P("STRUCTURES (close prices, 5% adverse slippage + B3 fees included)")
    P("=" * 78)
    for c in cands:
        P("")
        P(f"{c['name']}  [{c['expiry'].date()}, "
          f"{(c['expiry']-day).days}d]  — {c['note']}")
        for l in c["legs"]:
            side = "BUY " if l["qty"] > 0 else "SELL"
            liq = "OK" if l["traded"] >= MIN_VOL_MULT * N_SHARES else "THIN"
            P(f"    {side} {abs(l['qty']):>6,}  {l['symbol']:<12} K={l['K']:>6.2f}"
              f"  R$ {l['px']:>5.2f}   IV {l['iv']*100:>5.1f}%"
              f"   Δ {l['delta']:>+5.2f}   vol {l['traded']:>9,} {liq}")
        pct = c["cost"] / (N_SHARES * S)
        P(f"    NET COST  R$ {c['cost']:>9,.0f}  = {pct:.2%} of your share notional"
          f"   (annualised {pct*365/(c['expiry']-day).days:.1%})")
        if len(c["legs"]) == 2:
            w = abs(c["legs"][0]["K"] - c["legs"][1]["K"])
            mp = w * N_SHARES
            ivb, ivs = c["legs"][0]["iv"] * 100, c["legs"][1]["iv"] * 100
            P(f"    Protection band R$ {c['legs'][1]['K']:.2f}-{c['legs'][0]['K']:.2f}"
              f"  → max payoff R$ {mp:,.0f};  cost/protection {c['cost']/mp:.0%}")
            P(f"    Vol bought {ivb:.1f}% vs sold {ivs:.1f}%  →  "
              f"{'you EARN' if ivs>ivb else 'you PAY'} {abs(ivs-ivb):.1f} pts of skew")
            floor = (c["legs"][1]["K"] - S) * N_SHARES - c["cost"]
            P(f"    Below R$ {c['legs'][1]['K']:.2f} you are unprotected again; "
              f"P&L there = R$ {floor:,.0f} + further downside")
        else:
            ivb = c["legs"][0]["iv"] * 100
            P(f"    Vol bought {ivb:.1f}%  (vs RV63 {rv63:.0f}% → paying "
              f"{ivb-rv63:.0f} pts of premium)")

    # ── payoff grids ────────────────────────────────────────────────────────
    for exp_grp, label in ((E_OCT, "OCTOBER expiry (round 1 only)"),
                           (E_NOV, "NOVEMBER expiry (rounds 1 and 2)")):
        P("")
        P("")
        P(f"BOOK P&L AT {label}")
        P("=" * 78)
        grid = np.array([0.60, 0.70, 0.80, 0.90, 0.95, 1.00, 1.10, 1.20]) * S
        P("  " + "spot".ljust(12) + "".join(f"{g:>9.2f}" for g in grid))
        P("  " + "move".ljust(12) + "".join(f"{g/S-1:>8.0%} " for g in grid))
        P("-" * 78)
        base = (grid - S) * N_SHARES
        P("  " + "shares only".ljust(12)
          + "".join(f"{v/1000:>8.0f}k" for v in base))
        for c in cands:
            if c["expiry"] != exp_grp:
                continue
            pnl = base - c["cost"]
            for l in c["legs"]:
                intr = (np.maximum(grid - l["K"], 0) if l["cp"] == "C"
                        else np.maximum(l["K"] - grid, 0))
                pnl = pnl + l["qty"] * intr
            P("  " + c["name"].split(".")[0].ljust(12)
              + "".join(f"{v/1000:>8.0f}k" for v in pnl))
        P("")
        P("  R$ thousands, P&L on the 18,500 shares vs today's spot.")
        P("  Covered calls NOT included — they cap the two right-hand columns")
        P("  once their strike is known (set CALL_STRIKE/CALL_EXPIRY).")

    # ── historical yardstick ────────────────────────────────────────────────
    P("")
    P("")
    P("SIZING YARDSTICK — PETR4 Aug-Dec drawdown, last six election cycles")
    P("-" * 78)
    P("  2002 -27%   2006 -17%   2010 -18%   2014 -63%   2018 -24%   2022 -43%")
    P("  median -25%   |   4 of 6 cycles worse than -20%")
    P("")
    P(f"  A -25% move costs the unhedged share position R$ "
      f"{0.25*N_SHARES*S:,.0f}.")
    P("  Judge each structure's cost against that, not against zero.")

    rep = "\n".join(L)
    (RES / "hedge_petr4_report.txt").write_text(rep)
    print(rep)


if __name__ == "__main__":
    main()
