"""
screener_metrics.py
===================
Builds the metrics tables behind the options screener app.

Sources (all local caches):
  IBOV  : data/ibov_options_all[_YYYY].parquet (bdi 74/75) spot data/ibov_daily.csv
  PETR4 / VALE3 / VALE5 : data/equity_options[_YYYY].parquet (bdi 78/82)
                          spot data/equity_spot.parquet
  (option caches are partitioned by year — read via data_cache.load_options)
  BRAV3 : data/brav3_options_recent.csv               spot data/brav3_equity_recent.csv
  Rates : data/rates_cdi.csv (daily CDI, r_cc)

Outputs (data/screener/):
  chain_latest.parquet    per-option metrics on the latest common date:
                          mid, iv, delta, moneyness, dte, smile_resid,
                          iv_minus_rv, cp_gap, parity_resid, below_intrinsic
  history_daily.parquet   per (underlying, date): spot, rv21, iv_atm, spread,
                          spread percentile (expanding)

Metrics lineage: iv_minus_rv is the VRP entry signal (backtest_short_vol*),
cp_gap the covered-call flow gap (covered_call_flow.py), parity_resid /
implied borrow from implied_carry.py, below_intrinsic from the BRAV3 work.

HORIZON MATCHING (Burghardt & Lane 1990)
----------------------------------------
`iv_minus_rv` compares every option's implied vol to a fixed 21-day realised
vol, whatever its time to expiry. On the current chain that means judging a
725-day option against a 21-day window. Burghardt & Lane, "How to tell if
options are cheap" (Journal of Portfolio Management 16(2):72-78, 1990), name
this directly: an implied vol is a forecast over the option's REMAINING LIFE,
so "the only option for which a thirty-trading-day historical volatility is an
appropriate standard is an option with thirty trading days remaining to
expiration."

Their fix is the volatility cone -- the historical DISTRIBUTION of realised vol
measured over the MATCHING horizon. Two facts make it matter:

  * Cones narrow with horizon. Short-horizon realised vol is far more variable
    than long-horizon (their Eurodollars: 1-month realised ranged 10-57%,
    3-month 12-35%, 6-month narrower still). So a 5-point deviation is ordinary
    at 21 days and extreme at 252, and a single fixed-window spread cannot
    express that difference.
  * A trailing window spikes at the START of a crisis, which makes options look
    cheap exactly when they are dear. Their worked example: after October 1987,
    30-day realised was enormous, so June-1988 Eurodollar options at 30% implied
    looked cheap -- but 30% sat above the top of the nine-month cone, and
    selling them was highly profitable.

Corrected metrics, added alongside the originals rather than replacing them so
existing consumers keep working:

  rv_matched     realised vol over a window matching the option's own DTE
  iv_minus_rv_h  implied minus rv_matched -- the horizon-matched spread
  cone_pct       position of implied vol within the trailing cone at that
                 horizon, in [0, 100]; 100 = above every realised outcome of
                 that horizon in the lookback window
  cone_z         same comparison in cone standard deviations. Use this to RANK:
                 cone_pct saturates at 100 for 27% of the current chain (59% of
                 the 70-150 day bucket), where cone_z keeps discriminating

`iv_minus_rv` is retained and is correct only for options near 21 trading days
(~30 calendar days) to expiry. Prefer `iv_minus_rv_h` / `cone_pct`.

CAVEAT -- read the cone metrics near the money. All three compare an option's
implied vol to the UNDERLYING's realised vol, which is an at-the-money concept.
The smile inflates IV mechanically as you move out: on the current chain median
`cone_z` runs 0.8 at |moneyness| < 0.05 and 5.2 beyond 0.35, so an unfiltered
`cone_z` ranking returns deep wings rather than genuinely rich options. Filter
to near-ATM, or read `cone_z` next to `smile_resid`, which is what strips the
skew out.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm
from scipy.optimize import brentq

from data_cache import load_options

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT = BASE / "data" / "screener"
OUT.mkdir(parents=True, exist_ok=True)

ANN = 252
ATM_BAND = 0.05          # |K/F - 1| for ATM bucket
ATM_DTE = (15, 60)
HIST_YEARS = 3
STALE_DAYS = 30          # drop an underlying from chain_latest if its newest
                         # chain is this far behind the freshest one
FRESH_BDAYS = 3          # warn if the newest chain is older than this many
                         # business days (nightly job silently stopped ingesting)

# volatility cones (Burghardt & Lane 1990)
CONE_LOOKBACK = 504      # trailing window for the cone; the paper used ~2 years
CONE_MIN_OBS = 60        # refuse to quote a percentile on less than this
CONE_MAX_HORIZON = 252   # clamp: beyond a year the cone has too few draws here


# ── Black-Scholes ────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sig, cp):
    if T <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sig, cp):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, cp):
    intrinsic = max(S - K, 0.0) if cp == "C" else max(K * np.exp(-r * T) - S, 0.0)
    if price <= intrinsic + 1e-9 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, cp) - price,
                      0.02, 3.0, xtol=1e-6)
    except ValueError:
        return np.nan


# ── volatility cones ─────────────────────────────────────────────────────────
def dte_to_trading_days(dte_calendar):
    """Calendar days to expiry -> the trading-day horizon to measure RV over.

    Option DTE is quoted in calendar days; realised vol is measured in trading
    days. Converting keeps the two on the same clock.
    """
    h = int(round(float(dte_calendar) * ANN / 365.0))
    return int(np.clip(h, 2, CONE_MAX_HORIZON))


class ConeBook:
    """Per-underlying volatility cones, cached by horizon.

    A cone at horizon h is the distribution of annualised realised vol measured
    over trailing h-trading-day windows. `position` returns where an implied vol
    sits inside that distribution, using only windows that closed STRICTLY
    BEFORE the quote date -- so the reading is causal and can be used as a
    signal without lookahead.

    Windows overlap (daily increments), which Burghardt & Lane note adds little
    information over monthly steps. Treat the effective sample as roughly
    lookback/horizon independent draws, not `lookback`.
    """

    def __init__(self, spots):
        self._spots = spots
        self._rv = {}
        self._hist = {}

    def rv_series(self, und, horizon):
        """Annualised RV over trailing `horizon`-day windows, indexed by date."""
        key = (und, horizon)
        if key not in self._rv:
            spot = self._spots.get(und)
            if spot is None or len(spot) <= horizon:
                self._rv[key] = pd.Series(dtype=float)
            else:
                lr = np.log(spot.astype(float)).diff()
                self._rv[key] = lr.rolling(horizon).std() * np.sqrt(ANN)
        return self._rv[key]

    def matched_rv(self, und, day, horizon):
        """RV over the `horizon`-day window ending on or before `day`."""
        s = self.rv_series(und, horizon).dropna()
        s = s[s.index <= day]
        return float(s.iloc[-1]) if len(s) else np.nan

    def cone(self, und, day, horizon, lookback=CONE_LOOKBACK):
        """Trailing realised-vol draws for the cone at `horizon`, all from
        windows that closed STRICTLY BEFORE `day` (no lookahead). None if too
        few. Cached: every option on a chain at the same horizon shares one."""
        key = (und, horizon, day, lookback)
        if key not in self._hist:
            s = self.rv_series(und, horizon).dropna()
            s = s[s.index < day]
            self._hist[key] = (s.iloc[-lookback:].to_numpy()
                               if len(s) >= CONE_MIN_OBS else None)
        return self._hist[key]

    def position(self, und, day, iv, horizon, lookback=CONE_LOOKBACK):
        """Percentile of `iv` within the trailing cone at `horizon`, in [0, 100].

        `iv` and the cone are both in decimal units (0.30 = 30%).
        """
        if iv != iv:
            return np.nan
        hist = self.cone(und, day, horizon, lookback)
        if hist is None:
            return np.nan
        return float((hist < iv).mean() * 100.0)

    def zscore(self, und, day, iv, horizon, lookback=CONE_LOOKBACK):
        """Distance from the cone's centre, in cone standard deviations.

        `position` saturates: once implied sits above every realised outcome in
        the lookback it reads 100 and stops ranking. On this chain that happens
        for 27% of options overall and 59% of the 70-150 day bucket -- i.e.
        exactly where the long-dated names live. This keeps discriminating above
        the top of the cone, which is what the screener needs for ranking.
        """
        if iv != iv:
            return np.nan
        hist = self.cone(und, day, horizon, lookback)
        if hist is None:
            return np.nan
        sd = hist.std()
        if not sd:
            return np.nan
        return float((iv - hist.mean()) / sd)


# ── loaders ──────────────────────────────────────────────────────────────────
def load_rates():
    r = pd.read_csv(BASE / "data" / "rates_cdi.csv", parse_dates=["date"])
    s = r.dropna(subset=["r_cc"]).set_index("date")["r_cc"]
    return s.reindex(pd.date_range(s.index.min(), s.index.max())).ffill()


def load_chains():
    frames = []

    ibo = load_options("ibov_options_all")
    ibo["type"] = np.where(ibo["bdi_code"] == 74, "C", "P")
    ibo["underlying"] = "IBOV"
    frames.append(ibo)

    eq = load_options("equity_options")
    eq["type"] = np.where(eq["bdi_code"] == 78, "C", "P")
    frames.append(eq)

    br_p = BASE / "data" / "brav3_options_recent.csv"
    if br_p.exists():
        br = pd.read_csv(br_p, parse_dates=["refdate", "maturity_date"])
        br["type"] = np.where(br["bdi_code"] == 78, "C", "P")
        br["underlying"] = "BRAV3"
        frames.append(br)

    cols = ["refdate", "underlying", "symbol", "type", "strike_price", "close",
            "best_bid", "best_ask", "maturity_date", "traded_contracts", "volume"]
    out = pd.concat([f[[c for c in cols if c in f.columns]] for f in frames],
                    ignore_index=True)
    out["refdate"] = pd.to_datetime(out["refdate"])
    out["maturity_date"] = pd.to_datetime(out["maturity_date"])
    return out[(out["close"] > 0) & (out["strike_price"] > 0)]


def load_spots():
    spots = {}
    ib = pd.read_csv(BASE / "data" / "ibov_daily.csv", parse_dates=["date"])
    spots["IBOV"] = ib.set_index("date")["ibov_close"].sort_index()

    eq = pq.read_table(str(BASE / "data" / "equity_spot.parquet")).to_pandas()
    eq["refdate"] = pd.to_datetime(eq["refdate"])
    for sym, g in eq[eq["close"] > 0].groupby("symbol"):
        spots[sym] = g.drop_duplicates("refdate").set_index("refdate")["close"].sort_index()

    br_p = BASE / "data" / "brav3_equity_recent.csv"
    if br_p.exists():
        br = pd.read_csv(br_p, parse_dates=["refdate"])
        col = "close" if "close" in br.columns else br.columns[-1]
        s = br.drop_duplicates("refdate").set_index("refdate")[col].sort_index()
        # extend with Yahoo history if present (for RV)
        yh_p = BASE / "data" / "spot_yahoo_BRAV3.csv"
        if yh_p.exists():
            yh = pd.read_csv(yh_p, parse_dates=["date"]).set_index("date")["close"]
            s = yh.combine_first(s)
        spots["BRAV3"] = s
    return spots


# ── history: per-underlying ATM IV vs RV ─────────────────────────────────────
def build_history(chains, spots, r_curve, cones=None):
    cones = cones or ConeBook(spots)
    cutoff = chains["refdate"].max() - pd.DateOffset(years=HIST_YEARS)
    ch = chains[chains["refdate"] >= cutoff].copy()
    rows = []
    for und, g_und in ch.groupby("underlying"):
        spot = spots.get(und)
        if spot is None:
            continue
        rv = np.log(spot).diff().rolling(21).std() * np.sqrt(ANN)
        for day, g in g_und.groupby("refdate"):
            if day not in spot.index:
                continue
            S = float(spot.loc[day])
            r = float(r_curve.loc[day]) if day in r_curve.index else 0.10
            g = g.copy()
            g["dte"] = (g["maturity_date"] - day).dt.days
            g = g[(g["dte"] >= ATM_DTE[0]) & (g["dte"] <= ATM_DTE[1])
                  & (g["traded_contracts"] > 0)]
            g = g[np.abs(g["strike_price"] / S - 1) <= ATM_BAND]
            if len(g) < 2:
                continue
            ivs = [implied_vol(row.close, S, row.strike_price, row.dte / 365.0,
                               r, row.type) for row in g.itertuples()]
            ivs = [v for v in ivs if not np.isnan(v)]
            if len(ivs) < 2:
                continue
            iv_atm = float(np.median(ivs))
            rv_t = float(rv.loc[day]) if day in rv.index and not np.isnan(rv.loc[day]) else np.nan

            # horizon-matched: measure realised vol over the ATM bucket's own
            # life, not a fixed 21 days (Burghardt & Lane 1990)
            horizon = dte_to_trading_days(float(np.median(g["dte"])))
            rv_h = cones.matched_rv(und, day, horizon)
            cone_pct = cones.position(und, day, iv_atm, horizon)
            cone_z = cones.zscore(und, day, iv_atm, horizon)

            rows.append({"underlying": und, "date": day, "spot": S,
                         "iv_atm": iv_atm * 100,
                         "rv21": rv_t * 100 if rv_t == rv_t else np.nan,
                         "spread": (iv_atm - rv_t) * 100 if rv_t == rv_t else np.nan,
                         "atm_horizon": horizon,
                         "rv_matched": rv_h * 100 if rv_h == rv_h else np.nan,
                         "spread_h": (iv_atm - rv_h) * 100 if rv_h == rv_h else np.nan,
                         "cone_pct": cone_pct, "cone_z": cone_z})
    hist = pd.DataFrame(rows).sort_values(["underlying", "date"])
    for col, out in (("spread", "spread_pctile"), ("spread_h", "spread_h_pctile")):
        hist[out] = (hist.groupby("underlying")[col]
                     .transform(lambda s: s.expanding(60)
                                .apply(lambda w: (w[:-1] < w[-1]).mean(),
                                       raw=True) * 100))
    return hist


# ── latest chain metrics ─────────────────────────────────────────────────────
def build_chain_latest(chains, spots, r_curve, hist, cones=None):
    cones = cones or ConeBook(spots)
    rows = []
    latest_any = chains["refdate"].max()
    for und, g_und in chains.groupby("underlying"):
        spot = spots.get(und)
        if spot is None:
            continue
        day = g_und["refdate"].max()
        if day < latest_any - pd.Timedelta(days=STALE_DAYS):
            # delisted or dead feed (VALE5 stopped in 2017): not a live chain
            print(f"  {und}: last chain {day.date()}, skipping as stale")
            continue
        if day not in spot.index:
            day2 = spot.index[spot.index <= day]
            if len(day2) == 0:
                continue
            S = float(spot.loc[day2[-1]])
        else:
            S = float(spot.loc[day])
        r = float(r_curve.loc[day]) if day in r_curve.index else 0.10
        rv_series = np.log(spot).diff().rolling(21).std() * np.sqrt(ANN)
        rv = float(rv_series.reindex([day]).ffill().iloc[-1]) if len(rv_series) else np.nan

        g = g_und[g_und["refdate"] == day].copy()
        g["dte"] = (g["maturity_date"] - day).dt.days
        g = g[g["dte"] > 0]

        bid = pd.to_numeric(g.get("best_bid"), errors="coerce")
        ask = pd.to_numeric(g.get("best_ask"), errors="coerce")
        g["mid"] = np.where((bid > 0) & (ask >= bid), (bid + ask) / 2, g["close"])

        # one matched-RV / cone reading per distinct horizon on this chain
        horizons = {int(d): dte_to_trading_days(d) for d in g["dte"].unique()}
        matched = {h: cones.matched_rv(und, day, h) for h in set(horizons.values())}

        for row in g.itertuples():
            T = row.dte / 365.0
            F = S * np.exp(r * T)
            intr = max(S - row.strike_price, 0) if row.type == "C" \
                else max(row.strike_price - S, 0)
            iv = implied_vol(row.mid, S, row.strike_price, T, r, row.type)
            rows.append({
                "underlying": und, "date": day, "symbol": row.symbol,
                "type": row.type, "K": row.strike_price, "expiry": row.maturity_date,
                "dte": row.dte, "S": S, "mid": row.mid, "close": row.close,
                "contracts": row.traded_contracts,
                "moneyness": float(np.log(row.strike_price / F)),
                "iv": iv * 100 if iv == iv else np.nan,
                "delta": bs_delta(S, row.strike_price, T, r, iv, row.type)
                         if iv == iv else np.nan,
                # legacy fixed-21d spread: correct only near 21 trading days
                "iv_minus_rv": (iv - rv) * 100 if (iv == iv and rv == rv) else np.nan,
                # horizon-matched replacements
                "rv_horizon": horizons[int(row.dte)],
                "rv_matched": (matched[horizons[int(row.dte)]] * 100
                               if matched[horizons[int(row.dte)]] ==
                               matched[horizons[int(row.dte)]] else np.nan),
                "iv_minus_rv_h": ((iv - matched[horizons[int(row.dte)]]) * 100
                                  if (iv == iv and matched[horizons[int(row.dte)]] ==
                                      matched[horizons[int(row.dte)]]) else np.nan),
                "cone_pct": cones.position(und, day, iv, horizons[int(row.dte)]),
                "cone_z": cones.zscore(und, day, iv, horizons[int(row.dte)]),
                "below_intrinsic": bool(row.mid < intr - 0.01),
                "intrinsic": intr,
            })
    chain = pd.DataFrame(rows)

    # smile residual: quadratic IV(k) fit per (underlying, expiry), >=5 pts
    # (explicit loop, not groupby.apply — pandas 3 drops grouping columns there)
    chain["smile_resid"] = np.nan
    for _, idx in chain.groupby(["underlying", "expiry"]).groups.items():
        gg = chain.loc[idx].dropna(subset=["iv"])
        if len(gg) < 5:
            continue
        c = np.polyfit(gg["moneyness"], gg["iv"], 2)
        chain.loc[idx, "smile_resid"] = (chain.loc[idx, "iv"]
                                         - np.polyval(c, chain.loc[idx, "moneyness"]))

    # same-strike call-put IV gap and parity residual
    piv = chain.pivot_table(index=["underlying", "expiry", "K"], columns="type",
                            values=["iv", "mid"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["cp_gap"] = piv.get("iv_C", np.nan) - piv.get("iv_P", np.nan)
    chain = chain.merge(piv[["underlying", "expiry", "K", "cp_gap", "mid_C", "mid_P"]],
                        on=["underlying", "expiry", "K"], how="left")
    # parity residual: (C-P) - (S - K e^{-rT}), using per-underlying r
    rT = chain["dte"] / 365.0
    r_map = chain["date"].map(lambda d: float(r_curve.loc[d])
                              if d in r_curve.index else 0.10)
    chain["parity_resid"] = (chain["mid_C"] - chain["mid_P"]
                             - (chain["S"] - chain["K"] * np.exp(-r_map * rT)))

    # attach underlying-level context
    ctx = hist.sort_values("date").groupby("underlying").last().reset_index()
    ctx_cols = [c for c in ["underlying", "iv_atm", "rv21", "spread",
                            "spread_pctile", "rv_matched", "spread_h",
                            "spread_h_pctile", "cone_pct", "cone_z",
                            "atm_horizon"]
                if c in ctx.columns]
    chain = chain.merge(ctx[ctx_cols], on="underlying", how="left",
                        suffixes=("", "_und"))
    return chain


def quick_iv_chain(chains, spots, r_curve, und, day):
    """Minimal (moneyness, dte, iv) chain for one underlying/day — used to
    build the t-1 surface for the cVolGAN score."""
    spot = spots[und]
    if day not in spot.index:
        return pd.DataFrame()
    S = float(spot.loc[day])
    r = float(r_curve.loc[day]) if day in r_curve.index else 0.10
    g = chains[(chains["underlying"] == und) & (chains["refdate"] == day)].copy()
    g["dte"] = (g["maturity_date"] - day).dt.days
    g = g[(g["dte"] > 0) & (g["close"] > 0)]
    rows = []
    for row in g.itertuples():
        T = row.dte / 365.0
        F = S * np.exp(r * T)
        iv = implied_vol(row.close, S, row.strike_price, T, r, row.type)
        if iv == iv:
            rows.append({"moneyness": float(np.log(row.strike_price / F)),
                         "dte": row.dte, "iv": iv * 100})
    return pd.DataFrame(rows)


def attach_volgan_score(chain, chains, spots, r_curve):
    """volgan_pctile for IBOV rows: percentile of today's IV in the cVolGAN
    conditional distribution (low = cheap vs model, high = rich)."""
    chain["volgan_pctile"] = np.nan
    try:
        from volgan_score import volgan_percentiles
    except Exception as e:
        print(f"  volgan score unavailable: {e}")
        return chain
    ib = chain[chain["underlying"] == "IBOV"]
    if ib.empty:
        return chain
    day = ib["date"].max()
    prev_days = sorted(chains[(chains["underlying"] == "IBOV")
                              & (chains["refdate"] < day)]["refdate"].unique())
    if not prev_days:
        return chain
    tm1 = pd.Timestamp(prev_days[-1])
    chain_tm1 = quick_iv_chain(chains, spots, r_curve, "IBOV", tm1)
    f = volgan_percentiles(ib, chain_tm1, spots["IBOV"].loc[:tm1])
    if f is None:
        print("  volgan score: insufficient surface data")
        return chain
    mask = chain["underlying"] == "IBOV"
    chain.loc[mask, "volgan_pctile"] = f(
        chain.loc[mask, "moneyness"].values,
        chain.loc[mask, "dte"].values / 365.0)
    print(f"  volgan score attached (cond date {tm1.date()})")
    return chain


def main():
    print("Building screener metrics ...")
    r_curve = load_rates()
    chains = load_chains()
    spots = load_spots()
    print(f"  chains: {len(chains):,} rows, underlyings "
          f"{sorted(chains['underlying'].unique())}")

    cones = ConeBook(spots)          # shared cache across both passes
    hist = build_history(chains, spots, r_curve, cones)
    pq.write_table(pa.Table.from_pandas(hist, preserve_index=False),
                   str(OUT / "history_daily.parquet"))
    print(f"  history_daily: {len(hist):,} rows")

    chain = build_chain_latest(chains, spots, r_curve, hist, cones)
    chain = attach_volgan_score(chain, chains, spots, r_curve)
    pq.write_table(pa.Table.from_pandas(chain, preserve_index=False),
                   str(OUT / "chain_latest.parquet"))
    for und, g in chain.groupby("underlying"):
        print(f"  {und}: {len(g)} options on {g['date'].max().date()}, "
              f"{g['iv'].notna().sum()} with IV, "
              f"{g['cone_pct'].notna().sum()} with a cone reading")

    # freshness guard: the nightly job exits 0 on "no new file" (holidays), so
    # a feed that quietly stops would otherwise never surface. "::warning::" is
    # picked up by GitHub Actions as an annotation on the run.
    newest = chain["date"].max()
    age = len(pd.bdate_range(newest, pd.Timestamp.today().normalize())) - 1
    if age > FRESH_BDAYS:
        print(f"::warning::screener data is {age} business days old "
              f"(newest chain {newest.date()}) — check B3 ingest")
    print("Done.")


if __name__ == "__main__":
    main()
