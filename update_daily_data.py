"""
update_daily_data.py
====================
Pure-Python daily data updater — no R/rb3 needed for increments.

For every missing business day between the newest cached date and today:
  1. download COTAHIST_D{DDMMYYYY}.ZIP from B3
  2. fixed-width parse the TIPREG-01 rows
  3. filter to what the screener tracks:
       - IBOV index options  (bdi 74/75, spec startswith "IBO")
       - equity options for tracked underlying ISINs (bdi 78/82)
       - spot rows for tracked symbols (bdi 02)
  4. append to the parquet caches, deduped on (refdate, symbol)

Then run screener_metrics.py to refresh the app tables.

Usage:
  python3 update_daily_data.py            # catch up from last cached date
  python3 update_daily_data.py --days 10  # only look back N calendar days
"""

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).parent
DATA = BASE / "data"

B3_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{d}.ZIP"

TRACKED_ISINS = {}           # filled at runtime from existing cache (robust to
                             # ISIN guesses) + these known values as fallback
FALLBACK_ISINS = {
    "BRPETRACNPR6": "PETR4",
    "BRVALEACNOR0": "VALE3",
}
SPOT_SYMBOLS = ["PETR4", "VALE3", "BRAV3", "BOVA11"]
EQUITY_UNDERLYING_SYMBOL_PREFIXES = {"BRAV": "BRAV3"}   # prefix match fallback


def parse_cotahist_daily(raw: bytes) -> pd.DataFrame:
    """Fixed-width parse of a COTAHIST file (official layout, 1-based positions)."""
    rows = []
    for line in raw.decode("latin-1").splitlines():
        if len(line) < 245 or line[0:2] != "01":
            continue
        rows.append((
            line[2:10],                    # refdate AAAAMMDD
            int(line[10:12]),              # bdi_code
            line[12:24].strip(),           # symbol
            int(line[24:27]),              # tpmerc
            line[39:49].strip(),           # specification
            float(line[108:121]) / 100,    # close
            float(line[121:134]) / 100,    # best_bid
            float(line[134:147]) / 100,    # best_ask
            float(line[152:170]),          # traded contracts
            float(line[170:188]) / 100,    # volume
            float(line[188:201]) / 100,    # strike
            line[202:210],                 # maturity AAAAMMDD
            line[230:242].strip(),         # isin (underlying's, for options)
        ))
    df = pd.DataFrame(rows, columns=[
        "refdate", "bdi_code", "symbol", "tpmerc", "specification_code",
        "close", "best_bid", "best_ask", "traded_contracts", "volume",
        "strike_price", "maturity_date", "isin"])
    df["refdate"] = pd.to_datetime(df["refdate"], format="%Y%m%d", errors="coerce")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%Y%m%d",
                                         errors="coerce")
    return df.dropna(subset=["refdate"])


class FetchError(Exception):
    """Transient problem talking to B3 (5xx, timeout, truncated zip). Distinct
    from a 404, which is the normal 'no trading day / not published yet'."""


def fetch_day(day: pd.Timestamp):
    url = B3_URL.format(d=day.strftime("%d%m%Y"))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None                    # holiday / not published yet
        raise FetchError(f"HTTP {e.code} from B3") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise FetchError(f"network error talking to B3: {e}") from e
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            txt = [n for n in z.namelist() if n.upper().endswith(".TXT")]
            if not txt:
                raise FetchError("zip has no .TXT member")
            return parse_cotahist_daily(z.read(txt[0]))
    except zipfile.BadZipFile as e:
        raise FetchError(f"bad zip ({len(blob)} bytes)") from e


def load_cache(path):
    if path.exists():
        return pq.read_table(str(path)).to_pandas()
    return None


def save_cache(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def append_dedupe(cache, new, path):
    if cache is None:
        out = new
    else:
        out = pd.concat([cache, new], ignore_index=True)
        out = out.drop_duplicates(["refdate", "symbol"], keep="last")
    save_cache(out, path)
    return out


def discover_isins(eq_cache):
    """ISIN→underlying map from the existing cache + fallbacks."""
    m = dict(FALLBACK_ISINS)
    if eq_cache is not None and "isin" in eq_cache.columns:
        got = (eq_cache.dropna(subset=["isin"])
               .groupby("isin")["underlying"].first().to_dict())
        m.update({k: v for k, v in got.items() if isinstance(k, str) and k})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="only look back N calendar days (default: since cache)")
    args = ap.parse_args()

    ibo_p = DATA / "ibov_options_all.parquet"
    eq_p = DATA / "equity_options.parquet"
    spot_p = DATA / "equity_spot.parquet"

    ibo = load_cache(ibo_p)
    eq = load_cache(eq_p)
    spot = load_cache(spot_p)
    isin_map = discover_isins(eq)

    last_cached = max(
        (c["refdate"].max() for c in (ibo, eq) if c is not None and len(c)),
        default=pd.Timestamp("2026-01-01"))
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=args.days)
             if args.days else last_cached + pd.Timedelta(days=1))
    days = pd.bdate_range(start, pd.Timestamp.today().normalize())
    if len(days) == 0:
        print("Nothing to update — caches already current.")
        return

    print(f"Updating {len(days)} business days: "
          f"{days[0].date()} → {days[-1].date()}")
    n_ok = 0
    for day in days:
        try:
            df = fetch_day(day)
        except FetchError as e:
            # Stop here rather than skip: the next run resumes from the newest
            # cached date, so ingesting a LATER day now would leave this one
            # as a permanent hole. Exit 0 so the rest of the nightly pipeline
            # still refreshes on the data we do have.
            print(f"::warning::{day.date()}: {e} — stopping; will retry next run")
            break
        if df is None:
            print(f"  {day.date()}: no file (holiday or not yet published)")
            continue

        new_ibo = df[(df["bdi_code"].isin([74, 75]))
                     & df["specification_code"].str.startswith("IBO")].copy()

        opt = df[df["bdi_code"].isin([78, 82])].copy()
        opt["underlying"] = opt["isin"].map(isin_map)
        # prefix fallback for names whose ISIN we don't know yet
        for pre, und in EQUITY_UNDERLYING_SYMBOL_PREFIXES.items():
            mask = opt["underlying"].isna() & opt["symbol"].str.startswith(pre)
            opt.loc[mask, "underlying"] = und
            # learn the ISIN for next time
            isins = opt.loc[mask, "isin"].dropna().unique()
            for i in isins:
                if i:
                    isin_map[i] = und
        new_eq = opt.dropna(subset=["underlying"])

        new_spot = df[(df["bdi_code"] == 2)
                      & df["symbol"].isin(SPOT_SYMBOLS)].copy()

        drop = ["tpmerc"]
        ibo = append_dedupe(ibo, new_ibo.drop(columns=drop), ibo_p)
        eq = append_dedupe(eq, new_eq.drop(columns=drop), eq_p)
        new_spot = new_spot[["refdate", "symbol", "close", "volume"]]
        spot = append_dedupe(spot, new_spot, spot_p)

        n_ok += 1
        print(f"  {day.date()}: ibov_opts +{len(new_ibo)}, "
              f"eq_opts +{len(new_eq)}, spots +{len(new_spot)}")

    print(f"\n{n_ok} days ingested. Now run: python3 screener_metrics.py")


if __name__ == "__main__":
    main()
