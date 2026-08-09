"""
extract_equity_options.py
=========================
One pass over the rb3 COTAHIST parquet repo, caching what the equity-options
alpha work needs:

  - data/equity_spot.parquet     : PETR4/VALE3 daily closes (BDI 02)
  - data/equity_options.parquet  : their options (BDI 78 calls / 82 puts),
                                   matched by underlying ISIN

COTAHIST option rows carry the *underlying's* ISIN, which is how we map
option series to PETR4 (BRPETRACNPR6) / VALE3 (BRVALEACNOR0) without
symbol-prefix guessing.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).parent
DATA = BASE / "data" / "rb3_repository" / "db" / "staging" / "b3-cotahist-yearly"

UNDERLYINGS = {
    "BRPETRACNPR6": "PETR4",
    "BRVALEACNOR0": "VALE3",
    # VALE was VALE5 (PNA, BRVALEACNPA3) before the 2017 share unification
    "BRVALEACNPA3": "VALE5",
}
SPOT_SYMBOLS = ["PETR4", "VALE3", "VALE5"]

COLS = ["refdate", "bdi_code", "symbol", "specification_code", "isin",
        "strike_price", "close", "best_bid", "best_ask",
        "maturity_date", "volume", "traded_contracts"]

spot_frames, opt_frames = [], []
for y in range(2000, 2027):
    p = DATA / f"year={y}" / "part-0.parquet"
    if not p.exists():
        continue
    df = pq.read_table(str(p), columns=COLS).to_pandas()

    spot = df[(df["bdi_code"] == 2) & (df["symbol"].isin(SPOT_SYMBOLS))][
        ["refdate", "symbol", "close", "volume"]].copy()

    opts = df[df["bdi_code"].isin([78, 82])
              & df["isin"].isin(UNDERLYINGS)
              & (df["strike_price"] > 0)].copy()
    opts["underlying"] = opts["isin"].map(UNDERLYINGS)

    spot_frames.append(spot)
    opt_frames.append(opts)
    print(f"  {y}: spot {len(spot):>4}  options {len(opts):>7,}")
    del df

spot = pd.concat(spot_frames, ignore_index=True)
spot["refdate"] = pd.to_datetime(spot["refdate"])
opts = pd.concat(opt_frames, ignore_index=True)
opts["refdate"] = pd.to_datetime(opts["refdate"])
opts["maturity_date"] = pd.to_datetime(opts["maturity_date"])

pq.write_table(pa.Table.from_pandas(spot, preserve_index=False),
               str(BASE / "data" / "equity_spot.parquet"))
pq.write_table(pa.Table.from_pandas(opts, preserve_index=False),
               str(BASE / "data" / "equity_options.parquet"))
print(f"\nSaved: equity_spot.parquet ({len(spot):,}) | "
      f"equity_options.parquet ({len(opts):,})")
print(opts.groupby("underlying").size())
