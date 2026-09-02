"""
data_cache.py
=============
Partitioned option caches, so the nightly job stops rewriting 34MB of history.

  data/<name>.parquet         frozen history, everything before PARTITION_FROM.
                              Never touched by the nightly job.
  data/<name>_YYYY.parquet    one file per year from PARTITION_FROM on. The
                              nightly appends to the current year's file only,
                              so each data commit adds a few MB of pack instead
                              of a fresh copy of the whole history (the repo
                              grew 32->66MB in the first three weeks).

Names: "equity_options", "ibov_options_all". equity_spot.parquet is 300KB and
stays a single file.

Always read through `load_options(name)` — it concatenates history + year files
and dedupes on (refdate, symbol), so consumers never see the split.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path(__file__).parent / "data"
PARTITION_FROM = 2026
KEY = ["refdate", "symbol"]


def history_path(name):
    return DATA / f"{name}.parquet"


def year_path(name, year):
    return DATA / f"{name}_{year}.parquet"


def cache_files(name):
    """History file (if present) followed by year files in order."""
    files = [history_path(name)] if history_path(name).exists() else []
    files += sorted(DATA.glob(f"{name}_[0-9][0-9][0-9][0-9].parquet"))
    return files


def _read(path):
    df = pq.read_table(str(path)).to_pandas()
    df["refdate"] = pd.to_datetime(df["refdate"])
    return df


def _write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def load_options(name):
    """All rows across history + year files; None if nothing cached yet."""
    files = cache_files(name)
    if not files:
        return None
    out = pd.concat([_read(p) for p in files], ignore_index=True)
    return out.drop_duplicates(KEY, keep="last").reset_index(drop=True)


def append_options(name, new):
    """Merge `new` rows into the cache, rewriting only the year files they
    touch. Rows before PARTITION_FROM go to the history file (backfills only —
    the nightly never produces them). Returns rows written per file."""
    if new is None or new.empty:
        return {}
    new = new.copy()
    new["refdate"] = pd.to_datetime(new["refdate"])
    written = {}
    for year, chunk in new.groupby(new["refdate"].dt.year):
        path = history_path(name) if year < PARTITION_FROM else year_path(name, year)
        old = _read(path) if path.exists() else None
        out = chunk if old is None else pd.concat([old, chunk], ignore_index=True)
        out = out.drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
        _write(out, path)
        written[path.name] = len(out)
    return written


def split_history(name):
    """One-time migration: move rows >= PARTITION_FROM out of the history file
    into year files. Idempotent; prints what it did."""
    hp = history_path(name)
    if not hp.exists():
        print(f"{name}: no history file")
        return
    df = _read(hp)
    recent = df[df["refdate"].dt.year >= PARTITION_FROM]
    if recent.empty:
        print(f"{name}: history already ends before {PARTITION_FROM}")
        return
    frozen = df[df["refdate"].dt.year < PARTITION_FROM]
    append_options(name, recent)
    _write(frozen.sort_values(KEY).reset_index(drop=True), hp)
    print(f"{name}: history {len(frozen):,} rows (to {frozen['refdate'].max().date()}), "
          f"moved {len(recent):,} rows into year files")


if __name__ == "__main__":
    for n in ("equity_options", "ibov_options_all"):
        split_history(n)
