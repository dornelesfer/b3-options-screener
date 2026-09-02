# Deploying the B3 Options Screener

Architecture: **GitHub repo + GitHub Actions nightly refresh + Streamlit
Community Cloud** (free tier is enough). The app serves EOD data; the nightly
Action keeps it current — B3 publishes the daily COTAHIST file in the evening
(BRT), the Action ingests it at 22:30 BRT and commits the refreshed parquet
caches, and Streamlit Cloud redeploys automatically on push.

## One-time setup (user actions)

1. Create a **private** GitHub repo and push this folder — **DONE**:
   `github.com/dornelesfer/b3-options-screener`, branch `master`, nightly
   Action live since 2026-08-09. (For reference, the original command:)

```bash
cd /Users/fd/volgan_b3_starter
git init && git add -A && git commit -m "B3 options screener"
gh repo create b3-options-screener --private --source . --push
```

   The `.gitignore` keeps the 3GB `rb3_repository`, legacy `raw/processed`
   dumps, models and logs OUT of the repo. What ships: the compact parquet
   caches (~100MB), the app, the pipeline scripts, and the cVolGAN weights.

2. On https://share.streamlit.io → New app → pick the repo →
   main file `app_options_screener.py`. (For a private repo, grant the
   Streamlit GitHub app access when prompted.)

3. Actions tab → enable workflows → run **nightly-data-refresh** once manually
   (workflow_dispatch) and confirm it commits and the app redeploys.

## Data freshness contract

| Piece | Freshness | Source |
|---|---|---|
| Option chains (IBOV, PETR4, VALE3, BRAV3) | last B3 trading day | `update_daily_data.py`, COTAHIST_D direct from B3, pure Python |
| CDI / rates | daily | BCB SGS API (`download_bcb_series.py`) |
| IBOV / equity spots | last trading day | COTAHIST spot rows (+ BCB/Yahoo history) |
| Screener metrics | rebuilt nightly | `screener_metrics.py` |

Intraday data is NOT available on this stack — everything is end-of-day.

## Local run

```bash
python3 -m streamlit run app_options_screener.py --server.port 8601
```

## Adding an underlying

Add its spot symbol to `SPOT_SYMBOLS` and its ISIN (or symbol-prefix fallback)
in `update_daily_data.py`; it flows through metrics and the app automatically
after the next refresh. Options are matched to underlyings via the ISIN carried
on COTAHIST option rows.

## Troubleshooting

- Action found no new file → B3 holiday, or the run fired before B3 published;
  the next night catches up (the updater loops all missing days).
- App shows stale dates → check the Action's commit history; the app caption
  prints per-underlying as-of dates.
- Local pandas can't read/write parquet directly on the dev Mac (old pyarrow) —
  scripts already use `pyarrow.parquet` everywhere; keep doing that.
