# HANDOFF — B3 Options Screener: state, architecture, and path to deployment

> Written for any future Claude session (or human) picking this up cold.
> Read this + `MEMORY.md` files in `~/.claude/projects/-Users-fd-Prediction-Markets/memory/`
> before touching anything. Last updated: 2026-08 (screener built and verified locally).

## What this project is

A research pipeline on B3 (Brazilian exchange) options that culminated in:
1. A **vol-risk-premium short-strangle strategy** on IBOV index options
   (Sharpe ~0.85 backtested — the flagship; see `backtest_short_vol_v2.py`).
2. A **Streamlit options screener** (`app_options_screener.py`) showing per-option
   cheap/rich metrics with filters by underlying. Built, styled (dark, dataviz
   palette), verified in browser on port 8601.
3. The **user's goal now: deploy the screener with self-updating data.**

Owner context: BRL-based investor; portfolio = VT + NTN-B (IPCA+) + this overlay;
applying for Cayman PR (CUC position — unrelated to this repo). Full history in
memory files: `volgan-b3-vrp-pipeline.md`, `portfolio-lab-vt-tests.md`.

## Architecture (all paths relative to /Users/fd/volgan_b3_starter)

```
DATA LAYER
  data/rb3_repository/…            3GB parquet repo (rb3 R package cache, 2000-2026)
  data/ibov_options_all.parquet    IBOV index options cache (bdi 74/75)
  data/equity_options.parquet      PETR4/VALE3/VALE5 options (bdi 78/82, by ISIN)
  data/equity_spot.parquet         their spots (bdi 02)
  data/brav3_*.csv                 BRAV3 chain+spot (from the OPA investigation)
  data/rates_cdi.csv               daily CDI/SELIC (BCB SGS 12/11) — refetch: download_bcb_series.py
  data/ibov_daily.csv              IBOV closes (BCB SGS 7 + Yahoo ^BVSP merged)
  data/spot_yahoo_*.csv            Yahoo spot backfills (PETR4, VALE3, BRAV3)

METRICS LAYER
  screener_metrics.py              builds data/screener/{chain_latest,history_daily}.parquet
                                   metrics: iv, iv_minus_rv, smile_resid, cp_gap,
                                   parity_resid, below_intrinsic (+ volgan_pctile if
                                   volgan_score.py present — see WORK REMAINING)
  volgan_score.py                  (in progress) cVolGAN conditional RV score, IBOV only.
                                   Loads results/volgan_cond_G_weights.npy — a numpy
                                   dict {l1_W,l1_b,l2_W,l2_b,l3_W,l3_b}; generator is
                                   3 Linear layers 62→128→256→28 with softplus on the
                                   two hidden layers, linear out. Norm stats in
                                   results/volgan_cond_norm.npy {cond_mean,cond_std,
                                   tgt_mean,tgt_std}. Condition vec (30): [r_{t-1},
                                   r_{t-2}, RV21_{t-1}, logIV surface t-1 (27 = 9k×3T)].
                                   Output (28): [r_t, ΔlogIV (27)]. Grids: K_GRID=
                                   [-.24..-.24 step .06] 9 pts, T_GRID=[1,2,3]/12.
                                   Source of truth: train_volgan_br_conditional.py.

APP LAYER
  app_options_screener.py          Streamlit app; reads data/screener/*.parquet.
                                   Cache keyed on parquet mtime. Dark theme pinned in
                                   .streamlit/config.toml. Palette: calls #3987e5,
                                   puts #d95926 (dataviz skill dark steps).
  Launch: python3 -m streamlit run app_options_screener.py --server.port 8601
  (launch.json config "options-screener" in /Users/fd/Prediction Markets/.claude/)

UPDATE LAYER (the deployment blocker — see WORK REMAINING)
  update_daily_data.py             (in progress) pure-Python daily COTAHIST fetcher:
                                   downloads COTAHIST_D{DDMMYYYY}.ZIP from
                                   https://bvmf.bmfbovespa.com.br/InstDados/SerHist/,
                                   fixed-width parse, appends to the three parquet
                                   caches, dedupes on (refdate,symbol). Kills the R
                                   dependency for daily increments (rb3/R only needed
                                   for historical rebuilds).
```

## COTAHIST fixed-width layout (1-based positions, TIPREG "01" rows)

| field | pos | notes |
|---|---|---|
| DATA (refdate) | 3–10 | AAAAMMDD |
| CODBDI | 11–12 | 02 spot lot; 74/75 index C/P; 78/82 equity C/P |
| CODNEG (symbol) | 13–24 | |
| TPMERC | 25–27 | 010 spot, 070 call, 080 put |
| ESPECI (spec) | 40–49 | |
| PREULT (close) | 109–121 | ÷100 |
| PREOFC (best bid) | 122–134 | ÷100 |
| PREOFV (best ask) | 135–147 | ÷100 |
| TOTNEG | 148–152 | trades |
| QUATOT (contracts) | 153–170 | |
| VOLTOT (volume) | 171–188 | ÷100 |
| PREEXE (strike) | 189–201 | ÷100 |
| DATVEN (maturity) | 203–210 | AAAAMMDD |
| CODISI (ISIN) | 231–242 | for OPTIONS this is the UNDERLYING's ISIN |

Tracked ISINs: PETR4=BRPETRACNPR6, VALE3=BRVALEACNOR0, BRAV3=BRBRAVACNOR1
(verify BRAV3 ISIN from a data row — it was matched by symbol prefix in the
investigation). IBOV index options: filter specification startswith "IBO" +
bdi 74/75 (spec changed to "IBO/" in Mar-2025 — always use startswith).

## Deployment design (agreed with user)

Target: **GitHub repo + GitHub Actions nightly + Streamlit Community Cloud.**
- Nightly Action (~22:30 BRT, after B3 publishes EOD): run `update_daily_data.py`
  (fetch new COTAHIST_D files, append caches) → `download_bcb_series.py` (rates)
  → `screener_metrics.py` → commit refreshed `data/screener/*.parquet` (small)
  + incremental cache parquets. Big historical parquets (rb3_repository) stay OUT
  of the repo (.gitignore) — only the compact caches go in.
- Streamlit Cloud auto-redeploys on push → app always serves last EOD data.
- Answer to "will deployed version be up to date": YES once the nightly job is
  live; data is EOD (B3 publishes after close), never intraday.

## STATUS (updated 2026-09-01)

DEPLOYED (2026-08-09) and running:
- Repo: github.com/dornelesfer/b3-options-screener (private, branch `master`).
  The nightly Action has committed data every trading day since 2026-08-12;
  one failure (08-11, BCB API 502 killed the job — now fail-soft, see below).
- Streamlit Community Cloud connection is a browser/OAuth step only the user
  can do (share.streamlit.io → New app → this repo → `app_options_screener.py`).
  Whether it has been done is not visible from the repo.
- Local dev stack is Python 3.9 / pandas 2 / Streamlit 1.32; CI and Cloud run
  3.12 / pandas 3 / Streamlit 1.6x. Test on the modern stack before pushing:
  `uv venv /tmp/b3venv --python 3.12 && uv pip install -r requirements.txt`.
  (pandas 3 made `groupby.apply` drop grouping columns — fixed 08-09.)

Added 2026-09-01 (uncommitted at time of writing — verify, commit, push):
- Horizon-matched vol metrics (Burghardt & Lane cones): `rv_matched`,
  `iv_minus_rv_h`, `cone_pct`, `cone_z` on chain_latest, plus `spread_h*` on
  history_daily. `cone_z` is the default ranking metric. Read near the money.
- App: `Max |log-moneyness|` filter (default 0.30) — deep-ITM prints carry IVs
  of 100-250% on centavos of extrinsic value and their inflated IV drags delta
  back toward 0.5, so delta cannot screen them; moneyness can. App tolerates a
  parquet that predates new columns (code lands on Cloud before the nightly
  rebuild). Streamlit width API shim for 1.32 vs 1.50+.
- Pipeline hardening: `update_daily_data.py` stops (exit 0, `::warning::`) on
  a non-404 B3 error instead of crashing, and never ingests a later day ahead
  of a failed one (that hole would be permanent). `download_bcb_series.py` is
  incremental (`--full` to rebuild), fail-soft, and append-only on the CSVs
  (`float_precision="round_trip"` — otherwise every row re-serialises with a
  different last digit). `screener_metrics.py` skips underlyings whose chain is
  >30d stale (VALE5) and prints a `::warning::` if the newest chain is >3 bdays
  old. Workflow: two cron slots (22:47 / 01:47 BRT), concurrency guard,
  `git pull --rebase` before push, python 3.12, 30-min timeout.

DONE and verified (2026-08-08):
- `update_daily_data.py` WORKS — backfilled Apr→Aug (90 days) direct from B3,
  pure Python; all caches current through last close; dedupe verified.
- `download_bcb_series.py` now also merges Yahoo ^BVSP into ibov_daily.csv
  (BCB SGS-7 died in 2019 — this was a silent staleness bug, fixed).
- `screener_metrics.py` + app verified in browser with fresh data, all four
  underlyings same-day. App cache keyed on parquet mtime (NOTE: st.cache_data
  IGNORES underscore-prefixed args — param must be `mtime`, not `_mtime`).
- Deploy scaffolding written: `requirements.txt` (no torch needed),
  `.gitignore`, `.github/workflows/nightly.yml`, `README_DEPLOY.md`.
- `volgan_score.py` wired end-to-end (chain → surfaces → conditional samples →
  per-option percentile via griddata). Norm-stat arrays need np.ravel — shapes
  are (1,30)/(1,28) on disk.

## WORK REMAINING (in order)

1. **cVolGAN score is DEGENERATE — diagnose before trusting.** Scores cluster
   at ~100 for every option: the generator's conditional ΔlogIV dispersion is
   far narrower than realized day-to-day surface moves (classic WGAN
   weight-clipping (0.01) collapse, and/or the t-1→t interpolated-surface gap
   exceeds the model's noise floor). Diagnosis path: (a) sample the generator on a few
   historical conditions, compare std(ΔlogIV) per node vs realized std from
   surfaces_train.npy — expect model std ≈ 10x too small; (b) if confirmed,
   retrain with WGAN-GP (gradient penalty, no clipping) or reduce N_CRIT;
   (c) alternatively rescale sampled ΔlogIV to match realized vol-of-vol as a
   stopgap. The app marks the metric EXPERIMENTAL with a warning until fixed.
2. **Confirm Streamlit Community Cloud is connected** (user, browser only).
   GitHub side is done: repo, Action, nightly commits all live.
3. ~~Repo growth~~ DONE 2026-09-01: option caches partitioned by year
   (`data_cache.py`): `<name>.parquet` frozen through 2025, `<name>_2026.parquet`
   (3.6MB / 0.5MB) is all the nightly rewrites now — was 34MB + 3.4MB per
   night. All readers go through `load_options(name)`. After an rb3 rebuild
   with `extract_equity_options.py`, run `python data_cache.py` to re-split.
   Same commit: `volgan_score.surface_from_chain` now averages IV over
   duplicate (k,T) points — call and put at the same strike were fed to
   griddata as duplicates and the score depended on row order (diffs up to
   56 percentile points). Still EXPERIMENTAL for the dispersion reason above.
4. Nice-to-have: more underlyings (add ISIN to update_daily_data.py — flows
   through automatically); BOVA11 spot already collected for future use;
   consider a small "strategy signal" tile showing the strangle entry rule
   (spread pctile > 40 at ~30 DTE) from backtest_short_vol_v2.py. Untracked
   research scripts `election_risk_petr4.py`, `election_drawdown_detail.py`,
   `hedge_petr4.py` (PETR4 election hedge, priced on closes) are worth adding
   to the repo; their outputs go to results/ which is gitignored.

## Gotchas (learned the hard way)

- **pandas↔pyarrow mismatch on this machine**: `pd.read_parquet`/`to_parquet`
  FAIL. Always `pyarrow.parquet.pq.read_table(...).to_pandas()` and
  `pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)`.
- Anaconda python3 = 3.9. Streamlit 1.32, plotly 5.18 installed.
- B3 daily file for date D appears evening of D (BRT); weekends/holidays have no
  file — treat HTTP 404 as "no trading day", not error.
- COTAHIST daily ZIPs are latin-1 encoded; strip trailing whitespace on symbols.
- The old PCP backtest numbers (R$200M etc.) in results/ are ARTIFACTS — never
  quote them (see memory: volgan-b3-vrp-pipeline).
- BRAV3 options carry an OPA (Ecopetrol tender, auction 2026-08-05) — parity
  residuals and below-intrinsic flags on BRAV3 are the EVENT, not mispricing.
- Streamlit cache is keyed on chain_latest.parquet mtime — rebuilding metrics is
  enough; no app restart needed, just browser refresh.
