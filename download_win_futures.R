# =============================================================================
# download_win_futures.R
# =============================================================================
# Downloads WIN (IBOV mini-futures) and IND (full-size IBOV futures) daily
# settlement prices from B3 using the rb3 R package.
#
# Output: data/win_futures_settlements.parquet
#          Columns: refdate, contract, maturity, settle_price, volume,
#                   open_interest, last_price
#
# Prerequisites (run once):
#   install.packages("rb3")
#   install.packages("bizdays")
#   install.packages("arrow")   # for parquet output
#   install.packages("tidyverse")
#
# Usage:
#   Rscript download_win_futures.R
# =============================================================================

suppressPackageStartupMessages({
  library(rb3)
  library(bizdays)
  library(arrow)
  library(dplyr)
  library(lubridate)
})

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE  <- as.Date("2010-01-04")
END_DATE    <- Sys.Date()    # today; rb3 will fetch up to what B3 has published
OUT_DIR     <- file.path(dirname(rstudioapi::getSourceEditorContext()$path %||% "."),
                         "data")
# If running from Rscript (not RStudio), use script directory:
args <- commandArgs(trailingOnly = FALSE)
script_dir <- tryCatch({
  dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
}, error = function(e) ".")
OUT_DIR <- file.path(script_dir, "data")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUT_FILE <- file.path(OUT_DIR, "win_futures_settlements.parquet")

cat("=================================================================\n")
cat("WIN/IND Futures Data Download via rb3\n")
cat(sprintf("  Date range: %s → %s\n", START_DATE, END_DATE))
cat(sprintf("  Output:     %s\n", OUT_FILE))
cat("=================================================================\n\n")

# ── Business calendar (B3/ANBIMA) ──────────────────────────────────────────────
# rb3 ships its own holiday calendar; use it
load_builtin_calendars()
bizdays.options$set(default.calendar = "Brazil/ANBIMA")

biz_dates <- bizseq(START_DATE, END_DATE, "Brazil/ANBIMA")
cat(sprintf("Business days to process: %d\n\n", length(biz_dates)))

# ── Helper: fetch one day ──────────────────────────────────────────────────────
fetch_day <- function(refdate) {
  tryCatch({
    # fetch_marketdata downloads the BDI file for that date and caches it
    # bd_futures() extracts futures settlement prices
    df <- futures_get(
      refdate = refdate,
      cache_folder = file.path(OUT_DIR, "rb3_cache")
    )
    if (is.null(df) || nrow(df) == 0) return(NULL)

    # Filter WIN (IBOV mini) and IND (IBOV full)
    win_ind <- df %>%
      filter(grepl("^WIN|^IND", commodity)) %>%
      mutate(refdate = as.Date(refdate)) %>%
      select(
        refdate,
        contract     = symbol,
        commodity,
        maturity     = maturity_date,
        settle_price = price_previous_day,   # end-of-day settlement
        volume       = volume_contracts,
        open_interest
      )
    win_ind
  }, error = function(e) {
    message(sprintf("  [SKIP] %s: %s", refdate, condenseMessage(e)))
    NULL
  })
}

# ── Alternative approach: use cotahist_get + derivatives_get ──────────────────
# rb3 can fetch the full "COTAHIST" file (which has BDI codes including futures)
# OR use the BMF BDI file which has futures specifically.
# The most reliable method for futures is futures_get() using the BMF Ajuste files.

# Let's use the rb3-recommended approach for futures:
fetch_futures_rb3 <- function(refdate) {
  tryCatch({
    # yc_brl() fetches yield curve — but for futures prices, use:
    df <- futures_get(refdate = refdate,
                      cache_folder = file.path(OUT_DIR, "rb3_cache"))
    if (is.null(df) || nrow(df) == 0) return(NULL)

    # WIN = mini IBOV futures (multiplier R$0.20/pt)
    # IND = full IBOV futures (multiplier R$1.00/pt)
    df %>%
      filter(grepl("^(WIN|IND)", symbol)) %>%
      transmute(
        refdate      = as.Date(refdate),
        contract     = symbol,
        commodity    = substr(symbol, 1, 3),
        maturity     = maturity_date,
        settle_price = price_previous_day,
        volume       = volume_contracts,
        open_interest = open_interest
      )
  }, error = function(e) NULL)
}

# ── Main download loop ─────────────────────────────────────────────────────────
# Process in yearly batches and write incrementally
all_data <- list()

pb_total <- length(biz_dates)
for (i in seq_along(biz_dates)) {
  d <- biz_dates[i]

  if (i %% 50 == 1) {
    cat(sprintf("  Processing %s (%d/%d)...\n", d, i, pb_total))
  }

  result <- fetch_futures_rb3(d)
  if (!is.null(result) && nrow(result) > 0) {
    all_data[[length(all_data) + 1]] <- result
  }

  # Write checkpoint every 200 days
  if (i %% 200 == 0 && length(all_data) > 0) {
    combined <- bind_rows(all_data)
    arrow::write_parquet(combined, OUT_FILE)
    cat(sprintf("  [Checkpoint] %d rows saved\n", nrow(combined)))
  }
}

cat("\nFinalising...\n")
if (length(all_data) == 0) {
  cat("ERROR: No data downloaded. Check rb3 installation and B3 connectivity.\n")
  cat("\nTroubleshooting:\n")
  cat("  1. Run: rb3::futures_get(refdate = as.Date('2024-01-02'))\n")
  cat("  2. Check if B3's server is reachable from your network\n")
  cat("  3. Verify rb3 version: packageVersion('rb3')\n")
  quit(status = 1)
}

final <- bind_rows(all_data) %>%
  arrange(refdate, contract) %>%
  distinct()

cat(sprintf("\nDownload complete!\n"))
cat(sprintf("  Total rows:     %d\n", nrow(final)))
cat(sprintf("  Date range:     %s → %s\n",
            min(final$refdate), max(final$refdate)))
cat(sprintf("  Contracts:      %s\n",
            paste(sort(unique(final$commodity)), collapse = ", ")))
cat(sprintf("  Output file:    %s\n", OUT_FILE))

arrow::write_parquet(final, OUT_FILE)
cat("\nDone. Parquet file ready for pcp_3leg_win.py\n")

# ── Quick sanity check ────────────────────────────────────────────────────────
cat("\n--- Sample output (last 10 rows) ---\n")
print(tail(final, 10))

cat("\n--- Rows per year ---\n")
final %>%
  mutate(year = year(refdate)) %>%
  count(year, commodity) %>%
  tidyr::pivot_wider(names_from = commodity, values_from = n) %>%
  print()
