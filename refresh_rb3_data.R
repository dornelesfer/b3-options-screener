# refresh_rb3_data.R
# ==================
# Pulls the latest B3 COTAHIST data using the rb3 package.
# Run this on your local machine (needs R + rb3 installed).
#
# What this does:
#   1. Re-downloads / updates the yearly file for 2025 (which the local parquet
#      only covers through Aug 2025 — and the spec_code changed to 'IBO/' in Mar 2025)
#   2. Downloads 2026 data (January 2026 through today)
#
# After running, re-run backtest_vxbr.py to get the full up-to-date time series.
#
# Usage:
#   cd volgan_b3_starter
#   Rscript refresh_rb3_data.R

cat("=============================================================\n")
cat(" rb3 Data Refresh — pulling latest COTAHIST through today\n")
cat("=============================================================\n\n")

repo_path <- file.path(getwd(), "data", "rb3_repository")
options(rb3.cachedir = repo_path)

suppressPackageStartupMessages({
  library(rb3)
  library(dplyr)
})

current_year <- as.integer(format(Sys.Date(), "%Y"))
years_to_refresh <- 2025:current_year

cat("📅 Years to refresh:", paste(years_to_refresh, collapse=", "), "\n\n")

for (year in years_to_refresh) {
  cat(sprintf("📥 Fetching %d yearly COTAHIST file ...\n", year))
  tryCatch({
    # force = TRUE re-downloads even if cached, ensuring we get the latest data
    data <- fetch_marketdata("b3-cotahist-yearly", year = year, force = TRUE)
    if (!is.null(data) && nrow(data) > 0) {
      # Filter IBOV index options to preview
      df <- as.data.frame(data)
      df$spec_clean <- trimws(df$specification_code)
      ibov <- df[startsWith(df$spec_clean, "IBO") & df$bdi_code %in% c(74L, 75L), ]
      cat(sprintf("   ✅ %d total records, %d IBOV index option records\n",
                  nrow(df), nrow(ibov)))
      if (nrow(ibov) > 0) {
        ibov$refdate <- as.Date(ibov$refdate)
        cat(sprintf("   📅 IBOV options date range: %s to %s\n",
                    min(ibov$refdate), max(ibov$refdate)))
      }
    } else {
      cat(sprintf("   ⚠  No data returned for %d\n", year))
    }
  }, error = function(e) {
    cat(sprintf("   ❌ Error for %d: %s\n", year, e$message))
  })
  cat("\n")
}

cat("✅ Refresh complete.\n")
cat("   Now re-run:  python3 backtest_vxbr.py\n\n")
cat("NOTE: The 2025 parquet's spec_code changed from 'IBO' to 'IBO/' in March 2025.\n")
cat("      backtest_vxbr.py uses str.startswith('IBO') and handles both formats.\n")
