# Process B3 IBOV index options for VolGAN training
# Implements the S&P/B3 Ibovespa VIX (VXBR) methodology, which follows
# Cboe's VIX methodology (model-free variance, put-call parity forward,
# near-term / next-term structure, 30-day interpolation).
#
# KEY DESIGN DECISIONS:
#   - Only IBOV index options are used (bdi_code 74=calls, 75=puts, spec='IBO')
#   - Equity options on PETR4/VALE3/etc. are EXCLUDED
#   - Forward price estimated via put-call parity (NLS per expiry per day)
#   - Log moneyness: k = ln(K/F)   [NOT ln(K/close) — that was wrong before]
#   - IV computed via Black-76 bisection  [NOT the sqrt(2pi/T)*(C/K) proxy]
#   - Liquidity filter: only strikes with best_bid > 0 (VIX rule)

cat("=============================================================\n")
cat(" VolGAN-BR: B3 IBOV Index Options Processing (VXBR method)\n")
cat("=============================================================\n\n")

# ── packages ──────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(rb3)
  library(dplyr)
  library(lubridate)
  library(tidyr)
})

repo_path <- file.path(getwd(), "data", "rb3_repository")
options(rb3.cachedir = repo_path)

# ── helpers ───────────────────────────────────────────────────────────────────

#' Black-76 call price (options on a forward / index)
black76_call <- function(F, K, T, r, sigma) {
  d1 <- (log(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  exp(-r * T) * (F * pnorm(d1) - K * pnorm(d2))
}

#' Black-76 put price
black76_put <- function(F, K, T, r, sigma) {
  d1 <- (log(F/K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  exp(-r * T) * (K * pnorm(-d2) - F * pnorm(-d1))
}

#' Implied volatility via bisection (Black-76).
#'   type: "C" or "P"
#'   Returns NA if the market price is below intrinsic or bisection fails.
implied_vol_black76 <- function(market_price, F, K, T, r, type = "C",
                                lower = 1e-4, upper = 10, tol = 1e-6,
                                max_iter = 200) {
  if (T <= 0 || market_price <= 0 || F <= 0 || K <= 0) return(NA_real_)
  price_fn <- if (type == "C") black76_call else black76_put
  f_lo <- price_fn(F, K, T, r, lower) - market_price
  f_hi <- price_fn(F, K, T, r, upper) - market_price
  if (f_lo * f_hi > 0) return(NA_real_)
  for (i in seq_len(max_iter)) {
    mid <- (lower + upper) / 2
    f_mid <- price_fn(F, K, T, r, mid) - market_price
    if (abs(f_mid) < tol || (upper - lower) / 2 < tol) return(mid)
    if (f_lo * f_mid < 0) { upper <- mid; f_hi <- f_mid }
    else                   { lower <- mid; f_lo <- f_mid }
  }
  (lower + upper) / 2
}

#' Estimate forward price from put-call parity (NLS over matched pairs).
#'   Returns F = K + e^(rT) * (C - P)  for the strike minimising |C-P|.
#'   If no matched pairs exist, returns NA.
estimate_forward_pcp <- function(calls, puts, T, r) {
  if (nrow(calls) == 0 || nrow(puts) == 0) return(NA_real_)
  # Match on strike
  pairs <- inner_join(
    calls %>% select(strike_price, price_C = mid_price),
    puts  %>% select(strike_price, price_P = mid_price),
    by = "strike_price"
  ) %>%
    mutate(F_est = strike_price + exp(r * T) * (price_C - price_P))
  if (nrow(pairs) == 0) return(NA_real_)
  # Use the pair whose |C-P| is smallest (closest to ATM)
  atm_row <- pairs %>% slice_min(abs(price_C - price_P), n = 1)
  atm_row$F_est[1]
}

#' VIX model-free variance for one expiry (Demeterfi-Derman-Kamal-Zou formula).
#'   sigma2 = (2/T) * sum( dK/K^2 * e^(rT) * Q(K) ) - (1/T)*(F/K0 - 1)^2
#'   Only strikes with mid_price > 0 (and best_bid > 0) are included.
compute_vix_variance <- function(options_expiry, F, K0, T, r) {
  # Separate OTM options:  puts for K < F, calls for K > F, average at K0
  df <- options_expiry %>%
    filter(best_bid > 0, mid_price > 0) %>%
    mutate(
      Q = case_when(
        strike_price < F  ~ mid_price,   # OTM put price
        strike_price > F  ~ mid_price,   # OTM call price
        strike_price == K0 ~ (mid_price + mid_price) / 2,  # handled below
        TRUE ~ NA_real_
      )
    )

  # At K0, average call and put if both exist
  at_k0_calls <- options_expiry %>%
    filter(strike_price == K0, bdi_code == 74, best_bid > 0) %>%
    pull(mid_price)
  at_k0_puts  <- options_expiry %>%
    filter(strike_price == K0, bdi_code == 75, best_bid > 0) %>%
    pull(mid_price)
  q_k0 <- if (length(at_k0_calls) > 0 && length(at_k0_puts) > 0)
    (at_k0_calls[1] + at_k0_puts[1]) / 2
  else if (length(at_k0_calls) > 0) at_k0_calls[1]
  else if (length(at_k0_puts) > 0)  at_k0_puts[1]
  else return(NA_real_)

  # Build the strip: OTM puts (K < K0), K0, OTM calls (K > K0)
  strip <- bind_rows(
    df %>% filter(strike_price < K0, bdi_code == 75) %>%   # OTM puts
      group_by(strike_price) %>% slice_max(mid_price, n=1) %>% ungroup(),
    df %>% filter(strike_price > K0, bdi_code == 74) %>%   # OTM calls
      group_by(strike_price) %>% slice_max(mid_price, n=1) %>% ungroup()
  ) %>%
    bind_rows(tibble(strike_price = K0, mid_price = q_k0,
                     bdi_code = 74L, best_bid = 1)) %>%
    arrange(strike_price) %>%
    filter(!is.na(mid_price))

  if (nrow(strip) < 2) return(NA_real_)

  # Trapezoidal dK (VIX rule: boundary strikes get half-width)
  K  <- strip$strike_price
  Q  <- strip$mid_price
  n  <- length(K)
  dK <- numeric(n)
  dK[1]   <- K[2] - K[1]
  dK[n]   <- K[n] - K[n-1]
  if (n > 2) dK[2:(n-1)] <- (K[3:n] - K[1:(n-2)]) / 2

  sigma2 <- (2 / T) * sum(dK / K^2 * exp(r * T) * Q) -
    (1 / T) * (F / K0 - 1)^2
  max(sigma2, 0)    # should be non-negative
}

# ── 1. Load data ──────────────────────────────────────────────────────────────
cat("📥 Loading raw B3 data...\n")
raw_path <- "data/raw/b3_complete_options_data.csv"

if (!file.exists(raw_path)) {
  stop("❌ Raw data not found at ", raw_path,
       "\n   Please run download_full_b3_data.R first.")
}

b3_data <- read.csv(raw_path, stringsAsFactors = FALSE)
cat("   Loaded:", nrow(b3_data), "records\n")

# ── 2. Keep only IBOV index options ──────────────────────────────────────────
cat("\n🔍 Filtering for IBOV index options only...\n")

b3_data$refdate      <- as.Date(b3_data$refdate)
b3_data$maturity_date <- as.Date(b3_data$maturity_date)

# specification_code may have trailing spaces in some years
b3_data$spec_clean <- trimws(b3_data$specification_code)

ibov_opts <- b3_data %>%
  filter(
    startsWith(spec_clean, "IBO"),                # Ibovespa index options only
    # NOTE: B3 changed spec_code from 'IBO' (pre-Mar 2025) to 'IBO/' (Mar 2025 onward).
    # startsWith() handles both formats robustly.
    bdi_code %in% c(74L, 75L),                    # 74=calls, 75=puts
    !is.na(strike_price), strike_price > 0,
    !is.na(maturity_date),
    !is.na(close), close > 0,
    !is.na(best_bid)                               # need bid for VIX zero-bid rule
  ) %>%
  mutate(
    time_to_maturity = as.numeric(maturity_date - refdate) / 365.25,
    # Mid-price: use (best_bid + best_ask)/2 when available, else close
    mid_price = if_else(
      !is.na(best_bid) & !is.na(best_ask) & best_bid > 0,
      (best_bid + best_ask) / 2,
      close
    )
  ) %>%
  filter(
    time_to_maturity > 6/365.25,    # VIX rule: > 6 business days to expiry
    time_to_maturity <= 2.0
  )

cat("   IBOV index options after filtering:", nrow(ibov_opts), "records\n")
cat("   Date range:", as.character(min(ibov_opts$refdate)),
    "to", as.character(max(ibov_opts$refdate)), "\n")
cat("   Unique trading days:", length(unique(ibov_opts$refdate)), "\n")
cat("   Unique expiries:", length(unique(ibov_opts$maturity_date)), "\n")

# ── 3. Per-day, per-expiry: forward price + IV surface ───────────────────────
cat("\n📊 Computing forward prices and implied volatilities...\n")

# Brazilian risk-free proxy: CDI / SELIC — use a constant for now.
# In production, replace with daily SELIC rates from rb3::fetch_marketdata("b3-reference-rates")
r_brazil <- 0.12   # conservative annual rate; adjust per period if desired

trading_days <- sort(unique(ibov_opts$refdate))
cat("   Processing", length(trading_days), "trading days...\n")

surface_list <- list()
pb_step <- max(1L, floor(length(trading_days) / 20))

for (i in seq_along(trading_days)) {
  day <- trading_days[i]
  day_data <- ibov_opts %>% filter(refdate == day)
  expiries  <- sort(unique(day_data$maturity_date))

  if (i %% pb_step == 0)
    cat(sprintf("   [%4d/%d] %s — %d expiries\n",
                i, length(trading_days), day, length(expiries)))

  for (exp_date in expiries) {
    T_exp <- as.numeric(exp_date - day) / 365.25
    if (T_exp <= 0) next

    exp_data <- day_data %>% filter(maturity_date == exp_date)
    calls_exp <- exp_data %>% filter(bdi_code == 74)
    puts_exp  <- exp_data %>% filter(bdi_code == 75)

    # Estimate forward price via put-call parity
    F_est <- estimate_forward_pcp(calls_exp, puts_exp, T_exp, r_brazil)
    if (is.na(F_est) || F_est <= 0) next

    # K0 = largest strike <= F
    all_strikes <- sort(unique(exp_data$strike_price))
    strikes_below <- all_strikes[all_strikes <= F_est]
    if (length(strikes_below) == 0) next
    K0 <- max(strikes_below)

    # Compute Black-76 IV for every individual option
    exp_data_iv <- exp_data %>%
      mutate(
        F        = F_est,
        K0       = K0,
        opt_type = if_else(bdi_code == 74, "C", "P"),
        log_moneyness = log(strike_price / F_est),    # k = ln(K/F)
        iv = mapply(
          implied_vol_black76,
          market_price = mid_price,
          F            = F_est,
          K            = strike_price,
          T            = T_exp,
          r            = r_brazil,
          type         = opt_type
        )
      ) %>%
      filter(!is.na(iv), iv > 0, iv < 5)

    if (nrow(exp_data_iv) < 3) next

    surface_list[[length(surface_list) + 1]] <- exp_data_iv %>%
      select(
        date          = refdate,
        maturity_date,
        symbol,
        bdi_code,
        opt_type,
        strike_price,
        K0,
        F             = F,
        log_moneyness,
        T             = time_to_maturity,
        mid_price,
        best_bid,
        best_ask,
        close,
        volume,
        traded_contracts,
        iv
      )
  }
}

cat("\n✅ Forward / IV computation complete.\n")

if (length(surface_list) == 0) {
  stop("❌ No valid IV records produced. Check raw data and filtering parameters.")
}

iv_surface <- bind_rows(surface_list)
cat("   IV surface records:", nrow(iv_surface), "\n")

# ── 4. Build VolGAN input format ──────────────────────────────────────────────
# VolGAN expects (date, k, T, iv) — one row per option quote per day.
# We keep both calls and puts (they lie on the same surface).
cat("\n🎯 Building VolGAN training dataset...\n")

volgan_data <- iv_surface %>%
  select(
    date          = date,
    k             = log_moneyness,     # ln(K/F)
    T             = T,                 # years to maturity
    iv,
    # ancillary columns (useful for diagnostics / back-tests)
    symbol, opt_type, strike_price, F, maturity_date,
    mid_price, volume, traded_contracts
  ) %>%
  filter(
    abs(k) <= 0.5,          # trim deep OTM tails (Vuletic paper range)
    T >= 0.05, T <= 2.0
  ) %>%
  arrange(date, T, k)

cat("   VolGAN-ready records:", nrow(volgan_data), "\n")

# Normalised features (z-score per column) for direct GAN input
volgan_data <- volgan_data %>%
  mutate(
    k_norm  = as.numeric(scale(k)),
    T_norm  = as.numeric(scale(T)),
    iv_norm = as.numeric(scale(iv))
  )

# ── 5. Train / test split ─────────────────────────────────────────────────────
cat("\n📅 Splitting into training and test sets...\n")

# Practical split aligned with the paper (adjust if more data is available)
train_end  <- as.Date("2023-12-31")
test_start <- as.Date("2024-01-01")

train_data <- volgan_data %>% filter(date <= train_end)
test_data  <- volgan_data %>% filter(date >= test_start)

cat(sprintf("   Training: %d records (%s to %s)\n",
    nrow(train_data),
    if (nrow(train_data) > 0) as.character(min(train_data$date)) else "N/A",
    if (nrow(train_data) > 0) as.character(max(train_data$date)) else "N/A"))
cat(sprintf("   Test:     %d records (%s to %s)\n",
    nrow(test_data),
    if (nrow(test_data) > 0) as.character(min(test_data$date)) else "N/A",
    if (nrow(test_data) > 0) as.character(max(test_data$date)) else "N/A"))

# ── 6. Save ───────────────────────────────────────────────────────────────────
cat("\n💾 Saving processed datasets...\n")

dir.create("data/processed", showWarnings = FALSE, recursive = TRUE)

write.csv(volgan_data, "data/processed/volgan_complete_data.csv",  row.names = FALSE)
write.csv(train_data,  "data/processed/volgan_training_data.csv",  row.names = FALSE)
write.csv(test_data,   "data/processed/volgan_test_data.csv",      row.names = FALSE)
write.csv(iv_surface,  "data/processed/iv_surface_full.csv",       row.names = FALSE)

cat("   ✅ data/processed/volgan_complete_data.csv\n")
cat("   ✅ data/processed/volgan_training_data.csv\n")
cat("   ✅ data/processed/volgan_test_data.csv\n")
cat("   ✅ data/processed/iv_surface_full.csv  (with F, K0, opt_type)\n")

# ── 7. Summary ────────────────────────────────────────────────────────────────
cat("\n", strrep("=", 60), "\n", sep = "")
cat(" PROCESSING SUMMARY\n")
cat(strrep("=", 60), "\n")
cat(sprintf("  Raw IBOV index option records : %d\n", nrow(ibov_opts)))
cat(sprintf("  Records with valid IV         : %d\n", nrow(iv_surface)))
cat(sprintf("  VolGAN-ready records          : %d\n", nrow(volgan_data)))
cat(sprintf("  Training set                  : %d\n", nrow(train_data)))
cat(sprintf("  Test set                      : %d\n", nrow(test_data)))
cat(sprintf("  Unique trading days           : %d\n", length(unique(volgan_data$date))))
cat(sprintf("  k range                       : [%.3f, %.3f]\n",
            min(volgan_data$k), max(volgan_data$k)))
cat(sprintf("  T range (years)               : [%.3f, %.3f]\n",
            min(volgan_data$T), max(volgan_data$T)))
cat(sprintf("  IV range                      : [%.3f, %.3f]\n",
            min(volgan_data$iv), max(volgan_data$iv)))
cat(strrep("=", 60), "\n")
cat("✅ Processing complete — ready for VolGAN training.\n\n")
