# Download Brazilian options data using rb3 package
library(rb3)
library(dplyr)

cat("🚀 Downloading Brazilian options data with rb3...\n")

# Check available templates
cat("📋 Available templates:\n")
templates <- list_templates()
print(templates)

# First, we need to download the data files
cat("\n📥 Step 1: Downloading market data files...\n")

# Set reference date (today)
refdate <- Sys.Date()
cat("   - Using reference date:", as.character(refdate), "\n")

# Create metadata for daily COTAHIST
cat("   - Creating metadata for daily COTAHIST...\n")
daily_meta <- template_meta_create_or_load("b3-cotahist-daily", refdate = refdate)

# Download daily data
cat("   - Downloading daily COTAHIST files...\n")
daily_files <- download_marketdata(daily_meta)
cat("   - Downloaded daily files:", length(daily_files), "files\n")

# Create metadata for yearly COTAHIST (requires year argument)
cat("   - Creating metadata for yearly COTAHIST...\n")
current_year <- as.numeric(format(refdate, "%Y"))
cat("   - Using year:", current_year, "\n")
yearly_meta <- template_meta_create_or_load("b3-cotahist-yearly", year = current_year)

# Download yearly data
cat("   - Downloading yearly COTAHIST files...\n")
yearly_files <- download_marketdata(yearly_meta)
cat("   - Downloaded yearly files:", length(yearly_files), "files\n")

# Now try to read the data
cat("\n📊 Step 2: Reading downloaded data...\n")

# Try daily data first (this should work since we downloaded files)
cat("📥 Reading daily COTAHIST data...\n")
daily_data <- cotahist_get("daily")
cat("✅ Daily data has", nrow(daily_data), "records\n")

if (nrow(daily_data) > 0) {
  cat("📋 Daily data columns:\n")
  print(colnames(daily_data))
  
  # Filter for options data
  cat("🔍 Filtering for options data...\n")
  
  # Get equity options
  equity_options <- cotahist_filter_equity_options(daily_data)
  cat("   - Equity options:", nrow(equity_options), "records\n")
  
  # Get index options  
  index_options <- cotahist_filter_index_options(daily_data)
  cat("   - Index options:", nrow(index_options), "records\n")
  
  # Combine the data (only if we have data)
  all_options <- data.frame()
  
  if (nrow(equity_options) > 0 || nrow(index_options) > 0) {
    if (nrow(equity_options) > 0 && nrow(index_options) > 0) {
      all_options <- bind_rows(
        equity_options %>% mutate(type = "equity"),
        index_options %>% mutate(type = "index")
      )
    } else if (nrow(equity_options) > 0) {
      all_options <- equity_options %>% mutate(type = "equity")
    } else {
      all_options <- index_options %>% mutate(type = "index")
    }
    
    cat("📊 Total options records:", nrow(all_options), "\n")
    
    # Show date range
    cat("📅 Date range in data:", 
        as.character(min(all_options$refdate, na.rm = TRUE)), "to",
        as.character(max(all_options$refdate, na.rm = TRUE)), "\n")
    
    # Save the data
    cat("💾 Saving data...\n")
    write.csv(all_options, "data/raw/b3_daily_options.csv", row.names = FALSE)
    cat("✅ Data saved to data/raw/b3_daily_options.csv\n")
    
    # Show sample of the data
    cat("\n📋 Sample data:\n")
    print(head(all_options, 5))
    
  } else {
    cat("❌ No options data found in daily COTAHIST\n")
    cat("   This might mean the data doesn't contain options or needs different filtering\n")
    
    # Show what data we do have
    cat("\n📊 Available data summary:\n")
    cat("   Total records:", nrow(daily_data), "\n")
    if ("bdi_code" %in% colnames(daily_data)) {
      cat("   BDI codes present:", unique(daily_data$bdi_code), "\n")
    }
    if ("instrument_market" %in% colnames(daily_data)) {
      cat("   Instrument markets:", unique(daily_data$instrument_market), "\n")
    }
  }
} else {
  cat("❌ Daily data is still empty after download\n")
  cat("   This suggests the data might need to be processed differently\n")
}

# Try yearly data as well
cat("\n📥 Reading yearly COTAHIST data...\n")
yearly_data <- cotahist_get("yearly")
cat("✅ Yearly data has", nrow(yearly_data), "records\n")

if (nrow(yearly_data) > 0) {
  cat("📋 Yearly data columns:\n")
  print(colnames(yearly_data))
  
  yearly_options <- cotahist_filter_equity_options(yearly_data)
  cat("   - Yearly equity options:", nrow(yearly_options), "records\n")
  
  if (nrow(yearly_options) > 0) {
    cat("✅ Found yearly options data:", nrow(yearly_options), "records\n")
    write.csv(yearly_options, "data/raw/b3_yearly_options.csv", row.names = FALSE)
    cat("✅ Yearly data saved to data/raw/b3_yearly_options.csv\n")
    
    cat("\n📋 Sample yearly data:\n")
    print(head(yearly_options, 5))
  } else {
    cat("❌ No options data available in yearly data\n")
  }
} else {
  cat("❌ Yearly data is still empty after download\n")
}

cat("\n🔍 Summary:\n")
cat("Downloaded files - Daily:", length(daily_files), "Yearly:", length(yearly_files), "\n")
cat("Data records - Daily:", nrow(daily_data), "Yearly:", nrow(yearly_data), "\n")
