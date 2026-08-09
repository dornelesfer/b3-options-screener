# Download full historical Brazilian options data using rb3 package
# Covers the date range from Milena Vuletic's paper: 2000-2024
cat("🚀 Downloading full historical B3 data (2000-2024)...\n")

# Set the cache directory BEFORE loading rb3 package
repo_path <- file.path(getwd(), "data", "rb3_repository")
cat("📁 Repository path:", repo_path, "\n")

# Set the repository path for rb3 BEFORE loading the package
cat("🔧 Configuring rb3 cache directory...\n")
options(rb3.cachedir = repo_path)

# Now load the packages
cat("📦 Loading packages...\n")
library(rb3)
library(dplyr)

# Check repository contents
cat("📁 Current repository files:", length(list.files(repo_path, recursive = TRUE)), "\n")

# Define the full date range for VolGAN training
start_year <- 2000
end_year <- 2024
cat("\n📅 Downloading data for years:", start_year, "to", end_year, "\n")
cat("📊 Total years to download:", end_year - start_year + 1, "\n")

# Download yearly COTAHIST data for the full range
cat("\n📥 Step 1: Downloading yearly COTAHIST data...\n")
download_summary <- data.frame()

for (year in start_year:end_year) {
  cat("   - Downloading year:", year, "... ")
  
  tryCatch({
    # Download data for this year
    yearly_data <- fetch_marketdata("b3-cotahist-yearly", year = year)
    
    if (nrow(yearly_data) > 0) {
      # Convert to data frame and get record count
      yearly_df <- as.data.frame(yearly_data)
      record_count <- nrow(yearly_df)
      
      # Filter for IBOV INDEX options ONLY (bdi_code 74 = calls, 75 = puts)
      # specification_code == 'IBO' is the canonical identifier for Ibovespa index options.
      # Equity options (bdi_code 78/82) must NOT be included — the VXBR methodology
      # (S&P/B3 Ibovespa VIX, following Cboe's VIX methodology) uses only IBOV index options.
      index_options <- cotahist_filter_index_options(yearly_df)

      total_options <- nrow(index_options)

      cat("✅", record_count, "total records,", total_options, "IBOV index options records\n")

      # Store summary
      download_summary <- rbind(download_summary, data.frame(
        year = year,
        total_records = record_count,
        equity_options = 0,       # intentionally excluded
        index_options = nrow(index_options),
        total_options = total_options,
        status = "Success"
      ))
      
    } else {
      cat("❌ No data available\n")
      download_summary <- rbind(download_summary, data.frame(
        year = year,
        total_records = 0,
        equity_options = 0,
        index_options = 0,
        total_options = 0,
        status = "No data"
      ))
    }
    
  }, error = function(e) {
    cat("❌ Error:", e$message, "\n")
    download_summary <- rbind(download_summary, data.frame(
      year = year,
      total_records = 0,
      equity_options = 0,
      index_options = 0,
      total_options = 0,
      status = paste("Error:", e$message)
    ))
  })
  
  # Small delay to be respectful to B3 servers
  Sys.sleep(1)
}

# Show download summary
cat("\n📊 Download Summary:\n")
print(download_summary)

# Calculate totals
total_records <- sum(download_summary$total_records)
total_options <- sum(download_summary$total_options)
successful_years <- sum(download_summary$status == "Success")

cat("\n🎯 Final Results:\n")
cat("   - Total years processed:", nrow(download_summary), "\n")
cat("   - Successful downloads:", successful_years, "\n")
cat("   - Total records downloaded:", total_records, "\n")
cat("   - Total options records:", total_options, "\n")

# Now try to read all the data
cat("\n📊 Step 2: Reading all downloaded data...\n")

tryCatch({
  # Get all yearly data
  all_yearly_data <- cotahist_get("yearly")
  cat("✅ Total yearly data records:", nrow(all_yearly_data), "\n")
  
  if (nrow(all_yearly_data) > 0) {
    # Convert to data frame
    cat("🔄 Converting to data frame...\n")
    all_df <- as.data.frame(all_yearly_data)
    cat("✅ Converted to data frame:", nrow(all_df), "records\n")
    
    # Filter for IBOV INDEX options ONLY
    # We strictly exclude equity options — VXBR uses only Ibovespa index options
    cat("🔍 Filtering for IBOV index options only (excluding equity options)...\n")
    index_options <- cotahist_filter_index_options(all_df)

    cat("   - IBOV index options:", nrow(index_options), "records\n")
    cat("   - Equity options: EXCLUDED (not used for VXBR methodology)\n")

    # Save index options only
    if (nrow(index_options) > 0) {
      cat("\n💾 Saving IBOV index options dataset...\n")
      all_options <- index_options %>% mutate(type = "index")
      
      # Save the complete dataset
      write.csv(all_options, "data/raw/b3_complete_options_data.csv", row.names = FALSE)
      cat("✅ Complete options data saved to data/raw/b3_complete_options_data.csv\n")
      
      # Show data summary
      cat("\n📋 Data Summary:\n")
      cat("   - Total options records:", nrow(all_options), "\n")
      cat("   - Date range:", as.character(min(all_options$refdate, na.rm = TRUE)), "to", 
          as.character(max(all_options$refdate, na.rm = TRUE)), "\n")
      cat("   - Unique symbols:", length(unique(all_options$symbol)), "\n")
      
      # Show sample data
      cat("\n📋 Sample data:\n")
      print(head(all_options, 5))
      
    } else {
      cat("❌ No options data found in the complete dataset\n")
    }
    
  } else {
    cat("❌ No yearly data available\n")
  }
  
}, error = function(e) {
  cat("❌ Error reading complete data:", e$message, "\n")
})

cat("\n🔍 Final Summary:\n")
cat("Repository path:", repo_path, "\n")
cat("Repository files:", length(list.files(repo_path, recursive = TRUE)), "\n")
cat("Download summary saved in download_summary variable\n")
