# Test fetch_marketdata function for COTAHIST data
cat("🚀 Testing fetch_marketdata function...\n")

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

# Check current year
current_year <- as.numeric(format(Sys.Date(), "%Y"))
cat("📅 Current year:", current_year, "\n")

# Try fetch_marketdata for yearly COTAHIST
cat("\n📥 Testing fetch_marketdata for yearly COTAHIST...\n")
cat("   - Fetching data for year:", current_year, "\n")

tryCatch({
  yearly_data <- fetch_marketdata("b3-cotahist-yearly", year = current_year)
  cat("✅ fetch_marketdata result:", nrow(yearly_data), "records\n")
  
  if (nrow(yearly_data) > 0) {
    cat("📋 Data columns:\n")
    print(colnames(yearly_data))
    
    # Filter for options data
    cat("🔍 Filtering for options data...\n")
    
    # Get equity options
    equity_options <- cotahist_filter_equity_options(yearly_data)
    cat("   - Equity options:", nrow(equity_options), "records\n")
    
    # Get index options  
    index_options <- cotahist_filter_index_options(yearly_data)
    cat("   - Index options:", nrow(index_options), "records\n")
    
    # Show sample data
    if (nrow(yearly_data) > 0) {
      cat("\n📋 Sample data:\n")
      print(head(yearly_data, 5))
    }
  }
  
}, error = function(e) {
  cat("❌ Error with fetch_marketdata:", e$message, "\n")
})

# Also try the previous year
cat("\n📥 Testing fetch_marketdata for previous year...\n")
cat("   - Fetching data for year:", current_year - 1, "\n")

tryCatch({
  prev_year_data <- fetch_marketdata("b3-cotahist-yearly", year = current_year - 1)
  cat("✅ fetch_marketdata result:", nrow(prev_year_data), "records\n")
  
  if (nrow(prev_year_data) > 0) {
    cat("📋 Data columns:\n")
    print(colnames(prev_year_data))
    
    # Show sample data
    cat("\n📋 Sample data:\n")
    print(head(prev_year_data, 5))
  }
  
}, error = function(e) {
  cat("❌ Error with fetch_marketdata:", e$message, "\n")
})

cat("\n🔍 Summary:\n")
cat("Repository path:", repo_path, "\n")
cat("Repository files:", length(list.files(repo_path, recursive = TRUE)), "\n")
