# Test cotahist_get after data has been downloaded
cat("🚀 Testing cotahist_get with downloaded data...\n")

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
cat("📁 Repository files:", length(list.files(repo_path, recursive = TRUE)), "\n")

# Try to get the data using cotahist_get
cat("\n📥 Testing cotahist_get for yearly data...\n")

tryCatch({
  yearly_data <- cotahist_get("yearly")
  cat("✅ cotahist_get result:", nrow(yearly_data), "records\n")
  
  if (nrow(yearly_data) > 0) {
    cat("📋 Data columns:\n")
    print(colnames(yearly_data))
    
    # Convert Arrow data to data frame
    cat("\n🔄 Converting Arrow data to data frame...\n")
    yearly_df <- as.data.frame(yearly_data)
    cat("✅ Converted to data frame:", nrow(yearly_df), "records\n")
    
    # Show sample data
    cat("\n📋 Sample data:\n")
    print(head(yearly_df, 5))
    
    # Filter for options data
    cat("\n🔍 Filtering for options data...\n")
    
    # Get equity options
    equity_options <- cotahist_filter_equity_options(yearly_df)
    cat("   - Equity options:", nrow(equity_options), "records\n")
    
    # Get index options  
    index_options <- cotahist_filter_index_options(yearly_df)
    cat("   - Index options:", nrow(index_options), "records\n")
    
    # Save the data if we have any
    if (nrow(equity_options) > 0 || nrow(index_options) > 0) {
      cat("\n💾 Saving options data...\n")
      
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
      
      write.csv(all_options, "data/raw/b3_options_data.csv", row.names = FALSE)
      cat("✅ Options data saved to data/raw/b3_options_data.csv\n")
      
      # Show sample of options data
      cat("\n📋 Sample options data:\n")
      print(head(all_options, 5))
    }
    
  } else {
    cat("❌ No data returned from cotahist_get\n")
  }
  
}, error = function(e) {
  cat("❌ Error with cotahist_get:", e$message, "\n")
})

# Also try daily data
cat("\n📥 Testing cotahist_get for daily data...\n")

tryCatch({
  daily_data <- cotahist_get("daily")
  cat("✅ cotahist_get daily result:", nrow(daily_data), "records\n")
  
  if (nrow(daily_data) > 0) {
    cat("📋 Daily data columns:\n")
    print(colnames(daily_data))
    
    # Convert to data frame
    daily_df <- as.data.frame(daily_data)
    cat("✅ Converted daily data to data frame:", nrow(daily_df), "records\n")
    
    # Show sample data
    cat("\n📋 Sample daily data:\n")
    print(head(daily_df, 5))
  } else {
    cat("❌ No daily data returned\n")
  }
  
}, error = function(e) {
  cat("❌ Error with cotahist_get daily:", e$message, "\n")
})

cat("\n🔍 Final Summary:\n")
cat("Repository path:", repo_path, "\n")
cat("Repository files:", length(list.files(repo_path, recursive = TRUE)), "\n")
