# Set up proper rb3 repository and download Brazilian options data

# IMPORTANT: Set the cache directory BEFORE loading rb3 package
cat("🚀 Setting up proper rb3 repository...\n")

# Set up repository in project directory
repo_path <- file.path(getwd(), "data", "rb3_repository")
cat("📁 Repository path:", repo_path, "\n")

# Create repository if it doesn't exist
if (!dir.exists(repo_path)) {
  dir.create(repo_path, recursive = TRUE)
  cat("✅ Created repository directory\n")
}

# Set the repository path for rb3 BEFORE loading the package
cat("🔧 Configuring rb3 cache directory...\n")
options(rb3.cachedir = repo_path)

# Now load the packages
cat("📦 Loading packages...\n")
library(rb3)
library(dplyr)

# Check if repository is working
cat("📋 Available templates:\n")
templates <- list_templates()
print(templates)

# Set reference date (today)
refdate <- Sys.Date()
cat("\n📅 Using reference date:", as.character(refdate), "\n")

# Create metadata for daily COTAHIST
cat("\n📥 Step 1: Downloading daily COTAHIST data...\n")
cat("   - Creating metadata for daily COTAHIST...\n")
daily_meta <- template_meta_create_or_load("b3-cotahist-daily", refdate = refdate)

# Download daily data
cat("   - Downloading daily COTAHIST files...\n")
daily_files <- download_marketdata(daily_meta)
cat("   - Downloaded daily files:", length(daily_files), "files\n")

# Create metadata for yearly COTAHIST
cat("\n📥 Step 2: Downloading yearly COTAHIST data...\n")
current_year <- as.numeric(format(refdate, "%Y"))
cat("   - Using year:", current_year, "\n")
yearly_meta <- template_meta_create_or_load("b3-cotahist-yearly", year = current_year)

# Download yearly data
cat("   - Downloading yearly COTAHIST files...\n")
yearly_files <- download_marketdata(yearly_meta)
cat("   - Downloaded yearly files:", length(yearly_files), "files\n")

# Check what files are in the repository
cat("\n📁 Repository contents:\n")
repo_files <- list.files(repo_path, recursive = TRUE, full.names = TRUE)
print(repo_files)

# Now try to read the data
cat("\n📊 Step 3: Reading downloaded data...\n")

# Try daily data
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
  
  # Combine the data
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
    
    # Save the data
    cat("💾 Saving data...\n")
    write.csv(all_options, "data/raw/b3_options_data.csv", row.names = FALSE)
    cat("✅ Data saved to data/raw/b3_options_data.csv\n")
    
    # Show sample
    cat("\n📋 Sample data:\n")
    print(head(all_options, 5))
    
  } else {
    cat("❌ No options data found\n")
  }
} else {
  cat("❌ Daily data is empty\n")
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
  }
} else {
  cat("❌ Yearly data is empty\n")
}

cat("\n🔍 Final Summary:\n")
cat("Repository path:", repo_path, "\n")
cat("Repository files:", length(repo_files), "\n")
cat("Downloaded - Daily:", length(daily_files), "Yearly:", length(yearly_files), "\n")
cat("Data records - Daily:", nrow(daily_data), "Yearly:", nrow(yearly_data), "\n")
