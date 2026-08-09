# fetch_recent_daily.R — pull the last few COTAHIST daily files into the repo
repo_path <- file.path(getwd(), "data", "rb3_repository")
options(rb3.cachedir = repo_path)
suppressPackageStartupMessages(library(rb3))

dates <- seq(Sys.Date() - 8, Sys.Date(), by = "day")
dates <- dates[!weekdays(dates) %in% c("Saturday", "Sunday", "sábado", "domingo")]

for (d in as.list(dates)) {
  cat(sprintf("fetching %s ... ", format(d)))
  tryCatch({
    fetch_marketdata("b3-cotahist-daily", refdate = d)
    cat("ok\n")
  }, error = function(e) cat(sprintf("skip (%s)\n", substr(e$message, 1, 60))))
}
cat("done\n")
