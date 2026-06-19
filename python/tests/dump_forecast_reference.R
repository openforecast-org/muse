#!/usr/bin/env Rscript
# Dump R forecast.pts outputs (mean + prediction/confidence/none intervals,
# incl. one- and two-sided and vector levels) for the Python forecaster to
# match.  Fixed-lambda and auto-lambda specs on AirPassengers.
suppressMessages(devtools::load_all("../..", quiet = TRUE))

specs <- list(
    list(name = "1LT", model = "1LT"),
    list(name = "0LT", model = "0LT"),       # lambda 0 (log) interval branch
    list(name = "0.5LT", model = "0.5LT"),
    list(name = "1DT", model = "1DT"),
    list(name = "ZZZ", model = "ZZZ"),
    list(name = "1LT_ar1", model = "1LT", ar = 1, ma = 0)
)

y <- as.numeric(AirPassengers)
h <- 12
out <- list()
for (sp in specs) {
    ord <- list(ar = if (is.null(sp$ar)) 0 else sp$ar,
                ma = if (is.null(sp$ma)) 0 else sp$ma, select = FALSE)
    m <- pts(y, model = sp$model, lags = 12, h = 0, orders = ord)
    rec <- list(lambda = m$lambda)
    fp  <- forecast(m, h = h, interval = "prediction", level = c(0.8, 0.95))
    rec$pred_mean  <- as.numeric(fp$mean)
    rec$pred_lower <- as.numeric(as.matrix(fp$lower))   # column-major flatten
    rec$pred_upper <- as.numeric(as.matrix(fp$upper))
    fc <- forecast(m, h = h, interval = "confidence", level = 0.95)
    rec$conf_lower <- as.numeric(as.matrix(fc$lower))
    rec$conf_upper <- as.numeric(as.matrix(fc$upper))
    fu <- forecast(m, h = h, interval = "prediction", level = 0.95, side = "upper")
    rec$up_lower <- as.numeric(as.matrix(fu$lower))
    rec$up_upper <- as.numeric(as.matrix(fu$upper))
    out[[sp$name]] <- rec
}
jsonlite::write_json(out, "forecast_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null")
cat("Wrote forecast_reference.json with", length(out), "cases\n")
