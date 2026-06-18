#!/usr/bin/env Rscript
# Dump R outliers="use" outputs (spike-injected AirPassengers) for parity.
suppressMessages(devtools::load_all("../..", quiet = TRUE))

cases <- list(
    list(name = "1LT_spike60", model = "1LT", idx = 60, mult = 2.0, level = 0.99),
    list(name = "ZZZ_spike100", model = "ZZZ", idx = 100, mult = 1.8, level = 0.99),
    list(name = "1LT_lvl95",   model = "1LT", idx = 60, mult = 2.0, level = 0.95)
)
out <- list()
for (cs in cases) {
    y <- as.numeric(AirPassengers); y[cs$idx] <- y[cs$idx] * cs$mult
    m <- pts(y, model = cs$model, lags = 12, h = 0,
             outliers = "use", level = cs$level)
    out[[cs$name]] <- list(
        model     = m$model,
        nParam    = m$nParam,
        logLik    = as.numeric(logLik(m)),
        coefNames = names(coef(m)),
        coef      = as.numeric(coef(m)),
        det_time  = as.integer(m$outliersDetected$time),
        det_type  = as.character(m$outliersDetected$type)
    )
}
jsonlite::write_json(out, "outlier_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null")
cat("Wrote outlier_reference.json with", length(out), "cases\n")
