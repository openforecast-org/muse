#!/usr/bin/env Rscript
# Dump R summary.pts coefficient table + variance proportions for parity.
suppressMessages(devtools::load_all("../..", quiet = TRUE))

specs <- list(
    list(name = "1LT", model = "1LT"),
    list(name = "1DT", model = "1DT"),       # has Damping
    list(name = "1GT", model = "1GT"),       # deterministic Slope row
    list(name = "ZZZ", model = "ZZZ"),
    list(name = "1LT_ar1", model = "1LT", ar = 1, ma = 0)
)
y <- as.numeric(AirPassengers)
out <- list()
for (sp in specs) {
    ord <- list(ar = if (is.null(sp$ar)) 0 else sp$ar,
                ma = if (is.null(sp$ma)) 0 else sp$ma, select = FALSE)
    m <- pts(y, model = sp$model, lags = 12, h = 0, orders = ord)
    s <- summary(m)
    cm <- s$coefficients
    pm <- s$proportions
    out[[sp$name]] <- list(
        coef_names = rownames(cm),
        estimate   = as.numeric(cm[, "Estimate"]),
        std_error  = as.numeric(cm[, "Std. Error"]),
        lower      = as.numeric(cm[, "Lower"]),
        upper      = as.numeric(cm[, "Upper"]),
        prop_names = rownames(pm),
        proportion = as.numeric(pm[, "Proportion"]),
        prop_se    = as.numeric(pm[, "Std. Error"])
    )
}
jsonlite::write_json(out, "summary_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null")
cat("Wrote summary_reference.json with", length(out), "cases\n")
