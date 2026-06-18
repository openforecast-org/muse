#!/usr/bin/env Rscript
# Dump R diagnostics / confint / accuracy outputs for Python parity.
suppressMessages(devtools::load_all("../..", quiet = TRUE))

specs <- list(
    list(name = "1LT", model = "1LT"),
    list(name = "0LT", model = "0LT"),
    list(name = "1LT_ar1", model = "1LT", ar = 1, ma = 0),
    list(name = "ZZZ", model = "ZZZ")
)
y <- as.numeric(AirPassengers)
h <- 12
out <- list()
for (sp in specs) {
    ord <- list(ar = if (is.null(sp$ar)) 0 else sp$ar,
                ma = if (is.null(sp$ma)) 0 else sp$ma, select = FALSE)
    m <- pts(y, model = sp$model, lags = 12, h = h, holdout = TRUE, orders = ord)
    ci <- suppressWarnings(confint(m, level = 0.9))
    acc <- accuracy(m)
    out[[sp$name]] <- list(
        rstandard = as.numeric(rstandard(m)),
        rstudent  = as.numeric(rstudent(m)),
        pointLik  = as.numeric(pointLik(m)),
        ci_lower  = as.numeric(ci[, 1]),
        ci_upper  = as.numeric(ci[, 2]),
        acc_names = names(acc),
        acc_vals  = as.numeric(acc)
    )
}
jsonlite::write_json(out, "diag_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null")
cat("Wrote diag_reference.json with", length(out), "cases\n")
