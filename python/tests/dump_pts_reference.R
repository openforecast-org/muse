#!/usr/bin/env Rscript
# Dump full pts() method outputs for fixed-lambda specs so the Python PTS
# class can be checked end to end (estimation + ICs + fitted/residuals).
suppressMessages(devtools::load_all("../..", quiet = TRUE))

specs <- list(
    list(name = "1NN",    model = "1NN"),
    list(name = "1LN",    model = "1LN"),
    list(name = "1LT",    model = "1LT"),
    list(name = "1DT",    model = "1DT"),
    list(name = "1ND",    model = "1ND"),
    list(name = "0LT",    model = "0LT"),
    list(name = "0.5LT",  model = "0.5LT"),
    list(name = "1GT",    model = "1GT"),       # td drift -> nParam + 1
    list(name = "1ZZ",    model = "1ZZ"),       # engine structural selection
    list(name = "1ZN",    model = "1ZN"),
    list(name = "1LT_ar1", model = "1LT", ar = 1, ma = 0),
    list(name = "1LT_ma1", model = "1LT", ar = 0, ma = 1),
    # auto-lambda screen (power Z): decomposition + Guerrero
    list(name = "ZNN",  model = "ZNN"),
    list(name = "ZLT",  model = "ZLT"),
    list(name = "ZZZ",  model = "ZZZ")
)

y <- as.numeric(AirPassengers)
out <- list()
for (sp in specs) {
    ord <- list(ar = if (is.null(sp$ar)) 0 else sp$ar,
                ma = if (is.null(sp$ma)) 0 else sp$ma, select = FALSE)
    m <- pts(y, model = sp$model, lags = 12, h = 0, orders = ord)
    out[[sp$name]] <- list(
        model    = m$model,
        lambda   = m$lambda,
        coef     = as.numeric(coef(m)),
        coefNames = names(coef(m)),
        logLik   = as.numeric(logLik(m)),
        AIC      = as.numeric(AIC(m)),
        BIC      = as.numeric(BIC(m)),
        AICc     = as.numeric(greybox::AICc(m)),
        BICc     = as.numeric(greybox::BICc(m)),
        nobs     = nobs(m),
        nParam   = m$nParam,
        sigma    = as.numeric(sigma(m)),
        fitted   = as.numeric(fitted(m)),
        residuals = as.numeric(residuals(m))
    )
}
jsonlite::write_json(out, "pts_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null")
cat("Wrote pts_reference.json with", length(out), "cases\n")
