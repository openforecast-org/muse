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
    list(name = "ZZZ",  model = "ZZZ"),
    list(name = "1LT_sel", model = "1LT", ar = 2, ma = 2, select = TRUE),
    list(name = "ZZZ_sel", model = "ZZZ", ar = 2, ma = 2, select = TRUE),
    # SARMA: ar/ma vectors, seasonal lag from the top-level lags (=12)
    list(name = "1LT_sar10", model = "1LT", ar = c(1, 1), ma = c(0, 0)),
    list(name = "1LT_sma01", model = "1LT", ar = c(0, 0), ma = c(1, 1)),
    list(name = "1LT_smix",  model = "1LT", ar = c(1, 0), ma = c(0, 1)),
    # missing values: the screen + engine must handle gaps; positions
    # (1-based) match NA_POS in test_pts_parity.py (0-based 40,41,42).
    list(name = "ZZZ_na", model = "ZZZ", na = c(41, 42, 43))
)

y <- as.numeric(AirPassengers)
out <- list()
for (sp in specs) {
    ord <- list(ar = if (is.null(sp$ar)) 0 else sp$ar,
                ma = if (is.null(sp$ma)) 0 else sp$ma,
                select = isTRUE(sp$select))
    yi <- y
    # NB: use [["na"]] (exact), not $na -- $ partial-matches "name" for specs
    # with no na field, which would index yi by the model string and append.
    if (!is.null(sp[["na"]])) yi[sp[["na"]]] <- NA
    m <- pts(yi, model = sp$model, lags = 12, h = 0, orders = ord)
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
        residuals = as.numeric(residuals(m)),
        orders = list(ar = orders(m)$ar, ma = orders(m)$ma)
    )
}
jsonlite::write_json(out, "pts_reference.json", auto_unbox = TRUE,
                     digits = 16, null = "null", na = "null")
cat("Wrote pts_reference.json with", length(out), "cases\n")
