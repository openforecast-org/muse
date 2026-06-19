#!/usr/bin/env Rscript
# Dump exact engine inputs + R outputs for a battery of fixed-spec fits, so
# the Python binding can be fed byte-identical inputs and checked for parity.
suppressMessages(devtools::load_all("../..", quiet = TRUE))

# Fixed-lambda specs only (no auto-lambda screen / no Z selection) so the
# comparison is a pure engine round-trip with identical inputs on both sides.
cases <- list(
    list(name = "air_1NN",   y = as.numeric(AirPassengers), model = "1NN",   lags = 12),
    list(name = "air_1LT",   y = as.numeric(AirPassengers), model = "1LT",   lags = 12),
    list(name = "air_0LT",   y = as.numeric(AirPassengers), model = "0LT",   lags = 12),
    list(name = "air_0.5LT", y = as.numeric(AirPassengers), model = "0.5LT", lags = 12),
    list(name = "air_1DT",   y = as.numeric(AirPassengers), model = "1DT",   lags = 12),
    list(name = "air_1LT_ar1", y = as.numeric(AirPassengers), model = "1LT", lags = 12,
         ar = 1, ma = 0)
)

out <- list()
for (cs in cases) {
    uc <- pts_to_uc(cs$model,
                    armaOrders = c(if (is.null(cs$ar)) 0 else cs$ar,
                                   if (is.null(cs$ma)) 0 else cs$ma))
    args <- .pts_uc_inputs(y = cs$y, u = NULL, modelUC = uc$modelU, h = 0L,
                           lambda = uc$lambda, criterion = "bicc",
                           lags = cs$lags, verbose = FALSE, armaIdent = FALSE,
                           irregularOptions = "arma(0,0)", outlier = 0,
                           lambdaLower = -Inf)
    res <- .pts_call_uc("all", args)

    out[[cs$name]] <- list(
        # inputs (exactly what the engine received)
        inputs = list(
            command = "all", y = args$y,
            u = as.numeric(args$u), u_nrow = nrow(args$u), u_ncol = ncol(args$u),
            model = args$model, h = args$h, lambda = args$lambda,
            outlier = args$outlier, tTest = args$tTest, criterion = args$criterion,
            periods = args$periods, rhos = args$rhos, verbose = args$verbose,
            stepwise = args$stepwise, p0 = args$p, armaFlag = args$arma,
            TVP = args$TVP, seas = args$seas, trendOptions = args$trendOptions,
            seasonalOptions = args$seasonalOptions,
            irregularOptions = args$irregularOptions,
            nsim = 1L, seed = 0L, lambdaLower = args$lambdaLower
        ),
        # R engine outputs to match
        r = list(
            model = res$model, lambda = res$lambda,
            p = as.numeric(res$p), parNames = names(res$p),
            criteria = as.numeric(res$criteria),
            objFunValue = res$objFunValue,
            yForLen = length(res$yFor),
            comp_dim = dim(res$comp)
        )
    )
}

jsonlite::write_json(out, "reference.json", auto_unbox = TRUE, digits = 16,
                     null = "null")
cat("Wrote reference.json with", length(out), "cases\n")
