# S3 methods for class "pts". Convention: mirror smooth/adam's API so that
# users of those packages encounter no surprises.

#' @title S3 methods for objects of class \code{pts}
#' @name pts-methods
#' @description Standard accessors and printing for fitted \code{pts}
#' objects. \code{forecast(object, h)} generates forecasts from the fitted
#' parameters without re-running the optimiser; \code{predict(object)}
#' returns the in-sample fitted values.
#' @param x,object A fitted object of class \code{"pts"}.
#' @param h forecast horizon (steps ahead).
#' @param level confidence level for prediction intervals (default 0.95).
#' @param newdata unused (reserved for forecast paths with new regressors).
#' @param ... further arguments passed to underlying generics.
#' @template authors
NULL

#' @rdname pts-methods
#' @export
print.pts <- function(x, ...){
    cat("PTS state-space model\n")
    cat("  spec:   ", x$model, "   (UC: ", x$modelUC, ")\n", sep = "")
    cat("  lambda: ", format(x$lambda, digits = 5),
        "   lags: ", x$lags,
        "   nobs: ", nobs(x),
        "   nParam: ", x$nParam, "\n", sep = "")
    ll <- as.numeric(x$logLik)
    if (length(ll) == 1 && is.finite(ll)){
        cat("  ",
            "logLik=", format(ll,         digits = 5),
            "  AIC=",  format(AIC(x),     digits = 5),
            "  BIC=",  format(BIC(x),     digits = 5),
            "  AICc=", format(AICc.pts(x), digits = 5), "\n", sep = "")
    }
    if (!is.null(x$cppOutput) && nzchar(paste(x$cppOutput, collapse = ""))){
        cat("\n")
        cat(x$cppOutput)
    }
    invisible(x)
}

#' @rdname pts-methods
#' @export
summary.pts <- function(object, ...){
    print(object, ...)
    invisible(object)
}

# plot.pts and plot.pts.forecast intentionally NOT defined.  With the
# class chain c("pts", "smooth"), plot(m) dispatches to plot.smooth (a
# 4-panel diagnostic) and plot(forecast(m, h)) dispatches to
# plot.smooth.forecast.  See smooth/R/methods.R:1188 and :1880.

#' @rdname pts-methods
#' @export
fitted.pts <- function(object, ...) object$fitted

#' @rdname pts-methods
#' @export
residuals.pts <- function(object, ...) object$residuals

#' @rdname pts-methods
#' @export
coef.pts <- function(object, ...) object$B

#' @rdname pts-methods
#' @export
vcov.pts <- function(object, ...) object$vcov

#' @rdname pts-methods
#' @export
nobs.pts <- function(object, all = FALSE, ...){
    # Adam-style: nobs(object, all = FALSE) -> obsInSample (training only).
    #             nobs(object, all = TRUE)  -> obsAll      (incl. holdout).
    # Matches the dispatch at smooth/R/adam.R:7035.
    obsInSample <- sum(!is.na(object$data))
    if (isTRUE(all)) obsInSample + length(object$holdout) else obsInSample
}

#' @rdname pts-methods
#' @export
logLik.pts <- function(object, ...){
    out <- as.numeric(object$logLik)
    attr(out, "df")   <- object$nParam
    attr(out, "nobs") <- nobs(object)
    class(out) <- "logLik"
    out
}

#' @rdname pts-methods
#' @export
predict.pts <- function(object, newdata = NULL, ...){
    # In-sample fitted values (same shape as fitted(object)).  Out-of-sample
    # forecasts go through forecast().
    fitted.pts(object)
}

#' @rdname pts-methods
#' @description \code{forecast.pts} uses the C++ \code{forecastOnly} entry
#' point: it skips re-estimation and just propagates the Kalman filter
#' \code{h} steps forward from the fitted state, so changing \code{h} is
#' cheap.
#' @export
forecast.pts <- function(object, h = 10, level = 0.95, newdata = NULL, ...){
    if (!is.numeric(h) || length(h) != 1 || h < 1)
        stop("`h` must be a positive integer.", call. = FALSE)
    # Reconstruct the UCompC inputs from the fitted object's slots; no
    # need for a separate $forecast_args cache.  When the model was fit
    # with regressors, newdata supplies their future values for the
    # forecast horizon (see .pts_forecast_inputs).
    args <- .pts_forecast_inputs(object, h, newdata = newdata)
    out  <- .pts_call_uc("forecastOnly", args)

    # Engine returns yFor / yForV on the Box-Cox scale.  Build the
    # prediction interval by endpoint transformation:
    #   lower_orig = invBoxCox(yFor_bc - z*se),  upper_orig = invBoxCox(yFor_bc + z*se)
    # This preserves coverage and gives asymmetric intervals on the
    # original scale whenever lambda != 1.
    yFor_bc <- .pts_wrap_oos(as.numeric(out$yFor),  object$data)
    yForV   <- .pts_wrap_oos(as.numeric(out$yForV), object$data)
    z       <- stats::qnorm(1 - (1 - level) / 2)
    se      <- sqrt(yForV)
    lambda  <- args$lambda

    mean_out  <- .inv_box_cox(yFor_bc,          lambda)
    lower_out <- .inv_box_cox(yFor_bc - z * se, lambda)
    upper_out <- .inv_box_cox(yFor_bc + z * se, lambda)

    ret <- list(model    = object,
                mean     = mean_out,
                lower    = lower_out,
                upper    = upper_out,
                variance = yForV,           # documented: BC-scale variance
                level    = level,
                interval = "prediction",    # read by plot.smooth.forecast
                method   = object$model)
    class(ret) <- c("pts.forecast", "smooth.forecast")
    ret
}

#' @export
print.pts.forecast <- function(x, ...){
    cat(x$method, "forecast,", length(x$mean), "steps ahead\n")
    cat("  Confidence level:", format(100 * x$level, digits = 4), "%\n", sep = "")
    df <- data.frame(`Point Forecast` = as.numeric(x$mean),
                     Lo                = as.numeric(x$lower),
                     Hi                = as.numeric(x$upper),
                     check.names = FALSE)
    print(round(df, 4))
    invisible(x)
}

# plot.pts.forecast intentionally not defined; class chain
# c("pts.forecast", "smooth.forecast") dispatches to plot.smooth.forecast.
