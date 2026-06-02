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
    if (length(x$IC) == 3 && all(!is.na(x$IC))){
        cat("  ",
            "logLik=", format(as.numeric(x$logLik), digits = 5),
            "  AIC=",  format(x$IC["AIC"],  digits = 5),
            "  BIC=",  format(x$IC["BIC"],  digits = 5),
            "  AICc=", format(x$IC["AICc"], digits = 5), "\n", sep = "")
    }
    if (!is.null(x$table) && nzchar(paste(x$table, collapse = ""))){
        cat("\n")
        cat(x$table)
    }
    invisible(x)
}

#' @rdname pts-methods
#' @export
summary.pts <- function(object, ...){
    print(object, ...)
    invisible(object)
}

#' @rdname pts-methods
#' @export
plot.pts <- function(x, ...){
    if (is.null(x$comp) || length(x$comp) < 2)
        stop("pts object has no components; refit with pts().", call. = FALSE)
    if (is.ts(x$comp))
        plot(x$comp, main = "PTS decomposition", ...)
    else
        plot(stats::ts(x$comp, frequency = x$lags), main = "PTS decomposition", ...)
}

#' @rdname pts-methods
#' @export
fitted.pts <- function(object, ...) object$fitted

#' @rdname pts-methods
#' @export
residuals.pts <- function(object, ...) object$residuals

#' @rdname pts-methods
#' @export
coef.pts <- function(object, ...) object$p

#' @rdname pts-methods
#' @export
vcov.pts <- function(object, ...) object$covp

#' @rdname pts-methods
#' @export
nobs.pts <- function(object, ...) sum(!is.na(object$y))

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
    # In-sample fitted values. Out-of-sample forecasts go through forecast().
    object$fitted
}

#' @rdname pts-methods
#' @description \code{forecast.pts} uses the C++ \code{forecastOnly} entry
#' point: it skips re-estimation and just propagates the Kalman filter
#' \code{h} steps forward from the fitted state, so changing \code{h} is
#' cheap.
#' @export
forecast.pts <- function(object, h = 10, level = 0.95, ...){
    if (!is.numeric(h) || length(h) != 1 || h < 1)
        stop("`h` must be a positive integer.", call. = FALSE)
    if (is.null(object$forecast_args))
        stop("pts object has no cached forecast inputs; cannot forecast.",
             call. = FALSE)
    args   <- object$forecast_args
    args$h <- as.integer(h)
    out    <- .pts_call_uc("forecastOnly", args)

    yFor  <- .pts_ts_forecast(as.numeric(out$yFor),  object$y)
    yForV <- .pts_ts_forecast(as.numeric(out$yForV), object$y)

    z   <- stats::qnorm(1 - (1 - level) / 2)
    se  <- sqrt(yForV)
    lo  <- yFor - z * se
    hi  <- yFor + z * se

    ret <- list(model    = object,
                mean     = yFor,
                lower    = lo,
                upper    = hi,
                variance = yForV,
                level    = level,
                method   = paste0("PTS(", object$model, ")"))
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

#' @export
plot.pts.forecast <- function(x, ...){
    y <- x$model$y
    fmean <- x$mean
    ymin <- min(c(y, x$lower), na.rm = TRUE)
    ymax <- max(c(y, x$upper), na.rm = TRUE)
    if (is.ts(y) && is.ts(fmean)){
        xlim <- c(stats::time(y)[1], stats::time(fmean)[length(fmean)])
        plot(y, xlim = xlim, ylim = c(ymin, ymax),
             ylab = "y", main = x$method, ...)
    } else {
        plot(c(y, fmean), type = "n", ylim = c(ymin, ymax),
             ylab = "y", main = x$method, ...)
        lines(seq_along(y), as.numeric(y))
    }
    lines(fmean, col = "blue", lwd = 2)
    lines(x$lower, col = "blue", lty = 2)
    lines(x$upper, col = "blue", lty = 2)
}
