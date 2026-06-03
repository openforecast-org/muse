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
#' @description \code{print.pts} mirrors \code{smooth::print.adam}
#' (\code{smooth/R/adam.R:5862}) section by section, substituting the
#' PTS-specific MSOE concepts where ETS-specific ones do not apply:
#' the "initialisation" line becomes the Box-Cox \eqn{\lambda} block, and
#' the "Persistence vector g" block becomes the MSOE innovation variances
#' (the same variances are otherwise the structural pieces of the B
#' parameter vector).  The C++ validation diagnostics live on
#' \code{$cppOutput} should the user want them.
#' @export
print.pts <- function(x, digits = 4, ...){
    # --- elapsed time + model spec line (adam.R:5872-5875) ---
    elapsed <- if (inherits(x$timeElapsed, "proc_time"))
                   as.numeric(x$timeElapsed["elapsed"])
               else if (inherits(x$timeElapsed, "difftime"))
                   as.numeric(x$timeElapsed, units = "secs")
               else suppressWarnings(as.numeric(x$timeElapsed))
    if (!is.null(elapsed) && is.finite(elapsed))
        cat("Time elapsed:", round(elapsed, 2), "seconds")
    fnName <- tryCatch(utils::tail(all.vars(x$call[[1]]), 1),
                       error = function(e) "pts")
    if (!nzchar(fnName)) fnName <- "pts"
    cat("\nModel estimated using ", fnName, "() function: ", x$model, sep = "")

    # --- Box-Cox transform line (replaces adam's `With ... initialisation`) ---
    lam <- x$lambda
    if (is.finite(lam) && abs(lam - 1) < 1e-8){
        cat("\nWith no Box-Cox transform (lambda = 1)")
    } else if (is.finite(lam) && abs(lam) < 1e-8){
        cat("\nWith log transform (Box-Cox lambda = 0)")
    } else {
        cat("\nWith Box-Cox lambda = ", round(lam, digits), sep = "")
    }

    # --- distribution line (adam.R:5898-5917) ---
    distrib <- switch(as.character(x$distribution),
                      "dnorm"     = "Normal",
                      "dlaplace"  = "Laplace",
                      "ds"        = "S",
                      "dlogis"    = "Logistic",
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma"    = "Gamma",
                      as.character(x$distribution))
    cat("\nDistribution assumed in the model:", distrib)

    # --- loss line (adam.R:5919-5925) ---
    cat("\nLoss function type:", x$loss)
    if (!is.null(x$lossValue) && is.finite(x$lossValue))
        cat("; Loss function value:", round(x$lossValue, digits))

    # --- intercept / drift (adam.R:5928-5930); skip when NA for pts ---
    if (!is.null(x$constant) && !is.na(x$constant))
        cat("\nIntercept/Drift value:", round(x$constant, digits))

    # --- MSOE innovation variances (pts analog of adam's persistence
    #     vector g at adam.R:5947-5966).  Filter B to drop the ARMA / Beta
    #     rows; what remains are the variances of state innovations. ---
    B  <- coef(x)
    nm <- names(B)
    isArma <- grepl("^(AR|MA)\\(", nm)
    isXreg <- grepl("^Beta",       nm)
    vars   <- B[!(isArma | isXreg)]
    if (length(vars) > 0){
        cat("\nInnovation variances:\n")
        # Variances live on whatever scale the BC residuals do, often very
        # small (1e-5 .. 1e-3), so signif() preserves magnitude where
        # round(., digits) would zero them out.
        print(signif(vars, digits))
    }

    # --- ARMA parameters of the irregular component (adam.R:5976-6013) ---
    if (!is.null(x$orders) && (x$orders$ar > 0 || x$orders$ma > 0)){
        cat("\nARMA parameters of the irregular component:\n")
        if (x$orders$ar > 0){
            arVals <- B[grepl("^AR\\(", nm)]
            if (length(arVals)) print(round(arVals, digits))
        }
        if (x$orders$ma > 0){
            maVals <- B[grepl("^MA\\(", nm)]
            if (length(maVals)) print(round(maVals, digits))
        }
    }

    # --- Beta block when the model carries xregs (PTS-specific addition,
    #     since adam folds xreg info into the persistence vector header) ---
    if (any(isXreg)){
        cat("\nRegressor coefficients:\n")
        print(signif(B[isXreg], digits))
    }

    # --- sample size / nparam / df (adam.R:6024-6026) ---
    cat("\nSample size:", nobs(x))
    cat("\nNumber of estimated parameters:", nparam(x))
    cat("\nNumber of degrees of freedom:", nobs(x) - nparam(x))

    # --- information criteria as a named vector (adam.R:6036-6039) ---
    ICs <- c(AIC  = AIC(x),
             AICc = AICc.pts(x),
             BIC  = BIC(x),
             BICc = BICc.pts(x))
    cat("\nInformation criteria:\n")
    print(round(ICs, digits))

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
