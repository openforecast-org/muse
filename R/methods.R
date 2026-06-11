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

    # --- Harmonic periods (only when seasonal) ---
    if (!is.null(x$lagsAll) && length(x$lagsAll) > 1){
        perStr <- paste(formatC(x$lagsAll, format = "f", digits = 1), collapse = " / ")
        cat("\nPeriods:", perStr)
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

    # --- Parameters block: variance params shown as proportions (Tasks 1-3).
    #     Structural variances (Level, Slope, Seas*) are divided by their sum
    #     so they show the relative contribution of each component.
    #     Irregular is dropped.  Damping is shown with its raw value.
    #     ARMA and Beta (xreg) params get their own sections below. ---
    B  <- coef(x)
    nm <- names(B)
    isArma  <- grepl("^(AR|MA)\\(", nm)
    isXreg  <- grepl("^Beta",       nm)
    isDamp  <- nm == "Damping"
    isIrr   <- nm == "Irregular"
    isVar   <- !(isArma | isXreg | isDamp | isIrr)

    varVals <- B[isVar]
    dampVal <- B[isDamp]

    if (length(varVals) > 0){
        S <- sum(varVals)
        props <- if (S > 0) varVals / S else varVals
        cat("\nParameters:\n")
        print(signif(props, digits))
    }
    if (length(dampVal) > 0){
        cat("\nDamping:\n")
        print(round(dampVal, digits))
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
    # AIC / BIC come from stats; AICc / BICc come from greybox via the
    # c("pts", "smooth") dispatch chain.  No pts-local formulas.
    ICs <- c(AIC  = AIC(x),
             AICc = AICc(x),
             BIC  = BIC(x),
             BICc = BICc(x))
    cat("\nInformation criteria:\n")
    print(round(ICs, digits))

    invisible(x)
}

# plot.pts.forecast intentionally NOT defined: c("pts.forecast",
# "smooth.forecast") dispatches to plot.smooth.forecast.
#
# plot.pts is a thin pre-processor for panel 12 (state decomposition).
# plot.smooth's non-ETS branch (smooth/R/methods.R:1680-1707) does
# cbind(x$states, residuals(x)) without prepending actuals, so on the
# raw pts $states the user sees [Level, Slope, Seasonal, residuals] but
# no actuals row.  We prepend an "actuals" column (anchored with NA at
# row 1 to align with the t = 0 row of $states) and delegate the rest
# to plot.smooth via NextMethod.
#' @rdname pts-methods
#' @export
plot.pts <- function(x, which = c(1, 2, 4, 6), ...){
    if (12 %in% which){
        states <- x$states
        if (!is.null(states) && !is.null(colnames(states))){
            mat    <- unclass(states)
            actVec <- c(NA_real_, as.numeric(x$data))
            n      <- nrow(mat)
            if (length(actVec) > n) actVec <- actVec[seq_len(n)]
            else if (length(actVec) < n)
                actVec <- c(actVec, rep(NA_real_, n - length(actVec)))
            newMat <- cbind(actuals = actVec, mat)
            x$states <- if (is.ts(states))
                            stats::ts(newMat,
                                      start     = stats::start(states),
                                      frequency = stats::frequency(states))
                        else if (inherits(states, "zoo"))
                            zoo::zoo(newMat, order.by = stats::time(states))
                        else newMat
        }
    }
    NextMethod()
}

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
#' cheap.  \code{interval} selects the variance source:
#' \describe{
#'   \item{"prediction" (default)}{state propagation + future shocks
#'     (the engine's \code{yForV}).  Bands the next observation.}
#'   \item{"confidence"}{conditional-mean variance.  In a state-space
#'     model \eqn{var(E[y_{t+h}\,|\,obs]) = var(y_{t+h}\,|\,obs) - \sigma^2},
#'     where \eqn{\sigma^2} is the BC-scale residual variance: we read it
#'     straight off the fitted scale, no reforecast needed.}
#'   \item{"simulated"}{empirical quantiles of \code{nsim} forward
#'     paths from \code{simulate.pts()}.  Returns the path matrix in
#'     \code{$scenarios} when \code{scenarios = TRUE}.}
#'   \item{"none"}{no bands; \code{lower = upper = mean}.}
#' }
#' \code{level} accepts a vector; \code{lower} / \code{upper} are then
#' \eqn{h \times nLevels} matrices (or \eqn{1 \times nLevels} when
#' \code{cumulative = TRUE}).  \code{side} produces two-sided
#' / upper-only / lower-only intervals -- the absent side is set to
#' \eqn{\mp\infty} on the BC scale and back-transformed to the
#' original-scale support boundary (0 for \eqn{\lambda > 0}, \eqn{-\infty}
#' for the identity transform).  \code{cumulative = TRUE} collapses the
#' horizon into one total -- exact for \code{interval = "simulated"}
#' (sum within each path); the engine does not expose cross-step state
#' covariance, so the other intervals fall back to simulation totals.
#' @param interval interval type; one of \code{"prediction"},
#'   \code{"confidence"}, \code{"simulated"}, or \code{"none"}.
#' @param side one of \code{"both"}, \code{"upper"}, \code{"lower"} --
#'   selects two-sided vs one-sided intervals.
#' @param cumulative if \code{TRUE}, return the cumulative h-step total.
#' @param nsim number of simulated paths when
#'   \code{interval = "simulated"} or under the cumulative fallback
#'   (default \code{10000}).
#' @param scenarios if \code{TRUE} and \code{interval = "simulated"},
#'   return the simulated path matrix as \code{$scenarios}.
#' @export
forecast.pts <- function(object, h = 10, newdata = NULL,
                         interval = c("prediction", "confidence",
                                      "simulated", "none"),
                         level = 0.95,
                         side = c("both", "upper", "lower"),
                         cumulative = FALSE,
                         nsim = NULL,
                         scenarios = FALSE,
                         ...){
    if (!is.numeric(h) || length(h) != 1 || h < 1)
        stop("`h` must be a positive integer.", call. = FALSE)
    interval <- match.arg(interval)
    side     <- match.arg(side)
    if (!is.numeric(level) || any(level <= 0) || any(level >= 1))
        stop("`level` must be in (0, 1) (a scalar or numeric vector).", call. = FALSE)
    if (is.null(nsim)) nsim <- 10000L
    if (!is.numeric(nsim) || length(nsim) != 1 || nsim < 1)
        stop("`nsim` must be a positive integer.", call. = FALSE)
    nsim <- as.integer(nsim)
    nLevels <- length(level)

    # Lower / upper tail probabilities -- vectors of length nLevels.
    # Mirror smooth/R/adam.R:7333-7344.
    qLow <- switch(side, both = (1 - level) / 2, upper = rep(0, nLevels),
                   lower = 1 - level)
    qUp  <- switch(side, both = (1 + level) / 2, upper = level,
                   lower = rep(1, nLevels))

    # Engine call: yFor (BC point forecast) and yForV (prediction var).
    args      <- .pts_forecast_inputs(object, h, newdata = newdata)
    out       <- .pts_call_uc("forecastOnly", args)
    yFor_bc   <- .pts_wrap_oos(as.numeric(out$yFor),  object$data)
    yForVpred <- .pts_wrap_oos(as.numeric(out$yForV), object$data)
    lambda    <- args$lambda
    mean_out  <- .inv_box_cox(yFor_bc, lambda)
    # Confidence variance = prediction variance minus the obs-noise
    # contribution.  For a SSM y_t = Z a_t + eps with var(eps) = sigma^2,
    # var(E[y_{t+h}|obs]) = var(y_{t+h}|obs) - sigma^2.  In PTS the
    # irregular noise lives in Q, but the MLE residual variance
    # `object$scale^2` (BC scale) is precisely the per-step obs-noise
    # contribution to var(y).  Clamp at 0 to keep things sensible.
    sigma2BC  <- as.numeric(object$scale) ^ 2
    yForVconf <- pmax(0, .pts_wrap_oos(as.numeric(out$yForV) - sigma2BC,
                                       object$data))

    # Simulated paths -- cached for cumulative / scenarios reuse.
    pathsCache <- NULL
    drawPaths  <- function(){
        if (is.null(pathsCache))
            pathsCache <<- as.matrix(simulate(object, nsim = nsim, h = h, ...)$data)
        pathsCache
    }

    # Build a (h x nLevels) matrix on the original scale by evaluating
    # `f(p)` per level p; preserves the time index of `template` (a
    # length-h ts/zoo vector).
    .mkBands <- function(f, probs, template){
        mat <- matrix(NA_real_, nrow = length(template), ncol = nLevels)
        for (j in seq_along(probs)) mat[, j] <- as.numeric(f(probs[j]))
        if (is.ts(template))
            stats::ts(mat, start = stats::start(template),
                      frequency = stats::frequency(template))
        else if (inherits(template, "zoo"))
            zoo::zoo(mat, order.by = stats::time(template))
        else mat
    }

    # Per-step intervals.  Cumulative collapse, if any, overwrites
    # these below.
    if (interval == "none"){
        lower_out   <- .mkBands(function(p) mean_out, qLow, mean_out)
        upper_out   <- .mkBands(function(p) mean_out, qUp,  mean_out)
        varianceOut <- yForVpred
    } else if (interval %in% c("prediction", "confidence")){
        varianceOut <- if (interval == "prediction") yForVpred else yForVconf
        se          <- sqrt(as.numeric(varianceOut))
        # qnorm(0) = -Inf, qnorm(1) = +Inf -- pass through.  Where
        # the BC-scale bound is +/- Inf, .inv_box_cox returns the
        # support boundary (0 for lambda > 0, +/- Inf for identity).
        bcQuant <- function(p) {
            z <- stats::qnorm(p)
            if (is.finite(z))
                .inv_box_cox(yFor_bc + z * se, lambda)
            else if (z == -Inf)
                .inv_box_cox(stats::ts(rep(-Inf, h),
                                       start = stats::start(yFor_bc),
                                       frequency = stats::frequency(yFor_bc)),
                             lambda)
            else
                stats::ts(rep(Inf, h), start = stats::start(yFor_bc),
                          frequency = stats::frequency(yFor_bc))
        }
        lower_out <- .mkBands(bcQuant, qLow, mean_out)
        upper_out <- .mkBands(bcQuant, qUp,  mean_out)
    } else if (interval == "simulated"){
        paths <- drawPaths()
        qPaths <- function(p)
            if (p <= 0) rep(-Inf, h)
            else if (p >= 1) rep(Inf, h)
            else apply(paths, 1, stats::quantile, probs = p, na.rm = TRUE)
        lower_out   <- .mkBands(qPaths, qLow, mean_out)
        upper_out   <- .mkBands(qPaths, qUp,  mean_out)
        varianceOut <- .pts_wrap_oos(apply(paths, 1, stats::var, na.rm = TRUE),
                                     object$data)
    }

    if (isTRUE(cumulative)){
        # Cumulative collapse: 1 x nLevels matrices.  Point forecast is
        # the sum of per-step point forecasts; the interval uses
        # simulation totals (exact for "simulated", approximate
        # otherwise since the engine does not expose cross-step state
        # covariance).
        meanScalar <- sum(as.numeric(mean_out), na.rm = TRUE)
        if (interval == "none"){
            mean_out    <- meanScalar
            lower_out   <- matrix(meanScalar, nrow = 1, ncol = nLevels)
            upper_out   <- matrix(meanScalar, nrow = 1, ncol = nLevels)
            varianceOut <- 0
        } else {
            totals      <- colSums(drawPaths())
            qTot <- function(p)
                if (p <= 0) -Inf
                else if (p >= 1) Inf
                else unname(stats::quantile(totals, probs = p, na.rm = TRUE))
            mean_out    <- meanScalar
            lower_out   <- matrix(vapply(qLow, qTot, numeric(1)), nrow = 1)
            upper_out   <- matrix(vapply(qUp,  qTot, numeric(1)), nrow = 1)
            varianceOut <- stats::var(totals, na.rm = TRUE)
        }
    }

    scenariosMat <- if (interval == "simulated" && isTRUE(scenarios)) drawPaths() else NULL

    ret <- list(model      = object,
                mean       = mean_out,
                lower      = lower_out,
                upper      = upper_out,
                variance   = varianceOut,        # BC-scale variance source
                level      = level,
                interval   = interval,           # read by plot.smooth.forecast
                side       = side,
                cumulative = isTRUE(cumulative),
                method     = object$model)
    if (!is.null(scenariosMat)) ret$scenarios <- scenariosMat
    class(ret) <- c("pts.forecast", "smooth.forecast")
    ret
}

#' @export
print.pts.forecast <- function(x, ...){
    cat(x$method, "forecast,", length(as.numeric(x$mean)), "steps ahead\n")
    cat("  Confidence level: ",
        paste0(format(100 * x$level, digits = 4), "%", collapse = ", "), "\n", sep = "")
    lower <- as.matrix(x$lower); upper <- as.matrix(x$upper)
    nLevels <- ncol(lower)
    df <- data.frame(`Point Forecast` = as.numeric(x$mean), check.names = FALSE)
    for (j in seq_len(nLevels)){
        tag <- if (nLevels == 1) ""
               else paste0(" ", format(100 * x$level[j], digits = 4), "%")
        df[[paste0("Lo", tag)]] <- lower[, j]
        df[[paste0("Hi", tag)]] <- upper[, j]
    }
    print(round(df, 4))
    invisible(x)
}

# plot.pts.forecast intentionally not defined; class chain
# c("pts.forecast", "smooth.forecast") dispatches to plot.smooth.forecast.
