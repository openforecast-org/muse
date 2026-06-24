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
#' @param digits number of significant digits used in printed output.
#' @param which integer vector of plot panels to draw; passed through to
#'   \code{plot.smooth} (see \code{?smooth::plot.adam}).  Defaults to
#'   \code{c(1, 2, 4, 6)}.
#' @param all logical; if \code{TRUE} the holdout sample is included in
#'   the observation count.  Default \code{FALSE}.
#' @param ... further arguments passed to underlying generics.
#' @examples
#' model <- pts(AirPassengers, model = "1LT", h = 12, holdout = TRUE)
#' print(model)
#' fitted(model)
#' residuals(model)
#' # forecast 12 steps ahead with 95% prediction intervals
#' forecast(model, h = 12, interval = "prediction", level = 0.95)
#' @references \itemize{
#'   \item Granger, C. W. J., & Newbold, P. (1976).  Forecasting transformed
#'     series.  \emph{Journal of the Royal Statistical Society: Series B
#'     (Methodological)}, 38(2), 189-203.
#'     \doi{10.1111/j.2517-6161.1976.tb01585.x}
#'   \item Pankratz, A., & Dudley, U. (1987).  Forecasts of power-transformed
#'     series.  \emph{Journal of Forecasting}, 6(4), 239-248.
#'     \doi{10.1002/for.3980060403}
#'   \item Guerrero, V. M. (1993).  Time-series analysis supported by power
#'     transformations.  \emph{Journal of Forecasting}, 12(1), 37-48.
#'     \doi{10.1002/for.3980120104}
#' }
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

    # --- Harmonic periods (only meaningful for the trigonometric "T"
    #     seasonal -- the engine's `equal` shape with a harmonic
    #     expansion).  Discrete "D" and none "N" don't have a
    #     multi-period decomposition; skip the line there. ---
    seasLetter <- sub(".*,([A-Z])\\)$", "\\1", as.character(x$model))
    if (identical(seasLetter, "T") &&
        !is.null(x$lagsAll) && length(x$lagsAll) > 1){
        perStr <- paste(formatC(x$lagsAll, format = "f", digits = 1),
                        collapse = " / ")
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

    # --- Variance parameters: one row per noise source, showing both the
    #     absolute variance and the proportion of total noise it carries.
    #     A single row per parameter (no separate footer for the
    #     concentrated value) keeps the layout unambiguous.  Damping is
    #     a [0, 1] coefficient, not a variance, so it stays on its own
    #     line.  ARMA and Beta blocks come further below. ---
    B  <- coef(x)
    nm <- names(B)
    isArma   <- grepl("^S?(AR|MA)\\(", nm)   # matches AR/MA + SAR/SMA
    isXreg   <- grepl("^Beta",       nm)
    isOutlier<- grepl("^(AO|LS|SC)[0-9]+$", nm)   # engine outlier dummies
    isDamp   <- nm == "Damping"
    isVar    <- !(isArma | isXreg | isDamp | isOutlier)   # includes Irregular

    varVals <- B[isVar]
    dampVal <- B[isDamp]

    # Concentrated parameter(s): NaN on vcov diagonal.
    cv      <- x$vcov
    concNms <- if (is.null(dim(cv))) character(0)
               else nm[which(is.nan(diag(cv)))]

    if (length(varVals) > 0){
        S     <- sum(varVals)
        props <- if (S > 0) varVals / S else varVals
        rn    <- names(varVals)
        if (length(concNms) > 0){
            mark <- rn %in% concNms
            rn[mark] <- paste0(rn[mark], " (*)")
        }
        tbl <- cbind(Variance   = signif(varVals, digits),
                     Proportion = signif(props,   digits))
        rownames(tbl) <- rn

        cat("\nVariance parameters:\n")
        print(tbl, na.print = "-")
        if (length(concNms) > 0)
            cat("(*) concentrated out\n")

        # Deterministic slope under the global ("G") trend is a constant
        # drift, not a variance -- keep it out of the table above and
        # report it on its own line.
        if (!is.null(x$modelUC) && startsWith(x$modelUC, "td/") &&
            is.matrix(x$comp) && "Slope" %in% colnames(x$comp)){
            cat("\nDeterministic slope (drift per period): ",
                signif(as.numeric(x$comp[1, "Slope"]), digits), "\n", sep = "")
        }
    }
    if (length(dampVal) > 0)
        cat("\nDamping: ", round(unname(dampVal), digits), "\n", sep = "")

    # --- ARMA parameters of the irregular component (adam.R:5976-6013) ---
    # `orders$ar` and `$ma` may be length-1 (non-seasonal) or length-2
    # (seasonal); `any()` makes the scalar / vector cases uniform.  Names
    # are AR(i) / SAR(i) / MA(i) / SMA(i), so the patterns match both
    # non-seasonal and seasonal blocks in one go.
    if (!is.null(x$orders) && (any(x$orders$ar > 0) || any(x$orders$ma > 0))){
        cat("\nARMA parameters of the irregular component:\n")
        if (any(x$orders$ar > 0)){
            arVals <- B[grepl("^S?AR\\(", nm)]
            if (length(arVals)) print(round(arVals, digits))
        }
        if (any(x$orders$ma > 0)){
            maVals <- B[grepl("^S?MA\\(", nm)]
            if (length(maVals)) print(round(maVals, digits))
        }
    }

    # --- Beta block when the model carries xregs (PTS-specific addition,
    #     since adam folds xreg info into the persistence vector header) ---
    if (any(isXreg)){
        cat("\nRegressor coefficients:\n")
        print(signif(B[isXreg], digits))
    }

    # --- Outlier-dummy coefficients (only when outliers were used).
    #     One-liner above already says how many of each type; this is the
    #     per-event breakdown so the user can see magnitudes.
    if (any(isOutlier)){
        cat("\nOutlier coefficients:\n")
        print(signif(B[isOutlier], digits))
    }

    # --- sample size / nparam / df (adam.R:6024-6026) ---
    # --- missing-values + outlier-detection one-liners (skip when none) ---
    nMissing <- if (!is.null(x$data)) sum(!is.finite(as.numeric(x$data))) else 0L
    if (nMissing > 0L)
        cat("\n", nMissing,
            if (nMissing == 1L) " missing value in the data"
            else                " missing values in the data",
            sep = "")
    od <- x$outliersDetected
    if (!is.null(od) && nrow(od) > 0L &&
        !identical(x$outliers, "ignore")){
        tally <- table(factor(od$type, levels = c("AO", "LS", "SC")))
        bits  <- character(0)
        if (tally[["AO"]] > 0L)
            bits <- c(bits, sprintf("%d AO", tally[["AO"]]))
        if (tally[["LS"]] > 0L)
            bits <- c(bits, sprintf("%d LS", tally[["LS"]]))
        if (tally[["SC"]] > 0L)
            bits <- c(bits, sprintf("%d SC", tally[["SC"]]))
        cat("\nOutliers detected: ", paste(bits, collapse = ", "), sep = "")
        cat("\n")
    }

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
    attr(out, "df")   <- nparam(object)   # [Estimated, nParamAll] of $nParam
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
#' @param biasadj point-forecast back-transform: \code{FALSE} (median,
#'   the default) or \code{TRUE} (bias-corrected mean).  Defaults to the
#'   value the model was fitted with (\code{object$biasadj}).
#' @export
forecast.pts <- function(object, h = 10, newdata = NULL,
                         interval = c("prediction", "confidence",
                                      "simulated", "none"),
                         level = 0.95,
                         side = c("both", "upper", "lower"),
                         cumulative = FALSE,
                         nsim = NULL,
                         scenarios = FALSE,
                         biasadj = NULL,
                         ...){
    if (!is.numeric(h) || length(h) != 1 || h < 1)
        stop("`h` must be a positive integer.", call. = FALSE)
    # Point forecast: bias-corrected MEAN (TRUE) vs MEDIAN (FALSE).  Defaults to
    # the value the model was fitted with (object$biasadj), overridable here.
    if (is.null(biasadj)) biasadj <- isTRUE(object$biasadj)
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
    # Point forecast.  biasadj = FALSE (default): conditional MEDIAN g^{-1}(mu);
    # biasadj = TRUE: bias-corrected conditional MEAN.  Either way the interval
    # quantiles below stay median-style (exact quantiles of the back-transformed
    # distribution -- they must NOT be bias-corrected).
    mean_out  <- if (isTRUE(biasadj)) .inv_box_cox_mean(yFor_bc, yForVpred, lambda)
                 else                 .inv_box_cox(yFor_bc, lambda)
    # Confidence variance = prediction variance minus the obs-noise
    # contribution.  For a SSM y_t = Z a_t + eps with var(eps) = sigma^2,
    # var(E[y_{t+h}|obs]) = var(y_{t+h}|obs) - sigma^2.  In PTS the
    # irregular noise lives in Q, but the MLE residual variance
    # `object$scale^2` (BC scale) is precisely the per-step obs-noise
    # contribution to var(y).  Clamp at 0 to keep things sensible.
    sigma2BC  <- as.numeric(object$scale) ^ 2
    yForVconf <- pmax(0, .pts_wrap_oos(as.numeric(out$yForV) - sigma2BC,
                                       object$data))

    # Simulated paths from the *terminal* state -- these are forward
    # forecasts, not in-sample replays.  Bypasses simulate.pts (which
    # starts from the initial state) by calling .pts_forecast_paths
    # directly.  Cached for cumulative / scenarios reuse.
    pathsCache <- NULL
    drawPaths  <- function(){
        if (is.null(pathsCache))
            pathsCache <<- as.matrix(.pts_forecast_paths(object,
                                                        nsim = nsim,
                                                        h    = h))
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
        # Quantiles of the back-transformed forecast distribution.
        #
        # For lambda >= 0 the BC inverse covers all of (0, infinity), so
        # the standard formula y_q = invBoxCox(mu_BC + z*se, lambda)
        # gives a proper quantile of a proper CDF.
        #
        # For lambda < 0 the BC inverse only covers x < -1/lambda; the
        # underlying Normal puts non-zero mass above that boundary,
        # which corresponds to a point mass at +infinity in y.  The
        # unrenormalized CDF then saturates at
        #     P_valid = pnorm(-1/lambda, mu_BC, se)
        # and the naive quantile diverges to +infinity for any
        # p > P_valid.  Renormalize to the truncated distribution on
        # (0, infinity) -- i.e. condition on the underlying Normal
        # falling within the valid BC inverse support -- before
        # inverting.  This gives a proper CDF and a finite quantile
        # for every p in (0, 1).  See Granger & Newbold (1976),
        # Pankratz & Dudley (1987), Guerrero (1993) for the standard
        # truncated-distribution treatment of BC at lambda < 0.
        as_ts <- function(v) stats::ts(v,
                                       start = stats::start(yFor_bc),
                                       frequency = stats::frequency(yFor_bc))
        bcQuant <- function(p) {
            if (lambda < 0) {
                x_max   <- -1 / lambda
                P_valid <- stats::pnorm(x_max, yFor_bc, se)
                x_q     <- stats::qnorm(p * P_valid, yFor_bc, se)
                return(.inv_box_cox(as_ts(as.numeric(x_q)), lambda))
            }
            z <- stats::qnorm(p)
            if (is.finite(z))
                .inv_box_cox(yFor_bc + z * se, lambda)
            else if (z == -Inf)
                .inv_box_cox(as_ts(rep(-Inf, h)), lambda)
            else
                as_ts(rep(Inf, h))
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
