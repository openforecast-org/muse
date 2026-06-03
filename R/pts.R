#' @title pts: Power / Trend / Seasonal state-space model
#'
#' @description Estimates a PTS (Power / Trend / Seasonal) state-space model
#' for a univariate time series.  This is the user-facing entry point of the
#' \pkg{muse} package and mirrors the calling convention used elsewhere in
#' the \pkg{smooth} family: \code{pts()} estimates the model, and
#' \code{\link{forecast.pts}} produces forecasts from the fitted object
#' without re-estimating.
#'
#' @param y univariate time series (numeric vector or \code{ts}).
#' @param model 3-letter PTS specification string.  The three positions
#' encode Power / Trend / Seasonal:
#' \itemize{
#'   \item Power: \code{Z} to estimate Box-Cox \eqn{\lambda}, or a numeric
#'     value (e.g. \code{"0"}, \code{"0.5"}, \code{"1"}).
#'   \item Trend: \code{Z} (auto), \code{N} (none / random walk),
#'     \code{L} (local linear), \code{D} (damped / smooth random walk),
#'     \code{G} (global / deterministic).
#'   \item Seasonal: \code{Z} (auto), \code{N} (none), \code{D} (discrete /
#'     linear), \code{T} (trigonometric / equal).
#' }
#' @param lags seasonal period (default \code{frequency(y)}).
#' @param h forecast horizon. If \code{h > 0} a forecast is computed at fit
#' time and cached on the object; \code{forecast(object, h)} can later
#' recompute for a different horizon cheaply.
#' @param holdout logical. If \code{TRUE} and \code{h > 0}, the last \code{h}
#' observations of \code{y} are withheld from estimation and returned in
#' \code{$holdout} for later accuracy assessment.
#' @param criterion information criterion used for automatic model selection
#' (\code{"aic"}, \code{"bic"} or \code{"aicc"}).
#' @param armaIdent logical: search for an ARMA structure in the irregular
#' component.
#' @param verbose logical: print intermediate optimisation output.
#' @param u optional matrix of external regressors.
#'
#' @return An object of class \code{c("pts", "smooth")}.  Slot names mirror
#' \code{smooth::adam()}'s return list where the concept is shared; pts-only
#' extensions are flagged below.
#' \itemize{
#'   \item Inputs / spec: \code{y, model, modelUC*, lags, lambda*}
#'   \item Parameters: \code{B} (estimated parameter vector), \code{covp*}
#'     (parameter covariance), \code{nParam}
#'   \item In-sample fit: \code{fitted, residuals, states} plus
#'     pts-specific \code{comp*} (additive BC-scale decomposition with
#'     Error/Fit columns)
#'   \item Cached forecast: \code{forecast} (original scale, if \code{h > 0})
#'     and \code{forecast_args*} for cheap re-forecasting
#'   \item Likelihood + diagnostics: \code{logLik},
#'     \code{table*} (C++ validation text)
#'   \item Scalars read by \code{plot.smooth} / diagnostics:
#'     \code{distribution = "dnorm"}, \code{loss = "likelihood"},
#'     \code{occurrence = NULL}, \code{holdout}
#'   \item Bookkeeping: \code{call, timeElapsed}
#' }
#' AIC / AICc / BIC / BICc are derived on demand via the methods, not
#' stored on the object.  (* = pts-specific extension.)
#'
#' @seealso \code{\link{forecast.pts}}
#'
#' @template authors
#' @export
pts <- function(y, model = "ZZZ", lags = stats::frequency(y), h = 0,
                holdout = FALSE, criterion = c("aic", "bic", "aicc"),
                armaIdent = FALSE, verbose = FALSE, u = NULL){
    cl  <- match.call()
    tic <- proc.time()
    criterion <- match.arg(criterion)
    if (!is.numeric(h) || length(h) != 1 || h < 0)
        stop("`h` must be a non-negative integer.", call. = FALSE)

    held <- NULL
    if (holdout && h > 0){
        if (length(y) <= h)
            stop("`holdout = TRUE` requires `length(y) > h`.", call. = FALSE)
        n <- length(y) - h
        if (is.ts(y)){
            held <- stats::window(y, start = stats::time(y)[n + 1L])
            y    <- stats::window(y, end   = stats::time(y)[n])
        } else {
            held <- y[(n + 1L):length(y)]
            y    <- y[seq_len(n)]
        }
    }

    # The C++ engine always wants h >= 1; ask for one forecast point even
    # when the user asked for none, and drop the resulting cache below.
    h_fit <- max(as.integer(h), 1L)

    res <- .pts_fit(y = y, u = u, model = model, lags = lags, h = h_fit,
                    criterion = criterion, armaIdent = armaIdent,
                    verbose = verbose)

    cachedFor <- if (h > 0) res$yFor else NULL

    # Structural state evolution (adam-aligned): comp without Error / Fit,
    # truncated to in-sample length so plot.smooth's plot8 (which = 11, 12)
    # can cbind residuals to it.
    statesMat <- NULL
    if (is.matrix(res$comp) && ncol(res$comp) >= 3){
        ns   <- length(y)
        cols <- setdiff(colnames(res$comp), c("Error", "Fit"))
        statesMat <- res$comp[, cols, drop = FALSE]
        if (nrow(statesMat) > ns){
            if (is.ts(statesMat))
                statesMat <- stats::window(statesMat, end = stats::time(y)[ns])
            else
                statesMat <- statesMat[seq_len(ns), , drop = FALSE]
        }
    }

    out <- list(
        ## --- inputs / spec ---
        y          = y,                 # smooth::actuals.smooth reads $y
        model      = uc_to_pts(res$modelUC, res$lambda),
        modelUC    = res$modelUC,       # pts-specific UC string
        lags       = lags,
        lambda     = res$lambda,        # pts-specific Box-Cox parameter
        ## --- parameters (adam name = B) ---
        B          = res$p,
        covp       = res$covp,          # vcov source (we have it directly; adam has $FI instead)
        nParam     = length(res$p),
        ## --- in-sample ---
        fitted     = res$fitted,        # original scale (back-transformed)
        residuals  = res$residuals,     # BC scale (engine innovations)
        comp       = res$comp,          # pts-specific BC-scale additive decomposition
        states     = statesMat,         # adam-aligned structural state evolution
        ## --- forecast cache (adam name = forecast; no variance slot) ---
        forecast      = cachedFor,
        forecast_args = res$forecast_args,
        ## --- likelihood + diagnostics ---
        logLik       = res$logLik,
        table        = res$table,       # pts-specific C++ validation text block
        ## --- smooth/adam-aligned scalars for plot.smooth dispatch ---
        distribution = "dnorm",
        loss         = "likelihood",
        occurrence   = NULL,            # is.occurrence(NULL) == FALSE
        holdout      = NULL,            # overwritten below if holdout = TRUE
        ## --- bookkeeping ---
        call         = cl,
        timeElapsed  = proc.time() - tic
    )
    if (!is.null(held)) out$holdout <- held
    class(out) <- c("pts", "smooth")
    out
}

#' @rdname pts
#' @description \code{auto.pts(y, ...)} is a thin wrapper that forces full
#'   automatic selection (\code{model = "ZZZ"}).
#' @export
auto.pts <- function(y, ...){
    pts(y, model = "ZZZ", ...)
}
