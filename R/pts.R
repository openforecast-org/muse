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
#' @return An object of class \code{c("pts", "smooth")} with components:
#' \itemize{
#'   \item \code{y, u, model, modelUC, lags, lambda} -- inputs / spec
#'   \item \code{p, p0, covp, parNames, nParam} -- parameters
#'   \item \code{fitted, residuals, comp} -- in-sample fit
#'   \item \code{yFor, yForV} -- cached forecast (if \code{h > 0})
#'   \item \code{logLik, IC} -- likelihood + AIC/BIC/AICc
#'   \item \code{table} -- printable validation table
#'   \item \code{call, timeElapsed} -- bookkeeping
#' }
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

    out <- list(
        y         = y,
        u         = u,
        model     = uc_to_pts(res$modelUC, res$lambda),
        modelUC   = res$modelUC,
        lags      = lags,
        lambda    = res$lambda,
        p         = res$p,
        p0        = res$p0,
        covp      = res$covp,
        parNames  = names(res$p),
        nParam    = length(res$p),
        comp      = res$comp,           # BC scale, additive (engine native)
        fitted    = res$fitted,         # original scale (back-transformed)
        residuals = res$residuals,      # BC scale (engine innovations)
        yFor      = if (h > 0) res$yFor  else NULL,
        yForV     = if (h > 0) res$yForV else NULL,
        logLik    = res$logLik,
        IC        = res$IC,
        table     = res$table,
        forecast_args = res$forecast_args,
        call      = cl,
        timeElapsed = proc.time() - tic
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
