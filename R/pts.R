#' @title pts: Power / Trend / Seasonal state-space model
#'
#' @description Estimates a PTS (Power / Trend / Seasonal) state-space model
#' for a univariate time series. This is the user-facing entry point of the
#' \code{muse} package and mirrors the calling convention used elsewhere in
#' the \pkg{smooth} family: \code{pts()} estimates the model, and
#' \code{\link{forecast.pts}} produces forecasts from the fitted object
#' without re-estimating.
#'
#' @param y univariate time series (numeric vector or \code{ts}).
#' @param model 3-letter PTS specification string. The three positions encode
#' Power / Trend / Seasonal:
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
#'   \item \code{fitted, residuals, comp} -- in-sample
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
    cl <- match.call()
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

    # PTS() requires h >= 1; ask for at least one forecast point even if the
    # user wants none, and drop it from the public slots below.
    h_fit <- max(as.integer(h), 1L)
    m <- PTS(y, u = u, model = model, s = lags, h = h_fit,
             criterion = criterion, armaIdent = armaIdent, verbose = verbose)

    out <- list(
        y         = y,
        u         = u,
        model     = m$model,
        modelUC   = m$modelUC$model,
        lags      = m$s,
        lambda    = m$lambda,
        p         = m$p,
        p0        = m$p0,
        covp      = m$modelUC$covp,
        parNames  = names(m$p),
        nParam    = length(m$p),
        comp      = m$comp,
        fitted    = if (is.matrix(m$comp)) m$comp[, "Fit"]   else NA,
        residuals = if (is.matrix(m$comp)) m$comp[, "Error"] else NA,
        yFor      = if (h > 0) m$yFor  else NULL,
        yForV     = if (h > 0) m$yForV else NULL,
        logLik    = if (length(m$modelUC$criteria) >= 1) m$modelUC$criteria[1] else NA_real_,
        IC        = if (length(m$modelUC$criteria) >= 4)
                        stats::setNames(m$modelUC$criteria[2:4], c("AIC", "BIC", "AICc"))
                    else NA_real_,
        table     = m$table,
        modelUC_  = m$modelUC,
        call      = cl,
        timeElapsed = proc.time() - tic
    )
    if (!is.null(held))
        out$holdout <- held

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
